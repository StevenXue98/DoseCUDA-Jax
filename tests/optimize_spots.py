"""
optimize_spots.py - Spot weight optimization using JAX automatic differentiation

Implements optimization based on Shen et al. paper equation (4):
  min_w  sum_{i in Target} (d_i - d_rx)^2  +  sum_{i in OAR} alpha * max(0, d_i - d_max)^2

where:
  - d_i = sum_j (D_ij * w_j) is the dose at voxel i
  - D_ij is the dose influence matrix (dose at voxel i from unit weight at spot j)
  - w_j >= 0 are the spot weights to optimize
  - d_rx is the prescription dose for target
  - d_max is the maximum allowed dose for OAR
  - alpha is the OAR penalty weight

Using JAX for:
  1. Automatic differentiation of the objective function
  2. GPU-accelerated dose computation
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax  # JAX optimization library
import SimpleITK as sitk

# Add paths
sys.path.append('/home/ubuntu/DoseCUDA-Jax')
sys.path.insert(0, '/home/ubuntu/DoseCUDA-Jax/DoseCUDA/Jax')

from DoseCUDA import IMPTDoseGrid, IMPTPlan, IMPTBeam
from impt_jax_fix import (
    computeIMPTPlanJax,
    _extract_beam_params, _extract_lut_data, _extract_spot_data, _extract_layer_data,
    _precompute_all_grids, compute_raytrace, _pencil_beam_single_layer,
    BeamParams, DoseParams, LUTData, SpotData, LayerData, PrecomputedGrids
)
from utils.matrad_converter import MatRadData, convert_to_dosecuda


class SpotWeightOptimizer:
    """Spot weight optimizer using JAX autodiff."""
    
    def __init__(self, dose_grid, plan, target_mask, target_rx,
                 oar_masks=None, oar_max_doses=None, oar_weights=None):
        """
        Initialize optimizer.
        
        Args:
            dose_grid: IMPTDoseGrid with CT loaded
            plan: IMPTPlan with beam(s) and spots defined
            target_mask: 3D binary mask for target (same shape as dose_grid.HU)
            target_rx: Prescription dose for target (Gy)
            oar_masks: Dict of {name: 3D mask} for OARs (optional)
            oar_max_doses: Dict of {name: max_dose} for OARs (optional)
            oar_weights: Dict of {name: penalty_weight} for OARs (optional)
        """
        self.dose_grid = dose_grid
        self.plan = plan
        self.target_mask = jnp.array(target_mask, dtype=jnp.float32)
        self.target_rx = target_rx
        
        self.oar_masks = {}
        self.oar_max_doses = {}
        self.oar_weights = {}
        
        if oar_masks is not None:
            for name, mask in oar_masks.items():
                self.oar_masks[name] = jnp.array(mask, dtype=jnp.float32)
                self.oar_max_doses[name] = oar_max_doses.get(name, target_rx * 0.5)
                self.oar_weights[name] = oar_weights.get(name, 100.0)
        
        # Count total spots across all beams
        self.n_spots = sum(beam.n_spots for beam in plan.beam_list)
        print(f"Total spots: {self.n_spots}")
        
        # Pre-extract data structures for efficient JAX computation
        self._prepare_jax_data()
        
    def _prepare_jax_data(self):
        """Extract data needed for JAX dose computation."""
        # Check isotropic spacing
        if self.dose_grid.spacing[0] != self.dose_grid.spacing[1] or \
           self.dose_grid.spacing[0] != self.dose_grid.spacing[2]:
            raise ValueError("Spacing must be isotropic")
        
        # Get RLSP from HU
        rlsp = self.dose_grid.RLSPFromHU(self.plan.machine_name)
        self.density_array = jnp.array(rlsp, dtype=jnp.float32)
        
        # Dose parameters
        self.dose_params = DoseParams(
            ni=int(self.dose_grid.size[0]),
            nj=int(self.dose_grid.size[1]),
            nk=int(self.dose_grid.size[2]),
            spacing=jnp.array(float(self.dose_grid.spacing[0]), dtype=jnp.float32)
        )
        
        # Extract beam data (for single beam case)
        # For multi-beam, we'd need to handle each beam separately
        beam = self.plan.beam_list[0]
        model_index = list(self.plan.dicom_rangeshifter_label.astype(str)).index(
            beam.dicom_rangeshifter_label
        )
        beam_model = self.plan.beam_models[model_index]
        
        self.beam_params = _extract_beam_params(beam, beam_model, self.dose_grid.origin)
        self.lut_data = _extract_lut_data(beam_model)
        
        # Pre-compute grids (shared for all spot weight iterations)
        ni, nj, nk = self.dose_params.ni, self.dose_params.nj, self.dose_params.nk
        self.grids = _precompute_all_grids(
            ni, nj, nk,
            self.dose_params.spacing,
            self.beam_params.iso_x, self.beam_params.iso_y, self.beam_params.iso_z,
            self.beam_params.src_x, self.beam_params.src_y, self.beam_params.src_z,
            self.beam_params.singa, self.beam_params.cosga,
            self.beam_params.sinta, self.beam_params.costa
        )
        
        # Pre-compute WET (doesn't depend on spot weights)
        print("Pre-computing WET (ray tracing)...")
        self.wet_array = compute_raytrace(
            self.beam_params, self.dose_params, self.density_array, self.grids
        )
        print(f"WET computed: shape={self.wet_array.shape}, max={float(self.wet_array.max()):.2f}")
        
        # Extract spot data (we'll modify weights during optimization)
        spot_list = np.array(beam.spot_list, dtype=np.float32)
        if spot_list.ndim == 1:
            spot_list = spot_list.reshape(1, -1)
        
        self.spots_x = jnp.array(spot_list[:, 0], dtype=jnp.float32)
        self.spots_y = jnp.array(spot_list[:, 1], dtype=jnp.float32)
        self.spots_energy_id = jnp.array(spot_list[:, 3].astype(np.int32))
        
        # Sort by energy_id
        sort_indices = jnp.argsort(self.spots_energy_id)
        self.spots_x = self.spots_x[sort_indices]
        self.spots_y = self.spots_y[sort_indices]
        self.spots_energy_id = self.spots_energy_id[sort_indices]
        
        # Build layer data
        self._build_layer_data(beam_model)
        
    def _build_layer_data(self, beam_model):
        """Build energy layer data from sorted spots."""
        spots_energy_id = np.array(self.spots_energy_id)
        n_spots = len(spots_energy_id)
        n_energies = beam_model.divergence_params.shape[0]
        
        layers_spot_start = []
        layers_n_spots = []
        layers_energy_id = []
        layers_r80 = []
        
        spot_start = 0
        for energy_id in range(n_energies):
            spot_count = 0
            end = spot_start
            while end < n_spots and spots_energy_id[end] == energy_id:
                spot_count += 1
                end += 1
            
            if spot_count == 0:
                continue
            
            layers_spot_start.append(spot_start)
            layers_n_spots.append(spot_count)
            layers_energy_id.append(energy_id)
            layers_r80.append(beam_model.divergence_params[energy_id, 1])
            
            spot_start += spot_count
        
        self.layers_spot_start = jnp.array(layers_spot_start, dtype=jnp.int32)
        self.layers_n_spots = jnp.array(layers_n_spots, dtype=jnp.int32)
        self.layers_energy_id = jnp.array(layers_energy_id, dtype=jnp.int32)
        self.layers_r80 = jnp.array(layers_r80, dtype=jnp.float32)
        self.n_layers = len(layers_spot_start)
        
        # Store beam model for divergence params
        self.divergence_params = jnp.array(beam_model.divergence_params.flatten(), dtype=jnp.float32)
        self.dvp_len = beam_model.divergence_params.shape[1]
        
        # Pre-extract Python constants for use in JIT-traced function
        # These are used in compute_dose_from_weights to avoid tracer issues
        self._layer_energy_ids = [int(e) for e in layers_energy_id]
        self._layer_spot_starts = [int(s) for s in layers_spot_start]
        self._layer_n_spots = [int(n) for n in layers_n_spots]
        
        # Pre-extract divergence coefficients for each layer
        self._layer_coef0 = []
        self._layer_coef1 = []
        self._layer_coef2 = []
        for energy_id in self._layer_energy_ids:
            base_idx = energy_id * self.dvp_len
            self._layer_coef0.append(self.divergence_params[base_idx + 2])
            self._layer_coef1.append(self.divergence_params[base_idx + 3])
            self._layer_coef2.append(self.divergence_params[base_idx + 4])
        
    def compute_dose_from_weights(self, spot_weights):
        """
        Compute dose distribution from spot weights.
        
        This is the differentiable forward model: weights -> dose
        
        Args:
            spot_weights: 1D array of spot weights (n_spots,)
            
        Returns:
            3D dose array
        """
        ni, nj, nk = self.dose_params.ni, self.dose_params.nj, self.dose_params.nk
        dose_array = jnp.zeros((ni, nj, nk), dtype=jnp.float32)
        
        # Process each layer - use Python loop with pre-extracted constants
        # (Layer structure is fixed, so we can use Python loop)
        for layer_idx in range(self.n_layers):
            # These are pre-computed Python ints, not traced values
            energy_id = self._layer_energy_ids[layer_idx]
            spot_start = self._layer_spot_starts[layer_idx]
            n_layer_spots = self._layer_n_spots[layer_idx]
            spot_end = spot_start + n_layer_spots
            
            r80 = self.layers_r80[layer_idx]
            
            # Get divergence coefficients (pre-extracted as JAX arrays)
            coef0 = self._layer_coef0[layer_idx]
            coef1 = self._layer_coef1[layer_idx]
            coef2 = self._layer_coef2[layer_idx]
            
            # Get LUT slices
            depths = self.lut_data.lut_depths[energy_id]
            sigmas = self.lut_data.lut_sigmas[energy_id]
            idds = self.lut_data.lut_idds[energy_id]
            
            # Get spots for this layer
            layer_spots_x = self.spots_x[spot_start:spot_end]
            layer_spots_y = self.spots_y[spot_start:spot_end]
            layer_spots_mu = spot_weights[spot_start:spot_end]  # Use optimized weights
            
            # Compute layer dose
            layer_dose = _pencil_beam_single_layer(
                ni, nj, nk, self.lut_data.lut_len,
                self.dose_params.spacing,
                self.beam_params.model_vsadx, self.beam_params.model_vsady,
                self.grids.vox_head_x, self.grids.vox_head_y, self.grids.vox_head_z,
                self.grids.distance_to_source,
                self.wet_array,
                r80,
                depths, sigmas, idds,
                coef0, coef1, coef2,
                layer_spots_x, layer_spots_y, layer_spots_mu
            )
            
            dose_array = dose_array + layer_dose
        
        # Apply fractions
        dose_array = dose_array * self.plan.n_fractions
        
        return dose_array
    
    def objective_function(self, spot_weights):
        """
        Compute objective function value (equation 4 from Shen's paper).
        
        L = sum_{i in Target} (d_i - d_rx)^2 + sum_{OAR} alpha * max(0, d_i - d_max)^2
        
        Args:
            spot_weights: 1D array of spot weights
            
        Returns:
            Scalar objective value
        """
        # Ensure non-negative weights
        spot_weights = jnp.maximum(spot_weights, 0.0)
        
        # Compute dose
        dose = self.compute_dose_from_weights(spot_weights)
        
        # Target term: squared deviation from prescription
        target_voxels = dose * self.target_mask
        n_target = jnp.sum(self.target_mask)
        target_term = jnp.sum(self.target_mask * (dose - self.target_rx) ** 2) / jnp.maximum(n_target, 1.0)
        
        # OAR terms: penalize dose above max
        oar_term = 0.0
        for name in self.oar_masks:
            mask = self.oar_masks[name]
            d_max = self.oar_max_doses[name]
            alpha = self.oar_weights[name]
            n_oar = jnp.sum(mask)
            
            # max(0, d - d_max)^2
            overdose = jnp.maximum(dose - d_max, 0.0)
            oar_term = oar_term + alpha * jnp.sum(mask * overdose ** 2) / jnp.maximum(n_oar, 1.0)
        
        return target_term + oar_term
    
    def optimize(self, initial_weights=None, n_iterations=100, learning_rate=0.1,
                 verbose=True):
        """
        Optimize spot weights using gradient descent.
        
        Args:
            initial_weights: Initial spot weights (default: uniform)
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizer
            verbose: Print progress
            
        Returns:
            Optimized spot weights
        """
        if initial_weights is None:
            # Initialize with small positive weights
            initial_weights = jnp.ones(self.n_spots, dtype=jnp.float32) * 0.5
        
        weights = jnp.array(initial_weights, dtype=jnp.float32)
        
        # Use Adam optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(weights)
        
        # JIT compile the loss and gradient computation
        @jit
        def loss_and_grad(w):
            return value_and_grad(self.objective_function)(w)
        
        # Optimization loop
        history = {'loss': [], 'target_dose_mean': [], 'target_dose_std': []}
        
        print(f"Starting optimization with {self.n_spots} spots, {n_iterations} iterations...")
        start_time = time.time()
        
        for i in range(n_iterations):
            # Compute loss and gradient
            loss, grads = loss_and_grad(weights)
            
            # Update weights
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)
            
            # Project to non-negative (ReLU projection)
            weights = jnp.maximum(weights, 0.0)
            
            # Record history
            history['loss'].append(float(loss))
            
            if verbose and (i % 10 == 0 or i == n_iterations - 1):
                # Compute dose statistics
                dose = self.compute_dose_from_weights(weights)
                target_dose = dose * self.target_mask
                mean_dose = float(jnp.sum(target_dose) / jnp.sum(self.target_mask))
                
                elapsed = time.time() - start_time
                print(f"  Iter {i:4d}: loss={float(loss):.4f}, "
                      f"target_mean={mean_dose:.2f} Gy, "
                      f"weights: min={float(weights.min()):.3f}, max={float(weights.max()):.3f}, "
                      f"time={elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.1f}s")
        
        return weights, history


def load_ct_from_nrrd(dose_grid, nrrd_path):
    """Load CT from NRRD file into DoseCUDA dose grid."""
    ct = sitk.ReadImage(nrrd_path)
    hu_zyx = np.array(sitk.GetArrayFromImage(ct), dtype=np.float32)
    dose_grid.HU = np.transpose(hu_zyx, (2, 1, 0))
    dose_grid.origin = np.array(ct.GetOrigin(), dtype=np.float32)
    dose_grid.spacing = np.array(ct.GetSpacing(), dtype=np.float32)
    dose_grid.size = np.array(dose_grid.HU.shape)


def main():
    """Run spot weight optimization on HEAD_AND_NECK phantom."""
    
    print("="*60)
    print("Spot Weight Optimization using JAX")
    print("="*60)
    
    # Output directory
    output_dir = '/home/ubuntu/DoseCUDA-Jax/test_phantom_output/optimization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load matRad data
    print("\nLoading HEAD_AND_NECK data...")
    matrad_data = MatRadData('HEAD_AND_NECK')
    
    # Get target info
    target_name = 'PTV70'
    target_rx = 70.0  # Gy (from matRad objectives)
    target_center = matrad_data.get_target_center(target_name).astype(np.float32)
    print(f"Target: {target_name}, Rx: {target_rx} Gy")
    print(f"Target center: {target_center}")
    
    # Create dose grid and load CT
    dose = IMPTDoseGrid()
    ct_path = '/home/ubuntu/DoseCUDA-Jax/test_phantom_output/head_and_neck/ct.nrrd'
    
    if not os.path.exists(ct_path):
        print("Converting CT...")
        convert_to_dosecuda(matrad_data, '/home/ubuntu/DoseCUDA-Jax/test_phantom_output/head_and_neck', 
                           resample_spacing=3.0)
    
    load_ct_from_nrrd(dose, ct_path)
    print(f"CT shape: {dose.HU.shape}")
    
    # Create plan and beam
    plan = IMPTPlan(machine_name="HitachiProbeatJHU")
    beam = IMPTBeam()
    beam.gantry_angle = 0.0
    beam.couch_angle = 0.0
    beam.iso = target_center
    beam.dicom_rangeshifter_label = '0'
    
    # Create spots covering target - more spots for better coverage
    beam_model = plan.beam_models[0]
    n_energy_layers = 10  # More energy layers
    n_rings = 3  # Multiple rings of spots
    spot_radius_max = 40.0
    
    n_energies = len(beam_model.energy_labels)
    energy_ids = np.linspace(15, min(85, n_energies-1), n_energy_layers, dtype=int)
    
    for energy_id in energy_ids:
        # Central spot
        beam.addSingleSpot(0.0, 0.0, 1.0, energy_id)
        # Multiple rings
        for ring in range(1, n_rings + 1):
            radius = spot_radius_max * ring / n_rings
            n_spots_ring = 8 * ring  # More spots in outer rings
            for k in range(n_spots_ring):
                theta = 2.0 * np.pi * k / n_spots_ring
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                beam.addSingleSpot(x, y, 1.0, energy_id)
    
    plan.addBeam(beam)
    print(f"Spots: {beam.n_spots}")
    
    # Get target mask (need to match CT shape after resampling)
    # Load from saved NRRD if available, otherwise create from matRad
    target_mask_path = f'/home/ubuntu/DoseCUDA-Jax/test_phantom_output/head_and_neck/mask_{target_name}.nrrd'
    if os.path.exists(target_mask_path):
        target_mask_img = sitk.ReadImage(target_mask_path)
        target_mask_zyx = np.array(sitk.GetArrayFromImage(target_mask_img), dtype=np.float32)
        target_mask = np.transpose(target_mask_zyx, (2, 1, 0))  # (z,y,x) -> (x,y,z)
    else:
        print(f"Warning: Target mask not found at {target_mask_path}")
        # Create simple spherical target
        nx, ny, nz = dose.HU.shape
        x = np.arange(nx) * dose.spacing[0] + dose.origin[0]
        y = np.arange(ny) * dose.spacing[1] + dose.origin[1]
        z = np.arange(nz) * dose.spacing[2] + dose.origin[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        dist = np.sqrt((X - target_center[0])**2 + (Y - target_center[1])**2 + (Z - target_center[2])**2)
        target_mask = (dist < 30.0).astype(np.float32)
    
    print(f"Target mask shape: {target_mask.shape}, voxels: {target_mask.sum():.0f}")
    
    # Create optimizer
    optimizer = SpotWeightOptimizer(
        dose_grid=dose,
        plan=plan,
        target_mask=target_mask,
        target_rx=target_rx,
    )
    
    # Run optimization
    optimized_weights, history = optimizer.optimize(
        n_iterations=200,
        learning_rate=2.0,  # Higher learning rate for faster convergence
        verbose=True
    )
    
    # Compute final dose
    print("\nComputing final dose distribution...")
    final_dose = optimizer.compute_dose_from_weights(optimized_weights)
    final_dose_np = np.array(final_dose)
    
    # Statistics
    target_dose = final_dose_np * np.array(target_mask)
    target_voxels = target_dose[target_mask > 0]
    print(f"\nFinal dose statistics in target:")
    print(f"  Mean: {target_voxels.mean():.2f} Gy (Rx: {target_rx} Gy)")
    print(f"  Std:  {target_voxels.std():.2f} Gy")
    print(f"  Min:  {target_voxels.min():.2f} Gy")
    print(f"  Max:  {target_voxels.max():.2f} Gy")
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    
    # Save optimized dose
    dose_img = sitk.GetImageFromArray(np.transpose(final_dose_np, (2, 1, 0)))
    dose_img.SetOrigin(dose.origin.tolist())
    dose_img.SetSpacing(dose.spacing.tolist())
    sitk.WriteImage(dose_img, os.path.join(output_dir, 'dose_optimized.nrrd'))
    
    # Save optimization history
    np.savez(os.path.join(output_dir, 'optimization_history.npz'),
             weights=np.array(optimized_weights),
             loss_history=np.array(history['loss']))
    
    print("Done!")
    

if __name__ == '__main__':
    main()
