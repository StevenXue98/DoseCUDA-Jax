#!/usr/bin/env python3
"""
Spot Weight Optimization using ADMM (Alternating Direction Method of Multipliers)

Based on Shen et al. "Beam angle optimization for proton therapy via 
group-sparsity based angle generation method" (Med Phys 2023)

Key insight: Precompute dose influence matrix D where D[i,j] = dose to voxel i 
from spot j with unit weight. Then dose = D @ x is just a matrix-vector multiply.

ADMM algorithm (Appendix A of the paper):
    x^{n+1} = argmin_x F(x) + (μ/2)||x - z^n + u^n||^2
    z^{n+1} = S(x^{n+1} + u^n)  # soft-thresholding for constraints
    u^{n+1} = u^n + x^{n+1} - z^{n+1}

For the least-squares objective F(x) = ||Dx - d_rx||^2, the x-update becomes:
    (D^T D + μI) x = D^T d_rx + μ(z - u)
which is solved by conjugate gradient.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import scipy.io as sio
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DoseCUDA import IMPTPlan, IMPTBeam, IMPTDoseGrid
from DoseCUDA.Jax.impt_jax_fix import (
    compute_raytrace, compute_dose, compute_impt_dose,
    DoseParams, BeamParams, LUTData, SpotData, LayerData,
    _precompute_all_grids
)


def _extract_beam_params(beam, beam_model, origin):
    """Extract BeamParams from beam object."""
    gantry_rad = np.deg2rad(float(beam.gantry_angle))
    table_rad = np.deg2rad(float(beam.couch_angle))
    
    # Source distance
    sad = (float(beam_model.VSADX) + float(beam_model.VSADY)) / 2.0
    
    # Compute source position
    # For gantry=0, beam comes from +Y direction toward -Y
    singa = np.sin(gantry_rad)
    cosga = np.cos(gantry_rad)
    sinta = np.sin(table_rad)
    costa = np.cos(table_rad)
    
    # Source position relative to isocenter
    # Gantry rotation: around Z axis, 0° = source at +Y
    src_x_rel = -sad * singa  # X component
    src_y_rel = sad * cosga   # Y component (positive for gantry=0)
    src_z_rel = 0.0           # Z component (couch rotation would change this)
    
    iso = np.array(beam.iso) - origin
    
    return BeamParams(
        iso_x=jnp.array(iso[0], dtype=jnp.float32),
        iso_y=jnp.array(iso[1], dtype=jnp.float32),
        iso_z=jnp.array(iso[2], dtype=jnp.float32),
        src_x=jnp.array(iso[0] + src_x_rel, dtype=jnp.float32),
        src_y=jnp.array(iso[1] + src_y_rel, dtype=jnp.float32),
        src_z=jnp.array(iso[2] + src_z_rel, dtype=jnp.float32),
        singa=jnp.array(singa, dtype=jnp.float32),
        cosga=jnp.array(cosga, dtype=jnp.float32),
        sinta=jnp.array(sinta, dtype=jnp.float32),
        costa=jnp.array(costa, dtype=jnp.float32),
        model_vsadx=jnp.array(float(beam_model.VSADX), dtype=jnp.float32),
        model_vsady=jnp.array(float(beam_model.VSADY), dtype=jnp.float32)
    )


def _extract_lut_data(beam_model):
    """Extract LUTData from beam model."""
    n_energies = len(beam_model.energy_labels)
    lut_len = beam_model.lut_depths.shape[1]
    
    # Beam model already has these as arrays with shape (n_energies, lut_len)
    lut_depths = np.array(beam_model.lut_depths, dtype=np.float32)
    lut_sigmas = np.array(beam_model.lut_sigmas, dtype=np.float32)
    lut_idds = np.array(beam_model.lut_idds, dtype=np.float32)
    
    return LUTData(
        lut_depths=jnp.array(lut_depths, dtype=jnp.float32),
        lut_sigmas=jnp.array(lut_sigmas, dtype=jnp.float32),
        lut_idds=jnp.array(lut_idds, dtype=jnp.float32),
        divergence_params=jnp.array(beam_model.divergence_params.flatten(), dtype=jnp.float32),
        dvp_len=beam_model.divergence_params.shape[1],
        lut_len=lut_len
    )


def compute_dose_influence_matrix(plan, dose_grid, beam_index=0, target_mask=None):
    """
    Compute dose influence matrix D where D[i,j] = dose to voxel i from spot j.
    
    Strategy: Compute dose for each spot independently by creating a single-spot
    beam configuration and calling the full dose engine.
    
    Args:
        plan: IMPTPlan object with spots defined
        dose_grid: DoseGrid object
        beam_index: Which beam to compute for
        target_mask: Boolean mask for voxels to include (if None, use all)
        
    Returns:
        D: Dose influence matrix (n_voxels, n_spots)
        voxel_indices: Indices of voxels in the full grid
    """
    print("Computing dose influence matrix...")
    
    # Get spot data from original beam
    beam = plan.beam_list[beam_index]
    spot_list = np.array(beam.spot_list, dtype=np.float32)
    if spot_list.ndim == 1:
        spot_list = spot_list.reshape(1, -1)
    
    n_spots = len(spot_list)
    ni, nj, nk = int(dose_grid.size[0]), int(dose_grid.size[1]), int(dose_grid.size[2])
    
    # Determine which voxels to compute
    if target_mask is not None:
        voxel_indices = np.where(target_mask.flatten())[0]
    else:
        voxel_indices = np.arange(ni * nj * nk)
    
    n_voxels = len(voxel_indices)
    print(f"  Computing dose for {n_spots} spots to {n_voxels} voxels...")
    
    # Allocate dose influence matrix
    D = np.zeros((n_voxels, n_spots), dtype=np.float32)
    
    # Get beam parameters (shared)
    model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
        beam.dicom_rangeshifter_label
    )
    beam_model = plan.beam_models[model_index]
    
    # Check isotropic spacing
    print(f"  Dose grid spacing: {dose_grid.spacing}")
    if dose_grid.spacing[0] != dose_grid.spacing[1] or \
       dose_grid.spacing[0] != dose_grid.spacing[2]:
        # Force isotropic spacing using the first dimension
        print(f"  Warning: Non-isotropic spacing. Using spacing[0]={dose_grid.spacing[0]} for all dimensions")
        spacing = dose_grid.spacing[0]
    else:
        spacing = dose_grid.spacing[0]
    
    # Check if HU data is actually density (matRad stores density, not HU)
    # Density ranges 0-2.5 g/cm³, HU ranges -1000 to +3000
    if dose_grid.HU.max() < 10:
        print(f"  Detected density data (max={dose_grid.HU.max():.2f}), using as RLSP directly")
        # matRad stores density (g/cm³), which is approximately equal to RLSP for soft tissue
        density_array = jnp.array(dose_grid.HU, dtype=jnp.float32)
    else:
        # Get RLSP from HU
        rlsp = dose_grid.RLSPFromHU(plan.machine_name)
        density_array = jnp.array(rlsp, dtype=jnp.float32)
    
    # Dose parameters
    dose_params = DoseParams(
        ni=ni, nj=nj, nk=nk,
        spacing=jnp.array(float(dose_grid.spacing[0]), dtype=jnp.float32)
    )
    
    beam_params = _extract_beam_params(beam, beam_model, dose_grid.origin)
    lut_data = _extract_lut_data(beam_model)
    
    # Pre-compute grids (shared for all spots)
    print("  Pre-computing grids...")
    grids = _precompute_all_grids(
        ni, nj, nk,
        dose_params.spacing,
        beam_params.iso_x, beam_params.iso_y, beam_params.iso_z,
        beam_params.src_x, beam_params.src_y, beam_params.src_z,
        beam_params.singa, beam_params.cosga,
        beam_params.sinta, beam_params.costa
    )
    
    # Pre-compute WET (ray tracing) - shared for all spots
    print("  Computing WET (ray tracing)...")
    wet_array = compute_raytrace(beam_params, dose_params, density_array, grids)
    print(f"  WET computed: shape={wet_array.shape}, max={float(wet_array.max()):.2f}")
    
    # Compute dose from each spot using the full dose function
    start_time = time.time()
    
    for j in range(n_spots):
        if j % 50 == 0:
            elapsed = time.time() - start_time
            rate = (j + 1) / elapsed if elapsed > 0 else 0
            eta = (n_spots - j) / rate if rate > 0 else 0
            print(f"    Spot {j}/{n_spots} ({100*j/n_spots:.1f}%), "
                  f"rate: {rate:.1f} spots/s, ETA: {eta:.1f}s")
        
        # Create single-spot data
        spot_x = spot_list[j, 0]
        spot_y = spot_list[j, 1]
        spot_weight = 1.0  # Unit weight
        energy_id = int(spot_list[j, 3])
        
        spot_data = SpotData(
            spots_x=jnp.array([spot_x], dtype=jnp.float32),
            spots_y=jnp.array([spot_y], dtype=jnp.float32),
            spots_mu=jnp.array([spot_weight], dtype=jnp.float32),
            spots_energy_id=jnp.array([energy_id], dtype=jnp.int32)
        )
        
        # Single layer
        r80 = beam_model.divergence_params[energy_id, 1]
        layer_data = LayerData(
            layers_spot_start=jnp.array([0], dtype=jnp.int32),
            layers_n_spots=jnp.array([1], dtype=jnp.int32),
            layers_energy_id=jnp.array([energy_id], dtype=jnp.int32),
            layers_r80=jnp.array([r80], dtype=jnp.float32),
            n_layers=1
        )
        
        # Compute dose
        dose_spot = compute_dose(
            beam_params, dose_params, lut_data, spot_data, layer_data,
            wet_array, grids
        )
        
        # Extract dose at target voxels
        dose_flat = np.array(dose_spot.flatten())
        D[:, j] = dose_flat[voxel_indices]
    
    elapsed = time.time() - start_time
    print(f"  Dose influence matrix computed in {elapsed:.1f}s")
    print(f"  Matrix shape: {D.shape}, size: {D.nbytes / 1e6:.1f} MB")
    print(f"  Non-zero fraction: {np.count_nonzero(D) / D.size:.4f}")
    
    return D, voxel_indices


class ADMMOptimizer:
    """
    ADMM-based spot weight optimizer.
    
    Solves: min_x ||Dx - d_rx||^2 subject to x >= 0
    
    Using ADMM with:
        x-update: (D^T D + μI) x = D^T d_rx + μ(z - u)
        z-update: z = max(0, x + u)  (projection to non-negative)
        u-update: u = u + x - z
    """
    
    def __init__(self, D, d_rx, mu=1.0, max_cg_iters=50, cg_tol=1e-6):
        """
        Args:
            D: Dose influence matrix (n_voxels, n_spots)
            d_rx: Prescription dose vector (n_voxels,)
            mu: ADMM penalty parameter
            max_cg_iters: Max iterations for conjugate gradient
            cg_tol: Tolerance for conjugate gradient
        """
        self.D = jnp.array(D, dtype=jnp.float32)
        self.d_rx = jnp.array(d_rx, dtype=jnp.float32)
        self.mu = mu
        self.max_cg_iters = max_cg_iters
        self.cg_tol = cg_tol
        
        self.n_voxels, self.n_spots = D.shape
        
        # Precompute D^T D (this is the expensive part, but done once)
        print("Precomputing D^T D...")
        start = time.time()
        self.DtD = self.D.T @ self.D
        print(f"  D^T D shape: {self.DtD.shape}, computed in {time.time()-start:.2f}s")
        
        # Precompute D^T d_rx
        self.Dt_drx = self.D.T @ self.d_rx
        
        # JIT compile the key functions
        self._setup_jit_functions()
    
    def _setup_jit_functions(self):
        """Setup JIT-compiled functions."""
        DtD = self.DtD
        mu = self.mu
        max_cg_iters = self.max_cg_iters
        Dt_drx = self.Dt_drx
        
        @jit
        def x_update(z, u):
            """Solve (D^T D + μI) x = D^T d_rx + μ(z - u) using CG."""
            b = Dt_drx + mu * (z - u)
            
            def matvec(x):
                return DtD @ x + mu * x
            
            # CG solver
            x = z  # Initial guess
            r = b - matvec(x)
            p = r
            rsold = jnp.dot(r, r)
            
            def cg_step(carry, _):
                x, r, p, rsold = carry
                Ap = matvec(p)
                alpha = rsold / (jnp.dot(p, Ap) + 1e-10)
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = jnp.dot(r, r)
                p = r + (rsnew / (rsold + 1e-10)) * p
                return (x, r, p, rsnew), None
            
            (x, _, _, _), _ = jax.lax.scan(cg_step, (x, r, p, rsold), None, length=max_cg_iters)
            return x
        
        @jit
        def z_update(x, u):
            """z-update: project to non-negative."""
            return jnp.maximum(x + u, 0.0)
        
        D_local = self.D
        n_voxels = self.n_voxels
        d_rx_local = self.d_rx
        
        @jit
        def compute_objective(x):
            """Compute objective: ||Dx - d_rx||^2"""
            d = D_local @ x
            return jnp.sum((d - d_rx_local) ** 2) / n_voxels
        
        self._x_update = x_update
        self._z_update = z_update
        self._compute_objective = compute_objective
    
    def optimize(self, x0=None, n_iterations=100, verbose=True):
        """
        Run ADMM optimization.
        
        Args:
            x0: Initial spot weights (if None, use uniform)
            n_iterations: Number of ADMM iterations
            verbose: Print progress
            
        Returns:
            x: Optimized spot weights
            history: Dictionary with optimization history
        """
        # Initialize
        if x0 is None:
            x = jnp.ones(self.n_spots, dtype=jnp.float32)
        else:
            x = jnp.array(x0, dtype=jnp.float32)
        
        z = x
        u = jnp.zeros_like(x)
        
        history = {
            'objective': [],
            'primal_residual': [],
            'dual_residual': [],
            'time': []
        }
        
        start_time = time.time()
        
        if verbose:
            print(f"Starting ADMM optimization with {self.n_spots} spots, {n_iterations} iterations...")
        
        for k in range(n_iterations):
            z_old = z
            
            # ADMM updates
            x = self._x_update(z, u)
            z = self._z_update(x, u)
            u = u + x - z
            
            # Compute metrics (every 10 iterations to save time)
            if k % 10 == 0 or k == n_iterations - 1:
                obj = float(self._compute_objective(z))
                primal_res = float(jnp.linalg.norm(x - z))
                dual_res = float(self.mu * jnp.linalg.norm(z - z_old))
                
                history['objective'].append(obj)
                history['primal_residual'].append(primal_res)
                history['dual_residual'].append(dual_res)
                history['time'].append(time.time() - start_time)
                
                if verbose:
                    dose_mean = float(jnp.mean(self.D @ z))
                    rx_mean = float(jnp.mean(self.d_rx))
                    print(f"  Iter {k:4d}: obj={obj:.2f}, primal={primal_res:.2f}, "
                          f"dual={dual_res:.4f}, dose_mean={dose_mean:.2f}/{rx_mean:.1f} Gy")
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Optimization completed in {elapsed:.2f}s ({1000*elapsed/n_iterations:.2f} ms/iter)")
        
        return np.array(z), history


def main():
    """Main optimization routine."""
    print("=" * 60)
    print("Spot Weight Optimization using ADMM")
    print("=" * 60)
    
    # Load matRad data
    print("\nLoading HEAD_AND_NECK data...")
    data_path = "/home/ubuntu/DoseCUDA-Jax/data/matrad/phantoms/HEAD_AND_NECK.mat"
    mat = sio.loadmat(data_path, simplify_cells=True)
    
    ct_data = mat['ct']
    cst_data = mat['cst']
    
    # Find PTV70 target
    target_name = None
    target_rx = None
    target_mask = None
    ct_shape = ct_data['cube'].shape
    
    for i, row in enumerate(cst_data):
        name = row[1]
        objectives = row[5]
        
        # Check if objectives is a mat_struct (has __dict__)
        if hasattr(objectives, '__dict__') and hasattr(objectives, 'className'):
            # Check for SquaredDeviation (target objective)
            if 'SquaredDeviation' in objectives.className:
                target_name = name
                target_rx = float(objectives.parameters)
                # Get mask from indices (1-based in MATLAB, Fortran order)
                indices = row[3]
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten() - 1  # Convert to 0-based
                # MATLAB uses Fortran (column-major) order for linear indexing
                i_idx, j_idx, k_idx = np.unravel_index(indices, ct_shape, order='F')
                mask = np.zeros(ct_shape, dtype=bool)
                mask[i_idx, j_idx, k_idx] = True
                target_mask = mask
                print(f"Found target: {name}, Rx: {target_rx} Gy")
        
        # Stop after finding PTV70 (highest priority target)
        if target_name == 'PTV70':
            break
    
    print(f"Target: {target_name}, Rx: {target_rx} Gy")
    print(f"Target voxels: {np.sum(target_mask)}")
    
    # Create plan
    plan = IMPTPlan("HitachiProbeatJHU")
    
    # Set CT data
    ct_cube = ct_data['cube'].astype(np.float32)
    # Resolution is a dict with x, y, z keys
    ct_resolution = np.array([ct_data['resolution']['x'], 
                               ct_data['resolution']['y'], 
                               ct_data['resolution']['z']], dtype=np.float64)
    
    # Use matRad coordinate vectors to get correct origin
    x_coords = np.array(ct_data['x']).flatten()
    y_coords = np.array(ct_data['y']).flatten()
    z_coords = np.array(ct_data['z']).flatten()
    ct_origin = np.array([x_coords[0], y_coords[0], z_coords[0]], dtype=np.float64)
    print(f"CT origin (from matRad): {ct_origin}")
    
    # Create dose grid directly
    dose_grid = IMPTDoseGrid()
    dose_grid.HU = ct_cube
    dose_grid.origin = ct_origin
    dose_grid.spacing = ct_resolution
    dose_grid.size = np.array(ct_cube.shape)
    
    # Find target center for beam setup using matRad coordinates
    target_indices = np.where(target_mask)
    target_center = np.array([
        x_coords[int(np.mean(target_indices[0]))],  # X from indices along axis 0
        y_coords[int(np.mean(target_indices[1]))],  # Y from indices along axis 1
        z_coords[int(np.mean(target_indices[2]))]   # Z from indices along axis 2
    ])
    print(f"Target center: {target_center}")
    
    # Add beam
    beam = IMPTBeam()
    beam.gantry_angle = 0.0
    beam.couch_angle = 0.0
    beam.iso = target_center
    beam.dicom_rangeshifter_label = "0"  # Use the available rangeshifter label
    plan.addBeam(beam)
    
    # Create spots covering target - use a grid of spots
    beam_model = plan.beam_models[0]
    
    # Calculate spot grid from target geometry
    # For gantry 0°: BEV x -> patient X, BEV y -> patient Z
    # Target bounding box (computed earlier from matRad indices):
    target_where = np.where(target_mask)
    target_x_min = x_coords[target_where[0].min()]
    target_x_max = x_coords[target_where[0].max()]
    target_y_min = y_coords[target_where[1].min()]  # Depth direction
    target_y_max = y_coords[target_where[1].max()]
    target_z_min = z_coords[target_where[2].min()]
    target_z_max = z_coords[target_where[2].max()]
    
    print(f"\nTarget bounding box:")
    print(f"  X (BEV x): [{target_x_min:.1f}, {target_x_max:.1f}] mm")
    print(f"  Y (depth): [{target_y_min:.1f}, {target_y_max:.1f}] mm")
    print(f"  Z (BEV y): [{target_z_min:.1f}, {target_z_max:.1f}] mm")
    
    # Spot grid in BEV coordinates (relative to isocenter = target_center)
    # Add margin of ~5mm on each side
    margin = 5.0
    spot_spacing = 5.0  # mm - typical clinical spacing
    
    bev_x_min = target_x_min - target_center[0] - margin
    bev_x_max = target_x_max - target_center[0] + margin
    bev_y_min = target_z_min - target_center[2] - margin  # Z -> BEV y
    bev_y_max = target_z_max - target_center[2] + margin
    
    print(f"\nSpot grid (BEV coordinates, relative to isocenter):")
    print(f"  BEV x: [{bev_x_min:.1f}, {bev_x_max:.1f}] mm")
    print(f"  BEV y: [{bev_y_min:.1f}, {bev_y_max:.1f}] mm")
    print(f"  Spacing: {spot_spacing} mm")
    
    # Energy selection based on target depth
    # Need to find energies whose R80 matches the WET to target
    # For now, estimate WET ≈ geometric depth * avg density
    # Surface is at max Y where density > 0.1
    density_profile = ct_cube[:, :, ct_cube.shape[2]//2].mean(axis=0)  # Avg density vs Y
    surface_idx = np.where(density_profile > 0.1)[0][-1]  # Last Y index with tissue
    surface_y = y_coords[surface_idx]
    
    # WET from surface to target (rough estimate: depth * avg_density)
    avg_density = ct_cube[target_mask].mean()
    wet_proximal = (surface_y - target_y_max) * avg_density
    wet_distal = (surface_y - target_y_min) * avg_density
    
    print(f"\nDepth estimation:")
    print(f"  Surface Y: {surface_y:.1f} mm")
    print(f"  Avg target density: {avg_density:.2f} g/cm³")
    print(f"  WET to proximal: {wet_proximal:.1f} mm")
    print(f"  WET to distal: {wet_distal:.1f} mm")
    
    # Find energy indices that span this WET range
    n_energies = len(beam_model.energy_labels)
    r80_values = beam_model.divergence_params[:, 1]  # R80 for each energy
    
    # Find energies with R80 in range [wet_proximal - margin, wet_distal + margin]
    wet_margin = 20.0  # mm
    valid_energies = np.where((r80_values >= wet_proximal - wet_margin) & 
                               (r80_values <= wet_distal + wet_margin))[0]
    
    if len(valid_energies) == 0:
        print("Warning: No energies found in range, using full range")
        valid_energies = np.arange(n_energies)
    
    # Subsample to reasonable number of layers
    n_energy_layers = min(20, len(valid_energies))
    energy_ids = valid_energies[np.linspace(0, len(valid_energies)-1, n_energy_layers, dtype=int)]
    
    print(f"\nEnergy selection:")
    print(f"  Energy range: {beam_model.energy_labels[energy_ids[0]]:.1f} - {beam_model.energy_labels[energy_ids[-1]]:.1f} MeV")
    print(f"  R80 range: {r80_values[energy_ids[0]]:.1f} - {r80_values[energy_ids[-1]]:.1f} mm")
    print(f"  Number of layers: {n_energy_layers}")
    
    # Create grid of spots
    spot_positions = []
    for x in np.arange(bev_x_min, bev_x_max + spot_spacing/2, spot_spacing):
        for y in np.arange(bev_y_min, bev_y_max + spot_spacing/2, spot_spacing):
            spot_positions.append((x, y))
    
    for energy_id in energy_ids:
        for x, y in spot_positions:
            beam.addSingleSpot(x, y, 1.0, energy_id)
    
    n_spots = len(beam.spot_list)
    print(f"\nCreated {n_spots} spots ({len(spot_positions)} positions × {n_energy_layers} energies)")
    
    # Compute dose influence matrix (only for target voxels)
    D, voxel_indices = compute_dose_influence_matrix(
        plan, dose_grid, beam_index=0, target_mask=target_mask
    )
    
    # Apply fractions to D
    D = D * plan.n_fractions
    
    # Create prescription dose vector (for target voxels only)
    d_rx = np.full(len(voxel_indices), target_rx, dtype=np.float32)
    
    # Run ADMM optimization
    # Compute smart initial weights: scale so mean dose ≈ prescription
    # With unit weights, dose is D @ 1. Scale factor needed = d_rx / mean(D @ 1)
    unit_dose = D.sum(axis=1)  # Dose from unit weights to each voxel
    mean_unit_dose = unit_dose.mean()
    scale_factor = target_rx / (mean_unit_dose + 1e-10)
    x0 = np.ones(n_spots, dtype=np.float32) * scale_factor
    print(f"\nInitial weight scale factor: {scale_factor:.1f}")
    print(f"Initial mean dose: {(D @ x0).mean():.2f} Gy")
    
    # ADMM optimization (following Shen et al.)
    # μ should balance the objective ||Dx - d||^2 and constraint penalty
    # A reasonable choice is μ = trace(D^T D) / n_spots
    DtD_diag_sum = np.sum(np.sum(D ** 2, axis=0))  # trace(D^T D)
    mu = DtD_diag_sum / n_spots
    print(f"ADMM μ parameter: {mu:.6f}")
    
    optimizer = ADMMOptimizer(D, d_rx, mu=mu, max_cg_iters=100)
    
    x_opt, history = optimizer.optimize(
        x0=x0,
        n_iterations=200,
        verbose=True
    )
    
    # Compute final dose
    print("\nFinal dose statistics in target:")
    final_dose = D @ x_opt
    print(f"  Mean: {np.mean(final_dose):.2f} Gy (Rx: {target_rx} Gy)")
    print(f"  Std:  {np.std(final_dose):.2f} Gy")
    print(f"  Min:  {np.min(final_dose):.2f} Gy")
    print(f"  Max:  {np.max(final_dose):.2f} Gy")
    print(f"  D95:  {np.percentile(final_dose, 5):.2f} Gy")
    print(f"  D5:   {np.percentile(final_dose, 95):.2f} Gy")
    
    # Spot weight statistics
    print(f"\nSpot weight statistics:")
    print(f"  Non-zero spots: {np.sum(x_opt > 0.01)}/{n_spots}")
    print(f"  Mean weight: {np.mean(x_opt):.4f}")
    print(f"  Max weight:  {np.max(x_opt):.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
