"""
jax_impt.py - Jax implementation of computing IMPT dose.

Usage:
    from DoseCUDA.Jax.jax_impt import computeIMPTPlanJax
    
    dose_jax = computeIMPTPlanJax(dose_grid, plan)
"""

import os
import sys
import numpy as np

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import jax
import jax.numpy as jnp

from impt_classes_jax import (
    IMPTBeam_jax, IMPTDose_jax,
    proton_spot_jax, proton_raytrace_jax
)


def _import_layers_from_spots(spots_x, spots_y, spots_mu, spots_energy_id,
                               divergence_params, dvp_len):
    """
    Construct layer data from sorted spots (replicates CUDA importLayers).
    
    Args:
        spots_x, spots_y, spots_mu: Spot position and weight arrays
        spots_energy_id: Energy ID for each spot (must be sorted)
        divergence_params: Divergence parameters array [n_energies, dvp_len]
        dvp_len: Length of divergence params per energy
        
    Returns:
        Dictionary with layer arrays
    """
    n_spots = len(spots_energy_id)
    n_energies = divergence_params.shape[0]
    
    layers_spot_start = []
    layers_n_spots = []
    layers_energy_id = []
    layers_r80 = []
    layers_energy = []
    
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
        layers_energy.append(divergence_params[energy_id, 0])
        layers_r80.append(divergence_params[energy_id, 1])
        
        spot_start += spot_count
    
    return {
        'layers_spot_start': np.array(layers_spot_start, dtype=np.int32),
        'layers_n_spots': np.array(layers_n_spots, dtype=np.int32),
        'layers_energy_id': np.array(layers_energy_id, dtype=np.int32),
        'layers_r80': np.array(layers_r80, dtype=np.float32),
        'layers_energy': np.array(layers_energy, dtype=np.float32),
    }


def _create_jax_beam(original_beam, beam_model, dose_grid_origin=None):
    """
    Convert original DoseCUDA beam and beam model to JAX IMPTBeam.
    
    Args:
        original_beam: Original IMPTBeam object with spot data
        beam_model: IMPTBeamModel with lookup tables
        dose_grid_origin: Origin of the dose grid (for adjusted isocenter calculation)
        
    Returns:
        Configured JaxIMPTBeam ready for dose calculation
    """
    # CUDA computes adjusted_isocenter = iso - origin (see dosemodule.cu lines 161-164)
    # This accounts for the dose grid's position in physical space
    if dose_grid_origin is not None:
        adjusted_iso = jnp.array([
            original_beam.iso[0] - dose_grid_origin[0],
            original_beam.iso[1] - dose_grid_origin[1],
            original_beam.iso[2] - dose_grid_origin[2]
        ], dtype=jnp.float32)
    else:
        adjusted_iso = jnp.array(original_beam.iso, dtype=jnp.float32)
    
    # CUDA adds 180 degrees to gantry angle (see dosemodule.cu line 151)
    # adjusted_ga = fmodf(ga + 180.0f, 360.0f)
    adjusted_gantry = (float(original_beam.gantry_angle) + 180.0) % 360.0
    
    jax_beam = IMPTBeam_jax(
        iso=adjusted_iso,
        gantry_angle=adjusted_gantry,
        couch_angle=float(original_beam.couch_angle),
        model_vsadx=float(beam_model.VSADX),
        model_vsady=float(beam_model.VSADY)
    )
    
    # Set lookup tables
    jax_beam.set_lut_data(
        lut_depths=jnp.array(beam_model.lut_depths, dtype=jnp.float32),
        lut_sigmas=jnp.array(beam_model.lut_sigmas, dtype=jnp.float32),
        lut_idds=jnp.array(beam_model.lut_idds, dtype=jnp.float32)
    )
    
    # Set divergence parameters
    dvp_len = beam_model.divergence_params.shape[1]
    n_energies = beam_model.divergence_params.shape[0]
    jax_beam.set_divergence_params(
        divergence_params=jnp.array(beam_model.divergence_params.flatten(), dtype=jnp.float32),
        dvp_len=dvp_len,
        n_energies=n_energies
    )
    
    # Extract and sort spot data
    spot_list = np.array(original_beam.spot_list, dtype=np.float32)
    if spot_list.ndim == 1:
        spot_list = spot_list.reshape(1, -1)
    
    spots_x = spot_list[:, 0]
    spots_y = spot_list[:, 1]
    spots_mu = spot_list[:, 2]
    spots_energy_id = spot_list[:, 3].astype(np.int32)
    
    # Sort spots by energy_id
    sort_indices = np.argsort(spots_energy_id)
    spots_x = spots_x[sort_indices]
    spots_y = spots_y[sort_indices]
    spots_mu = spots_mu[sort_indices]
    spots_energy_id = spots_energy_id[sort_indices]
    
    # Set spots
    jax_beam.set_spots(
        spots_x=jnp.array(spots_x, dtype=jnp.float32),
        spots_y=jnp.array(spots_y, dtype=jnp.float32),
        spots_mu=jnp.array(spots_mu, dtype=jnp.float32),
        spots_energy_id=jnp.array(spots_energy_id, dtype=jnp.int32)
    )
    
    # Import layers
    layer_data = _import_layers_from_spots(
        spots_x, spots_y, spots_mu, spots_energy_id,
        beam_model.divergence_params, dvp_len
    )
    
    jax_beam.set_layers(
        layers_spot_start=jnp.array(layer_data['layers_spot_start']),
        layers_n_spots=jnp.array(layer_data['layers_n_spots']),
        layers_energy_id=jnp.array(layer_data['layers_energy_id']),
        layers_r80=jnp.array(layer_data['layers_r80']),
        layers_energy=jnp.array(layer_data['layers_energy'])
    )
    
    return jax_beam


def _create_jax_dose_grid(dose_grid):
    """
    Convert original IMPTDoseGrid to JAX IMPTDose.
    
    Args:
        dose_grid: Original IMPTDoseGrid with CT data
        
    Returns:
        JAX IMPTDose object
    """
    img_sz = (int(dose_grid.size[0]), int(dose_grid.size[1]), int(dose_grid.size[2]))
    spacing = float(dose_grid.spacing[0])
    origin = (float(dose_grid.origin[0]), float(dose_grid.origin[1]), float(dose_grid.origin[2]))
    return IMPTDose_jax(img_sz=img_sz, spacing=spacing, origin=origin)


def computeIMPTPlanJax(dose_grid, plan):
    """
    Compute IMPT dose using JAX implementation.
    
    Args:
        dose_grid: IMPTDoseGrid object with CT/phantom data
        plan: IMPTPlan object with beam list
        
    Returns:
        3D numpy array of dose values (same shape as dose_grid.size)
    """
    # Check isotropic spacing
    if dose_grid.spacing[0] != dose_grid.spacing[1] or dose_grid.spacing[0] != dose_grid.spacing[2]:
        raise ValueError("Spacing must be isotropic for IMPT dose calculation")
    
    # Get RLSP from HU
    rlsp = dose_grid.RLSPFromHU(plan.machine_name)
    
    # Create JAX dose grid
    jax_dose = _create_jax_dose_grid(dose_grid)
    density_array = jnp.array(rlsp.flatten(), dtype=jnp.float32)
    
    # Accumulate dose from all beams
    total_dose = np.zeros(dose_grid.size, dtype=np.float32)
    
    for beam in plan.beam_list:
        # Get beam model
        try:
            model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
                beam.dicom_rangeshifter_label
            )
        except ValueError:
            raise ValueError(f"Beam model not found for rangeshifter ID {beam.dicom_rangeshifter_label}")
        
        beam_model = plan.beam_models[model_index]
        
        # Convert to JAX beam (pass dose grid origin for adjusted isocenter)
        jax_beam = _create_jax_beam(beam, beam_model, dose_grid.origin)
        
        # Compute WET via ray tracing
        wet_array = proton_raytrace_jax(jax_beam, jax_dose, density_array)
        
        # Compute dose
        dose_array = proton_spot_jax(jax_beam, jax_dose, density_array, wet_array)
        
        # Accumulate
        total_dose += np.array(dose_array).reshape(dose_grid.size)
    
    # Apply fractions
    total_dose *= plan.n_fractions
    
    return total_dose
