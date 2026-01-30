"""
impt_jax.py - Pure functional JAX implementation of IMPT dose calculation.

Usage:
    from DoseCUDA.Jax.impt_jax import computeIMPTPlanJax
    
    dose_jax = computeIMPTPlanJax(dose_grid, plan)
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.ndimage import map_coordinates
from functools import partial
from typing import Tuple, NamedTuple


# =============================================================================
# Data structures using NamedTuples
# =============================================================================

class BeamParams(NamedTuple):
    """Beam geometry parameters."""
    iso_x: jnp.ndarray
    iso_y: jnp.ndarray
    iso_z: jnp.ndarray
    src_x: jnp.ndarray
    src_y: jnp.ndarray
    src_z: jnp.ndarray
    singa: jnp.ndarray
    cosga: jnp.ndarray
    sinta: jnp.ndarray
    costa: jnp.ndarray
    model_vsadx: jnp.ndarray
    model_vsady: jnp.ndarray


class DoseParams(NamedTuple):
    """Dose grid parameters."""
    ni: int
    nj: int
    nk: int
    spacing: jnp.ndarray


class LUTData(NamedTuple):
    """Lookup table data."""
    lut_depths: jnp.ndarray      # [n_energies, lut_len]
    lut_sigmas: jnp.ndarray      # [n_energies, lut_len]
    lut_idds: jnp.ndarray        # [n_energies, lut_len]
    divergence_params: jnp.ndarray  # [n_energies * dvp_len]
    dvp_len: int
    lut_len: int


class SpotData(NamedTuple):
    """Spot position and weight data."""
    spots_x: jnp.ndarray
    spots_y: jnp.ndarray
    spots_mu: jnp.ndarray
    spots_energy_id: jnp.ndarray


class LayerData(NamedTuple):
    """Energy layer data."""
    layers_spot_start: jnp.ndarray
    layers_n_spots: jnp.ndarray
    layers_energy_id: jnp.ndarray
    layers_r80: jnp.ndarray
    n_layers: int


class PrecomputedGrids(NamedTuple):
    """Precomputed voxel grids for efficient reuse."""
    i_indices: jnp.ndarray
    j_indices: jnp.ndarray
    k_indices: jnp.ndarray
    vox_xyz_x: jnp.ndarray
    vox_xyz_y: jnp.ndarray
    vox_xyz_z: jnp.ndarray
    vox_head_x: jnp.ndarray
    vox_head_y: jnp.ndarray
    vox_head_z: jnp.ndarray
    distance_to_source: jnp.ndarray
    uvec_x: jnp.ndarray  # Unit vector toward source (for raytrace)
    uvec_y: jnp.ndarray
    uvec_z: jnp.ndarray


# =============================================================================
# Pure JIT-compiled helper functions
# =============================================================================

@jax.jit
def _sqr(x: jnp.ndarray) -> jnp.ndarray:
    """Square of input."""
    return x * x


@partial(jax.jit, static_argnums=(1,))
def _interpolate_lut(wet: jnp.ndarray, lut_len: int,
                     depths: jnp.ndarray, sigmas: jnp.ndarray,
                     idds: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate dose and sigma from lookup tables."""
    i = jnp.searchsorted(depths, wet, side='left')
    
    at_end = i >= lut_len
    at_start = i <= 0
    
    i_lo = jnp.clip(i - 1, 0, lut_len - 1)
    i_hi = jnp.clip(i, 0, lut_len - 1)
    
    denom = depths[i_hi] - depths[i_lo]
    factor = jnp.where(denom > 0, (wet - depths[i_lo]) / denom, 0.0)
    
    idd_interp = idds[i_lo] + factor * (idds[i_hi] - idds[i_lo])
    sigma_interp = sigmas[i_lo] + factor * (sigmas[i_hi] - sigmas[i_lo])
    
    idd = jnp.where(at_end, idds[-1], jnp.where(at_start, idds[0], idd_interp))
    sigma = jnp.where(at_end, sigmas[-1], jnp.where(at_start, sigmas[0], sigma_interp))
    
    return idd, sigma


@jax.jit
def _sigma_air(wet: jnp.ndarray, distance_to_source: jnp.ndarray,
               r80: jnp.ndarray, coef0: jnp.ndarray, 
               coef1: jnp.ndarray, coef2: jnp.ndarray) -> jnp.ndarray:
    """Calculate sigma from air/multiple scattering."""
    d = distance_to_source - wet + (0.7 * r80)
    return coef0 * d * d + coef1 * d + coef2


@jax.jit
def _nuclear_halo(wet: jnp.ndarray, r80: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate nuclear halo effects."""
    wet_clipped = jnp.clip(wet, 0.1, r80 - 0.1)
    
    halo_sigma = 2.85 + (0.0014 * r80 * jnp.log(wet_clipped + 3.0)) + \
                (0.06 * wet_clipped) - (7.4e-5 * wet_clipped**2) - \
                ((0.22 * r80) / _sqr(wet_clipped - r80 - 5.0))
    
    halo_weight = 0.052 * jnp.log(1.13 + (wet_clipped / (11.2 - (0.023 * r80)))) + \
                 (0.35 * ((0.0017 * r80**2) - r80) / (_sqr(r80 + 3.0) - _sqr(wet_clipped))) - \
                 (1.61e-9 * wet_clipped * _sqr(r80 + 3.0))
    
    halo_sigma = jnp.maximum(halo_sigma, 0.0)
    halo_weight = jnp.clip(halo_weight, 0.0, 0.9)
    
    return halo_sigma, halo_weight


# =============================================================================
# Pre-computation functions (run once per beam)
# =============================================================================

@partial(jax.jit, static_argnums=(0, 1, 2))
def _precompute_all_grids(ni: int, nj: int, nk: int,
                          spacing: jnp.ndarray,
                          iso_x: jnp.ndarray, iso_y: jnp.ndarray, iso_z: jnp.ndarray,
                          src_x: jnp.ndarray, src_y: jnp.ndarray, src_z: jnp.ndarray,
                          singa: jnp.ndarray, cosga: jnp.ndarray,
                          sinta: jnp.ndarray, costa: jnp.ndarray) -> PrecomputedGrids:
    """
    Pre-compute ALL voxel grids and coordinate transforms once.
    
    This includes everything needed for both raytrace AND pencil beam kernels:
    - Voxel indices
    - Physical XYZ coordinates
    - BEV/head coordinates  
    - Distance to source
    - Unit vectors toward source (for ray tracing)
    """
    # Create voxel index grids (once!)
    i_indices, j_indices, k_indices = jnp.meshgrid(
        jnp.arange(ni),
        jnp.arange(nj),
        jnp.arange(nk),
        indexing='ij'
    )
    
    # Convert to physical coordinates
    vox_xyz_x = i_indices.astype(jnp.float32) * spacing - iso_x
    vox_xyz_y = j_indices.astype(jnp.float32) * spacing - iso_y
    vox_xyz_z = k_indices.astype(jnp.float32) * spacing - iso_z
    
    # Convert to BEV coordinates (image_to_head transform)
    xt = vox_xyz_x * costa + vox_xyz_z * (-sinta)
    yt = vox_xyz_y
    zt = -vox_xyz_x * (-sinta) + vox_xyz_z * costa
    
    xg = xt * cosga - yt * (-singa)
    yg = xt * (-singa) + yt * cosga
    zg = zt
    
    vox_head_x = -xg
    vox_head_y = zg
    vox_head_z = yg
    
    # Calculate distances and unit vectors to source (for raytrace)
    dx = src_x - vox_xyz_x
    dy = src_y - vox_xyz_y
    dz = src_z - vox_xyz_z
    distance_to_source = jnp.sqrt(dx**2 + dy**2 + dz**2)
    
    # Unit vectors toward source
    uvec_x = dx / distance_to_source
    uvec_y = dy / distance_to_source
    uvec_z = dz / distance_to_source
    
    return PrecomputedGrids(
        i_indices=i_indices,
        j_indices=j_indices,
        k_indices=k_indices,
        vox_xyz_x=vox_xyz_x,
        vox_xyz_y=vox_xyz_y,
        vox_xyz_z=vox_xyz_z,
        vox_head_x=vox_head_x,
        vox_head_y=vox_head_y,
        vox_head_z=vox_head_z,
        distance_to_source=distance_to_source,
        uvec_x=uvec_x,
        uvec_y=uvec_y,
        uvec_z=uvec_z
    )


# =============================================================================
# Ray tracing kernels
# =============================================================================

@partial(jax.jit, static_argnums=(0, 1, 2, 7))
def _raytrace_kernel_optimized(ni: int, nj: int, nk: int,
                                spacing: jnp.ndarray,
                                iso_x: jnp.ndarray, iso_y: jnp.ndarray, iso_z: jnp.ndarray,
                                max_steps: int,
                                density_3d: jnp.ndarray,
                                vox_xyz_x: jnp.ndarray, vox_xyz_y: jnp.ndarray, vox_xyz_z: jnp.ndarray,
                                uvec_x: jnp.ndarray, uvec_y: jnp.ndarray, uvec_z: jnp.ndarray) -> jnp.ndarray:
    """
    Optimized ray tracing kernel using precomputed grids.
    
    Takes precomputed voxel coordinates and unit vectors instead of
    recomputing meshgrid and unit vectors internally.
    """
    step_length = 1.0  # mm
    
    # Initialize WET accumulator
    wet_sum = jnp.full((ni, nj, nk), -0.05, dtype=jnp.float32)
    
    def ray_step(step, wet_sum):
        ray_length = step * step_length
        
        ray_x = vox_xyz_x + uvec_x * ray_length
        ray_y = vox_xyz_y + uvec_y * ray_length
        ray_z = vox_xyz_z + uvec_z * ray_length
        
        # Convert ray position to array index coordinates for map_coordinates
        # Note: scipy/JAX map_coordinates uses (i, j, k) directly for voxel [i][j][k]
        # (No +0.5 offset needed, unlike CUDA 3D textures which use a half-texel offset)
        tex_x = (ray_x + iso_x) / spacing
        tex_y = (ray_y + iso_y) / spacing
        tex_z = (ray_z + iso_z) / spacing
        
        within_bounds = (tex_x >= 0) & (tex_x < ni) & \
                       (tex_y >= 0) & (tex_y < nj) & \
                       (tex_z >= 0) & (tex_z < nk)
        
        coords = jnp.stack([tex_x, tex_y, tex_z], axis=0)
        density = map_coordinates(density_3d, coords, order=1, mode='constant', cval=0.0)
        
        delta_wet = jnp.where(within_bounds, 
                              jnp.maximum(density, 0.0) * step_length / 10.0,
                              0.0)
        
        return wet_sum + delta_wet
    
    wet_array = lax.fori_loop(0, max_steps, ray_step, wet_sum)
    # Keep in (ni, nj, nk) = (x, y, z) order for internal consistency
    
    return wet_array


# =============================================================================
# WET smoothing kernel (lateral averaging for scattering correction)
# =============================================================================

def _point_head_to_image(head_x, head_y, head_z, singa, cosga, sinta, costa):
    """
    Transform from head coordinates back to image coordinates.
    Inverse of pointXYZImageToHead.
    """
    # Convert from DICOM nozzle coords back to patient coords
    # head->x = -xg, head->y = zg, head->z = yg
    # So: xz = -head_x, yz = head_z, zz = head_y
    xz = -head_x
    yz = head_z
    zz = head_y
    
    # Inverse gantry rotation (rotate about z-axis, positive direction)
    # Note: CUDA uses singa (not -singa) for inverse
    xg = xz * cosga - yz * singa
    yg = xz * singa + yz * cosga
    zg = zz
    
    # Inverse table rotation (rotate about y-axis, positive direction)
    xt = xg * costa + zg * sinta
    yt = yg
    zt = -xg * sinta + zg * costa
    
    return xt, yt, zt


@partial(jax.jit, static_argnums=(0, 1, 2))
def _smooth_wet_kernel(ni: int, nj: int, nk: int,
                       spacing: jnp.ndarray,
                       wet_array: jnp.ndarray,
                       vox_head_x: jnp.ndarray,
                       vox_head_y: jnp.ndarray,
                       vox_head_z: jnp.ndarray,
                       singa: jnp.ndarray, cosga: jnp.ndarray,
                       sinta: jnp.ndarray, costa: jnp.ndarray,
                       iso_x: jnp.ndarray, iso_y: jnp.ndarray, iso_z: jnp.ndarray) -> jnp.ndarray:
    """
    Smooth WET array by lateral averaging perpendicular to beam direction.
    
    Samples 6 directions at 60° intervals, walking outward up to 
    min(10mm, center_wet*10) in each direction.
    """
    # wet_array is (ni, nj, nk) = (x, y, z) order
    
    # Max distance to sample: min(center_wet * 10, 10) mm
    max_dr = jnp.minimum(wet_array * 10.0, 10.0)
    
    # Pre-compute all direction vectors (6 directions at 60 degree intervals)
    # and all distances (1-10 mm)
    n_directions = 6
    n_distances = 10
    
    # Create arrays of angles and distances
    angles = jnp.arange(n_directions) * (jnp.pi / 3.0)  # 0, 60, 120, 180, 240, 300 degrees
    distances = jnp.arange(1, n_distances + 1, dtype=jnp.float32)  # 1, 2, ..., 10 mm
    
    # Create meshgrid of all angle/distance combinations: (n_directions, n_distances)
    angle_grid, dist_grid = jnp.meshgrid(angles, distances, indexing='ij')
    # Flatten to (n_directions * n_distances,)
    all_angles = angle_grid.flatten()
    all_dists = dist_grid.flatten()
    n_samples = len(all_angles)
    
    # Direction components in head x-y plane
    cos_angles = jnp.cos(all_angles)  # (n_samples,)
    sin_angles = jnp.sin(all_angles)  # (n_samples,)
    
    # Initialize sums with center values
    wet_sum = wet_array.copy()
    n_voxels = jnp.ones_like(wet_array)
    
    # Process all sample points using vmap for efficiency
    # For each sample offset, compute contribution to all voxels
    def process_sample(carry, sample_idx):
        wet_sum, n_voxels = carry
        
        dr = all_dists[sample_idx]
        cos_a = cos_angles[sample_idx]
        sin_a = sin_angles[sample_idx]
        
        # Offset in head coordinates (in x-y plane perpendicular to beam)
        offset_head_x = dr * cos_a
        offset_head_y = dr * sin_a
        
        # New head position
        new_head_x = vox_head_x + offset_head_x
        new_head_y = vox_head_y + offset_head_y
        new_head_z = vox_head_z  # unchanged
        
        # Transform back to image coordinates
        new_img_x, new_img_y, new_img_z = _point_head_to_image(
            new_head_x, new_head_y, new_head_z,
            singa, cosga, sinta, costa
        )
        
        # Convert to voxel indices: ijk = (xyz + iso) / spacing
        new_i = jnp.round((new_img_x + iso_x) / spacing).astype(jnp.int32)
        new_j = jnp.round((new_img_y + iso_y) / spacing).astype(jnp.int32)
        new_k = jnp.round((new_img_z + iso_z) / spacing).astype(jnp.int32)
        
        # Check bounds
        within_bounds = (new_i >= 0) & (new_i < ni) & \
                       (new_j >= 0) & (new_j < nj) & \
                       (new_k >= 0) & (new_k < nk)
        
        # Check distance is within max for this voxel
        within_range = dr < max_dr
        valid = within_bounds & within_range
        
        # Clip for safe indexing
        safe_i = jnp.clip(new_i, 0, ni - 1)
        safe_j = jnp.clip(new_j, 0, nj - 1)
        safe_k = jnp.clip(new_k, 0, nk - 1)
        
        # Sample neighbor WET
        neighbor_wet = wet_array[safe_i, safe_j, safe_k]
        
        # Accumulate where valid
        wet_sum = wet_sum + jnp.where(valid, neighbor_wet, 0.0)
        n_voxels = n_voxels + jnp.where(valid, 1.0, 0.0)
        
        return (wet_sum, n_voxels), None
    
    # Use lax.scan to iterate over all samples (compiles to a single loop)
    (wet_sum, n_voxels), _ = lax.scan(process_sample, (wet_sum, n_voxels), jnp.arange(n_samples))
    
    # Return average
    return wet_sum / n_voxels


# =============================================================================
# Pencil beam kernel
# =============================================================================

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _pencil_beam_single_layer(ni: int, nj: int, nk: int, lut_len: int,
                                  spacing: jnp.ndarray,
                                  # Beam params
                                  model_vsadx: jnp.ndarray, model_vsady: jnp.ndarray,
                                  # Pre-computed grids
                                  vox_head_x: jnp.ndarray, vox_head_y: jnp.ndarray, vox_head_z: jnp.ndarray,
                                  distance_to_source: jnp.ndarray,
                                  # Layer-specific data
                                  wet_array: jnp.ndarray,
                                  r80: jnp.ndarray,
                                  depths: jnp.ndarray, sigmas: jnp.ndarray, idds: jnp.ndarray,
                                  coef0: jnp.ndarray, coef1: jnp.ndarray, coef2: jnp.ndarray,
                                  spots_x: jnp.ndarray, spots_y: jnp.ndarray,
                                  spots_mu: jnp.ndarray) -> jnp.ndarray:
    """
    Optimized pencil beam kernel - processes spots sequentially using lax.fori_loop
    to reduce memory usage from O(ni*nj*nk*n_spots) to O(ni*nj*nk).
    """
    # wet_array is already 3D (ni, nj, nk), convert to mm
    wet = wet_array * 10.0
    
    # Check R80 limit
    r80_limit = 1.1 * r80
    valid_mask = wet <= r80_limit
    
    # Get LUT interpolation results (inlined for efficiency)
    i = jnp.searchsorted(depths, wet, side='left')
    at_end = i >= lut_len
    at_start = i <= 0
    i_lo = jnp.clip(i - 1, 0, lut_len - 1)
    i_hi = jnp.clip(i, 0, lut_len - 1)
    denom = depths[i_hi] - depths[i_lo]
    factor = jnp.where(denom > 0, (wet - depths[i_lo]) / denom, 0.0)
    idd_interp = idds[i_lo] + factor * (idds[i_hi] - idds[i_lo])
    sigma_ms_interp = sigmas[i_lo] + factor * (sigmas[i_hi] - sigmas[i_lo])
    idd = jnp.where(at_end, idds[-1], jnp.where(at_start, idds[0], idd_interp))
    sigma_ms = jnp.where(at_end, sigmas[-1], jnp.where(at_start, sigmas[0], sigma_ms_interp))
    
    # Sigma calculations (inlined)
    d = distance_to_source - wet + (0.7 * r80)
    sigma_air = coef0 * d * d + coef1 * d + coef2
    sigma_total = sigma_air + sigma_ms
    
    # Nuclear halo (inlined)
    wet_clipped = jnp.clip(wet, 0.1, r80 - 0.1)
    wet_minus_r80_minus5_sqr = (wet_clipped - r80 - 5.0) ** 2
    r80_plus3_sqr = (r80 + 3.0) ** 2
    
    halo_sigma = 2.85 + (0.0014 * r80 * jnp.log(wet_clipped + 3.0)) + \
                (0.06 * wet_clipped) - (7.4e-5 * wet_clipped**2) - \
                ((0.22 * r80) / wet_minus_r80_minus5_sqr)
    
    halo_weight = 0.052 * jnp.log(1.13 + (wet_clipped / (11.2 - (0.023 * r80)))) + \
                 (0.35 * ((0.0017 * r80**2) - r80) / (r80_plus3_sqr - wet_clipped**2)) - \
                 (1.61e-9 * wet_clipped * r80_plus3_sqr)
    
    halo_sigma = jnp.maximum(halo_sigma, 0.0)
    halo_weight = jnp.clip(halo_weight, 0.0, 0.9)
    
    sigma_halo_total = jnp.sqrt(sigma_total**2 + halo_sigma**2)
    
    # Pre-compute scaling factors (avoid recomputation per spot)
    sigma_total_sqr = sigma_total ** 2
    sigma_halo_total_sqr = sigma_halo_total ** 2
    
    primary_scal = jnp.where(sigma_total > 0, -0.5 / sigma_total_sqr, -jnp.inf)
    halo_scal = jnp.where(sigma_halo_total > 0, -0.5 / sigma_halo_total_sqr, -jnp.inf)
    
    two_pi = 2.0 * jnp.pi
    primary_dose_factor = (1.0 - halo_weight) * idd / (two_pi * sigma_total_sqr + 1e-8)
    halo_dose_factor = halo_weight * idd / (two_pi * sigma_halo_total_sqr + 1e-8)
    
    # Precompute reciprocals for VSAD (avoid division per spot)
    inv_vsadx = 1.0 / model_vsadx
    inv_vsady = 1.0 / model_vsady
    
    n_spots = spots_x.shape[0]
    
    # Process spots sequentially to reduce memory
    def process_spot(spot_id, total_dose):
        spot_x = spots_x[spot_id]
        spot_y = spots_y[spot_id]
        spot_mu = spots_mu[spot_id]
        
        # Compute cax_distance for this spot
        tangent_x = spot_x * inv_vsadx
        tangent_y = spot_y * inv_vsady
        tangent_z = -1.0
        
        ray_x = vox_head_x - spot_x
        ray_y = vox_head_y - spot_y
        ray_z = vox_head_z
        
        tansqr = tangent_x**2 + tangent_y**2 + tangent_z**2
        raysqr = ray_x**2 + ray_y**2 + ray_z**2
        dotprd = tangent_x * ray_x + tangent_y * ray_y + tangent_z * ray_z
        
        distance_to_cax_sqr = raysqr - (dotprd**2 / tansqr)
        
        # Compute dose from this spot
        primary_dose = primary_dose_factor * jnp.exp(primary_scal * distance_to_cax_sqr)
        halo_dose = halo_dose_factor * jnp.exp(halo_scal * distance_to_cax_sqr)
        
        spot_dose = spot_mu * (primary_dose + halo_dose)
        
        return total_dose + spot_dose
    
    # Use fori_loop to iterate over spots
    total_dose = lax.fori_loop(0, n_spots, process_spot, jnp.zeros((ni, nj, nk), dtype=jnp.float32))
    
    # Apply valid mask
    dose_array = jnp.where(valid_mask, total_dose, 0.0)
    
    # Transpose from (ni, nj, nk) = (x, y, z) to (nk, nj, ni) = (z, y, x) order
    # This matches the output format expected by SimpleITK/NRRD
    return dose_array.transpose(2, 1, 0)


# =============================================================================
# High-level computation functions
# =============================================================================

def compute_raytrace_pure(beam_params: BeamParams, dose_params: DoseParams,
                          density_array: jnp.ndarray,
                          grids: PrecomputedGrids) -> jnp.ndarray:
    """
    Compute WET using ray tracing with lateral smoothing - pure functional interface.
    
    Args:
        beam_params: Beam geometry parameters
        dose_params: Dose grid parameters  
        density_array: Density array (3D, shape ni x nj x nk)
        grids: Precomputed voxel grids
        
    Returns:
        Smoothed WET array (3D, shape ni x nj x nk)
    """
    ni, nj, nk = dose_params.ni, dose_params.nj, dose_params.nk
    
    # density_array is already 3D
    density_3d = density_array
    
    # Calculate max steps (ensure we use Python floats to avoid JAX array)
    spacing_val = float(dose_params.spacing)
    max_dist = float(jnp.sqrt(
        (ni * spacing_val)**2 + 
        (nj * spacing_val)**2 + 
        (nk * spacing_val)**2
    )) + 500.0
    max_steps = int(max_dist) + 10
    
    # Step 1: Ray trace to get raw WET
    raw_wet = _raytrace_kernel_optimized(
        ni, nj, nk,
        dose_params.spacing,
        beam_params.iso_x, beam_params.iso_y, beam_params.iso_z,
        max_steps,
        density_3d,
        grids.vox_xyz_x, grids.vox_xyz_y, grids.vox_xyz_z,
        grids.uvec_x, grids.uvec_y, grids.uvec_z
    )
    
    # Step 2: Apply lateral smoothing (accounts for proton scattering)
    smoothed_wet = _smooth_wet_kernel(
        ni, nj, nk,
        dose_params.spacing,
        raw_wet,
        grids.vox_head_x, grids.vox_head_y, grids.vox_head_z,
        beam_params.singa, beam_params.cosga,
        beam_params.sinta, beam_params.costa,
        beam_params.iso_x, beam_params.iso_y, beam_params.iso_z
    )
    
    return smoothed_wet


def compute_dose_pure(beam_params: BeamParams, dose_params: DoseParams,
                      lut_data: LUTData, spot_data: SpotData, layer_data: LayerData,
                      wet_array: jnp.ndarray,
                      grids: PrecomputedGrids = None) -> jnp.ndarray:
    """
    Compute proton dose using optimized kernel that reduces memory usage.
    
    This version uses _pencil_beam_single_layer which processes spots sequentially
    using lax.fori_loop, reducing memory from O(ni*nj*nk*n_spots) to O(ni*nj*nk).
    
    Args:
        beam_params: Beam geometry parameters
        dose_params: Dose grid parameters
        lut_data: Lookup table data
        spot_data: Spot position and weight data
        layer_data: Energy layer data
        wet_array: Water equivalent thickness array (3D, shape ni x nj x nk)
        grids: Optional precomputed grids (if None, will be computed)
        
    Returns:
        Dose array (3D, shape nk x nj x ni in z,y,x order for NRRD)
    """
    ni, nj, nk = dose_params.ni, dose_params.nj, dose_params.nk
    n_layers = layer_data.n_layers
    
    # Pre-compute voxel grids if not provided
    if grids is None:
        grids = _precompute_all_grids(
            ni, nj, nk,
            dose_params.spacing,
            beam_params.iso_x, beam_params.iso_y, beam_params.iso_z,
            beam_params.src_x, beam_params.src_y, beam_params.src_z,
            beam_params.singa, beam_params.cosga,
            beam_params.sinta, beam_params.costa
        )
    
    # Initialize dose accumulator (in output z,y,x order)
    dose_array = jnp.zeros((nk, nj, ni), dtype=jnp.float32)
    
    # Process each layer with Python loop
    for layer_id in range(n_layers):
        energy_id = int(layer_data.layers_energy_id[layer_id])
        r80 = layer_data.layers_r80[layer_id]
        
        # Get divergence coefficients
        base_idx = energy_id * lut_data.dvp_len
        coef0 = lut_data.divergence_params[base_idx + 2]
        coef1 = lut_data.divergence_params[base_idx + 3]
        coef2 = lut_data.divergence_params[base_idx + 4]
        
        # Get LUT slices for this energy
        depths = lut_data.lut_depths[energy_id]
        sigmas = lut_data.lut_sigmas[energy_id]
        idds = lut_data.lut_idds[energy_id]
        
        # Get spots for this layer
        spot_start = int(layer_data.layers_spot_start[layer_id])
        n_spots = int(layer_data.layers_n_spots[layer_id])
        spot_end = spot_start + n_spots
        
        spots_x = spot_data.spots_x[spot_start:spot_end]
        spots_y = spot_data.spots_y[spot_start:spot_end]
        spots_mu = spot_data.spots_mu[spot_start:spot_end]
        
        # Compute layer dose using optimized kernel (reduced memory)
        layer_dose = _pencil_beam_single_layer(
            ni, nj, nk, lut_data.lut_len,
            dose_params.spacing,
            beam_params.model_vsadx, beam_params.model_vsady,
            grids.vox_head_x, grids.vox_head_y, grids.vox_head_z,
            grids.distance_to_source,
            wet_array,
            r80,
            depths, sigmas, idds,
            coef0, coef1, coef2,
            spots_x, spots_y, spots_mu
        )
        
        dose_array = dose_array + layer_dose
    
    return dose_array


def compute_impt_dose_optimized(beam_params: BeamParams, dose_params: DoseParams,
                                 density_array: jnp.ndarray, lut_data: LUTData,
                                 spot_data: SpotData, layer_data: LayerData) -> jnp.ndarray:
    """
    Compute full IMPT dose with optimized grid precomputation.
    
    This is the most efficient API - it:
    1. Precomputes all voxel grids ONCE
    2. Uses them for ray tracing
    3. Reuses them for dose calculation
    
    This avoids duplicate meshgrid creation and coordinate transforms.
    
    Args:
        beam_params: Beam geometry parameters
        dose_params: Dose grid parameters
        density_array: Density array (3D, shape ni x nj x nk)
        lut_data: Lookup table data
        spot_data: Spot position and weight data
        layer_data: Energy layer data
        
    Returns:
        Dose array (3D, shape nk x nj x ni in z,y,x order for NRRD)
    """
    ni, nj, nk = dose_params.ni, dose_params.nj, dose_params.nk
    
    # Precompute all grids ONCE
    grids = _precompute_all_grids(
        ni, nj, nk,
        dose_params.spacing,
        beam_params.iso_x, beam_params.iso_y, beam_params.iso_z,
        beam_params.src_x, beam_params.src_y, beam_params.src_z,
        beam_params.singa, beam_params.cosga,
        beam_params.sinta, beam_params.costa
    )
    
    # Compute WET using precomputed grids
    wet_array = compute_raytrace_pure(beam_params, dose_params, density_array, grids)
    
    # Compute dose using memory-optimized kernel
    dose_array = compute_dose_pure(
        beam_params, dose_params, lut_data, spot_data, layer_data, wet_array, grids
    )
    
    return dose_array


# =============================================================================
# Data extraction helpers (convert from legacy objects to pure data structures)
# =============================================================================

def _extract_beam_params(original_beam, beam_model, dose_grid_origin) -> BeamParams:
    """Extract beam parameters from original beam object into pure data structures."""
    # Adjust isocenter (CUDA does: adjusted_iso = iso - origin)
    adjusted_iso = [
        original_beam.iso[0] - dose_grid_origin[0],
        original_beam.iso[1] - dose_grid_origin[1],
        original_beam.iso[2] - dose_grid_origin[2]
    ]
    
    # Adjust gantry angle (CUDA adds 180 degrees)
    adjusted_gantry = (float(original_beam.gantry_angle) + 180.0) % 360.0
    couch_angle = float(original_beam.couch_angle)
    
    # Source distance is average of model parameters
    src_dist = (float(beam_model.VSADX) + float(beam_model.VSADY)) / 2.0
    
    # Precompute trig functions
    ga = jnp.deg2rad(adjusted_gantry)
    ta = jnp.deg2rad(couch_angle)
    
    singa = jnp.sin(ga)
    cosga = jnp.cos(ga)
    sinta = jnp.sin(ta)
    costa = jnp.cos(ta)
    
    # Compute source position
    xg = -src_dist * singa
    yg = src_dist * cosga
    
    xt = xg * costa
    yt = yg
    zt = -xg * sinta
    
    return BeamParams(
        iso_x=jnp.array(adjusted_iso[0], dtype=jnp.float32),
        iso_y=jnp.array(adjusted_iso[1], dtype=jnp.float32),
        iso_z=jnp.array(adjusted_iso[2], dtype=jnp.float32),
        src_x=xt,
        src_y=yt,
        src_z=zt,
        singa=singa,
        cosga=cosga,
        sinta=sinta,
        costa=costa,
        model_vsadx=jnp.array(float(beam_model.VSADX), dtype=jnp.float32),
        model_vsady=jnp.array(float(beam_model.VSADY), dtype=jnp.float32)
    )


def _extract_lut_data(beam_model) -> LUTData:
    """Extract lookup table data from beam model."""
    return LUTData(
        lut_depths=jnp.array(beam_model.lut_depths, dtype=jnp.float32),
        lut_sigmas=jnp.array(beam_model.lut_sigmas, dtype=jnp.float32),
        lut_idds=jnp.array(beam_model.lut_idds, dtype=jnp.float32),
        divergence_params=jnp.array(beam_model.divergence_params.flatten(), dtype=jnp.float32),
        dvp_len=beam_model.divergence_params.shape[1],
        lut_len=400  # LUT_LENGTH constant
    )


def _extract_spot_data(original_beam, beam_model) -> SpotData:
    """Extract and sort spot data."""
    spot_list = np.array(original_beam.spot_list, dtype=np.float32)
    if spot_list.ndim == 1:
        spot_list = spot_list.reshape(1, -1)
    
    spots_x = spot_list[:, 0]
    spots_y = spot_list[:, 1]
    spots_mu = spot_list[:, 2]
    spots_energy_id = spot_list[:, 3].astype(np.int32)
    
    # Sort by energy_id
    sort_indices = np.argsort(spots_energy_id)
    spots_x = spots_x[sort_indices]
    spots_y = spots_y[sort_indices]
    spots_mu = spots_mu[sort_indices]
    spots_energy_id = spots_energy_id[sort_indices]
    
    return SpotData(
        spots_x=jnp.array(spots_x, dtype=jnp.float32),
        spots_y=jnp.array(spots_y, dtype=jnp.float32),
        spots_mu=jnp.array(spots_mu, dtype=jnp.float32),
        spots_energy_id=jnp.array(spots_energy_id, dtype=jnp.int32)
    )


def _extract_layer_data(spot_data: SpotData, beam_model) -> LayerData:
    """Construct layer data from sorted spots."""
    spots_energy_id = np.array(spot_data.spots_energy_id)
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
    
    return LayerData(
        layers_spot_start=jnp.array(layers_spot_start, dtype=jnp.int32),
        layers_n_spots=jnp.array(layers_n_spots, dtype=jnp.int32),
        layers_energy_id=jnp.array(layers_energy_id, dtype=jnp.int32),
        layers_r80=jnp.array(layers_r80, dtype=jnp.float32),
        n_layers=len(layers_spot_start)
    )


# =============================================================================
# Main entry point (compatible with class-based jax_impt.py API)
# =============================================================================

def computeIMPTPlanJax(dose_grid, plan):
    """
    Compute IMPT dose using pure functional JAX implementation.
    
    This is a drop-in replacement for the CUDA implementation that uses
    pure JAX functions internally for better JIT efficiency.
    
    Args:
        dose_grid: IMPTDoseGrid object with CT/phantom data
        plan: IMPTPlan object with beam list
        
    Returns:
        3D numpy array of dose values (same shape as dose_grid.size)
    """
    # Check isotropic spacing
    if dose_grid.spacing[0] != dose_grid.spacing[1] or dose_grid.spacing[0] != dose_grid.spacing[2]:
        raise ValueError("Spacing must be isotropic for IMPT dose calculation")
    
    # Get RLSP from HU - keep as 3D array
    rlsp = dose_grid.RLSPFromHU(plan.machine_name)
    density_array = jnp.array(rlsp, dtype=jnp.float32)
    
    # Create dose parameters
    dose_params = DoseParams(
        ni=int(dose_grid.size[0]),
        nj=int(dose_grid.size[1]),
        nk=int(dose_grid.size[2]),
        spacing=jnp.array(float(dose_grid.spacing[0]), dtype=jnp.float32)
    )
    
    # Output shape is (nk, nj, ni) = (z, y, x) for NRRD compatibility
    output_shape = (dose_params.nk, dose_params.nj, dose_params.ni)
    total_dose = np.zeros(output_shape, dtype=np.float32)
    
    for beam in plan.beam_list:
        # Get beam model
        try:
            model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
                beam.dicom_rangeshifter_label
            )
        except ValueError:
            raise ValueError(f"Beam model not found for rangeshifter ID {beam.dicom_rangeshifter_label}")
        
        beam_model = plan.beam_models[model_index]
        
        # Extract pure data structures
        beam_params = _extract_beam_params(beam, beam_model, dose_grid.origin)
        lut_data = _extract_lut_data(beam_model)
        spot_data = _extract_spot_data(beam, beam_model)
        layer_data = _extract_layer_data(spot_data, beam_model)
        
        # Use optimized API that precomputes grids once for raytrace + dose
        # Returns 3D array in (z, y, x) order
        dose_array = compute_impt_dose_optimized(
            beam_params, dose_params, density_array, lut_data, spot_data, layer_data
        )
        
        # Accumulate (both are now 3D arrays in same order)
        total_dose += np.array(dose_array)
    
    # Apply fractions
    total_dose *= plan.n_fractions
    
    return total_dose
