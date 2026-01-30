"""
IMPTClasses.py - JAX implementation of IMPT (Intensity Modulated Proton Therapy) dose kernels

Fully differentiable JAX implementations of IMPT proton dose calculations including:
- Proton LUT interpolation
- Nuclear halo modeling
- Multiple scattering and air spread calculations
- Ray tracing for WET calculation
- Pencil beam kernel for dose computation
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from cuda_classes_jax import (
    CudaBeam_jax, CudaDose_jax, PointXYZ_jax, PointIJK_jax,
    lower_bound_jax, sqr_jax, clamp_jax, xyz_dotproduct_jax, xyz_crossproduct_jax
)

# Lookup table constants
LUT_LENGTH = 400


@dataclass
class Spot_jax:
    """Proton spot parameters"""
    x: jnp.ndarray
    y: jnp.ndarray
    mu: jnp.ndarray  # Weight/fluence
    energy_id: jnp.ndarray


@dataclass
class Layer_jax:
    """Energy layer information"""
    spot_start: jnp.ndarray
    n_spots: jnp.ndarray
    energy_id: jnp.ndarray
    r80: jnp.ndarray  # 80% dose depth
    energy: jnp.ndarray


class IMPTBeam_jax(CudaBeam_jax):
    """
    JAX implementation of IMPT beam geometry and dose calculations.
    
    Extends CudaBeam with proton-specific calculations:
    - Lookup tables for IDD and sigma
    - Multiple scattering modeling
    - Nuclear halo effects
    - Energy-dependent stopping power
    """
    
    def __init__(self, iso: jnp.ndarray, gantry_angle: float, 
                 couch_angle: float, model_vsadx: float, model_vsady: float):
        """
        Initialize IMPT beam
        
        Args:
            iso: Isocenter position [x, y, z]
            gantry_angle: Gantry rotation in degrees
            couch_angle: Couch rotation in degrees
            model_vsadx, model_vsady: Beam model parameters
        """
        # Source distance is average of model parameters
        src_dist = (model_vsadx + model_vsady) / 2.0
        
        super().__init__(iso, gantry_angle, couch_angle, src_dist)
        
        # Beam model parameters
        self.model_vsadx = model_vsadx
        self.model_vsady = model_vsady
        
        # Will be populated with actual data
        self.n_energies = 0
        self.layers = None
        self.n_layers = 0
        self.spots = None
        self.n_spots = 0
        
        # Lookup tables
        self.divergence_params = None  # R80, energy, coefficients
        self.dvp_len = 0
        
        self.lut_depths = None
        self.lut_sigmas = None
        self.lut_idds = None
        self.lut_len = LUT_LENGTH
    
    def set_lut_data(self, lut_depths: jnp.ndarray, lut_sigmas: jnp.ndarray, 
                     lut_idds: jnp.ndarray):
        """
        Set lookup tables for proton dose calculation
        
        Args:
            lut_depths: Depth lookup table [n_energies, lut_len]
            lut_sigmas: Sigma (scattering) lookup table [n_energies, lut_len]
            lut_idds: IDD (dose depth curve) lookup table [n_energies, lut_len]
        """
        self.lut_depths = lut_depths
        self.lut_sigmas = lut_sigmas
        self.lut_idds = lut_idds
    
    def set_divergence_params(self, divergence_params: jnp.ndarray, dvp_len: int, 
                             n_energies: int):
        """
        Set divergence parameters (R80, energy, expansion coefficients)
        
        Args:
            divergence_params: Parameter array [n_energies * dvp_len]
            dvp_len: Length per energy (stride)
            n_energies: Number of energies
        """
        self.divergence_params = divergence_params
        self.dvp_len = dvp_len
        self.n_energies = n_energies
    
    def set_spots(self, spots_x: jnp.ndarray, spots_y: jnp.ndarray, 
                  spots_mu: jnp.ndarray, spots_energy_id: jnp.ndarray):
        """
        Set spot positions and weights
        
        Args:
            spots_x, spots_y: Spot positions
            spots_mu: Spot weights/fluences
            spots_energy_id: Energy ID for each spot
        """
        self.n_spots = spots_x.shape[0]
        self.spots_x = spots_x
        self.spots_y = spots_y
        self.spots_mu = spots_mu
        self.spots_energy_id = spots_energy_id.astype(jnp.int32)
    
    def set_layers(self, layers_spot_start: jnp.ndarray, layers_n_spots: jnp.ndarray,
                   layers_energy_id: jnp.ndarray, layers_r80: jnp.ndarray, 
                   layers_energy: jnp.ndarray):
        """
        Set energy layer information
        
        Args:
            layers_spot_start: Starting spot index for each layer
            layers_n_spots: Number of spots in each layer
            layers_energy_id: Energy ID for each layer
            layers_r80: R80 depth for each layer
            layers_energy: Energy for each layer
        """
        self.n_layers = layers_spot_start.shape[0]
        self.layers_spot_start = layers_spot_start.astype(jnp.int32)
        self.layers_n_spots = layers_n_spots.astype(jnp.int32)
        self.layers_energy_id = layers_energy_id.astype(jnp.int32)
        self.layers_r80 = layers_r80
        self.layers_energy = layers_energy
    
    def interpolate_proton_lut(self, wet: jnp.ndarray, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Interpolate proton dose and scattering from lookup tables
        
        Uses linear interpolation on depth-dose (IDD) and sigma tables

        Args:
            wet: Water equivalent thickness in cm
            layer_id: Layer index
            
        Returns:
            (idd, sigma) - Dose and scattering at WET depth
        """
        energy_id = self.layers_energy_id[layer_id]
        
        depths = self.lut_depths[energy_id]
        sigmas = self.lut_sigmas[energy_id]
        idds = self.lut_idds[energy_id]
        
        # Find insertion point: first index where depths[i] >= wet
        i = lower_bound_jax(depths, wet)
        
        # CUDA logic:
        # if (i == lut_len) { use last value }
        # else if (i == 0) { use first value }
        # else { interpolate between i-1 and i }
        
        at_end = i >= self.lut_len
        at_start = i <= 0
        
        # Safe indices for interpolation (clamped to valid range)
        i_lo = jnp.clip(i - 1, 0, self.lut_len - 1)  # i-1, but at least 0
        i_hi = jnp.clip(i, 0, self.lut_len - 1)      # i, but at most lut_len-1
        
        # Linear interpolation factor
        # factor = (wet - depths[i-1]) / (depths[i] - depths[i-1])
        denom = depths[i_hi] - depths[i_lo]
        factor = jnp.where(denom > 0, (wet - depths[i_lo]) / denom, 0.0)
        
        # Interpolated values
        idd_interp = idds[i_lo] + factor * (idds[i_hi] - idds[i_lo])
        sigma_interp = sigmas[i_lo] + factor * (sigmas[i_hi] - sigmas[i_lo])
        
        # Select based on boundary conditions
        idd = jnp.where(at_end, idds[-1],
                       jnp.where(at_start, idds[0], idd_interp))
        sigma = jnp.where(at_end, sigmas[-1],
                         jnp.where(at_start, sigmas[0], sigma_interp))
        
        return idd, sigma
    
    def sigma_air(self, wet: jnp.ndarray, distance_to_source: jnp.ndarray, 
                  layer_id: int) -> jnp.ndarray:
        """
        Calculate sigma from air/multiple scattering
        
        Empirical formula based on R80 and distance to source
        
        Args:
            wet: Water equivalent thickness
            distance_to_source: Distance from voxel to source
            layer_id: Layer index
            
        Returns:
            Sigma due to air scattering
        """
        r80 = self.layers_r80[layer_id]
        
        # Get divergence parameters (R80, energy, coefficients)
        energy_id = self.layers_energy_id[layer_id]
        base_idx = energy_id * self.dvp_len
        
        # Coefficients are at indices base_idx + 2, +3, +4
        coef0 = self.divergence_params[base_idx + 2]
        coef1 = self.divergence_params[base_idx + 3]
        coef2 = self.divergence_params[base_idx + 4] if self.dvp_len > 4 else 0.0
        
        d = distance_to_source - wet + (0.7 * r80)
        
        # Quadratic expansion: coef0*d^2 + coef1*d + coef2
        sigma = coef0 * d * d + coef1 * d + coef2
        
        return sigma
    
    def nuclear_halo(self, wet: jnp.ndarray, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate nuclear halo effects (secondary protons and fragmentation)
        
        Args:
            wet: Water equivalent thickness
            layer_id: Layer index
            
        Returns:
            (halo_sigma, halo_weight) - Sigma and weight for halo component
        """
        wet_clipped = clamp_jax(wet, 0.1, self.layers_r80[layer_id] - 0.1)
        r80 = self.layers_r80[layer_id]
        
        # Empirical halo parameters
        halo_sigma = 2.85 + (0.0014 * r80 * jnp.log(wet_clipped + 3.0)) + \
                    (0.06 * wet_clipped) - (7.4e-5 * wet_clipped**2) - \
                    ((0.22 * r80) / sqr_jax(wet_clipped - r80 - 5.0))
        
        halo_weight = 0.052 * jnp.log(1.13 + (wet_clipped / (11.2 - (0.023 * r80)))) + \
                     (0.35 * ((0.0017 * r80**2) - r80) / (sqr_jax(r80 + 3.0) - sqr_jax(wet_clipped))) - \
                     (1.61e-9 * wet_clipped * sqr_jax(r80 + 3.0))
        
        halo_sigma = jnp.maximum(halo_sigma, 0.0)
        halo_weight = clamp_jax(halo_weight, 0.0, 0.9)
        
        return halo_sigma, halo_weight
    
    def cax_distance(self, spot_x: jnp.ndarray, spot_y: jnp.ndarray, 
                     vox_head: PointXYZ_jax) -> jnp.ndarray:
        """
        Compute squared distance from voxel to pencil beam axis
        
        Args:
            spot_x, spot_y: Spot position
            vox_head: Voxel position in BEV coordinates
            
        Returns:
            Squared distance to beam axis
        """
        # Normalize spot position by model parameters
        tangent_x = spot_x / self.model_vsadx
        tangent_y = spot_y / self.model_vsady
        tangent_z = -1.0
        
        ray_x = vox_head.x - spot_x
        ray_y = vox_head.y - spot_y
        ray_z = vox_head.z
        
        tansqr = tangent_x**2 + tangent_y**2 + tangent_z**2
        raysqr = ray_x**2 + ray_y**2 + ray_z**2
        dotprd = tangent_x * ray_x + tangent_y * ray_y + tangent_z * ray_z
        
        return raysqr - (dotprd**2 / tansqr)


class IMPTDose_jax(CudaDose_jax):
    """JAX implementation of dose grid for IMPT calculations"""
    
    def __init__(self, img_sz: Tuple[int, int, int], spacing: float,
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        super().__init__(img_sz, spacing, origin)


# =============================================================================
# JIT-compiled algorithms for IMPT dose calculation
# =============================================================================

@partial(jax.jit, static_argnums=(1, 2, 3))
def _interpolate_proton_lut_jit(wet: jnp.ndarray, lut_len: int, 
                                 energy_id: int, dvp_len: int,
                                 lut_depths: jnp.ndarray, 
                                 lut_sigmas: jnp.ndarray,
                                 lut_idds: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled LUT interpolation.
    """
    depths = lut_depths[energy_id]
    sigmas = lut_sigmas[energy_id]
    idds = lut_idds[energy_id]
    
    i = jnp.searchsorted(depths, wet, side='left')
    
    at_end = i >= lut_len
    at_start = i <= 0
    
    i_lo = jnp.clip(i - 1, 0, lut_len - 1)
    i_hi = jnp.clip(i, 0, lut_len - 1)
    
    denom = depths[i_hi] - depths[i_lo]
    factor = jnp.where(denom > 0, (wet - depths[i_lo]) / denom, 0.0)
    
    idd_interp = idds[i_lo] + factor * (idds[i_hi] - idds[i_lo])
    sigma_interp = sigmas[i_lo] + factor * (sigmas[i_hi] - sigmas[i_lo])
    
    idd = jnp.where(at_end, idds[-1],
                   jnp.where(at_start, idds[0], idd_interp))
    sigma = jnp.where(at_end, sigmas[-1],
                     jnp.where(at_start, sigmas[0], sigma_interp))
    
    return idd, sigma


@jax.jit
def _sigma_air_jit(wet: jnp.ndarray, distance_to_source: jnp.ndarray,
                   r80: jnp.ndarray, coef0: jnp.ndarray, 
                   coef1: jnp.ndarray, coef2: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled sigma air calculation."""
    d = distance_to_source - wet + (0.7 * r80)
    return coef0 * d * d + coef1 * d + coef2


@jax.jit
def _nuclear_halo_jit(wet: jnp.ndarray, r80: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled nuclear halo calculation."""
    wet_clipped = jnp.clip(wet, 0.1, r80 - 0.1)
    
    halo_sigma = 2.85 + (0.0014 * r80 * jnp.log(wet_clipped + 3.0)) + \
                (0.06 * wet_clipped) - (7.4e-5 * wet_clipped**2) - \
                ((0.22 * r80) / sqr_jax(wet_clipped - r80 - 5.0))
    
    halo_weight = 0.052 * jnp.log(1.13 + (wet_clipped / (11.2 - (0.023 * r80)))) + \
                 (0.35 * ((0.0017 * r80**2) - r80) / (sqr_jax(r80 + 3.0) - sqr_jax(wet_clipped))) - \
                 (1.61e-9 * wet_clipped * sqr_jax(r80 + 3.0))
    
    halo_sigma = jnp.maximum(halo_sigma, 0.0)
    halo_weight = jnp.clip(halo_weight, 0.0, 0.9)
    
    return halo_sigma, halo_weight


@partial(jax.jit, static_argnums=(0, 1, 2))
def _proton_raytrace_kernel_jit(ni: int, nj: int, nk: int,
                                 spacing: float,
                                 iso_x: jnp.ndarray, iso_y: jnp.ndarray, iso_z: jnp.ndarray,
                                 src_x: jnp.ndarray, src_y: jnp.ndarray, src_z: jnp.ndarray,
                                 density_3d: jnp.ndarray,
                                 max_steps: int) -> jnp.ndarray:
    """
    JIT-compiled ray tracing kernel.
    """
    from jax.scipy.ndimage import map_coordinates
    
    # Create voxel grids
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
    
    # Compute unit vectors from voxel to source
    dx = src_x - vox_xyz_x
    dy = src_y - vox_xyz_y
    dz = src_z - vox_xyz_z
    norm = jnp.sqrt(dx**2 + dy**2 + dz**2)
    uvec_x = dx / norm
    uvec_y = dy / norm
    uvec_z = dz / norm
    
    step_length = 1.0  # mm
    
    # Initialize WET accumulator
    wet_sum = jnp.full((ni, nj, nk), -0.05, dtype=jnp.float32)
    
    def ray_step(step, wet_sum):
        ray_length = step * step_length
        
        ray_x = vox_xyz_x + uvec_x * ray_length
        ray_y = vox_xyz_y + uvec_y * ray_length
        ray_z = vox_xyz_z + uvec_z * ray_length
        
        tex_x = (ray_x + iso_x) / spacing + 0.5
        tex_y = (ray_y + iso_y) / spacing + 0.5
        tex_z = (ray_z + iso_z) / spacing + 0.5
        
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
    # The final reshape to dose_grid.size (z, y, x) happens in the main function
    
    return wet_array.flatten()


@partial(jax.jit, static_argnums=(0, 1, 2))
def _pencil_beam_kernel_jit(ni: int, nj: int, nk: int,
                             spacing: float,
                             iso_x: jnp.ndarray, iso_y: jnp.ndarray, iso_z: jnp.ndarray,
                             src_x: jnp.ndarray, src_y: jnp.ndarray, src_z: jnp.ndarray,
                             singa: jnp.ndarray, cosga: jnp.ndarray,
                             sinta: jnp.ndarray, costa: jnp.ndarray,
                             model_vsadx: float, model_vsady: float,
                             wet_array: jnp.ndarray,
                             r80: jnp.ndarray,
                             idd: jnp.ndarray, sigma_ms: jnp.ndarray,
                             coef0: jnp.ndarray, coef1: jnp.ndarray, coef2: jnp.ndarray,
                             spots_x: jnp.ndarray, spots_y: jnp.ndarray,
                             spots_mu: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled pencil beam kernel for a single layer.
    All beam parameters are passed as arrays for JIT compatibility.
    """
    # Create voxel grids
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
    
    # Reshape wet_array to 3D
    wet_3d = wet_array.reshape(ni, nj, nk)
    wet = wet_3d * 10.0  # Convert to mm
    
    # Check R80 limit
    r80_limit = 1.1 * r80
    valid_mask = wet <= r80_limit
    
    # Calculate distances to source
    distance_to_source = jnp.sqrt((src_x - vox_xyz_x)**2 + 
                                  (src_y - vox_xyz_y)**2 + 
                                  (src_z - vox_xyz_z)**2)
    
    # Sigma calculations
    sigma_air = _sigma_air_jit(wet, distance_to_source, r80, coef0, coef1, coef2)
    sigma_total = sigma_air + sigma_ms
    
    sigma_halo, halo_weight = _nuclear_halo_jit(wet, r80)
    sigma_halo_total = jnp.hypot(sigma_total, sigma_halo)
    
    # Pre-compute scaling factors
    primary_scal = jnp.where(sigma_total > 0, -0.5 / sqr_jax(sigma_total), -jnp.inf)
    halo_scal = jnp.where(sigma_halo_total > 0, -0.5 / sqr_jax(sigma_halo_total), -jnp.inf)
    
    primary_dose_factor = (1.0 - halo_weight) * idd / (2.0 * jnp.pi * sqr_jax(sigma_total) + 1e-8)
    halo_dose_factor = halo_weight * idd / (2.0 * jnp.pi * sqr_jax(sigma_halo_total) + 1e-8)
    
    # Vectorized dose calculation over all spots
    vox_head_x_exp = vox_head_x[..., jnp.newaxis]
    vox_head_y_exp = vox_head_y[..., jnp.newaxis]
    vox_head_z_exp = vox_head_z[..., jnp.newaxis]
    
    spots_x_exp = spots_x[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    spots_y_exp = spots_y[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    spots_mu_exp = spots_mu[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    
    # Compute cax_distance for all voxel-spot pairs
    tangent_x = spots_x_exp / model_vsadx
    tangent_y = spots_y_exp / model_vsady
    tangent_z = -1.0
    
    ray_x = vox_head_x_exp - spots_x_exp
    ray_y = vox_head_y_exp - spots_y_exp
    ray_z = vox_head_z_exp
    
    tansqr = tangent_x**2 + tangent_y**2 + tangent_z**2
    raysqr = ray_x**2 + ray_y**2 + ray_z**2
    dotprd = tangent_x * ray_x + tangent_y * ray_y + tangent_z * ray_z
    
    distance_to_cax_sqr = raysqr - (dotprd**2 / tansqr)
    
    # Expand dose factors for broadcasting
    primary_scal_exp = primary_scal[..., jnp.newaxis]
    halo_scal_exp = halo_scal[..., jnp.newaxis]
    primary_dose_factor_exp = primary_dose_factor[..., jnp.newaxis]
    halo_dose_factor_exp = halo_dose_factor[..., jnp.newaxis]
    
    # Compute doses for all spots
    primary_dose = primary_dose_factor_exp * jnp.exp(primary_scal_exp * distance_to_cax_sqr)
    halo_dose = halo_dose_factor_exp * jnp.exp(halo_scal_exp * distance_to_cax_sqr)
    
    dose_per_spot = spots_mu_exp * (primary_dose + halo_dose)
    
    # Sum over all spots
    total_dose = jnp.sum(dose_per_spot, axis=-1)
    
    # Apply valid mask
    dose_array = jnp.where(valid_mask, total_dose, 0.0)
    
    # Transpose from (ni, nj, nk) = (x, y, z) to (nk, nj, ni) = (z, y, x) order
    # to match dose_grid.size which is [z, y, x]
    dose_array = dose_array.transpose(2, 1, 0)
    
    return dose_array.flatten()


# =============================================================================
# Original wrapper functions (call JIT-compiled kernels)
# =============================================================================

@partial(jax.jit, static_argnums=(0, 1, 2))
def _smooth_ray_kernel_jit(ni: int, nj: int, nk: int,
                            wet_array: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled smoothing kernel using 3D convolution."""
    wet_3d = wet_array.reshape(ni, nj, nk)
    
    # Use a simple 3x3x3 averaging kernel
    # Pad the array to handle boundaries
    padded = jnp.pad(wet_3d, 1, mode='edge')
    
    # Compute 6-neighbor average + center
    smoothed = (
        padded[1:-1, 1:-1, 1:-1] +  # center
        padded[:-2, 1:-1, 1:-1] +   # i-1
        padded[2:, 1:-1, 1:-1] +    # i+1
        padded[1:-1, :-2, 1:-1] +   # j-1
        padded[1:-1, 2:, 1:-1] +    # j+1
        padded[1:-1, 1:-1, :-2] +   # k-1
        padded[1:-1, 1:-1, 2:]      # k+1
    ) / 7.0
    
    return smoothed.flatten()


def smooth_ray_kernel_jax(beam: IMPTBeam_jax, dose: IMPTDose_jax, 
                     wet_array: jnp.ndarray) -> jnp.ndarray:
    """
    Smooth WET array using 6-directional averaging.
    Wrapper that calls JIT-compiled kernel.
    
    Args:
        beam: IMPT beam object
        dose: Dose grid object
        wet_array: Water equivalent thickness array
        
    Returns:
        Smoothed WET array
    """
    ni = int(dose.img_sz.i)
    nj = int(dose.img_sz.j)
    nk = int(dose.img_sz.k)
    
    return _smooth_ray_kernel_jit(ni, nj, nk, wet_array)


def pencil_beam_kernel_jax(beam: IMPTBeam_jax, dose: IMPTDose_jax, 
                       wet_array: jnp.ndarray, 
                       layer_id: int) -> jnp.ndarray:
    """
    Calculate dose contribution from pencil beams in given layer.
    Wrapper that calls JIT-compiled kernel.

    Args:
        beam: IMPT beam object
        dose: Dose grid object
        wet_array: Water equivalent thickness array (flat)
        layer_id: Energy layer index
        
    Returns:
        Dose array contribution from this layer (flat)
    """
    # Get image dimensions as Python ints for static args
    ni = int(dose.img_sz.i)
    nj = int(dose.img_sz.j)
    nk = int(dose.img_sz.k)
    
    # Extract layer-specific data
    energy_id = int(beam.layers_energy_id[layer_id])
    r80 = beam.layers_r80[layer_id]
    
    # Get LUT interpolation results
    wet_3d = wet_array.reshape(ni, nj, nk) if wet_array.ndim == 1 else wet_array
    wet_mm = wet_3d * 10.0
    idd, sigma_ms = _interpolate_proton_lut_jit(
        wet_mm, beam.lut_len, energy_id, beam.dvp_len,
        beam.lut_depths, beam.lut_sigmas, beam.lut_idds
    )
    
    # Get divergence coefficients
    base_idx = energy_id * beam.dvp_len
    coef0 = beam.divergence_params[base_idx + 2]
    coef1 = beam.divergence_params[base_idx + 3]
    coef2 = beam.divergence_params[base_idx + 4] if beam.dvp_len > 4 else jnp.array(0.0)
    
    # Get spot data for this layer
    spot_start = int(beam.layers_spot_start[layer_id])
    n_spots_layer = int(beam.layers_n_spots[layer_id])
    spot_end = spot_start + n_spots_layer
    
    spots_x = beam.spots_x[spot_start:spot_end]
    spots_y = beam.spots_y[spot_start:spot_end]
    spots_mu = beam.spots_mu[spot_start:spot_end]
    
    # Call JIT-compiled kernel
    return _pencil_beam_kernel_jit(
        ni, nj, nk,
        dose.spacing,
        beam.iso.x, beam.iso.y, beam.iso.z,
        beam.src.x, beam.src.y, beam.src.z,
        beam.singa, beam.cosga, beam.sinta, beam.costa,
        beam.model_vsadx, beam.model_vsady,
        wet_array,
        r80,
        idd, sigma_ms,
        coef0, coef1, coef2,
        spots_x, spots_y, spots_mu
    )


def proton_raytrace_jax(beam: IMPTBeam_jax, dose: IMPTDose_jax, 
                    density_array: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate WET using ray tracing through density array.
    Wrapper that calls JIT-compiled kernel.
    
    Args:
        beam: IMPT beam object
        dose: Dose grid object
        density_array: Density array (g/cc), flat
        
    Returns:
        WET array (flat)
    """
    # Get dimensions as Python ints
    ni = int(dose.img_sz.i)
    nj = int(dose.img_sz.j)
    nk = int(dose.img_sz.k)
    
    # Reshape density to 3D
    density_3d = density_array.reshape(ni, nj, nk)
    
    # Calculate max steps needed
    # Approximate max distance from any voxel to source
    max_dist = jnp.sqrt(
        (ni * dose.spacing)**2 + 
        (nj * dose.spacing)**2 + 
        (nk * dose.spacing)**2
    ) + 500.0  # Add buffer for source distance
    max_steps = int(max_dist) + 10
    
    # Call JIT-compiled kernel
    return _proton_raytrace_kernel_jit(
        ni, nj, nk,
        dose.spacing,
        beam.iso.x, beam.iso.y, beam.iso.z,
        beam.src.x, beam.src.y, beam.src.z,
        density_3d,
        max_steps
    )


def proton_spot_jax(beam: IMPTBeam_jax, dose: IMPTDose_jax, 
                density_array: jnp.ndarray, 
                wet_array: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate proton dose using pencil beam model for all layers.
    
    Args:
        beam: IMPT beam object
        dose: Dose grid object
        density_array: Density array (unused, kept for API compatibility)
        wet_array: Water equivalent thickness array
        
    Returns:
        Dose array (flat)
    """
    ni = int(dose.img_sz.i)
    nj = int(dose.img_sz.j)
    nk = int(dose.img_sz.k)
    
    # Initialize dose array
    dose_array = jnp.zeros(dose.num_voxels, dtype=jnp.float32)
    
    # Process each energy layer using JIT-compiled kernel
    for layer_id in range(beam.n_layers):
        layer_dose = pencil_beam_kernel_jax(beam, dose, wet_array, layer_id)
        dose_array = dose_array + layer_dose
    
    return dose_array
