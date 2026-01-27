"""
double_gaussian_pb_jax.py

JAX implementation of the double-Gaussian proton pencil beam model from 
Bhattacharya et al. (2025), Eqs. (1)-(6):

D(x) = MU * IDD(E, z_w) * K_t(E, r_perp, z_w)
K_t = (1 - u_n) * K_c + u_n * K_n
K_c(r) = 1/(2*pi*sigma_c^2) * exp(-r^2/(2*sigma_c^2))
K_n(r) = 1/(2*pi*sigma_n^2) * exp(-r^2/(2*sigma_n^2))
sigma_c(z, z_w) = sigma_air(z) + sigma_mcs(z_w)

Notes:
- This file focuses on the *kernel/dose* given z_w and geometry.
- z_w (water-equivalent path length) is assumed precomputed externally
(ray tracing / stopping power integration).
- u_n and sigma_n are provided as lookup tables vs z_w per energy layer
(consistent with the paper's Soukup-based approach).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
import CudaClasses


class DoubleGaussianPBModel(NamedTuple):
    # 1D depth grid (mm) used by all lookup tables
    zw_grid_mm: jnp.ndarray # (Nz,)

    # Lookup tables vs zw for each discrete energy index
    idd_cgy_mm2_per_mu: jnp.ndarray # (nE, Nz)
    sigma_mcs_mm: jnp.ndarray # (nE, Nz)

    # sigma_air(z) = a0 + a1*z + a2*z^2 (mm), coefficients per energy
    # z is distance from effective source along beam direction (mm)
    sigma_air_coeffs: jnp.ndarray # (nE, 3)

    # Nuclear halo models as tables vs zw (per energy)
    # (These are derived from Soukup et al. in the paper, but formulas are not
    # printed in Bhattacharya et al.; providing tables is the simplest drop-in.)
    un_halo: jnp.ndarray # (nE, Nz) in [0, 1]
    sigma_n_mm: jnp.ndarray # (nE, Nz) (mm)


# ---------- utilities ----------

def interp1d_linear_clamp(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    """
    Linear interpolation with x clamped to [xp[0], xp[-1]].
    Works under jit/vmap; xp must be 1D sorted ascending.
    """
    x = jnp.clip(x, xp[0], xp[-1])
    idx = jnp.searchsorted(xp, x, side="right") - 1
    idx = jnp.clip(idx, 0, xp.shape[0] - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def interp1d_linear_zero_outside(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    """
    Linear interpolation but returns 0 outside [xp[0], xp[-1]].
    Useful for IDD so dose becomes 0 beyond modeled range.
    """
    y = interp1d_linear_clamp(x, xp, fp)
    in_range = (x >= xp[0]) & (x <= xp[-1])
    return jnp.where(in_range, y, 0.0)

def gaussian2d_normed(r2_mm2: jnp.ndarray, sigma_mm: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Normalized 2D Gaussian: (1/(2*pi*sigma^2)) * exp(-r^2/(2*sigma^2))
    sigma_mm can be voxel-dependent (vector).
    """
    sigma_mm = jnp.maximum(sigma_mm, eps)
    sigma2 = sigma_mm * sigma_mm
    return jnp.exp(-0.5 * r2_mm2 / sigma2) / (2.0 * jnp.pi * sigma2)

# ---------- core model ----------
def dose_for_layer_beamcoords(
    *,
    x_mm: jnp.ndarray, # (V,) voxel x in beam coordinates (mm)
    y_mm: jnp.ndarray, # (V,) voxel y in beam coordinates (mm)
    z_mm: jnp.ndarray, # (V,) voxel distance from effective source along beam axis (
    zw_mm: jnp.ndarray, # (V,) voxel water-equivalent path length (mm)
    energy_idx: jnp.ndarray, # scalar int32/int64
    spots_xy_mu_mm: jnp.ndarray,# (S, 3): [x_spot_mm, y_spot_mm, MU]
    spot_mask: jnp.ndarray, # (S,) bool or 0/1
    model: DoubleGaussianPBModel,
    ) -> jnp.ndarray:
    """
    Dose for one energy layer (one energy_idx), summing all spots in that layer.
    Implements Bhattacharya et al. Eqs. (1)-(6) in beam coordinates.
    """
    # Gather per-energy curves/coeffs
    idd_curve = model.idd_cgy_mm2_per_mu[energy_idx] # (Nz,)
    mcs_curve = model.sigma_mcs_mm[energy_idx] # (Nz,)
    un_curve = model.un_halo[energy_idx] # (Nz,)
    sn_curve = model.sigma_n_mm[energy_idx] # (Nz,)
    a = model.sigma_air_coeffs[energy_idx] # (3,)

    # Interpolate voxel-dependent depth terms
    idd = interp1d_linear_zero_outside(zw_mm, model.zw_grid_mm, idd_curve) # (V,)
    sigma_mcs = interp1d_linear_clamp(zw_mm, model.zw_grid_mm, mcs_curve) # (V,)
    un = interp1d_linear_clamp(zw_mm, model.zw_grid_mm, un_curve) # (V,)
    sigma_n = interp1d_linear_clamp(zw_mm, model.zw_grid_mm, sn_curve) # (V,)
    
    # Eq (5): sigma_c = sigma_air(z) + sigma_mcs(zw)
    sigma_air = a[0] + a[1] * z_mm + a[2] * (z_mm * z_mm) # (V,)
    sigma_c = sigma_air + sigma_mcs # (V,)
    
    # Safety clamps to avoid inf/nan (important even where idd==0)
    un = jnp.clip(un, 0.0, 1.0)
    sigma_c = jnp.maximum(sigma_c, 1e-6)
    sigma_n = jnp.maximum(sigma_n, 1e-6)

    # Spot mask as float to zero out padded spots
    m = spot_mask.astype(idd.dtype)

    V = zw_mm.shape[0]
    dose0 = jnp.zeros((V,), dtype=idd.dtype)

    # Loop spots without materializing (S, V) arrays.
    # This matches the "loop over spots per voxel/energy" conceptually,
    # but written as a JAX-friendly reduction.
    def body(i, dose_acc):
        spot = spots_xy_mu_mm[i] # (3,)
        mu = spot[2] * m[i] # scalar

        dx = x_mm - spot[0]
        dy = y_mm - spot[1]
        r2 = dx * dx + dy * dy # (V,)

        kc = gaussian2d_normed(r2, sigma_c) # Eq (4)
        kn = gaussian2d_normed(r2, sigma_n) # Eq (6)
        
        kt = (1.0 - un) * kc + un * kn # Eq (3)
        return dose_acc + (mu * idd * kt) # Eq (1)

    dose = lax.fori_loop(0, spots_xy_mu_mm.shape[0], body, dose0)
    return dose

def dose_for_beam_beamcoords(
    *,
    x_mm: jnp.ndarray, # (V,)
    y_mm: jnp.ndarray, # (V,)
    z_mm: jnp.ndarray, # (V,)
    zw_mm: jnp.ndarray, # (V,)
    layer_energy_idx: jnp.ndarray, # (L,) int
    layer_spots_xy_mu_mm: jnp.ndarray, # (L, Smax, 3)
    layer_spot_mask: jnp.ndarray, # (L, Smax) bool or 0/1
    model: DoubleGaussianPBModel,
    ) -> jnp.ndarray:
    """
    Dose for a single beam angle, summing across energy layers.
    Inputs are padded to fixed Smax per layer with a mask.
    """
    V = zw_mm.shape[0]
    dose0 = jnp.zeros((V,), dtype=zw_mm.dtype)

    def body(l, dose_acc):
        e_idx = layer_energy_idx[l]
        spots = layer_spots_xy_mu_mm[l]
        mask = layer_spot_mask[l]
        d = dose_for_layer_beamcoords(
            x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, zw_mm=zw_mm,
            energy_idx=e_idx, spots_xy_mu_mm=spots, spot_mask=mask,
            model=model,
        )
        return dose_acc + d

    return lax.fori_loop(0, layer_energy_idx.shape[0], body, dose0)

# JIT wrappers (optional)
dose_for_layer_beamcoords_jit = jax.jit(dose_for_layer_beamcoords)
dose_for_beam_beamcoords_jit = jax.jit(dose_for_beam_beamcoords)

