# impt_jax.py - Pure Functional JAX Implementation of IMPT Dose Calculation

## Overview

The impt_jax.py module provides a JAX implementation of Intensity Modulated Proton Therapy (IMPT) dose calculation using the **double-Gaussian pencil beam algorithm**. It works in parallel with the CUDA implementation, interfacing with existing Python modules in the original codebase. It is written in a functional style to maximize the benefit of JAX's XLA compilation.

**Reference Paper:**
> Bhattacharya M, Reamy C, Li H, Lee J, Hrinivich WT. *A Python package for fast GPU‐based proton pencil beam dose calculation.* Journal of Applied Clinical Medical Physics. 2025 Apr 11:e70093. DOI: [10.1002/acm2.70093](http://doi.org/10.1002/acm2.70093)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Structures](#data-structures)
3. [Function Reference](#function-reference)
   - [Helper Functions](#helper-functions)
   - [Grid Precomputation](#grid-precomputation)
   - [Ray Tracing](#ray-tracing)
   - [WET Smoothing](#wet-smoothing)
   - [Pencil Beam Kernel](#pencil-beam-kernel)
   - [High-Level Computation](#high-level-computation)
   - [Data Extraction Helpers](#data-extraction-helpers)
   - [Main Entry Point](#main-entry-point)
4. [Function Dependency Flowchart](#function-dependency-flowchart)
5. [Physical Formulas and Algorithms](#physical-formulas-and-algorithms)

---

## Architecture Overview

The IMPT dose calculation pipeline consists of three main stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IMPT DOSE CALCULATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────────┐    ┌────────────────────────┐   │
│  │   Input     │    │  Ray Tracing    │    │   Pencil Beam Dose     │   │
│  │   Data      │───>│  (WET Calc)     │───>│   Calculation          │   │
│  │             │    │                 │    │                        │   │
│  │ • CT/Phantom│    │ • Line integral │    │ • Double-Gaussian      │   │
│  │ • Beam geo  │    │   of RLSP       │    │   lateral profile      │   │
│  │ • Spot data │    │ • Lateral       │    │ • IDD from LUT         │   │
│  │ • LUT data  │    │   smoothing     │    │ • Nuclear halo         │   │
│  └─────────────┘    └─────────────────┘    └────────────────────────┘   │
│                              │                         │                │
│                              ▼                         ▼                │
│                    ┌─────────────────┐       ┌─────────────────┐        │
│                    │   WET Array     │       │   Dose Array    │        │
│                    │  (ni×nj×nk)     │       │  (nk×nj×ni)     │        │
│                    └─────────────────┘       └─────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

The module uses `NamedTuple` classes for clean, immutable data passing:

### `BeamParams`
```python
class BeamParams(NamedTuple):
    """Beam geometry parameters."""
    iso_x, iso_y, iso_z      # Isocenter position (adjusted: iso - origin) [mm]
    src_x, src_y, src_z      # Virtual source position [mm]
    singa, cosga             # Sin/cos of gantry angle (+180°)
    sinta, costa             # Sin/cos of couch/table angle
    model_vsadx, model_vsady # Virtual SAD in x and y directions [mm]
```

**Source:** Extracted from `IMPTBeam` object (gantry_angle, couch_angle, iso) and `IMPTBeamModel` (VSADX, VSADY).

---

### `DoseParams`
```python
class DoseParams(NamedTuple):
    """Dose grid parameters."""
    ni, nj, nk  # Grid dimensions (x, y, z voxel counts)
    spacing     # Isotropic voxel spacing [mm]
```

**Source:** Extracted from `IMPTDoseGrid.size` and `IMPTDoseGrid.spacing`.

---

### `LUTData`
```python
class LUTData(NamedTuple):
    """Lookup table data for depth-dose and sigma curves."""
    lut_depths         # [n_energies, 400] - Depth values [mm]
    lut_sigmas         # [n_energies, 400] - Multiple scattering sigma [mm]
    lut_idds           # [n_energies, 400] - Integrated Depth Dose values
    divergence_params  # Flattened [n_energies * dvp_len] - Air scattering coefficients
    dvp_len            # Number of divergence params per energy (typically 5)
    lut_len            # LUT length (400)
```

**Source:** Loaded from machine-specific CSV files in `lookuptables/protons/<machine_name>/` by `IMPTBeamModel`.

The divergence_params contain per-energy:
- `[0]`: Energy label
- `[1]`: R80 (80% range depth) [mm]
- `[2]`: coef0 (quadratic air scattering coefficient)
- `[3]`: coef1 (linear air scattering coefficient)
- `[4]`: coef2 (constant air scattering coefficient)

---

### `SpotData`
```python
class SpotData(NamedTuple):
    """Spot position and weight data."""
    spots_x         # X position at isocenter plane [mm]
    spots_y         # Y position at isocenter plane [mm]
    spots_mu        # Monitor units (beam weight)
    spots_energy_id # Energy layer index
```

**Source:** Extracted from `IMPTBeam.spot_list`, sorted by energy_id for efficient layer processing.

---

### `LayerData`
```python
class LayerData(NamedTuple):
    """Energy layer data for efficient layer-by-layer processing."""
    layers_spot_start  # Starting index in spot arrays for each layer
    layers_n_spots     # Number of spots in each layer
    layers_energy_id   # Energy index for each layer
    layers_r80         # R80 (80% range) for each layer [mm]
    n_layers           # Total number of active energy layers
```

**Source:** Constructed from sorted `SpotData` by `_extract_layer_data()`.

---

### `PrecomputedGrids`
```python
class PrecomputedGrids(NamedTuple):
    """Precomputed voxel grids for efficient reuse."""
    i_indices, j_indices, k_indices  # Voxel index meshgrids
    vox_xyz_x, vox_xyz_y, vox_xyz_z  # Physical coordinates relative to isocenter [mm]
    vox_head_x, vox_head_y, vox_head_z  # Beam's Eye View (BEV) coordinates [mm]
    distance_to_source               # Distance from each voxel to virtual source [mm]
    uvec_x, uvec_y, uvec_z           # Unit vectors pointing toward source
```

**Purpose:** Computed once per beam, reused for both ray tracing and dose calculation to avoid redundant meshgrid creation.

---

## Function Reference

### Helper Functions

#### `_sqr(x)`
```python
@jax.jit
def _sqr(x: jnp.ndarray) -> jnp.ndarray
```

**Purpose:** Simple square function, JIT-compiled for efficiency.

**Input:** Any JAX array `x`  
**Output:** Element-wise `x²`

---

#### `_interpolate_lut(wet, lut_len, depths, sigmas, idds)`
```python
@partial(jax.jit, static_argnums=(1,))
def _interpolate_lut(wet, lut_len, depths, sigmas, idds) -> Tuple[idd, sigma]
```

**Purpose:** Linear interpolation of dose (IDD) and sigma from lookup tables at a given water-equivalent depth.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `wet` | Water equivalent thickness at each voxel [mm] |
| `lut_len` | Length of lookup table (400) |
| `depths` | LUT depth values [mm] |
| `sigmas` | LUT sigma values (multiple scattering) [mm] |
| `idds` | LUT integrated depth dose values |

**Output:** Tuple of (idd, sigma) arrays with same shape as `wet`

**Algorithm:**
1. Find insertion index `i` using binary search (`searchsorted`)
2. Handle boundary cases (at start/end of LUT)
3. Linear interpolation: `value = lo + factor * (hi - lo)` where `factor = (wet - depth_lo) / (depth_hi - depth_lo)`

**Connection to Paper:** The LUT contains pre-computed depth-dose curves (IDDs) and multiple scattering sigma values for each proton energy, as described in Section 2.2 of the referenced paper.

---

#### `_sigma_air(wet, distance_to_source, r80, coef0, coef1, coef2)`
```python
@jax.jit
def _sigma_air(wet, distance_to_source, r80, coef0, coef1, coef2) -> jnp.ndarray
```

**Purpose:** Calculate the sigma contribution from beam divergence and scattering in air.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `wet` | Water equivalent thickness [mm] |
| `distance_to_source` | Distance from voxel to virtual source [mm] |
| `r80` | 80% range depth for this energy [mm] |
| `coef0, coef1, coef2` | Energy-specific polynomial coefficients |

**Output:** Air scattering sigma [mm²]

**Formula:**
$$\sigma_{air} = c_0 \cdot d^2 + c_1 \cdot d + c_2$$

where:
$$d = \text{distance\_to\_source} - \text{WET} + 0.7 \times R_{80}$$

**Connection to Paper:** This models the beam divergence from the virtual source and in-air scattering, as part of the double-Gaussian model (Section 2.2).

---

#### `_nuclear_halo(wet, r80)`
```python
@jax.jit
def _nuclear_halo(wet, r80) -> Tuple[halo_sigma, halo_weight]
```

**Purpose:** Calculate nuclear interaction halo parameters for the secondary Gaussian.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `wet` | Water equivalent thickness [mm] |
| `r80` | 80% range depth [mm] |

**Output:** Tuple of (halo_sigma [mm], halo_weight [0-0.9])

**Formulas (empirical fits):**

$$\sigma_{halo} = 2.85 + 0.0014 \cdot R_{80} \cdot \ln(\text{WET} + 3) + 0.06 \cdot \text{WET} - 7.4 \times 10^{-5} \cdot \text{WET}^2 - \frac{0.22 \cdot R_{80}}{(\text{WET} - R_{80} - 5)^2}$$

$$w_{halo} = 0.052 \cdot \ln\left(1.13 + \frac{\text{WET}}{11.2 - 0.023 \cdot R_{80}}\right) + \frac{0.35 \cdot (0.0017 \cdot R_{80}^2 - R_{80})}{(R_{80} + 3)^2 - \text{WET}^2} - 1.61 \times 10^{-9} \cdot \text{WET} \cdot (R_{80} + 3)^2$$

**Connection to Paper:** The nuclear halo models the wide-angle scattering from nuclear interactions, forming the "tail" of the double-Gaussian distribution (Section 2.2).

---

### Grid Precomputation

#### `_precompute_all_grids(ni, nj, nk, spacing, iso_x, iso_y, iso_z, src_x, src_y, src_z, singa, cosga, sinta, costa)`
```python
@partial(jax.jit, static_argnums=(0, 1, 2))
def _precompute_all_grids(...) -> PrecomputedGrids
```

**Purpose:** Pre-compute ALL voxel grids and coordinate transforms once per beam. This is a key optimization that avoids redundant meshgrid creation.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `ni, nj, nk` | Grid dimensions (static for JIT) |
| `spacing` | Voxel spacing [mm] |
| `iso_*` | Isocenter coordinates [mm] |
| `src_*` | Virtual source coordinates [mm] |
| `sin/cos ga/ta` | Gantry and table angle trig values |

**Output:** `PrecomputedGrids` containing all voxel data

**Algorithm:**

1. **Create voxel index meshgrids:**
   ```python
   i, j, k = meshgrid(arange(ni), arange(nj), arange(nk))
   ```

2. **Convert to physical coordinates (relative to isocenter):**
   ```python
   vox_xyz = index * spacing - iso
   ```

3. **Transform to Beam's Eye View (BEV/head) coordinates:**
   
   First, apply couch rotation (about Y-axis):
   $$x_t = x \cdot \cos\theta_T + z \cdot (-\sin\theta_T)$$
   $$y_t = y$$
   $$z_t = -x \cdot (-\sin\theta_T) + z \cdot \cos\theta_T$$
   
   Then, apply gantry rotation (about Z-axis):
   $$x_g = x_t \cdot \cos\theta_G - y_t \cdot (-\sin\theta_G)$$
   $$y_g = x_t \cdot (-\sin\theta_G) + y_t \cdot \cos\theta_G$$
   $$z_g = z_t$$
   
   Finally, convert to head coordinate system:
   $$\text{head}_x = -x_g, \quad \text{head}_y = z_g, \quad \text{head}_z = y_g$$

4. **Calculate distance and unit vectors to source:**
   $$d = \sqrt{(src - vox)^2}$$
   $$\hat{u} = (src - vox) / d$$

---

### Ray Tracing

#### `_raytrace_kernel_optimized(ni, nj, nk, spacing, iso_x, iso_y, iso_z, max_steps, density_3d, vox_xyz_*, uvec_*)`
```python
@partial(jax.jit, static_argnums=(0, 1, 2, 7))
def _raytrace_kernel_optimized(...) -> jnp.ndarray
```

**Purpose:** Compute Water Equivalent Thickness (WET) for each voxel by ray tracing from the voxel toward the source.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `ni, nj, nk` | Grid dimensions (static) |
| `spacing` | Voxel spacing [mm] |
| `iso_*` | Isocenter coordinates [mm] |
| `max_steps` | Maximum ray steps (static, ~grid diagonal + 500) |
| `density_3d` | 3D RLSP (Relative Linear Stopping Power) array |
| `vox_xyz_*` | Precomputed voxel physical coordinates |
| `uvec_*` | Precomputed unit vectors toward source |

**Output:** WET array `[ni, nj, nk]` in g/cm²

**Algorithm (Siddon-style ray marching):**

```
for step in 0..max_steps:
    ray_position = voxel_position + unit_vector * (step * step_length)
    density = trilinear_interpolate(density_3d, ray_position)
    if within_bounds:
        WET += density * step_length / 10  # mm to cm
```

**Code Blocks Explained:**

1. **Initialization:**
   ```python
   wet_sum = jnp.full((ni, nj, nk), -0.05, dtype=jnp.float32)
   ```
   Small negative offset to avoid numerical issues at surface.

2. **Ray step function (inside `lax.fori_loop`):**
   - Advances ray by `step_length` (1 mm) in source direction
   - Converts ray position to texture coordinates for interpolation
   - Uses `map_coordinates` for trilinear density sampling
   - Accumulates WET contribution: `density * step_length / 10`

3. **Bounds checking:**
   ```python
   within_bounds = (tex_x >= 0) & (tex_x < ni) & ...
   ```
   Only accumulate WET while ray is inside the volume.

**Connection to Paper:** WET calculation via ray tracing is described in Section 2.2, enabling heterogeneity corrections.

---

### WET Smoothing

#### `_point_head_to_image(head_x, head_y, head_z, singa, cosga, sinta, costa)`
```python
def _point_head_to_image(...) -> Tuple[xt, yt, zt]
```

**Purpose:** Transform coordinates from BEV (head) back to patient (image) coordinates.

**Algorithm:** Inverse of the forward transform in `_precompute_all_grids`:
1. Recover intermediate coordinates from head: `xz = -head_x, yz = head_z, zz = head_y`
2. Inverse gantry rotation
3. Inverse couch rotation

---

#### `_smooth_wet_kernel(ni, nj, nk, spacing, wet_array, vox_head_*, sin/cos_*, iso_*)`
```python
@partial(jax.jit, static_argnums=(0, 1, 2))
def _smooth_wet_kernel(...) -> jnp.ndarray
```

**Purpose:** Apply lateral WET smoothing to account for proton scattering effects.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `wet_array` | Raw WET from ray tracing [ni, nj, nk] |
| `vox_head_*` | BEV coordinates for each voxel |
| Beam geometry parameters for coordinate transforms |

**Output:** Smoothed WET array [ni, nj, nk]

**Algorithm:**

1. **Define sampling pattern:** 6 directions × 10 distances
   - Angles: 0°, 60°, 120°, 180°, 240°, 300° (hexagonal pattern)
   - Distances: 1 to 10 mm

2. **For each sample direction/distance:**
   - Offset in BEV x-y plane (perpendicular to beam)
   - Transform back to image coordinates
   - Sample neighbor WET if within max range: `min(center_WET * 10, 10 mm)`

3. **Average all valid samples:**
   ```python
   smoothed_wet = sum(valid_samples) / count(valid_samples)
   ```

**Connection to Paper:** Lateral WET smoothing accounts for the lateral scatter equilibrium effects on proton range (Section 2.2).

---

### Pencil Beam Kernel

#### `_pencil_beam_single_layer(ni, nj, nk, lut_len, spacing, model_vsadx, model_vsady, vox_head_*, distance_to_source, wet_array, r80, depths, sigmas, idds, coef*, spots_*)`
```python
@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _pencil_beam_single_layer(...) -> jnp.ndarray
```

**Purpose:** Compute dose contribution from all spots in a single energy layer using the double-Gaussian pencil beam model.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `ni, nj, nk, lut_len` | Grid and LUT dimensions (static) |
| `spacing` | Voxel spacing [mm] |
| `model_vsadx, model_vsady` | Virtual SAD values [mm] |
| `vox_head_*` | BEV coordinates |
| `distance_to_source` | Distance to virtual source [mm] |
| `wet_array` | Smoothed WET [g/cm²] |
| `r80` | 80% range for this energy [mm] |
| `depths, sigmas, idds` | LUT slices for this energy |
| `coef0, coef1, coef2` | Air scattering coefficients |
| `spots_x, spots_y, spots_mu` | Spot positions and weights |

**Output:** Dose array [nk, nj, ni] (z, y, x order for NRRD)

**Algorithm (Double-Gaussian Model):**

1. **Convert WET to mm and check validity:**
   ```python
   wet = wet_array * 10.0  # g/cm² to mm
   valid_mask = wet <= 1.1 * r80  # Beyond range = no dose
   ```

2. **LUT interpolation for IDD and σ_MS:**
   ```python
   idd, sigma_ms = interpolate_lut(wet, depths, sigmas, idds)
   ```

3. **Calculate total sigma (primary Gaussian):**
   $$\sigma_{total}^2 = \sigma_{air}^2 + \sigma_{MS}^2$$

4. **Calculate nuclear halo parameters:**
   $$\sigma_{halo,total}^2 = \sigma_{total}^2 + \sigma_{halo}^2$$

5. **Pre-compute Gaussian scaling factors:**
   ```python
   primary_scal = -0.5 / sigma_total²
   halo_scal = -0.5 / sigma_halo_total²
   ```

6. **Process each spot using `lax.fori_loop`:**
   
   For each spot at position $(x_s, y_s)$:
   
   a. **Calculate distance to Central Axis (CAX):**
   
   The spot defines a ray from virtual source through $(x_s, y_s)$ at isocenter plane:
   $$\vec{t} = \left(\frac{x_s}{VSAD_x}, \frac{y_s}{VSAD_y}, -1\right)$$
   
   Distance from voxel to this ray:
   $$d_{CAX}^2 = |\vec{r}|^2 - \frac{(\vec{t} \cdot \vec{r})^2}{|\vec{t}|^2}$$
   
   where $\vec{r} = (\text{head}_x - x_s, \text{head}_y - y_s, \text{head}_z)$

   b. **Apply double-Gaussian formula:**
   
   $$D = \text{MU} \times \left[ (1 - w_{halo}) \cdot \frac{\text{IDD}}{2\pi\sigma_{total}^2} \cdot e^{-\frac{d_{CAX}^2}{2\sigma_{total}^2}} + w_{halo} \cdot \frac{\text{IDD}}{2\pi\sigma_{halo}^2} \cdot e^{-\frac{d_{CAX}^2}{2\sigma_{halo}^2}} \right]$$

7. **Accumulate dose and transpose to output order:**
   ```python
   dose_array = dose.transpose(2, 1, 0)  # (x,y,z) → (z,y,x)
   ```

**Connection to Paper:** This implements the core double-Gaussian pencil beam algorithm from Equation (1-3) in the paper (Section 2.2).

---

### High-Level Computation

#### `compute_raytrace_pure(beam_params, dose_params, density_array, grids)`
```python
def compute_raytrace_pure(...) -> jnp.ndarray
```

**Purpose:** Complete WET computation pipeline with smoothing.

**Flow:**
1. Calculate max ray steps from grid diagonal
2. Call `_raytrace_kernel_optimized` for raw WET
3. Call `_smooth_wet_kernel` for lateral averaging

**Output:** Smoothed WET array [ni, nj, nk]

---

#### `compute_dose_pure(beam_params, dose_params, lut_data, spot_data, layer_data, wet_array, grids)`
```python
def compute_dose_pure(...) -> jnp.ndarray
```

**Purpose:** Compute dose from all energy layers using precomputed WET.

**Algorithm:**
```python
for layer in layers:
    extract layer-specific LUT data
    extract spots for this layer
    layer_dose = _pencil_beam_single_layer(...)
    total_dose += layer_dose
```

**Output:** Total dose array [nk, nj, ni]

---

#### `compute_impt_dose_optimized(beam_params, dose_params, density_array, lut_data, spot_data, layer_data)`
```python
def compute_impt_dose_optimized(...) -> jnp.ndarray
```

**Purpose:** Most efficient API - combines grid precomputation, ray tracing, and dose calculation.

**Key Optimization:**
- Precomputes all grids ONCE
- Uses same grids for both ray tracing and dose calculation
- Avoids duplicate meshgrid creation and coordinate transforms

**Flow:**
```
_precompute_all_grids() → grids
compute_raytrace_pure(grids) → wet_array
compute_dose_pure(grids, wet_array) → dose_array
```

---

### Data Extraction Helpers

#### `_extract_beam_params(original_beam, beam_model, dose_grid_origin)`
```python
def _extract_beam_params(...) -> BeamParams
```

**Purpose:** Convert legacy `IMPTBeam` object to pure `BeamParams` NamedTuple.

**Key transformations:**
- Adjust isocenter: `adjusted_iso = iso - origin`
- Adjust gantry angle: `+180°` (CUDA convention)
- Calculate source position from VSAD and angles

---

#### `_extract_lut_data(beam_model)`
```python
def _extract_lut_data(beam_model) -> LUTData
```

**Purpose:** Extract lookup tables from `IMPTBeamModel` to pure JAX arrays.

---

#### `_extract_spot_data(original_beam, beam_model)`
```python
def _extract_spot_data(...) -> SpotData
```

**Purpose:** Extract and sort spots by energy ID for efficient layer processing.

---

#### `_extract_layer_data(spot_data, beam_model)`
```python
def _extract_layer_data(...) -> LayerData
```

**Purpose:** Construct layer boundaries from sorted spot data.

---

### Main Entry Point

#### `computeIMPTPlanJax(dose_grid, plan)`
```python
def computeIMPTPlanJax(dose_grid: IMPTDoseGrid, plan: IMPTPlan) -> np.ndarray
```

**Purpose:** Drop-in replacement for CUDA `computeIMPTPlan`. Computes complete IMPT dose for all beams.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `dose_grid` | `IMPTDoseGrid` object containing CT/phantom data, RLSP conversion |
| `plan` | `IMPTPlan` object with beam list, machine name, fractionation |

**Output:** 3D numpy array of dose values, shape `(nk, nj, ni)` matching `dose_grid.size` order

**Algorithm:**
```python
1. Validate isotropic spacing
2. Get RLSP from HU using plan.machine_name
3. Create DoseParams from grid dimensions
4. For each beam in plan.beam_list:
   a. Find matching beam model by rangeshifter label
   b. Extract all pure data structures
   c. Call compute_impt_dose_optimized()
   d. Accumulate dose
5. Multiply by plan.n_fractions
6. Return total dose
```

**Usage Example:**
```python
from DoseCUDA.Jax.impt_jax import computeIMPTPlanJax

# Load CT and plan (using existing DoseCUDA infrastructure)
dose_grid = IMPTDoseGrid()
dose_grid.loadFromCT(ct_directory)
plan = IMPTPlan("HitachiProbeatJHU")
plan.loadFromDicom(plan_file)

# Compute dose using JAX backend
dose = computeIMPTPlanJax(dose_grid, plan)

# dose is a 3D numpy array with shape (nz, ny, nx)
```

---

## Function Dependency Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FUNCTION DEPENDENCY GRAPH                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────┐
                              │   computeIMPTPlanJax    │  ◄── MAIN ENTRY POINT
                              │   (User-facing API)     │
                              └───────────┬─────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
    │ _extract_beam_params│   │  _extract_lut_data  │   │ _extract_spot_data  │
    └─────────────────────┘   └─────────────────────┘   └──────────┬──────────┘
                                                                   │
                                                                   ▼
                                                        ┌─────────────────────┐
                                                        │ _extract_layer_data │
                                                        └─────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────┐
                        │     compute_impt_dose_optimized     │  ◄── HIGH-LEVEL API
                        └─────────────────┬───────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           │                           │
    ┌─────────────────────────┐           │                           │
    │  _precompute_all_grids  │◄──────────┤                           │
    │  (Grid precomputation)  │           │                           │
    └─────────────────────────┘           │                           │
              │                           │                           │
              │              ┌────────────┴────────────┐              │
              │              │                         │              │
              ▼              ▼                         ▼              │
    ┌─────────────────────────────┐       ┌─────────────────────────┐ │
    │   compute_raytrace_pure     │       │    compute_dose_pure    │◄┘
    │   (WET calculation)         │       │    (Dose calculation)   │
    └─────────────┬───────────────┘       └───────────┬─────────────┘
                  │                                   │
        ┌─────────┴─────────┐                         │
        │                   │                         │
        ▼                   ▼                         ▼
┌───────────────────┐ ┌───────────────────┐ ┌─────────────────────────────┐
│_raytrace_kernel_  │ │ _smooth_wet_kernel│ │ _pencil_beam_single_layer   │
│    optimized      │ │                   │ │ (Double-Gaussian kernel)    │
└───────────────────┘ └─────────┬─────────┘ └─────────────┬───────────────┘
        │                       │                         │
        │                       │           ┌─────────────┼─────────────┐
        │                       │           │             │             │
        │                       ▼           ▼             ▼             ▼
        │             ┌─────────────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐
        │             │_point_head_to_  │ │  _sqr   │ │_sigma_  │ │_nuclear_  │
        │             │     image       │ │         │ │  air    │ │  halo     │
        │             └─────────────────┘ └─────────┘ └─────────┘ └───────────┘
        │
        ▼
  ┌───────────────────────────┐
  │ jax.scipy.ndimage.        │
  │   map_coordinates         │
  │ (Trilinear interpolation) │
  └───────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   LEGEND                                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐                                                                        │
│  │ Function│  Normal function                                                       │
│  └─────────┘                                                                        │
│                                                                                     │
│       │                                                                             │
│       ▼       Function call / data flow                                             │
│                                                                                     │
│      ◄──     Input dependency                                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Physical Formulas and Algorithms

### 1. Water Equivalent Thickness (WET)

WET is computed by integrating the Relative Linear Stopping Power (RLSP) along the ray from voxel to source:

$$\text{WET}(x,y,z) = \int_0^{d_{source}} \rho_{RLSP}(x + u_x \cdot t, y + u_y \cdot t, z + u_z \cdot t) \, dt$$

where $(u_x, u_y, u_z)$ is the unit vector toward the source.

### 2. Double-Gaussian Lateral Profile

The lateral dose distribution combines a primary (core) Gaussian and a secondary (halo) Gaussian:

$$D_{lateral}(r) = (1 - w_{halo}) \cdot G(r, \sigma_{primary}) + w_{halo} \cdot G(r, \sigma_{halo})$$

where:
$$G(r, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{r^2}{2\sigma^2}\right)$$

### 3. Total Sigma Calculation

$$\sigma_{primary}^2 = \sigma_{MS}^2(\text{WET}) + \sigma_{air}^2(d, R_{80})$$

$$\sigma_{halo}^2 = \sigma_{primary}^2 + \sigma_{nuclear}^2(\text{WET}, R_{80})$$

### 4. Complete Dose Formula

For a spot with monitor units MU at position $(x_s, y_s)$:

$$D(x,y,z) = \text{MU} \times \text{IDD}(\text{WET}) \times D_{lateral}(d_{CAX})$$

where $d_{CAX}$ is the distance from voxel to the central axis of the proton pencil beam.

---

## Performance Considerations

1. **Grid Precomputation:** Computing all voxel grids once saves ~40% computation vs. repeated meshgrid calls.

2. **Memory Optimization:** The `_pencil_beam_single_layer` kernel uses `lax.fori_loop` to process spots sequentially, reducing memory from O(ni×nj×nk×n_spots) to O(ni×nj×nk).

3. **JIT Compilation:** All kernels are JIT-compiled with static array dimensions, enabling efficient XLA optimization.

4. **Static Arguments:** Grid dimensions are marked as `static_argnums` to allow shape-dependent optimizations.

---

## Coordinate System Reference

| System | X | Y | Z | Notes |
|--------|---|---|---|-------|
| Image/DICOM | Right-Left | Anterior-Posterior | Inferior-Superior | Patient coordinates |
| BEV/Head | Lateral (cross-plane) | Depth (beam direction) | Vertical (in-plane) | Rotated by gantry+couch |
| Array Index | i (fastest) | j | k (slowest) | C-order memory layout |

---

## Error Handling

- **Non-isotropic spacing:** Raises `ValueError` if `spacing[0] != spacing[1] != spacing[2]`
- **Missing beam model:** Raises `ValueError` if rangeshifter label not found in plan

---

## License

This code is part of the DoseCUDA project, licensed under GPL-2.0. See the main [LICENSE](../../LICENSE) file for details.

**NOT FOR CLINICAL USE**
