"""
Profile JAX IMPT dose calculation to identify bottlenecks.
"""
import os
import sys
import time
import numpy as np

# Add Jax folder to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
jax_dir = os.path.join(os.path.dirname(script_dir), "DoseCUDA", "Jax")
sys.path.insert(0, jax_dir)

from DoseCUDA import IMPTDoseGrid, IMPTPlan, IMPTBeam
import jax
import jax.numpy as jnp

# Import internal functions for profiling
from impt_jax_fix import (
    _precompute_all_grids,
    _raytrace_kernel,
    _smooth_wet_kernel,
    _pencil_beam_single_layer,
    _extract_beam_params,
    _extract_lut_data,
    _extract_spot_data,
    _extract_layer_data,
    DoseParams,
    computeIMPTPlanJax
)


def create_test_plan():
    """Create test plan with 98 spots."""
    plan = IMPTPlan()
    beam = IMPTBeam()
    
    beam.dicom_rangeshifter_label = '0'
    n_spots = 98
    for energy_id in range(n_spots):
        theta = 2.0 * 3.14159 * energy_id / n_spots
        spot_x = 100.0 * np.cos(theta)
        spot_y = 100.0 * np.sin(theta)
        mu = 0.2
        beam.addSingleSpot(spot_x, spot_y, mu, energy_id)
    
    plan.addBeam(beam)
    return plan


def profile_step(name, func, *args, n_runs=3, warmup=1):
    """Profile a single step with warmup and multiple runs."""
    # Warmup runs (includes JIT compilation)
    for _ in range(warmup):
        result = func(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  {name}: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    return result, avg_time


def main():
    print("=" * 60)
    print("JAX IMPT Profiling")
    print("=" * 60)
    
    # Setup
    dose_grid = IMPTDoseGrid()
    dose_grid.createCubePhantom()  # Default 138x138x138
    plan = create_test_plan()
    
    print(f"\nPhantom size: {dose_grid.size}")
    print(f"Number of voxels: {np.prod(dose_grid.size):,}")
    print(f"Number of spots: 98")
    print(f"Number of energy layers: 98")
    
    # Get beam model
    beam = plan.beam_list[0]
    model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
        beam.dicom_rangeshifter_label
    )
    beam_model = plan.beam_models[model_index]
    
    # Extract data structures
    beam_params = _extract_beam_params(beam, beam_model, dose_grid.origin)
    lut_data = _extract_lut_data(beam_model)
    spot_data = _extract_spot_data(beam, beam_model)
    layer_data = _extract_layer_data(spot_data, beam_model)
    
    dose_params = DoseParams(
        ni=int(dose_grid.size[0]),
        nj=int(dose_grid.size[1]),
        nk=int(dose_grid.size[2]),
        spacing=jnp.array(float(dose_grid.spacing[0]), dtype=jnp.float32)
    )
    
    # Get density array
    rlsp = dose_grid.RLSPFromHU(plan.machine_name)
    density_array = jnp.array(rlsp, dtype=jnp.float32)
    
    ni, nj, nk = dose_params.ni, dose_params.nj, dose_params.nk
    spacing = dose_params.spacing
    
    print("\n" + "-" * 60)
    print("STEP-BY-STEP PROFILING (3 runs each, after warmup)")
    print("-" * 60)
    
    # 1. Grid precomputation
    grids, t1 = profile_step(
        "1. Precompute grids",
        lambda: _precompute_all_grids(
            ni, nj, nk, spacing,
            beam_params.iso_x, beam_params.iso_y, beam_params.iso_z,
            beam_params.src_x, beam_params.src_y, beam_params.src_z,
            beam_params.singa, beam_params.cosga,
            beam_params.sinta, beam_params.costa
        )
    )
    
    # Calculate max_steps for raytrace
    spacing_val = float(spacing)
    max_dist = float(jnp.sqrt(
        (ni * spacing_val)**2 + 
        (nj * spacing_val)**2 + 
        (nk * spacing_val)**2
    )) + 500.0
    max_steps = int(max_dist) + 10
    print(f"     (max_steps = {max_steps})")
    
    # 2. Ray tracing
    raw_wet, t2 = profile_step(
        "2. Ray tracing",
        lambda: _raytrace_kernel(
            ni, nj, nk, spacing,
            beam_params.iso_x, beam_params.iso_y, beam_params.iso_z,
            max_steps,
            density_array,
            grids.vox_xyz_x, grids.vox_xyz_y, grids.vox_xyz_z,
            grids.uvec_x, grids.uvec_y, grids.uvec_z
        )
    )
    
    # 3. WET smoothing
    smoothed_wet, t3 = profile_step(
        "3. WET smoothing",
        lambda: _smooth_wet_kernel(
            ni, nj, nk, spacing,
            raw_wet,
            grids.vox_head_x, grids.vox_head_y, grids.vox_head_z,
            beam_params.singa, beam_params.cosga,
            beam_params.sinta, beam_params.costa,
            beam_params.iso_x, beam_params.iso_y, beam_params.iso_z
        )
    )
    
    wet_array = jnp.maximum(smoothed_wet, 0.0)
    
    # 4. Pencil beam for single layer (to get per-layer timing)
    # Use first layer with spots
    for layer_id in range(layer_data.n_layers):
        n_layer_spots = int(layer_data.layers_n_spots[layer_id])
        if n_layer_spots > 0:
            energy_id = int(layer_data.layers_energy_id[layer_id])
            r80 = layer_data.layers_r80[layer_id]
            base_idx = energy_id * lut_data.dvp_len
            coef0 = lut_data.divergence_params[base_idx + 2]
            coef1 = lut_data.divergence_params[base_idx + 3]
            coef2 = lut_data.divergence_params[base_idx + 4]
            depths = lut_data.lut_depths[energy_id]
            sigmas = lut_data.lut_sigmas[energy_id]
            idds = lut_data.lut_idds[energy_id]
            
            spot_start = int(layer_data.layers_spot_start[layer_id])
            spot_end = spot_start + n_layer_spots
            spots_x = spot_data.spots_x[spot_start:spot_end]
            spots_y = spot_data.spots_y[spot_start:spot_end]
            spots_mu = spot_data.spots_mu[spot_start:spot_end]
            
            _, t4_single = profile_step(
                f"4. Pencil beam (1 layer, {n_layer_spots} spot(s))",
                lambda: _pencil_beam_single_layer(
                    ni, nj, nk, lut_data.lut_len, spacing,
                    beam_params.model_vsadx, beam_params.model_vsady,
                    grids.vox_head_x, grids.vox_head_y, grids.vox_head_z,
                    grids.distance_to_source,
                    wet_array, r80,
                    depths, sigmas, idds,
                    coef0, coef1, coef2,
                    spots_x, spots_y, spots_mu
                )
            )
            break
    
    # 5. Full pencil beam (all layers)
    print(f"\n  5. Pencil beam (all {layer_data.n_layers} layers):")
    
    # Warmup
    total_dose = jnp.zeros((ni, nj, nk), dtype=jnp.float32)
    for layer_id in range(layer_data.n_layers):
        energy_id = int(layer_data.layers_energy_id[layer_id])
        r80 = layer_data.layers_r80[layer_id]
        base_idx = energy_id * lut_data.dvp_len
        coef0 = lut_data.divergence_params[base_idx + 2]
        coef1 = lut_data.divergence_params[base_idx + 3]
        coef2 = lut_data.divergence_params[base_idx + 4]
        depths = lut_data.lut_depths[energy_id]
        sigmas = lut_data.lut_sigmas[energy_id]
        idds = lut_data.lut_idds[energy_id]
        
        spot_start = int(layer_data.layers_spot_start[layer_id])
        n_layer_spots = int(layer_data.layers_n_spots[layer_id])
        spot_end = spot_start + n_layer_spots
        spots_x = spot_data.spots_x[spot_start:spot_end]
        spots_y = spot_data.spots_y[spot_start:spot_end]
        spots_mu = spot_data.spots_mu[spot_start:spot_end]
        
        layer_dose = _pencil_beam_single_layer(
            ni, nj, nk, lut_data.lut_len, spacing,
            beam_params.model_vsadx, beam_params.model_vsady,
            grids.vox_head_x, grids.vox_head_y, grids.vox_head_z,
            grids.distance_to_source,
            wet_array, r80,
            depths, sigmas, idds,
            coef0, coef1, coef2,
            spots_x, spots_y, spots_mu
        )
        total_dose = total_dose + layer_dose
    total_dose.block_until_ready()
    
    # Timed run
    times = []
    for _ in range(3):
        start = time.perf_counter()
        total_dose = jnp.zeros((ni, nj, nk), dtype=jnp.float32)
        for layer_id in range(layer_data.n_layers):
            energy_id = int(layer_data.layers_energy_id[layer_id])
            r80 = layer_data.layers_r80[layer_id]
            base_idx = energy_id * lut_data.dvp_len
            coef0 = lut_data.divergence_params[base_idx + 2]
            coef1 = lut_data.divergence_params[base_idx + 3]
            coef2 = lut_data.divergence_params[base_idx + 4]
            depths = lut_data.lut_depths[energy_id]
            sigmas = lut_data.lut_sigmas[energy_id]
            idds = lut_data.lut_idds[energy_id]
            
            spot_start = int(layer_data.layers_spot_start[layer_id])
            n_layer_spots = int(layer_data.layers_n_spots[layer_id])
            spot_end = spot_start + n_layer_spots
            spots_x = spot_data.spots_x[spot_start:spot_end]
            spots_y = spot_data.spots_y[spot_start:spot_end]
            spots_mu = spot_data.spots_mu[spot_start:spot_end]
            
            layer_dose = _pencil_beam_single_layer(
                ni, nj, nk, lut_data.lut_len, spacing,
                beam_params.model_vsadx, beam_params.model_vsady,
                grids.vox_head_x, grids.vox_head_y, grids.vox_head_z,
                grids.distance_to_source,
                wet_array, r80,
                depths, sigmas, idds,
                coef0, coef1, coef2,
                spots_x, spots_y, spots_mu
            )
            total_dose = total_dose + layer_dose
        total_dose.block_until_ready()
        times.append(time.perf_counter() - start)
    
    t5 = np.mean(times)
    print(f"     Total: {t5*1000:.2f} ms (±{np.std(times)*1000:.2f} ms)")
    print(f"     Per layer avg: {t5*1000/layer_data.n_layers:.2f} ms")
    
    # Summary
    total_component_time = t1 + t2 + t3 + t5
    print("\n" + "-" * 60)
    print("TIMING SUMMARY")
    print("-" * 60)
    print(f"  Grid precompute:    {t1*1000:8.2f} ms ({100*t1/total_component_time:5.1f}%)")
    print(f"  Ray tracing:        {t2*1000:8.2f} ms ({100*t2/total_component_time:5.1f}%)")
    print(f"  WET smoothing:      {t3*1000:8.2f} ms ({100*t3/total_component_time:5.1f}%)")
    print(f"  Pencil beam (all):  {t5*1000:8.2f} ms ({100*t5/total_component_time:5.1f}%)")
    print(f"  ---------------------------------")
    print(f"  Total components:   {total_component_time*1000:8.2f} ms")
    
    # Full end-to-end timing
    print("\n" + "-" * 60)
    print("END-TO-END TIMING (computeIMPTPlanJax)")
    print("-" * 60)
    
    # Warmup
    _ = computeIMPTPlanJax(dose_grid, plan)
    
    # Timed runs
    times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = computeIMPTPlanJax(dose_grid, plan)
        times.append(time.perf_counter() - start)
    
    print(f"  End-to-end: {np.mean(times)*1000:.2f} ms (±{np.std(times)*1000:.2f} ms)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
