"""Diagnose end-to-end angle derivatives through WET and dose calculation.

Unlike ``validate_frozen_wet_angle_gradients.py``, this script recomputes ray
tracing and WET smoothing at every gantry/couch perturbation.  The step-size
sweep is intentionally diagnostic: discrete voxel bounds and rounded smoothing
indices may prevent the implemented autodiff derivative from matching finite
differences over every angular scale.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from impt_jax import (  # noqa: E402
    _precompute_all_grids,
    _raytrace_kernel,
    _smooth_wet_kernel,
    beam_params_from_angles,
    compute_dose,
)
from validate_frozen_wet_angle_gradients import (  # noqa: E402
    FINITE_DIFFERENCE_STEPS_DEG,
    legacy_geometry_reference,
    relative_l2_error,
)
from validate_spot_weight_gradients import (  # noqa: E402
    SHAPE_ZYX,
    create_case,
    prepare_fixed_data,
)


MAX_FINAL_DOSE_DERIVATIVE_ERROR = 3.0e-3


def main():
    (
        baseline_beam_params,
        dose_params,
        lut_data,
        spot_data,
        layer_data,
        _baseline_grids,
        _baseline_wet,
        probe,
    ) = prepare_fixed_data()
    grid, plan, _beam = create_case()
    density = jnp.asarray(
        grid.RLSPFromHU(plan.machine_name), dtype=jnp.float32
    )
    angles_np, adjusted_iso_np, _legacy_params = legacy_geometry_reference()
    spacing_value = float(dose_params.spacing)
    grid_diagonal_squared = (
        (dose_params.ni * spacing_value) ** 2
        + (dose_params.nj * spacing_value) ** 2
        + (dose_params.nk * spacing_value) ** 2
    )
    max_steps = int(
        float(np.sqrt(np.float32(grid_diagonal_squared))) + 500.0
    ) + 10

    def wet_and_dose_from_angles(angles):
        beam_params = beam_params_from_angles(
            angles[0],
            angles[1],
            jnp.asarray(adjusted_iso_np),
            baseline_beam_params.model_vsadx,
            baseline_beam_params.model_vsady,
        )
        grids = _precompute_all_grids(
            dose_params.ni,
            dose_params.nj,
            dose_params.nk,
            dose_params.spacing,
            beam_params.iso_x,
            beam_params.iso_y,
            beam_params.iso_z,
            beam_params.src_x,
            beam_params.src_y,
            beam_params.src_z,
            beam_params.singa,
            beam_params.cosga,
            beam_params.sinta,
            beam_params.costa,
        )
        raw_wet = _raytrace_kernel(
            dose_params.ni,
            dose_params.nj,
            dose_params.nk,
            dose_params.spacing,
            beam_params.iso_x,
            beam_params.iso_y,
            beam_params.iso_z,
            max_steps,
            density,
            grids.vox_xyz_x,
            grids.vox_xyz_y,
            grids.vox_xyz_z,
            grids.uvec_x,
            grids.uvec_y,
            grids.uvec_z,
        )
        wet = _smooth_wet_kernel(
            dose_params.ni,
            dose_params.nj,
            dose_params.nk,
            dose_params.spacing,
            raw_wet,
            grids.vox_head_x,
            grids.vox_head_y,
            grids.vox_head_z,
            beam_params.singa,
            beam_params.cosga,
            beam_params.sinta,
            beam_params.costa,
            beam_params.iso_x,
            beam_params.iso_y,
            beam_params.iso_z,
        )
        dose = compute_dose(
            beam_params,
            dose_params,
            lut_data,
            spot_data,
            layer_data,
            wet,
            grids,
        )
        return raw_wet, wet, dose

    def objective(angles):
        _raw_wet, _wet, dose = wet_and_dose_from_angles(angles)
        return jnp.sum(dose * probe)

    angles = jnp.asarray(angles_np)
    compiled_outputs = jax.jit(wet_and_dose_from_angles)
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    objective_value, autodiff_gradient = value_and_gradient(angles)
    objective_value.block_until_ready()

    autodiff_np = np.asarray(autodiff_gradient, dtype=np.float64)
    component_scale = np.maximum(
        np.abs(autodiff_np), 0.05 * np.max(np.abs(autodiff_np))
    )
    probe_np = np.asarray(probe, dtype=np.float64)

    component_results = []
    for step in FINITE_DIFFERENCE_STEPS_DEG:
        finite_difference = np.empty_like(autodiff_np)
        for angle_index in range(angles_np.size):
            perturbation = np.zeros_like(angles_np)
            perturbation[angle_index] = np.float32(step)
            _plus_raw_wet, _plus_wet, plus_dose = compiled_outputs(
                jnp.asarray(angles_np + perturbation)
            )
            _minus_raw_wet, _minus_wet, minus_dose = compiled_outputs(
                jnp.asarray(angles_np - perturbation)
            )
            plus = float(jnp.sum(plus_dose * probe))
            minus = float(jnp.sum(minus_dose * probe))
            finite_difference[angle_index] = (plus - minus) / (2.0 * step)
        component_results.append(
            (
                step,
                finite_difference,
                relative_l2_error(finite_difference, autodiff_np),
                float(
                    np.max(
                        np.abs(finite_difference - autodiff_np) / component_scale
                    )
                ),
            )
        )

    direction = np.asarray((0.8, -0.6), dtype=np.float32)
    direction_jax = jnp.asarray(direction)
    (_raw_wet, _wet, _dose), (raw_wet_jvp, wet_jvp, dose_jvp) = jax.jvp(
        wet_and_dose_from_angles,
        (angles,),
        (direction_jax,),
    )
    raw_wet_jvp_np = np.asarray(raw_wet_jvp, dtype=np.float64)
    wet_jvp_np = np.asarray(wet_jvp, dtype=np.float64)
    dose_jvp_np = np.asarray(dose_jvp, dtype=np.float64)
    autodiff_directional = float(np.sum(autodiff_np * direction))

    directional_results = []
    for step in FINITE_DIFFERENCE_STEPS_DEG:
        plus_raw_wet, plus_wet, plus_dose = compiled_outputs(
            angles + np.float32(step) * direction_jax
        )
        minus_raw_wet, minus_wet, minus_dose = compiled_outputs(
            angles - np.float32(step) * direction_jax
        )
        finite_difference_raw_wet = (
            np.asarray(plus_raw_wet, dtype=np.float64)
            - np.asarray(minus_raw_wet, dtype=np.float64)
        ) / (2.0 * step)
        finite_difference_wet = (
            np.asarray(plus_wet, dtype=np.float64)
            - np.asarray(minus_wet, dtype=np.float64)
        ) / (2.0 * step)
        finite_difference_dose = (
            np.asarray(plus_dose, dtype=np.float64)
            - np.asarray(minus_dose, dtype=np.float64)
        ) / (2.0 * step)
        finite_difference_directional = float(
            np.sum(finite_difference_dose * probe_np)
        )
        directional_results.append(
            (
                step,
                finite_difference_directional,
                abs(finite_difference_directional - autodiff_directional)
                / max(abs(autodiff_directional), 1.0e-12),
                relative_l2_error(finite_difference_raw_wet, raw_wet_jvp_np),
                relative_l2_error(finite_difference_wet, wet_jvp_np),
                relative_l2_error(finite_difference_dose, dose_jvp_np),
                float(np.linalg.norm(finite_difference_raw_wet)),
                float(np.linalg.norm(finite_difference_wet)),
                float(np.linalg.norm(finite_difference_dose)),
            )
        )

    best_component = min(component_results, key=lambda result: result[3])
    best_directional = min(directional_results, key=lambda result: result[2])
    best_raw_wet_jvp = min(directional_results, key=lambda result: result[3])
    best_wet_jvp = min(directional_results, key=lambda result: result[4])
    best_dose_jvp = min(directional_results, key=lambda result: result[5])

    print(f"JAX device: {jax.devices()[0]}")
    print(f"shape (z,y,x): {SHAPE_ZYX}")
    print(f"angles (gantry,couch) degrees: {angles_np.tolist()}")
    print(f"objective: {float(objective_value):.9g}")
    print(f"autodiff gradient per degree: {autodiff_np.tolist()}")
    print("component central-difference sweep:")
    for step, finite_difference, l2_error, max_scaled_error in component_results:
        print(
            f"  h={step:g} deg: gradient={finite_difference.tolist()}, "
            f"relative L2 error={l2_error:.6g}, "
            f"max scaled component error={max_scaled_error:.6g}"
        )
    print(f"autodiff directional derivative: {autodiff_directional:.9g}")
    print(f"autodiff raw-WET JVP L2 norm: {np.linalg.norm(raw_wet_jvp_np):.9g}")
    print(f"autodiff smoothed-WET JVP L2 norm: {np.linalg.norm(wet_jvp_np):.9g}")
    print(f"autodiff dose JVP L2 norm: {np.linalg.norm(dose_jvp_np):.9g}")
    print("directional central-difference sweep:")
    for result in directional_results:
        (
            step,
            scalar,
            scalar_error,
            raw_wet_error,
            wet_error,
            dose_error,
            raw_wet_norm,
            wet_norm,
            dose_norm,
        ) = result
        print(
            f"  h={step:g} deg: scalar={scalar:.9g}, "
            f"scalar error={scalar_error:.6g}, raw/smoothed WET JVP errors="
            f"{raw_wet_error:.6g}/{wet_error:.6g}, dose JVP error={dose_error:.6g}, "
            f"FD norms={raw_wet_norm:.6g}/{wet_norm:.6g}/{dose_norm:.6g}"
        )
    print(
        f"best max scaled component error: {best_component[3]:.6g} "
        f"at h={best_component[0]:g} deg"
    )
    print(
        f"best directional scalar error: {best_directional[2]:.6g} "
        f"at h={best_directional[0]:g} deg"
    )
    print(
        f"best raw-WET JVP error: {best_raw_wet_jvp[3]:.6g} "
        f"at h={best_raw_wet_jvp[0]:g} deg"
    )
    print(
        f"best smoothed-WET JVP error: {best_wet_jvp[4]:.6g} "
        f"at h={best_wet_jvp[0]:g} deg"
    )
    print(
        f"best dose JVP error: {best_dose_jvp[5]:.6g} "
        f"at h={best_dose_jvp[0]:g} deg"
    )

    failures = []
    if not np.all(np.isfinite(autodiff_np)):
        failures.append("reverse-mode objective gradient is non-finite")
    if (
        not np.all(np.isfinite(raw_wet_jvp_np))
        or not np.all(np.isfinite(wet_jvp_np))
        or not np.all(np.isfinite(dose_jvp_np))
    ):
        failures.append("a forward-mode JVP is non-finite")
    if best_component[3] > MAX_FINAL_DOSE_DERIVATIVE_ERROR:
        failures.append("objective-angle components disagree with finite differences")
    if best_directional[2] > MAX_FINAL_DOSE_DERIVATIVE_ERROR:
        failures.append("objective directional derivative disagrees with finite differences")
    if best_dose_jvp[5] > MAX_FINAL_DOSE_DERIVATIVE_ERROR:
        failures.append("full-dose JVP disagrees with finite differences")
    if failures:
        raise SystemExit(
            "Full-angle diagnostic failed:\n- " + "\n- ".join(failures)
        )

    print("Full angle-gradient diagnostic passed final-dose checks.")


if __name__ == "__main__":
    main()
