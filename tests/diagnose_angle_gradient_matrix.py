"""Survey full angle derivatives across geometries and material layouts.

Each case recomputes ray tracing, rounded WET smoothing, and dose.  Autodiff
is compared with a central-difference step sweep for both a scalar objective
and the complete dose/WET directional derivatives.
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
    beam_params_from_angles,
    compute_dose,
    compute_raytrace,
)
from validate_frozen_wet_angle_gradients import (  # noqa: E402
    FINITE_DIFFERENCE_STEPS_DEG,
    relative_l2_error,
)
from validate_spot_weight_gradients import (  # noqa: E402
    SHAPE_ZYX,
    create_case,
    prepare_fixed_data,
)


ANGLE_CASES = (
    ("baseline_oblique", (23.0, -11.0)),
    ("opposite_oblique", (37.0, 13.0)),
    ("near_lateral", (91.0, -7.0)),
    ("far_oblique", (143.0, 22.0)),
)
ROI_OBJECTIVE_NAMES = ("target_mean", "target_square", "oar_square", "combined")
MAX_PROBE_COMPONENT_ERROR = 5.0e-3
MAX_BAO_VECTOR_ERROR = 5.0e-3
MAX_SCALAR_DIRECTIONAL_ERROR = 5.0e-3
MAX_DOSE_JVP_ERROR = 4.0e-2
MAX_ROI_RELATIVE_ERROR = 5.0e-3
MAX_ROI_ABSOLUTE_ERROR = 2.0e-9


def roi_objectives(dose, target_mask, oar_mask, target_rx):
    target_mean = np.mean(dose[target_mask])
    target_square = np.mean((dose[target_mask] - target_rx) ** 2)
    oar_square = np.mean(dose[oar_mask] ** 2)
    return np.asarray(
        (
            target_mean,
            target_square,
            oar_square,
            target_square + 10.0 * oar_square,
        ),
        dtype=np.float64,
    )


def roi_objective_directionals(
    dose, dose_jvp, target_mask, oar_mask, target_rx
):
    target_mean = np.mean(dose_jvp[target_mask])
    target_square = np.mean(
        2.0 * (dose[target_mask] - target_rx) * dose_jvp[target_mask]
    )
    oar_square = np.mean(2.0 * dose[oar_mask] * dose_jvp[oar_mask])
    return np.asarray(
        (
            target_mean,
            target_square,
            oar_square,
            target_square + 10.0 * oar_square,
        ),
        dtype=np.float64,
    )


def create_densities():
    homogeneous_grid, homogeneous_plan, _beam = create_case()
    homogeneous = jnp.asarray(
        homogeneous_grid.RLSPFromHU(homogeneous_plan.machine_name),
        dtype=jnp.float32,
    )

    heterogeneous_grid, heterogeneous_plan, _beam = create_case()
    nz, ny, nx = heterogeneous_grid.HU.shape
    edge = max(2, round(5.0 / float(heterogeneous_grid.spacing[0])))
    heterogeneous_grid.HU[
        nz // 4:nz // 4 + 3, edge:-edge, edge:-edge
    ] = 850.0
    heterogeneous_grid.HU[
        edge:-edge, ny // 2 - 2:ny // 2 + 2, edge:-edge
    ] = -450.0
    heterogeneous_grid.HU[
        edge:-edge, edge:-edge, 3 * nx // 4:3 * nx // 4 + 3
    ] = 300.0
    heterogeneous = jnp.asarray(
        heterogeneous_grid.RLSPFromHU(heterogeneous_plan.machine_name),
        dtype=jnp.float32,
    )
    return (("homogeneous", homogeneous), ("heterogeneous", heterogeneous))


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
    adjusted_iso = jnp.asarray(
        (
            baseline_beam_params.iso_x,
            baseline_beam_params.iso_y,
            baseline_beam_params.iso_z,
        ),
        dtype=jnp.float32,
    )

    def wet_and_dose(angles, density):
        beam_params = beam_params_from_angles(
            angles[0],
            angles[1],
            adjusted_iso,
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
        wet = compute_raytrace(beam_params, dose_params, density, grids)
        dose = compute_dose(
            beam_params,
            dose_params,
            lut_data,
            spot_data,
            layer_data,
            wet,
            grids,
        )
        return wet, dose

    def objective(angles, density):
        _wet, dose = wet_and_dose(angles, density)
        return jnp.sum(dose * probe)

    compiled_outputs = jax.jit(wet_and_dose)
    value_and_gradient = jax.jit(jax.value_and_grad(objective, argnums=0))

    @jax.jit
    def outputs_and_directional_jvp(angles, density, direction):
        return jax.jvp(
            lambda candidate: wet_and_dose(candidate, density),
            (angles,),
            (direction,),
        )

    direction = jnp.asarray((0.8, -0.6), dtype=jnp.float32)
    probe_np = np.asarray(probe, dtype=np.float64)
    z, y, x = np.ogrid[
        :dose_params.ni, :dose_params.nj, :dose_params.nk
    ]
    target_mask = (
        (z - 0.58 * dose_params.ni) ** 2
        + (y - 0.44 * dose_params.nj) ** 2
        + (x - 0.61 * dose_params.nk) ** 2
    ) <= 4.5 ** 2
    oar_mask = (
        (z - 0.38 * dose_params.ni) ** 2
        + (y - 0.68 * dose_params.nj) ** 2
        + (x - 0.35 * dose_params.nk) ** 2
    ) <= 4.0 ** 2
    target_mask_jax = jnp.asarray(target_mask)
    oar_mask_jax = jnp.asarray(oar_mask)

    def bao_objective(angles, density, target_rx):
        _wet, dose = wet_and_dose(angles, density)
        target_square = jnp.mean((dose[target_mask_jax] - target_rx) ** 2)
        oar_square = jnp.mean(dose[oar_mask_jax] ** 2)
        return target_square + 10.0 * oar_square

    bao_value_and_gradient = jax.jit(
        jax.value_and_grad(bao_objective, argnums=0)
    )
    results = []

    for material_name, density in create_densities():
        for angle_name, angle_values in ANGLE_CASES:
            angles_np = np.asarray(angle_values, dtype=np.float32)
            angles = jnp.asarray(angles_np)
            objective_value, autodiff_gradient = value_and_gradient(angles, density)
            objective_value.block_until_ready()
            autodiff_np = np.asarray(autodiff_gradient, dtype=np.float64)
            component_scale = np.maximum(
                np.abs(autodiff_np), 0.05 * np.max(np.abs(autodiff_np))
            )
            _baseline_wet_for_rx, baseline_dose_for_rx = compiled_outputs(
                angles, density
            )
            baseline_dose_for_rx_np = np.asarray(
                baseline_dose_for_rx, dtype=np.float64
            )
            target_rx = 0.8 * np.max(baseline_dose_for_rx_np[target_mask])
            bao_value, bao_gradient = bao_value_and_gradient(
                angles, density, jnp.asarray(target_rx, dtype=jnp.float32)
            )
            bao_value.block_until_ready()
            bao_gradient_np = np.asarray(bao_gradient, dtype=np.float64)
            bao_component_scale = np.maximum(
                np.abs(bao_gradient_np),
                0.05 * np.max(np.abs(bao_gradient_np)),
            )

            component_errors = []
            bao_component_errors = []
            for step in FINITE_DIFFERENCE_STEPS_DEG:
                finite_difference = np.empty_like(autodiff_np)
                bao_finite_difference = np.empty_like(bao_gradient_np)
                for angle_index in range(angles_np.size):
                    perturbation = np.zeros_like(angles_np)
                    perturbation[angle_index] = np.float32(step)
                    _plus_wet, plus_dose = compiled_outputs(
                        jnp.asarray(angles_np + perturbation), density
                    )
                    _minus_wet, minus_dose = compiled_outputs(
                        jnp.asarray(angles_np - perturbation), density
                    )
                    plus = float(jnp.sum(plus_dose * probe))
                    minus = float(jnp.sum(minus_dose * probe))
                    finite_difference[angle_index] = (
                        plus - minus
                    ) / (2.0 * step)
                    plus_bao = roi_objectives(
                        np.asarray(plus_dose, dtype=np.float64),
                        target_mask,
                        oar_mask,
                        target_rx,
                    )[3]
                    minus_bao = roi_objectives(
                        np.asarray(minus_dose, dtype=np.float64),
                        target_mask,
                        oar_mask,
                        target_rx,
                    )[3]
                    bao_finite_difference[angle_index] = (
                        plus_bao - minus_bao
                    ) / (2.0 * step)
                component_errors.append(
                    (
                        step,
                        float(
                            np.max(
                                np.abs(finite_difference - autodiff_np)
                                / component_scale
                            )
                        ),
                    )
                )
                bao_component_errors.append(
                    (
                        step,
                        float(
                            np.max(
                                np.abs(bao_finite_difference - bao_gradient_np)
                                / bao_component_scale
                            )
                        ),
                        bao_finite_difference.copy(),
                        relative_l2_error(
                            bao_finite_difference, bao_gradient_np
                        ),
                    )
                )

            (_baseline_wet, baseline_dose), (wet_jvp, dose_jvp) = (
                outputs_and_directional_jvp(
                    angles, density, direction
                )
            )
            wet_jvp_np = np.asarray(wet_jvp, dtype=np.float64)
            dose_jvp_np = np.asarray(dose_jvp, dtype=np.float64)
            baseline_dose_np = np.asarray(baseline_dose, dtype=np.float64)
            dose_support = baseline_dose_np > 1.0e-3 * np.max(baseline_dose_np)
            autodiff_roi_directionals = roi_objective_directionals(
                baseline_dose_np,
                dose_jvp_np,
                target_mask,
                oar_mask,
                target_rx,
            )
            autodiff_directional = float(
                np.sum(autodiff_np * np.asarray(direction))
            )
            directional_errors = []
            roi_errors = []

            for step in FINITE_DIFFERENCE_STEPS_DEG:
                plus_wet, plus_dose = compiled_outputs(
                    angles + np.float32(step) * direction, density
                )
                minus_wet, minus_dose = compiled_outputs(
                    angles - np.float32(step) * direction, density
                )
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
                finite_difference_roi = (
                    roi_objectives(
                        np.asarray(plus_dose, dtype=np.float64),
                        target_mask,
                        oar_mask,
                        target_rx,
                    )
                    - roi_objectives(
                        np.asarray(minus_dose, dtype=np.float64),
                        target_mask,
                        oar_mask,
                        target_rx,
                    )
                ) / (2.0 * step)
                roi_errors.append(
                    (
                        step,
                        np.abs(
                            finite_difference_roi - autodiff_roi_directionals
                        )
                        / np.maximum(
                            np.abs(autodiff_roi_directionals), 1.0e-12
                        ),
                        finite_difference_roi,
                    )
                )
                directional_errors.append(
                    (
                        step,
                        abs(finite_difference_directional - autodiff_directional)
                        / max(abs(autodiff_directional), 1.0e-12),
                        relative_l2_error(finite_difference_wet, wet_jvp_np),
                        relative_l2_error(finite_difference_dose, dose_jvp_np),
                        relative_l2_error(
                            finite_difference_wet[dose_support],
                            wet_jvp_np[dose_support],
                        ),
                        relative_l2_error(
                            finite_difference_dose[dose_support],
                            dose_jvp_np[dose_support],
                        ),
                    )
                )

            best_component = min(component_errors, key=lambda result: result[1])
            best_bao_component = min(
                bao_component_errors, key=lambda result: result[1]
            )
            best_bao_vector = min(
                bao_component_errors, key=lambda result: result[3]
            )
            best_scalar = min(directional_errors, key=lambda result: result[1])
            best_wet = min(directional_errors, key=lambda result: result[2])
            best_dose = min(directional_errors, key=lambda result: result[3])
            best_wet_support = min(
                directional_errors, key=lambda result: result[4]
            )
            best_dose_support = min(
                directional_errors, key=lambda result: result[5]
            )
            best_roi = []
            for objective_index, objective_name in enumerate(ROI_OBJECTIVE_NAMES):
                step, errors, finite_difference_roi = min(
                    roi_errors, key=lambda result: result[1][objective_index]
                )
                best_roi.append(
                    (
                        objective_name,
                        step,
                        float(errors[objective_index]),
                        float(autodiff_roi_directionals[objective_index]),
                        float(finite_difference_roi[objective_index]),
                    )
                )
            worst_roi = max(best_roi, key=lambda result: result[2])
            results.append(
                {
                    "name": f"{material_name}/{angle_name}",
                    "objective": float(objective_value),
                    "gradient": autodiff_np,
                    "component": best_component,
                    "bao_component": best_bao_component,
                    "bao_vector": best_bao_vector,
                    "bao_gradient": bao_gradient_np,
                    "scalar": best_scalar,
                    "wet": best_wet,
                    "dose": best_dose,
                    "wet_support": best_wet_support,
                    "dose_support": best_dose_support,
                    "roi": best_roi,
                    "worst_roi": worst_roi,
                }
            )

    print(f"JAX device: {jax.devices()[0]}")
    print(f"shape (z,y,x): {SHAPE_ZYX}")
    print(
        "case                              objective    gradient [gantry,couch]       "
        "probe%/BAO-vector% (h)  scalar% (h)  dose%/support%  worst ROI% (name,h)"
    )
    print("-" * 150)
    for result in results:
        component_step, component_error = result["component"]
        (
            bao_vector_step,
            _bao_vector_component,
            _bao_vector_finite_difference,
            bao_vector_error,
        ) = result["bao_vector"]
        scalar_step, scalar_error, *_scalar_rest = result["scalar"]
        dose_step, _dose_scalar, _dose_wet, dose_error, *_dose_rest = result["dose"]
        dose_support_step = result["dose_support"][0]
        dose_support_error = result["dose_support"][5]
        roi_name, roi_step, roi_error, _roi_ad, _roi_fd = result["worst_roi"]
        gradient = result["gradient"]
        print(
            f"{result['name']:33} {result['objective']:10.7f}  "
            f"[{gradient[0]: .6e},{gradient[1]: .6e}]  "
            f"{100.0 * component_error:7.3f}({component_step:g})/"
            f"{100.0 * bao_vector_error:7.3f}({bao_vector_step:g})  "
            f"{100.0 * scalar_error:8.4f} ({scalar_step:g})  "
            f"{100.0 * dose_error:7.3f}({dose_step:g})/"
            f"{100.0 * dose_support_error:7.3f}({dose_support_step:g})  "
            f"{100.0 * roi_error:8.3f} ({roi_name},{roi_step:g})"
        )

    print("\nSmoothed-WET derivative error (full volume / dose support):")
    for result in results:
        wet_step, _wet_scalar, wet_error, *_wet_rest = result["wet"]
        wet_support_step = result["wet_support"][0]
        wet_support_error = result["wet_support"][4]
        print(
            f"  {result['name']:33} {100.0 * wet_error:8.3f}% ({wet_step:g}) / "
            f"{100.0 * wet_support_error:8.3f}% ({wet_support_step:g})"
        )

    print("\nWorst ROI-objective derivative details:")
    for result in results:
        roi_name, roi_step, roi_error, roi_ad, roi_fd = result["worst_roi"]
        print(
            f"  {result['name']:33} {roi_name:13} h={roi_step:g}: "
            f"autodiff={roi_ad:.9g}, finite-difference={roi_fd:.9g}, "
            f"relative error={100.0 * roi_error:.4f}%"
        )

    print("\nBAO-style reverse-gradient details:")
    for result in results:
        step, error, finite_difference, vector_error = result["bao_component"]
        gradient = result["bao_gradient"]
        print(
            f"  {result['name']:33} h={step:g}: "
            f"autodiff={gradient.tolist()}, finite-difference="
            f"{finite_difference.tolist()}, max scaled error={100.0 * error:.4f}%, "
            f"vector L2 error={100.0 * vector_error:.4f}%"
        )

    failures = []
    for result in results:
        name = result["name"]
        if (
            not np.all(np.isfinite(result["gradient"]))
            or not np.all(np.isfinite(result["bao_gradient"]))
        ):
            failures.append(f"{name}: a reverse-mode gradient is non-finite")
        if result["component"][1] > MAX_PROBE_COMPONENT_ERROR:
            failures.append(f"{name}: probe-objective component error is too large")
        if result["bao_vector"][3] > MAX_BAO_VECTOR_ERROR:
            failures.append(f"{name}: BAO gradient-vector error is too large")
        if result["scalar"][1] > MAX_SCALAR_DIRECTIONAL_ERROR:
            failures.append(f"{name}: scalar directional error is too large")
        if result["dose"][3] > MAX_DOSE_JVP_ERROR:
            failures.append(f"{name}: voxel-dose JVP error is too large")
        for objective_name, _step, _relative_error, autodiff, finite_difference in result["roi"]:
            allowed_error = (
                MAX_ROI_ABSOLUTE_ERROR
                + MAX_ROI_RELATIVE_ERROR * abs(autodiff)
            )
            if abs(finite_difference - autodiff) > allowed_error:
                failures.append(
                    f"{name}: {objective_name} directional error is too large"
                )
    if failures:
        raise SystemExit(
            "Angle-gradient matrix failed:\n- " + "\n- ".join(failures)
        )

    print("Angle-gradient matrix passed final-dose and ROI-objective checks.")


if __name__ == "__main__":
    main()
