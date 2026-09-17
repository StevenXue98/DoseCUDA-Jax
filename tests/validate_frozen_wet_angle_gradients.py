"""Validate gantry/couch autodiff through the smooth dose geometry path.

The WET array is deliberately held fixed at the baseline beam angle.  This
test covers angle construction, image/head coordinate transforms, source
geometry, and pencil-beam dose.  It does not validate differentiation through
ray tracing or WET smoothing.
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
)
from validate_spot_weight_gradients import (  # noqa: E402
    SHAPE_ZYX,
    create_case,
    prepare_fixed_data,
)


FINITE_DIFFERENCE_STEPS_DEG = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def relative_l2_error(actual, expected):
    numerator = np.linalg.norm(np.asarray(actual) - np.asarray(expected))
    denominator = max(np.linalg.norm(np.asarray(expected)), 1.0e-12)
    return float(numerator / denominator)


def legacy_geometry_reference():
    """Evaluate the pre-refactor beam-parameter formulas independently."""
    grid, plan, beam = create_case()
    model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
        beam.dicom_rangeshifter_label
    )
    beam_model = plan.beam_models[model_index]

    adjusted_iso = np.asarray(beam.iso, dtype=np.float32) - np.asarray(
        grid.origin, dtype=np.float32
    )
    adjusted_gantry = (float(beam.gantry_angle) + 180.0) % 360.0
    couch_angle = float(beam.couch_angle)
    src_dist = (float(beam_model.VSADX) + float(beam_model.VSADY)) / 2.0

    ga = jnp.deg2rad(adjusted_gantry)
    ta = jnp.deg2rad(couch_angle)
    singa = jnp.sin(ga)
    cosga = jnp.cos(ga)
    sinta = jnp.sin(ta)
    costa = jnp.cos(ta)
    xg = -src_dist * singa
    yg = src_dist * cosga

    values = (
        adjusted_iso[0],
        adjusted_iso[1],
        adjusted_iso[2],
        xg * costa,
        yg,
        -xg * sinta,
        singa,
        cosga,
        sinta,
        costa,
        np.float32(beam_model.VSADX),
        np.float32(beam_model.VSADY),
    )
    angles = np.asarray(
        (beam.gantry_angle, beam.couch_angle), dtype=np.float32
    )
    return angles, adjusted_iso, np.asarray(values, dtype=np.float32)


def beam_params_as_array(beam_params):
    return np.asarray(tuple(beam_params), dtype=np.float32)


def main():
    (
        baseline_beam_params,
        dose_params,
        lut_data,
        spot_data,
        layer_data,
        _baseline_grids,
        fixed_wet,
        probe,
    ) = prepare_fixed_data()
    angles_np, adjusted_iso_np, legacy_params = legacy_geometry_reference()

    def beam_from_angles(angles):
        return beam_params_from_angles(
            angles[0],
            angles[1],
            jnp.asarray(adjusted_iso_np),
            baseline_beam_params.model_vsadx,
            baseline_beam_params.model_vsady,
        )

    def dose_from_angles(angles):
        beam_params = beam_from_angles(angles)
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
        return compute_dose(
            beam_params,
            dose_params,
            lut_data,
            spot_data,
            layer_data,
            fixed_wet,
            grids,
        )

    def objective(angles):
        return jnp.sum(dose_from_angles(angles) * probe)

    angles = jnp.asarray(angles_np)
    generated_params = beam_params_as_array(beam_from_angles(angles))
    baseline_params = beam_params_as_array(baseline_beam_params)
    legacy_max_abs = float(np.max(np.abs(generated_params - legacy_params)))
    extractor_max_abs = float(np.max(np.abs(generated_params - baseline_params)))

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    objective_value, autodiff_gradient = value_and_gradient(angles)
    objective_value.block_until_ready()
    autodiff_np = np.asarray(autodiff_gradient, dtype=np.float64)
    component_scale = np.maximum(
        np.abs(autodiff_np), 0.05 * np.max(np.abs(autodiff_np))
    )

    component_results = []
    for step in FINITE_DIFFERENCE_STEPS_DEG:
        finite_difference = np.empty_like(autodiff_np)
        for angle_index in range(angles_np.size):
            perturbation = np.zeros_like(angles_np)
            perturbation[angle_index] = np.float32(step)
            plus = float(objective(jnp.asarray(angles_np + perturbation)))
            minus = float(objective(jnp.asarray(angles_np - perturbation)))
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
    _, dose_jvp = jax.jvp(dose_from_angles, (angles,), (direction_jax,))
    dose_jvp_np = np.asarray(dose_jvp, dtype=np.float64)
    autodiff_directional = float(np.sum(autodiff_np * direction))

    directional_results = []
    for step in FINITE_DIFFERENCE_STEPS_DEG:
        plus_dose = np.asarray(
            dose_from_angles(angles + np.float32(step) * direction_jax),
            dtype=np.float64,
        )
        minus_dose = np.asarray(
            dose_from_angles(angles - np.float32(step) * direction_jax),
            dtype=np.float64,
        )
        finite_difference_dose = (plus_dose - minus_dose) / (2.0 * step)
        finite_difference_directional = float(
            np.sum(finite_difference_dose * np.asarray(probe))
        )
        directional_results.append(
            (
                step,
                finite_difference_directional,
                abs(finite_difference_directional - autodiff_directional)
                / max(abs(autodiff_directional), 1.0e-12),
                relative_l2_error(finite_difference_dose, dose_jvp_np),
            )
        )

    best_component = min(component_results, key=lambda result: result[3])
    best_directional = min(directional_results, key=lambda result: result[2])
    best_dose_jvp = min(directional_results, key=lambda result: result[3])

    print(f"JAX device: {jax.devices()[0]}")
    print(f"shape (z,y,x): {SHAPE_ZYX}")
    print(f"angles (gantry,couch) degrees: {angles_np.tolist()}")
    print(f"legacy geometry max absolute difference: {legacy_max_abs:.9g}")
    print(f"extractor geometry max absolute difference: {extractor_max_abs:.9g}")
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
    print("directional central-difference sweep:")
    for step, finite_difference, scalar_error, dose_error in directional_results:
        print(
            f"  h={step:g} deg: scalar={finite_difference:.9g}, "
            f"scalar relative error={scalar_error:.6g}, "
            f"full-dose JVP relative L2 error={dose_error:.6g}"
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
        f"best full-dose JVP error: {best_dose_jvp[3]:.6g} "
        f"at h={best_dose_jvp[0]:g} deg"
    )

    failures = []
    if legacy_max_abs != 0.0 or extractor_max_abs != 0.0:
        failures.append("traceable angle construction changed baseline beam geometry")
    if not np.all(np.isfinite(autodiff_np)):
        failures.append("autodiff gradients must be finite")
    if np.linalg.norm(autodiff_np) <= 1.0e-8:
        failures.append("the validation objective has a negligible angle gradient")
    if best_component[3] > 3.0e-3:
        failures.append("angle derivatives disagree with central differences")
    if best_directional[2] > 3.0e-3:
        failures.append("directional derivative disagrees with central differences")
    if best_dose_jvp[3] > 3.0e-3:
        failures.append("full-dose angle JVP disagrees with central differences")
    if failures:
        raise SystemExit(
            "Frozen-WET angle gradient validation failed:\n- "
            + "\n- ".join(failures)
        )

    print("Frozen-WET angle gradient validation passed.")


if __name__ == "__main__":
    main()
