"""Validate JAX spot-position derivatives against central differences.

Beam geometry, WET, energy layers, and spot weights are fixed.  Only the
scanning-magnet spot coordinates are traced.  This isolates the smooth
lateral pencil-beam calculation from the discrete WET ray-tracing path that
will need separate treatment for beam-angle optimization.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from impt_jax import compute_dose  # noqa: E402
from validate_spot_weight_gradients import (  # noqa: E402
    SHAPE_ZYX,
    prepare_fixed_data,
)


FINITE_DIFFERENCE_STEPS_MM = (0.5, 0.25, 0.125, 0.0625, 0.03125)


def relative_l2_error(actual, expected):
    numerator = np.linalg.norm(np.asarray(actual) - np.asarray(expected))
    denominator = max(np.linalg.norm(np.asarray(expected)), 1.0e-12)
    return float(numerator / denominator)


def main():
    (
        beam_params,
        dose_params,
        lut_data,
        spot_data,
        layer_data,
        grids,
        wet,
        probe,
    ) = prepare_fixed_data()

    def dose_from_positions(positions):
        variable_spots = spot_data._replace(
            spots_x=positions[:, 0],
            spots_y=positions[:, 1],
        )
        return compute_dose(
            beam_params,
            dose_params,
            lut_data,
            variable_spots,
            layer_data,
            wet,
            grids,
        )

    def objective(positions):
        return jnp.sum(dose_from_positions(positions) * probe)

    positions = jnp.stack((spot_data.spots_x, spot_data.spots_y), axis=1)
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    objective_value, autodiff_gradient = value_and_gradient(positions)
    objective_value.block_until_ready()

    positions_np = np.asarray(positions, dtype=np.float32)
    autodiff_np = np.asarray(autodiff_gradient, dtype=np.float64)
    component_scale = np.maximum(
        np.abs(autodiff_np), 0.05 * np.max(np.abs(autodiff_np))
    )

    component_results = []
    for step in FINITE_DIFFERENCE_STEPS_MM:
        finite_difference = np.empty_like(autodiff_np)
        for spot_index in range(positions_np.shape[0]):
            for axis_index in range(positions_np.shape[1]):
                perturbation = np.zeros_like(positions_np)
                perturbation[spot_index, axis_index] = np.float32(step)
                plus = float(objective(jnp.asarray(positions_np + perturbation)))
                minus = float(objective(jnp.asarray(positions_np - perturbation)))
                finite_difference[spot_index, axis_index] = (
                    plus - minus
                ) / (2.0 * step)
        l2_error = relative_l2_error(finite_difference, autodiff_np)
        max_scaled_error = float(
            np.max(np.abs(finite_difference - autodiff_np) / component_scale)
        )
        component_results.append(
            (step, finite_difference, l2_error, max_scaled_error)
        )

    direction = np.asarray(
        ((0.60, -0.20), (-0.35, 0.50), (0.25, 0.40)), dtype=np.float32
    )
    direction /= np.linalg.norm(direction)
    direction_jax = jnp.asarray(direction)

    _, dose_jvp = jax.jvp(
        dose_from_positions,
        (positions,),
        (direction_jax,),
    )
    dose_jvp_np = np.asarray(dose_jvp, dtype=np.float64)
    autodiff_directional = float(np.sum(autodiff_np * direction))

    directional_results = []
    for step in FINITE_DIFFERENCE_STEPS_MM:
        plus_dose = np.asarray(
            dose_from_positions(positions + np.float32(step) * direction_jax),
            dtype=np.float64,
        )
        minus_dose = np.asarray(
            dose_from_positions(positions - np.float32(step) * direction_jax),
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
    print(f"sorted energy IDs: {np.asarray(spot_data.spots_energy_id).tolist()}")
    print(f"spot positions (x,y) mm:\n{positions_np}")
    print(f"objective: {float(objective_value):.9g}")
    print(f"autodiff gradient (d objective / d [x,y]):\n{autodiff_np}")
    print("component central-difference sweep:")
    for step, finite_difference, l2_error, max_scaled_error in component_results:
        print(
            f"  h={step:g} mm: relative L2 error={l2_error:.6g}, "
            f"max scaled component error={max_scaled_error:.6g}"
        )
        print(f"{finite_difference}")
    print(f"autodiff directional derivative: {autodiff_directional:.9g}")
    print("directional central-difference sweep:")
    for step, finite_difference, scalar_error, dose_error in directional_results:
        print(
            f"  h={step:g} mm: scalar={finite_difference:.9g}, "
            f"scalar relative error={scalar_error:.6g}, "
            f"full-dose JVP relative L2 error={dose_error:.6g}"
        )
    print(
        f"best max scaled component error: {best_component[3]:.6g} "
        f"at h={best_component[0]:g} mm"
    )
    print(
        f"best directional scalar error: {best_directional[2]:.6g} "
        f"at h={best_directional[0]:g} mm"
    )
    print(
        f"best full-dose JVP error: {best_dose_jvp[3]:.6g} "
        f"at h={best_dose_jvp[0]:g} mm"
    )

    failures = []
    if not np.all(np.isfinite(autodiff_np)):
        failures.append("autodiff gradients must be finite")
    if np.linalg.norm(autodiff_np) <= 1.0e-8:
        failures.append("the validation objective has a negligible gradient")
    if best_component[3] > 2.0e-3:
        failures.append("component derivatives disagree with central differences")
    if best_directional[2] > 2.0e-3:
        failures.append("scalar directional derivative disagrees with central differences")
    if best_dose_jvp[3] > 2.0e-3:
        failures.append("full-dose JVP disagrees with central differences")
    if failures:
        raise SystemExit(
            "Spot-position gradient validation failed:\n- " + "\n- ".join(failures)
        )

    print("Spot-position gradient validation passed.")


if __name__ == "__main__":
    main()
