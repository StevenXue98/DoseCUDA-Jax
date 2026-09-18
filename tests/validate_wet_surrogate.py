"""Validate exact-forward/smooth-backward WET behavior.

The surrogate is intentionally not a true derivative of its rounded forward
model. This test verifies the two contracts separately:

1. Forward WET and dose exactly equal the CUDA-compatible rounded JAX path.
2. At the WET boundary, its angle gradient equals the differentiable smoother.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from diagnose_angle_gradient_matrix import ANGLE_CASES, create_densities  # noqa: E402
from diagnose_rounded_wet_landscape import LANDSCAPE_CASES  # noqa: E402
from impt_jax import (  # noqa: E402
    _precompute_all_grids,
    beam_params_from_angles,
    compute_dose,
    compute_raytrace,
    compute_raytrace_differentiable,
    compute_raytrace_surrogate,
)
from validate_spot_weight_gradients import prepare_fixed_data  # noqa: E402


TRANSITION_WIDTH_MM = 0.25


def main():
    (
        baseline_beam_params,
        dose_params,
        lut_data,
        spot_data,
        layer_data,
        _baseline_grids,
        _baseline_wet,
        _probe,
    ) = prepare_fixed_data()
    adjusted_iso = jnp.asarray(
        (
            baseline_beam_params.iso_x,
            baseline_beam_params.iso_y,
            baseline_beam_params.iso_z,
        ),
        dtype=jnp.float32,
    )

    def geometry(angles):
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
        return beam_params, grids

    def rounded_model(angles, density):
        beam_params, grids = geometry(angles)
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

    def surrogate_model(angles, density):
        beam_params, grids = geometry(angles)
        wet = compute_raytrace_surrogate(
            beam_params,
            dose_params,
            density,
            grids,
            transition_width_mm=TRANSITION_WIDTH_MM,
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
        return wet, dose

    def differentiable_wet(angles, density):
        beam_params, grids = geometry(angles)
        return compute_raytrace_differentiable(
            beam_params,
            dose_params,
            density,
            grids,
            transition_width_mm=TRANSITION_WIDTH_MM,
        )

    rounded_model = jax.jit(rounded_model)
    surrogate_model = jax.jit(surrogate_model)
    densities = dict(create_densities())
    angle_cases = dict(ANGLE_CASES)

    print(f"JAX device: {jax.devices()[0]}")
    print("case                                WET exact  dose exact")
    print("-" * 61)
    for material_name, angle_case_name in LANDSCAPE_CASES:
        density = densities[material_name]
        angles = jnp.asarray(angle_cases[angle_case_name], dtype=jnp.float32)
        rounded_wet, rounded_dose = rounded_model(angles, density)
        surrogate_wet, surrogate_dose = surrogate_model(angles, density)
        wet_exact = np.array_equal(np.asarray(rounded_wet), np.asarray(surrogate_wet))
        dose_exact = np.array_equal(
            np.asarray(rounded_dose), np.asarray(surrogate_dose)
        )
        case_name = f"{material_name}/{angle_case_name}"
        print(f"{case_name:35} {str(wet_exact):9}  {dose_exact}")
        if not wet_exact or not dose_exact:
            raise AssertionError(f"{case_name}: surrogate changed forward values")

    probe = jnp.linspace(
        -0.5,
        0.5,
        dose_params.ni * dose_params.nj * dose_params.nk,
        dtype=jnp.float32,
    ).reshape((dose_params.ni, dose_params.nj, dose_params.nk))
    density = densities["heterogeneous"]
    angles = jnp.asarray(angle_cases["far_oblique"], dtype=jnp.float32)

    def surrogate_projection(angle_values):
        wet, _dose = surrogate_model(angle_values, density)
        return jnp.sum(wet * probe)

    def differentiable_projection(angle_values):
        wet = differentiable_wet(angle_values, density)
        return jnp.sum(wet * probe)

    surrogate_gradient = np.asarray(jax.grad(surrogate_projection)(angles))
    differentiable_gradient = np.asarray(
        jax.grad(differentiable_projection)(angles)
    )
    difference = np.max(
        np.abs(surrogate_gradient - differentiable_gradient)
    )
    print(f"surrogate WET gradient:      {surrogate_gradient}")
    print(f"differentiable WET gradient: {differentiable_gradient}")
    print(f"maximum absolute difference: {difference:.9g}")
    np.testing.assert_allclose(
        surrogate_gradient,
        differentiable_gradient,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    print("WET surrogate validation passed.")


if __name__ == "__main__":
    main()
