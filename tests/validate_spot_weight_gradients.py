"""Validate JAX spot-weight derivatives against independent numerical checks.

Geometry, WET, energy layers, and spot positions are fixed.  Only spot monitor
units are traced, so the physical dose must be a linear function of the input
weight vector.  This test deliberately uses the low-level JAX dose interface;
the legacy object wrapper converts results through NumPy and is not an
end-to-end differentiable API.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from DoseCUDA import IMPTBeam, IMPTDoseGrid, IMPTPlan  # noqa: E402
from impt_jax import (  # noqa: E402
    DoseParams,
    _extract_beam_params,
    _extract_layer_data,
    _extract_lut_data,
    _extract_spot_data,
    _precompute_all_grids,
    compute_dose,
    compute_raytrace,
)


SHAPE_ZYX = (24, 28, 32)
FINITE_DIFFERENCE_STEP = np.float32(1.0e-2)


def create_case():
    grid = IMPTDoseGrid()
    grid.createCubePhantom(size=SHAPE_ZYX)

    plan = IMPTPlan()
    beam = IMPTBeam()
    beam.dicom_rangeshifter_label = "0"
    beam.gantry_angle = 23.0
    beam.couch_angle = -11.0
    beam.iso = np.asarray((4.0, -5.0, 7.0), dtype=np.float32)

    # Two spots share a layer and one uses a second energy layer.  Supplying
    # them out of energy order also exercises the normal extraction/sort path.
    beam.addSingleSpot(14.0, 5.0, 0.65, 57)
    beam.addSingleSpot(-11.0, 8.0, 0.80, 38)
    beam.addSingleSpot(3.0, -9.0, 1.10, 38)
    plan.addBeam(beam)
    return grid, plan, beam


def prepare_fixed_data():
    grid, plan, beam = create_case()
    model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
        beam.dicom_rangeshifter_label
    )
    beam_model = plan.beam_models[model_index]

    beam_params = _extract_beam_params(beam, beam_model, grid.origin)
    dose_params = DoseParams(
        ni=int(grid.size[0]),
        nj=int(grid.size[1]),
        nk=int(grid.size[2]),
        spacing=jnp.asarray(float(grid.spacing[0]), dtype=jnp.float32),
    )
    lut_data = _extract_lut_data(beam_model)
    spot_data = _extract_spot_data(beam, beam_model)
    layer_data = _extract_layer_data(spot_data, beam_model)

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
    density = jnp.asarray(grid.RLSPFromHU(plan.machine_name), dtype=jnp.float32)
    wet = compute_raytrace(beam_params, dose_params, density, grids)

    # An asymmetric, smooth probe makes the scalar objective sensitive to all
    # three spots without introducing any weight-dependent masks.
    z, y, x = jnp.meshgrid(
        jnp.arange(dose_params.ni, dtype=jnp.float32),
        jnp.arange(dose_params.nj, dtype=jnp.float32),
        jnp.arange(dose_params.nk, dtype=jnp.float32),
        indexing="ij",
    )
    probe = jnp.exp(
        -(
            ((z - 0.58 * dose_params.ni) / 5.0) ** 2
            + ((y - 0.43 * dose_params.nj) / 6.0) ** 2
            + ((x - 0.61 * dose_params.nk) / 7.0) ** 2
        )
    )
    probe = probe / jnp.sum(probe)
    return beam_params, dose_params, lut_data, spot_data, layer_data, grids, wet, probe


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

    def dose_from_weights(weights):
        variable_spots = spot_data._replace(spots_mu=weights)
        return compute_dose(
            beam_params,
            dose_params,
            lut_data,
            variable_spots,
            layer_data,
            wet,
            grids,
        )

    def objective(weights):
        return jnp.sum(dose_from_weights(weights) * probe)

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    weights = spot_data.spots_mu
    objective_value, autodiff_gradient = value_and_gradient(weights)
    objective_value.block_until_ready()

    weights_np = np.asarray(weights, dtype=np.float32)
    autodiff_np = np.asarray(autodiff_gradient, dtype=np.float64)

    finite_difference = np.empty_like(autodiff_np)
    for index in range(weights_np.size):
        perturbation = np.zeros_like(weights_np)
        perturbation[index] = FINITE_DIFFERENCE_STEP
        plus = float(objective(jnp.asarray(weights_np + perturbation)))
        minus = float(objective(jnp.asarray(weights_np - perturbation)))
        finite_difference[index] = (
            plus - minus
        ) / (2.0 * float(FINITE_DIFFERENCE_STEP))

    basis_gradient = np.empty_like(autodiff_np)
    basis_doses = []
    for index in range(weights_np.size):
        basis = np.zeros_like(weights_np)
        basis[index] = 1.0
        basis_dose = np.asarray(dose_from_weights(jnp.asarray(basis)))
        basis_doses.append(basis_dose)
        basis_gradient[index] = float(np.sum(basis_dose * np.asarray(probe)))

    dose = np.asarray(dose_from_weights(weights))
    reconstructed_dose = np.sum(
        np.asarray(basis_doses) * weights_np[:, np.newaxis, np.newaxis, np.newaxis],
        axis=0,
    )

    second_weights = jnp.asarray((0.35, 1.25, 0.90), dtype=jnp.float32)
    second_gradient = np.asarray(jax.grad(objective)(second_weights), dtype=np.float64)

    gradient_scale = np.maximum(np.abs(basis_gradient), 1.0e-12)
    finite_relative_error = np.abs(autodiff_np - finite_difference) / gradient_scale
    basis_relative_error = np.abs(autodiff_np - basis_gradient) / gradient_scale
    dose_scale = max(float(np.max(np.abs(dose))), 1.0e-12)
    dose_linearity_error = float(np.max(np.abs(dose - reconstructed_dose)) / dose_scale)
    gradient_invariance_error = float(
        np.max(np.abs(autodiff_np - second_gradient) / gradient_scale)
    )

    print(f"JAX device: {jax.devices()[0]}")
    print(f"shape (z,y,x): {SHAPE_ZYX}")
    print(f"sorted energy IDs: {np.asarray(spot_data.spots_energy_id).tolist()}")
    print(f"weights: {weights_np.tolist()}")
    print(f"objective: {float(objective_value):.9g}")
    print(f"autodiff gradient:       {autodiff_np.tolist()}")
    print(f"finite-difference grad:  {finite_difference.tolist()}")
    print(f"unit-dose response grad: {basis_gradient.tolist()}")
    print(f"max finite-difference relative error: {finite_relative_error.max():.6g}")
    print(f"max unit-dose relative error: {basis_relative_error.max():.6g}")
    print(f"full-dose linearity error: {dose_linearity_error:.6g}")
    print(f"gradient invariance error: {gradient_invariance_error:.6g}")

    failures = []
    if not np.all(np.isfinite(autodiff_np)) or np.any(autodiff_np <= 0.0):
        failures.append("autodiff gradients must be finite and positive")
    if finite_relative_error.max() > 2.0e-3:
        failures.append("autodiff and central finite differences disagree")
    if basis_relative_error.max() > 2.0e-5:
        failures.append("autodiff and unit-dose responses disagree")
    if dose_linearity_error > 2.0e-6:
        failures.append("dose is not linear in spot weights")
    if gradient_invariance_error > 2.0e-5:
        failures.append("gradient changes with spot weights for a linear objective")
    if failures:
        raise SystemExit("Spot-weight gradient validation failed:\n- " + "\n- ".join(failures))

    print("Spot-weight gradient validation passed.")


if __name__ == "__main__":
    main()
