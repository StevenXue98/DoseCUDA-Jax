#!/usr/bin/env python3
"""Validate CUDA spot-weight reverse mode and projected optimization."""

import os
import sys

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from DoseCUDA import IMPTBeam  # noqa: E402
from DoseCUDA.impt_weight_optimization import (  # noqa: E402
    FixedGeometryBeamDose,
    FixedGeometryPlanDose,
    projected_gradient_descent,
)
from validate_spot_weight_gradients import create_case  # noqa: E402


FINITE_DIFFERENCE_STEP = np.float32(1.0e-2)


def central_difference(value, weights):
    result = np.empty_like(weights, dtype=np.float64)
    for index in range(weights.size):
        perturbation = np.zeros_like(weights)
        perturbation[index] = FINITE_DIFFERENCE_STEP
        plus = value(weights + perturbation)
        minus = value(weights - perturbation)
        result[index] = (plus - minus) / (2.0 * float(FINITE_DIFFERENCE_STEP))
    return result


def make_probe(shape):
    z, y, x = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    probe = np.exp(
        -(
            ((z - 0.58 * shape[0]) / 5.0) ** 2
            + ((y - 0.43 * shape[1]) / 6.0) ** 2
            + ((x - 0.61 * shape[2]) / 7.0) ** 2
        )
    )
    return np.ascontiguousarray(probe / np.sum(probe), dtype=np.float32)


def main():
    grid, plan, _beam = create_case()
    operator = FixedGeometryBeamDose(grid, plan)
    weights = operator.weights
    probe = make_probe(tuple(int(value) for value in grid.size))

    def linear_loss(dose):
        return float(np.sum(dose * probe)), probe

    linear_value, linear_gradient = operator.value_and_gradient(weights, linear_loss)
    linear_fd = central_difference(
        lambda trial: linear_loss(operator.dose(trial))[0],
        weights,
    )

    target_weights = np.asarray((1.15, 0.45, 0.90), dtype=np.float32)
    target_dose = operator.dose(target_weights)
    dose_scale = max(float(np.mean(target_dose * target_dose)), 1.0e-12)

    def quadratic_loss(dose):
        residual = dose - target_dose
        value = 0.5 * float(np.mean(residual * residual)) / dose_scale
        dose_gradient = residual / np.float32(residual.size * dose_scale)
        return value, np.ascontiguousarray(dose_gradient, dtype=np.float32)

    quadratic_value, quadratic_gradient = operator.value_and_gradient(
        weights, quadratic_loss
    )
    quadratic_fd = central_difference(
        lambda trial: quadratic_loss(operator.dose(trial))[0],
        weights,
    )

    initial_weights = np.asarray((0.30, 1.20, 0.25), dtype=np.float32)
    optimization = projected_gradient_descent(
        lambda trial: operator.value_and_gradient(trial, quadratic_loss),
        initial_weights,
        max_iterations=200,
        initial_step=1.0,
    )

    second_beam = IMPTBeam()
    second_beam.dicom_rangeshifter_label = "0"
    second_beam.gantry_angle = -47.0
    second_beam.couch_angle = 9.0
    second_beam.iso = np.asarray((-3.0, 6.0, -4.0), dtype=np.float32)
    second_beam.addSingleSpot(-7.0, 4.0, 0.55, 44)
    second_beam.addSingleSpot(9.0, -6.0, 0.75, 31)
    plan.addBeam(second_beam)
    plan_operator = FixedGeometryPlanDose(grid, plan)
    plan_weights = plan_operator.weights
    plan_value, plan_gradient = plan_operator.value_and_gradient(
        plan_weights, linear_loss
    )
    plan_fd = central_difference(
        lambda trial: linear_loss(plan_operator.dose(trial))[0],
        plan_weights,
    )

    linear_scale = np.maximum(np.abs(linear_fd), 1.0e-10)
    quadratic_scale = np.maximum(np.abs(quadratic_fd), 1.0e-10)
    linear_relative_error = np.abs(linear_gradient - linear_fd) / linear_scale
    quadratic_relative_error = (
        np.abs(quadratic_gradient - quadratic_fd) / quadratic_scale
    )
    target_weight_error = float(
        np.max(np.abs(optimization.weights - target_weights))
    )
    plan_scale = np.maximum(np.abs(plan_fd), 1.0e-10)
    plan_relative_error = np.abs(plan_gradient - plan_fd) / plan_scale

    print(f"original-order energy IDs: {plan.beam_list[0].spot_list[:, 3].tolist()}")
    print(f"linear objective: {linear_value:.9g}")
    print(f"CUDA linear VJP: {linear_gradient.tolist()}")
    print(f"linear finite difference: {linear_fd.tolist()}")
    print(f"max linear relative error: {linear_relative_error.max():.6g}")
    print(f"quadratic objective: {quadratic_value:.9g}")
    print(f"CUDA quadratic gradient: {quadratic_gradient.tolist()}")
    print(f"quadratic finite difference: {quadratic_fd.tolist()}")
    print(f"max quadratic relative error: {quadratic_relative_error.max():.6g}")
    print(f"optimizer iterations: {optimization.iterations}")
    print(f"optimizer converged: {optimization.converged}")
    print(f"initial objective: {optimization.objective_history[0]:.9g}")
    print(f"final objective: {optimization.objective_history[-1]:.9g}")
    print(f"target weights: {target_weights.tolist()}")
    print(f"optimized weights: {optimization.weights.tolist()}")
    print(f"max target-weight error: {target_weight_error:.6g}")
    print(f"two-beam spot count: {plan_operator.n_spots}")
    print(f"two-beam linear objective: {plan_value:.9g}")
    print(f"two-beam CUDA VJP: {plan_gradient.tolist()}")
    print(f"two-beam finite difference: {plan_fd.tolist()}")
    print(f"max two-beam relative error: {plan_relative_error.max():.6g}")

    failures = []
    if linear_relative_error.max() > 3.0e-3:
        failures.append("CUDA linear VJP disagrees with central finite differences")
    if quadratic_relative_error.max() > 3.0e-3:
        failures.append("CUDA quadratic chain rule disagrees with finite differences")
    if optimization.objective_history[-1] > 1.0e-7:
        failures.append("projected optimization did not recover the target dose")
    if target_weight_error > 2.0e-3:
        failures.append("projected optimization did not recover the target weights")
    if plan_relative_error.max() > 3.0e-3:
        failures.append("multi-beam CUDA VJP disagrees with finite differences")
    if failures:
        raise SystemExit("CUDA spot-weight validation failed:\n- " + "\n- ".join(failures))

    print("CUDA spot-weight VJP and optimization validation passed.")


if __name__ == "__main__":
    main()
