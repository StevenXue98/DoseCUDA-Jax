#!/usr/bin/env python3
"""Validate and time one toy target/OAR CUDA spot-weight optimization.

The masks and dose limits are fixed geometric test inputs, not patient data or
clinical prescriptions. No files or reference outputs are modified.
"""

import os
import sys
from time import perf_counter

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from DoseCUDA.impt_weight_optimization import (  # noqa: E402
    FixedGeometryBeamDose,
    make_target_oar_loss,
    projected_gradient_descent,
)
from validate_spot_weight_gradients import create_case  # noqa: E402


def main():
    grid, plan, _ = create_case()
    started = perf_counter()
    operator = FixedGeometryBeamDose(grid, plan)
    setup_seconds = perf_counter() - started
    shape = tuple(int(value) for value in grid.size)
    if shape != (24, 28, 32):
        raise ValueError("toy masks require the 24 x 28 x 32 validation grid")

    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    target_mask = (z - 15) ** 2 + (y - 25) ** 2 + (x - 7) ** 2 <= 2**2
    oar_mask = (z - 10) ** 2 + (y - 25) ** 2 + (x - 13) ** 2 <= 2**2
    prescription = 0.50
    oar_limit = 0.30
    loss = make_target_oar_loss(
        target_mask, prescription, oar_mask, oar_limit, oar_weight=0.25
    )
    initial_weights = operator.weights

    # Check the complete chain rule for this objective before profiling.
    _, analytic_gradient = operator.value_and_gradient(initial_weights, loss)
    finite_difference = np.empty_like(analytic_gradient)
    step = np.float32(1.0e-2)
    for index in range(operator.n_spots):
        plus = initial_weights.copy()
        minus = initial_weights.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            loss(operator.dose(plus))[0] - loss(operator.dose(minus))[0]
        ) / (2.0 * step)
    np.testing.assert_allclose(
        analytic_gradient, finite_difference, rtol=3.0e-3, atol=3.0e-4
    )

    # The CUDA calls return host arrays, so each timing includes the current
    # synchronization and transfer cost. It does not isolate that cost.
    timings = {"dose": 0.0, "loss": 0.0, "vjp": 0.0}
    evaluations = 0

    def timed_value_and_gradient(weights):
        nonlocal evaluations
        started = perf_counter()
        dose = operator.dose(weights)
        after_dose = perf_counter()
        value, dose_gradient = loss(dose)
        after_loss = perf_counter()
        gradient = operator.weight_vjp(dose_gradient)
        after_vjp = perf_counter()
        timings["dose"] += after_dose - started
        timings["loss"] += after_loss - after_dose
        timings["vjp"] += after_vjp - after_loss
        evaluations += 1
        return value, gradient

    started = perf_counter()
    result = projected_gradient_descent(
        timed_value_and_gradient,
        initial_weights,
        max_iterations=50,
        initial_step=0.1,
    )
    solver_seconds = perf_counter() - started
    initial_dose = operator.dose(initial_weights)
    final_dose = operator.dose(result.weights)

    print(f"target voxels: {int(np.count_nonzero(target_mask))}")
    print(f"OAR voxels: {int(np.count_nonzero(oar_mask))}")
    print(f"target prescription: {prescription:.3f} toy dose units")
    print(f"OAR limit: {oar_limit:.3f} toy dose units")
    print(f"gradient finite difference: {finite_difference.tolist()}")
    print(f"gradient CUDA VJP: {analytic_gradient.tolist()}")
    print(f"objective: {result.objective_history[0]:.6g} -> "
          f"{result.objective_history[-1]:.6g}")
    print(f"target mean dose: {np.mean(initial_dose[target_mask]):.6g} -> "
          f"{np.mean(final_dose[target_mask]):.6g}")
    print(f"OAR mean dose: {np.mean(initial_dose[oar_mask]):.6g} -> "
          f"{np.mean(final_dose[oar_mask]):.6g}")
    print(f"OAR max dose: {np.max(initial_dose[oar_mask]):.6g} -> "
          f"{np.max(final_dose[oar_mask]):.6g}")
    print("OAR voxels above limit: "
          f"{np.count_nonzero(initial_dose[oar_mask] > oar_limit)} -> "
          f"{np.count_nonzero(final_dose[oar_mask] > oar_limit)}")
    print(f"weights: {initial_weights.tolist()} -> {result.weights.tolist()}")
    print(f"accepted iterations: {result.iterations}; converged: {result.converged}")
    print(f"objective evaluations (including line search): {evaluations}")
    print(f"cached-WET setup: {setup_seconds * 1000:.2f} ms")
    print(f"complete solver: {solver_seconds * 1000:.2f} ms")
    for stage in ("dose", "loss", "vjp"):
        print(f"  {stage}: {timings[stage] * 1000:.2f} ms total, "
              f"{timings[stage] * 1000 / evaluations:.2f} ms/evaluation")

    if result.objective_history[-1] >= result.objective_history[0]:
        raise SystemExit("target/OAR objective did not decrease")


if __name__ == "__main__":
    main()
