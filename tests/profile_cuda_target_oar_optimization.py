#!/usr/bin/env python3
"""Validate and time one toy target/OAR CUDA spot-weight optimization.

The masks and dose limits are fixed geometric test inputs, not patient data or
clinical prescriptions. No files or reference outputs are modified.
"""

import os
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import minimize


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from DoseCUDA.impt_weight_optimization import (  # noqa: E402
    FixedGeometryBeamDose,
    bounded_lbfgsb,
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
    oar_weight = 0.25
    loss = make_target_oar_loss(
        target_mask, prescription, oar_mask, oar_limit, oar_weight=oar_weight
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

    # Independent tiny-problem reference: assemble unit-spot dose columns,
    # formulate the same objective directly in float64, and use a bounded
    # optimizer. This matrix is a test oracle, not the production dose path.
    basis = np.stack(
        [
            operator.dose(np.eye(operator.n_spots, dtype=np.float32)[index])
            for index in range(operator.n_spots)
        ],
        axis=-1,
    ).astype(np.float64)
    np.testing.assert_allclose(
        np.tensordot(basis, initial_weights.astype(np.float64), axes=([-1], [0])),
        initial_dose,
        rtol=1.0e-4,
        atol=1.0e-5,
    )
    trial_weights = np.asarray((0.3, 1.2, 0.4), dtype=np.float32)
    np.testing.assert_allclose(
        np.tensordot(basis, trial_weights.astype(np.float64), axes=([-1], [0])),
        operator.dose(trial_weights),
        rtol=1.0e-4,
        atol=1.0e-5,
    )
    target_columns = basis[target_mask]
    oar_columns = basis[oar_mask]
    scale = prescription * prescription

    def reference_value_and_gradient(weights):
        target_error = target_columns @ weights - prescription
        oar_excess = np.maximum(oar_columns @ weights - oar_limit, 0.0)
        value = (
            np.mean(target_error**2)
            + oar_weight * np.mean(oar_excess**2)
        ) / scale
        gradient = (
            2.0 * target_columns.T @ target_error / target_columns.shape[0]
            + 2.0 * oar_weight * oar_columns.T @ oar_excess / oar_columns.shape[0]
        ) / scale
        return value, gradient

    reference_initial_value, reference_initial_gradient = (
        reference_value_and_gradient(initial_weights.astype(np.float64))
    )
    np.testing.assert_allclose(
        reference_initial_value, loss(initial_dose)[0], atol=1.0e-6
    )
    np.testing.assert_allclose(
        reference_initial_gradient, analytic_gradient, rtol=3.0e-3, atol=3.0e-4
    )

    reference = minimize(
        reference_value_and_gradient,
        initial_weights.astype(np.float64),
        jac=True,
        bounds=[(0.0, None)] * operator.n_spots,
        method="L-BFGS-B",
        options={"ftol": 1.0e-14, "gtol": 1.0e-9, "maxiter": 2000},
    )
    if not reference.success:
        raise SystemExit(f"independent bounded solve failed: {reference.message}")
    reference_dose = operator.dose(reference.x.astype(np.float32))
    reference_cuda_value = loss(reference_dose)[0]
    np.testing.assert_allclose(reference.fun, reference_cuda_value, atol=1.0e-6)

    cuda_lbfgsb = bounded_lbfgsb(
        lambda weights: operator.value_and_gradient(weights, loss),
        initial_weights,
        relative_tolerance=1.0e-12,
        gradient_tolerance=1.0e-6,
        stationarity_tolerance=1.0e-4,
    )
    cuda_lbfgsb_value = cuda_lbfgsb.objective

    refined = projected_gradient_descent(
        lambda weights: operator.value_and_gradient(weights, loss),
        result.weights,
        max_iterations=2000,
        initial_step=1.0,
        gradient_tolerance=1.0e-5,
        relative_tolerance=1.0e-10,
    )
    refined_value, refined_gradient = operator.value_and_gradient(
        refined.weights, loss
    )
    projected_gradient = refined.weights - np.maximum(
        refined.weights - refined_gradient, 0.0
    )
    print(f"independent bounded optimum: {reference.fun:.9g}")
    print(f"50-step objective gap: {result.objective_history[-1] - reference.fun:.6g}")
    print(f"CUDA bounded solve: {cuda_lbfgsb_value:.9g}; "
          f"gap: {cuda_lbfgsb_value - reference.fun:.6g}; "
          f"converged: {cuda_lbfgsb.converged}; "
          f"projected-gradient norm: "
          f"{cuda_lbfgsb.projected_gradient_norm:.6g}")
    print(f"CUDA bounded weights: {cuda_lbfgsb.weights.tolist()}")
    print(f"refined CUDA objective: {refined_value:.9g}; "
          f"gap: {refined_value - reference.fun:.6g}")
    print(f"refined CUDA iterations: {refined.iterations}; "
          f"converged: {refined.converged}")
    print(f"refined projected-gradient norm: "
          f"{np.linalg.norm(projected_gradient, ord=np.inf):.6g}")
    print(f"reference weights: {reference.x.tolist()}")
    print(f"refined CUDA weights: {refined.weights.tolist()}")
    print(f"reference target mean: {np.mean(reference_dose[target_mask]):.6g}")
    print(f"reference OAR max: {np.max(reference_dose[oar_mask]):.6g}; "
          f"voxels above limit: "
          f"{np.count_nonzero(reference_dose[oar_mask] > oar_limit)}")

    if refined_value - reference.fun > 1.0e-4:
        raise SystemExit("CUDA weight solve remains above independent optimum")
    if cuda_lbfgsb_value - reference.fun > 1.0e-4:
        raise SystemExit("CUDA bounded solve remains above independent optimum")
    if not cuda_lbfgsb.converged:
        raise SystemExit("CUDA bounded solve did not meet stationarity tolerance")


if __name__ == "__main__":
    main()
