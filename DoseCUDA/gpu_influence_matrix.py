"""Experimental GPU-resident fixed-angle dose-influence matrix.

DoseCUDA constructs each unit-spot column once. cuBLAS then evaluates dose
and the weight gradient in float64 from a persistent device matrix. SciPy
controls the small nonnegative weight vector on the host; this intentionally
does not claim that the entire optimizer is device-resident.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from . import dose_kernels
from .impt_weight_optimization import (
    FixedGeometryPlanDose,
    InfluenceMatrixPlanDose,
    bounded_slsqp,
)


class GPUInfluenceMatrixObjective:
    """Keep a C-order (voxels, spots) float64 matrix on one CUDA device."""

    def __init__(self, matrix, shape, target_mask, oar_mask, normal_mask, *,
                 prescription, oar_limit, normal_limit, oar_weight=1.0,
                 normal_weight=1.0, gpu_id=0):
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be a nonempty finite 2D array")
        self.shape = tuple(int(value) for value in shape)
        if np.prod(self.shape) != matrix.shape[0]:
            raise ValueError("matrix rows must match the dose-grid shape")
        self.n_spots = matrix.shape[1]
        masks = []
        for name, mask in (("target", target_mask), ("OAR", oar_mask),
                           ("normal", normal_mask)):
            mask = np.asarray(mask)
            if (mask.shape != self.shape or not np.any(mask) or
                    not np.all((mask == 0) | (mask == 1))):
                raise ValueError(f"{name} must be a nonempty binary dose mask")
            masks.append(np.ascontiguousarray(mask, dtype=np.uint8).ravel())
        if np.any(masks[0] + masks[1] + masks[2] > 1):
            raise ValueError("structure masks must be disjoint")
        settings = (prescription, oar_limit, normal_limit, oar_weight,
                    normal_weight)
        if (not np.all(np.isfinite(settings)) or prescription <= 0 or
                min(oar_limit, normal_limit, oar_weight, normal_weight) < 0):
            raise ValueError("dose limits and penalty weights must be valid")
        self._context = dose_kernels.proton_gpu_matrix_create(
            matrix, *masks, *map(float, settings), int(gpu_id))

    def _weights(self, weights):
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if (weights.shape != (self.n_spots,) or
                not np.all(np.isfinite(weights)) or np.any(weights < 0)):
            raise ValueError("weights must be finite, nonnegative, and match spots")
        return weights

    def value_and_gradient(self, weights):
        result = dose_kernels.proton_gpu_matrix_evaluate(
            self._context, self._weights(weights))
        return result["objective"], result["gradient"]

    def dose(self, weights):
        return dose_kernels.proton_gpu_matrix_dose(
            self._context, self._weights(weights)).reshape(self.shape)

    def weight_vjp(self, dose_adjoint):
        adjoint = np.asarray(dose_adjoint, dtype=np.float64)
        if adjoint.shape != self.shape or not np.all(np.isfinite(adjoint)):
            raise ValueError("dose adjoint must be finite and match the grid")
        return dose_kernels.proton_gpu_matrix_weight_vjp(
            self._context, np.ascontiguousarray(adjoint).ravel())

    def value_and_gradient_for(self, weights, loss):
        """Use an arbitrary CPU loss with GPU-resident D and D-transpose.

        The dose and its adjoint cross the CPU/GPU boundary each callback;
        this keeps multi-structure research losses modular at modest sizes.
        """
        dose = self.dose(weights)
        value, dose_adjoint = loss(dose)
        return float(value), self.weight_vjp(dose_adjoint)


@dataclass(frozen=True)
class GPUInfluenceSolveResult:
    weights: np.ndarray
    objective: float
    iterations: int
    evaluations: int
    projected_gradient_norm: float
    converged: bool
    message: str
    matrix_build_seconds: float
    upload_seconds: float
    optimization_seconds: float


def solve_gpu_influence_weights(
    operator: FixedGeometryPlanDose, target_mask, oar_mask, normal_mask, *,
    prescription, oar_limit, normal_limit, initial_weights=None,
    max_matrix_elements=10_000_000, max_iterations=1000,
    stationarity_tolerance=1.0e-6,
) -> GPUInfluenceSolveResult:
    """Build DoseCUDA columns, upload them once, then solve joint weights.

    Timings include the complete matrix build and upload. Beam geometry/WET
    preparation occurs when ``operator`` is constructed and is not included.
    """
    if not isinstance(operator, FixedGeometryPlanDose):
        raise TypeError("operator must be a FixedGeometryPlanDose")
    started = perf_counter()
    influence = InfluenceMatrixPlanDose(
        operator, max_elements=max_matrix_elements)
    matrix_build_seconds = perf_counter() - started
    started = perf_counter()
    gpu_objective = GPUInfluenceMatrixObjective(
        influence.matrix, influence.shape, target_mask, oar_mask, normal_mask,
        prescription=prescription, oar_limit=oar_limit,
        normal_limit=normal_limit, gpu_id=operator.beams[0].gpu_id)
    upload_seconds = perf_counter() - started
    weights = operator.weights if initial_weights is None else initial_weights
    started = perf_counter()
    result = bounded_slsqp(
        gpu_objective.value_and_gradient, weights, weight_scale=0.02,
        objective_scale=1.0e4, max_iterations=max_iterations,
        stationarity_tolerance=stationarity_tolerance)
    optimization_seconds = perf_counter() - started
    return GPUInfluenceSolveResult(
        weights=result.weights, objective=result.objective,
        iterations=result.iterations, evaluations=result.evaluations,
        projected_gradient_norm=result.projected_gradient_norm,
        converged=result.converged, message=result.message,
        matrix_build_seconds=matrix_build_seconds,
        upload_seconds=upload_seconds,
        optimization_seconds=optimization_seconds)
