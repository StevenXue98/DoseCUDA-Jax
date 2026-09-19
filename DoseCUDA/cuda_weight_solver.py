"""Experimental matrix-free, float32, resident-GPU spot-weight optimization.

This is intentionally separate from the established CPU/SciPy reference path.
It reuses the original DoseCUDA forward kernel and its spot-weight VJP, while
keeping the fixed-angle beam buffers, weights, dose, and gradients on the GPU
for the whole solve. The treatment objective is currently the toy
target/OAR/normal-tissue squared-hinge loss, not a clinical prescription.
"""

from dataclasses import dataclass

import numpy as np

from . import dose_kernels
from .impt_weight_optimization import FixedGeometryPlanDose


@dataclass(frozen=True)
class CUDAWeightSolveResult:
    weights: np.ndarray
    objective: float
    projected_gradient_norm: float
    iterations: int
    forward_evaluations: int
    gradient_evaluations: int
    converged: bool


def solve_cuda_spot_weights(
    operator: FixedGeometryPlanDose,
    target_mask,
    oar_mask,
    normal_mask,
    *,
    prescription: float,
    oar_limit: float,
    normal_limit: float,
    initial_weights=None,
    method="fista",
    max_iterations=500,
    gradient_tolerance=1.0e-5,
) -> CUDAWeightSolveResult:
    """Optimize all beam spots jointly using experimental CUDA methods.

    WET is precomputed by ``FixedGeometryPlanDose``. This function constructs
    no influence matrix and does not alter the original CUDA forward kernels.
    The L-BFGS candidate keeps dose/gradient buffers on the GPU but handles
    its small weight/gradient vectors on the CPU; the other methods keep their
    optimization vectors on-device.
    The returned gradient norm is unscaled in spot-weight units; compare dose
    and loss externally when deciding whether a given tolerance is adequate.
    """
    if not isinstance(operator, FixedGeometryPlanDose):
        raise TypeError("operator must be a FixedGeometryPlanDose")
    if method not in ("pg", "fista", "cg", "lbfgs"):
        raise ValueError("method must be 'pg', 'fista', 'cg', or 'lbfgs'")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    limits = np.asarray((prescription, oar_limit, normal_limit,
                         gradient_tolerance), dtype=np.float64)
    if (not np.all(np.isfinite(limits)) or prescription <= 0 or
            oar_limit < 0 or normal_limit < 0 or gradient_tolerance <= 0):
        raise ValueError("prescription, limits, and tolerance must be finite")
    shape = tuple(int(value) for value in operator.beams[0].dose_grid.size)
    masks = []
    for name, mask in (("target", target_mask), ("OAR", oar_mask),
                       ("normal", normal_mask)):
        array = np.asarray(mask)
        if array.shape != shape or not np.any(array) or not np.all(
                (array == 0) | (array == 1)):
            raise ValueError(f"{name} must be a nonempty binary dose-grid mask")
        masks.append(np.ascontiguousarray(array, dtype=np.uint8))
    if np.any((masks[0] + masks[1] + masks[2]) > 1):
        raise ValueError("target, OAR, and normal masks must be disjoint")
    weights = (operator.weights if initial_weights is None else
               np.asarray(initial_weights, dtype=np.float32))
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    if (weights.shape != (operator.n_spots,) or
            not np.all(np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("initial weights must be finite and nonnegative")
    gpu_ids = {beam.gpu_id for beam in operator.beams}
    scales = {beam.dose_scale for beam in operator.beams}
    if len(gpu_ids) != 1 or len(scales) != 1:
        raise ValueError("all beams must use the same GPU and fraction scale")
    raw = dose_kernels.proton_optimize_spot_weights_cuda(
        tuple(beam.beam_model for beam in operator.beams),
        operator.beams[0]._rlsp,
        tuple(beam._wet for beam in operator.beams),
        tuple(beam.beam for beam in operator.beams),
        tuple(masks),
        float(prescription), float(oar_limit), float(normal_limit),
        float(scales.pop()), weights, max_iterations,
        float(gradient_tolerance), method, int(gpu_ids.pop()),
    )
    return CUDAWeightSolveResult(
        weights=raw["weights"],
        objective=float(raw["objective"]),
        projected_gradient_norm=float(raw["projected_gradient"]),
        iterations=int(raw["iterations"]),
        forward_evaluations=int(raw["forward_evaluations"]),
        gradient_evaluations=int(raw["gradient_evaluations"]),
        converged=bool(raw["converged"]),
    )
