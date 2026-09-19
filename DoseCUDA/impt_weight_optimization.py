"""Fixed-geometry IMPT spot-weight optimization without an AD framework.

The CUDA extension supplies the loss-agnostic vector-Jacobian product
``dL/dDose -> dL/dMU``.  Treatment objectives remain ordinary Python callables
that return a scalar loss and a voxel adjoint, so changing the objective does
not require changing the dose kernel.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import dose_kernels
from .plan import VolumeObject


LossAndDoseGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]
ValueAndWeightGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


def make_target_oar_loss(
    target_mask, prescription, oar_mask, oar_limit, oar_weight=1.0
) -> LossAndDoseGradient:
    """Create a dimensionless target-prescription/OAR-overdose objective.

    Both terms are mean squared dose errors divided by ``prescription**2``.
    The target term penalizes under- and over-dose; the OAR term penalizes only
    dose above its limit. This is a research objective, not a clinical plan
    acceptance criterion.
    """
    target_mask = np.asarray(target_mask)
    oar_mask = np.asarray(oar_mask)
    if target_mask.ndim != 3 or target_mask.shape != oar_mask.shape:
        raise ValueError("target and OAR masks must have the same 3D shape")
    for name, mask in (("target", target_mask), ("OAR", oar_mask)):
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} mask must contain only zeros and ones")
        if not np.any(mask):
            raise ValueError(f"{name} mask must not be empty")
    if not np.isfinite(prescription) or prescription <= 0:
        raise ValueError("prescription must be finite and positive")
    if not np.isfinite(oar_limit) or oar_limit < 0:
        raise ValueError("oar_limit must be finite and nonnegative")
    if not np.isfinite(oar_weight) or oar_weight < 0:
        raise ValueError("oar_weight must be finite and nonnegative")

    target_mask = target_mask.astype(bool, copy=True)
    oar_mask = oar_mask.astype(bool, copy=True)
    prescription = float(prescription)
    oar_limit = float(oar_limit)
    oar_weight = float(oar_weight)
    scale = prescription * prescription
    target_count = int(np.count_nonzero(target_mask))
    oar_count = int(np.count_nonzero(oar_mask))

    def loss(dose: np.ndarray) -> tuple[float, np.ndarray]:
        dose = np.asarray(dose)
        if dose.dtype not in (np.dtype("float32"), np.dtype("float64")):
            dose = dose.astype(np.float32)
        if dose.shape != target_mask.shape or not np.all(np.isfinite(dose)):
            raise ValueError("dose must be finite and match the mask shape")

        target_error = dose[target_mask] - prescription
        oar_excess = np.maximum(dose[oar_mask] - oar_limit, 0.0)
        value = (
            np.sum(target_error * target_error, dtype=np.float64) / target_count
            + oar_weight
            * np.sum(oar_excess * oar_excess, dtype=np.float64)
            / oar_count
        ) / scale

        dose_gradient = np.zeros(dose.shape, dtype=dose.dtype)
        dose_gradient[target_mask] += 2.0 * target_error / (target_count * scale)
        dose_gradient[oar_mask] += (
            2.0 * oar_weight * oar_excess / (oar_count * scale)
        )
        return float(value), dose_gradient

    return loss


def make_target_oar_normal_tissue_loss(
    target_mask,
    prescription,
    oar_mask,
    oar_limit,
    normal_tissue_mask,
    normal_tissue_limit,
    *,
    oar_weight=1.0,
    normal_tissue_weight=1.0,
) -> LossAndDoseGradient:
    """Target deviation plus one-sided OAR and normal-tissue overdose.

    Each structure contributes a mean squared error normalized by the square
    of the target prescription. The caller chooses structures, limits, and
    weights; this function does not encode a clinical prescription.
    """
    base_loss = make_target_oar_loss(
        target_mask, prescription, oar_mask, oar_limit, oar_weight
    )
    normal_tissue_mask = np.asarray(normal_tissue_mask)
    target_mask = np.asarray(target_mask)
    oar_mask = np.asarray(oar_mask)
    if normal_tissue_mask.shape != target_mask.shape:
        raise ValueError("normal-tissue mask must match target/OAR shape")
    if not np.all((normal_tissue_mask == 0) | (normal_tissue_mask == 1)):
        raise ValueError("normal-tissue mask must contain only zeros and ones")
    if not np.any(normal_tissue_mask):
        raise ValueError("normal-tissue mask must not be empty")
    normal_tissue_mask = normal_tissue_mask.astype(bool, copy=True)
    if np.any(normal_tissue_mask & (target_mask.astype(bool) | oar_mask.astype(bool))):
        raise ValueError("normal-tissue mask must exclude target and OAR")
    if not np.isfinite(normal_tissue_limit) or normal_tissue_limit < 0:
        raise ValueError("normal-tissue limit must be finite and nonnegative")
    if not np.isfinite(normal_tissue_weight) or normal_tissue_weight < 0:
        raise ValueError("normal-tissue weight must be finite and nonnegative")

    count = int(np.count_nonzero(normal_tissue_mask))
    scale = float(prescription) ** 2
    limit = float(normal_tissue_limit)
    weight = float(normal_tissue_weight)

    def loss(dose: np.ndarray) -> tuple[float, np.ndarray]:
        value, dose_gradient = base_loss(dose)
        dose = np.asarray(dose, dtype=dose_gradient.dtype)
        excess = np.maximum(dose[normal_tissue_mask] - limit, 0.0)
        value += weight * np.sum(excess * excess, dtype=np.float64) / (count * scale)
        dose_gradient[normal_tissue_mask] += 2.0 * weight * excess / (count * scale)
        return float(value), dose_gradient

    return loss


class FixedGeometryBeamDose:
    """One-beam CUDA dose operator with cached WET and a spot-weight VJP."""

    def __init__(self, dose_grid, plan, beam_index=0, gpu_id=0):
        dose_grid._validate_geometry()
        if not np.allclose(dose_grid.spacing, dose_grid.spacing[0]):
            raise ValueError("Spacing must be isotropic for IMPT dose calculation")
        if beam_index < 0 or beam_index >= len(plan.beam_list):
            raise IndexError("beam_index is outside plan.beam_list")

        self.dose_grid = dose_grid
        self.plan = plan
        self.beam = plan.beam_list[beam_index]
        self.gpu_id = int(gpu_id)
        self.dose_scale = float(plan.n_fractions)

        try:
            model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
                self.beam.dicom_rangeshifter_label
            )
        except ValueError as error:
            raise ValueError(
                "Beam model not found for rangeshifter ID "
                f"{self.beam.dicom_rangeshifter_label}"
            ) from error
        self.beam_model = plan.beam_models[model_index]

        rlsp = np.ascontiguousarray(
            dose_grid.RLSPFromHU(plan.machine_name), dtype=np.float32
        )
        self._rlsp = VolumeObject()
        self._rlsp.voxel_data = rlsp
        self._rlsp.origin = np.asarray(dose_grid.origin, dtype=np.float32)
        self._rlsp.spacing = np.asarray(dose_grid.spacing, dtype=np.float32)

        wet_data = dose_kernels.proton_raytrace_cuda(
            self.beam_model,
            self._rlsp,
            self.beam,
            self.gpu_id,
        )
        self._wet = VolumeObject()
        self._wet.voxel_data = np.ascontiguousarray(wet_data, dtype=np.float32)
        self._wet.origin = self._rlsp.origin
        self._wet.spacing = self._rlsp.spacing

    @property
    def n_spots(self):
        return int(self.beam.n_spots)

    @property
    def weights(self):
        return np.asarray(self.beam.spot_list[:, 2], dtype=np.float32).copy()

    def _validate_weights(self, weights):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (self.n_spots,):
            raise ValueError(
                f"weights must have shape ({self.n_spots},), got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        return weights

    def dose(self, weights):
        """Return fraction-scaled dose for weights in the beam's original order."""
        weights = self._validate_weights(weights)
        self.beam.spot_list[:, 2] = weights
        dose = dose_kernels.proton_spot_cuda(
            self.beam_model,
            self._rlsp,
            self._wet,
            self.beam,
            self.gpu_id,
        )
        return np.asarray(dose, dtype=np.float32) * self.dose_scale

    def weight_vjp(self, dose_gradient):
        """Apply ``(dDose/dWeights).T`` to an arbitrary voxel gradient."""
        dose_gradient = np.ascontiguousarray(dose_gradient, dtype=np.float32)
        expected_shape = tuple(int(value) for value in self.dose_grid.size)
        if dose_gradient.shape != expected_shape:
            raise ValueError(
                f"dose_gradient must have shape {expected_shape}, "
                f"got {dose_gradient.shape}"
            )
        if not np.all(np.isfinite(dose_gradient)):
            raise ValueError("dose_gradient must be finite")

        # dose() returns n_fractions times the raw beam dose, so its transpose
        # product requires the same chain-rule factor.
        raw_gradient = dose_gradient * np.float32(self.dose_scale)
        return np.asarray(
            dose_kernels.proton_spot_weight_vjp_cuda(
                self.beam_model,
                self._rlsp,
                self._wet,
                self.beam,
                raw_gradient,
                self.gpu_id,
            ),
            dtype=np.float32,
        )

    def value_and_gradient(self, weights, loss: LossAndDoseGradient):
        """Evaluate an arbitrary dose loss and its spot-weight gradient."""
        dose = self.dose(weights)
        value, dose_gradient = loss(dose)
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("loss must be finite")
        return value, self.weight_vjp(dose_gradient)


class FixedGeometryPlanDose:
    """Multi-beam fixed-geometry operator with one concatenated weight vector."""

    def __init__(self, dose_grid, plan, gpu_id=0):
        if not plan.beam_list:
            raise ValueError("plan must contain at least one beam")
        self.beams = tuple(
            FixedGeometryBeamDose(dose_grid, plan, beam_index, gpu_id)
            for beam_index in range(len(plan.beam_list))
        )
        counts = np.asarray([beam.n_spots for beam in self.beams], dtype=np.int64)
        self._offsets = np.concatenate((np.asarray([0]), np.cumsum(counts)))

    @property
    def n_spots(self):
        return int(self._offsets[-1])

    @property
    def weights(self):
        return np.concatenate([beam.weights for beam in self.beams])

    def _split(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.n_spots,):
            raise ValueError(
                f"weights must have shape ({self.n_spots},), got {values.shape}"
            )
        return tuple(
            values[self._offsets[index] : self._offsets[index + 1]]
            for index in range(len(self.beams))
        )

    def dose(self, weights):
        beam_weights = self._split(weights)
        total = None
        for beam, weights_for_beam in zip(self.beams, beam_weights):
            beam_dose = beam.dose(weights_for_beam)
            if total is None:
                total = beam_dose
            else:
                total += beam_dose
        return total

    def weight_vjp(self, dose_gradient):
        return np.concatenate(
            [beam.weight_vjp(dose_gradient) for beam in self.beams]
        )

    def value_and_gradient(self, weights, loss: LossAndDoseGradient):
        dose = self.dose(weights)
        value, dose_gradient = loss(dose)
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("loss must be finite")
        return value, self.weight_vjp(dose_gradient)


class InfluenceMatrixPlanDose:
    """Small fixed plan represented by CUDA-computed unit-spot dose columns.

    This optional operator trades memory for a smooth float64 weight objective.
    It is useful for small BAO reference cases, not large clinical matrices.
    Beam geometry and each dose column still come from the original CUDA model.
    """

    def __init__(self, fixed_plan: FixedGeometryPlanDose, *, max_elements=10_000_000):
        shape = tuple(int(value) for value in fixed_plan.beams[0].dose_grid.size)
        n_voxels = int(np.prod(shape))
        self.n_spots = fixed_plan.n_spots
        if self.n_spots * n_voxels > max_elements:
            raise ValueError("influence matrix exceeds max_elements")
        self.shape = shape
        original_weights = fixed_plan.weights
        matrix = np.empty((n_voxels, self.n_spots), dtype=np.float64)
        unit = np.zeros(self.n_spots, dtype=np.float32)
        try:
            for index in range(self.n_spots):
                unit[index] = 1.0
                matrix[:, index] = fixed_plan.dose(unit).ravel()
                unit[index] = 0.0
        finally:
            fixed_plan.dose(original_weights)
        self.matrix = matrix

    def dose(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.n_spots,) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be a finite vector matching the spots")
        return np.einsum("ij,j->i", self.matrix, weights).reshape(self.shape)

    def weight_vjp(self, dose_gradient):
        dose_gradient = np.asarray(dose_gradient, dtype=np.float64)
        if dose_gradient.shape != self.shape or not np.all(np.isfinite(dose_gradient)):
            raise ValueError("dose_gradient must be finite and match the dose grid")
        return np.einsum("ij,i->j", self.matrix, dose_gradient.ravel())

    def value_and_gradient(self, weights, loss: LossAndDoseGradient):
        value, dose_gradient = loss(self.dose(weights))
        if not np.isfinite(value):
            raise ValueError("loss must be finite")
        return float(value), self.weight_vjp(dose_gradient)


@dataclass(frozen=True)
class ProjectedGradientResult:
    weights: np.ndarray
    objective_history: tuple[float, ...]
    iterations: int
    converged: bool


@dataclass(frozen=True)
class BoundedLBFGSBResult:
    weights: np.ndarray
    objective: float
    iterations: int
    evaluations: int
    projected_gradient_norm: float
    converged: bool
    message: str


@dataclass(frozen=True)
class BoundedSLSQPResult:
    weights: np.ndarray
    objective: float
    iterations: int
    evaluations: int
    projected_gradient_norm: float
    converged: bool
    message: str


def bounded_slsqp(
    value_and_gradient: ValueAndWeightGradient,
    initial_weights,
    *,
    weight_scale=1.0,
    objective_scale=1.0,
    max_iterations=1000,
    absolute_tolerance=1.0e-12,
    stationarity_tolerance=1.0e-6,
) -> BoundedSLSQPResult:
    """Nonnegative SLSQP solve with explicit variable/objective scaling.

    Scaling affects the optimizer only; the result and stationarity check use
    the original weights and objective units. For high-accuracy small cases,
    pair this with ``InfluenceMatrixPlanDose`` to avoid float32 line-search
    noise from repeated CUDA dose evaluation.
    """
    from scipy.optimize import minimize

    weights = np.asarray(initial_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("initial_weights must be a nonempty finite 1D array")
    settings = np.asarray((weight_scale, objective_scale, absolute_tolerance,
                           stationarity_tolerance), dtype=np.float64)
    if not np.all(np.isfinite(settings)) or np.any(settings <= 0):
        raise ValueError("scales and tolerances must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    weights = np.maximum(weights, 0.0)

    def scaled_value_and_gradient(scaled_weights):
        value, gradient = value_and_gradient(weight_scale * scaled_weights)
        value = float(value)
        gradient = np.asarray(gradient, dtype=np.float64)
        if (not np.isfinite(value) or gradient.shape != weights.shape or
                not np.all(np.isfinite(gradient))):
            raise ValueError("callback must return a finite value and gradient")
        return objective_scale * value, objective_scale * weight_scale * gradient

    result = minimize(
        scaled_value_and_gradient,
        weights / weight_scale,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, None)] * weights.size,
        options={"ftol": float(absolute_tolerance),
                 "maxiter": int(max_iterations)},
    )
    final_weights = np.maximum(weight_scale * result.x, 0.0)
    final_value, final_gradient = value_and_gradient(final_weights)
    final_gradient = np.asarray(final_gradient, dtype=np.float64)
    projected = final_weights - np.maximum(final_weights - final_gradient, 0.0)
    projected_norm = float(np.linalg.norm(projected, ord=np.inf))
    return BoundedSLSQPResult(
        weights=final_weights,
        objective=float(final_value),
        iterations=int(result.nit),
        evaluations=int(result.nfev) + 1,
        projected_gradient_norm=projected_norm,
        converged=bool(result.success and projected_norm <= stationarity_tolerance),
        message=str(result.message),
    )


def bounded_lbfgsb(
    value_and_gradient: ValueAndWeightGradient,
    initial_weights,
    *,
    max_iterations=1000,
    gradient_tolerance=1.0e-6,
    stationarity_tolerance=1.0e-4,
    relative_tolerance=1.0e-12,
) -> BoundedLBFGSBResult:
    """Optimize nonnegative spot weights using a dose/VJP callback.

    SciPy chooses trial weights on the CPU. The supplied callback may use CUDA
    for dose and weight gradients; no dose-influence matrix is constructed.
    ``gradient_tolerance`` controls SciPy's search; the looser
    ``stationarity_tolerance`` checks the returned float32 CUDA weights.
    ``converged`` requires that projected-gradient check, not just SciPy's
    relative-objective stopping condition.
    """
    from scipy.optimize import minimize

    weights = np.asarray(initial_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("initial_weights must be a nonempty finite 1D array")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    tolerances = np.asarray(
        (gradient_tolerance, stationarity_tolerance, relative_tolerance),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances <= 0):
        raise ValueError("optimizer tolerances must be positive")
    weights = np.maximum(weights, 0.0)

    def checked_value_and_gradient(trial):
        value, gradient = value_and_gradient(trial)
        value = float(value)
        gradient = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(value):
            raise ValueError("objective must be finite")
        if gradient.shape != weights.shape or not np.all(np.isfinite(gradient)):
            raise ValueError("weight gradient must be finite and match weights")
        return value, gradient

    result = minimize(
        checked_value_and_gradient,
        weights,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * weights.size,
        options={
            "maxiter": int(max_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": float(relative_tolerance),
        },
    )
    final_weights = np.maximum(result.x, 0.0).astype(np.float32)
    final_value, final_gradient = checked_value_and_gradient(final_weights)
    projected_gradient = final_weights - np.maximum(
        final_weights - final_gradient, 0.0
    )
    projected_norm = float(np.linalg.norm(projected_gradient, ord=np.inf))
    return BoundedLBFGSBResult(
        weights=final_weights,
        objective=final_value,
        iterations=int(result.nit),
        evaluations=int(result.nfev) + 1,
        projected_gradient_norm=projected_norm,
        converged=bool(result.success and projected_norm <= stationarity_tolerance),
        message=str(result.message),
    )


def projected_gradient_descent(
    value_and_gradient: ValueAndWeightGradient,
    initial_weights,
    *,
    max_iterations=200,
    initial_step=1.0,
    gradient_tolerance=1.0e-6,
    relative_tolerance=1.0e-8,
    armijo=1.0e-4,
    backtracking=0.5,
    minimum_step=1.0e-10,
):
    """Minimize a differentiable objective subject to nonnegative weights.

    ``converged`` means the projected-gradient tolerance was met. A small
    objective change can stop the search without certifying stationarity.
    """
    weights = np.maximum(np.asarray(initial_weights, dtype=np.float32), 0.0)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError("initial_weights must be a finite one-dimensional array")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if initial_step <= 0.0:
        raise ValueError("initial_step must be positive")
    if not 0.0 < backtracking < 1.0:
        raise ValueError("backtracking must lie strictly between zero and one")

    value, gradient = value_and_gradient(weights)
    gradient = np.asarray(gradient, dtype=np.float32)
    history = [float(value)]
    step_hint = float(initial_step)
    converged = False

    for iteration in range(1, max_iterations + 1):
        projected_gradient = weights - np.maximum(weights - gradient, 0.0)
        if float(np.linalg.norm(projected_gradient, ord=np.inf)) <= gradient_tolerance:
            converged = True
            break

        step = step_hint
        accepted = False
        while step >= minimum_step:
            candidate = np.maximum(weights - step * gradient, 0.0).astype(np.float32)
            direction = candidate - weights
            candidate_value, candidate_gradient = value_and_gradient(candidate)
            sufficient_decrease = value + armijo * float(np.dot(gradient, direction))
            if candidate_value <= sufficient_decrease:
                accepted = True
                break
            step *= backtracking

        if not accepted:
            break

        improvement = value - candidate_value
        scale = max(1.0, abs(value))
        weights = candidate
        value = float(candidate_value)
        gradient = np.asarray(candidate_gradient, dtype=np.float32)
        history.append(value)
        step_hint = min(float(initial_step), step / backtracking)

        if improvement <= relative_tolerance * scale:
            projected_gradient = weights - np.maximum(weights - gradient, 0.0)
            converged = (
                float(np.linalg.norm(projected_gradient, ord=np.inf))
                <= gradient_tolerance
            )
            break

    return ProjectedGradientResult(
        weights=weights.copy(),
        objective_history=tuple(history),
        iterations=len(history) - 1,
        converged=converged,
    )
