#!/usr/bin/env python3
"""Check that the small-plan double-precision inner solve is start-stable."""

import json

import numpy as np

from DoseCUDA.impt_weight_optimization import InfluenceMatrixPlanDose
from run_toy_bao_baseline import make_case
from run_toy_three_beam_joint import make_joint_operator, solve_subset


ANGLES = ((-56.0, 50.0), (-56.0, 52.0),
          (-56.0, 54.0), (-60.0, 60.0))


def main():
    case = make_case()
    base_spot_weights = np.asarray(case[2].spot_list[:, 2], dtype=np.float32)
    base = np.concatenate((base_spot_weights, base_spot_weights))
    operator = make_joint_operator(ANGLES[1], *case[:3])
    influence = InfluenceMatrixPlanDose(operator)
    trial = np.linspace(0.001, 0.04, operator.n_spots)
    direct_dose = operator.dose(trial)
    basis_dose = influence.dose(trial)
    max_dose_error = float(np.max(np.abs(direct_dose - basis_dose)))
    np.testing.assert_allclose(direct_dose, basis_dose,
                               rtol=5.0e-5, atol=5.0e-6)
    _, weight_gradient = influence.value_and_gradient(trial, case[-1])
    for index in (0, 17, 44, 45, 89):
        plus, minus = trial.copy(), trial.copy()
        plus[index] += 1.0e-5
        minus[index] -= 1.0e-5
        finite_difference = (
            influence.value_and_gradient(plus, case[-1])[0]
            - influence.value_and_gradient(minus, case[-1])[0]
        ) / (2.0e-5)
        np.testing.assert_allclose(weight_gradient[index], finite_difference,
                                   rtol=1.0e-5, atol=1.0e-8)
    rows = []
    for angles in ANGLES:
        attempts = []
        for name, initial in (("default", None), ("low", 0.2 * base),
                              ("high", 3.0 * base)):
            result = solve_subset(angles, case, initial,
                                  backend="influence_slsqp")
            attempts.append({"initialization": name, "loss": result["loss"],
                             "projected_gradient_norm": result[
                                 "projected_gradient_norm"],
                             "seconds": result["solve_seconds"]})
        losses = [attempt["loss"] for attempt in attempts]
        relative_spread = (max(losses) - min(losses)) / min(losses)
        if relative_spread > 1.0e-3:
            raise AssertionError(f"inner solve varies by {relative_spread:.2%} "
                                 f"at {angles}")
        rows.append({"angles_deg": angles, "relative_loss_spread": relative_spread,
                     "attempts": attempts})
        print(f"{angles}: loss {min(losses):.9g}..{max(losses):.9g}; "
              f"relative spread {relative_spread:.3%}", flush=True)
    print(json.dumps({"max_basis_dose_error": max_dose_error,
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
