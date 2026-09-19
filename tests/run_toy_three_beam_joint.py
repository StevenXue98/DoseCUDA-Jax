#!/usr/bin/env python3
"""Joint three-beam spot-weight solves on the synthetic full-voxel BAO case.

For each fixed angle triple, solve every nonempty beam subset with the same
target/OAR/normal-tissue objective. This probes coupled weight optimization
and beam use; it is not a search over angle triples or a clinical plan.
"""

import argparse
import csv
import json
import os
from copy import copy
from itertools import combinations
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from DoseCUDA.impt_weight_optimization import (  # noqa: E402
    FixedGeometryPlanDose,
    InfluenceMatrixPlanDose,
    bounded_slsqp,
)
from run_toy_bao_baseline import (  # noqa: E402
    NORMAL_LIMIT,
    OAR_LIMIT,
    ROOT,
    RX,
    make_angle_beam,
    make_case,
    solve_weights,
)


ANGLE_TRIPLES = ((-60.0, 0.0, 60.0),
                 (-20.0, 23.0, 75.0),
                 (-50.0, 23.0, 75.0))
MONOTONICITY_TOLERANCE = 2.0e-5  # Near the float32 CUDA/solver loss variation.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_three_beam"))
    return parser.parse_args()


def make_joint_operator(angles, grid, plan, base_beam):
    angle_plan = copy(plan)
    angle_plan.beam_list = [make_angle_beam(angle, grid, base_beam)
                            for angle in angles]
    return FixedGeometryPlanDose(grid, angle_plan)


def solve_subset(angles, case, initial_weights=None, *, backend="cuda_lbfgsb"):
    if backend not in ("cuda_lbfgsb", "influence_slsqp"):
        raise ValueError("unsupported inner-solver backend")
    grid, plan, beam, body, target, oar, normal, loss = case
    operator = make_joint_operator(angles, grid, plan, beam)
    initial = (np.concatenate([member.weights for member in operator.beams])
               if initial_weights is None else
               np.asarray(initial_weights, dtype=np.float32))
    started = perf_counter()
    if backend == "cuda_lbfgsb":
        result, restarts, evaluations = solve_weights(operator, loss, initial)
    else:
        influence = InfluenceMatrixPlanDose(operator)
        result = bounded_slsqp(
            lambda weights: influence.value_and_gradient(weights, loss),
            initial, weight_scale=0.02, objective_scale=1.0e4,
            stationarity_tolerance=1.0e-6)
        restarts, evaluations = 0, result.evaluations
    seconds = perf_counter() - started
    if not result.converged:
        raise RuntimeError(
            f"joint inner solve failed for {angles}: "
            f"projected gradient {result.projected_gradient_norm:.3g}"
        )

    beam_doses = [member.dose(weights) for member, weights in zip(
        operator.beams, np.split(result.weights, len(angles))
    )]
    total_dose = np.sum(np.asarray(beam_doses), axis=0, dtype=np.float32)
    checked_loss, _ = loss(total_dose)
    if not np.isclose(result.objective, checked_loss, rtol=1.0e-3, atol=1.0e-7):
        raise AssertionError("beam dose sum does not reproduce optimized loss")

    beam_target_means = [float(np.mean(dose[target])) for dose in beam_doses]
    target_mean = float(np.mean(total_dose[target]))
    active_spots = [int(np.count_nonzero(weights > 1.0e-4)) for weights in
                    np.split(result.weights, len(angles))]
    target_error = total_dose[target] - RX
    oar_excess = np.maximum(total_dose[oar] - OAR_LIMIT, 0)
    normal_excess = np.maximum(total_dose[normal] - NORMAL_LIMIT, 0)
    return {
        "angles_deg": list(angles),
        "beam_count": len(angles),
        "spot_count": operator.n_spots,
        "loss": result.objective,
        "target_term": float(np.mean(target_error**2) / RX**2),
        "oar_term": float(np.mean(oar_excess**2) / RX**2),
        "normal_term": float(np.mean(normal_excess**2) / RX**2),
        "target_mean": target_mean,
        "target_d95": float(np.percentile(total_dose[target], 5)),
        "oar_max": float(np.max(total_dose[oar])),
        "body_max": float(np.max(total_dose[body])),
        "beam_target_mean_dose": beam_target_means,
        "beam_target_dose_fraction": [
            contribution / target_mean for contribution in beam_target_means
        ],
        "beam_active_spots": active_spots,
        "projected_gradient_norm": result.projected_gradient_norm,
        "restarts": restarts,
        "evaluations": evaluations,
        "solve_seconds": seconds,
        "weights": result.weights,
    }


def without_weights(row):
    return {key: value for key, value in row.items() if key != "weights"}


def run():
    args = parse_args()
    case = make_case()
    solved = {}
    triples = []
    for triple in ANGLE_TRIPLES:
        subsets = []
        for size in (1, 2, 3):
            for angles in combinations(triple, size):
                if angles not in solved:
                    solved[angles] = solve_subset(angles, case)
                subsets.append(solved[angles])
        singles = [row for row in subsets if row["beam_count"] == 1]
        pairs = [row for row in subsets if row["beam_count"] == 2]
        joint = solved[triple]
        best_single = min(singles, key=lambda row: row["loss"])
        best_pair = min(pairs, key=lambda row: row["loss"])
        if best_pair["loss"] > best_single["loss"] + MONOTONICITY_TOLERANCE:
            raise AssertionError(
                f"{triple}: best two-beam subset is worse than a feasible single"
            )
        if joint["loss"] > min(best_single["loss"], best_pair["loss"]) + \
                MONOTONICITY_TOLERANCE:
            raise AssertionError(
                f"{triple}: joint optimum is worse than a feasible subset"
            )
        triples.append({
            "angles_deg": list(triple),
            "best_single": without_weights(best_single),
            "best_pair": without_weights(best_pair),
            "joint": without_weights(joint),
            "all_subsets": [without_weights(row) for row in subsets],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "case": "same synthetic full-voxel toy objective and spot layout as BAO scan",
        "angle_triples_deg": [list(triple) for triple in ANGLE_TRIPLES],
        "inner_solve_count": len(solved),
        "total_inner_evaluations": sum(row["evaluations"] for row in solved.values()),
        "total_inner_solve_seconds": sum(row["solve_seconds"] for row in solved.values()),
        "triples": triples,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_path = os.path.join(args.output_dir, "triple_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "angles_deg", "best_single_loss", "best_pair_loss", "joint_loss",
            "joint_target_d95", "joint_oar_max", "beam_target_dose_fraction",
            "beam_active_spots", "joint_solve_seconds",
        ])
        writer.writeheader()
        for item in triples:
            joint = item["joint"]
            writer.writerow({
                "angles_deg": item["angles_deg"],
                "best_single_loss": item["best_single"]["loss"],
                "best_pair_loss": item["best_pair"]["loss"],
                "joint_loss": joint["loss"],
                "joint_target_d95": joint["target_d95"],
                "joint_oar_max": joint["oar_max"],
                "beam_target_dose_fraction": joint["beam_target_dose_fraction"],
                "beam_active_spots": joint["beam_active_spots"],
                "joint_solve_seconds": joint["solve_seconds"],
            })

    labels = [", ".join(f"{angle:g}°" for angle in triple)
              for triple in ANGLE_TRIPLES]
    positions = np.arange(len(triples))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), layout="constrained")
    for offset, key, name in ((-0.25, "best_single", "Best one beam"),
                              (0.0, "best_pair", "Best two beams"),
                              (0.25, "joint", "Joint three beams")):
        axes[0].bar(positions + offset,
                    [item[key]["loss"] for item in triples], width=0.23,
                    label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Optimized toy loss (log scale)")
    axes[0].set_xticks(positions, labels)
    axes[0].legend()
    base = np.zeros(len(triples))
    for index in range(3):
        fractions = [item["joint"]["beam_target_dose_fraction"][index]
                     for item in triples]
        axes[1].bar(positions, fractions, bottom=base, width=0.6,
                    label=f"Beam {index + 1}")
        base += fractions
    axes[1].set_ylabel("Share of mean target dose")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xticks(positions, labels)
    axes[1].legend()
    plot_path = os.path.join(args.output_dir, "joint_beam_comparison.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    weights_path = os.path.join(args.output_dir, "joint_weights.npz")
    np.savez_compressed(
        weights_path,
        angles_deg=np.asarray(ANGLE_TRIPLES),
        weights=np.stack([solved[triple]["weights"] for triple in ANGLE_TRIPLES]),
        base_spot_list=case[2].spot_list,
    )
    print(f"Solved {len(solved)} distinct beam subsets; "
          f"{summary['total_inner_evaluations']} dose/VJP evaluations; "
          f"{summary['total_inner_solve_seconds']:.2f} s in inner solves")
    for item in triples:
        single, pair, joint = (item[key] for key in
                               ("best_single", "best_pair", "joint"))
        print(f"Angles {item['angles_deg']}: loss one/two/three="
              f"{single['loss']:.6g}/{pair['loss']:.6g}/{joint['loss']:.6g}; "
              f"D95={joint['target_d95']:.4f}; OAR max={joint['oar_max']:.4f}; "
              f"target shares={[round(value, 3) for value in joint['beam_target_dose_fraction']]}; "
              f"active spots={joint['beam_active_spots']}")
    print(f"Plot: {plot_path}\nCSV: {csv_path}\nSummary: {summary_path}"
          f"\nWeights: {weights_path}")


if __name__ == "__main__":
    run()
