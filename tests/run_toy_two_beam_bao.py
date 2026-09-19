#!/usr/bin/env python3
"""Two-angle BAO search versus a coarse CUDA reference on the toy phantom.

The angle-search direction comes from a Gaussian-smoothed, fixed-weight
candidate-plan loss. Every accepted angle pair is scored after jointly
re-optimizing all 90 spot weights with the original unsmoothed CUDA model.
This is neither a global BAO certificate nor a physical error/SAM objective.
"""

import argparse
import csv
import json
import os
from itertools import combinations_with_replacement

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402

from run_toy_bao_baseline import ROOT, make_case  # noqa: E402
from run_toy_three_beam_joint import (  # noqa: E402
    make_joint_operator,
    solve_subset,
    without_weights,
)


STARTS = ((-80.0, -20.0), (-20.0, 75.0), (23.0, 75.0))
MIN_IMPROVEMENT = 1.0e-5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-step", type=float, default=10.0,
                        help="Coarse reference spacing in degrees")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--pairs", type=int, default=8,
                        help="Antithetic Gaussian pairs per search step")
    parser.add_argument("--initial-step", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--reuse-reference", action="store_true",
                        help="Reuse a completed coarse grid in output-dir")
    parser.add_argument("--output-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam"))
    return parser.parse_args()


def canonical_pair(angles):
    values = np.asarray(angles, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("expected two finite gantry angles")
    if np.any(values < -90.0) or np.any(values > 90.0):
        raise ValueError("gantry angles must lie in [-90, 90] degrees")
    return tuple(round(float(value), 6) for value in np.sort(values))


def reflect_angle(angle):
    folded = (float(angle) + 90.0) % 360.0
    return -90.0 + min(folded, 360.0 - folded)


def check_beam_exchange_symmetry(case):
    """Verify angle sorting does not change dose if weight blocks are swapped."""
    grid, plan, beam = case[:3]
    first = np.linspace(0.002, 0.025, 45, dtype=np.float32)
    second = np.linspace(0.027, 0.004, 45, dtype=np.float32)
    forward = make_joint_operator((-35.0, 55.0), grid, plan, beam).dose(
        np.concatenate((first, second)))
    exchanged = make_joint_operator((55.0, -35.0), grid, plan, beam).dose(
        np.concatenate((second, first)))
    np.testing.assert_allclose(forward, exchanged, rtol=5.0e-5, atol=5.0e-6)
    return float(np.max(np.abs(forward - exchanged)))


class PairExperiment:
    def __init__(self):
        self.case = make_case()
        self.exact_rows = {}
        self.fixed_operators = {}
        self.fixed_forward_calls = 0
        self.inner_evaluations = 0
        self.warm_start_retries = 0

    def exact(self, angles, initial_weights=None):
        key = canonical_pair(angles)
        if key not in self.exact_rows:
            try:
                row = solve_subset(key, self.case, initial_weights)
            except RuntimeError:
                if initial_weights is None:
                    raise
                # A warm start can occasionally stall in float32; retry from
                # the same independent base vector used by the coarse grid.
                self.warm_start_retries += 1
                row = solve_subset(key, self.case)
            if row["spot_count"] != 90:
                raise AssertionError("two-beam candidate must have 90 spots")
            self.exact_rows[key] = row
            self.inner_evaluations += row["evaluations"]
        return self.exact_rows[key]

    def fixed_weight_loss(self, angles, weights):
        # Keep each weight block attached to its beam under perturbation.
        # Sorting angles here would silently exchange spot-weight blocks.
        key = tuple(round(reflect_angle(angle), 6) for angle in angles)
        if key not in self.fixed_operators:
            grid, plan, beam = self.case[:3]
            self.fixed_operators[key] = make_joint_operator(key, grid, plan, beam)
        dose = self.fixed_operators[key].dose(weights)
        self.fixed_forward_calls += 1
        return self.case[-1](dose)[0]

    def gaussian_gradient(self, angles, weights, sigma, pairs, rng):
        z = rng.standard_normal((pairs, 2))
        angles = np.asarray(angles, dtype=np.float64)
        differences = np.asarray([
            self.fixed_weight_loss(angles + sigma * direction, weights)
            - self.fixed_weight_loss(angles - sigma * direction, weights)
            for direction in z
        ])
        return np.mean(differences[:, None] * z, axis=0) / (2.0 * sigma)


def search_from(experiment, start, args, rng):
    current = experiment.exact(start)
    history = [{"angles_deg": current["angles_deg"], "loss": current["loss"]}]
    proposals = []
    for _ in range(args.max_steps):
        gradient = experiment.gaussian_gradient(
            current["angles_deg"], current["weights"], args.sigma, args.pairs, rng)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm):
            raise RuntimeError("nonfinite two-angle Gaussian direction")
        if norm < 1.0e-9:
            break
        direction = -gradient / norm
        accepted = None
        trials = []
        for index in range(6):
            step = args.initial_step / 2.0**index
            proposal = np.clip(np.asarray(current["angles_deg"]) + step * direction,
                               -90.0, 90.0)
            candidate_angles = canonical_pair(proposal)
            if candidate_angles == tuple(current["angles_deg"]):
                continue
            candidate = experiment.exact(candidate_angles, current["weights"])
            trials.append({"angles_deg": candidate["angles_deg"],
                           "loss": candidate["loss"]})
            if candidate["loss"] < current["loss"] - MIN_IMPROVEMENT:
                accepted = candidate
                break
        proposals.append({
            "from_angles_deg": current["angles_deg"],
            "gaussian_gradient": gradient.tolist(),
            "trials": trials,
            "accepted_angles_deg": None if accepted is None else
                                   accepted["angles_deg"],
        })
        if accepted is None:
            break
        current = accepted
        history.append({"angles_deg": current["angles_deg"],
                        "loss": current["loss"]})
    return {"start_angles_deg": list(start),
            "final": without_weights(current),
            "accepted_steps": len(history) - 1,
            "history": history, "proposals": proposals,
            "weights": current["weights"]}


def build_reference(experiment, angles):
    rows = []
    for index, first in enumerate(angles):
        for second in angles[index:]:
            pair = (first, second)
            try:
                row = experiment.exact(pair)
            except RuntimeError:
                # Preserve the stationarity requirement; only the starting
                # weights change when an independent float32 solve stalls.
                nearest = min(experiment.exact_rows.values(), key=lambda solved:
                              np.linalg.norm(np.asarray(solved["angles_deg"]) - pair))
                experiment.warm_start_retries += 1
                row = experiment.exact(pair, nearest["weights"])
            rows.append(row)
        print(f"Coarse grid: {index + 1}/{len(angles)} rows, "
              f"{len(rows)} unique angle pairs", flush=True)
    expected = len(angles) * (len(angles) + 1) // 2
    if len(rows) != expected:
        raise AssertionError("incomplete angle-pair reference")
    return rows


def load_reference(directory, step, angles):
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as handle:
        previous = json.load(handle)
    expected_count = len(angles) * (len(angles) + 1) // 2
    if (previous["grid_step_deg"] != step or
            previous["grid_unique_pairs"] != expected_count):
        raise ValueError("saved coarse reference does not match grid-step")
    rows = []
    with open(os.path.join(directory, "coarse_angle_pairs.csv"), newline="",
              encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "angles_deg": [float(raw["angle_1_deg"]),
                               float(raw["angle_2_deg"])],
                "loss": float(raw["loss"]),
                "target_d95": float(raw["target_d95"]),
                "oar_max": float(raw["oar_max"]),
                "beam_target_dose_fraction": [
                    float(raw["beam_1_target_share"]),
                    float(raw["beam_2_target_share"])],
                "beam_active_spots": [int(raw["beam_1_active_spots"]),
                                      int(raw["beam_2_active_spots"])],
                "solve_seconds": float(raw["solve_seconds"]),
                "evaluations": int(raw["evaluations"]),
                "projected_gradient_norm": float(raw["projected_gradient_norm"]),
            })
    expected_pairs = {canonical_pair(pair) for pair in
                      combinations_with_replacement(angles, 2)}
    actual_pairs = {canonical_pair(row["angles_deg"]) for row in rows}
    if len(rows) != expected_count or actual_pairs != expected_pairs:
        raise ValueError("saved coarse reference is incomplete")
    return rows, previous["grid_warm_start_retries"]


def save_results(args, grid_angles, reference, grid_retries, exchange_error,
                 runs, search_experiment):
    os.makedirs(args.output_dir, exist_ok=True)
    grid_best = min(reference, key=lambda row: row["loss"])
    winner = min(runs, key=lambda run: run["final"]["loss"])

    csv_path = os.path.join(args.output_dir, "coarse_angle_pairs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "angle_1_deg", "angle_2_deg", "loss", "target_d95", "oar_max",
            "beam_1_target_share", "beam_2_target_share",
            "beam_1_active_spots", "beam_2_active_spots", "solve_seconds",
            "evaluations", "projected_gradient_norm",
        ])
        writer.writeheader()
        for row in reference:
            writer.writerow({
                "angle_1_deg": row["angles_deg"][0],
                "angle_2_deg": row["angles_deg"][1],
                "loss": row["loss"],
                "target_d95": row["target_d95"],
                "oar_max": row["oar_max"],
                "beam_1_target_share": row["beam_target_dose_fraction"][0],
                "beam_2_target_share": row["beam_target_dose_fraction"][1],
                "beam_1_active_spots": row["beam_active_spots"][0],
                "beam_2_active_spots": row["beam_active_spots"][1],
                "solve_seconds": row["solve_seconds"],
                "evaluations": row["evaluations"],
                "projected_gradient_norm": row["projected_gradient_norm"],
            })

    positions = {float(angle): index for index, angle in enumerate(grid_angles)}
    landscape = np.full((len(grid_angles), len(grid_angles)), np.nan)
    for row in reference:
        first, second = row["angles_deg"]
        i, j = positions[first], positions[second]
        landscape[i, j] = landscape[j, i] = row["loss"]
    if not np.all(np.isfinite(landscape)) or np.any(landscape <= 0):
        raise AssertionError("incomplete or nonpositive loss landscape")
    fig, axis = plt.subplots(figsize=(9, 8), layout="constrained")
    extent = (-90 - args.grid_step / 2, 90 + args.grid_step / 2,
              -90 - args.grid_step / 2, 90 + args.grid_step / 2)
    image = axis.imshow(landscape, origin="lower", extent=extent,
                        norm=LogNorm(vmin=np.min(landscape),
                                     vmax=np.max(landscape)),
                        cmap="viridis", interpolation="nearest", aspect="equal")
    fig.colorbar(image, ax=axis, label="Jointly optimized toy loss (log color)")
    for run in runs:
        path = np.asarray([point["angles_deg"] for point in run["history"]])
        axis.plot(path[:, 0], path[:, 1], marker="o", linewidth=1.5,
                  label=f"start {run['start_angles_deg']}")
    axis.scatter(*grid_best["angles_deg"], marker="*", s=180, color="red",
                 edgecolor="white", label="Coarse-grid minimum", zorder=5)
    axis.set_xlim(-90, 90)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("Gantry angle 1 (degrees)")
    axis.set_ylabel("Gantry angle 2 (degrees)")
    axis.legend(loc="lower right", fontsize="small")
    plot_path = os.path.join(args.output_dir, "pair_landscape_and_paths.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary = {
        "case": "same full-voxel synthetic target/OAR/normal-tissue toy case",
        "not_robust_delivery": True,
        "grid_step_deg": args.grid_step,
        "grid_unique_pairs": len(reference),
        "grid_best": without_weights(grid_best),
        "grid_inner_evaluations": sum(row["evaluations"] for row in reference),
        "grid_warm_start_retries": grid_retries,
        "beam_exchange_max_abs_dose_error": exchange_error,
        "grid_inner_solve_seconds": sum(row["solve_seconds"] for row in reference),
        "search_starts_deg": [list(start) for start in STARTS],
        "sigma_deg": args.sigma,
        "antithetic_pairs": args.pairs,
        "initial_step_deg": args.initial_step,
        "max_steps": args.max_steps,
        "min_accepted_improvement": MIN_IMPROVEMENT,
        "seed": args.seed,
        "search_exact_inner_solves": len(search_experiment.exact_rows),
        "search_inner_evaluations": search_experiment.inner_evaluations,
        "search_warm_start_retries": search_experiment.warm_start_retries,
        "search_fixed_weight_forwards": search_experiment.fixed_forward_calls,
        "search_inner_solve_seconds": sum(
            row["solve_seconds"] for row in search_experiment.exact_rows.values()),
        "winner_start_deg": winner["start_angles_deg"],
        "winner_final": winner["final"],
        "runs": [{key: value for key, value in run.items() if key != "weights"}
                 for run in runs],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    weights_path = os.path.join(args.output_dir, "winner_weights.npz")
    np.savez_compressed(weights_path,
                        angles_deg=np.asarray(winner["final"]["angles_deg"]),
                        weights=winner["weights"],
                        base_spot_list=search_experiment.case[2].spot_list)
    return grid_best, winner, (plot_path, csv_path, summary_path, weights_path)


def main():
    args = parse_args()
    ratio = 180.0 / args.grid_step if args.grid_step > 0 else np.nan
    if (not np.isfinite(ratio) or not np.isclose(ratio, round(ratio), atol=1.0e-9)
            or not np.isfinite(args.sigma) or args.sigma <= 0
            or args.pairs <= 0 or not np.isfinite(args.initial_step)
            or args.initial_step <= 0 or args.max_steps <= 0):
        raise ValueError("invalid grid or search settings")
    grid_angles = np.linspace(-90, 90, round(ratio) + 1)
    symmetry_case = make_case()
    exchange_error = check_beam_exchange_symmetry(symmetry_case)
    if args.reuse_reference:
        reference, grid_retries = load_reference(
            args.output_dir, args.grid_step, grid_angles)
        print(f"Reused {len(reference)} converged coarse-grid pairs", flush=True)
    else:
        reference_experiment = PairExperiment()
        reference = build_reference(reference_experiment, grid_angles)
        grid_retries = reference_experiment.warm_start_retries
    search_experiment = PairExperiment()  # No free reuse of grid inner solves.
    rng = np.random.default_rng(args.seed)
    runs = [search_from(search_experiment, start, args, rng) for start in STARTS]
    grid_best, winner, paths = save_results(
        args, grid_angles, reference, grid_retries, exchange_error, runs,
        search_experiment)
    print(f"Coarse grid best: {grid_best['angles_deg']}, "
          f"loss={grid_best['loss']:.6g}")
    for run in runs:
        print(f"Start {run['start_angles_deg']} -> "
              f"{run['final']['angles_deg']}; "
              f"loss={run['final']['loss']:.6g}; "
              f"accepted={run['accepted_steps']}")
    print(f"Search winner: {winner['final']['angles_deg']}, "
          f"loss={winner['final']['loss']:.6g}, "
          f"grid relative gap={winner['final']['loss'] / grid_best['loss'] - 1:+.2%}")
    print(f"Search cost: {len(search_experiment.exact_rows)} fresh inner solves, "
          f"{search_experiment.fixed_forward_calls} fixed-weight forwards")
    print("Plot: {}\nGrid CSV: {}\nSummary: {}\nWeights: {}".format(*paths))


if __name__ == "__main__":
    main()
