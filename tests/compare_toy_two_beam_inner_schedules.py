#!/usr/bin/env python3
"""Matched-start 20/50/100-step alternating BAO versus full inner solves.

All four methods use the same toy case, starts, Gaussian probe vectors at each
outer cycle, angle bounds, line-search steps, and original DoseCUDA forward
model. Partial methods accept angles at fixed carried weights. The full-inner
method reoptimizes weights for every candidate angle before acceptance.
Retrospective full solves score every visited angle on one common scale.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402

from run_toy_three_beam_joint import solve_subset, without_weights  # noqa: E402
from run_toy_two_beam_bao import STARTS, canonical_pair  # noqa: E402
from run_toy_two_beam_alternating import (  # noqa: E402
    ExactToyLoss, MIN_DECREASE, OUTPUT, gaussian_direction, load_grid,
    try_angle_move, weight_block,
)


METHODS = {"20 steps": 20, "50 steps": 50, "100 steps": 100,
           "full inner solve": None}
COLORS = {"20 steps": "#f59e0b", "50 steps": "#06b6d4",
          "100 steps": "#8b5cf6", "full inner solve": "#ef4444"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-moves", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--gaussian-pairs", type=int, default=8)
    parser.add_argument("--initial-step", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--starts", type=int, default=len(STARTS))
    parser.add_argument("--output-dir", type=Path,
                        default=OUTPUT / "matched_schedules")
    args = parser.parse_args()
    if (args.max_moves < 1 or args.sigma <= 0 or args.gaussian_pairs < 1 or
            args.initial_step <= 0 or not 1 <= args.starts <= len(STARTS)):
        parser.error("invalid comparison limits")
    return args


def shared_probes(seed, start_index, cycle, pairs):
    """Common random numbers for methods at the same start/outer cycle."""
    return np.random.default_rng(
        np.random.SeedSequence([seed, start_index, cycle])
    ).standard_normal((pairs, 2))


class FullInnerReference:
    def __init__(self, experiment):
        self.experiment = experiment
        self.cache = {}
        self.solve_count = 0
        self.solve_seconds = 0.0

    def solve(self, angles, fallback_weights=None):
        key = canonical_pair(angles)
        if key not in self.cache:
            try:
                row = solve_subset(key, self.experiment.case,
                                   backend="influence_slsqp")
            except RuntimeError:
                if fallback_weights is None:
                    raise
                row = solve_subset(key, self.experiment.case, fallback_weights,
                                   backend="influence_slsqp")
            row["influence_objective"] = row["loss"]
            row["loss"] = self.experiment.loss(key, row["weights"])
            self.cache[key] = row
            self.solve_count += 1
            self.solve_seconds += row["solve_seconds"]
        return self.cache[key]


def run_matched(experiment, start, method, steps, args, start_index):
    started = perf_counter()
    fwd_before, vjp_before = experiment.forward_calls, experiment.vjp_calls
    reference = FullInnerReference(experiment)
    angles = np.asarray(start, dtype=np.float64)
    if steps is None:
        solved = reference.solve(angles)
        weights = solved["weights"].copy()
        current_loss = float(solved["loss"])
    else:
        weights = experiment.operator(angles).weights.copy()
        current_loss = experiment.loss(angles, weights)
    path = [{"angles_deg": angles.tolist(), "joint_loss": current_loss,
             "search_seconds": perf_counter() - started,
             "forward_calls": experiment.forward_calls - fwd_before}]
    cycles = []
    for cycle in range(args.max_moves):
        block = None
        if steps is not None:
            candidate, block = weight_block(experiment, angles, weights, steps)
            checked = experiment.loss(angles, candidate)
            block["accepted"] = checked <= current_loss + 1.0e-8
            if block["accepted"]:
                weights, current_loss = candidate, checked
        probes = shared_probes(args.seed, start_index, cycle, args.gaussian_pairs)
        gradient, direction = gaussian_direction(
            experiment, angles, weights, sigma=args.sigma,
            pairs=args.gaussian_pairs, probes=probes)
        fixed_trial_losses = []
        if steps is None:
            def candidate_loss(proposal, _):
                fixed_trial_losses.append(experiment.loss(proposal, weights))
                return reference.solve(proposal, weights)["loss"]
        else:
            candidate_loss = experiment.loss
        moved, trials = try_angle_move(
            candidate_loss, angles, weights, current_loss, direction,
            args.initial_step)
        if steps is None:
            for trial, fixed_loss in zip(trials, fixed_trial_losses):
                trial["loss_with_current_weights"] = fixed_loss
        cycles.append({"cycle": cycle + 1, "weight_block": block,
                       "gradient": gradient.tolist(),
                       "loss_before_angle": current_loss, "trials": trials,
                       "angle_accepted": moved is not None})
        if moved is None:
            break
        angles, current_loss = moved
        if steps is None:
            solved = reference.solve(angles)
            weights = solved["weights"].copy()
        path.append({"angles_deg": angles.tolist(), "joint_loss": current_loss,
                     "search_seconds": perf_counter() - started,
                     "forward_calls": experiment.forward_calls - fwd_before})
    search_seconds = perf_counter() - started
    final_joint_loss = (float(current_loss) if steps is None else
                        float(experiment.loss(angles, weights)))
    if abs(final_joint_loss - current_loss) > 1.0e-8:
        raise AssertionError("DoseCUDA did not reproduce final joint loss")
    # These solves are excluded from partial-method search time and never
    # influence its path; they put all visited angles on the same score scale.
    for point in path:
        row = reference.solve(point["angles_deg"])
        point["fully_reoptimized_loss"] = float(row["loss"])
    final_reference = reference.solve(angles)
    return {"start_angles_deg": list(start), "method": method,
            "weight_steps_per_cycle": steps, "path": path,
            "cycles": cycles, "final_angles_deg": angles.tolist(),
            "final_joint_loss": final_joint_loss,
            "final_fully_reoptimized": without_weights(final_reference),
            "accepted_angle_moves": len(path) - 1,
            "cuda_forward_calls": experiment.forward_calls - fwd_before,
            "cuda_vjp_calls": experiment.vjp_calls - vjp_before,
            "full_inner_solves": reference.solve_count,
            "full_inner_solve_seconds": reference.solve_seconds,
            "search_seconds": search_seconds,
            "total_seconds_including_retrospective": perf_counter() - started}


def plot_results(output, surface, runs, grid_best):
    fig, axes = plt.subplots(1, len(STARTS), figsize=(17, 5), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, start in zip(axes, STARTS):
        image = ax.imshow(surface.T, origin="lower", extent=(-90, 90, -90, 90),
                          norm=LogNorm(vmin=np.nanmin(surface),
                                       vmax=np.nanpercentile(surface, 95)),
                          cmap="viridis", aspect="equal")
        for run in runs:
            if tuple(run["start_angles_deg"]) != start:
                continue
            points = np.sort(np.asarray([point["angles_deg"] for point in
                                         run["path"]]), axis=1)
            ax.plot(points[:, 0], points[:, 1], "o-", markersize=3,
                    linewidth=1.5, color=COLORS[run["method"]],
                    label=run["method"])
            ax.scatter(*points[-1], marker="x", s=50,
                       color=COLORS[run["method"]])
        ax.scatter(*grid_best["angles_deg"], marker="*", s=80, color="white",
                   edgecolor="black", linewidth=0.5, label="2° grid best")
        ax.set_title(f"Same start {start}")
        ax.set_xlabel("Lower angle (°)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Upper angle (°)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.colorbar(image, ax=axes, label="Fully reoptimized DoseCUDA loss")
    fig.savefig(output / "matched_angle_paths.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(STARTS), figsize=(17, 5),
                             constrained_layout=True)
    for ax, start in zip(axes, STARTS):
        ax.imshow(surface.T, origin="lower", extent=(-90, 90, -90, 90),
                  norm=LogNorm(vmin=np.nanmin(surface),
                               vmax=np.nanpercentile(surface, 95)),
                  cmap="viridis", aspect="auto")
        all_points = []
        for run in runs:
            if tuple(run["start_angles_deg"]) != start:
                continue
            points = np.sort(np.asarray([point["angles_deg"] for point in
                                         run["path"]]), axis=1)
            all_points.extend(points)
            ax.plot(points[:, 0], points[:, 1], "o-", markersize=4,
                    linewidth=1.8, color=COLORS[run["method"]],
                    label=run["method"])
            ax.scatter(*points[-1], marker="x", s=65,
                       color=COLORS[run["method"]])
        all_points = np.asarray(all_points)
        ax.set_xlim(max(-90, np.min(all_points[:, 0]) - 5),
                    min(90, np.max(all_points[:, 0]) + 5))
        ax.set_ylim(max(-90, np.min(all_points[:, 1]) - 5),
                    min(90, np.max(all_points[:, 1]) + 5))
        ax.set_title(f"Same start {start}; zoomed paths")
        ax.set_xlabel("Lower angle (°)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Upper angle (°)")
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "matched_angle_paths_zoom.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(STARTS), figsize=(16, 4), sharey=True,
                             constrained_layout=True)
    for ax, start in zip(axes, STARTS):
        for run in runs:
            if tuple(run["start_angles_deg"]) != start:
                continue
            ax.plot(range(len(run["path"])),
                    [point["fully_reoptimized_loss"] for point in run["path"]],
                    "o-", markersize=3, linewidth=1.5,
                    color=COLORS[run["method"]], label=run["method"])
        ax.axhline(grid_best["loss"], color="black", linestyle="--",
                   linewidth=1, label="2° grid best")
        ax.set_title(f"Same start {start}")
        ax.set_xlabel("Accepted angle moves")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Fully reoptimized loss at visited angles")
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "matched_reoptimized_loss_paths.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    grid, surface = load_grid()
    experiment = ExactToyLoss()
    runs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start_index, start in enumerate(STARTS[:args.starts]):
        for method, steps in METHODS.items():
            run = run_matched(experiment, start, method, steps, args, start_index)
            run["ratio_to_grid_best"] = (
                run["final_fully_reoptimized"]["loss"] /
                grid["grid_best"]["loss"])
            runs.append(run)
            print(f"{start} | {method}: {run['accepted_angle_moves']} moves, "
                  f"angles {np.sort(run['final_angles_deg']).round(2).tolist()}, "
                  f"joint {run['final_joint_loss']:.7g}, "
                  f"full {run['final_fully_reoptimized']['loss']:.7g}, "
                  f"search {run['search_seconds']:.2f}s", flush=True)
            with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump({"case": "matched-start full-voxel 90-spot toy BAO",
                           "comparison": "same starts and cycle-indexed Gaussian probes",
                           "config": {"max_moves": args.max_moves,
                                      "sigma_deg": args.sigma,
                                      "gaussian_pairs": args.gaussian_pairs,
                                      "initial_step_deg": args.initial_step,
                                      "seed": args.seed,
                                      "min_exact_decrease": MIN_DECREASE},
                           "grid_best": grid["grid_best"],
                           "grid_is_sampled_not_global": True,
                           "runs": runs}, handle, indent=2)
    if args.starts == len(STARTS):
        plot_results(args.output_dir, surface, runs, grid["grid_best"])
    print(f"Saved {len(runs)} matched runs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
