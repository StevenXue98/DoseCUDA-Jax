#!/usr/bin/env python3
"""Paired short-horizon weight solves for the exact two-beam DoseCUDA toy.

At every cycle, fork the current weights. Give the unchanged angle and each
trial angle the same number of L-BFGS-B iterations, then retain the plan with
the lowest exact CUDA loss. The unchanged current plan is a safety option.
Full inner solves are used only for retrospective scoring, never selection.
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

from compare_toy_two_beam_inner_schedules import (  # noqa: E402
    FullInnerReference, shared_probes,
)
from run_toy_three_beam_joint import without_weights  # noqa: E402
from run_toy_two_beam_bao import STARTS  # noqa: E402
from run_toy_two_beam_alternating import (  # noqa: E402
    ExactToyLoss, MIN_DECREASE, OUTPUT, gaussian_direction, load_grid,
    weight_block,
)


STEPS = (5, 10, 20)
COLORS = {5: "#d946ef", 10: "#22c55e", 20: "#f59e0b"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--gaussian-pairs", type=int, default=8)
    parser.add_argument("--initial-step", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--starts", type=int, default=len(STARTS))
    parser.add_argument("--output-dir", type=Path,
                        default=OUTPUT / "paired_branches")
    args = parser.parse_args()
    if (not args.steps or any(n < 1 for n in args.steps) or
            len(args.steps) != len(set(args.steps)) or
            args.max_cycles < 1 or args.sigma <= 0 or
            args.gaussian_pairs < 1 or args.initial_step <= 0 or
            not 1 <= args.starts <= len(STARTS)):
        parser.error("invalid experiment limits")
    return args


def paired_trial(experiment, angles, weights, current_loss, direction,
                 steps, initial_step, *, weight_solver=weight_block,
                 min_decrease=MIN_DECREASE):
    """Compare equal-budget branches; return a recoverable winning state.

    The incumbent continuation is solved once and reused across line-search
    candidates. Every candidate starts from the *same pre-fork weights*.
    """
    angles = np.asarray(angles, dtype=np.float64)
    source_weights = np.asarray(weights).copy()
    stay_weights, stay_block = weight_solver(
        experiment, angles, source_weights.copy(), steps)
    stay_loss = float(experiment.loss(angles, stay_weights))
    winner = {"angles": angles.copy(), "weights": source_weights.copy(),
              "loss": float(current_loss), "branch": "original"}
    if stay_loss < winner["loss"]:
        winner = {"angles": angles.copy(), "weights": stay_weights.copy(),
                  "loss": stay_loss, "branch": "stay"}
    trials = []
    if direction is not None:
        for divisor in (1, 2, 4, 8, 16, 32):
            proposal = np.clip(angles + initial_step * direction / divisor,
                               -90.0, 90.0)
            if np.array_equal(proposal, angles):
                continue
            move_weights, block = weight_solver(
                experiment, proposal, source_weights.copy(), steps)
            move_loss = float(experiment.loss(proposal, move_weights))
            trials.append({"angles_deg": proposal.tolist(),
                           "loss": move_loss, "weight_block": block,
                           "beats_stay": move_loss < stay_loss - min_decrease})
            if move_loss < stay_loss - min_decrease and move_loss < current_loss:
                winner = {"angles": proposal.copy(),
                          "weights": move_weights.copy(),
                          "loss": move_loss, "branch": "move"}
                break
    if winner["loss"] > current_loss + 1.0e-10:
        raise AssertionError("paired selection increased exact joint loss")
    return winner, {"stay_loss": stay_loss, "stay_weight_block": stay_block,
                    "trials": trials}


def run_start(experiment, start, steps, args, start_index):
    started = perf_counter()
    fwd_before, vjp_before = experiment.forward_calls, experiment.vjp_calls
    reference = FullInnerReference(experiment)
    angles = np.asarray(start, dtype=np.float64)
    weights = experiment.operator(angles).weights.copy()
    weights, initial_block = weight_block(experiment, angles, weights, steps)
    current_loss = experiment.loss(angles, weights)
    path = [{"angles_deg": angles.tolist(), "joint_loss": current_loss,
             "branch": "initial", "forward_calls": experiment.forward_calls - fwd_before,
             "search_seconds": perf_counter() - started}]
    cycles = []
    for cycle in range(args.max_cycles):
        probes = shared_probes(args.seed, start_index, cycle,
                               args.gaussian_pairs)
        gradient, direction = gaussian_direction(
            experiment, angles, weights, sigma=args.sigma,
            pairs=args.gaussian_pairs, probes=probes)
        before = current_loss
        winner, details = paired_trial(
            experiment, angles, weights, current_loss, direction, steps,
            args.initial_step)
        angles, weights, current_loss = (winner["angles"], winner["weights"],
                                         winner["loss"])
        cycles.append({"cycle": cycle + 1, "loss_before": before,
                       "gradient": gradient.tolist(),
                       "selected_branch": winner["branch"],
                       "selected_loss": current_loss, **details})
        path.append({"angles_deg": angles.tolist(),
                     "joint_loss": current_loss, "branch": winner["branch"],
                     "forward_calls": experiment.forward_calls - fwd_before,
                     "search_seconds": perf_counter() - started})
        print(f"{start} | {steps} steps | cycle {cycle + 1}: "
              f"{winner['branch']}, angle {np.sort(angles).round(2).tolist()}, "
              f"joint loss {current_loss:.7g}", flush=True)
        if winner["branch"] == "original":
            break
    search_seconds = perf_counter() - started
    reproduced = experiment.loss(angles, weights)
    if abs(reproduced - current_loss) > 1.0e-8:
        raise AssertionError("final exact CUDA loss was not reproduced")
    search_forward_calls = experiment.forward_calls - fwd_before
    search_vjp_calls = experiment.vjp_calls - vjp_before
    # Reference solves are outside the search and do not select branches.
    for point in path:
        point["fully_reoptimized_loss"] = float(
            reference.solve(point["angles_deg"])["loss"])
    final_reference = reference.solve(angles)
    return {"start_angles_deg": list(start), "weight_steps_per_branch": steps,
            "initial_weight_block": initial_block, "path": path,
            "cycles": cycles, "final_angles_deg": angles.tolist(),
            "final_joint_loss": current_loss,
            "final_fully_reoptimized": without_weights(final_reference),
            "accepted_angle_moves": sum(c["selected_branch"] == "move"
                                        for c in cycles),
            "stay_cycles": sum(c["selected_branch"] == "stay" for c in cycles),
            "cuda_forward_calls": search_forward_calls,
            "cuda_vjp_calls": search_vjp_calls,
            "search_seconds": search_seconds,
            "full_inner_solves_for_scoring": reference.solve_count,
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
            points = np.sort(np.asarray([p["angles_deg"] for p in run["path"]]),
                             axis=1)
            ax.plot(points[:, 0], points[:, 1], "o-", markersize=3,
                    color=COLORS.get(run["weight_steps_per_branch"]),
                    label=f"paired {run['weight_steps_per_branch']}")
            ax.scatter(*points[-1], marker="x", s=55,
                       color=COLORS.get(run["weight_steps_per_branch"]))
        ax.scatter(*np.sort(start), marker="s", s=50, facecolor="white",
                   edgecolor="black", label="start")
        ax.scatter(*grid_best["angles_deg"], marker="*", color="red", s=80,
                   label="2° grid best")
        ax.set_title(f"Start {start}")
        ax.set_xlabel("Lower angle (°)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Upper angle (°)")
    axes[-1].legend(fontsize=8)
    fig.colorbar(image, ax=axes, label="Fully reoptimized DoseCUDA loss")
    fig.savefig(output / "paired_paths_on_2deg_grid.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    grid, surface = load_grid()
    experiment = ExactToyLoss()
    runs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start_index, start in enumerate(STARTS[:args.starts]):
        for steps in args.steps:
            run = run_start(experiment, start, steps, args, start_index)
            run["ratio_to_grid_best"] = (
                run["final_fully_reoptimized"]["loss"] /
                grid["grid_best"]["loss"])
            runs.append(run)
            with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump({"case": "full-voxel two-beam 90-spot toy",
                           "method": "paired equal-iteration branch continuation",
                           "config": {"steps": args.steps,
                                      "max_cycles": args.max_cycles,
                                      "sigma_deg": args.sigma,
                                      "gaussian_pairs": args.gaussian_pairs,
                                      "initial_step_deg": args.initial_step,
                                      "seed": args.seed,
                                      "min_move_decrease": MIN_DECREASE},
                           "grid_best": grid["grid_best"],
                           "grid_is_sampled_not_global": True,
                           "runs": runs}, handle, indent=2)
            print(f"RESULT {start} | {steps}: {run['accepted_angle_moves']} moves, "
                  f"joint {run['final_joint_loss']:.7g}, "
                  f"fully reoptimized {run['final_fully_reoptimized']['loss']:.7g}, "
                  f"search {run['search_seconds']:.2f}s", flush=True)
    if args.starts == len(STARTS):
        plot_results(args.output_dir, surface, runs, grid["grid_best"])
    print(f"Saved {len(runs)} paired runs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
