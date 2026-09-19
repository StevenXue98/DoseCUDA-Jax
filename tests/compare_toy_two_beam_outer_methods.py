#!/usr/bin/env python3
"""Compare nominal two-beam searches against the saved 2-degree CUDA grid.

Gaussian probes hold the current spot weights fixed; finite differences probe
the bilevel objective after a new joint weight solve at each offset. Both
methods accept steps only after an exact joint weight re-optimization. This
experiment is not SAM and does not model delivery uncertainty.
"""

import argparse
import csv
import json
import os
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402

from compute_toy_two_beam_reference import (  # noqa: E402
    case_signature, reference_landscape, write_json_atomic,
)
from run_toy_bao_baseline import ROOT, make_case  # noqa: E402
from run_toy_three_beam_joint import solve_subset, without_weights  # noqa: E402
from run_toy_two_beam_bao import PairExperiment, canonical_pair  # noqa: E402


STARTS = (
    (-80.0, -20.0), (-20.0, 75.0), (23.0, 75.0),
    (-80.0, 50.0), (-50.0, -50.0), (45.0, 80.0),
)
METHODS = ("gaussian_sigma2", "fd_h2", "fd_h4")
REPEAT_ANGLES = ((-56.0, 50.0), (-56.0, 52.0),
                 (-56.0, 54.0), (-60.0, 60.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam_2deg"))
    parser.add_argument("--output-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam_outer_comparison_24pairs"))
    parser.add_argument("--start-limit", type=int, default=len(STARTS))
    parser.add_argument("--time-budget", type=float, default=50.0,
                        help="Approximate wall-clock seconds per method/start")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--initial-step", type=float, default=12.0)
    parser.add_argument("--gaussian-pairs", type=int, default=24)
    parser.add_argument("--min-improvement", type=float, default=1.0e-5,
                        help="Minimum exact-loss drop to accept a step; "
                             "chosen to reject most repeat-solve noise")
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def load_grid(directory):
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["grid_step_deg"] != 2.0 or summary["grid_unique_pairs"] != 4186:
        raise ValueError("expected the completed 2-degree, 4186-pair reference")
    angles = np.arange(-90.0, 90.0 + 1.0, 2.0)
    rows = []
    with open(os.path.join(directory, "angle_pairs.csv"), newline="",
              encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append({"angles_deg": [float(raw["angle_1_deg"]),
                                         float(raw["angle_2_deg"])],
                         "loss": float(raw["loss"])})
    if len(rows) != 4186 or len({tuple(row["angles_deg"]) for row in rows}) != 4186:
        raise ValueError("reference CSV has missing or duplicate pairs")
    matrix = reference_landscape(rows, angles)
    if not np.isclose(np.min(matrix), summary["grid_best"]["loss"], rtol=1e-8):
        raise ValueError("reference CSV and summary disagree on the minimum")
    return summary, angles, matrix, {
        tuple(row["angles_deg"]): row["loss"] for row in rows
    }


def repeat_solver_check(grid_losses):
    """Vary only the weight initialization at a few jagged-grid locations."""
    case = make_case()
    initial_spot_weights = np.asarray(case[2].spot_list[:, 2], dtype=np.float32)
    base = np.concatenate((initial_spot_weights, initial_spot_weights))
    results = []
    for angles in REPEAT_ANGLES:
        attempts = []
        for name, initial in (("default", None), ("low", 0.2 * base),
                              ("high", 3.0 * base)):
            solved = solve_subset(angles, case, initial)
            attempts.append({"initialization": name,
                             "loss": solved["loss"],
                             "projected_gradient_norm": solved[
                                 "projected_gradient_norm"],
                             "solve_seconds": solved["solve_seconds"]})
        losses = [attempt["loss"] for attempt in attempts]
        row = {"angles_deg": list(angles),
               "saved_grid_loss": grid_losses[angles],
               "attempts": attempts,
               "repeat_range": max(losses) - min(losses),
               "saved_to_best_repeat_gap": grid_losses[angles] - min(losses)}
        results.append(row)
        print(f"Repeat {angles}: grid={grid_losses[angles]:.8g}, "
              f"new range={min(losses):.8g}..{max(losses):.8g}", flush=True)
    return results


def bilevel_fd_gradient(experiment, angles, weights, step):
    center = np.asarray(angles, dtype=np.float64)
    gradient = np.empty(2, dtype=np.float64)
    for axis in range(2):
        plus = center.copy()
        minus = center.copy()
        plus[axis] = min(90.0, plus[axis] + step)
        minus[axis] = max(-90.0, minus[axis] - step)
        high = experiment.exact(plus, weights)["loss"]
        low = experiment.exact(minus, weights)["loss"]
        gradient[axis] = (high - low) / (plus[axis] - minus[axis])
    return gradient


def search(method, start, args, seed):
    experiment = PairExperiment()  # No sharing of optimizations between methods.
    rng = np.random.default_rng(seed)
    started = perf_counter()
    current = experiment.exact(start)
    history = [{"angles_deg": current["angles_deg"], "loss": current["loss"],
                "elapsed_seconds": perf_counter() - started}]
    proposals = []
    stop_reason = "max_steps"
    for _ in range(args.max_steps):
        if perf_counter() - started >= args.time_budget:
            stop_reason = "time_budget"
            break
        if method == "gaussian_sigma2":
            gradient = experiment.gaussian_gradient(
                current["angles_deg"], current["weights"], 2.0,
                args.gaussian_pairs, rng)
        else:
            gradient = bilevel_fd_gradient(
                experiment, current["angles_deg"], current["weights"],
                2.0 if method == "fd_h2" else 4.0)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm):
            raise RuntimeError(f"nonfinite direction for {method} at {start}")
        if norm < 1.0e-12:
            stop_reason = "zero_direction"
            break
        if perf_counter() - started >= args.time_budget:
            stop_reason = "time_budget_after_gradient"
            break
        direction = -gradient / norm
        accepted = None
        trials = []
        for backtrack in range(6):
            if perf_counter() - started >= args.time_budget:
                stop_reason = "time_budget_during_line_search"
                break
            step = args.initial_step / (2.0 ** backtrack)
            trial_angles = canonical_pair(np.clip(
                np.asarray(current["angles_deg"]) + step * direction,
                -90.0, 90.0))
            if trial_angles == tuple(current["angles_deg"]):
                continue
            candidate = experiment.exact(trial_angles, current["weights"])
            trials.append({"angles_deg": candidate["angles_deg"],
                           "loss": candidate["loss"]})
            if candidate["loss"] <= current["loss"] - args.min_improvement:
                accepted = candidate
                break
        proposals.append({"from_angles_deg": current["angles_deg"],
                          "gradient": gradient.tolist(), "trials": trials,
                          "accepted_angles_deg": None if accepted is None else
                          accepted["angles_deg"]})
        if accepted is None:
            if not stop_reason.startswith("time_budget"):
                stop_reason = "no_sufficient_decrease"
            break
        current = accepted
        history.append({"angles_deg": current["angles_deg"],
                        "loss": current["loss"],
                        "elapsed_seconds": perf_counter() - started})
    search_seconds = perf_counter() - started
    base = np.asarray(experiment.case[2].spot_list[:, 2], dtype=np.float32)
    low_start = 0.2 * np.concatenate((base, base))
    verification = [current, solve_subset(current["angles_deg"], experiment.case),
                    solve_subset(current["angles_deg"], experiment.case, low_start)]
    verified = min(verification, key=lambda row: row["loss"])
    verification_seconds = perf_counter() - started - search_seconds
    return {"method": method, "start_angles_deg": list(start),
            "final": without_weights(current), "history": history,
            "verified_final": without_weights(verified),
            "verification_losses": [row["loss"] for row in verification],
            "verification_seconds": verification_seconds,
            "proposals": proposals, "stop_reason": stop_reason,
            "wall_seconds": search_seconds,
            "exact_inner_solves": len(experiment.exact_rows),
            "inner_evaluations": experiment.inner_evaluations,
            "inner_solve_seconds": sum(row["solve_seconds"] for row in
                                       experiment.exact_rows.values()),
            "fixed_weight_forwards": experiment.fixed_forward_calls,
            "warm_start_retries": experiment.warm_start_retries}


def plot_paths(path, matrix, grid_best, runs):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(18, 6.5),
                             sharex=True, sharey=True, layout="constrained")
    colors = plt.get_cmap("tab10").colors
    norm = LogNorm(vmin=float(np.min(matrix)), vmax=float(np.max(matrix)))
    for axis, method in zip(axes, METHODS):
        image = axis.imshow(matrix, origin="lower", extent=(-91, 91, -91, 91),
                            norm=norm, cmap="viridis", interpolation="nearest")
        for index, start in enumerate(STARTS):
            found = next((run for run in runs if run["method"] == method and
                          run["start_angles_deg"] == list(start)), None)
            if found is None:
                continue
            points = np.asarray([row["angles_deg"] for row in found["history"]])
            axis.plot(points[:, 0], points[:, 1], "-o", color=colors[index],
                      linewidth=1.6, markersize=4, label=str(start))
            axis.scatter(*points[-1], marker="x", s=65, color=colors[index],
                         linewidth=2)
        axis.scatter(*grid_best["angles_deg"], marker="*", s=160,
                     color="red", edgecolor="white", zorder=6)
        axis.set(title=method, xlabel="Angle 1 (degrees)",
                 xlim=(-90, 90), ylim=(-90, 90))
    axes[0].set_ylabel("Angle 2 (degrees)")
    axes[0].legend(title="Starts; x = final", loc="lower right", fontsize=7)
    fig.colorbar(image, ax=axes, label="Re-optimized toy loss (log color)",
                 fraction=0.025)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cost(path, grid_best, runs):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(15, 4), sharex=True,
                             sharey=True, layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for axis, method in zip(axes, METHODS):
        for index, start in enumerate(STARTS):
            found = next((run for run in runs if run["method"] == method and
                          run["start_angles_deg"] == list(start)), None)
            if found is None:
                continue
            axis.step([row["elapsed_seconds"] for row in found["history"]],
                      [row["loss"] / grid_best["loss"] for row in
                       found["history"]], where="post", color=colors[index],
                      linewidth=1.5, marker="o", markersize=3)
        axis.axhline(1.0, linestyle="--", linewidth=1, color="black")
        axis.set(title=method, xlabel="Search wall time (s)", yscale="log")
    axes[0].set_ylabel("Exact loss / 2-degree grid minimum")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if (not 1 <= args.start_limit <= len(STARTS) or args.time_budget <= 0 or
            args.max_steps <= 0 or args.initial_step <= 0 or
            args.gaussian_pairs <= 0 or args.min_improvement < 0):
        raise ValueError("invalid search settings")
    reference, _, matrix, grid_losses = load_grid(args.reference_dir)
    if case_signature(make_case()) != reference["case_signature"]:
        raise ValueError("saved reference belongs to a different toy case")
    os.makedirs(args.output_dir, exist_ok=True)
    config = {"start_limit": args.start_limit,
              "time_budget_seconds": args.time_budget,
              "max_steps": args.max_steps,
              "initial_step_deg": args.initial_step,
              "gaussian_pairs": args.gaussian_pairs,
              "min_improvement": args.min_improvement,
              "seed": args.seed,
              "methods": METHODS,
              "reference_case_signature": reference["case_signature"]}
    checkpoint_path = os.path.join(args.output_dir, "comparison_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as handle:
            state = json.load(handle)
        if state["config"] != json.loads(json.dumps(config)):
            raise ValueError("comparison checkpoint has different settings")
    else:
        state = {"config": config, "repeat_checks": None, "runs": []}
    if state["repeat_checks"] is None:
        state["repeat_checks"] = repeat_solver_check(grid_losses)
        write_json_atomic(checkpoint_path, state)
    total = len(METHODS) * args.start_limit
    for method_index, method in enumerate(METHODS):
        for start_index, start in enumerate(STARTS[:args.start_limit]):
            if any(run["method"] == method and
                   run["start_angles_deg"] == list(start) for run in state["runs"]):
                continue
            seed = args.seed + 10000 * method_index + start_index
            run = search(method, start, args, seed)
            state["runs"].append(run)
            write_json_atomic(checkpoint_path, state)
            print(f"Completed {len(state['runs'])}/{total}: {method} "
                  f"{start} -> {run['final']['angles_deg']}; "
                  f"loss={run['final']['loss']:.8g}, "
                  f"verified={run['verified_final']['loss']:.8g}; "
                  f"{run['wall_seconds']:.1f}s, "
                  f"{run['exact_inner_solves']} inner solves, "
                  f"{run['fixed_weight_forwards']} fixed forwards; "
                  f"stop={run['stop_reason']}", flush=True)
    grid_best = reference["grid_best"]
    state["grid_best"] = grid_best
    state["search_rerun_reference"] = False
    state["reference_directory"] = os.path.abspath(args.reference_dir)
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_json_atomic(summary_path, state)
    plot_paths(os.path.join(args.output_dir, "methods_on_2deg_reference.png"),
               matrix, grid_best, state["runs"])
    plot_cost(os.path.join(args.output_dir, "loss_vs_wall_time.png"),
              grid_best, state["runs"])
    for method in METHODS:
        rows = [run for run in state["runs"] if run["method"] == method]
        winner = min(rows, key=lambda row: row["verified_final"]["loss"])
        print(f"{method}: best verified loss "
              f"{winner['verified_final']['loss']:.8g} at "
              f"{winner['verified_final']['angles_deg']}, "
              f"grid gap={winner['verified_final']['loss']/grid_best['loss'] - 1:+.2%}",
              flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
