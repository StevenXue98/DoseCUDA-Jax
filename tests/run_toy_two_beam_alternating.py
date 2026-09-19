#!/usr/bin/env python3
"""Joint two-beam angle/weight descent on the full-voxel DoseCUDA toy case.

Weight blocks use matrix-free CUDA forward/VJP calls. Angle directions are
Gaussian-smoothed fixed-weight estimates, but every accepted move must lower
the *unsmoothed* DoseCUDA loss with those same weights. A rejected move in the
adaptive schedule triggers more weight steps before another angle attempt.
The accurate 2-degree grid is used only for retrospective comparison.
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from run_toy_bao_baseline import ROOT, make_case  # noqa: E402
from run_toy_three_beam_joint import make_joint_operator, solve_subset, without_weights  # noqa: E402
from run_toy_two_beam_bao import STARTS  # noqa: E402


GRID = Path(ROOT) / "test_phantom_output/bao_toy_two_beam_2deg_accurate"
OUTPUT = Path(ROOT) / "test_phantom_output/bao_toy_two_beam_alternating"
SCHEDULES = {"fixed20": (20,), "fixed100": (100,),
             "adaptive20_50_100": (20, 50, 100)}
MIN_DECREASE = 1.0e-7


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-moves", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--gaussian-pairs", type=int, default=8)
    parser.add_argument("--initial-step", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--starts", type=int, default=len(STARTS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (args.max_moves < 1 or args.sigma <= 0 or args.gaussian_pairs < 1 or
            args.initial_step <= 0 or not 1 <= args.starts <= len(STARTS)):
        parser.error("invalid experiment limits")
    return args


class ExactToyLoss:
    def __init__(self):
        self.case = make_case()
        self.forward_calls = 0
        self.vjp_calls = 0

    def operator(self, angles):
        grid, plan, beam = self.case[:3]
        return make_joint_operator(tuple(float(a) for a in angles), grid, plan, beam)

    def loss(self, angles, weights):
        dose = self.operator(angles).dose(weights)
        self.forward_calls += 1
        return float(self.case[-1](dose)[0])

    def value_and_gradient(self, operator, weights):
        self.forward_calls += 1
        self.vjp_calls += 1
        return operator.value_and_gradient(weights, self.case[-1])


def weight_block(experiment, angles, weights, iterations):
    """Spend at most `iterations` accepted L-BFGS-B steps at fixed angles."""
    operator = experiment.operator(angles)
    scale = 0.02
    initial = np.asarray(weights, dtype=np.float64)

    def objective(trial):
        value, gradient = experiment.value_and_gradient(operator, scale * trial)
        return 1.0e4 * value, 1.0e4 * scale * np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective, initial / scale, jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * len(initial),
        options={"maxiter": iterations, "gtol": 1.0e-12, "ftol": 1.0e-16,
                 "maxls": 40},
    )
    candidate = np.maximum(scale * result.x, 0.0).astype(np.float32)
    return candidate, {"iterations": int(result.nit),
                       "evaluations": int(result.nfev),
                       "message": str(result.message)}


def gaussian_direction(experiment, angles, weights, *, sigma, pairs,
                       rng=None, probes=None):
    if probes is None:
        if rng is None:
            raise ValueError("rng or explicit probes are required")
        probes = rng.standard_normal((pairs, 2))
    else:
        probes = np.asarray(probes, dtype=np.float64)
        if probes.shape != (pairs, 2):
            raise ValueError("probes must have shape (pairs, 2)")
    differences = np.asarray([
        experiment.loss(np.clip(angles + sigma * z, -90, 90), weights)
        - experiment.loss(np.clip(angles - sigma * z, -90, 90), weights)
        for z in probes
    ])
    gradient = np.mean(differences[:, None] * probes, axis=0) / (2 * sigma)
    norm = float(np.linalg.norm(gradient))
    return gradient, (-gradient / norm if norm > 1.0e-12 else None)


def try_angle_move(loss_at, angles, weights, current_loss, direction,
                   initial_step, min_decrease=MIN_DECREASE):
    """A rejected proposal never alters either angles or weights."""
    trials = []
    if direction is None:
        return None, trials
    for divisor in (1, 2, 4, 8, 16, 32):
        proposal = np.clip(angles + (initial_step / divisor) * direction,
                           -90.0, 90.0)
        if np.array_equal(proposal, angles):
            continue
        exact_loss = float(loss_at(proposal, weights))
        trials.append({"angles_deg": proposal.tolist(), "loss": exact_loss})
        if exact_loss < current_loss - min_decrease:
            return (proposal, exact_loss), trials
    return None, trials


def run_start(experiment, start, schedule, args, rng):
    started = perf_counter()
    forward_before, vjp_before = experiment.forward_calls, experiment.vjp_calls
    angles = np.asarray(start, dtype=np.float64)
    weights = experiment.operator(angles).weights.copy()
    current_loss = experiment.loss(angles, weights)
    history = [{"angles_deg": angles.tolist(), "loss": current_loss,
                "forward_calls": experiment.forward_calls - forward_before}]
    cycles = []
    for cycle in range(args.max_moves):
        accepted = False
        attempts = []
        spent = 0
        for target_steps in schedule:
            before = current_loss
            candidate, block = weight_block(experiment, angles, weights,
                                            target_steps - spent)
            spent = target_steps
            checked = experiment.loss(angles, candidate)
            if checked <= current_loss + 1.0e-8:
                weights, current_loss = candidate, checked
                block["weight_update_accepted"] = True
            else:
                block["weight_update_accepted"] = False
            gradient, direction = gaussian_direction(
                experiment, angles, weights, sigma=args.sigma,
                pairs=args.gaussian_pairs, rng=rng)
            moved, trials = try_angle_move(
                experiment.loss, angles, weights, current_loss, direction,
                args.initial_step)
            attempt = {"cumulative_weight_budget": target_steps,
                       "weight_block": block, "loss_before_weight": before,
                       "loss_after_weight": current_loss,
                       "angle_gradient": gradient.tolist(), "trials": trials,
                       "angle_accepted": moved is not None}
            attempts.append(attempt)
            if moved is not None:
                angles, current_loss = moved
                history.append({"angles_deg": angles.tolist(),
                                "loss": current_loss,
                                "forward_calls": experiment.forward_calls - forward_before})
                accepted = True
                break
        cycles.append({"cycle": cycle + 1, "attempts": attempts,
                       "accepted": accepted})
        print(f"{schedule}: start {start}, cycle {cycle + 1}, "
              f"angle {np.sort(angles).round(2).tolist()}, "
              f"joint loss {current_loss:.7g}, "
              f"attempted budgets {[a['cumulative_weight_budget'] for a in attempts]}, "
              f"accepted {accepted}", flush=True)
        if not accepted:
            break
    # This is a separate reference check; no influence matrix enters the loop.
    final_fixed_loss = experiment.loss(angles, weights)
    if abs(final_fixed_loss - current_loss) > 1.0e-8:
        raise AssertionError("final joint loss was not reproduced by DoseCUDA")
    search_seconds = perf_counter() - started
    reference = solve_subset(tuple(np.sort(angles)), experiment.case,
                             backend="influence_slsqp")
    return {"start_angles_deg": list(start), "schedule": list(schedule),
            "final_angles_deg": angles.tolist(),
            "final_joint_loss": final_fixed_loss,
            "final_reoptimized": without_weights(reference),
            "accepted_angle_moves": len(history) - 1,
            "weight_escalations": sum(len(c["attempts"]) - 1 for c in cycles),
            "forward_calls": experiment.forward_calls - forward_before,
            "vjp_calls": experiment.vjp_calls - vjp_before,
            "search_seconds": search_seconds,
            "reference_seconds": reference["solve_seconds"],
            "total_seconds": perf_counter() - started,
            "history": history, "cycles": cycles}


def load_grid():
    with (GRID / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["grid_step_deg"] != 2.0 or summary["grid_unique_pairs"] != 4186:
        raise ValueError("expected the completed accurate 2-degree grid")
    angles = np.arange(-90.0, 90.1, 2.0)
    surface = np.full((len(angles), len(angles)), np.nan)
    with (GRID / "angle_pairs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            i = int(round((float(row["angle_1_deg"]) + 90) / 2))
            j = int(round((float(row["angle_2_deg"]) + 90) / 2))
            surface[i, j] = surface[j, i] = float(row["loss"])
    if not np.all(np.isfinite(surface)):
        raise ValueError("accurate 2-degree grid is incomplete")
    return summary, surface


def plot_paths(output, surface, runs, grid_best):
    fig, axes = plt.subplots(1, len(STARTS), figsize=(16, 5), sharex=True,
                             sharey=True, constrained_layout=True)
    colors = {"fixed20": "#f59e0b", "fixed100": "#8b5cf6",
              "adaptive20_50_100": "#10b981"}
    for ax, start in zip(axes, STARTS):
        mesh = ax.imshow(surface.T, origin="lower", extent=(-90, 90, -90, 90),
                         norm=LogNorm(vmin=np.nanmin(surface),
                                      vmax=np.nanpercentile(surface, 95)),
                         cmap="viridis", aspect="equal")
        for run in runs:
            if tuple(run["start_angles_deg"]) != start:
                continue
            points = np.sort(np.asarray([row["angles_deg"] for row in
                                         run["history"]]), axis=1)
            name = run["strategy"]
            ax.plot(points[:, 0], points[:, 1], "o-", color=colors[name],
                    markersize=3, linewidth=1.5, label=name)
            ax.scatter(points[-1, 0], points[-1, 1], marker="x", s=55,
                       color=colors[name])
        ax.scatter(*grid_best["angles_deg"], marker="*", color="red", s=80,
                   edgecolors="white", linewidths=0.5, label="2° grid best")
        ax.set_title(f"Start {start}")
        ax.set_xlabel("Lower beam angle (°)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Upper beam angle (°)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.colorbar(mesh, ax=axes, label="Fully reoptimized DoseCUDA loss")
    fig.savefig(output / "alternating_paths_on_2deg_grid.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(STARTS), figsize=(15, 4), sharey=True,
                             constrained_layout=True)
    for ax, start in zip(axes, STARTS):
        for run in runs:
            if tuple(run["start_angles_deg"]) != start:
                continue
            points = run["history"]
            ax.plot([point["forward_calls"] for point in points],
                    [point["loss"] for point in points], "o-", markersize=3,
                    linewidth=1.5, label=run["strategy"],
                    color=colors[run["strategy"]])
            ax.scatter(run["forward_calls"], run["final_joint_loss"],
                       marker="x", color=colors[run["strategy"]])
        ax.axhline(grid_best["loss"], color="red", linestyle="--",
                   linewidth=1, label="2° grid best (fully reoptimized)")
        ax.set_title(f"Start {start}")
        ax.set_xlabel("Exact CUDA forward calls (incl. weight blocks)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Current joint angle-and-weight loss")
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "alternating_joint_loss_vs_cuda_calls.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    grid, surface = load_grid()
    experiment = ExactToyLoss()
    runs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start_index, start in enumerate(STARTS[:args.starts]):
        for strategy, schedule in SCHEDULES.items():
            rng = np.random.default_rng(args.seed + start_index)
            run = run_start(experiment, start, schedule, args, rng)
            run["strategy"] = strategy
            run["ratio_to_grid_best_after_reoptimization"] = (
                run["final_reoptimized"]["loss"] / grid["grid_best"]["loss"])
            runs.append(run)
            with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump({"case": "full-voxel two-beam toy, 90 joint spot weights",
                           "config": {"max_moves": args.max_moves,
                                      "sigma_deg": args.sigma,
                                      "gaussian_pairs": args.gaussian_pairs,
                                      "initial_step_deg": args.initial_step,
                                      "seed": args.seed,
                                      "min_exact_decrease": MIN_DECREASE},
                           "grid_best": grid["grid_best"],
                           "note": "2-degree grid is sampled reference, not global optimum",
                           "runs": runs}, handle, indent=2)
    if args.starts == len(STARTS):
        plot_paths(args.output_dir, surface, runs, grid["grid_best"])
    print(f"Saved {len(runs)} runs to {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
