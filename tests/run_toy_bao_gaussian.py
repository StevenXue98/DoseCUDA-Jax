#!/usr/bin/env python3
"""Multistart Gaussian-gradient BAO on the full-voxel synthetic CUDA case.

This is an angle-search experiment, not a robust-delivery/SAM objective.
The frozen-weight perturbations use the angle-dependent candidate spot lattice
from run_toy_bao_baseline; accepted angles get fresh optimized weights and are
judged only by the exact, unsmoothed DoseCUDA loss.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run_toy_bao_baseline import ROOT, make_case, make_operator, score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=float, nargs="+", default=[-20.0, 23.0, 75.0])
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian angle-smoothing scale, in degrees")
    parser.add_argument("--pairs", type=int, default=8,
                        help="Antithetic perturbation pairs per angle step")
    parser.add_argument("--initial-step", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--reference-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_baseline"))
    parser.add_argument("--output-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_gaussian"))
    return parser.parse_args()


def reflect_angle(angle, low=-90.0, high=90.0):
    """Mirror outside the allowed interval for a defined smoothed boundary."""
    width = high - low
    folded = (float(angle) - low) % (2.0 * width)
    return low + min(folded, 2.0 * width - folded)


class Experiment:
    def __init__(self):
        (self.grid, self.plan, self.beam, self.body, self.target, self.oar,
         self.normal, self.loss) = make_case()
        self._operators = {}
        self._exact = {}
        self.frozen_forward_calls = 0
        self.inner_forward_vjp_calls = 0

    @staticmethod
    def key(angle):
        return round(float(angle), 6)

    def operator(self, angle):
        key = self.key(angle)
        if key not in self._operators:
            self._operators[key] = make_operator(
                key, self.grid, self.plan, self.beam)
        return self._operators[key]

    def fixed_weight_loss(self, angle, weights):
        reflected = reflect_angle(angle)
        dose = self.operator(reflected).dose(weights)
        self.frozen_forward_calls += 1
        return self.loss(dose)[0]

    def exact(self, angle):
        key = self.key(angle)
        if not -90.0 <= key <= 90.0:
            raise ValueError("angle outside the reference domain")
        if key not in self._exact:
            row = score(key, self.grid, self.plan, self.beam, self.body,
                        self.target, self.oar, self.normal, self.loss)
            self.inner_forward_vjp_calls += row["evaluations"]
            if not row["converged"]:
                raise RuntimeError(f"inner spot-weight solve failed at {key} deg")
            self._exact[key] = row
        return self._exact[key]

    def gaussian_gradient(self, angle, weights, sigma, pairs, rng):
        """Antithetic score estimator of a Gaussian-smoothed frozen-weight loss."""
        z = rng.standard_normal(pairs)
        differences = np.asarray([
            self.fixed_weight_loss(angle + sigma * value, weights)
            - self.fixed_weight_loss(angle - sigma * value, weights)
            for value in z
        ])
        return float(np.mean(differences * z) / (2.0 * sigma))


def optimize_start(experiment, start, args, rng):
    current = experiment.exact(start)
    history = [{"angle_deg": current["angle_deg"], "loss": current["loss"]}]
    estimates = []
    proposals = []
    # The floor avoids accepting changes comparable to repeated float32
    # CUDA/inner-solver variation near the broad optimum.
    min_improvement = 1.0e-5
    for _ in range(args.max_steps):
        gradient = experiment.gaussian_gradient(
            current["angle_deg"], current["weights"], args.sigma, args.pairs, rng)
        estimates.append(gradient)
        if not np.isfinite(gradient):
            raise RuntimeError("nonfinite Gaussian angle-gradient estimate")
        if abs(gradient) < 1.0e-9:
            break

        direction = -np.sign(gradient)
        accepted = None
        trials = []
        for step in (args.initial_step / 2.0**index for index in range(6)):
            angle = float(np.clip(current["angle_deg"] + direction * step,
                                  -90.0, 90.0))
            if abs(angle - current["angle_deg"]) < 1.0e-6:
                continue
            candidate = experiment.exact(angle)
            trials.append({"angle_deg": angle, "loss": candidate["loss"]})
            if candidate["loss"] < current["loss"] - min_improvement:
                accepted = candidate
                break
        proposals.append({"from_angle_deg": current["angle_deg"],
                          "gaussian_gradient": gradient, "trials": trials,
                          "accepted_angle_deg": None if accepted is None else
                          accepted["angle_deg"]})
        if accepted is None:
            break
        current = accepted
        history.append({"angle_deg": current["angle_deg"],
                        "loss": current["loss"]})
    return {"start_deg": float(start), "final_deg": current["angle_deg"],
            "initial_loss": history[0]["loss"], "final_loss": current["loss"],
            "target_d95": current["target_d95"], "oar_max": current["oar_max"],
            "accepted_steps": len(history) - 1, "history": history,
            "gradient_estimates": estimates, "proposals": proposals,
            "weights": current["weights"]}


def load_reference(directory):
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    if (summary["angle_start"], summary["angle_stop"], summary["angle_step"]) != (
        -90.0, 90.0, 2.0
    ) or summary["converged_candidates"] != 91 or summary["spot_count_per_angle"] != 45:
        raise ValueError("reference must be the complete default 91-angle toy scan")
    expected_case = {"grid_voxels": 21504, "body_voxels": 10296,
                     "target_voxels": 33, "oar_voxels": 33,
                     "normal_voxels": 10230, "target_prescription": 0.5,
                     "oar_limit": 0.3, "normal_limit": 0.5,
                     "energy_ids": [34, 38, 42]}
    if any(summary.get(key) != value for key, value in expected_case.items()):
        raise ValueError("reference summary is not the matching full-voxel toy case")
    with open(os.path.join(directory, "exhaustive_angles.csv"), newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 91 or not all(row["converged"] == "True" for row in rows):
        raise ValueError("reference CSV must contain 91 converged candidates")
    return rows, summary


def main():
    args = parse_args()
    if (not np.isfinite(args.sigma) or args.sigma <= 0 or args.pairs <= 0
            or not np.isfinite(args.initial_step) or args.initial_step <= 0
            or args.max_steps <= 0 or not all(
                np.isfinite(start) and -90.0 <= start <= 90.0
                for start in args.starts)):
        raise ValueError("invalid starts or optimizer settings")
    reference_rows, reference_summary = load_reference(args.reference_dir)
    grid_best_loss = min(float(row["loss"]) for row in reference_rows)
    experiment = Experiment()
    rng = np.random.default_rng(args.seed)
    runs = [optimize_start(experiment, start, args, rng) for start in args.starts]
    winner = min(runs, key=lambda run: run["final_loss"])

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "multistart_vs_grid.png")
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    grid_angles = [float(row["angle_deg"]) for row in reference_rows]
    grid_losses = [float(row["loss"]) for row in reference_rows]
    for axis in axes:
        axis.plot(grid_angles, grid_losses, color="0.5", label="91-angle grid reference")
        for run in runs:
            axis.plot([point["angle_deg"] for point in run["history"]],
                      [point["loss"] for point in run["history"]],
                      marker="o", linewidth=1.5,
                      label=f"start {run['start_deg']:g}°")
        axis.grid(alpha=0.25)
        axis.set_ylabel("Exact, inner-optimized loss")
    axes[0].legend()
    axes[0].set_xlim(-90, 90)
    axes[0].set_xlabel("Gantry angle (degrees)")
    axes[1].set_ylim(0, 0.0025)
    axes[1].set_xlim(-90, -30)
    axes[1].axvline(reference_summary["best_on_grid"]["angle_deg"],
                    color="0.4", linestyle=":", linewidth=1,
                    label="Grid minimum")
    axes[1].legend()
    axes[1].set_xlabel("Gantry angle (degrees)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary = {
        "method": "antithetic Gaussian gradient of fixed-weight candidate-angle loss; "
                  "exact inner solve and exact loss for each accepted angle",
        "not_robust_delivery": True,
        "starts_deg": args.starts,
        "sigma_deg": args.sigma,
        "antithetic_pairs": args.pairs,
        "initial_step_deg": args.initial_step,
        "max_steps": args.max_steps,
        "minimum_accepted_loss_decrease": 1.0e-5,
        "seed": args.seed,
        "grid_reference_best_angle_deg": reference_summary["best_on_grid"]["angle_deg"],
        "grid_reference_best_loss": grid_best_loss,
        "grid_reference_near_optimal_angles_deg": reference_summary[
            "angles_within_5pct_of_best_loss"],
        "frozen_forward_calls": experiment.frozen_forward_calls,
        "exact_inner_solves": len(experiment._exact),
        "inner_forward_vjp_calls": experiment.inner_forward_vjp_calls,
        "winner_start_deg": winner["start_deg"],
        "winner_final_angle_deg": winner["final_deg"],
        "winner_final_loss": winner["final_loss"],
        "runs": [{key: value for key, value in run.items() if key != "weights"}
                 for run in runs],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    weights_path = os.path.join(args.output_dir, "winner_weights.npz")
    np.savez_compressed(weights_path, angle_deg=winner["final_deg"],
                        weights=winner["weights"],
                        base_spot_list=experiment.beam.spot_list)

    print(f"Grid reference: {reference_summary['best_on_grid']['angle_deg']:g} deg, "
          f"loss={grid_best_loss:.6g}")
    for run in runs:
        print(f"Start {run['start_deg']:g} deg -> {run['final_deg']:g} deg; "
              f"loss {run['initial_loss']:.6g} -> {run['final_loss']:.6g}; "
              f"accepted steps={run['accepted_steps']}")
    print(f"Winner: {winner['final_deg']:g} deg, loss={winner['final_loss']:.6g}; "
          f"grid relative gap={(winner['final_loss'] / grid_best_loss - 1):+.2%}")
    print(f"Cost: {experiment.frozen_forward_calls} frozen-weight forwards, "
          f"{len(experiment._exact)} exact inner solves, "
          f"{experiment.inner_forward_vjp_calls} inner dose/VJP calls")
    print(f"Plot: {plot_path}\nSummary: {summary_path}\nWeights: {weights_path}")


if __name__ == "__main__":
    main()
