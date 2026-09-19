#!/usr/bin/env python3
"""Scan one gantry angle with the rounded CUDA dose model and optimized weights.

This is a toy BAO landscape diagnostic, not a clinical plan or an angle
optimization algorithm. Every point gets fresh ray tracing/WET and starts its
weight solve from the same nominal-angle optimum.
"""

import argparse
import csv
import os
import sys
from copy import copy, deepcopy

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from DoseCUDA.impt_weight_optimization import (  # noqa: E402
    FixedGeometryPlanDose,
    bounded_lbfgsb,
    make_target_oar_loss,
)
from validate_spot_weight_gradients import create_case  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-width", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            REPOSITORY_ROOT, "test_phantom_output", "optimized_gantry_scan"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not np.isfinite(args.half_width) or args.half_width <= 0:
        raise ValueError("half-width must be finite and positive")
    if not np.isfinite(args.step) or args.step <= 0:
        raise ValueError("step must be finite and positive")
    half_steps = args.half_width / args.step
    if not np.isclose(half_steps, round(half_steps), atol=1.0e-9):
        raise ValueError("half-width must be an integer multiple of step")

    grid, plan, beam = create_case()
    nominal_angle = float(beam.gantry_angle)
    base_beam = deepcopy(beam)
    initial_weights = np.asarray(base_beam.spot_list[:, 2], dtype=np.float32).copy()
    shape = tuple(int(value) for value in grid.size)
    if shape != (24, 28, 32):
        raise ValueError("toy masks require the 24 x 28 x 32 validation grid")
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    target_mask = (z - 15) ** 2 + (y - 25) ** 2 + (x - 7) ** 2 <= 2**2
    oar_mask = (z - 10) ** 2 + (y - 25) ** 2 + (x - 13) ** 2 <= 2**2
    loss = make_target_oar_loss(target_mask, 0.50, oar_mask, 0.30, 0.25)

    def make_operator(angle):
        angle_plan = copy(plan)
        angle_beam = deepcopy(base_beam)
        angle_beam.gantry_angle = float(angle)
        angle_plan.beam_list = [angle_beam]
        return FixedGeometryPlanDose(grid, angle_plan)

    nominal_operator = make_operator(nominal_angle)
    nominal_solution = bounded_lbfgsb(
        lambda weights: nominal_operator.value_and_gradient(weights, loss),
        initial_weights,
    )
    if not nominal_solution.converged:
        raise RuntimeError(
            "nominal weight solve did not converge: " + nominal_solution.message
        )
    nominal_weights = nominal_solution.weights.copy()

    offsets = np.arange(-round(half_steps), round(half_steps) + 1)
    angles = nominal_angle + offsets * args.step
    rows = []
    for angle in angles:
        operator = make_operator(angle)
        fixed_loss = loss(operator.dose(nominal_weights))[0]
        solution = bounded_lbfgsb(
            lambda weights: operator.value_and_gradient(weights, loss),
            nominal_weights,
        )
        optimized_dose = operator.dose(solution.weights)
        row = {
            "gantry_angle_deg": float(angle),
            "fixed_weight_loss": float(fixed_loss),
            "optimized_loss": solution.objective,
            "converged": solution.converged,
            "projected_gradient_norm": solution.projected_gradient_norm,
            "iterations": solution.iterations,
            "evaluations": solution.evaluations,
            "target_mean_dose": float(np.mean(optimized_dose[target_mask])),
            "oar_max_dose": float(np.max(optimized_dose[oar_mask])),
            "full_grid_max_dose": float(np.max(optimized_dose)),
        }
        row.update(
            {f"spot_weight_{index}": float(weight)
             for index, weight in enumerate(solution.weights)}
        )
        rows.append(row)
        if solution.objective > fixed_loss + 1.0e-4:
            raise RuntimeError(
                f"optimized loss exceeded fixed-weight loss at {angle:.3f} deg"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "gantry_scan.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fixed = np.asarray([row["fixed_weight_loss"] for row in rows])
    optimized = np.asarray([row["optimized_loss"] for row in rows])
    weights = np.asarray(
        [[row[f"spot_weight_{index}"] for index in range(initial_weights.size)]
         for row in rows]
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(angles, fixed, label="Nominal weights held fixed")
    axes[0].plot(angles, optimized, label="Weights re-optimized")
    failed = np.asarray([not row["converged"] for row in rows], dtype=bool)
    if np.any(failed):
        axes[0].scatter(
            angles[failed], optimized[failed], marker="x", color="red",
            label="Solver check not passed", zorder=5,
        )
    axes[0].axvline(nominal_angle, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Toy target/OAR loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    for index in range(initial_weights.size):
        axes[1].plot(angles, weights[:, index], label=f"Spot {index + 1}")
    axes[1].axvline(nominal_angle, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Optimized spot weight")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    grid_max = [row["full_grid_max_dose"] for row in rows]
    axes[2].plot(angles, grid_max, color="tab:purple")
    axes[2].axvline(nominal_angle, color="0.5", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Gantry angle (degrees)")
    axes[2].set_ylabel("Full-grid maximum dose")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    png_path = os.path.join(args.output_dir, "gantry_scan.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    converged = sum(row["converged"] for row in rows)
    best = min(rows, key=lambda row: row["optimized_loss"])
    print(f"Nominal angle: {nominal_angle:.3f} deg")
    print(f"Scan: {len(rows)} points, {angles[0]:.3f} to {angles[-1]:.3f} deg")
    print(f"Nominal optimized loss: {nominal_solution.objective:.9g}")
    print(f"Lowest sampled toy loss angle: {best['gantry_angle_deg']:.3f} deg")
    print(f"Lowest sampled toy loss: {best['optimized_loss']:.9g}")
    print(f"Full-grid max dose at sampled best: "
          f"{best['full_grid_max_dose']:.6g}")
    print(f"Stationarity checks passed: {converged}/{len(rows)}")
    largest_weight = float(np.max(weights))
    print(f"Largest optimized spot weight: {largest_weight:.6g}")
    if largest_weight > 10.0 * float(np.max(nominal_weights)):
        print("WARNING: toy objective permits extreme spot weights outside its "
              "target/OAR masks; this scan is not a deliverable BAO plan.")
    print(f"CSV: {csv_path}")
    print(f"Plot: {png_path}")
    if converged != len(rows):
        print("WARNING: points marked with X did not pass the solver check; "
              "their optimized-loss values need independent verification.")


if __name__ == "__main__":
    main()
