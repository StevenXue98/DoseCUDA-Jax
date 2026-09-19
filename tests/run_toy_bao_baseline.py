#!/usr/bin/env python3
"""Exhaustive one-beam BAO reference on a fixed synthetic DoseCUDA case.

Every candidate angle gets the same target-centered spot lattice and an
independent nonnegative weight solve. This is an algorithm benchmark, not a
clinical plan or a continuous-angle optimum.
"""

import argparse
import csv
import json
import os
import sys
from copy import copy, deepcopy
from itertools import product

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from DoseCUDA.impt_weight_optimization import (  # noqa: E402
    FixedGeometryPlanDose,
    bounded_lbfgsb,
    make_target_oar_normal_tissue_loss,
)
from validate_spot_weight_gradients import create_case  # noqa: E402


RX = 0.50
OAR_LIMIT = 0.30
NORMAL_LIMIT = 0.50
NOMINAL_ANGLE = 23.0
TARGET_CENTER_ZYX = (15, 20, 7)
OAR_CENTER_ZYX = (15, 15, 9)
RADIUS_VOXELS = 2
ENERGY_IDS = (34, 38, 42)
X_OFFSETS_MM = (-12, -6, 0, 6, 12)
Y_OFFSETS_MM = (-6, 0, 6)
STATIONARITY_TOLERANCE = 2.0e-4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=-90.0)
    parser.add_argument("--stop", type=float, default=90.0)
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "test_phantom_output", "bao_toy_baseline"),
    )
    return parser.parse_args()


def spherical_mask(shape, center, radius):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return sum((coordinate - position) ** 2 for coordinate, position in zip(
        (z, y, x), center
    )) <= radius**2


def make_case():
    grid, plan, beam = create_case()
    shape = tuple(int(value) for value in grid.size)
    if shape != (24, 28, 32):
        raise ValueError("BAO toy case requires the 24 x 28 x 32 phantom")
    body = np.asarray(grid.HU) > -500.0
    target = spherical_mask(shape, TARGET_CENTER_ZYX, RADIUS_VOXELS)
    oar = spherical_mask(shape, OAR_CENTER_ZYX, RADIUS_VOXELS)
    if np.any(target & oar) or not np.all(body[target]) or not np.all(body[oar]):
        raise ValueError("target and OAR must be disjoint and inside water")
    normal = body & ~target & ~oar

    beam.resetSpots()
    for energy_id, x_offset, y_offset in product(
        ENERGY_IDS, X_OFFSETS_MM, Y_OFFSETS_MM
    ):
        beam.addSingleSpot(-23 + x_offset, 8 + y_offset, 0.02, energy_id)
    if beam.n_spots != 45:
        raise AssertionError("unexpected BAO candidate count")
    loss = make_target_oar_normal_tissue_loss(
        target, RX, oar, OAR_LIMIT, normal, NORMAL_LIMIT
    )
    return grid, plan, beam, body, target, oar, normal, loss


def projected_spot_x(angle, grid, beam):
    """Project target center using DoseCUDA's gantry+180 head convention."""
    z, y, x = TARGET_CENTER_ZYX
    target_xyz = np.asarray(grid.origin, dtype=np.float64) + np.asarray(
        (x, y, z), dtype=np.float64
    ) * np.asarray(grid.spacing, dtype=np.float64)
    relative = target_xyz - np.asarray(beam.iso, dtype=np.float64)
    couch = np.deg2rad(float(beam.couch_angle))
    rotated_x = relative[0] * np.cos(couch) - relative[2] * np.sin(couch)
    gantry = np.deg2rad(float(angle) + 180.0)
    return -rotated_x * np.cos(gantry) - relative[1] * np.sin(gantry)


def make_operator(angle, grid, plan, base_beam):
    angle_plan = copy(plan)
    angle_beam = deepcopy(base_beam)
    angle_beam.gantry_angle = float(angle)
    angle_beam.spot_list[:, 0] += np.float32(
        projected_spot_x(angle, grid, base_beam)
        - projected_spot_x(NOMINAL_ANGLE, grid, base_beam)
    )
    angle_plan.beam_list = [angle_beam]
    return FixedGeometryPlanDose(grid, angle_plan)


def solve_weights(operator, loss, initial_weights):
    """Retry float32 objective-change stalls; never waive stationarity."""
    attempts = []
    weights = initial_weights
    for _ in range(6):
        result = bounded_lbfgsb(
            lambda trial: operator.value_and_gradient(trial, loss),
            weights,
            stationarity_tolerance=STATIONARITY_TOLERANCE,
        )
        attempts.append(result)
        if result.converged:
            break
        weights = result.weights
    return attempts[-1], len(attempts) - 1, sum(
        attempt.evaluations for attempt in attempts
    )


def score(angle, grid, plan, beam, body, target, oar, normal, loss):
    operator = make_operator(angle, grid, plan, beam)
    result, restarts, evaluations = solve_weights(
        operator, loss, np.asarray(beam.spot_list[:, 2], dtype=np.float32)
    )
    dose = operator.dose(result.weights)
    target_error = dose[target] - RX
    oar_excess = np.maximum(dose[oar] - OAR_LIMIT, 0.0)
    normal_excess = np.maximum(dose[normal] - NORMAL_LIMIT, 0.0)
    scale = RX**2
    terms = {
        "target_term": float(np.mean(target_error**2) / scale),
        "oar_term": float(np.mean(oar_excess**2) / scale),
        "normal_term": float(np.mean(normal_excess**2) / scale),
    }
    if not np.isclose(result.objective, sum(terms.values()), rtol=1.0e-3,
                      atol=1.0e-7):
        raise AssertionError("reported terms do not reproduce the optimized loss")
    return {
        "angle_deg": float(angle),
        "loss": result.objective,
        "converged": result.converged,
        "projected_gradient_norm": result.projected_gradient_norm,
        "restarts": restarts,
        "evaluations": evaluations,
        **terms,
        "target_mean": float(np.mean(dose[target])),
        "target_d95": float(np.percentile(dose[target], 5)),
        "oar_max": float(np.max(dose[oar])),
        "oar_over_limit_voxels": int(np.count_nonzero(oar_excess > 0)),
        "body_max": float(np.max(dose[body])),
        "active_spots": int(np.count_nonzero(result.weights > 1.0e-4)),
        "weights": result.weights,
    }


def main():
    args = parse_args()
    values = (args.start, args.stop, args.step)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("angle bounds and step must be finite")
    if args.step <= 0 or args.stop <= args.start:
        raise ValueError("require positive step and stop > start")
    interval_count = (args.stop - args.start) / args.step
    if not np.isclose(interval_count, round(interval_count), atol=1.0e-9):
        raise ValueError("angle interval must be divisible by step")

    grid, plan, beam, body, target, oar, normal, loss = make_case()
    angles = np.linspace(args.start, args.stop, round(interval_count) + 1)
    nominal = score(NOMINAL_ANGLE, grid, plan, beam, body, target, oar, normal, loss)
    rows = [score(angle, grid, plan, beam, body, target, oar, normal, loss)
            for angle in angles]
    best = min(rows, key=lambda row: row["loss"])
    near_best = [row["angle_deg"] for row in rows
                 if row["loss"] <= 1.05 * best["loss"]]

    # Show that the named OAR materially changes the nominal inner solution.
    no_oar_loss = make_target_oar_normal_tissue_loss(
        target, RX, oar, OAR_LIMIT, normal, NORMAL_LIMIT, oar_weight=0.0
    )
    no_oar_operator = make_operator(NOMINAL_ANGLE, grid, plan, beam)
    no_oar_result, _, _ = solve_weights(
        no_oar_operator, no_oar_loss,
        np.asarray(beam.spot_list[:, 2], dtype=np.float32),
    )
    no_oar_dose = no_oar_operator.dose(no_oar_result.weights)
    if nominal["oar_term"] <= 0 or np.max(no_oar_dose[oar]) <= nominal["oar_max"]:
        raise AssertionError("toy OAR penalty did not create an active tradeoff")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "exhaustive_angles.csv")
    columns = [key for key in rows[0] if key != "weights"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(angles, [row["loss"] for row in rows], label="Inner-optimized loss")
    axes[0].set_ylabel("Three-structure toy loss")
    axes[1].plot(angles, [row["target_d95"] for row in rows], label="Target D95")
    axes[1].axhline(0.95 * RX, color="tab:green", linestyle=":",
                    label="95% of toy prescription")
    axes[1].set_ylabel("Target dose")
    axes[2].plot(angles, [row["oar_max"] for row in rows], label="OAR maximum")
    axes[2].axhline(OAR_LIMIT, color="tab:red", linestyle=":",
                    label="Toy OAR limit")
    axes[2].set_ylabel("OAR dose")
    axes[2].set_xlabel("Gantry angle (degrees)")
    for ax in axes:
        ax.axvline(NOMINAL_ANGLE, color="0.5", linestyle="--", linewidth=1)
        ax.axvline(best["angle_deg"], color="tab:orange", linestyle="--",
                   linewidth=1)
        ax.grid(alpha=0.25)
        ax.legend()
    failures = [row for row in rows if not row["converged"]]
    if failures:
        axes[0].scatter([row["angle_deg"] for row in failures],
                        [row["loss"] for row in failures], marker="x",
                        color="red", zorder=5, label="Unresolved inner solve")
        axes[0].legend()
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "exhaustive_angles.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary = {
        "case": "synthetic water phantom; not for clinical use",
        "dose_grid_shape_zyx": list(grid.HU.shape),
        "grid_voxels": int(grid.HU.size),
        "body_voxels": int(np.count_nonzero(body)),
        "target_voxels": int(np.count_nonzero(target)),
        "oar_voxels": int(np.count_nonzero(oar)),
        "normal_voxels": int(np.count_nonzero(normal)),
        "spot_count_per_angle": beam.n_spots,
        "energy_ids": list(ENERGY_IDS),
        "target_prescription": RX,
        "oar_limit": OAR_LIMIT,
        "normal_limit": NORMAL_LIMIT,
        "inner_stationarity_tolerance": STATIONARITY_TOLERANCE,
        "angle_start": args.start,
        "angle_stop": args.stop,
        "angle_step": args.step,
        "nominal": {key: value for key, value in nominal.items() if key != "weights"},
        "best_on_grid": {key: value for key, value in best.items() if key != "weights"},
        "angles_within_5pct_of_best_loss": near_best,
        "no_oar_nominal": {
            "converged": no_oar_result.converged,
            "target_d95": float(np.percentile(no_oar_dose[target], 5)),
            "oar_max": float(np.max(no_oar_dose[oar])),
        },
        "converged_candidates": len(rows) - len(failures),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    weights_path = os.path.join(args.output_dir, "selected_plan_weights.npz")
    np.savez_compressed(
        weights_path,
        best_angle_deg=best["angle_deg"],
        best_weights=best["weights"],
        nominal_angle_deg=NOMINAL_ANGLE,
        nominal_weights=nominal["weights"],
        base_spot_list=beam.spot_list,
    )

    print(f"Voxels: grid={grid.HU.size}, body={body.sum()}, "
          f"target={target.sum()}, OAR={oar.sum()}")
    print(f"Candidate spots per angle: {beam.n_spots}; "
          f"angles: {len(rows)} from {angles[0]:.1f} to {angles[-1]:.1f} deg")
    print(f"Inner solves passed: {len(rows) - len(failures)}/{len(rows)}")
    print(f"Nominal {NOMINAL_ANGLE:.1f} deg: loss={nominal['loss']:.6g}, "
          f"D95={nominal['target_d95']:.6g}, OAR max={nominal['oar_max']:.6g}")
    print(f"Best sampled {best['angle_deg']:.1f} deg: loss={best['loss']:.6g}, "
          f"D95={best['target_d95']:.6g}, OAR max={best['oar_max']:.6g}")
    print(f"Angles within 5% of best loss: {near_best}")
    print(f"Nominal without OAR penalty: D95="
          f"{summary['no_oar_nominal']['target_d95']:.6g}, "
          f"OAR max={summary['no_oar_nominal']['oar_max']:.6g}")
    print(f"CSV: {csv_path}\nPlot: {plot_path}\nSummary: {summary_path}"
          f"\nSelected weights: {weights_path}")
    if failures or not nominal["converged"] or not no_oar_result.converged:
        raise RuntimeError("some inner solves failed stationarity; exhaustive "
                           "reference is incomplete (see CSV)")


if __name__ == "__main__":
    main()
