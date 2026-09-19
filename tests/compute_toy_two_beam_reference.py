#!/usr/bin/env python3
"""Resumable fixed-angle two-beam DoseCUDA reference; no angle optimization.

Only unordered angle pairs are solved because the two candidate beams are
identical except for gantry angle. Every pair gets a fresh joint 90-weight
solve. An atomic checkpoint allows interruption and continuation without
repeating completed pairs.
"""

import argparse
import csv
import hashlib
import json
import os
from itertools import combinations_with_replacement
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402

from run_toy_bao_baseline import (  # noqa: E402
    NORMAL_LIMIT,
    OAR_LIMIT,
    ROOT,
    RX,
    STATIONARITY_TOLERANCE,
    make_case,
)
from run_toy_three_beam_joint import solve_subset, without_weights  # noqa: E402
from run_toy_two_beam_bao import check_beam_exchange_symmetry  # noqa: E402


CSV_COLUMNS = (
    "angle_1_deg", "angle_2_deg", "loss", "target_d95", "oar_max",
    "beam_1_target_share", "beam_2_target_share", "beam_1_active_spots",
    "beam_2_active_spots", "solve_seconds", "evaluations",
    "projected_gradient_norm",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-step", type=float, default=2.0)
    parser.add_argument("--output-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam_2deg"))
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-new-pairs", type=int, default=None,
                        help="Stop after this many new solves; useful to test resume")
    parser.add_argument("--search-summary", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam", "summary.json"),
                        help="Existing search paths to overlay; never rerun search")
    return parser.parse_args()


def case_signature(case):
    grid, plan, beam, _, target, oar, normal, _ = case
    digest = hashlib.sha256()
    for data in (grid.HU, grid.origin, grid.spacing, beam.spot_list,
                 beam.iso, target, oar, normal,
                 np.asarray((beam.couch_angle, RX, OAR_LIMIT, NORMAL_LIMIT,
                             STATIONARITY_TOLERANCE), dtype=np.float64)):
        digest.update(np.ascontiguousarray(data).tobytes())
    digest.update(str(plan.machine_name).encode("utf-8"))
    return digest.hexdigest()


def write_json_atomic(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_or_start_checkpoint(path, step, pairs, signature):
    if not os.path.exists(path):
        return {"schema_version": 1, "grid_step_deg": step,
                "case_signature": signature, "total_pairs": len(pairs),
                "rows": [], "last_weights": None,
                "last_angles_deg": None, "warm_start_retries": 0}
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    if (state.get("schema_version") != 1 or state.get("grid_step_deg") != step
            or state.get("case_signature") != signature
            or state.get("total_pairs") != len(pairs)):
        raise ValueError("checkpoint does not match this case and angle grid")
    rows = state.get("rows")
    if not isinstance(rows, list) or len(rows) > len(pairs):
        raise ValueError("invalid checkpoint row count")
    for index, row in enumerate(rows):
        if tuple(row["angles_deg"]) != tuple(pairs[index]):
            raise ValueError(f"checkpoint angle order differs at row {index}")
        if (row["beam_count"] != 2 or row["spot_count"] != 90
                or not np.isfinite(row["loss"])
                or row["projected_gradient_norm"] > STATIONARITY_TOLERANCE):
            raise ValueError(f"checkpoint has an uncertified solve at row {index}")
    if rows and (state.get("last_weights") is None or
                 len(state["last_weights"]) != 90):
        raise ValueError("checkpoint is missing its last 90-weight vector")
    return state


def write_reference_csv(path, rows):
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def reference_landscape(rows, angles):
    positions = {float(angle): index for index, angle in enumerate(angles)}
    matrix = np.full((len(angles), len(angles)), np.nan)
    for row in rows:
        first, second = row["angles_deg"]
        i, j = positions[first], positions[second]
        matrix[i, j] = matrix[j, i] = row["loss"]
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
        raise AssertionError("completed reference has missing/invalid cells")
    return matrix


def plot_heatmap(path, matrix, angles, rows, runs, step):
    best = min(rows, key=lambda row: row["loss"])
    fig, axis = plt.subplots(figsize=(9, 8), layout="constrained")
    image = axis.imshow(matrix, origin="lower",
                        extent=(-90 - step / 2, 90 + step / 2,
                                -90 - step / 2, 90 + step / 2),
                        norm=LogNorm(vmin=float(np.min(matrix)),
                                     vmax=float(np.max(matrix))),
                        cmap="viridis", interpolation="nearest", aspect="equal")
    fig.colorbar(image, ax=axis, label="Jointly optimized toy loss (log color)")
    for run in runs:
        points = np.asarray([point["angles_deg"] for point in run["history"]])
        axis.plot(points[:, 0], points[:, 1], marker="o", linewidth=1.5,
                  label=f"prior start {run['start_angles_deg']}")
    axis.scatter(*best["angles_deg"], marker="*", s=180, color="red",
                 edgecolor="white", label="2-degree grid minimum", zorder=5)
    axis.set(xlim=(-90, 90), ylim=(-90, 90),
             xlabel="Gantry angle 1 (degrees)",
             ylabel="Gantry angle 2 (degrees)")
    axis.legend(loc="lower right", fontsize="small")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    ratio = 180.0 / args.grid_step if args.grid_step > 0 else np.nan
    if (not np.isfinite(ratio) or not np.isclose(ratio, round(ratio), atol=1.0e-9)
            or args.checkpoint_every <= 0 or args.progress_every <= 0
            or args.max_new_pairs is not None and args.max_new_pairs <= 0):
        raise ValueError("invalid grid or checkpoint settings")
    angles = np.linspace(-90.0, 90.0, round(ratio) + 1)
    pairs = [tuple(float(value) for value in pair)
             for pair in combinations_with_replacement(angles, 2)]
    case = make_case()
    signature = case_signature(case)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "reference_checkpoint.json")
    state = load_or_start_checkpoint(checkpoint_path, args.grid_step,
                                     pairs, signature)
    start_count = len(state["rows"])
    print(f"Reference grid: {len(angles)} angles, {len(pairs)} unique pairs; "
          f"resuming at {start_count}/{len(pairs)}", flush=True)
    exchange_error = check_beam_exchange_symmetry(case)
    started = perf_counter()
    new_count = 0
    for index in range(start_count, len(pairs)):
        pair = pairs[index]
        try:
            result = solve_subset(pair, case)
        except RuntimeError:
            previous = state["last_weights"]
            if previous is None:
                raise
            state["warm_start_retries"] += 1
            result = solve_subset(pair, case, np.asarray(previous, dtype=np.float32))
        state["rows"].append(without_weights(result))
        state["last_weights"] = result["weights"].tolist()
        state["last_angles_deg"] = list(pair)
        new_count += 1
        completed = index + 1
        if completed % args.checkpoint_every == 0 or completed == len(pairs):
            write_json_atomic(checkpoint_path, state)
        if completed % args.progress_every == 0 or completed == len(pairs):
            total_solve_seconds = sum(row["solve_seconds"] for row in state["rows"])
            mean = total_solve_seconds / completed
            eta = mean * (len(pairs) - completed) / 60.0
            print(f"Solved {completed}/{len(pairs)} pairs; "
                  f"mean inner solve {mean:.2f} s; estimated {eta:.1f} min left; "
                  f"this run {perf_counter() - started:.1f} s", flush=True)
        if args.max_new_pairs is not None and new_count >= args.max_new_pairs:
            write_json_atomic(checkpoint_path, state)
            print(f"Paused after {new_count} new pairs; resume with same command "
                  f"without --max-new-pairs", flush=True)
            return
    write_json_atomic(checkpoint_path, state)

    with open(args.search_summary, encoding="utf-8") as handle:
        previous_search = json.load(handle)
    runs = previous_search["runs"]
    rows = state["rows"]
    best = min(rows, key=lambda row: row["loss"])
    matrix = reference_landscape(rows, angles)
    csv_path = os.path.join(args.output_dir, "angle_pairs.csv")
    write_reference_csv(csv_path, rows)
    plot_path = os.path.join(args.output_dir, "pair_landscape_and_paths.png")
    plot_heatmap(plot_path, matrix, angles, rows, runs, args.grid_step)
    summary = {
        "case": "same full-voxel synthetic target/OAR/normal-tissue toy case",
        "case_signature": signature,
        "grid_step_deg": args.grid_step,
        "grid_unique_pairs": len(rows),
        "grid_best": best,
        "grid_inner_evaluations": sum(row["evaluations"] for row in rows),
        "grid_inner_solve_seconds": sum(row["solve_seconds"] for row in rows),
        "grid_warm_start_retries": state["warm_start_retries"],
        "beam_exchange_max_abs_dose_error": exchange_error,
        "search_rerun": False,
        "prior_search_summary": os.path.abspath(args.search_summary),
        "runs": runs,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_json_atomic(summary_path, summary)
    print(f"Finished {len(rows)} certified pairs. Best sampled angles "
          f"{best['angles_deg']}, loss={best['loss']:.8g}; "
          f"{summary['grid_inner_solve_seconds']/60:.1f} min in inner solves; "
          f"warm-start retries={state['warm_start_retries']}", flush=True)
    print(f"Heatmap: {plot_path}\nGrid CSV: {csv_path}\nSummary: {summary_path}\n"
          f"Checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
