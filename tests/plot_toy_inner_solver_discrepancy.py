#!/usr/bin/env python3
"""Compare legacy and accurate inner solves on the same 2-degree angle grid."""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402

from compute_toy_two_beam_reference import write_json_atomic  # noqa: E402
from run_toy_bao_baseline import ROOT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam_2deg"))
    parser.add_argument("--accurate-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam_2deg_accurate"))
    return parser.parse_args()


def load_matrix(directory):
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["grid_step_deg"] != 2.0 or summary["grid_unique_pairs"] != 4186:
        raise ValueError(f"{directory} is not a complete 2-degree reference")
    angles = np.arange(-90.0, 90.0 + 1.0, 2.0)
    positions = {float(angle): index for index, angle in enumerate(angles)}
    matrix = np.full((len(angles), len(angles)), np.nan)
    with open(os.path.join(directory, "angle_pairs.csv"), newline="",
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            i = positions[float(row["angle_1_deg"])]
            j = positions[float(row["angle_2_deg"])]
            if np.isfinite(matrix[i, j]):
                raise ValueError("duplicate angle pair")
            matrix[i, j] = matrix[j, i] = float(row["loss"])
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
        raise ValueError("reference has missing or nonpositive losses")
    return summary, matrix


def main():
    args = parse_args()
    legacy, old = load_matrix(args.legacy_dir)
    accurate, new = load_matrix(args.accurate_dir)
    if legacy["case_signature"] != accurate["case_signature"]:
        raise ValueError("references use different toy cases")
    upper = np.triu_indices_from(old)
    excess = (old - new) / new
    unique_excess = excess[upper]
    stats = {
        "legacy_backend": legacy.get("inner_backend", "cuda_lbfgsb"),
        "accurate_backend": accurate.get("inner_backend"),
        "unique_pairs": len(unique_excess),
        "legacy_grid_best": legacy["grid_best"],
        "accurate_grid_best": accurate["grid_best"],
        "median_legacy_excess_percent": float(np.median(unique_excess) * 100),
        "p95_legacy_excess_percent": float(np.percentile(unique_excess, 95) * 100),
        "max_legacy_excess_percent": float(np.max(unique_excess) * 100),
        "pairs_legacy_more_than_1_percent_high": int(np.sum(unique_excess > 0.01)),
        "pairs_legacy_more_than_5_percent_high": int(np.sum(unique_excess > 0.05)),
        "pairs_legacy_lower_by_more_than_0_1_percent": int(
            np.sum(unique_excess < -0.001)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True,
                             layout="constrained")
    extent = (-91, 91, -91, 91)
    common = LogNorm(vmin=float(min(np.min(old), np.min(new))),
                     vmax=float(max(np.max(old), np.max(new))))
    for axis, matrix, title in zip(axes[:2], (old, new),
                                   ("Legacy CUDA-callback inner solve",
                                    "Accurate influence-matrix inner solve")):
        image = axis.imshow(matrix, origin="lower", extent=extent,
                            norm=common, cmap="viridis", interpolation="nearest")
        axis.set_title(title)
    fig.colorbar(image, ax=axes[:2], label="Re-optimized toy loss (log color)",
                 fraction=0.025)
    difference = axes[2].imshow(
        excess * 100, origin="lower", extent=extent,
        norm=TwoSlopeNorm(vmin=-5, vcenter=0, vmax=20),
        cmap="coolwarm", interpolation="nearest")
    axes[2].set_title("Legacy loss excess over accurate (%)")
    fig.colorbar(difference, ax=axes[2], label="Percent; clipped to [-5, 20]")
    for axis in axes:
        axis.set(xlabel="Angle 1 (degrees)", xlim=(-90, 90), ylim=(-90, 90))
    axes[0].set_ylabel("Angle 2 (degrees)")
    plot_path = os.path.join(args.accurate_dir, "inner_solver_discrepancy.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    summary_path = os.path.join(args.accurate_dir, "inner_solver_discrepancy.json")
    write_json_atomic(summary_path, stats)
    print(json.dumps(stats, indent=2))
    print(f"Plot: {plot_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
