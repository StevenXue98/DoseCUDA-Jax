#!/usr/bin/env python3
"""Render a saved two-beam toy BAO landscape and optional search paths in 3D.

The surface joins 10-degree (or other saved grid) samples for display only;
interpolated points are not additional DoseCUDA calculations. Path heights
use their independently evaluated, jointly re-optimized exact losses.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run_toy_bao_baseline import ROOT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=os.path.join(
        ROOT, "test_phantom_output", "bao_toy_two_beam"))
    parser.add_argument("--elev", type=float, default=27.0,
                        help="Camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=-60.0,
                        help="Camera azimuth in degrees")
    parser.add_argument("--output", default=None,
                        help="PNG path; defaults inside input-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.input_dir, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    grid_csv = os.path.join(args.input_dir, "angle_pairs.csv")
    if not os.path.exists(grid_csv):
        grid_csv = os.path.join(args.input_dir, "coarse_angle_pairs.csv")
    with open(grid_csv, newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    count = round(180.0 / summary["grid_step_deg"]) + 1
    angles = np.linspace(-90.0, 90.0, count)
    if len(rows) != count * (count + 1) // 2:
        raise ValueError("saved angle-pair grid is incomplete")
    positions = {round(float(angle), 6): index for index, angle in enumerate(angles)}
    landscape = np.full((count, count), np.nan)
    for row in rows:
        first = positions[round(float(row["angle_1_deg"]), 6)]
        second = positions[round(float(row["angle_2_deg"]), 6)]
        loss = float(row["loss"])
        if not np.isfinite(loss) or loss <= 0:
            raise ValueError("saved losses must be finite and positive")
        landscape[first, second] = landscape[second, first] = loss
    if not np.all(np.isfinite(landscape)):
        raise ValueError("saved angle-pair grid has missing cells")

    x, y = np.meshgrid(angles, angles, indexing="ij")
    z = np.log10(landscape)
    fig = plt.figure(figsize=(11, 9), layout="constrained")
    axis = fig.add_subplot(111, projection="3d", computed_zorder=False)
    surface = axis.plot_surface(
        x, y, z, cmap="viridis", alpha=0.57,
        edgecolor=(0.1, 0.1, 0.1, 0.12), linewidth=0.2,
        antialiased=True, shade=False, zorder=1,
    )
    fig.colorbar(surface, ax=axis, shrink=0.60, pad=0.10,
                 label="log10(jointly optimized loss)")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, run in enumerate(summary["runs"]):
        history = run["history"]
        path = np.asarray([point["angles_deg"] for point in history])
        heights = np.log10([point["loss"] for point in history])
        color = colors[index % len(colors)]
        axis.plot(path[:, 0], path[:, 1], heights, color=color,
                  linewidth=3.3, marker="o", markersize=5.5,
                  label=f"start {run['start_angles_deg']}", zorder=10)
        axis.scatter(path[0, 0], path[0, 1], heights[0], color=color,
                     marker="^", s=90, depthshade=False, zorder=11)
        axis.scatter(path[-1, 0], path[-1, 1], heights[-1], color=color,
                     marker="X", s=105, depthshade=False, zorder=11)

    best = summary["grid_best"]
    axis.scatter(*best["angles_deg"], np.log10(best["loss"]),
                 color="red", marker="*", s=180, edgecolor="white",
                 depthshade=False, label="Sampled-grid minimum", zorder=12)
    axis.set(xlim=(-90, 90), ylim=(-90, 90),
             xlabel="Gantry angle 1 (degrees)",
             ylabel="Gantry angle 2 (degrees)",
             zlabel="log10(loss)")
    axis.set_box_aspect((1, 1, 0.72))
    axis.view_init(elev=args.elev, azim=args.azim)
    axis.legend(loc="upper left", fontsize="small")
    if summary["runs"]:
        fig.suptitle("Two-beam toy BAO: sampled loss surface and exact search paths\n"
                     "Triangles = starts; X = final plans; surface between grid "
                     "points is visual interpolation")
    else:
        fig.suptitle("Two-beam toy BAO: sampled loss surface\n"
                     "Surface between grid points is visual interpolation, "
                     "not additional dose calculations")
    output = args.output or os.path.join(args.input_dir, "pair_landscape_3d.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"3D plot: {output}")


if __name__ == "__main__":
    main()
