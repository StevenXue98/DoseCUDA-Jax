"""Render a fresh non-cubic ring case without touching reference outputs."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from DoseCUDA import IMPTBeam, IMPTDoseGrid, IMPTPlan  # noqa: E402
from impt_jax import computeIMPTPlanJax  # noqa: E402


SHAPE_ZYX = (96, 112, 128)


def create_ring_plan():
    plan = IMPTPlan()
    beam = IMPTBeam()
    beam.dicom_rangeshifter_label = "0"

    number_of_spots = 98
    for energy_id in range(number_of_spots):
        theta = 2.0 * np.pi * energy_id / number_of_spots
        beam.addSingleSpot(
            100.0 * np.cos(theta),
            100.0 * np.sin(theta),
            0.2,
            energy_id,
        )

    plan.addBeam(beam)
    return plan


def write_volume(array, grid, path):
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetOrigin(np.asarray(grid.origin, dtype=float).tolist())
    image.SetSpacing(np.asarray(grid.spacing, dtype=float).tolist())
    sitk.WriteImage(image, path)


def plane(volume, axis, index):
    return np.take(volume, index, axis=axis)


def plane_geometry(grid, axis):
    nz, ny, nx = grid.HU.shape
    x0, y0, z0 = grid.origin
    sx, sy, sz = grid.spacing
    if axis == 0:
        return (x0, x0 + (nx - 1) * sx, y0, y0 + (ny - 1) * sy), "x (mm)", "y (mm)", "z"
    if axis == 1:
        return (x0, x0 + (nx - 1) * sx, z0, z0 + (nz - 1) * sz), "x (mm)", "z (mm)", "y"
    return (y0, y0 + (ny - 1) * sy, z0, z0 + (nz - 1) * sz), "y (mm)", "z (mm)", "x"


def render_orthogonal_comparison(ct, cuda, jax, grid, output_path):
    peak_index = np.unravel_index(cuda.argmax(), cuda.shape)
    difference = jax - cuda
    dose_maximum = max(float(cuda.max()), float(jax.max()))
    difference_limit = float(np.max(np.abs(difference)))

    figure, axes = plt.subplots(3, 4, figsize=(18, 14), constrained_layout=True)
    column_titles = ("CT", "CUDA dose", "JAX dose", "JAX - CUDA")

    for axis, index in enumerate(peak_index):
        extent, xlabel, ylabel, axis_name = plane_geometry(grid, axis)
        images = (
            axes[axis, 0].imshow(
                plane(ct, axis, index), cmap="gray", vmin=-1000.0, vmax=200.0,
                origin="lower", extent=extent, aspect="equal"
            ),
            axes[axis, 1].imshow(
                plane(cuda, axis, index), cmap="turbo", vmin=0.0, vmax=dose_maximum,
                origin="lower", extent=extent, aspect="equal"
            ),
            axes[axis, 2].imshow(
                plane(jax, axis, index), cmap="turbo", vmin=0.0, vmax=dose_maximum,
                origin="lower", extent=extent, aspect="equal"
            ),
            axes[axis, 3].imshow(
                plane(difference, axis, index), cmap="RdBu_r",
                vmin=-difference_limit, vmax=difference_limit,
                origin="lower", extent=extent, aspect="equal"
            ),
        )

        for column, (plot_axis, image) in enumerate(zip(axes[axis], images)):
            if axis == 0:
                plot_axis.set_title(column_titles[column])
            plot_axis.set_xlabel(xlabel)
            plot_axis.set_ylabel(ylabel)
            figure.colorbar(image, ax=plot_axis, fraction=0.046, pad=0.04)

        physical_coordinate = grid.origin[2 - axis] + index * grid.spacing[2 - axis]
        axes[axis, 0].text(
            0.02,
            0.96,
            f"{axis_name} index {index}\n{physical_coordinate:.1f} mm",
            transform=axes[axis, 0].transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
        )

    figure.suptitle(
        f"Non-cubic phantom {SHAPE_ZYX} (z, y, x); orthogonal planes through CUDA peak {peak_index}",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join("test_phantom_output", "noncubic_indexing"),
    )
    arguments = parser.parse_args()
    output_directory = os.path.abspath(arguments.output_dir)
    os.makedirs(output_directory, exist_ok=True)

    plan = create_ring_plan()

    cuda_grid = IMPTDoseGrid()
    cuda_grid.createCubePhantom(size=SHAPE_ZYX)
    cuda_grid.computeIMPTPlan(plan)
    cuda = np.asarray(cuda_grid.dose, dtype=np.float32)

    jax_grid = IMPTDoseGrid()
    jax_grid.createCubePhantom(size=SHAPE_ZYX)
    jax = np.asarray(computeIMPTPlanJax(jax_grid, plan), dtype=np.float32)

    ct_path = os.path.join(output_directory, "noncubic_phantom_ct.nrrd")
    cuda_path = os.path.join(output_directory, "noncubic_impt_dose_cuda.nrrd")
    jax_path = os.path.join(output_directory, "noncubic_impt_dose_jax.nrrd")
    image_path = os.path.join(output_directory, "noncubic_cuda_jax_orthogonal.png")

    write_volume(cuda_grid.HU, cuda_grid, ct_path)
    write_volume(cuda, cuda_grid, cuda_path)
    write_volume(jax, jax_grid, jax_path)
    render_orthogonal_comparison(cuda_grid.HU, cuda, jax, cuda_grid, image_path)

    difference = np.abs(cuda - jax)
    correlation = float(np.corrcoef(cuda.ravel(), jax.ravel())[0, 1])
    print(f"shape (z,y,x): {cuda.shape}")
    print(f"origin (x,y,z): {tuple(cuda_grid.origin)}")
    print(f"CUDA/JAX correlation: {correlation:.12g}")
    print(f"max difference as percent of CUDA peak: {100.0 * difference.max() / cuda.max():.9g}%")
    print(f"CT NRRD:        {ct_path}")
    print(f"CUDA NRRD:      {cuda_path}")
    print(f"JAX NRRD:       {jax_path}")
    print(f"comparison PNG: {image_path}")


if __name__ == "__main__":
    main()
