"""Recompute and render the cube reference case without touching golden files."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import SimpleITK as sitk


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAX_DIRECTORY = os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax")
sys.path.insert(0, JAX_DIRECTORY)

from DoseCUDA import IMPTDoseGrid  # noqa: E402
from impt_jax_fix import computeIMPTPlanJax  # noqa: E402
from validate_cuda_jax_reference import create_test_plan  # noqa: E402


def write_dose(array, dose_grid, path):
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetOrigin(tuple(dose_grid.origin))
    image.SetSpacing(tuple(dose_grid.spacing))
    sitk.WriteImage(image, path)


def render_comparison(cuda_path, jax_path, ct_path, output_path, slice_index):
    cuda_data, _ = nrrd.read(cuda_path)
    jax_data, _ = nrrd.read(jax_path)
    ct_data, _ = nrrd.read(ct_path)

    cuda_slice = cuda_data[:, slice_index, :]
    jax_slice = jax_data[:, slice_index, :]
    ct_slice = ct_data[:, slice_index, :]
    difference_slice = jax_slice - cuda_slice

    dose_maximum = max(float(cuda_data.max()), float(jax_data.max()))
    difference_maximum = float(np.max(np.abs(difference_slice)))

    figure, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    ct_plot = axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(f"Cube CT (slice {slice_index})")
    figure.colorbar(ct_plot, ax=axes[0], fraction=0.046)

    cuda_plot = axes[1].imshow(cuda_slice, cmap="jet", vmin=0.0, vmax=dose_maximum)
    axes[1].set_title(f"Fresh CUDA dose (slice {slice_index})")
    figure.colorbar(cuda_plot, ax=axes[1], fraction=0.046)

    jax_plot = axes[2].imshow(jax_slice, cmap="jet", vmin=0.0, vmax=dose_maximum)
    axes[2].set_title(f"Fresh JAX dose (slice {slice_index})")
    figure.colorbar(jax_plot, ax=axes[2], fraction=0.046)

    difference_plot = axes[3].imshow(
        difference_slice,
        cmap="RdBu_r",
        vmin=-difference_maximum,
        vmax=difference_maximum,
    )
    axes[3].set_title("JAX - CUDA")
    figure.colorbar(difference_plot, ax=axes[3], fraction=0.046)

    for axis in axes:
        axis.invert_yaxis()
        axis.set_xlabel("voxel")
        axis.set_ylabel("voxel")

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=int, default=15, dest="slice_index")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("test_phantom_output", "native_gtx1080ti"),
    )
    arguments = parser.parse_args()

    output_directory = os.path.abspath(arguments.output_dir)
    os.makedirs(output_directory, exist_ok=True)

    cuda_grid = IMPTDoseGrid()
    cuda_grid.createCubePhantom(size=[138, 138, 138])
    cuda_grid.computeIMPTPlan(create_test_plan())
    cuda_result = np.asarray(cuda_grid.dose, dtype=np.float32)

    jax_grid = IMPTDoseGrid()
    jax_grid.createCubePhantom(size=[138, 138, 138])
    jax_result = np.asarray(
        computeIMPTPlanJax(jax_grid, create_test_plan()), dtype=np.float32
    )

    cuda_path = os.path.join(output_directory, "cube_impt_dose_cuda.nrrd")
    jax_path = os.path.join(output_directory, "cube_impt_dose_jax.nrrd")
    ct_path = os.path.join(output_directory, "cube_phantom_ct.nrrd")
    image_path = os.path.join(
        output_directory, f"cube_cuda_jax_slice_{arguments.slice_index:03d}.png"
    )

    write_dose(cuda_result, cuda_grid, cuda_path)
    write_dose(jax_result, jax_grid, jax_path)
    cuda_grid.writeCTNRRD(ct_path)
    render_comparison(
        cuda_path,
        jax_path,
        ct_path,
        image_path,
        arguments.slice_index,
    )

    print(f"Fresh CUDA NRRD: {cuda_path}")
    print(f"Fresh JAX NRRD:  {jax_path}")
    print(f"Fresh CT NRRD:   {ct_path}")
    print(f"Rendered PNG:    {image_path}")


if __name__ == "__main__":
    main()
