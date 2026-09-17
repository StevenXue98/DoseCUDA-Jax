"""Validate native CUDA and JAX against the committed cube references.

This test performs all comparisons in memory and never overwrites the golden
NRRD files under test_phantom_output.
"""

import os
import sys
import time

import numpy as np
import SimpleITK as sitk


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAX_DIRECTORY = os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax")
sys.path.insert(0, JAX_DIRECTORY)

from DoseCUDA import IMPTBeam, IMPTDoseGrid, IMPTPlan  # noqa: E402
from impt_jax_fix import computeIMPTPlanJax  # noqa: E402


def create_test_plan():
    plan = IMPTPlan()
    beam = IMPTBeam()
    beam.dicom_rangeshifter_label = "0"

    number_of_spots = 98
    for energy_id in range(number_of_spots):
        theta = 2.0 * 3.14159 * energy_id / number_of_spots
        beam.addSingleSpot(
            100.0 * np.cos(theta),
            100.0 * np.sin(theta),
            0.2,
            energy_id,
        )

    plan.addBeam(beam)
    return plan


def load_reference(filename):
    path = os.path.join(REPOSITORY_ROOT, "test_phantom_output", filename)
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32, copy=False)


def comparison_metrics(actual, reference):
    absolute_difference = np.abs(actual - reference)
    reference_scale = float(np.max(np.abs(reference)))
    return {
        "max_abs": float(absolute_difference.max()),
        "max_pct": float(100.0 * absolute_difference.max() / reference_scale),
        "mean_abs": float(absolute_difference.mean()),
        "correlation": float(np.corrcoef(actual.ravel(), reference.ravel())[0, 1]),
    }


def print_metrics(label, actual, reference):
    metrics = comparison_metrics(actual, reference)
    print(
        f"{label}: shape={actual.shape}, max={actual.max():.9g}, "
        f"max_abs={metrics['max_abs']:.9g}, max_pct={metrics['max_pct']:.9g}, "
        f"mean_abs={metrics['mean_abs']:.9g}, "
        f"corr={metrics['correlation']:.12g}"
    )
    return metrics


def main():
    cuda_grid = IMPTDoseGrid()
    cuda_grid.createCubePhantom(size=[138, 138, 138])
    start = time.perf_counter()
    cuda_grid.computeIMPTPlan(create_test_plan())
    cuda_result = np.asarray(cuda_grid.dose, dtype=np.float32)
    print(f"CUDA seconds: {time.perf_counter() - start:.6f}")

    jax_grid = IMPTDoseGrid()
    jax_grid.createCubePhantom(size=[138, 138, 138])
    start = time.perf_counter()
    jax_result = np.asarray(
        computeIMPTPlanJax(jax_grid, create_test_plan()), dtype=np.float32
    )
    print(f"JAX first-call seconds: {time.perf_counter() - start:.6f}")

    cuda_history = print_metrics(
        "CUDA vs historical CUDA",
        cuda_result,
        load_reference("cube_impt_dose.nrrd"),
    )
    jax_history = print_metrics(
        "JAX vs historical JAX",
        jax_result,
        load_reference("cube_impt_dose_jax.nrrd"),
    )
    cuda_jax = print_metrics("new CUDA vs new JAX", cuda_result, jax_result)

    if cuda_history["max_abs"] > 2.0e-7:
        raise SystemExit("CUDA result does not reproduce the historical CUDA reference")
    if jax_history["max_abs"] > 2.0e-7:
        raise SystemExit("JAX result does not reproduce the historical JAX reference")
    if cuda_jax["max_pct"] > 0.3 or cuda_jax["correlation"] < 0.999999:
        raise SystemExit("CUDA/JAX agreement is outside the established baseline")

    print("Reference validation passed.")


if __name__ == "__main__":
    main()
