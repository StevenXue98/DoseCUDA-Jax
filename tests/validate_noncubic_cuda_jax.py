"""CUDA/JAX regression for axis swaps hidden by the historical 138-cube test.

This validation is entirely in memory and never modifies golden output files.
"""

import os
import sys

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from DoseCUDA import IMPTBeam, IMPTDoseGrid, IMPTPlan  # noqa: E402
from impt_jax import computeIMPTPlanJax  # noqa: E402


ARRAY_SHAPE_ZYX = (48, 60, 72)
EXPECTED_ORIGIN_XYZ = (-108.0, -90.0, -72.0)


def create_asymmetric_plan():
    plan = IMPTPlan()
    beam = IMPTBeam()
    beam.dicom_rangeshifter_label = "0"
    beam.iso = np.array((7.0, -11.0, 13.0), dtype=np.float32)
    beam.addSingleSpot(24.0, -15.0, 1.0, 45)
    plan.addBeam(beam)
    return plan


def create_grid():
    grid = IMPTDoseGrid()
    grid.createCubePhantom(size=ARRAY_SHAPE_ZYX)
    if grid.HU.shape != ARRAY_SHAPE_ZYX:
        raise AssertionError(f"HU shape is {grid.HU.shape}, expected {ARRAY_SHAPE_ZYX}")
    np.testing.assert_array_equal(grid.origin, EXPECTED_ORIGIN_XYZ)
    return grid


def main():
    plan = create_asymmetric_plan()

    cuda_grid = create_grid()
    cuda_grid.computeIMPTPlan(plan)
    cuda_dose = np.asarray(cuda_grid.dose, dtype=np.float32)

    jax_grid = create_grid()
    jax_dose = np.asarray(computeIMPTPlanJax(jax_grid, plan), dtype=np.float32)

    if cuda_dose.shape != ARRAY_SHAPE_ZYX or jax_dose.shape != ARRAY_SHAPE_ZYX:
        raise SystemExit(
            f"shape mismatch: CUDA={cuda_dose.shape}, JAX={jax_dose.shape}, "
            f"expected={ARRAY_SHAPE_ZYX}"
        )

    absolute_difference = np.abs(cuda_dose - jax_dose)
    max_percentage = float(100.0 * absolute_difference.max() / cuda_dose.max())
    correlation = float(np.corrcoef(cuda_dose.ravel(), jax_dose.ravel())[0, 1])
    cuda_peak = np.unravel_index(cuda_dose.argmax(), cuda_dose.shape)
    jax_peak = np.unravel_index(jax_dose.argmax(), jax_dose.shape)

    print(f"array shape (z,y,x): {cuda_dose.shape}")
    print(f"physical origin (x,y,z): {tuple(cuda_grid.origin)}")
    print(f"peak voxel CUDA/JAX: {cuda_peak} / {jax_peak}")
    print(f"max abs difference: {absolute_difference.max():.9g}")
    print(f"max difference as percent of CUDA peak: {max_percentage:.9g}%")
    print(f"correlation: {correlation:.12g}")

    if cuda_peak != jax_peak:
        raise SystemExit("CUDA and JAX peak voxels do not match")
    if max_percentage > 0.3 or correlation < 0.999999:
        raise SystemExit("CUDA/JAX agreement is outside the established baseline")

    print("Non-cubic indexing validation passed.")


if __name__ == "__main__":
    main()
