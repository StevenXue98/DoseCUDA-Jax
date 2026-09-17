"""Read-only CUDA/JAX regression matrix for IMPT geometry and materials.

The cases are intentionally small enough for routine development runs. They
exercise conventions that a symmetric homogeneous cube cannot validate.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from DoseCUDA import IMPTBeam, IMPTDoseGrid, IMPTPlan  # noqa: E402
from impt_jax import (  # noqa: E402
    DoseParams,
    _extract_beam_params,
    _precompute_all_grids,
    computeIMPTPlanJax,
    compute_raytrace,
)


@dataclass(frozen=True)
class BeamCase:
    gantry: float
    couch: float
    isocenter_xyz: tuple
    spots: tuple


@dataclass(frozen=True)
class RegressionCase:
    name: str
    shape_zyx: tuple
    beams: tuple
    heterogeneous: bool = False
    fractions: int = 1


CASES = (
    RegressionCase(
        name="single_spot_z_short",
        shape_zyx=(48, 60, 72),
        beams=(BeamCase(0.0, 0.0, (7.0, -11.0, 13.0), ((24.0, -15.0, 1.0, 45),)),),
    ),
    RegressionCase(
        name="single_spot_x_short",
        shape_zyx=(72, 60, 48),
        beams=(BeamCase(0.0, 0.0, (-8.0, 6.0, -12.0), ((-18.0, 12.0, 0.8, 38),)),),
    ),
    RegressionCase(
        name="oblique_heterogeneous",
        shape_zyx=(48, 60, 72),
        beams=(
            BeamCase(
                37.0,
                13.0,
                (-8.0, 5.0, 10.0),
                ((-16.0, 11.0, 0.7, 28), (8.0, -14.0, 1.0, 45), (19.0, 6.0, 0.4, 62)),
            ),
        ),
        heterogeneous=True,
    ),
    RegressionCase(
        name="two_beam_fractionated",
        shape_zyx=(48, 60, 72),
        beams=(
            BeamCase(0.0, 0.0, (4.0, -6.0, 9.0), ((-12.0, 8.0, 0.6, 34), (15.0, -9.0, 0.9, 52))),
            BeamCase(91.0, -7.0, (4.0, -6.0, 9.0), ((7.0, 13.0, 0.5, 31), (-17.0, -5.0, 0.75, 57))),
        ),
        heterogeneous=True,
        fractions=3,
    ),
)


# Ray integration at a grazing volume boundary is sensitive to CUDA's
# approximate rnorm3df and hardware texture interpolation.  A few voxels can
# therefore differ by one 1 mm step even when the distributions agree.  Keep
# a generous hard-outlier guard, while using percentile and correlation checks
# to catch broad regressions.
WET_MAX_PERCENT = 3.5
WET_P99_PERCENT = 0.01
WET_MIN_CORRELATION = 0.999999


def create_grid(case):
    grid = IMPTDoseGrid()
    grid.createCubePhantom(size=case.shape_zyx)

    if case.heterogeneous:
        nz, ny, nx = case.shape_zyx
        edge = round(10.0 / float(grid.spacing[0]))
        grid.HU[nz // 4:nz // 4 + 4, edge:-edge, edge:-edge] = 850.0
        grid.HU[edge:-edge, ny // 2 - 3:ny // 2 + 3, edge:-edge] = -450.0
        grid.HU[edge:-edge, edge:-edge, 3 * nx // 4:3 * nx // 4 + 5] = 300.0

    return grid


def create_plan(case):
    plan = IMPTPlan()
    plan.n_fractions = case.fractions

    for beam_case in case.beams:
        beam = IMPTBeam()
        beam.dicom_rangeshifter_label = "0"
        beam.gantry_angle = beam_case.gantry
        beam.couch_angle = beam_case.couch
        beam.iso = np.asarray(beam_case.isocenter_xyz, dtype=np.float32)
        for spot in beam_case.spots:
            beam.addSingleSpot(*spot)
        plan.addBeam(beam)

    return plan


def comparison_metrics(cuda_dose, jax_dose):
    difference = np.abs(cuda_dose - jax_dose)
    peak = float(cuda_dose.max())
    normalized = 100.0 * difference / peak
    max_index = np.unravel_index(difference.argmax(), difference.shape)
    material_mask = cuda_dose > 0.1
    return {
        "cuda_peak": peak,
        "jax_peak": float(jax_dose.max()),
        "max_pct": float(normalized.max()),
        "p99_pct": float(np.percentile(normalized, 99.0)),
        "mean_pct": float(normalized.mean()),
        "correlation": float(np.corrcoef(cuda_dose.ravel(), jax_dose.ravel())[0, 1]),
        "cuda_peak_index": np.unravel_index(cuda_dose.argmax(), cuda_dose.shape),
        "jax_peak_index": np.unravel_index(jax_dose.argmax(), jax_dose.shape),
        "max_difference_index": max_index,
        "cuda_at_max_difference": float(cuda_dose[max_index]),
        "jax_at_max_difference": float(jax_dose[max_index]),
        "cuda_min": float(cuda_dose.min()),
        "jax_min": float(jax_dose.min()),
        "cuda_negative_count": int(np.count_nonzero(cuda_dose < 0.0)),
        "jax_negative_count": int(np.count_nonzero(jax_dose < 0.0)),
        "material_max_pct": (
            float(normalized[material_mask].max()) if np.any(material_mask) else 0.0
        ),
        "material_p99_pct": (
            float(np.percentile(normalized[material_mask], 99.0))
            if np.any(material_mask)
            else 0.0
        ),
    }


def compute_jax_wet(grid, plan, beam):
    model_index = list(plan.dicom_rangeshifter_label.astype(str)).index(
        beam.dicom_rangeshifter_label
    )
    beam_model = plan.beam_models[model_index]
    beam_params = _extract_beam_params(beam, beam_model, grid.origin)
    dose_params = DoseParams(
        ni=int(grid.size[0]),
        nj=int(grid.size[1]),
        nk=int(grid.size[2]),
        spacing=jnp.asarray(float(grid.spacing[0]), dtype=jnp.float32),
    )
    grids = _precompute_all_grids(
        dose_params.ni,
        dose_params.nj,
        dose_params.nk,
        dose_params.spacing,
        beam_params.iso_x,
        beam_params.iso_y,
        beam_params.iso_z,
        beam_params.src_x,
        beam_params.src_y,
        beam_params.src_z,
        beam_params.singa,
        beam_params.cosga,
        beam_params.sinta,
        beam_params.costa,
    )
    density = jnp.asarray(grid.RLSPFromHU(plan.machine_name), dtype=jnp.float32)
    return np.asarray(compute_raytrace(beam_params, dose_params, density, grids))


def run_case(case):
    plan = create_plan(case)

    cuda_grid = create_grid(case)
    cuda_wets = [wet.voxel_data for wet in cuda_grid.computeIMPTWET(plan)]
    cuda_grid.computeIMPTPlan(plan)
    cuda_dose = np.asarray(cuda_grid.dose, dtype=np.float32)

    jax_grid = create_grid(case)
    jax_wets = [compute_jax_wet(jax_grid, plan, beam) for beam in plan.beam_list]
    jax_dose = np.asarray(computeIMPTPlanJax(jax_grid, plan), dtype=np.float32)

    if cuda_dose.shape != case.shape_zyx or jax_dose.shape != case.shape_zyx:
        raise AssertionError(
            f"{case.name}: shape mismatch CUDA={cuda_dose.shape}, "
            f"JAX={jax_dose.shape}, expected={case.shape_zyx}"
        )
    if cuda_dose.max() <= 0.0 or jax_dose.max() <= 0.0:
        raise AssertionError(f"{case.name}: expected a nonzero dose distribution")

    return {
        "dose": comparison_metrics(cuda_dose, jax_dose),
        "wet": [
            comparison_metrics(np.asarray(cuda_wet), np.asarray(jax_wet))
            for cuda_wet, jax_wet in zip(cuda_wets, jax_wets)
        ],
    }


def main():
    failed = []
    print(
        "case                         shape          max%       p99%      mean%       corr          peaks"
    )
    print("-" * 112)

    for case in CASES:
        result = run_case(case)
        metrics = result["dose"]
        print(
            f"{case.name:28} {str(case.shape_zyx):14} "
            f"{metrics['max_pct']:9.5f} {metrics['p99_pct']:10.6f} "
            f"{metrics['mean_pct']:10.7f} {metrics['correlation']:.10f}  "
            f"{metrics['cuda_peak_index']} / {metrics['jax_peak_index']}"
        )

        if metrics["max_pct"] > 0.3:
            failed.append(f"{case.name}: max difference exceeds 0.3% of CUDA peak")
        if metrics["p99_pct"] > 0.01:
            failed.append(f"{case.name}: 99th-percentile difference exceeds 0.01%")
        if metrics["mean_pct"] > 0.001:
            failed.append(f"{case.name}: mean difference exceeds 0.001%")
        if metrics["correlation"] < 0.999999:
            failed.append(f"{case.name}: correlation is below 0.999999")
        if metrics["cuda_peak_index"] != metrics["jax_peak_index"]:
            failed.append(f"{case.name}: peak voxel indices differ")

        for beam_index, wet_metrics in enumerate(result["wet"]):
            if wet_metrics["max_pct"] > WET_MAX_PERCENT:
                failed.append(
                    f"{case.name} beam {beam_index}: WET max difference exceeds "
                    f"{WET_MAX_PERCENT}%"
                )
            if wet_metrics["p99_pct"] > WET_P99_PERCENT:
                failed.append(
                    f"{case.name} beam {beam_index}: WET 99th-percentile difference "
                    f"exceeds {WET_P99_PERCENT}%"
                )
            if wet_metrics["correlation"] < WET_MIN_CORRELATION:
                failed.append(
                    f"{case.name} beam {beam_index}: WET correlation is below "
                    f"{WET_MIN_CORRELATION}"
                )

        wet_max = max(metrics["max_pct"] for metrics in result["wet"])
        wet_p99 = max(metrics["p99_pct"] for metrics in result["wet"])
        wet_corr = min(metrics["correlation"] for metrics in result["wet"])
        print(
            f"  WET across {len(result['wet'])} beam(s): "
            f"max={wet_max:.6f}% p99={wet_p99:.6f}% corr={wet_corr:.10f}"
        )
        for beam_index, wet_metrics in enumerate(result["wet"]):
            print(
                f"    beam {beam_index}: max at {wet_metrics['max_difference_index']} "
                f"CUDA={wet_metrics['cuda_at_max_difference']:.7f} "
                f"JAX={wet_metrics['jax_at_max_difference']:.7f}; "
                f"min={wet_metrics['cuda_min']:.7f}/{wet_metrics['jax_min']:.7f}; "
                f"negative={wet_metrics['cuda_negative_count']}/"
                f"{wet_metrics['jax_negative_count']}; "
                f"WET>0.1 max={wet_metrics['material_max_pct']:.6f}% "
                f"p99={wet_metrics['material_p99_pct']:.6f}%"
            )

    if failed:
        raise SystemExit("Regression matrix failed:\n- " + "\n- ".join(failed))

    print("CUDA/JAX regression matrix passed.")


if __name__ == "__main__":
    main()
