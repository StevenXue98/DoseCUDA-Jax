"""Compare forward WET and dose from rounded and differentiable smoothers.

This is a characterization diagnostic, not a CUDA parity test.  The rounded
JAX model is the reference because that is the path already validated against
the original CUDA implementation.
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from diagnose_angle_gradient_matrix import ANGLE_CASES, create_densities  # noqa: E402
from diagnose_rounded_wet_landscape import LANDSCAPE_CASES  # noqa: E402
from impt_jax import (  # noqa: E402
    _precompute_all_grids,
    beam_params_from_angles,
    compute_dose,
    compute_raytrace,
    compute_raytrace_differentiable,
)
from validate_spot_weight_gradients import prepare_fixed_data  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition-width-mm",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50),
        help="One or more sigmoid transition widths to compare.",
    )
    return parser.parse_args()


def metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    difference = candidate - reference
    absolute = np.abs(difference)
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-12)
    reference_peak = max(float(np.max(np.abs(reference))), 1.0e-12)
    return {
        "relative_l2_pct": 100.0 * float(np.linalg.norm(difference)) / reference_norm,
        "peak_normalized_max_pct": 100.0 * float(np.max(absolute)) / reference_peak,
        "peak_normalized_p99_pct": 100.0 * float(np.percentile(absolute, 99.0)) / reference_peak,
        "mean_absolute": float(np.mean(absolute)),
        "correlation": float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]),
    }


def main():
    args = parse_args()
    if any(width <= 0.0 for width in args.transition_width_mm):
        raise SystemExit("All transition widths must be positive")

    (
        baseline_beam_params,
        dose_params,
        lut_data,
        spot_data,
        layer_data,
        _baseline_grids,
        _baseline_wet,
        _probe,
    ) = prepare_fixed_data()
    adjusted_iso = jnp.asarray(
        (
            baseline_beam_params.iso_x,
            baseline_beam_params.iso_y,
            baseline_beam_params.iso_z,
        ),
        dtype=jnp.float32,
    )

    def geometry(angles):
        beam_params = beam_params_from_angles(
            angles[0],
            angles[1],
            adjusted_iso,
            baseline_beam_params.model_vsadx,
            baseline_beam_params.model_vsady,
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
        return beam_params, grids

    def rounded_model(angles, density):
        beam_params, grids = geometry(angles)
        wet = compute_raytrace(beam_params, dose_params, density, grids)
        dose = compute_dose(
            beam_params,
            dose_params,
            lut_data,
            spot_data,
            layer_data,
            wet,
            grids,
        )
        return wet, dose

    def differentiable_model(angles, density, width):
        beam_params, grids = geometry(angles)
        wet = compute_raytrace_differentiable(
            beam_params,
            dose_params,
            density,
            grids,
            transition_width_mm=width,
        )
        dose = compute_dose(
            beam_params,
            dose_params,
            lut_data,
            spot_data,
            layer_data,
            wet,
            grids,
        )
        return wet, dose

    rounded_model = jax.jit(rounded_model)
    differentiable_model = jax.jit(differentiable_model)
    densities = dict(create_densities())
    angles = dict(ANGLE_CASES)

    print(f"JAX device: {jax.devices()[0]}")
    print("reference: CUDA-compatible rounded JAX smoother")
    print(
        "width  case                                "
        "WET L2%  WET MAE(cm)  dose L2%  dose max%  dose p99%  corr"
    )
    print("-" * 116)

    aggregate = {float(width): [] for width in args.transition_width_mm}
    for material_name, angle_case_name in LANDSCAPE_CASES:
        case_name = f"{material_name}/{angle_case_name}"
        density = densities[material_name]
        case_angles = jnp.asarray(angles[angle_case_name], dtype=jnp.float32)
        rounded_wet, rounded_dose = rounded_model(case_angles, density)

        for width in args.transition_width_mm:
            smooth_wet, smooth_dose = differentiable_model(
                case_angles, density, jnp.asarray(width, dtype=jnp.float32)
            )
            wet_metrics = metrics(rounded_wet, smooth_wet)
            dose_metrics = metrics(rounded_dose, smooth_dose)
            aggregate[float(width)].append((wet_metrics, dose_metrics))
            print(
                f"{width:5.2f}  {case_name:35} "
                f"{wet_metrics['relative_l2_pct']:7.3f}  "
                f"{wet_metrics['mean_absolute']:11.6f}  "
                f"{dose_metrics['relative_l2_pct']:8.3f}  "
                f"{dose_metrics['peak_normalized_max_pct']:9.3f}  "
                f"{dose_metrics['peak_normalized_p99_pct']:9.3f}  "
                f"{dose_metrics['correlation']:.7f}"
            )

    print("\nWorst case across the four material/angle cases:")
    for width, rows in aggregate.items():
        print(
            f"  {width:g} mm: "
            f"WET L2={max(row[0]['relative_l2_pct'] for row in rows):.3f}%, "
            f"dose L2={max(row[1]['relative_l2_pct'] for row in rows):.3f}%, "
            f"dose max={max(row[1]['peak_normalized_max_pct'] for row in rows):.3f}%, "
            f"min corr={min(row[1]['correlation'] for row in rows):.7f}"
        )


if __name__ == "__main__":
    main()
