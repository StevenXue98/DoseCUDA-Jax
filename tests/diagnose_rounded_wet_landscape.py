"""Map local BAO loss landscapes produced by rounded WET smoothing.

The target/OAR masks, prescription, spot parameters, and material volume are
fixed within each scan.  Only one beam angle varies.  Dense forward samples
are compared with sparse reverse-mode gradients and are also used to test
whether finite steps in the negative-gradient direction actually reduce the
recomputed loss.
"""

import argparse
import csv
import os
import sys

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "DoseCUDA", "Jax"))

from diagnose_angle_gradient_matrix import (  # noqa: E402
    ANGLE_CASES,
    create_densities,
)
from impt_jax import (  # noqa: E402
    _precompute_all_grids,
    beam_params_from_angles,
    compute_dose,
    compute_raytrace,
)
from validate_spot_weight_gradients import (  # noqa: E402
    SHAPE_ZYX,
    prepare_fixed_data,
)


LANDSCAPE_CASES = (
    ("homogeneous", "baseline_oblique"),
    ("homogeneous", "far_oblique"),
    ("heterogeneous", "near_lateral"),
    ("heterogeneous", "far_oblique"),
)
ANGLE_NAMES = ("gantry", "couch")
SCAN_HALF_WIDTH_DEG = 0.5
SCAN_STEP_DEG = 0.01
GRADIENT_SPACING_DEG = 0.05
FINITE_DIFFERENCE_HALF_STEP_DEG = 0.05
DESCENT_STEPS_DEG = (0.01, 0.05, 0.10, 0.25)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            REPOSITORY_ROOT, "test_phantom_output", "angular_landscape"
        ),
        help="Directory for diagnostic CSV and PNG files.",
    )
    return parser.parse_args()


def build_masks(dose_params):
    z, y, x = np.ogrid[
        :dose_params.ni, :dose_params.nj, :dose_params.nk
    ]
    target = (
        (z - 0.58 * dose_params.ni) ** 2
        + (y - 0.44 * dose_params.nj) ** 2
        + (x - 0.61 * dose_params.nk) ** 2
    ) <= 4.5 ** 2
    oar = (
        (z - 0.38 * dose_params.ni) ** 2
        + (y - 0.68 * dose_params.nj) ** 2
        + (x - 0.35 * dose_params.nk) ** 2
    ) <= 4.0 ** 2
    return jnp.asarray(target), jnp.asarray(oar)


def scan_metrics(losses, gradient_indices, autodiff, finite_difference):
    difference = autodiff - finite_difference
    relative_l2 = float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(autodiff), 1.0e-12)
    )
    gradient_floor = 0.05 * max(float(np.max(np.abs(autodiff))), 1.0e-12)
    scaled_error = np.abs(difference) / np.maximum(
        np.abs(autodiff), gradient_floor
    )
    active = (np.abs(autodiff) > gradient_floor) & (
        np.abs(finite_difference) > gradient_floor
    )
    sign_mismatches = int(
        np.count_nonzero(active & (np.sign(autodiff) != np.sign(finite_difference)))
    )

    adjacent_slopes = np.diff(losses) / SCAN_STEP_DEG
    slope_jumps = np.diff(adjacent_slopes)
    slope_scale = max(
        float(np.max(np.abs(autodiff))),
        float(np.max(np.abs(finite_difference))),
        1.0e-12,
    )

    descent = {}
    max_gradient = max(float(np.max(np.abs(autodiff))), 1.0e-12)
    for step in DESCENT_STEPS_DEG:
        offset = int(round(step / SCAN_STEP_DEG))
        attempts = 0
        decreases = 0
        worst_increase = 0.0
        model_errors = []
        for index, gradient in zip(gradient_indices, autodiff):
            if abs(gradient) < 0.01 * max_gradient:
                continue
            direction = -1 if gradient > 0.0 else 1
            destination = index + direction * offset
            if destination < 0 or destination >= losses.size:
                continue
            attempts += 1
            actual_change = float(losses[destination] - losses[index])
            predicted_change = -abs(float(gradient)) * step
            if actual_change < 0.0:
                decreases += 1
            worst_increase = max(worst_increase, actual_change)
            model_errors.append(
                abs(actual_change - predicted_change)
                / max(abs(actual_change), abs(predicted_change), 1.0e-12)
            )
        descent[step] = {
            "attempts": attempts,
            "decreases": decreases,
            "worst_increase": worst_increase,
            "median_model_error": (
                float(np.median(model_errors)) if model_errors else 0.0
            ),
        }

    return {
        "relative_l2": relative_l2,
        "max_scaled_error": float(np.max(scaled_error)),
        "sign_mismatches": sign_mismatches,
        "active_gradients": int(np.count_nonzero(active)),
        "max_normalized_slope_jump": (
            float(np.max(np.abs(slope_jumps))) / slope_scale
        ),
        "total_variation": float(np.sum(np.abs(np.diff(losses)))),
        "descent": descent,
    }


def write_outputs(
    output_dir,
    case_name,
    angle_name,
    varied_angles,
    losses,
    gradient_indices,
    autodiff,
    finite_difference,
):
    stem = f"{case_name.replace('/', '_')}_{angle_name}"
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    png_path = os.path.join(output_dir, f"{stem}.png")

    autodiff_dense = np.full(losses.shape, np.nan, dtype=np.float64)
    finite_difference_dense = np.full(losses.shape, np.nan, dtype=np.float64)
    autodiff_dense[gradient_indices] = autodiff
    finite_difference_dense[gradient_indices] = finite_difference
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("angle_degrees", "loss", "autodiff", "finite_difference")
        )
        writer.writerows(
            zip(varied_angles, losses, autodiff_dense, finite_difference_dense)
        )

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(varied_angles, losses, linewidth=1.2)
    axes[0].set_ylabel("BAO-style loss")
    axes[0].set_title(f"Rounded WET: {case_name}, varying {angle_name}")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        varied_angles[gradient_indices],
        finite_difference,
        "o-",
        markersize=3,
        label=f"central difference (h={FINITE_DIFFERENCE_HALF_STEP_DEG:g}°)",
    )
    axes[1].plot(
        varied_angles[gradient_indices],
        autodiff,
        "x-",
        markersize=4,
        label="autodiff",
    )
    axes[1].set_xlabel(f"{angle_name.capitalize()} angle (degrees)")
    axes[1].set_ylabel("d loss / d angle")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return csv_path, png_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
    target_mask, oar_mask = build_masks(dose_params)

    def dose_from_angles(angles, density):
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
        wet = compute_raytrace(beam_params, dose_params, density, grids)
        return compute_dose(
            beam_params,
            dose_params,
            lut_data,
            spot_data,
            layer_data,
            wet,
            grids,
        )

    def loss(angles, density, target_rx):
        dose = dose_from_angles(angles, density)
        target_square = jnp.mean((dose[target_mask] - target_rx) ** 2)
        oar_square = jnp.mean(dose[oar_mask] ** 2)
        return target_square + 10.0 * oar_square

    compiled_dose = jax.jit(dose_from_angles)
    compiled_value_and_gradient = jax.jit(
        jax.value_and_grad(loss, argnums=0)
    )

    @jax.jit
    def loss_batch(angle_batch, density, target_rx):
        return jax.lax.map(
            lambda angles: loss(angles, density, target_rx), angle_batch
        )

    @jax.jit
    def value_and_gradient_batch(angle_batch, density, target_rx):
        return jax.lax.map(
            lambda angles: compiled_value_and_gradient(
                angles, density, target_rx
            ),
            angle_batch,
        )

    densities = dict(create_densities())
    angle_cases = dict(ANGLE_CASES)
    offsets = np.arange(
        -SCAN_HALF_WIDTH_DEG,
        SCAN_HALF_WIDTH_DEG + 0.5 * SCAN_STEP_DEG,
        SCAN_STEP_DEG,
        dtype=np.float32,
    )
    gradient_stride = int(round(GRADIENT_SPACING_DEG / SCAN_STEP_DEG))
    finite_difference_radius = int(
        round(FINITE_DIFFERENCE_HALF_STEP_DEG / SCAN_STEP_DEG)
    )
    gradient_indices = np.arange(
        finite_difference_radius,
        offsets.size - finite_difference_radius,
        gradient_stride,
        dtype=np.int32,
    )
    all_results = []

    print(f"JAX device: {jax.devices()[0]}")
    print(f"shape (z,y,x): {SHAPE_ZYX}")
    print(
        f"scan: +/-{SCAN_HALF_WIDTH_DEG:g} deg at {SCAN_STEP_DEG:g} deg; "
        f"autodiff every {GRADIENT_SPACING_DEG:g} deg"
    )

    for material_name, angle_case_name in LANDSCAPE_CASES:
        density = densities[material_name]
        center = np.asarray(angle_cases[angle_case_name], dtype=np.float32)
        baseline_dose = np.asarray(compiled_dose(jnp.asarray(center), density))
        target_rx = np.float32(0.8 * np.max(baseline_dose[np.asarray(target_mask)]))
        case_name = f"{material_name}/{angle_case_name}"

        for angle_index, angle_name in enumerate(ANGLE_NAMES):
            angle_batch = np.repeat(center[np.newaxis, :], offsets.size, axis=0)
            angle_batch[:, angle_index] += offsets
            varied_angles = angle_batch[:, angle_index].astype(np.float64)
            losses = np.asarray(
                loss_batch(jnp.asarray(angle_batch), density, target_rx),
                dtype=np.float64,
            )
            gradient_batch = jnp.asarray(angle_batch[gradient_indices])
            _values, gradients = value_and_gradient_batch(
                gradient_batch, density, target_rx
            )
            autodiff = np.asarray(gradients, dtype=np.float64)[:, angle_index]
            finite_difference = (
                losses[gradient_indices + finite_difference_radius]
                - losses[gradient_indices - finite_difference_radius]
            ) / (2.0 * FINITE_DIFFERENCE_HALF_STEP_DEG)
            metrics = scan_metrics(
                losses, gradient_indices, autodiff, finite_difference
            )
            csv_path, png_path = write_outputs(
                args.output_dir,
                case_name,
                angle_name,
                varied_angles,
                losses,
                gradient_indices,
                autodiff,
                finite_difference,
            )
            all_results.append((case_name, angle_name, metrics))

            print(
                f"{case_name:35} {angle_name:7}: "
                f"grad L2={100.0 * metrics['relative_l2']:.3f}% "
                f"max-scaled={100.0 * metrics['max_scaled_error']:.3f}% "
                f"sign={metrics['sign_mismatches']}/{metrics['active_gradients']} "
                f"slope-jump={metrics['max_normalized_slope_jump']:.3f}"
            )
            for step, descent in metrics["descent"].items():
                print(
                    f"  descent {step:g} deg: {descent['decreases']}/"
                    f"{descent['attempts']} decreased; "
                    f"worst increase={descent['worst_increase']:.6g}; "
                    f"median model error={100.0 * descent['median_model_error']:.2f}%"
                )
            print(f"  outputs: {csv_path}, {png_path}")

    if any(
        not np.isfinite(metrics["relative_l2"])
        for _case, _angle, metrics in all_results
    ):
        raise SystemExit("Rounded-WET landscape diagnostic produced non-finite metrics")

    print("Rounded-WET angular landscape diagnostic completed.")


if __name__ == "__main__":
    main()
