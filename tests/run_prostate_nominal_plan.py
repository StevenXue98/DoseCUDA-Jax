#!/usr/bin/env python3
"""Research-only nominal two-field prostate plan on the 5 mm matRad CT.

This intentionally keeps the original DoseCUDA forward kernels unchanged.
It is not a clinical plan, dose calibration, or proof of robust BAO.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from DoseCUDA import IMPTBeam, IMPTPlan
from DoseCUDA.gpu_influence_matrix import GPUInfluenceMatrixObjective
from DoseCUDA.impt_weight_optimization import (
    FixedGeometryPlanDose, InfluenceMatrixPlanDose, bounded_slsqp,
)
from utils.matrad_converter import MatRadData, create_dosecuda_objects


ROOT = Path(__file__).resolve().parents[1]
ANGLES = (90.0, 270.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing-mm", type=float, default=8.0)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--max-matrix-gib", type=float, default=4.0)
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--max-solve-seconds", type=float, default=300.0)
    parser.add_argument("--solver", choices=("slsqp", "lbfgsb"), default="lbfgsb")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "test_phantom_output/bao_prostate_nominal/summary.json")
    args = parser.parse_args()
    if (args.spacing_mm <= 0 or args.layers < 2 or args.max_matrix_gib <= 0 or
            args.max_iterations < 1 or args.max_solve_seconds <= 0):
        parser.error("invalid run limits")
    return args


def target_points(grid, target, iso, angle, model):
    """Return target voxel positions projected to the BEV isocenter plane."""
    z, y, x = np.nonzero(target)
    xyz = np.column_stack((
        grid.origin[0] + x * grid.spacing[0],
        grid.origin[1] + y * grid.spacing[1],
        grid.origin[2] + z * grid.spacing[2],
    )) - np.asarray(iso)
    theta = np.deg2rad(angle)
    head_x = -xyz[:, 0] * np.cos(theta) - xyz[:, 1] * np.sin(theta)
    head_z = -xyz[:, 0] * np.sin(theta) + xyz[:, 1] * np.cos(theta)
    return np.column_stack((
        head_x / (1.0 - head_z / model.VSADX),
        xyz[:, 2] / (1.0 - head_z / model.VSADY),
    ))


def make_beam(angle, iso):
    beam = IMPTBeam()
    beam.gantry_angle = angle
    beam.couch_angle = 0.0
    beam.iso = np.asarray(iso, dtype=np.float32)
    beam.dicom_rangeshifter_label = "0"
    return beam


def place_spots(beam, bev, wet_mm, model, spacing_mm, layers):
    """Choose layers from target WET and retain lateral spots near its BEV."""
    r80 = model.divergence_params[:, 1]
    low, high = np.percentile(wet_mm, [2, 98])
    eligible = np.flatnonzero((r80 >= low - 10.0) & (r80 <= high + 10.0))
    if eligible.size < layers:
        raise ValueError("too few machine energy layers span the target WET")
    energy_ids = np.unique(eligible[np.linspace(0, eligible.size - 1,
                                                layers).round().astype(int)])
    layer_info = []
    for energy_id in energy_ids:
        depth = float(r80[energy_id])
        points = bev[np.abs(wet_mm - depth) <= 18.0]
        if len(points) == 0:
            continue
        start = np.floor((points.min(axis=0) - spacing_mm / 2) / spacing_mm) * spacing_mm
        stop = np.ceil((points.max(axis=0) + spacing_mm / 2) / spacing_mm) * spacing_mm
        xx = np.arange(start[0], stop[0] + spacing_mm / 2, spacing_mm)
        yy = np.arange(start[1], stop[1] + spacing_mm / 2, spacing_mm)
        candidate = np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)
        distance, _ = cKDTree(points).query(candidate)
        selected = candidate[distance <= 0.75 * spacing_mm]
        for sx, sy in selected:
            beam.addSingleSpot(float(sx), float(sy), 0.02, int(energy_id))
        layer_info.append({"energy_id": int(energy_id), "r80_mm": depth,
                           "spots": len(selected)})
    if beam.n_spots == 0:
        raise ValueError("no candidate spots cover the target projection")
    return layer_info


def make_loss(regions):
    """Relative mean-squared target deviations and one-sided OAR excesses."""
    prepared = []
    shape = None
    for name, mask, prescribed, penalty, one_sided in regions:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask) or (shape is not None and mask.shape != shape):
            raise ValueError(f"invalid {name} mask")
        shape = mask.shape
        prepared.append((name, mask, float(prescribed), float(penalty),
                         bool(one_sided), int(mask.sum())))

    def loss(dose):
        dose = np.asarray(dose, dtype=np.float64)
        if dose.shape != shape or not np.all(np.isfinite(dose)):
            raise ValueError("dose must be finite and match the structure grid")
        adjoint = np.zeros(shape, dtype=np.float64)
        value = 0.0
        for _, mask, threshold, penalty, one_sided, count in prepared:
            error = dose[mask] - threshold
            if one_sided:
                error = np.maximum(error, 0.0)
            value += penalty * float(np.dot(error, error)) / count
            adjoint[mask] += 2.0 * penalty * error / count
        return value, adjoint

    return loss


def metrics(dose, masks):
    result = {}
    for name, mask in masks.items():
        values = dose[mask]
        result[name] = {"mean": float(np.mean(values)),
                        "d95": float(np.percentile(values, 5)),
                        "d5": float(np.percentile(values, 95)),
                        "max": float(np.max(values))}
    return result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def solve_scaled_lbfgsb(callback, initial, max_iterations):
    """Nonnegative L-BFGS-B with the same explicit scales as SLSQP."""
    scale = float(initial[0])

    def scaled(trial):
        value, gradient = callback(scale * trial)
        return 100.0 * value, 100.0 * scale * np.asarray(gradient)

    result = minimize(
        scaled, initial / scale, jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * len(initial),
        options={"maxiter": max_iterations, "gtol": 1e-7,
                 "ftol": 1e-12, "maxls": 40},
    )
    weights = np.maximum(result.x * scale, 0.0)
    value, gradient = callback(weights)
    projected = weights - np.maximum(weights - gradient, 0.0)
    norm = float(np.max(np.abs(projected)))
    return SimpleNamespace(
        weights=weights, objective=float(value), iterations=int(result.nit),
        evaluations=int(result.nfev) + 1, projected_gradient_norm=norm,
        converged=bool(result.success and norm <= 1e-4),
        message=str(result.message),
    )


def main():
    args = parse_args()
    started = perf_counter()
    patient = MatRadData("PROSTATE")
    grid, structures = create_dosecuda_objects(patient, resample_spacing=5.0)
    raw = {name: structures[name]["mask"].astype(bool)
           for name in ("PTV_68", "PTV_56", "Rectum", "Bladder", "BODY")}
    high = raw["PTV_68"]
    low = raw["PTV_56"] & ~high
    target = high | low
    rectum = raw["Rectum"] & ~target
    bladder = raw["Bladder"] & ~target
    body = raw["BODY"] & ~target & ~rectum & ~bladder
    masks = {"PTV_68": high, "PTV_56_exclusive": low,
             "Rectum_exclusive": rectum, "Bladder_exclusive": bladder,
             "BODY_exclusive": body}
    selected = high | low | rectum | bladder | body
    if not all(np.any(mask) for mask in masks.values()):
        raise ValueError("a required resampled structure is empty")
    iso = patient.get_target_center("PTV_68")
    plan = IMPTPlan(machine_name="HitachiProbeatJHU")
    for angle in ANGLES:
        plan.addBeam(make_beam(angle, iso))
    operator = FixedGeometryPlanDose(grid, plan)
    summary = {
        "purpose": "nominal research plan; not clinical validation",
        "case": "matRad PROSTATE", "grid_spacing_mm": 5.0,
        "grid_shape_zyx": list(map(int, grid.HU.shape)),
        "angles_deg": ANGLES, "structure_voxels": {k: int(v.sum()) for k, v in masks.items()},
        "prescriptions": {"PTV_68": 1.0, "PTV_56_exclusive": 56 / 68,
                          "Rectum_exclusive": 50 / 68, "Bladder_exclusive": 50 / 68,
                          "BODY_exclusive": 30 / 68},
        "dose_units": "relative to 68 Gy prescription; DoseCUDA MU is not clinically calibrated",
        "spot_spacing_mm": args.spacing_mm, "requested_layers": args.layers,
        "max_matrix_gib": args.max_matrix_gib,
        "max_solve_seconds": args.max_solve_seconds,
        "solver": args.solver,
        "status": "preparing", "beams": [],
    }
    for beam_operator in operator.beams:
        beam = beam_operator.beam
        model = beam_operator.beam_model
        bev = target_points(grid, target, iso, beam.gantry_angle, model)
        wet = 10.0 * beam_operator._wet.voxel_data[target]
        layer_info = place_spots(beam, bev, wet, model, args.spacing_mm, args.layers)
        summary["beams"].append({"angle_deg": beam.gantry_angle,
                                 "wet_mm_quantiles": np.percentile(wet, [2, 50, 98]).tolist(),
                                 "layers": layer_info, "spots": beam.n_spots})
    n_spots = plan.beam_list[0].n_spots + plan.beam_list[1].n_spots
    n_voxels = int(selected.sum())
    matrix_gib = 8 * n_spots * n_voxels / 2**30
    summary.update({"spots": n_spots, "matrix_gib": matrix_gib,
                    "matrix_rows": n_voxels, "matrix_rows_policy":
                    "union of all scored structures, retaining all body voxels",
                    "geometry_seconds": perf_counter() - started})
    save(args.output, summary)
    if matrix_gib > args.max_matrix_gib:
        summary["status"] = "memory_cap"
        save(args.output, summary)
        return
    # Rebuild cached geometry after spots were added; WET itself is unchanged.
    operator = FixedGeometryPlanDose(grid, plan)
    initial = operator.weights.astype(np.float64)
    initial_dose = operator.dose(initial)
    target_mean = float(np.mean(initial_dose[target]))
    if target_mean <= 1.0e-10:
        raise RuntimeError("candidate beam layers give negligible target dose")
    initial *= 1.0 / target_mean
    loss = make_loss([
        ("PTV_68", high, 1.0, 1.0, False),
        ("PTV_56", low, 56 / 68, 1.0, False),
        ("Rectum", rectum, 50 / 68, 0.3, True),
        ("Bladder", bladder, 50 / 68, 0.3, True),
        ("BODY", body, 30 / 68, 0.1, True),
    ])
    matrix_loss = make_loss([
        ("PTV_68", high[selected], 1.0, 1.0, False),
        ("PTV_56", low[selected], 56 / 68, 1.0, False),
        ("Rectum", rectum[selected], 50 / 68, 0.3, True),
        ("Bladder", bladder[selected], 50 / 68, 0.3, True),
        ("BODY", body[selected], 30 / 68, 0.1, True),
    ])
    summary["initial_weight_per_spot"] = float(initial[0])
    summary["initial_metrics"] = metrics(operator.dose(initial), masks)
    summary["initial_loss"] = loss(operator.dose(initial))[0]
    print(f"{n_spots} spots, {matrix_gib:.2f} GiB matrix; building", flush=True)
    started = perf_counter()
    influence = InfluenceMatrixPlanDose(
        operator, max_elements=n_voxels * n_spots, row_mask=selected)
    summary["matrix_build_seconds"] = perf_counter() - started
    print(f"matrix built in {summary['matrix_build_seconds']:.2f}s", flush=True)
    save(args.output, summary)
    started = perf_counter()
    gpu = GPUInfluenceMatrixObjective(
        influence.matrix, influence.shape, high[selected], rectum[selected],
        body[selected],
        prescription=1.0, oar_limit=50 / 68, normal_limit=30 / 68)
    summary["matrix_upload_seconds"] = perf_counter() - started
    probe = influence.value_and_gradient(initial, matrix_loss)
    gpu_probe = gpu.value_and_gradient_for(initial, matrix_loss)
    summary["matrix_cpu_gpu_probe_loss_abs_error"] = abs(probe[0] - gpu_probe[0])
    summary["matrix_cpu_gpu_probe_gradient_max_abs_error"] = float(np.max(
        np.abs(probe[1] - gpu_probe[1])))
    if summary["matrix_cpu_gpu_probe_loss_abs_error"] > 1.0e-8:
        raise AssertionError("CPU/GPU matrix dose mismatch")
    save(args.output, summary)
    started = perf_counter()

    def callback(weights):
        if perf_counter() - started > args.max_solve_seconds:
            raise TimeoutError("inner solver reached wall-time cap")
        return gpu.value_and_gradient_for(weights, matrix_loss)

    try:
        if args.solver == "slsqp":
            result = bounded_slsqp(
                callback, initial, weight_scale=float(initial[0]),
                objective_scale=100.0, max_iterations=args.max_iterations,
                stationarity_tolerance=1.0e-4)
        else:
            result = solve_scaled_lbfgsb(callback, initial, args.max_iterations)
    except TimeoutError:
        summary["status"] = "time_cap"
        summary["solve_seconds"] = perf_counter() - started
        save(args.output, summary)
        print("solve reached time cap", flush=True)
        return
    summary.update({"solve_seconds": perf_counter() - started,
                    "solver_converged": result.converged,
                    "solver_message": result.message,
                    "iterations": result.iterations,
                    "evaluations": result.evaluations,
                    "projected_gradient_norm": result.projected_gradient_norm,
                    "optimized_loss_matrix": result.objective,
                    "active_spots": int(np.count_nonzero(result.weights > 1.0e-4))})
    fresh_dose = operator.dose(result.weights)
    fresh_loss = loss(fresh_dose)[0]
    summary["optimized_loss_fresh_dosecuda"] = fresh_loss
    summary["fresh_cuda_loss_relative_error"] = abs(fresh_loss - result.objective) / max(fresh_loss, 1e-12)
    summary["optimized_metrics"] = metrics(fresh_dose, masks)
    summary["coverage_fraction_high_95pct"] = float(np.mean(fresh_dose[high] >= 0.95))
    summary["coverage_fraction_low_95pct"] = float(np.mean(fresh_dose[low] >= 0.95 * 56 / 68))
    summary["oar_over_limit_fraction"] = {
        "Rectum_exclusive": float(np.mean(fresh_dose[rectum] > 50 / 68)),
        "Bladder_exclusive": float(np.mean(fresh_dose[bladder] > 50 / 68)),
        "BODY_exclusive": float(np.mean(fresh_dose[body] > 30 / 68)),
    }
    high_d95 = summary["optimized_metrics"]["PTV_68"]["d95"]
    low_d95 = summary["optimized_metrics"]["PTV_56_exclusive"]["d95"]
    summary["research_coverage_gate"] = {
        "criterion": "PTV_68 D95 >= 0.90 and PTV_56-exclusive D95 >= 0.90*(56/68)",
        "passed": bool(high_d95 >= 0.90 and low_d95 >= 0.90 * 56 / 68),
        "not_a_clinical_acceptance_criterion": True,
    }
    summary["status"] = (
        "usable_research_plan" if result.converged and
        summary["research_coverage_gate"]["passed"] else "research_gate_failed"
    )
    archive = args.output.with_suffix(".npz")
    np.savez_compressed(
        archive, weights=result.weights.astype(np.float32),
        dose=fresh_dose.astype(np.float32),
        beam_0_spots=plan.beam_list[0].spot_list,
        beam_1_spots=plan.beam_list[1].spot_list,
        origin=np.asarray(grid.origin), spacing=np.asarray(grid.spacing))
    summary["reproducibility_archive"] = str(archive)
    save(args.output, summary)
    print(f"{summary['status']}; D95 high {summary['optimized_metrics']['PTV_68']['d95']:.3f}; "
          f"D95 low {summary['optimized_metrics']['PTV_56_exclusive']['d95']:.3f}; "
          f"fresh mismatch {summary['fresh_cuda_loss_relative_error']:.2e}", flush=True)


if __name__ == "__main__":
    main()
