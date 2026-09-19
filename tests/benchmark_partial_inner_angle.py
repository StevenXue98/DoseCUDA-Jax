#!/usr/bin/env python3
"""Warm-started partial-weight/angle-signal experiment on matRad PROSTATE.

The saved nominal spot template (positions and energy IDs) is held fixed while
the two gantry angles move locally; weights retain their beam/spot identity.
Both matrix and matrix-free callbacks optimize the same five-region loss.
This is a research diagnostic, not a clinical treatment plan or BAO solver.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from DoseCUDA import IMPTPlan
from DoseCUDA.gpu_influence_matrix import GPUInfluenceMatrixObjective
from DoseCUDA.impt_weight_optimization import (
    FixedGeometryPlanDose, InfluenceMatrixPlanDose,
)
from tests.run_prostate_nominal_plan import make_beam, make_loss
from utils.matrad_converter import MatRadData, create_dosecuda_objects


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (0, 1, 5, 10, 20)
ANGLES = ((92.0, 270.0), (90.0, 272.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--start-case", type=int, default=0,
                        help="Zero-based case index; useful for bounded reruns")
    parser.add_argument("--angle-step-deg", type=float, default=1.0)
    parser.add_argument("--max-matrix-gib", type=float, default=4.0)
    parser.add_argument("--max-partial-seconds", type=float, default=120.0)
    parser.add_argument("--max-reference-seconds", type=float, default=300.0)
    parser.add_argument("--max-reference-iterations", type=int, default=2500)
    parser.add_argument("--archive", type=Path, default=ROOT /
                        "test_phantom_output/bao_prostate_nominal/summary.npz")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "test_phantom_output/bao_partial_inner/summary.json")
    args = parser.parse_args()
    if (args.cases not in (1, 2) or args.start_case not in (0, 1) or
            args.start_case + args.cases > len(ANGLES) or
            args.angle_step_deg <= 0 or
            args.max_matrix_gib <= 0 or args.max_partial_seconds <= 0 or
            args.max_reference_seconds <= 0 or
            args.max_reference_iterations < 1):
        parser.error("invalid experiment limits")
    return args


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def masks_and_losses(structures):
    raw = {name: structures[name]["mask"].astype(bool)
           for name in ("PTV_68", "PTV_56", "Rectum", "Bladder", "BODY")}
    high = raw["PTV_68"]
    low = raw["PTV_56"] & ~high
    target = high | low
    rectum = raw["Rectum"] & ~target
    bladder = raw["Bladder"] & ~target
    body = raw["BODY"] & ~target & ~rectum & ~bladder
    selected = high | low | rectum | bladder | body
    terms = (("PTV_68", high, 1.0, 1.0, False),
             ("PTV_56", low, 56 / 68, 1.0, False),
             ("Rectum", rectum, 50 / 68, 0.3, True),
             ("Bladder", bladder, 50 / 68, 0.3, True),
             ("BODY", body, 30 / 68, 0.1, True))
    full = make_loss(terms)
    selected_loss = make_loss([(name, mask[selected], limit, penalty, hinge)
                               for name, mask, limit, penalty, hinge in terms])
    return high, rectum, body, selected, full, selected_loss


def make_operator(grid, iso, spot_lists, angles):
    plan = IMPTPlan(machine_name="HitachiProbeatJHU")
    for angle, spots in zip(angles, spot_lists):
        beam = make_beam(float(angle), iso)
        beam.spot_list = np.ascontiguousarray(spots, dtype=np.float32).copy()
        beam.n_spots = len(spots)
        plan.addBeam(beam)
    return FixedGeometryPlanDose(grid, plan)


def partial_lbfgsb(value_and_gradient, initial, *, max_seconds, iterations=20):
    """Collect accepted L-BFGS-B iterates; return early-stop truthfully."""
    scale = float(np.median(initial[initial > 0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("a positive warm-start weight is required")
    snapshots = {0: np.asarray(initial, dtype=np.float64).copy()}
    started = perf_counter()
    evaluations = 0

    def scaled(trial):
        nonlocal evaluations
        if perf_counter() - started > max_seconds:
            raise TimeoutError("partial weight solve reached time cap")
        value, gradient = value_and_gradient(scale * trial)
        evaluations += 1
        return 100.0 * float(value), 100.0 * scale * np.asarray(gradient)

    def capture(trial):
        snapshots[len(snapshots)] = (scale * trial).copy()

    try:
        result = minimize(
            scaled, initial / scale, jac=True, method="L-BFGS-B",
            bounds=[(0.0, None)] * len(initial), callback=capture,
            options={"maxiter": iterations, "gtol": 1e-12, "ftol": 1e-16,
                     "maxls": 40},
        )
        message = str(result.message)
        success = bool(result.success)
    except TimeoutError:
        message = "time cap"
        success = False
    return snapshots, {"seconds": perf_counter() - started,
                       "iterations": len(snapshots) - 1,
                       "evaluations": evaluations, "success": success,
                       "message": message}


def full_lbfgsb(value_and_gradient, initial, *, max_seconds, max_iterations):
    scale = float(np.median(initial[initial > 0]))
    started = perf_counter()

    def scaled(trial):
        if perf_counter() - started > max_seconds:
            raise TimeoutError("reference weight solve reached time cap")
        value, gradient = value_and_gradient(scale * trial)
        return 100.0 * float(value), 100.0 * scale * np.asarray(gradient)

    result = minimize(
        scaled, initial / scale, jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * len(initial),
        options={"maxiter": max_iterations, "gtol": 1e-7, "ftol": 1e-12,
                 "maxls": 40},
    )
    weights = np.maximum(scale * result.x, 0.0)
    value, gradient = value_and_gradient(weights)
    projected = weights - np.maximum(weights - gradient, 0.0)
    return weights, {"seconds": perf_counter() - started,
                     "iterations": int(result.nit), "evaluations": int(result.nfev) + 1,
                     "loss": float(value), "projected_gradient_norm":
                     float(np.max(np.abs(projected))), "success": bool(result.success),
                     "message": str(result.message)}


def angle_signal(probes, weights, loss, step):
    """Central secant of exact rounded DoseCUDA loss, holding weights fixed."""
    signal = []
    for plus, minus in probes:
        plus_loss = loss(plus.dose(weights))[0]
        minus_loss = loss(minus.dose(weights))[0]
        signal.append((plus_loss - minus_loss) / (2.0 * step))
    return np.asarray(signal, dtype=np.float64)


def vector_comparison(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    norms = np.linalg.norm(candidate) * np.linalg.norm(reference)
    return {"candidate": candidate.tolist(),
            "cosine_vs_reference": float(np.dot(candidate, reference) / norms)
            if norms > 0 else None,
            "relative_l2_error": float(np.linalg.norm(candidate - reference) /
                                       max(np.linalg.norm(reference), 1e-12))}


def main():
    args = parse_args()
    with np.load(args.archive) as archive:
        initial = np.asarray(archive["weights"], dtype=np.float64)
        spot_lists = (archive["beam_0_spots"].copy(),
                      archive["beam_1_spots"].copy())
    patient = MatRadData("PROSTATE")
    grid, structures = create_dosecuda_objects(patient, resample_spacing=5.0)
    high, rectum, body, selected, full_loss, selected_loss = masks_and_losses(structures)
    iso = patient.get_target_center("PTV_68")
    row_count = int(selected.sum())
    matrix_gib = 8 * row_count * len(initial) / 2**30
    if matrix_gib > args.max_matrix_gib:
        raise ValueError(f"matrix needs {matrix_gib:.2f} GiB above cap")
    output = {"purpose": "partial-weight angle-signal and cost diagnostic",
              "case": "matRad PROSTATE, two beams, 5 mm research grid",
              "loss": "same five-region relative squared objective as nominal plan",
              "spot_mapping": "fixed baseline beam-local (x,y,energy_id) and index; "
                              "carry weight at same index as angles move",
              "spot_count": len(initial), "scored_rows": row_count,
              "matrix_gib": matrix_gib, "angle_step_deg": args.angle_step_deg,
              "checkpoints": CHECKPOINTS, "cases": []}
    save(args.output, output)

    for angles in ANGLES[args.start_case:args.start_case + args.cases]:
        case = {"angles_deg": angles, "status": "preparing"}
        output["cases"].append(case)
        save(args.output, output)
        started = perf_counter()
        operator = make_operator(grid, iso, spot_lists, angles)
        case["geometry_seconds"] = perf_counter() - started
        started = perf_counter()
        influence = InfluenceMatrixPlanDose(
            operator, max_elements=row_count * len(initial), row_mask=selected)
        case["matrix_build_seconds"] = perf_counter() - started
        print(f"{angles}: matrix built in {case['matrix_build_seconds']:.1f}s", flush=True)
        started = perf_counter()
        gpu = GPUInfluenceMatrixObjective(
            influence.matrix, influence.shape, high[selected], rectum[selected],
            body[selected], prescription=1.0, oar_limit=50 / 68,
            normal_limit=30 / 68)
        case["matrix_upload_seconds"] = perf_counter() - started
        matrix_callback = lambda w: gpu.value_and_gradient_for(w, selected_loss)
        free_callback = lambda w: operator.value_and_gradient(w, full_loss)
        started = perf_counter()
        matrix_probe = matrix_callback(initial)
        case["matrix_callback_seconds"] = perf_counter() - started
        started = perf_counter()
        free_probe = free_callback(initial)
        case["matrix_free_callback_seconds"] = perf_counter() - started
        case["probe_loss_relative_error"] = abs(matrix_probe[0] - free_probe[0]) / max(
            abs(matrix_probe[0]), 1e-12)
        case["probe_weight_gradient_relative_l2_error"] = float(np.linalg.norm(
            matrix_probe[1] - free_probe[1]) / max(np.linalg.norm(matrix_probe[1]), 1e-12))
        save(args.output, output)

        matrix_states, case["matrix_partial"] = partial_lbfgsb(
            matrix_callback, initial, max_seconds=args.max_partial_seconds)
        free_states, case["matrix_free_partial"] = partial_lbfgsb(
            free_callback, initial, max_seconds=args.max_partial_seconds)
        save(args.output, output)
        try:
            reference, case["reference_solve"] = full_lbfgsb(
                matrix_callback, initial, max_seconds=args.max_reference_seconds,
                max_iterations=args.max_reference_iterations)
        except TimeoutError:
            case["status"] = "reference_time_cap"
            save(args.output, output)
            continue
        case["reference_fresh_cuda_loss"] = float(full_loss(operator.dose(reference))[0])
        case["reference_loss_relative_check"] = abs(
            case["reference_fresh_cuda_loss"] - case["reference_solve"]["loss"]) / max(
                case["reference_fresh_cuda_loss"], 1e-12)
        case["reference_certified"] = bool(
            case["reference_solve"]["success"] and
            case["reference_solve"]["projected_gradient_norm"] <= 1e-6)
        archive_path = args.output.with_name(
            args.output.stem + f"_case_{args.start_case + len(output['cases']) - 1}.npz")
        stored = {"reference_weights": reference.astype(np.float64),
                  "angles_deg": np.asarray(angles, dtype=np.float64)}
        for name, states in (("matrix", matrix_states),
                             ("matrix_free", free_states)):
            for step, weights in states.items():
                stored[f"{name}_weights_{step}"] = weights.astype(np.float64)
        np.savez_compressed(archive_path, **stored)
        case["checkpoint_archive"] = str(archive_path)
        save(args.output, output)

        probes = []
        started = perf_counter()
        for axis in range(2):
            shifted = []
            for sign in (1.0, -1.0):
                trial = list(angles)
                trial[axis] += sign * args.angle_step_deg
                shifted.append(make_operator(grid, iso, spot_lists, trial))
            probes.append(tuple(shifted))
        case["angle_probe_geometry_seconds"] = perf_counter() - started
        started = perf_counter()
        reference_signal = angle_signal(probes, reference, full_loss,
                                        args.angle_step_deg)
        case["reference_angle_signal"] = reference_signal.tolist()
        case["reference_signal_seconds"] = perf_counter() - started
        case["checkpoints_result"] = {}
        for name, states, callback in (("matrix", matrix_states, matrix_callback),
                                       ("matrix_free", free_states, free_callback)):
            rows = []
            for k in CHECKPOINTS:
                if k not in states:
                    rows.append({"step": k, "available": False})
                    continue
                w = states[k]
                value = float(full_loss(operator.dose(w))[0])
                matrix_value, matrix_gradient = matrix_callback(w)
                free_value, free_gradient = free_callback(w)
                signal = angle_signal(probes, w, full_loss, args.angle_step_deg)
                row = {"step": k, "available": True, "exact_loss": value,
                       "gap_vs_reference": value / case["reference_fresh_cuda_loss"] - 1,
                       "matrix_loss_relative_error": abs(matrix_value - value) /
                       max(abs(value), 1e-12),
                       "matrix_free_loss_relative_error": abs(free_value - value) /
                       max(abs(value), 1e-12),
                       "weight_gradient_relative_l2_error": float(np.linalg.norm(
                           matrix_gradient - free_gradient) /
                           max(np.linalg.norm(matrix_gradient), 1e-12)),
                       "angle_signal": vector_comparison(signal, reference_signal)}
                rows.append(row)
            case["checkpoints_result"][name] = rows
            save(args.output, output)
        case["status"] = ("complete" if case["reference_certified"]
                          else "reference_not_certified")
        save(args.output, output)
        print(f"{angles}: reference loss {case['reference_fresh_cuda_loss']:.6g}; "
              f"matrix-free {case['matrix_free_partial']['iterations']} partial steps", flush=True)
        del probes, gpu, influence, operator
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
