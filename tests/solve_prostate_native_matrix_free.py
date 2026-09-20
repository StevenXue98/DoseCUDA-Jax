#!/usr/bin/env python3
"""One capped native-grid fixed-angle weight solve, with convergence trace.

This measures the cost of approaching the convex inner optimum at the saved
nominal two-field prostate geometry. It does not perform angle optimization,
build an influence matrix, or establish clinical plan quality.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from tests.benchmark_prostate_native_matrix_free import (
    ANGLES, ARCHIVE, GPUMemorySampler, gpu_used_mib, save,
)
from tests.benchmark_partial_inner_angle import make_operator, masks_and_losses
from utils.matrad_converter import MatRadData, create_dosecuda_objects


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test_phantom_output/bao_prostate_native_matrix_free/full_solve.json"
CHECKPOINTS = frozenset((5, 20, 50, 100, 200, 400, 600, 800, 1000,
                         1250, 1500, 1750, 2000, 2250, 2500, 3000))


def projected_gradient_norm(weights, gradient):
    weights = np.asarray(weights, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    projected = weights - np.maximum(weights - gradient, 0.0)
    return float(np.max(np.abs(projected)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--stationarity-tolerance", type=float, default=1e-4)
    parser.add_argument("--min-free-gpu-mib", type=int, default=2048)
    parser.add_argument("--max-used-gpu-mib", type=int, default=9500)
    args = parser.parse_args()
    if (args.max_seconds <= 0 or args.max_iterations < 20 or
            args.stationarity_tolerance <= 0 or args.min_free_gpu_mib < 0 or
            args.max_used_gpu_mib <= 0):
        parser.error("invalid full-solve limits")
    return args


def main():
    args = parse_args()
    started = perf_counter()
    baseline_used, baseline_free = gpu_used_mib()
    if baseline_free < args.min_free_gpu_mib:
        raise RuntimeError(f"only {baseline_free} MiB GPU memory free")
    with np.load(args.archive) as archive:
        initial = np.asarray(archive["weights"], dtype=np.float64)
        spot_lists = (archive["beam_0_spots"].copy(),
                      archive["beam_1_spots"].copy())
    if len(initial) != sum(len(spots) for spots in spot_lists):
        raise ValueError("archived spot and weight counts disagree")
    summary = {
        "purpose": "one capped fixed-angle inner convergence trace; not BAO or clinical validation",
        "case": "matRad PROSTATE native 3 mm", "angles_deg": ANGLES,
        "spot_count": len(initial), "matrix": "none; matrix-free DoseCUDA",
        "caps": {"max_seconds": args.max_seconds,
                 "max_iterations": args.max_iterations,
                 "stationarity_tolerance": args.stationarity_tolerance,
                 "max_used_gpu_mib": args.max_used_gpu_mib},
        "gpu_baseline_used_mib": baseline_used,
        "gpu_baseline_free_mib": baseline_free,
        "checkpoints": [], "solver_runs": [], "status": "preparing",
    }
    save(args.output, summary)
    weights_path = args.output.with_suffix(".npz")

    with GPUMemorySampler() as sampler:
        total_accepted = 0
        evaluations = 0
        try:
            def check_budget():
                if perf_counter() - started >= args.max_seconds:
                    raise TimeoutError("full-solve wall-time cap reached")
                if (sampler.peak_mib is not None and
                        sampler.peak_mib > args.max_used_gpu_mib):
                    raise MemoryError("sampled GPU memory cap reached")

            patient = MatRadData("PROSTATE")
            if not np.allclose(patient.spacing, 3.0):
                raise ValueError("expected native 3 mm isotropic CT")
            grid, structures = create_dosecuda_objects(
                patient, resample_spacing=3.0)
            high, _, _, _, loss, _ = masks_and_losses(structures)
            summary["grid_shape_zyx"] = list(map(int, grid.HU.shape))
            check_budget()
            geometry_started = perf_counter()
            operator = make_operator(
                grid, patient.get_target_center("PTV_68"), spot_lists, ANGLES)
            summary["geometry_seconds"] = perf_counter() - geometry_started
            save(args.output, summary)
            scale = float(np.median(initial[initial > 0]))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("the archived plan needs positive weights")

            def value_and_gradient(weights):
                nonlocal evaluations
                check_budget()
                value, gradient = operator.value_and_gradient(weights, loss)
                evaluations += 1
                if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                    raise ValueError("nonfinite inner objective/gradient")
                return float(value), np.asarray(gradient, dtype=np.float64)

            def record(iteration, weights, *, reason):
                value, gradient = value_and_gradient(weights)
                row = {
                    "iteration": iteration, "reason": reason,
                    "elapsed_seconds": perf_counter() - started,
                    "loss": value,
                    "projected_gradient_norm": projected_gradient_norm(
                        weights, gradient),
                    "active_spots": int(np.count_nonzero(weights > 1e-4)),
                    "evaluations": evaluations,
                }
                summary["checkpoints"].append(row)
                np.savez_compressed(weights_path,
                                    weights=np.asarray(weights, dtype=np.float32),
                                    iteration=np.asarray(iteration))
                save(args.output, summary)
                print(f"iteration {iteration}: loss {value:.7g}, "
                      f"projected gradient {row['projected_gradient_norm']:.3g}, "
                      f"elapsed {row['elapsed_seconds']:.1f}s", flush=True)
                return row

            current = np.maximum(initial, 0.0)
            record(0, current, reason="initial")
            summary["status"] = "solving"
            save(args.output, summary)
            for restart in range(3):
                check_budget()
                budget = args.max_iterations - total_accepted
                if budget <= 0:
                    break
                accepted_this_run = 0

                def scaled(trial):
                    value, gradient = value_and_gradient(scale * trial)
                    return 100.0 * value, 100.0 * scale * gradient

                def capture(trial):
                    nonlocal accepted_this_run, total_accepted
                    accepted_this_run += 1
                    total_accepted += 1
                    if total_accepted in CHECKPOINTS:
                        record(total_accepted, np.maximum(scale * trial, 0.0),
                               reason="checkpoint")
                    check_budget()

                result = minimize(
                    scaled, current / scale, jac=True, method="L-BFGS-B",
                    bounds=[(0.0, None)] * len(current), callback=capture,
                    options={"maxiter": budget, "gtol": 1e-12,
                             "ftol": 1e-16, "maxls": 40},
                )
                current = np.maximum(scale * result.x, 0.0)
                row = record(total_accepted, current, reason="solver_stop")
                summary["solver_runs"].append({
                    "restart": restart, "iterations": int(result.nit),
                    "evaluations": int(result.nfev),
                    "success": bool(result.success),
                    "message": str(result.message),
                })
                save(args.output, summary)
                if row["projected_gradient_norm"] <= args.stationarity_tolerance:
                    summary["status"] = "stationarity_met"
                    break
                if accepted_this_run == 0:
                    summary["status"] = "stalled"
                    break
            else:
                summary["status"] = "restart_limit"
            if summary["status"] == "solving":
                summary["status"] = "iteration_cap"
            summary["final_iterations"] = total_accepted
            summary["final_evaluations"] = evaluations
            summary["final_loss"] = summary["checkpoints"][-1]["loss"]
            summary["final_projected_gradient_norm"] = summary["checkpoints"][-1][
                "projected_gradient_norm"]
            dose = operator.dose(current)
            summary["final_PTV_68_d95"] = float(np.percentile(dose[high], 5))
        except (TimeoutError, MemoryError) as exc:
            summary["status"] = "time_cap" if isinstance(exc, TimeoutError) else "memory_cap"
            summary["stop_reason"] = str(exc)
            summary["final_iterations"] = total_accepted
            summary["final_evaluations"] = evaluations
        finally:
            summary["elapsed_seconds"] = perf_counter() - started
            summary["gpu_peak_sampled_mib"] = sampler.peak_mib
            summary["gpu_sampler_error"] = sampler.error
            save(args.output, summary)
    print(f"Status {summary['status']}; {total_accepted} accepted weight steps; "
          f"{summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
