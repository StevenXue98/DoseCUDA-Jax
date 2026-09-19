#!/usr/bin/env python3
"""Compare complete fixed-angle CPU/GPU matrix and matrix-free CUDA solves.

Each source angle gets a cold solve; a nearby angle is then solved from the
same accurate source weights by every method. Geometry/WET, dose-column
construction, GPU upload, and optimizer timings are reported separately.
This is a numerical/throughput benchmark on a synthetic case, not BAO.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from DoseCUDA.cuda_weight_solver import solve_cuda_spot_weights
from DoseCUDA.gpu_influence_matrix import GPUInfluenceMatrixObjective
from DoseCUDA.impt_weight_optimization import (
    InfluenceMatrixPlanDose,
    bounded_slsqp,
)
from run_toy_bao_baseline import NORMAL_LIMIT, OAR_LIMIT, ROOT, RX, make_case
from run_toy_three_beam_joint import make_joint_operator


SOURCE_ANGLES = (
    (-25.0, 40.0, 60.0),  # previously capped matrix-free FISTA
    (-20.0, 14.0, 53.0),  # previously capped matrix-free FISTA
    (4.0, 27.0, 67.0),   # previously capped matrix-free FISTA
    (-60.0, 0.0, 60.0),
    (-77.0, 12.0, 78.0),
    (-31.0, 36.0, 64.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=len(SOURCE_ANGLES))
    parser.add_argument("--fista-max-iterations", type=int, default=10000)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, default=Path(ROOT) /
                        "test_phantom_output/bao_gpu_influence/benchmark.json")
    return parser.parse_args()


def solve_case(angles, initial_weights, kind, case, args):
    started = perf_counter()
    operator = make_joint_operator(angles, *case[:3])
    geometry_seconds = perf_counter() - started

    started = perf_counter()
    influence = InfluenceMatrixPlanDose(operator)
    matrix_build_seconds = perf_counter() - started

    started = perf_counter()
    gpu_matrix = GPUInfluenceMatrixObjective(
        influence.matrix, influence.shape, case[4], case[5], case[6],
        prescription=RX, oar_limit=OAR_LIMIT, normal_limit=NORMAL_LIMIT,
        gpu_id=operator.beams[0].gpu_id)
    upload_seconds = perf_counter() - started

    initial = operator.weights if initial_weights is None else initial_weights
    probe = np.maximum(np.asarray(initial, dtype=np.float64), 0.0)
    cpu_value, cpu_gradient = influence.value_and_gradient(probe, case[-1])
    gpu_value, gpu_gradient = gpu_matrix.value_and_gradient(probe)
    gpu_dose = gpu_matrix.dose(probe)
    dose_error = float(np.max(np.abs(gpu_dose - influence.dose(probe))))
    gradient_error = float(np.max(np.abs(gpu_gradient - cpu_gradient)))
    if (abs(gpu_value - cpu_value) > 1.0e-11 or
            gradient_error > 1.0e-10 or dose_error > 1.0e-10):
        raise AssertionError("GPU matrix disagrees with the CPU matrix")

    solve_settings = dict(weight_scale=0.02, objective_scale=1.0e4,
                          max_iterations=1000, stationarity_tolerance=1.0e-6)
    started = perf_counter()
    reference = bounded_slsqp(
        lambda weights: influence.value_and_gradient(weights, case[-1]),
        initial, **solve_settings)
    cpu_solve_seconds = perf_counter() - started
    if not reference.converged:
        raise RuntimeError(f"CPU matrix reference failed at {angles}: "
                           f"{reference.message}")

    started = perf_counter()
    gpu_result = bounded_slsqp(
        gpu_matrix.value_and_gradient, initial, **solve_settings)
    gpu_solve_seconds = perf_counter() - started

    started = perf_counter()
    matrix_free = solve_cuda_spot_weights(
        operator, case[4], case[5], case[6], prescription=RX,
        oar_limit=OAR_LIMIT, normal_limit=NORMAL_LIMIT,
        initial_weights=initial, method="fista",
        max_iterations=args.fista_max_iterations,
        gradient_tolerance=args.gradient_tolerance)
    matrix_free_solve_seconds = perf_counter() - started

    rows = []
    for method, result, solve_seconds, build_seconds, transfer_seconds in (
        ("cpu_matrix_slsqp", reference, cpu_solve_seconds,
         matrix_build_seconds, 0.0),
        ("gpu_matrix_slsqp", gpu_result, gpu_solve_seconds,
         matrix_build_seconds, upload_seconds),
        ("cuda_matrix_free_fista", matrix_free,
         matrix_free_solve_seconds, 0.0, 0.0),
    ):
        exact_loss = case[-1](operator.dose(result.weights))[0]
        relative_check_error = abs(exact_loss - result.objective) / max(
            exact_loss, 1.0e-12)
        if relative_check_error > 5.0e-3:
            raise AssertionError(f"{method} loss disagrees with fresh DoseCUDA")
        row = {
            "angles_deg": angles, "initialization": kind, "method": method,
            "spot_count": operator.n_spots,
            "geometry_seconds": geometry_seconds,
            "matrix_build_seconds": build_seconds,
            "matrix_upload_seconds": transfer_seconds,
            "solve_seconds": solve_seconds,
            "full_seconds": geometry_seconds + build_seconds +
                            transfer_seconds + solve_seconds,
            "iterations": result.iterations,
            "loss": exact_loss,
            "relative_loss_gap_vs_cpu_matrix":
                exact_loss / reference.objective - 1.0,
            "projected_gradient_norm": result.projected_gradient_norm,
            "converged": result.converged,
            "fresh_cuda_loss_check_relative_error": relative_check_error,
            "gpu_cpu_probe_dose_max_abs_error": dose_error,
            "gpu_cpu_probe_gradient_max_abs_error": gradient_error,
        }
        rows.append(row)
        print(f"{kind:4s} {angles} {method:22s} "
              f"{row['full_seconds']:.3f}s total, "
              f"{result.iterations:5d} iter, "
              f"gap {row['relative_loss_gap_vs_cpu_matrix']:+.3%}, "
              f"converged={result.converged}", flush=True)
    return rows, reference.weights


def main():
    args = parse_args()
    if (args.sources < 1 or args.sources > len(SOURCE_ANGLES) or
            args.fista_max_iterations < 1 or
            not np.isfinite(args.gradient_tolerance) or
            args.gradient_tolerance <= 0):
        raise ValueError("invalid benchmark settings")
    case = make_case()
    started = perf_counter()
    rows = []
    for source in SOURCE_ANGLES[:args.sources]:
        source_rows, reference_weights = solve_case(
            source, None, "cold", case, args)
        rows.extend(source_rows)
        nearby = (source[0] + 2.0, source[1], source[2])
        warm_rows, _ = solve_case(
            nearby, reference_weights, "warm", case, args)
        rows.extend(warm_rows)
    summary = {
        "settings": {"sources": args.sources,
                     "fista_max_iterations": args.fista_max_iterations,
                     "gradient_tolerance": args.gradient_tolerance},
        "wall_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
