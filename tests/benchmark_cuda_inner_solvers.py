#!/usr/bin/env python3
"""Benchmark matrix-free resident-GPU weight solvers on 2/3-beam toy plans.

This measures complete inner-solver wall time after angle-specific WET setup.
Both CUDA methods start from the same weights. An optional accurate CPU solve
is used only as a loss reference; it is not part of the GPU timing.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from DoseCUDA.cuda_weight_solver import solve_cuda_spot_weights
from run_toy_bao_baseline import NORMAL_LIMIT, OAR_LIMIT, RX, make_case
from run_toy_three_beam_joint import make_joint_operator, solve_subset


CASES = ((-56.0, 52.0), (-20.0, 23.0, 75.0), (-60.0, 0.0, 60.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--warm-start-check", action="store_true",
                        help="Compare cold/warm three-beam FISTA at a nearby angle")
    parser.add_argument("--methods", default="pg,fista,cg,lbfgs",
                        help="Comma-separated subset of pg,fista,cg,lbfgs")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON file for reproducible benchmark results")
    return parser.parse_args()


def run():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    methods = tuple(args.methods.split(","))
    if not methods or any(method not in ("pg", "fista", "cg", "lbfgs")
                          for method in methods):
        raise ValueError("--methods must be a subset of pg,fista,cg,lbfgs")
    case = make_case()
    rows = []
    warm_rows = []
    for angles in CASES:
        started = perf_counter()
        operator = make_joint_operator(angles, *case[:3])
        geometry_seconds = perf_counter() - started
        initial_weights = operator.weights.copy()
        initial_loss = case[-1](operator.dose(initial_weights))[0]
        print(f"{angles}: {operator.n_spots} weights, initial loss "
              f"{initial_loss:.9g}, WET setup {geometry_seconds:.3f} s",
              flush=True)
        for method in methods:
            for repeat in range(args.repeats):
                started = perf_counter()
                result = solve_cuda_spot_weights(
                    operator, case[4], case[5], case[6],
                    prescription=RX, oar_limit=OAR_LIMIT,
                    normal_limit=NORMAL_LIMIT,
                    initial_weights=initial_weights,
                    method=method,
                    max_iterations=args.max_iterations,
                    gradient_tolerance=args.gradient_tolerance,
                )
                seconds = perf_counter() - started
                checked_loss = case[-1](operator.dose(result.weights))[0]
                row = {
                    "angles_deg": angles,
                    "spot_count": operator.n_spots,
                    "method": method,
                    "repeat": repeat,
                    "geometry_seconds": geometry_seconds,
                    "solve_seconds": seconds,
                    "initial_loss": initial_loss,
                    "gpu_loss": result.objective,
                    "checked_cuda_loss": checked_loss,
                    "loss_check_relative_error": abs(checked_loss - result.objective)
                    / max(checked_loss, 1.0e-12),
                    "projected_gradient_norm": result.projected_gradient_norm,
                    "iterations": result.iterations,
                    "forward_evaluations": result.forward_evaluations,
                    "gradient_evaluations": result.gradient_evaluations,
                    "converged": result.converged,
                }
                if row["loss_check_relative_error"] > 5.0e-3:
                    raise AssertionError("resident GPU loss differs from fresh CUDA dose")
                rows.append(row)
                print(f"  {method} repeat {repeat}: {seconds:.3f} s, "
                      f"loss {checked_loss:.9g}, PG "
                      f"{result.projected_gradient_norm:.3g}, "
                      f"{result.iterations} iterations, "
                      f"{result.forward_evaluations} forwards, "
                      f"converged={result.converged}", flush=True)
        if args.reference:
            reference = solve_subset(angles, case, backend="influence_slsqp")
            for row in rows:
                if tuple(row["angles_deg"]) == angles:
                    row["reference_loss"] = reference["loss"]
                    row["relative_loss_gap"] = (
                        row["checked_cuda_loss"] - reference["loss"]
                    ) / reference["loss"]
            print(f"  CPU accuracy reference: loss {reference['loss']:.9g} "
                  f"({reference['solve_seconds']:.3f} s)", flush=True)
    if args.warm_start_check:
        source_angles = (-20.0, 23.0, 75.0)
        target_angles = (-18.0, 23.0, 75.0)
        source = make_joint_operator(source_angles, *case[:3])
        optimized_source = solve_cuda_spot_weights(
            source, case[4], case[5], case[6], prescription=RX,
            oar_limit=OAR_LIMIT, normal_limit=NORMAL_LIMIT,
            method="fista", max_iterations=2000,
            gradient_tolerance=1.0e-5)
        target = make_joint_operator(target_angles, *case[:3])
        target_reference = solve_subset(
            target_angles, case, backend="influence_slsqp")["loss"]
        print(f"Nearby-angle warm start {source_angles} -> {target_angles}; "
              f"target reference {target_reference:.9g}", flush=True)
        for name, initial in (("cold", target.weights),
                              ("warm", optimized_source.weights)):
            for budget in (100, 250, 500):
                started = perf_counter()
                result = solve_cuda_spot_weights(
                    target, case[4], case[5], case[6],
                    prescription=RX, oar_limit=OAR_LIMIT,
                    normal_limit=NORMAL_LIMIT,
                    initial_weights=initial, method="fista",
                    max_iterations=budget, gradient_tolerance=1.0e-5)
                seconds = perf_counter() - started
                checked = case[-1](target.dose(result.weights))[0]
                warm_rows.append({
                    "source_angles_deg": source_angles,
                    "target_angles_deg": target_angles,
                    "initialization": name,
                    "max_iterations": budget,
                    "solve_seconds": seconds,
                    "loss": checked,
                    "reference_loss": target_reference,
                    "relative_loss_gap": checked / target_reference - 1,
                })
                print(f"  {name} {budget}: {seconds:.3f} s, loss {checked:.9g}, "
                      f"gap {(checked / target_reference - 1):.2%}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump({
                "max_iterations": args.max_iterations,
                "gradient_tolerance": args.gradient_tolerance,
                "repeats": args.repeats,
                "methods": methods,
                "rows": rows,
                "warm_start_rows": warm_rows,
            }, handle, indent=2)
        print(f"Saved {args.output}", flush=True)
    else:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run()
