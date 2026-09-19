#!/usr/bin/env python3
"""Time tolerance-stopped CUDA inner solves across many three-beam starts.

Each independent angle triple gets one cold spot-weight solve, then a few
deterministic nearby 2-degree proposals. Proposals reuse their parent's spot
weights and are accepted only if the reoptimized DoseCUDA loss falls. This
is a warm-start workload, not a complete BAO algorithm or a clinical plan.
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from DoseCUDA.cuda_weight_solver import solve_cuda_spot_weights
from run_toy_bao_baseline import NORMAL_LIMIT, OAR_LIMIT, ROOT, RX, make_case
from run_toy_three_beam_joint import make_joint_operator, solve_subset


_WORKER_CASE = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-iterations", type=int, default=10000,
                        help="Safety cap; convergence is tolerance-based")
    parser.add_argument("--reference-samples", type=int, default=8,
                        help="Number of starts to check at cold and final angles")
    parser.add_argument("--output", type=Path, default=Path(ROOT) /
                        "test_phantom_output/bao_cuda_inner_multistart/summary.json")
    return parser.parse_args()


def generate_paths(count, steps, seed):
    rng = np.random.default_rng(seed)
    paths = []
    seen = set()
    while len(paths) < count:
        start = np.sort(rng.integers(-80, 81, size=3)).astype(np.float64)
        if np.min(np.diff(start)) < 20:
            continue
        key = tuple(start)
        if key in seen:
            continue
        seen.add(key)
        proposals = []
        angles = start.copy()
        for _ in range(steps):
            for _ in range(100):
                trial = angles.copy()
                axis = int(rng.integers(0, 3))
                trial[axis] += float(rng.choice((-2, 2)))
                if (np.all(np.abs(trial) <= 90) and
                        np.min(np.diff(trial)) >= 15):
                    break
            else:
                raise RuntimeError("could not generate a valid nearby proposal")
            proposals.append(tuple(float(value) for value in trial))
            # Proposal coordinates are generated as a local path. The worker
            # later uses its *accepted* parent and the same coordinate step.
            angles = trial
        paths.append({"start_id": len(paths), "start_angles_deg": key,
                      "proposals_deg": proposals})
    return paths


def worker_initialize():
    global _WORKER_CASE
    _WORKER_CASE = make_case()


def solve_at(angles, initial_weights, *, start_id, step, tolerance,
             max_iterations):
    case = _WORKER_CASE
    started = perf_counter()
    operator = make_joint_operator(angles, *case[:3])
    geometry_seconds = perf_counter() - started
    started = perf_counter()
    result = solve_cuda_spot_weights(
        operator, case[4], case[5], case[6], prescription=RX,
        oar_limit=OAR_LIMIT, normal_limit=NORMAL_LIMIT,
        initial_weights=initial_weights, method="fista",
        gradient_tolerance=tolerance, max_iterations=max_iterations)
    solve_seconds = perf_counter() - started
    checked_loss = case[-1](operator.dose(result.weights))[0]
    relative_check_error = abs(checked_loss - result.objective) / max(
        checked_loss, 1.0e-12)
    if relative_check_error > 5.0e-3:
        raise AssertionError("resident and fresh DoseCUDA losses disagree")
    row = {
        "start_id": start_id, "step": step, "worker_pid": os.getpid(),
        "angles_deg": tuple(float(value) for value in angles),
        "kind": "cold" if step == 0 else "warm",
        "geometry_seconds": geometry_seconds,
        "solve_seconds": solve_seconds,
        "iterations": result.iterations,
        "forward_evaluations": result.forward_evaluations,
        "gradient_evaluations": result.gradient_evaluations,
        "projected_gradient_norm": result.projected_gradient_norm,
        "converged": result.converged,
        "hit_iteration_cap": bool(not result.converged and
                                  result.iterations >= max_iterations),
        "loss": checked_loss,
        "resident_loss_check_relative_error": relative_check_error,
    }
    return row, result.weights


def run_start(path, tolerance, max_iterations):
    start_id = path["start_id"]
    current_angles = tuple(path["start_angles_deg"])
    cold, weights = solve_at(
        current_angles, None, start_id=start_id, step=0,
        tolerance=tolerance, max_iterations=max_iterations)
    cold["accepted"] = True
    rows = [cold]
    current_loss = cold["loss"]
    for step, planned in enumerate(path["proposals_deg"], start=1):
        # Follow the pre-generated 2-degree coordinate *increment* from the
        # preceding planned point, applied to the last accepted angle set.
        preceding = (path["start_angles_deg"] if step == 1 else
                     path["proposals_deg"][step - 2])
        difference = np.asarray(planned) - np.asarray(preceding)
        candidate_angles = tuple(np.asarray(current_angles) + difference)
        if np.min(np.diff(candidate_angles)) < 15:
            continue
        warm, candidate_weights = solve_at(
            candidate_angles, weights, start_id=start_id, step=step,
            tolerance=tolerance, max_iterations=max_iterations)
        warm["parent_angles_deg"] = current_angles
        warm["parent_loss"] = current_loss
        accepted = bool(warm["converged"] and warm["loss"] < current_loss)
        warm["accepted"] = accepted
        rows.append(warm)
        if accepted:
            current_angles = candidate_angles
            current_loss = warm["loss"]
            weights = candidate_weights
    return {"start_id": start_id, "rows": rows,
            "final_angles_deg": current_angles, "final_loss": current_loss}


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return {"count": int(array.size), "min": float(np.min(array)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "max": float(np.max(array))}


def reference_checks(results, case, sample_count):
    if sample_count <= 0:
        return []
    selected = np.linspace(0, len(results) - 1,
                           min(sample_count, len(results)), dtype=int)
    checks = []
    choices = []
    seen = set()

    def select(label, row):
        key = (row["start_id"], tuple(row["angles_deg"]))
        if key not in seen:
            seen.add(key)
            choices.append((label, row))

    for index in selected:
        result = results[int(index)]
        select("cold", result["rows"][0])
        select("final", next(row for row in reversed(
            result["rows"]) if row["accepted"]))
    # A tolerance miss is precisely where the accuracy reference matters.
    nonconverged = [row for result in results for row in result["rows"]
                    if not row["converged"]]
    for row in nonconverged[:8]:
        select("iteration_cap" if row["hit_iteration_cap"] else
               "nonconverged", row)

    for label, row in choices:
        try:
            reference = solve_subset(
                row["angles_deg"], case, backend="influence_slsqp")
            checks.append({
                "start_id": row["start_id"], "kind": label,
                "angles_deg": row["angles_deg"],
                "gpu_loss": row["loss"],
                "reference_loss": reference["loss"],
                "relative_loss_gap": row["loss"] / reference["loss"] - 1.0,
                "gpu_converged": row["converged"],
                "gpu_iterations": row["iterations"],
            })
        except RuntimeError as error:
            checks.append({"start_id": row["start_id"], "kind": label,
                           "angles_deg": row["angles_deg"],
                           "gpu_loss": row["loss"],
                           "reference_error": str(error)})
    return checks


def main():
    args = parse_args()
    if (args.starts <= 0 or args.local_steps < 0 or args.workers <= 0 or
            args.max_iterations <= 0 or args.reference_samples < 0 or
            not np.isfinite(args.gradient_tolerance) or
            args.gradient_tolerance <= 0):
        raise ValueError("invalid multistart benchmark settings")
    paths = generate_paths(args.starts, args.local_steps, args.seed)
    started = perf_counter()
    results = []
    with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=worker_initialize) as pool:
        futures = [pool.submit(run_start, path, args.gradient_tolerance,
                               args.max_iterations) for path in paths]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            cold = result["rows"][0]
            warm = result["rows"][1:]
            print(f"start {result['start_id']:2d}: cold {cold['iterations']:4d} "
                  f"iter/{cold['solve_seconds']:.2f}s, "
                  f"{sum(row['converged'] for row in warm)}/{len(warm)} "
                  f"warm converged, {sum(row['accepted'] for row in warm)} "
                  f"accepted", flush=True)
    parallel_work_seconds = perf_counter() - started
    results.sort(key=lambda result: result["start_id"])
    rows = [row for result in results for row in result["rows"]]
    cold = [row for row in rows if row["kind"] == "cold"]
    warm = [row for row in rows if row["kind"] == "warm"]
    checks = reference_checks(results, make_case(), args.reference_samples)
    total_seconds = perf_counter() - started
    summary = {
        "settings": {"starts": args.starts, "local_steps": args.local_steps,
                     "workers": args.workers, "seed": args.seed,
                     "gradient_tolerance": args.gradient_tolerance,
                     "max_iterations": args.max_iterations,
                     "reference_samples": args.reference_samples},
        "parallel_work_seconds": parallel_work_seconds,
        "total_seconds_including_references": total_seconds,
        "cold_solve_seconds": distribution([row["solve_seconds"] for row in cold]),
        "warm_solve_seconds": distribution([row["solve_seconds"] for row in warm]),
        "cold_iterations": distribution([row["iterations"] for row in cold]),
        "warm_iterations": distribution([row["iterations"] for row in warm]),
        "cold_converged": sum(row["converged"] for row in cold),
        "warm_converged": sum(row["converged"] for row in warm),
        "cold_hit_iteration_cap": sum(row["hit_iteration_cap"] for row in cold),
        "warm_hit_iteration_cap": sum(row["hit_iteration_cap"] for row in warm),
        "accepted_warm_moves": sum(row["accepted"] for row in warm),
        "reference_loss_gaps": distribution([
            row["relative_loss_gap"] for row in checks
            if "relative_loss_gap" in row]),
        "reference_checks": checks,
        "starts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("starts", "reference_checks")}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
