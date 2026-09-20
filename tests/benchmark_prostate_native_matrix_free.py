#!/usr/bin/env python3
"""Bounded, matrix-free DoseCUDA feasibility check on native 3 mm PROSTATE CT.

Reuses the two-field, 3410-spot 5 mm research template and weights. This is a
throughput diagnostic, not a clinical plan or an optimized full-resolution BAO
result. No influence matrix is constructed and no source data are modified.
"""

import argparse
import json
import resource
import subprocess
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from DoseCUDA.impt_weight_optimization import FixedGeometryPlanDose
from tests.benchmark_partial_inner_angle import make_operator, masks_and_losses
from utils.matrad_converter import MatRadData, create_dosecuda_objects


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "test_phantom_output/bao_prostate_nominal/summary.npz"
OUTPUT = ROOT / "test_phantom_output/bao_prostate_native_matrix_free/summary.json"
ANGLES = (90.0, 270.0)
PROBES = ((92.0, 270.0), (88.0, 270.0),
          (90.0, 272.0), (90.0, 268.0))


def gpu_used_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5)
    line = result.stdout.strip().splitlines()[0]
    used, free = (int(part.strip()) for part in line.split(","))
    return used, free


class GPUMemorySampler:
    def __init__(self, period=0.3):
        self.period = period
        self.peak_mib = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                used, _ = gpu_used_mib()
                self.peak_mib = used if self.peak_mib is None else max(
                    self.peak_mib, used)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                self.error = str(exc)
                return
            self.stop_event.wait(self.period)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=6)


def save(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-total-seconds", type=float, default=240.0)
    parser.add_argument("--max-solver-seconds", type=float, default=120.0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=2048)
    parser.add_argument("--max-used-gpu-mib", type=int, default=9500)
    parser.add_argument("--max-iterations", type=int, default=20)
    args = parser.parse_args()
    if (args.max_total_seconds <= 0 or args.max_solver_seconds <= 0 or
            args.max_iterations < 5 or args.max_iterations > 100 or
            args.min_free_gpu_mib < 0 or args.max_used_gpu_mib <= 0):
        parser.error("invalid benchmark limits")
    return args


def partial_solve(operator, loss, initial, *, max_iterations, deadline,
                  sampler, max_used_gpu_mib):
    """Warm-start L-BFGS-B for at most max_iterations accepted steps."""
    scale = float(np.median(initial[initial > 0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("the archived plan needs positive initial weights")
    snapshots = {0: initial.copy()}
    checkpoints = {0: 0.0}
    evaluations = 0
    accepted = 0
    started = perf_counter()

    def objective(trial):
        nonlocal evaluations
        if perf_counter() > deadline:
            raise TimeoutError("benchmark time cap reached during weight solve")
        if (sampler.peak_mib is not None and
                sampler.peak_mib > max_used_gpu_mib):
            raise MemoryError("sampled GPU memory cap reached")
        value, gradient = operator.value_and_gradient(scale * trial, loss)
        evaluations += 1
        return 100.0 * float(value), 100.0 * scale * np.asarray(gradient)

    def capture(trial):
        nonlocal accepted
        accepted += 1
        iteration = accepted
        if iteration in (5, max_iterations):
            snapshots[iteration] = (scale * trial).copy()
            checkpoints[iteration] = perf_counter() - started
        if perf_counter() > deadline:
            raise TimeoutError("benchmark time cap reached after weight step")

    result = minimize(
        objective, initial / scale, jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * len(initial), callback=capture,
        options={"maxiter": max_iterations, "gtol": 1e-12,
                 "ftol": 1e-16, "maxls": 40},
    )
    final = np.maximum(scale * result.x, 0.0)
    snapshots[int(result.nit)] = final.copy()
    checkpoints[int(result.nit)] = perf_counter() - started
    return snapshots, checkpoints, {
        "iterations": int(result.nit), "evaluations": evaluations,
        "seconds": perf_counter() - started,
        "success": bool(result.success), "message": str(result.message),
    }


def main():
    args = parse_args()
    started = perf_counter()
    baseline_used, baseline_free = gpu_used_mib()
    if baseline_free < args.min_free_gpu_mib:
        raise RuntimeError(f"only {baseline_free} MiB GPU memory free")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    with np.load(args.archive) as archive:
        weights = np.asarray(archive["weights"], dtype=np.float64)
        spot_lists = (archive["beam_0_spots"].copy(),
                      archive["beam_1_spots"].copy())
    if len(weights) != sum(len(spots) for spots in spot_lists):
        raise ValueError("archived weights and spot lists disagree")
    summary = {
        "purpose": "native-resolution patient-anatomy throughput; not a clinical plan",
        "case": "matRad PROSTATE", "grid_spacing_mm": 3.0,
        "angles_deg": ANGLES, "spot_count": len(weights),
        "matrix": "none; original matrix-free DoseCUDA forward/VJP",
        "gpu_baseline_used_mib": baseline_used,
        "gpu_baseline_free_mib": baseline_free,
        "caps": {"max_total_seconds": args.max_total_seconds,
                 "max_solver_seconds": args.max_solver_seconds,
                 "max_used_gpu_mib": args.max_used_gpu_mib,
                 "max_iterations": args.max_iterations},
        "status": "preparing",
    }
    save(args.output, summary)

    def check_budget(sampler):
        if perf_counter() - started > args.max_total_seconds:
            raise TimeoutError("benchmark total time cap reached")
        if (sampler.peak_mib is not None and
                sampler.peak_mib > args.max_used_gpu_mib):
            raise MemoryError("sampled GPU memory cap reached")

    with GPUMemorySampler() as sampler:
        try:
            step_started = perf_counter()
            patient = MatRadData("PROSTATE")
            if not np.allclose(patient.spacing, 3.0):
                raise ValueError("expected native 3 mm isotropic PROSTATE CT")
            grid, structures = create_dosecuda_objects(
                patient, resample_spacing=3.0)
            summary["load_seconds"] = perf_counter() - step_started
            summary["grid_shape_zyx"] = list(map(int, grid.HU.shape))
            summary["voxel_count"] = int(grid.HU.size)
            high, rectum, body, selected, loss, _ = masks_and_losses(structures)
            summary["structure_voxels"] = {
                "PTV_68": int(high.sum()), "Rectum_exclusive": int(rectum.sum()),
                "BODY_exclusive": int(body.sum())}
            summary["scored_voxel_count"] = int(selected.sum())
            summary["estimated_dense_float64_matrix_gib"] = (
                int(selected.sum()) * len(weights) * 8 / 2**30)
            check_budget(sampler)
            save(args.output, summary)

            step_started = perf_counter()
            operator = make_operator(grid, patient.get_target_center("PTV_68"),
                                     spot_lists, ANGLES)
            if not isinstance(operator, FixedGeometryPlanDose):
                raise TypeError("expected matrix-free fixed-geometry operator")
            summary["geometry_seconds"] = perf_counter() - step_started
            check_budget(sampler)
            save(args.output, summary)

            step_started = perf_counter()
            dose = operator.dose(weights)
            summary["first_forward_seconds"] = perf_counter() - step_started
            summary["first_loss"] = float(loss(dose)[0])
            summary["first_target_d95"] = float(np.percentile(dose[high], 5))
            del dose
            check_budget(sampler)

            forward_times = []
            callback_times = []
            for _ in range(3):
                step_started = perf_counter()
                dose = operator.dose(weights)
                forward_times.append(perf_counter() - step_started)
                del dose
                step_started = perf_counter()
                value, gradient = operator.value_and_gradient(weights, loss)
                callback_times.append(perf_counter() - step_started)
                if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                    raise ValueError("nonfinite loss/weight gradient")
                check_budget(sampler)
            summary["forward_seconds"] = forward_times
            summary["forward_vjp_loss_seconds"] = callback_times
            summary["forward_vjp_loss_median_seconds"] = float(
                np.median(callback_times))
            save(args.output, summary)
            print(f"Native grid {grid.HU.shape}, {len(weights)} spots: "
                  f"geometry {summary['geometry_seconds']:.2f}s, "
                  f"forward/VJP median {np.median(callback_times):.3f}s",
                  flush=True)

            deadline = min(started + args.max_total_seconds,
                           perf_counter() + args.max_solver_seconds)
            snapshots, checkpoint_times, solve = partial_solve(
                operator, loss, weights, max_iterations=args.max_iterations,
                deadline=deadline, sampler=sampler,
                max_used_gpu_mib=args.max_used_gpu_mib)
            summary["weight_solve"] = solve
            summary["weight_checkpoints"] = []
            for iteration in sorted(snapshots):
                check_budget(sampler)
                trial_dose = operator.dose(snapshots[iteration])
                summary["weight_checkpoints"].append({
                    "iteration": iteration,
                    "solve_elapsed_seconds": checkpoint_times[iteration],
                    "exact_loss": float(loss(trial_dose)[0]),
                    "PTV_68_d95": float(np.percentile(trial_dose[high], 5)),
                    "active_spots": int(np.count_nonzero(
                        snapshots[iteration] > 1e-4)),
                })
            save(args.output, summary)

            final_weights = snapshots[solve["iterations"]]
            summary["angle_probes"] = []
            for angles in PROBES:
                check_budget(sampler)
                step_started = perf_counter()
                probe = make_operator(grid, patient.get_target_center("PTV_68"),
                                      spot_lists, angles)
                geometry_seconds = perf_counter() - step_started
                step_started = perf_counter()
                value = float(loss(probe.dose(final_weights))[0])
                forward_loss_seconds = perf_counter() - step_started
                summary["angle_probes"].append({
                    "angles_deg": angles, "geometry_seconds": geometry_seconds,
                    "forward_loss_seconds": forward_loss_seconds,
                    "exact_fixed_weight_loss": value,
                })
                del probe
                save(args.output, summary)
            summary["status"] = "complete"
        except (TimeoutError, MemoryError) as exc:
            summary["status"] = "time_cap" if isinstance(exc, TimeoutError) else "memory_cap"
            summary["stop_reason"] = str(exc)
        finally:
            summary["elapsed_seconds"] = perf_counter() - started
            summary["gpu_peak_sampled_mib"] = sampler.peak_mib
            summary["gpu_sampler_error"] = sampler.error
            summary["host_peak_rss_mib"] = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024.0
            save(args.output, summary)
    print(f"Status {summary['status']}; elapsed {summary['elapsed_seconds']:.1f}s; "
          f"sampled GPU peak {summary['gpu_peak_sampled_mib']} MiB; "
          f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
