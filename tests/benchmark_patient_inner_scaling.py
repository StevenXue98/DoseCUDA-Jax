#!/usr/bin/env python3
"""Bounded fixed-angle inner-solver scaling on matRad head-and-neck anatomy.

This is a throughput experiment, not a clinically validated treatment plan.
The same two beam angles, four energy layers, CT grid, structures, and loss
are used at every spot count. Only the nested lateral spot pattern grows.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from DoseCUDA import IMPTBeam, IMPTPlan
from DoseCUDA.gpu_influence_matrix import GPUInfluenceMatrixObjective
from DoseCUDA.impt_weight_optimization import (
    FixedGeometryPlanDose,
    InfluenceMatrixPlanDose,
    bounded_slsqp,
    make_target_oar_normal_tissue_loss,
)
from utils.matrad_converter import MatRadData, create_dosecuda_objects


ROOT = Path(__file__).resolve().parents[1]
ANGLES_DEG = (0.0, 180.0)
ENERGY_IDS = (25, 40, 55, 70)
SPOTS_PER_LAYER = (7, 13, 19)  # total spots: 56, 104, 152


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=3,
                        help="Run the first 1–3 nested spot counts")
    parser.add_argument("--max-matrix-gib", type=float, default=4.0)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--max-solve-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "test_phantom_output/bao_patient_inner_scaling/summary.json")
    args = parser.parse_args()
    if (args.levels not in (1, 2, 3) or args.max_matrix_gib <= 0 or
            args.max_iterations <= 0 or args.max_solve_seconds <= 0):
        parser.error("invalid benchmark limits")
    return args


def nested_spot_positions(count):
    positions = [(0.0, 0.0)]
    for radius in (15.0, 30.0, 45.0):
        for index in range(6):
            angle = index * np.pi / 3.0
            positions.append((radius * np.cos(angle),
                              radius * np.sin(angle)))
    return positions[:count]


def make_plan(center, spots_per_layer):
    plan = IMPTPlan(machine_name="HitachiProbeatJHU")
    model = plan.beam_models[0]
    if max(ENERGY_IDS) >= len(model.energy_labels):
        raise ValueError("selected energy IDs exceed the machine table")
    for angle in ANGLES_DEG:
        beam = IMPTBeam()
        beam.gantry_angle = angle
        beam.couch_angle = 0.0
        beam.iso = np.asarray(center, dtype=np.float32)
        beam.dicom_rangeshifter_label = "0"
        for energy_id in ENERGY_IDS:
            for x, y in nested_spot_positions(spots_per_layer):
                beam.addSingleSpot(x, y, 0.02, energy_id)
        plan.addBeam(beam)
    return plan


def load_case():
    patient = MatRadData("HEAD_AND_NECK")
    grid, structures = create_dosecuda_objects(patient, resample_spacing=3.0)
    def mask(name):
        result = structures[name]["mask"].astype(bool)
        if result.shape != grid.HU.shape:
            raise ValueError(f"{name} mask does not match the CT grid")
        return result
    target = mask("PTV70")
    oar = mask("BRAIN_STEM") & ~target
    normal = (grid.HU > -500.0) & ~target & ~oar
    if not (np.any(target) and np.any(oar) and np.any(normal)):
        raise ValueError("target/OAR/normal structures must be nonempty")
    return grid, patient.get_target_center("PTV70"), target, oar, normal


def save(output, summary):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    case_started = perf_counter()
    grid, center, target, oar, normal = load_case()
    case_load_seconds = perf_counter() - case_started
    n_voxels = int(np.prod(grid.HU.shape))
    summary = {
        "purpose": "patient-anatomy scaling, not a clinical-plan comparison",
        "patient": "matRad HEAD_AND_NECK",
        "ct_shape_zyx": list(map(int, grid.HU.shape)),
        "voxel_count": n_voxels,
        "case_load_seconds": case_load_seconds,
        "angles_deg": ANGLES_DEG,
        "energy_ids": ENERGY_IDS,
        "spot_counts": [len(ANGLES_DEG) * len(ENERGY_IDS) * value
                        for value in SPOTS_PER_LAYER[:args.levels]],
        "matrix_memory_limit_gib": args.max_matrix_gib,
        "solver_iteration_limit": args.max_iterations,
        "solver_time_limit_seconds": args.max_solve_seconds,
        "rows": [],
    }
    prescription = None
    for spots_per_layer in SPOTS_PER_LAYER[:args.levels]:
        count = len(ANGLES_DEG) * len(ENERGY_IDS) * spots_per_layer
        matrix_bytes = n_voxels * count * np.dtype("float64").itemsize
        gib = matrix_bytes / 2**30
        print(f"{count} spots: estimated dense matrix {gib:.2f} GiB", flush=True)
        if gib > args.max_matrix_gib:
            summary["rows"].append({"spot_count": count, "status": "memory_cap",
                                    "matrix_gib": gib})
            save(args.output, summary)
            continue
        row = {"spot_count": count, "matrix_gib": gib}
        started = perf_counter()
        operator = FixedGeometryPlanDose(grid, make_plan(center, spots_per_layer))
        row["geometry_seconds"] = perf_counter() - started
        if "energy_r80_mm" not in summary:
            model = operator.beams[0].beam_model
            summary["energy_r80_mm"] = [
                float(model.divergence_params[index, 1]) for index in ENERGY_IDS
            ]
            summary["target_wet_mm_quantiles_by_beam"] = [
                np.percentile(beam._wet.voxel_data[target] * 10.0,
                              [5, 50, 95]).tolist()
                for beam in operator.beams
            ]
        initial = operator.weights.astype(np.float64)
        started = perf_counter()
        initial_dose = operator.dose(initial)
        row["initial_forward_seconds"] = perf_counter() - started
        row["initial_target_mean_dose"] = float(np.mean(initial_dose[target]))
        if prescription is None:
            if row["initial_target_mean_dose"] <= 1.0e-8:
                raise RuntimeError("selected energies give negligible target dose")
            prescription = 2.0 * row["initial_target_mean_dose"]
            summary["synthetic_prescription"] = prescription
            summary["synthetic_oar_limit"] = 0.6 * prescription
            summary["synthetic_normal_limit"] = prescription
        loss = make_target_oar_normal_tissue_loss(
            target, prescription, oar, 0.6 * prescription,
            normal, prescription)
        row["initial_loss"] = loss(initial_dose)[0]
        started = perf_counter()
        influence = InfluenceMatrixPlanDose(
            operator, max_elements=n_voxels * count)
        row["matrix_build_seconds"] = perf_counter() - started
        print(f"  matrix built in {row['matrix_build_seconds']:.2f}s", flush=True)
        started = perf_counter()
        gpu = GPUInfluenceMatrixObjective(
            influence.matrix, influence.shape, target, oar, normal,
            prescription=prescription, oar_limit=0.6 * prescription,
            normal_limit=prescription)
        row["matrix_upload_seconds"] = perf_counter() - started
        probe_cpu = influence.value_and_gradient(initial, loss)
        probe_gpu = gpu.value_and_gradient(initial)
        row["probe_loss_relative_error"] = abs(probe_cpu[0] - probe_gpu[0]) / max(
            abs(probe_cpu[0]), 1.0e-12)
        row["probe_gradient_max_abs_error"] = float(np.max(np.abs(
            probe_cpu[1] - probe_gpu[1])))
        if row["probe_loss_relative_error"] > 1.0e-8:
            raise AssertionError("GPU and CPU influence losses disagree")
        started = perf_counter()
        def timed_callback(weights):
            if perf_counter() - started > args.max_solve_seconds:
                raise TimeoutError("GPU-matrix SLSQP exceeded time limit")
            return gpu.value_and_gradient(weights)
        try:
            result = bounded_slsqp(
                timed_callback, initial, weight_scale=0.02,
                objective_scale=1.0e4, max_iterations=args.max_iterations,
                stationarity_tolerance=1.0e-6)
            row["solve_seconds"] = perf_counter() - started
            row["iterations"] = result.iterations
            row["evaluations"] = result.evaluations
            row["converged"] = result.converged
            row["projected_gradient_norm"] = result.projected_gradient_norm
            row["loss"] = result.objective
            checked_dose = operator.dose(result.weights)
            checked_loss = loss(checked_dose)[0]
            row["fresh_cuda_loss_relative_error"] = abs(
                checked_loss - result.objective) / max(abs(checked_loss), 1.0e-12)
            row["final_target_mean_dose"] = float(np.mean(checked_dose[target]))
            row["final_target_d95"] = float(np.percentile(checked_dose[target], 5))
            row["final_oar_max_dose"] = float(np.max(checked_dose[oar]))
            row["active_spots"] = int(np.count_nonzero(result.weights > 1.0e-4))
            row["complete_seconds"] = sum(row[name] for name in (
                "geometry_seconds", "matrix_build_seconds",
                "matrix_upload_seconds", "solve_seconds"))
            row["status"] = "converged" if result.converged else "not_converged"
        except TimeoutError:
            row["solve_seconds"] = perf_counter() - started
            row["status"] = "time_cap"
        print(f"  {row['status']}: solve {row['solve_seconds']:.2f}s", flush=True)
        summary["rows"].append(row)
        save(args.output, summary)
        del gpu, influence, operator
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
