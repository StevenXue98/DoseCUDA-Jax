"""
Compare CUDA and JAX (class-based) IMPT dose calculations.
Runs both implementations on identical phantom/beam setup and reports differences.
"""
import os
import sys
import time
import numpy as np

# Add Jax folder to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
jax_dir = os.path.join(os.path.dirname(script_dir), "DoseCUDA", "Jax")
sys.path.insert(0, jax_dir)

from DoseCUDA import IMPTDoseGrid, IMPTPlan, IMPTBeam
from jax_impt import computeIMPTPlanJax


def create_test_plan():
    """Create identical plan and beam for both CUDA and JAX."""
    plan = IMPTPlan()
    beam = IMPTBeam()
    
    beam.dicom_rangeshifter_label = '0'
    n_spots = 98
    for energy_id in range(n_spots):
        theta = 2.0 * 3.14159 * energy_id / n_spots
        spot_x = 100.0 * np.cos(theta)
        spot_y = 100.0 * np.sin(theta)
        mu = 0.2
        beam.addSingleSpot(spot_x, spot_y, mu, energy_id)
    
    plan.addBeam(beam)
    return plan


def run_cuda_dose(dose, plan):
    """Run CUDA dose calculation and return numpy array."""
    dose.computeIMPTPlan(plan)
    return np.array(dose.dose)


def run_jax_dose(dose, plan):
    """Run JAX dose calculation and return numpy array."""
    return np.array(computeIMPTPlanJax(dose, plan))


def compare_doses(cuda_dose, jax_dose):
    """Compare two dose arrays and print statistics."""
    print("\n" + "="*60)
    print("DOSE COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nArray shapes: CUDA={cuda_dose.shape}, JAX={jax_dose.shape}")
    
    # Basic statistics
    print(f"\nCUDA dose - min: {cuda_dose.min():.6f}, max: {cuda_dose.max():.6f}, mean: {cuda_dose.mean():.6f}")
    print(f"JAX dose  - min: {jax_dose.min():.6f}, max: {jax_dose.max():.6f}, mean: {jax_dose.mean():.6f}")
    
    # Difference metrics
    diff = cuda_dose - jax_dose
    abs_diff = np.abs(diff)
    
    print(f"\nAbsolute difference:")
    print(f"  Max: {abs_diff.max():.6f}")
    print(f"  Mean: {abs_diff.mean():.6f}")
    print(f"  Std: {abs_diff.std():.6f}")
    
    # Relative difference (where dose > 0)
    nonzero_mask = cuda_dose > 1e-6
    if nonzero_mask.any():
        rel_diff = np.abs(diff[nonzero_mask]) / cuda_dose[nonzero_mask]
        print(f"\nRelative difference (where CUDA dose > 1e-6):")
        print(f"  Max: {rel_diff.max()*100:.4f}%")
        print(f"  Mean: {rel_diff.mean()*100:.4f}%")
        print(f"  Median: {np.median(rel_diff)*100:.4f}%")
    
    # Correlation
    if cuda_dose.std() > 0 and jax_dose.std() > 0:
        corr = np.corrcoef(cuda_dose.flatten(), jax_dose.flatten())[0, 1]
        print(f"\nPearson correlation: {corr:.8f}")
    
    # Voxel-level agreement
    tol_1pct = np.sum(abs_diff <= 0.01 * cuda_dose.max()) / cuda_dose.size * 100
    tol_5pct = np.sum(abs_diff <= 0.05 * cuda_dose.max()) / cuda_dose.size * 100
    print(f"\nVoxels within tolerance of max dose:")
    print(f"  Within 1%: {tol_1pct:.2f}%")
    print(f"  Within 5%: {tol_5pct:.2f}%")
    
    # Pass/fail summary
    print("\n" + "="*60)
    max_rel_diff = (abs_diff.max() / cuda_dose.max() * 100) if cuda_dose.max() > 0 else 0
    if max_rel_diff < 1.0:
        print("RESULT: PASS - Max difference < 1% of max dose")
    elif max_rel_diff < 5.0:
        print("RESULT: ACCEPTABLE - Max difference < 5% of max dose")
    else:
        print(f"RESULT: FAIL - Max difference = {max_rel_diff:.2f}% of max dose")
    print("="*60)
    
    return diff


def main():
    print("Setting up cube phantom...")
    
    # Create dose grids for CUDA and JAX
    dose_cuda = IMPTDoseGrid()
    dose_jax = IMPTDoseGrid()
    
    # Initialize identical phantoms
    dose_cuda.createCubePhantom()
    dose_jax.createCubePhantom()
    
    # Create identical plans
    plan_cuda = create_test_plan()
    plan_jax = create_test_plan()
    
    # Time CUDA dose calculation
    print("Running CUDA dose calculation...")
    start_time = time.perf_counter()
    cuda_result = run_cuda_dose(dose_cuda, plan_cuda)
    cuda_time = time.perf_counter() - start_time
    print(f"CUDA execution time: {cuda_time:.4f} seconds")
    
    # Time JAX dose calculation - first run includes JIT compilation
    print("\nRunning JAX dose calculation (first run - includes JIT compilation)...")
    start_time = time.perf_counter()
    jax_result = run_jax_dose(dose_jax, plan_jax)
    jax_first_time = time.perf_counter() - start_time
    print(f"JAX first run time (with JIT compilation): {jax_first_time:.4f} seconds")
    
    # Time JAX dose calculation - subsequent run (JIT cached)
    print("\nRunning JAX dose calculation (second run - JIT cached)...")
    dose_jax2 = IMPTDoseGrid()
    dose_jax2.createCubePhantom()
    plan_jax2 = create_test_plan()
    
    start_time = time.perf_counter()
    jax_result2 = run_jax_dose(dose_jax2, plan_jax2)
    jax_second_time = time.perf_counter() - start_time
    print(f"JAX second run time (JIT cached): {jax_second_time:.4f} seconds")
    
    # Print timing summary
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    print(f"CUDA execution time:           {cuda_time:.4f} seconds")
    print(f"JAX first run (with JIT):      {jax_first_time:.4f} seconds")
    print(f"JAX second run (JIT cached):   {jax_second_time:.4f} seconds")
    if jax_first_time > 0:
        print(f"JAX JIT compilation overhead:  {jax_first_time - jax_second_time:.4f} seconds")
    if cuda_time > 0:
        print(f"JAX speedup vs CUDA (cached):  {cuda_time / jax_second_time:.2f}x")
    print("="*60)
    
    # Compare results
    print("\n\nComparing CUDA vs Class-based JAX:")
    compare_doses(cuda_result, jax_result)


if __name__ == "__main__":
    main()
