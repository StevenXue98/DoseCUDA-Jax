# DoseCUDA

**NOT FOR CLINICAL USE**

**License**
This project is licensed under the GPL-2.0 License - see the [LICENSE](LICENSE) file for details.

**To use, please cite [our paper](http://doi.org/10.1002/acm2.70093):**
Bhattacharya M, Reamy C, Li H, Lee J, Hrinivich WT. A Python package for fast GPU‐based proton pencil beam dose calculation. Journal of Applied Clinical Medical Physics. 2025 Apr 11:e70093.

**DoseCUDA** is a Python package enabling GPU-based radiation dose calculation for research, development, and education. The package currently supports photon dose calculation using a collapsed cone convolution superposition algorithm and proton dose calculation using a double-Gaussian pencil beam algorithm. The default photon beam model corresponds to the 6 MV energy of a Varian Truebeam linear accelerator and the default proton beam model corresponds to a Hitachi Probeat synchrotron-based PBS delivery system with 98 discrete energies.

# Quickstart Guide

## Native Ubuntu environment for the GTX 1080 Ti

This repository includes a project-local Spack environment for CUDA 12.9 and
compute capability 6.1. It reuses the host CUDA installation at
`/usr/local/cuda-12.9`; Spack does not install or modify the NVIDIA driver.

Create the toolchain and Python environment from the repository root:

```bash
source "$HOME/.spack-src/share/spack/setup-env.sh"
spack -e . concretize
spack -e . install
./.spack-env/view/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-linux-lock.txt
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=61" \
  CUDACXX=/usr/local/cuda-12.9/bin/nvcc \
  PATH="$(pwd)/.spack-env/view/bin:/usr/local/cuda-12.9/bin:$PATH" \
  .venv/bin/python -m pip install --no-deps --editable .
```

Activate it in later shells and run the non-destructive reference validation:

```bash
source scripts/activate-dosecuda.sh
python tests/validate_cuda_jax_reference.py
```

The validation runs both implementations in memory and does not overwrite the
committed NRRD reference outputs. The CUDA golden is checked essentially
bit-for-bit; the legacy JAX golden is retained as a bounded historical drift
check because it was produced by the prototype with the indexing bugs fixed
in the canonical implementation.

For indexing and CUDA/JAX development, run the validation ladder from cheapest
to most comprehensive:

```bash
python -m unittest tests/test_indexing_contract.py
python tests/validate_noncubic_cuda_jax.py
python tests/validate_cuda_jax_matrix.py
python tests/validate_spot_weight_gradients.py
python tests/validate_cuda_spot_weight_vjp.py
python tests/validate_spot_position_gradients.py
python tests/validate_frozen_wet_angle_gradients.py
python tests/diagnose_full_angle_gradients.py
python tests/diagnose_angle_gradient_matrix.py
python tests/validate_wet_surrogate.py
python tests/validate_cuda_jax_reference.py
python tests/test_head_and_neck.py
```

All twelve are read-only by default. The CUDA/JAX matrix covers both non-cubic axis
orders, a heterogeneous oblique beam, and a two-beam fractionated plan. The
gradient validations compare JAX autodiff with central finite differences.
The JAX weight test also checks independent unit-spot dose responses. The CUDA
weight test validates the loss-agnostic reverse-mode product against linear and
quadratic finite differences, then recovers known nonnegative spot weights with
projected gradient descent. The position test holds WET and beam geometry fixed,
so it validates the smooth lateral pencil-beam path. The angle test rebuilds
beam geometry from traceable gantry and couch angles but holds WET fixed. The
full-angle diagnostic additionally recomputes ray tracing and WET smoothing. It
validates the final-dose gradient while reporting, but not hiding, the
discontinuous derivative introduced by rounded voxel selection during WET
smoothing. The angle-gradient matrix repeats that diagnostic at four beam
orientations in homogeneous and heterogeneous phantoms, including target and
OAR-style dose objectives.

For the first CUDA target/OAR optimization experiment, use a squared target
prescription error plus a squared OAR overdose penalty. The objective returns
its voxel-dose gradient to the existing CUDA spot-weight VJP; no dose kernel
or robust-scenario interface is needed. Run the CPU loss checks and the toy
phantom's full-iteration CUDA validation/profile with:

```bash
python -m unittest tests/test_cuda_target_oar_loss.py
python tests/profile_cuda_target_oar_optimization.py
```

The profile prints cached-WET setup, complete projected-gradient runtime,
objective-evaluation count, and timings for dose, Python loss, and weight VJP.
These timings include GPU execution and host transfers but do not separate
them. The same script then assembles a three-column unit-spot dose matrix as
an independent tiny-problem reference and compares a bounded CPU solve, the
CUDA projected-gradient result, and a bounded solve using the CUDA callback.
The matrix is only a validation oracle, not the production dose path. The masks
and dose limits are toy inputs, not a clinical prescription or validated
treatment plan.

For subsequent fixed-angle weight solves, `bounded_lbfgsb` accepts any
`value_and_gradient(weights)` callback, including
`FixedGeometryPlanDose.value_and_gradient(weights, loss)`. It supports a warm
start by passing the previous weight vector and returns the objective, weight
vector, iteration count, and projected-gradient norm. SciPy controls the
search on the CPU; CUDA still computes dose and the weight VJP. The existing
CUDA weight validation checks this path on a two-beam, five-spot case.

To scan one gantry angle with the rounded DoseCUDA forward model while
re-optimizing spot weights at every sampled angle:

```bash
python tests/scan_cuda_optimized_gantry.py
```

The default `three-structure` objective uses fully interior target and OAR
masks plus the remaining water phantom as normal tissue. It adds mean squared
target deviation and one-sided squared overdose penalties for OAR and normal
tissue, following the standard objective *form* used in
[matRad](https://matrad.readthedocs.io/en/latest/guide/planopt.html). For this
synthetic case, the target prescription is 0.50, the OAR limit is 0.30, the
normal-tissue limit is 0.50, and the three normalized term weights are 1.0.
These values are toy dose units, **not clinical prescriptions**. The old
boundary-straddling target/OAR case remains available with `--objective
legacy` for reproducing the original diagnostic. Neither case imposes
clinical spot deliverability constraints.

The default `covering15` spot set is a fixed 15-spot lattice (one energy ID,
five lateral x positions, three lateral y positions) covering the interior
toy target at the nominal angle. It is held unchanged as gantry angle varies;
only weights are re-optimized. Two additional trial energy layers were
discarded after all their optimized weights were zero in the 21–25 degree
diagnostic. The original three-spot gradient-validation
beam is still available with `--spot-set validation` and is selected
automatically with `--objective legacy`.

The script rebuilds ray tracing/WET at each angle, compares re-optimized
weights with nominal weights held fixed, and writes CSV and plots under the
ignored `test_phantom_output/optimized_gantry_scan/<objective>/<spot-set>/`
directory.
It is a small-phantom workflow diagnostic, not a treatment plan or validated
BAO objective. Points that do not pass the inner solver's stationarity check
are marked on the plot. The CSV reports target mean and D95 doses so a falling
objective is not mistaken for clinically acceptable coverage. The old
three-spot beam substantially undercovers the interior toy target.

For a reproducible **one-beam toy BAO baseline** with an active target/OAR
tradeoff, run:

```bash
python tests/run_toy_bao_baseline.py
```

This evaluates dose on the entire 24 × 28 × 32 grid (21,504 voxels). The
objective uses all 10,296 water/body voxels: 33 target, 33 OAR, and 10,230
other normal-tissue voxels. Air voxels are calculated but not penalized. Each
angle from −90° to 90° in 2° increments receives the same 45-spot,
three-energy beam's-eye-view lattice, shifted laterally to follow the target,
and an independently optimized nonnegative spot-weight vector. It reports the
best sampled angle, the nominal 23° result, and a nominal-angle no-OAR
ablation. CSV, a landscape plot, a JSON summary, and the selected spot
weights are saved under the
ignored `test_phantom_output/bao_toy_baseline/` directory. The inner-solver
stationarity check is 2e-4, reflecting float32 CUDA dose/gradient noise;
unresolved angles make the script fail rather than silently claiming an
exhaustive reference. "Exhaustive" means only exhaustive on this finite
angle grid and spot layout. The dose units, masks, and limits are synthetic,
not a clinical plan or an estimate of treatment quality.
Near the minimum, several neighboring angles are nearly tied; float32 CUDA
variation can change the exact 2° winner. The JSON reports all angles within
5% of the run's lowest loss, so the broad minimum is more informative than
the single best marker. OAR and normal-tissue limits are soft penalties, not
hard dose constraints.

To map the CUDA-compatible smoother's local BAO loss landscape and compare
autodiff slopes with dense forward evaluations:

```bash
python tests/diagnose_rounded_wet_landscape.py --smoother rounded
```

CSV data and plots are written under the ignored
`test_phantom_output/angular_landscape/` directory. This diagnostic varies
gantry and couch angles independently around representative smooth and
discontinuous cases; its target/OAR masks, prescription, spots, and weights
remain fixed within each scan.

An experimental BAO-specific WET smoother is available as a separate opt-in
path. It replaces rounded lateral voxel lookup with trilinear interpolation
and the hard radial cutoff with normalized sigmoid weights. The original
`compute_raytrace` function remains the CUDA-compatible default; call
`compute_raytrace_differentiable` explicitly to use the smooth model. Its
sigmoid transition width defaults to 0.25 mm and is an experimental numerical
parameter, not a calibrated machine parameter.

Compare its forward WET and dose with the rounded reference, then run the same
dense angular landscape diagnostic:

```bash
python tests/compare_wet_smoothers.py
python tests/diagnose_rounded_wet_landscape.py \
  --smoother differentiable --transition-width-mm 0.25
```

The smooth model intentionally does not reproduce CUDA exactly. The comparison
script reports that modeling drift separately from the CUDA/JAX regression
tests, so CUDA parity and BAO gradient behavior are not conflated.

A third experimental mode keeps the rounded forward WET and dose exactly while
substituting the differentiable smoother only at the WET backward boundary:

```bash
python tests/validate_wet_surrogate.py
python tests/diagnose_rounded_wet_landscape.py \
  --smoother surrogate --transition-width-mm 0.25
```

Call `compute_raytrace_surrogate` to select this behavior. Its custom gradient
is deliberately not the mathematical derivative of its returned rounded
forward values; it is an optimization heuristic that must be assessed by
recomputed rounded-loss descent. The validation checks that forward WET and
dose remain exactly equal to `compute_raytrace` and that the substituted WET
gradient equals `compute_raytrace_differentiable`.

To generate a fresh non-cubic ring visualization and explore arbitrary slices:

```bash
python -m pip install ipykernel ipywidgets  # once per environment
python tests/render_noncubic_comparison.py
```

Then open `tests/view_noncubic_indexing.ipynb` with the project `.venv` kernel.
Generated NRRDs and PNGs are isolated under
`test_phantom_output/noncubic_indexing/` and are not reference files.

## Prerequisites
Before installing DoseCUDA, ensure you have the following dependencies installed:
- **Python 3.6+**
- **CMake 3.15+**
- **CUDA Toolkit** (Ensure you have a compatible version for your GPU)
- **Git** (for cloning the repository)

Ensure that your GPU and CUDA drivers are properly set up before proceeding.

## Installation on Linux

Linux builds were tested on Ubuntu 20.04.4 LTS and Debian 12.

1. **Install Python** (if not already installed):
   ```
   sudo apt update
   sudo apt install python3 python3-pip
   ```
   
2. **(Optional but recommended)**: Create a virtual environment to isolate DoseCUDA and its dependencies:
   ```
   python3 -m venv dosecuda-env
   source dosecuda-env/bin/activate
   ```

3. **Clone the DoseCUDA repository**:
   ```
   git clone https://github.com/jhu-som-radiation-oncology/DoseCUDA
   cd DoseCUDA
   ```

4. **Install using pip**:
   ```
   python -m pip install .
   ```
   You may have to [override the host compiler](#additional-notes-linux) if yours is not supported.

5. **Verify the installation**:
   Run the test script to verify that DoseCUDA is installed correctly and working:
   ```
   python tests/test_phantom_impt.py
   ```
   or
   ```
   python tests/test_phantom_imrt.py
   ```

   If everything is installed correctly, you should see the dose calculation output in your terminal and files saved into `./test_phantom_output`.


6. **Deactivating the virtual environment** (if used):
   After using the package, you can deactivate the virtual environment:
   ```
   deactivate
   ```

## Additional Notes (Linux):
- **CUDA version**: Ensure that your CUDA version is compatible with your GPU and the CUDA Toolkit installed on your system.
- **Virtual environments**: Virtual environments help avoid conflicts between dependencies required by DoseCUDA and other Python projects you may have on your machine.
- **NVCC host compiler**: You can supply a host compiler override to pip by setting the environment variable `CUDAHOSTCXX` to the compiler command (e.g. `CUDAHOSTCXX=clang++`). Use this if your system's default is not supported.

## Installation on Windows

Windows builds have only been tested with Visual Studio 2022 on Windows 11 Enterprise Edition.

1. **Install Python** (if not already installed):
   - Download Python from the [official website](https://www.python.org/downloads/) and install it.
   - Make sure to check the option "Add Python to PATH" during the installation.

2. **Install MSVC**
   - Download Microsoft's Visual Studio installer from the [official website](https://visualstudio.microsoft.com/downloads/).
      - Install the x64/x86 build tools (C++ compiler and runtime libraries)
      - Install the CMake tools
      - (optional) Install Git for Windows

3. **Install the CUDA Toolkit**:
   - Download and install the appropriate version of the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).
   - Make sure the drivers for your GPU are compatible with this version.

4. **Install Git (if not installed through MSVC)**:
   - Download and install Git from the [official website](https://git-scm.com/download/win).
   - During the installation, choose "Git from the command line and also from 3rd-party software".

5. **Create a virtual environment** (optional but recommended):
   Open a command prompt (cmd) and run:
   ```cmd
   python -m venv dosecuda-env
   dosecuda-env\Scripts\activate
   ```

6. **Clone the DoseCUDA repository**:
   ```cmd
   git clone https://github.com/jhu-som-radiation-oncology/DoseCUDA
   cd DoseCUDA
   ```

7. **Install DoseCUDA using pip**:
   From a [**developer command prompt**](#additional-notes-windows):
   ```cmd
   python -m pip install .
   ```

8. **Verify the installation**:
   Run the test script to verify that DoseCUDA is installed correctly and working:
   ```cmd
   python tests\test_phantom_impt.py
   ```
   or
   ```
   python tests\test_phantom_imrt.py
   ```

   If everything is installed correctly, the dose calculation output should appear in the terminal, with files saved into `.\test_phantom_output`.

9. **Deactivate the virtual environment** (if used):
   After using the package, deactivate the virtual environment by typing:
   ```cmd
   deactivate
   ```

## Additional Notes (Windows):
- **PATH configuration**: Ensure Python, CMake, and CUDA are added to your system's PATH during their installations to avoid issues.
- **CUDA compatibility**: Ensure your GPU is compatible with the version of CUDA Toolkit you installed.
- **Developer command prompt**: Using a [developer command prompt](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell) ensures that all MSVC build tools are present in the environment's PATH. Use the `x64 Native Tools Command Prompt` for best results.

# Contact

For any questions or support regarding DoseCUDA, please reach out via email:
* **DoseCUDA@gmail.com**

# Developers

* Tom Hrinivich
* Calin Reamy
* Mahasweta Bhattacharya

# Funding Support
* The Commonwealth Fund

# JAX Implementation

Differentiable implementation of the PB algorithm in Jax.

## Array and coordinate convention

DoseCUDA uses the same convention as `SimpleITK.GetArrayFromImage`: CT, WET,
mask, and dose arrays have shape `(z, y, x)`. Physical metadata (`origin`,
`spacing`, and beam isocentres) is ordered `(x, y, z)`. `DoseGrid.size` is the
NumPy array shape, so it is also `(z, y, x)`. Do not transpose a SimpleITK
array when loading it into a dose grid or before writing a result with
`SimpleITK.GetImageFromArray`.

## Main Changes

1. **JAX Algorithm Implementation**
   - New DoseCUDA/Jax folder containing the JAX-based pencil beam algorithm
   - Canonical implementation: `DoseCUDA/Jax/impt_jax.py`

2. **Test Cases for Jax**
   - added test scripts for Jax in the folder tests:
     - JAX implementation tests
     - Numerical comparison between CUDA and JAX implementations
     - Jupyter notebook for dose visualization
   - Usage is similar to the original tests:
      ```cmd
      python tests\test_phantom_impt_jax.py
      ```
      or 
      ```cmd
      python tests\validate_cuda_jax_matrix.py
      ```

3. **Updated Dependencies**
   - added new package dependencies in pyproject.toml, with main addition being jax[cuda12]
   - Package can still be installed with original command with new Jax implementations:
      ```
      python -m pip install .
      ```

4. **CUDA Architecture Configuration**
   - Changed line 6 of CMakeLists.txt from `set(CMAKE_CUDA_ARCHITECTURES native)` to `CMAKE_CUDA_ARCHITECTURES 86` in order to successfully run on Lambda Cloud.
