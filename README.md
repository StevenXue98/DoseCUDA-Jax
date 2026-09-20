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

To test an actual multistart *angle search* against that finite-grid answer
key, run the baseline above and then:

```bash
python tests/run_toy_bao_gaussian.py
```

The default starts are −20°, 23°, and 75°. At each step, eight antithetic
Gaussian perturbation pairs (2° standard deviation) estimate a direction
while holding the current optimized spot weights fixed. The candidate spot
lattice follows the target as the nominal angle changes; this is a search
parameterization, **not** a physical delivery-error scenario or SAM. A
backtracked angle move is accepted only after re-optimizing its weights and
checking that the *unsmoothed* DoseCUDA loss falls by more than 1e-5. The
ignored `test_phantom_output/bao_toy_gaussian/` directory contains a plot,
per-start histories, exact trial losses, call counts, and the winning weights.
This is a safeguarded heuristic, not a gradient of the bilevel loss and not a
global-optimum guarantee. In particular, a 4° smoothing scale can point
opposite the re-optimized loss slope around 35° and stop that start early;
`--sigma 4` reproduces the sensitivity check.

To test whether the same toy objective benefits from **jointly optimized
three-beam weights** before searching over three angles, run:

```bash
python tests/run_toy_three_beam_joint.py
```

This freezes three representative angle triples and solves all one-, two-,
and three-beam subsets with the same 45 candidate spots per beam and the
same full-water-phantom objective. All spot weights in each subset are
optimized together; no separate beam-share variables are introduced. The
ignored `test_phantom_output/bao_toy_three_beam/` directory contains the
subset losses, joint target D95 and OAR dose, per-beam target-dose shares,
active-spot counts, solver costs, a comparison plot, and the joint weights.
The experiment checks that adding an available beam does not worsen the
optimized loss beyond float32 solver variation. These are fixed-angle toy
plans, not a three-angle BAO search or a clinical treatment plan.

For the coupled **two-angle outer search**, run:

```bash
python tests/run_toy_two_beam_bao.py
```

The script jointly optimizes 90 weights at each evaluated angle pair. Its
10° diagnostic grid has 190 unique pairs after exchanging otherwise
identical beams is accounted for. Three starts, none placed at the grid's
observed best pair, use a two-component Gaussian fixed-weight direction;
accepted moves must lower the original CUDA loss after a fresh joint weight
solve. The search and grid have separate solve caches and call counts. A
heatmap with search paths, grid CSV, detailed search history, and winning
weights are saved under ignored `test_phantom_output/bao_toy_two_beam/`.
To change search settings without rebuilding the grid, pass
`--reuse-reference` (for example, with `--sigma 4`). The grid is coarse, not
a global-optimum certificate; fixed-weight probes use the candidate beam
layout and are not physical setup errors or SAM perturbations.

To view the saved two-angle map as a three-dimensional surface with the exact
search paths overlaid, without rerunning CUDA:

```bash
python tests/plot_toy_two_beam_landscape_3d.py
```

The PNG appears beside the heatmap. Its height is `log10(loss)` so both the
high-loss ridge and low-loss valleys remain visible. The plotted surface
connects measured grid points for visualization; intermediate surface values
are **not** additional dose calculations. The colored path heights are exact
jointly re-optimized losses. Use `--elev`, `--azim`, and `--output` to save
other viewing angles.

To reproduce the original exploratory **reference-only** 2° scan, run:

```bash
python tests/compute_toy_two_beam_reference.py \
  --inner-backend cuda_lbfgsb \
  --output-dir test_phantom_output/bao_toy_two_beam_2deg \
  --search-summary test_phantom_output/bao_toy_two_beam/summary.json
python tests/plot_toy_two_beam_landscape_3d.py \
  --input-dir test_phantom_output/bao_toy_two_beam_2deg
```

The first command solves the coupled 90-weight inner problem at every one of
the 4,186 unique unordered pairs on a 2° grid. It does **not** rerun the
outer angle search: the existing search paths are only overlaid on its new
2D and 3D plots. The fine CSV, plots, and JSON summary are saved in the
ignored `test_phantom_output/bao_toy_two_beam_2deg/` directory, leaving the
10° results untouched. An atomic `reference_checkpoint.json` is written
every ten pairs; rerunning the same command resumes an interrupted scan.
The minimum is a sampled-grid reference, not a continuous global-optimum
certificate. Each grid point uses one inner solve. The comparison below found
that alternative weight initializations can change the solved loss by several
percent even when the existing projected-gradient criterion passes, so narrow
peaks and the exact ranking of nearby grid points need additional checking.

For a start-stable small-plan reference, use the separate accurate backend:

```bash
python tests/validate_toy_two_beam_inner_accuracy.py
python tests/compute_toy_two_beam_reference.py
python tests/plot_toy_two_beam_landscape_3d.py \
  --input-dir test_phantom_output/bao_toy_two_beam_2deg_accurate
python tests/plot_toy_inner_solver_discrepancy.py
```

This builds one unit-spot dose column per spot using DoseCUDA, then solves the
90-weight convex problem in float64 with bounded SQP. It checks the resulting
plan with a fresh DoseCUDA forward pass. The old CUDA-callback L-BFGS-B solver
remains available; the new backend is explicitly selected and is intended for
small reference cases because its full influence matrix does not scale to a
large treatment plan. The accurate grid uses its own resumable checkpoint and
does not overwrite the earlier 2° scan. No legacy search paths are overlaid,
since their loss heights came from the less accurate inner solver.
The final command compares the old and accurate grid values pair by pair and
plots where inner-solver error altered the landscape.

For a separate **matrix-free float32 CUDA inner-solver benchmark**, run:

```bash
python tests/benchmark_cuda_inner_solvers.py \
  --max-iterations 500 --gradient-tolerance 1e-5 --reference
python tests/benchmark_cuda_inner_solvers.py \
  --methods fista --max-iterations 2000 --gradient-tolerance 1e-5 \
  --reference --warm-start-check
```

The experimental `DoseCUDA.cuda_weight_solver.solve_cuda_spot_weights` path
keeps fixed-angle WET, dose, and weight-gradient buffers on the GPU, calls the
existing pencil-beam forward and weight-VJP kernels, and builds **no** dose
influence matrix. It benchmarks projected gradient, accelerated projected
gradient (FISTA), projected nonlinear conjugate gradient, and a host-assisted
L-BFGS variant with persistent GPU physics buffers. The CPU/SQP reference is
unchanged and is used only to measure loss gaps. The benchmark checks each
reported GPU loss with a fresh DoseCUDA forward pass. A warm-start experiment
reuses weights from one three-beam angle triple at a nearby triple, as an
actual outer BAO step could do. These are experimental solver candidates for
the synthetic objective, not clinically validated plans or replacements for
the accurate reference. In particular, a loose projected-gradient threshold
can still leave a large inner-loss gap on this ill-conditioned problem. The
projected methods keep their weight vectors on-device but still synchronize
small loss/line-search scalars with the host; this is not a zero-transfer
implementation. Use `--output` to save a JSON benchmark record under an
ignored output directory.

For a parallel **GPU-resident influence-matrix** inner solve and complete
cold/warm comparison, run:

```bash
python -m unittest discover -s tests -p test_gpu_influence_matrix.py
python tests/benchmark_gpu_influence_inner.py
```

`DoseCUDA.gpu_influence_matrix.solve_gpu_influence_weights` builds one
unit-spot dose column per spot with the unchanged DoseCUDA forward model,
uploads the fixed-angle matrix once, and uses float64 cuBLAS products for
`D @ weights` and `D.T @ dose_gradient`. SciPy SLSQP still controls the
small weight vector on the CPU; only that vector, its gradient, and the loss
cross the CPU/GPU boundary per callback. This is intentionally separate from
the matrix-free path. The benchmark counts geometry/WET setup, column build,
GPU upload, and solve time, checks the final loss with fresh DoseCUDA dose,
and compares against the existing accurate CPU matrix reference. The dense
matrix is suitable for this small synthetic case; memory and construction
cost must be reassessed for realistic voxel/spot counts.

To compare the earlier nominal continuous two-angle searches on the exploratory
map, run:

```bash
python tests/compare_toy_two_beam_outer_methods.py
```

This reuses the original 2° grid without rebuilding it. It still uses the
old CUDA-callback inner solver, so its method rankings are exploratory rather
than rankings against the accurate reference above. Six common starts compare
fixed-weight Gaussian directions (24 antithetic pairs, σ = 2°) with central
finite differences of the fully re-optimized loss (steps of 2° and 4°).
Every proposed move is accepted only after a fresh joint 90-weight solve
lowers the original CUDA loss. All methods use the same per-start time ceiling;
actual wall time and dose/inner-solve counts are reported because a search
may stop early. Three differently initialized inner solves at four grid
locations estimate optimizer variability, and each search endpoint is
rechecked from two additional weight starts. Results and path/cost plots go
to ignored `test_phantom_output/bao_toy_two_beam_outer_comparison_24pairs/`.
The comparison is a nominal optimizer diagnostic, not SAM or a robustness
result. Differences comparable to the repeat-solve spread are inconclusive.

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

## Patient-anatomy inner-solver scaling benchmark

From the repository root, after activating the Linux project environment, run
`python -m tests.benchmark_patient_inner_scaling`. It uses the checked-in matRad
head-and-neck CT and masks, two fixed beams, and nested 56/104/152-spot plans.
The CUDA dose engine builds a dense influence matrix; the GPU evaluates its
loss and gradient; CPU SLSQP chooses weights. The script records geometry,
matrix construction/upload, optimization, convergence, and a fresh DoseCUDA
loss check under `test_phantom_output/bao_patient_inner_scaling/summary.json`.
Its normalized target/brainstem/normal-tissue objective and spot layout are
synthetic: this is a scaling test on patient anatomy, **not** a clinical-plan
quality benchmark or a like-for-like comparison to matRad timing.

## Coarse prostate nominal-plan experiment

From the activated environment, run
`python -m tests.run_prostate_nominal_plan`. This uses the checked-in matRad
prostate CT and structures on a 5 mm grid, two fixed lateral fields,
target-projected spots, and machine energy layers selected from target WET.
The relative research objective includes the two PTV prescriptions and
rectum, bladder, and remaining body penalties. Target voxels take priority
over overlapping OAR voxels. It is **not** matRad's exact objective or a
clinically calibrated plan. Only scored voxels are retained in the GPU
influence matrix; the final dose and loss are independently recomputed by
the original DoseCUDA forward model on the full grid. Runtime and matrix
memory are capped. Scaled L-BFGS-B is the default inner solver; `--solver slsqp`
allows a slower comparison. Results go to ignored
`test_phantom_output/bao_prostate_nominal/summary.json` (with a matching `.npz`
archive of optimized weights, spots, and full-grid dose). Coverage and OAR
metrics must be inspected separately from numerical solver convergence.

## Partial inner solves and angle signals

`python -m tests.benchmark_partial_inner_angle` uses the saved prostate
`summary.npz` as a warm start, moves each of the two beams locally in a
separate case, and retains each beam's spot positions, energy IDs, and weight
indices. It compares matrix and matrix-free calls for the **same** five-region
loss, records 0/1/5/10/20 accepted L-BFGS-B weight iterates, and computes
fixed-weight central angular secants against a converged-weight reference.
The matrix build, both partial trajectories, full reference solve, and angle
probes are timed separately. These secants are not derivatives of the rounded
DoseCUDA model, and this script does not implement alternating BAO. Output is
ignored under `test_phantom_output/bao_partial_inner/`; each case also saves
an `.npz` archive of reference and partial-step weights for subsequent angle
step experiments.

## Alternating two-beam toy BAO

`python tests/run_toy_two_beam_alternating.py` is the original joint
coordinate-descent diagnostic on the full-voxel, 90-spot toy case. It changes
angles while holding old weights fixed for acceptance, then adjusts weights
at the new angle. This is **not** a fair partial-inner approximation to nested
BAO, because an otherwise good nominal angle may require different weights.
Its ignored outputs remain under `test_phantom_output/bao_toy_two_beam_alternating/`
for that diagnostic purpose.

For the corrected matched-start nominal BAO comparison, run
`python tests/compare_toy_two_beam_inner_schedules.py`. Each of the three
preselected angle pairs starts four paths: fixed 20, 50, or 100 matrix-free
weight steps per candidate angle, plus a reference method that fully solves spot
weights at every proposed angle before deciding whether to move. Methods at
the same start and outer cycle share Gaussian probe vectors and line-search
settings. **Every proposed angle gets its own weight solve before acceptance**:
20/50/100 steps in the partial paths or a full inner solve in the reference.
If all partial trial angles fail, the incumbent angle and its saved weights
are restored, the incumbent weight solve is extended toward 50/100/200 steps,
and the angle search is retried. Accepted candidates carry their optimized
weights forward without an immediate redundant inner solve. Every scored
loss is checked by the original DoseCUDA forward model. Retrospective full
solves score every visited angle on one common scale; they do not influence
partial-method decisions, and their time is reported separately. Outputs
are ignored under `test_phantom_output/bao_toy_two_beam_alternating/candidate_refit_schedules/`.
This remains a stochastic nonconvex toy search, not a clinical plan or a
one-seed performance ranking.

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
