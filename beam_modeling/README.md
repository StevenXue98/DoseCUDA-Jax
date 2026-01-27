# Proton Pencil Beam Modeling

The modeling process currently uses data from pristine Bragg peaks computed using an existing pencil beam model to build lookup tables for each proton beam energy, which are saved as `.csv` files.

## Overview

Each nominal beam energy is modeled using the following three components:
1. **IDD (Integrated Depth Dose) vs. z_w**: A lookup table representing dose deposition as a function of depth in water.
2. **σ_mcs (Multiple Coulomb Scattering) vs. z_w**: A lookup table representing lateral spot broadening due to scattering as a function of depth in water.
3. **σ_air (Air Scatter)**: A quadratic function describing the lateral spread of the proton beam as a function of distance from the effective proton source.

These components were modeled based on pristine Bragg peaks computed using Monte Carlo (MC) simulations in a clinical Treatment Planning System (TPS), RayStation 2023B (RaySearch Labs, Stockholm, Sweden). The default model was configured to match the synchrotron-based Probeat PBS delivery system (Hitachi, Tokyo, Japan), with 98 discrete energies ranging from 70.2 MeV to 228.7 MeV.

## Requirements

- Existing planning system or beam model capable of exporting physical dose grids.
- Digital water phantom - note that the lookup table script assumes that the material is water (i.e. RLSP = 1.0). It is recommended to override the phantom with water to ensure this is the case. 
- Python 3.x
- Numpy, Pandas for data processing
- Matplotlib for plotting (optional)

## Modeling Workflow

### Pristing Bragg Peak Calculation

   - Pristine Bragg peaks were generated for each energy using a digital water phantom in the TPS with 1 mm isotropic dose grid, 0.5% statistical uncertainty, and 0.2 MU per spot.
   - For each energy, Bragg peaks were computed at multiple effective SSDs (isocenter positions) for σ_air fitting.
   - ```impt_compute_spots.py``` is a Python script that can be run in RayStation to automate the dose calcualation and physical dose grid export for a list of energies. Note that this script needs to be edited to match the digital phantom, plan, and energy list for your beam model. 

### Look Up Table Generation

1. **IDD Calculation**:
   - The dose grids were analyzed to compute the Integrated Depth Dose (IDD) as a function of depth (z_w) at 1 mm resolution.
   - The IDD was normalized by the delivered MUs to create lookup tables with units of `cGy mm² / MU`, inherently accounting for reference dose calibration.
   - The R80 (depth at which 80% of the peak dose is reached) was derived from the final IDD for each energy.

2. **Lateral Spot Profile Calculation**:
   - The lateral spot profiles were derived from the pristine Bragg peak cross-sections at 1 mm depth increments.
   - A 2D Gaussian distribution was fitted to compute the lateral beam spread (σ_c) as a function of depth for each beam energy and source-to-surface distance (SSD).
   - A reference depth (z_(w,ref)) of 0.7 * R80 was defined for each energy to improve lateral spot profile accuracy near the Bragg peak.

3. **Air Scatter (σ_air) Fitting**:
   - The lateral spot profile at z_(w,ref) was used to fit a quadratic function representing σ_air as a function of distance to the proton source for each energy.
   - The fitting was performed across four SSDs: 163.5 cm, 153.5 cm, 143.5 cm, and 138.5 cm, corresponding to typical clinical setups.

4. **Multiple Coulomb Scattering (σ_mcs) Fitting**:
   - The lookup table for σ_mcs was computed by averaging the difference between σ_c at different depths and σ_c at the reference depth (z_(w,ref)).

### Running the Lookup Table Script

1. Ensure the TPS dose grids for all energies are exported to a designated directory.
2. Edit `energies.csv` to contain the energy labels and complete path to the directories containing the physical dose grids exported from RayStation in the `dcm_directory` column.
3. Run the script to generate the IDD lookup tables, σ_mcs, and σ_air parameters as `.csv` files.

```bash
python impt_lookup_tables.py "directory-containing-energies.csv"
```
4. The results, including the updated `energies.csv` file should be placed in `DoseCUDA/lookuptables/`.
5. Note that the distance from isocenter to the x and y scanning magnets are defined in `dose_kernels/model_params.h` and should also be updated to match your machine before re-building DoseCUDA with new lookup tables. 
