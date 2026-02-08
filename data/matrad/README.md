# matRad Data Files

This directory contains patient phantom data from **matRad** - an open source 
treatment planning system for radiation therapy research.

## Source

These files are copied from the matRad repository:
- Repository: https://github.com/e0404/matRad
- License: GNU General Public License v3.0
- Citation: 
  ```
  Wieser HP, Cisternas E, Wahl N, Ulrich S, Stadler A, Deber H, et al. 
  Development of the open-source dose calculation and optimization toolkit 
  matRad. Med Phys. 2017;44(6):2556-2568. doi:10.1002/mp.12251
  ```

## Phantoms (data/matrad/phantoms/)

| File | Description | CT Size | Structures |
|------|-------------|---------|------------|
| HEAD_AND_NECK.mat | Head & neck patient | 161×161×67 | 23 (CTV63, PTV70, brainstem, etc.) |
| PROSTATE.mat | Prostate patient | 183×183×90 | 10 (PTV_68, rectum, bladder, etc.) |
| LIVER.mat | Liver patient | 217×217×168 | 26 (GTV, liver, kidneys, etc.) |
| BOXPHANTOM.mat | Simple box phantom | 160×160×160 | 2 (BODY, OuterTarget) |

## Test Data (data/matrad/testdata/)

Pre-computed treatment planning data for testing:

| File | Description | Contents |
|------|-------------|----------|
| protons_testData.mat | Proton plan test case | ct, cst, pln, stf, dij, resultGUI |
| carbon_testData.mat | Carbon ion test case | ct, cst, pln, stf, dij, resultGUI |
| helium_testData.mat | Helium ion test case | ct, cst, pln, stf, dij, resultGUI |

## Data Structure

Each phantom `.mat` file contains:
- `ct.cubeHU`: 3D HU array (MATLAB x,y,z ordering)
- `ct.resolution`: Voxel spacing {x, y, z} in mm
- `ct.x, ct.y, ct.z`: Coordinate arrays
- `cst`: Structure set (N×6 cell array)
  - Column 0: Structure index
  - Column 1: Structure name
  - Column 2: Type ('TARGET' or 'OAR')
  - Column 3: Voxel indices (1-based, Fortran order)
  - Column 4: Parameters
  - Column 5: Optimization objectives

## Usage

```python
from utils.matrad_converter import MatRadData, MATRAD_PHANTOMS

# Load HEAD_AND_NECK phantom
data = MatRadData('HEAD_AND_NECK')

# Get CT info
print(data.cubeHU.shape)
print(data.get_target_names())

# Get target center for isocenter
center = data.get_target_center('CTV63')
```

See `utils/matrad_converter.py` for full API documentation.
