"""
matrad_converter.py - Convert matRad .mat phantoms to DoseCUDA format

matRad (.mat) files contain:
  - ct.cubeHU: 3D HU array (x, y, z) in MATLAB order
  - ct.resolution: {x, y, z} spacing in mm
  - ct.x, ct.y, ct.z: coordinate arrays  
  - cst: (N, 6) array with structure info:
      [0]: index, [1]: name, [2]: type (TARGET/OAR), 
      [3]: voxel indices, [4]: parameters, [5]: objectives

DoseCUDA expects:
  - HU array in (z, y, x) SimpleITK order
  - origin in mm
  - spacing (isotropic preferred)
  - beam geometry: gantry_angle, couch_angle, iso
  - spot_list: [x, y, mu, energy_id]
"""

import os
import scipy.io as sio
import numpy as np
import SimpleITK as sitk


# =============================================================================
# Data paths - relative to this file's location
# =============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'data', 'matrad')

MATRAD_PHANTOMS_DIR = os.path.join(_DATA_DIR, 'phantoms')
MATRAD_TESTDATA_DIR = os.path.join(_DATA_DIR, 'testdata')

# Available phantoms (name -> filename mapping)
MATRAD_PHANTOMS = {
    'HEAD_AND_NECK': 'HEAD_AND_NECK.mat',
    'PROSTATE': 'PROSTATE.mat',
    'LIVER': 'LIVER.mat',
    'BOXPHANTOM': 'BOXPHANTOM.mat',
}

# Available test data
MATRAD_TESTDATA = {
    'protons': 'protons_testData.mat',
    'carbon': 'carbon_testData.mat',
    'helium': 'helium_testData.mat',
}


def get_phantom_path(name):
    """Get full path to a phantom file.
    
    Args:
        name: Phantom name (e.g., 'HEAD_AND_NECK', 'PROSTATE') or full path
        
    Returns:
        Full path to the .mat file
    """
    if os.path.isfile(name):
        return name
    
    if name in MATRAD_PHANTOMS:
        return os.path.join(MATRAD_PHANTOMS_DIR, MATRAD_PHANTOMS[name])
    
    # Try adding .mat extension
    mat_name = name if name.endswith('.mat') else f'{name}.mat'
    path = os.path.join(MATRAD_PHANTOMS_DIR, mat_name)
    if os.path.isfile(path):
        return path
    
    raise FileNotFoundError(f"Phantom '{name}' not found. Available: {list(MATRAD_PHANTOMS.keys())}")


def get_testdata_path(name):
    """Get full path to a test data file.
    
    Args:
        name: Test data name (e.g., 'protons', 'carbon') or full path
        
    Returns:
        Full path to the .mat file
    """
    if os.path.isfile(name):
        return name
    
    if name in MATRAD_TESTDATA:
        return os.path.join(MATRAD_TESTDATA_DIR, MATRAD_TESTDATA[name])
    
    raise FileNotFoundError(f"Test data '{name}' not found. Available: {list(MATRAD_TESTDATA.keys())}")


class MatRadData:
    """Container for matRad phantom data."""
    
    def __init__(self, name_or_path):
        """Load matRad .mat file.
        
        Args:
            name_or_path: Either a phantom name (e.g., 'HEAD_AND_NECK', 'PROSTATE')
                          or a full path to a .mat file
        """
        self.mat_path = get_phantom_path(name_or_path)
        mat_data = sio.loadmat(self.mat_path, struct_as_record=False, squeeze_me=True)
        
        self.ct = mat_data['ct']
        self.cst = mat_data['cst']
        
        # Extract CT data
        self.cubeHU = self.ct.cubeHU  # (x, y, z) MATLAB order
        self.resolution = self.ct.resolution
        self.spacing = np.array([
            self.resolution.x,
            self.resolution.y, 
            self.resolution.z
        ], dtype=np.float32)
        
        # Coordinate arrays (voxel centers) - may not exist in all phantoms
        if hasattr(self.ct, 'x') and hasattr(self.ct, 'y') and hasattr(self.ct, 'z'):
            self.x = self.ct.x
            self.y = self.ct.y
            self.z = self.ct.z
            # Origin: first voxel position
            self.origin = np.array([
                self.x[0],
                self.y[0],
                self.z[0]
            ], dtype=np.float32)
        else:
            # Generate coordinate arrays for phantoms that don't have them
            # Assume centered at origin
            nx, ny, nz = self.cubeHU.shape
            self.x = np.arange(nx) * self.spacing[0] - (nx - 1) * self.spacing[0] / 2
            self.y = np.arange(ny) * self.spacing[1] - (ny - 1) * self.spacing[1] / 2
            self.z = np.arange(nz) * self.spacing[2] - (nz - 1) * self.spacing[2] / 2
            self.origin = np.array([self.x[0], self.y[0], self.z[0]], dtype=np.float32)
        
        # Parse structures
        self.structures = self._parse_structures()
        
    def _parse_structures(self):
        """Parse CST structure array into dict."""
        structures = {}
        for i in range(self.cst.shape[0]):
            struct = {
                'index': int(self.cst[i, 0]),
                'name': str(self.cst[i, 1]),
                'type': str(self.cst[i, 2]),  # TARGET or OAR
                'voxel_indices': self.cst[i, 3],  # Linear indices in MATLAB 1-based
                'parameters': self.cst[i, 4],
                'objectives': self.cst[i, 5],
            }
            structures[struct['name']] = struct
        return structures
    
    def get_target_names(self):
        """Get list of target structure names."""
        return [name for name, s in self.structures.items() if s['type'] == 'TARGET']
    
    def get_oar_names(self):
        """Get list of OAR structure names."""
        return [name for name, s in self.structures.items() if s['type'] == 'OAR']
    
    def get_structure_mask(self, name):
        """Get 3D binary mask for a structure.
        
        Args:
            name: Structure name (e.g., 'CTV63', 'BRAIN_STEM')
            
        Returns:
            3D numpy array mask in (z, y, x) SimpleITK order
        """
        if name not in self.structures:
            raise ValueError(f"Structure '{name}' not found. Available: {list(self.structures.keys())}")
        
        struct = self.structures[name]
        voxel_indices = struct['voxel_indices']
        
        # Convert from MATLAB 1-based linear indices to 3D indices
        # MATLAB uses column-major (Fortran) order
        shape = self.cubeHU.shape  # (x, y, z) in MATLAB
        
        if hasattr(voxel_indices, '__len__'):
            linear_idx = np.array(voxel_indices, dtype=np.int64) - 1  # 0-based
        else:
            linear_idx = np.array([int(voxel_indices) - 1])
        
        # Create mask in MATLAB (x, y, z) order
        mask_matlab = np.zeros(shape, dtype=np.uint8)
        
        # Unravel linear indices (Fortran order in MATLAB)
        ix, iy, iz = np.unravel_index(linear_idx, shape, order='F')
        mask_matlab[ix, iy, iz] = 1
        
        # Transpose to SimpleITK (z, y, x) order
        mask_sitk = np.transpose(mask_matlab, (2, 1, 0))
        
        return mask_sitk
    
    def get_target_center(self, name):
        """Get center of mass for a target structure in mm.
        
        Returns coordinates suitable for isocenter placement.
        """
        mask = self.get_structure_mask(name)
        
        # Find indices of mask voxels
        indices = np.argwhere(mask > 0)  # (z, y, x)
        center_idx = indices.mean(axis=0)
        
        # Convert to mm coordinates
        # SimpleITK order: (z, y, x)
        center_mm = np.array([
            self.origin[0] + center_idx[2] * self.spacing[0],  # x
            self.origin[1] + center_idx[1] * self.spacing[1],  # y
            self.origin[2] + center_idx[0] * self.spacing[2],  # z
        ])
        
        return center_mm


def convert_to_dosecuda(matrad_data, output_dir, resample_spacing=None):
    """Convert matRad data to DoseCUDA-compatible format.
    
    Args:
        matrad_data: MatRadData instance
        output_dir: Directory to save output files
        resample_spacing: If set, resample CT to this isotropic spacing (mm)
        
    Returns:
        Dict with paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get HU cube - transpose from MATLAB (x,y,z) to SimpleITK (z,y,x)
    cubeHU = np.transpose(matrad_data.cubeHU, (2, 1, 0)).astype(np.float32)
    
    # Create SimpleITK image
    ct_img = sitk.GetImageFromArray(cubeHU)
    ct_img.SetOrigin(matrad_data.origin.tolist())
    ct_img.SetSpacing(matrad_data.spacing.tolist())
    
    # Resample if requested (DoseCUDA requires isotropic spacing)
    if resample_spacing is not None:
        # Calculate new size
        orig_size = ct_img.GetSize()
        orig_spacing = ct_img.GetSpacing()
        new_size = [
            int(round(orig_size[i] * orig_spacing[i] / resample_spacing))
            for i in range(3)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([resample_spacing] * 3)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(ct_img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000)
        
        ct_img = resampler.Execute(ct_img)
    
    # Save CT as NRRD
    ct_path = os.path.join(output_dir, 'ct.nrrd')
    sitk.WriteImage(ct_img, ct_path)
    
    # Save structure masks
    mask_paths = {}
    for name, struct in matrad_data.structures.items():
        try:
            mask = matrad_data.get_structure_mask(name)
            mask_img = sitk.GetImageFromArray(mask)
            mask_img.SetOrigin(matrad_data.origin.tolist())
            mask_img.SetSpacing(matrad_data.spacing.tolist())
            
            # Resample mask if CT was resampled
            if resample_spacing is not None:
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing([resample_spacing] * 3)
                resampler.SetSize(ct_img.GetSize())
                resampler.SetOutputOrigin(ct_img.GetOrigin())
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                mask_img = resampler.Execute(mask_img)
            
            # Sanitize filename
            safe_name = name.replace(' ', '_').replace('/', '_')
            mask_path = os.path.join(output_dir, f'mask_{safe_name}.nrrd')
            sitk.WriteImage(mask_img, mask_path)
            mask_paths[name] = mask_path
        except Exception as e:
            print(f"Warning: Could not create mask for '{name}': {e}")
    
    return {
        'ct_path': ct_path,
        'mask_paths': mask_paths,
        'origin': ct_img.GetOrigin(),
        'spacing': ct_img.GetSpacing(),
        'size': ct_img.GetSize(),
    }


def create_dosecuda_objects(matrad_data, resample_spacing=3.0):
    """Create DoseCUDA objects directly from matRad data.
    
    Args:
        matrad_data: MatRadData instance
        resample_spacing: Isotropic spacing in mm (default 3.0)
        
    Returns:
        (dose_grid, structures_dict) tuple
    """
    # Import DoseCUDA 
    from DoseCUDA import IMPTDoseGrid
    
    dose_grid = IMPTDoseGrid()
    
    # Get HU cube - transpose from MATLAB (x,y,z) to SimpleITK (z,y,x)
    cubeHU = np.transpose(matrad_data.cubeHU, (2, 1, 0)).astype(np.float32)
    
    # Create SimpleITK image for resampling
    ct_img = sitk.GetImageFromArray(cubeHU)
    ct_img.SetOrigin(matrad_data.origin.tolist())
    ct_img.SetSpacing(matrad_data.spacing.tolist())
    
    # Resample to isotropic
    orig_size = ct_img.GetSize()
    orig_spacing = ct_img.GetSpacing()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / resample_spacing))
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([resample_spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(ct_img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    ct_img = resampler.Execute(ct_img)
    
    # Set dose grid properties
    dose_grid.HU = np.array(sitk.GetArrayFromImage(ct_img), dtype=np.float32)
    dose_grid.origin = np.array(ct_img.GetOrigin(), dtype=np.float32)
    dose_grid.spacing = np.array(ct_img.GetSpacing(), dtype=np.float32)
    dose_grid.size = np.array(dose_grid.HU.shape)
    
    # Resample structure masks
    structures = {}
    for name, struct in matrad_data.structures.items():
        try:
            mask = matrad_data.get_structure_mask(name)
            mask_img = sitk.GetImageFromArray(mask)
            mask_img.SetOrigin(matrad_data.origin.tolist())
            mask_img.SetSpacing(matrad_data.spacing.tolist())
            
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            mask_img = resampler.Execute(mask_img)
            
            structures[name] = {
                'type': struct['type'],
                'mask': np.array(sitk.GetArrayFromImage(mask_img), dtype=np.uint8),
            }
        except Exception as e:
            print(f"Warning: Could not process structure '{name}': {e}")
    
    return dose_grid, structures


class MatRadTestData:
    """Container for matRad pre-computed test data (plans, dose influence matrices)."""
    
    def __init__(self, name_or_path):
        """Load matRad test data .mat file.
        
        Args:
            name_or_path: Either a test data name (e.g., 'protons', 'carbon')
                          or a full path to a .mat file
        """
        self.mat_path = get_testdata_path(name_or_path)
        mat_data = sio.loadmat(self.mat_path, struct_as_record=False, squeeze_me=True)
        
        self.ct = mat_data.get('ct')
        self.cst = mat_data.get('cst')
        self.pln = mat_data.get('pln')  # Plan parameters
        self.stf = mat_data.get('stf')  # Steering file (beam geometry)
        self.dij = mat_data.get('dij')  # Dose influence matrix
        self.resultGUI = mat_data.get('resultGUI')  # Pre-computed results
        
    def get_beam_info(self, beam_idx=0):
        """Get beam geometry information.
        
        Returns dict with gantry_angle, couch_angle, isocenter, SAD.
        """
        if self.stf is None:
            return None
            
        beam = self.stf[beam_idx] if hasattr(self.stf, '__len__') else self.stf
        
        return {
            'gantry_angle': float(beam.gantryAngle),
            'couch_angle': float(beam.couchAngle),
            'isocenter': np.array(beam.isoCenter, dtype=np.float32),
            'SAD': float(beam.SAD),
            'n_rays': int(beam.numOfRays),
        }
    
    def get_plan_info(self):
        """Get treatment plan parameters."""
        if self.pln is None:
            return None
            
        return {
            'radiation_mode': str(self.pln.radiationMode),
            'machine': str(self.pln.machine),
            'n_fractions': int(self.pln.numOfFractions),
            'n_beams': int(self.pln.propStf.numOfBeams),
            'gantry_angles': np.array(self.pln.propStf.gantryAngles),
            'couch_angles': np.array(self.pln.propStf.couchAngles),
        }


def list_available_data():
    """Print available phantoms and test data."""
    print("Available matRad Phantoms:")
    print("-" * 40)
    for name, filename in MATRAD_PHANTOMS.items():
        path = os.path.join(MATRAD_PHANTOMS_DIR, filename)
        exists = "✓" if os.path.exists(path) else "✗"
        size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
        print(f"  {exists} {name:20s} ({size:.1f} MB)")
    
    print("\nAvailable matRad Test Data:")
    print("-" * 40)
    for name, filename in MATRAD_TESTDATA.items():
        path = os.path.join(MATRAD_TESTDATA_DIR, filename)
        exists = "✓" if os.path.exists(path) else "✗"
        size = os.path.getsize(path) / 1e3 if os.path.exists(path) else 0
        print(f"  {exists} {name:20s} ({size:.1f} KB)")


# =============================================================================
# Example usage / test
# =============================================================================

if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("matRad Data Converter - Available Data")
    print("=" * 60)
    list_available_data()
    
    print("\n" + "=" * 60)
    print("Testing HEAD_AND_NECK phantom")
    print("=" * 60)
    
    # Test loading by name (not path)
    print("\nLoading matRad data by name...")
    data = MatRadData('HEAD_AND_NECK')
    
    print(f"\n=== CT Info ===")
    print(f"  Loaded from: {data.mat_path}")
    print(f"  HU shape (MATLAB): {data.cubeHU.shape}")
    print(f"  Spacing: {data.spacing} mm")
    print(f"  Origin: {data.origin} mm")
    print(f"  HU range: [{data.cubeHU.min():.0f}, {data.cubeHU.max():.0f}]")
    
    print(f"\n=== Structures ({len(data.structures)}) ===")
    print("  Targets:", data.get_target_names())
    print("  OARs:", data.get_oar_names()[:5], "...")
    
    # Get target center
    targets = data.get_target_names()
    if targets:
        target = targets[0]
        center = data.get_target_center(target)
        print(f"\n=== Target '{target}' ===")
        print(f"  Center (isocenter): {center} mm")
    
    print("\n" + "=" * 60)
    print("Testing protons_testData")
    print("=" * 60)
    
    test_data = MatRadTestData('protons')
    print(f"\nLoaded from: {test_data.mat_path}")
    
    plan_info = test_data.get_plan_info()
    if plan_info:
        print(f"\n=== Plan Info ===")
        print(f"  Radiation mode: {plan_info['radiation_mode']}")
        print(f"  Machine: {plan_info['machine']}")
        print(f"  Fractions: {plan_info['n_fractions']}")
        print(f"  Number of beams: {plan_info['n_beams']}")
        print(f"  Gantry angles: {plan_info['gantry_angles']}")
    
    beam_info = test_data.get_beam_info(0)
    if beam_info:
        print(f"\n=== Beam 0 Info ===")
        print(f"  Gantry angle: {beam_info['gantry_angle']}°")
        print(f"  Couch angle: {beam_info['couch_angle']}°")
        print(f"  Isocenter: {beam_info['isocenter']}")
        print(f"  SAD: {beam_info['SAD']} mm")
        print(f"  Number of rays: {beam_info['n_rays']}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
