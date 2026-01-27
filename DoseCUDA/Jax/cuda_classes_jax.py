"""
CudaClasses.py - JAX implementation of CUDA beam and dose geometry operations

Differentiable JAX implementations of core beam geometry transformations and 
dose computation support structures from the CUDA kernels.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class PointXYZ_jax:
    """3D Cartesian coordinates"""
    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray


@dataclass
class PointIJK_jax:
    """3D Voxel/Image indices"""
    i: jnp.ndarray
    j: jnp.ndarray
    k: jnp.ndarray


class CudaBeam_jax:
    """
    JAX implementation of beam geometry and coordinate transformations.
    
    Handles:
    - Isocenter and source positioning
    - Gantry and couch angle transformations
    - Image to head frame conversions
    - Distance calculations to source and CAX
    """
    
    def __init__(self, iso: jnp.ndarray, gantry_angle: float, 
                 couch_angle: float, src_dist: float):
        """
        Initialize beam geometry
        
        Args:
            iso: Isocenter position [x, y, z]
            gantry_angle: Gantry rotation in degrees
            couch_angle: Couch rotation in degrees
            src_dist: Source to isocenter distance in cm
        """
        self.iso = PointXYZ_jax(iso[0], iso[1], iso[2])
        self.gantry_angle = gantry_angle
        self.couch_angle = couch_angle
        
        # Convert angles to radians and precompute trig functions
        ga = jnp.deg2rad(gantry_angle)
        ta = jnp.deg2rad(couch_angle)
        
        self.singa = jnp.sin(ga)
        self.cosga = jnp.cos(ga)
        self.sinta = jnp.sin(ta)
        self.costa = jnp.cos(ta)
        
        # Compute source position based on gantry and table rotation
        # Starting at [0, SAD, 0], apply rotations
        xg = -src_dist * self.singa
        yg = src_dist * self.cosga
        
        xt = xg * self.costa
        yt = yg
        zt = -xg * self.sinta
        
        self.src = PointXYZ_jax(xt, yt, zt)
    
    def unit_vector_to_source(self, point_xyz: PointXYZ_jax) -> PointXYZ_jax:
        """
        Compute unit vector from point to source
        
        Args:
            point_xyz: Point in image coordinates
            
        Returns:
            Unit vector pointing toward source
        """
        dx = self.src.x - point_xyz.x
        dy = self.src.y - point_xyz.y
        dz = self.src.z - point_xyz.z
        
        norm = jnp.sqrt(dx**2 + dy**2 + dz**2)
        
        return PointXYZ_jax(dx / norm, dy / norm, dz / norm)
    
    def distance_to_source(self, point_xyz: PointXYZ_jax) -> jnp.ndarray:
        """
        Compute Euclidean distance from point to source
        
        Args:
            point_xyz: Point in image coordinates
            
        Returns:
            Distance to source
        """
        dx = self.src.x - point_xyz.x
        dy = self.src.y - point_xyz.y
        dz = self.src.z - point_xyz.z
        
        return jnp.sqrt(dx**2 + dy**2 + dz**2)
    
    def point_xyz_image_to_head(self, point_img: PointXYZ_jax) -> PointXYZ_jax:
        """
        Transform from DICOM image coordinates to BEV (Beam's Eye View) coordinates
        
        Applies:
        1. Table rotation (about y-axis)
        2. Gantry rotation (about z-axis)
        3. Coordinate swap to DICOM nozzle frame
        
        Args:
            point_img: Point in DICOM image coordinates
            
        Returns:
            Point in BEV coordinates
        """
        # Table rotation - rotate about y-axis
        xt = point_img.x * self.costa + point_img.z * (-self.sinta)
        yt = point_img.y
        zt = -point_img.x * (-self.sinta) + point_img.z * self.costa
        
        # Gantry rotation - rotate about z-axis
        xg = xt * self.cosga - yt * (-self.singa)
        yg = xt * (-self.singa) + yt * self.cosga
        zg = zt
        
        # Swap to DICOM nozzle coordinates (matching CUDA exactly)
        # For an AP beam:
        #   beam travels in negative z direction
        #   positive x is to the patient's left
        #   positive y is to the patient's superior
        # head_x = -xg, head_y = zg, head_z = yg
        point_head = PointXYZ_jax(-xg, zg, yg)
        
        return point_head
    
    def point_xyz_head_to_image(self, point_head: PointXYZ_jax) -> PointXYZ_jax:
        """
        Transform from BEV (Beam's Eye View) to DICOM image coordinates
        
        Inverse of point_xyz_image_to_head
        
        Args:
            point_head: Point in BEV coordinates
            
        Returns:
            Point in DICOM image coordinates
        """
        # Convert back to DICOM patient LPS coordinates
        xz = -point_head.x
        yz = point_head.z
        zz = point_head.y
        
        # Gantry rotation - rotate about z-axis (negative direction)
        xg = xz * self.cosga - yz * self.singa
        yg = xz * self.singa + yz * self.cosga
        zg = zz
        
        # Table rotation - rotate about y-axis (negative direction)
        xt = xg * self.costa + zg * self.sinta
        yt = yg
        zt = -xg * self.sinta + zg * self.costa
        
        point_img = PointXYZ_jax(xt, yt, zt)
        
        return point_img
    
    def point_xyz_closest_cax_point(self, point_xyz: PointXYZ_jax) -> PointXYZ_jax:
        """
        Find the closest point on the central axis to given point
        
        Args:
            point_xyz: Point in image coordinates
            
        Returns:
            Closest point on CAX
        """
        d1x = self.iso.x - self.src.x
        d1y = self.iso.y - self.src.y
        d1z = self.iso.z - self.src.z
        
        d2x = point_xyz.x - self.src.x
        d2y = point_xyz.y - self.src.y
        d2z = point_xyz.z - self.src.z
        
        t = (d1x * d2x + d1y * d2y + d1z * d2z) / (d1x**2 + d1y**2 + d1z**2)
        
        point_cax = PointXYZ_jax(
            t * d1x + self.src.x,
            t * d1y + self.src.y,
            t * d1z + self.src.z
        )
        
        return point_cax
    
    def point_xyz_distance_to_cax(self, point_head_xyz: PointXYZ_jax) -> jnp.ndarray:
        """
        Compute distance from point to central axis
        
        Args:
            point_head_xyz: Point in BEV coordinates
            
        Returns:
            Distance to CAX
        """
        return jnp.sqrt(point_head_xyz.x**2 + point_head_xyz.z**2)


class CudaDose_jax:
    """
    JAX implementation of dose grid and coordinate transformations.
    
    Manages:
    - Voxel grid dimensions and spacing
    - IJK (voxel) to XYZ (physical) coordinate conversions
    - Dose, density, and WET array storage
    """
    
    def __init__(self, img_sz: Tuple[int, int, int], spacing: float, 
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize dose grid
        
        Args:
            img_sz: Image dimensions [z, y, x] (k, j, i)
            spacing: Voxel spacing in cm
            origin: Physical coordinates of voxel (0,0,0)
        """
        self.img_sz = PointIJK_jax(jnp.asarray(img_sz[2]), jnp.asarray(img_sz[1]), jnp.asarray(img_sz[0]))
        self.spacing = spacing
        self.origin = PointXYZ_jax(jnp.array(origin[0]), jnp.array(origin[1]), jnp.array(origin[2]))
        self.num_voxels = self.img_sz.i * self.img_sz.j * self.img_sz.k
        
        # Initialize arrays (will be populated with actual data)
        self.dose_array = None
        self.density_array = None
        self.wet_array = None
    
    def point_ijk_within_image(self, point_ijk: PointIJK_jax) -> jnp.ndarray:
        """
        Check if voxel coordinates are within image bounds
        
        Args:
            point_ijk: Voxel coordinates
            
        Returns:
            Boolean array indicating if point is within bounds
        """
        return (point_ijk.i < self.img_sz.i) & \
               (point_ijk.j < self.img_sz.j) & \
               (point_ijk.k < self.img_sz.k) & \
               (point_ijk.i >= 0) & \
               (point_ijk.j >= 0) & \
               (point_ijk.k >= 0)
    
    def point_ijk_to_index(self, point_ijk: PointIJK_jax) -> jnp.ndarray:
        """
        Convert IJK voxel coordinates to linear index
        
        Args:
            point_ijk: Voxel coordinates
            
        Returns:
            Linear index into dose array
        """
        return point_ijk.i + self.img_sz.i * (point_ijk.j + self.img_sz.j * point_ijk.k)
    
    def point_ijk_to_xyz(self, point_ijk: PointIJK_jax, beam: CudaBeam_jax) -> PointXYZ_jax:
        """
        Convert voxel coordinates to physical coordinates
        
        Args:
            point_ijk: Voxel coordinates
            beam: Beam object for isocenter information
            
        Returns:
            Point in physical coordinates
        """
        point_xyz = PointXYZ_jax(
            jnp.asarray(point_ijk.i, dtype=jnp.float32) * self.spacing - beam.iso.x,
            jnp.asarray(point_ijk.j, dtype=jnp.float32) * self.spacing - beam.iso.y,
            jnp.asarray(point_ijk.k, dtype=jnp.float32) * self.spacing - beam.iso.z
        )
        
        return point_xyz
    
    def point_xyz_to_ijk(self, point_xyz: PointXYZ_jax, beam: CudaBeam_jax) -> PointIJK_jax:
        """
        Convert physical coordinates to voxel coordinates
        
        Args:
            point_xyz: Point in physical coordinates
            beam: Beam object for isocenter information
            
        Returns:
            Voxel coordinates
        """
        point_ijk = PointIJK_jax(
            jnp.round((point_xyz.x + beam.iso.x) / self.spacing).astype(jnp.int32),
            jnp.round((point_xyz.y + beam.iso.y) / self.spacing).astype(jnp.int32),
            jnp.round((point_xyz.z + beam.iso.z) / self.spacing).astype(jnp.int32)
        )
        
        return point_ijk
    
    def point_xyz_to_texture_xyz(self, point_xyz: PointXYZ_jax, beam: CudaBeam_jax) -> PointXYZ_jax:
        """
        Convert physical coordinates to texture coordinates (0 to size)
        
        Args:
            point_xyz: Point in physical coordinates
            beam: Beam object for isocenter information
            
        Returns:
            Point in texture coordinates
        """
        tex_xyz = PointXYZ_jax(
            (point_xyz.x + beam.iso.x) / self.spacing + 0.5,
            (point_xyz.y + beam.iso.y) / self.spacing + 0.5,
            (point_xyz.z + beam.iso.z) / self.spacing + 0.5
        )
        
        return tex_xyz
    
    def texture_xyz_within_image(self, point_xyz: PointXYZ_jax) -> jnp.ndarray:
        """
        Check if texture coordinates are within image bounds
        
        Args:
            point_xyz: Point in texture coordinates
            
        Returns:
            Boolean array indicating if point is within bounds
        """
        return (point_xyz.x >= 0.0) & (point_xyz.x < jnp.asarray(self.img_sz.i, dtype=jnp.float32)) & \
               (point_xyz.y >= 0.0) & (point_xyz.y < jnp.asarray(self.img_sz.j, dtype=jnp.float32)) & \
               (point_xyz.z >= 0.0) & (point_xyz.z < jnp.asarray(self.img_sz.k, dtype=jnp.float32))


# Utility functions that are differentiable
def lower_bound_jax(data: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """
    Differentiable binary search for lower bound
    
    Finds the first index i such that data[i] >= key
    
    Args:
        data: Sorted 1D array
        key: Search key
        
    Returns:
        Index of lower bound
    """
    # Using searchsorted for differentiable binary search
    return jnp.searchsorted(data, key, side='left')


def sqr_jax(x: jnp.ndarray) -> jnp.ndarray:
    """Square of input"""
    return x * x


def clamp_jax(x: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Clamp value to [lo, hi]"""
    return jnp.clip(x, lo, hi)


def xyz_dotproduct_jax(a: PointXYZ_jax, b: PointXYZ_jax) -> jnp.ndarray:
    """3D dot product"""
    return a.x * b.x + a.y * b.y + a.z * b.z


def xyz_crossproduct_jax(a: PointXYZ_jax, b: PointXYZ_jax) -> PointXYZ_jax:
    """3D cross product"""
    return PointXYZ_jax(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )
