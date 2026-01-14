"""
LUT (Look-Up Table) Implementation
colour library互換

3D LUTと1D LUT×3（3x1D）をサポート
.cubeファイル読み込みとTrilinear補間
"""

import numpy as np
import re
from typing import Union, Tuple
from pathlib import Path


class LUT3D:
    """
    3D LUT (Look-Up Table) class
    
    Attributes:
        name: LUT name
        domain: array shape (2, 3) - [[min_r, min_g, min_b], [max_r, max_g, max_b]]
        size: LUT size (e.g., 33 for 33×33×33 LUT)
        table: array shape (size, size, size, 3) - LUT data
    """
    
    def __init__(self, table: np.ndarray, name: str = "LUT3D", 
                 domain: np.ndarray = None, size: int = None):
        """
        Initialize 3D LUT.
        
        Parameters:
            table: shape (size, size, size, 3)
            name: LUT name
            domain: shape (2, 3) - input range [[min], [max]]
            size: LUT size (derived from table if not provided)
        """
        self.table = np.asarray(table, dtype=np.float32)
        self.name = name
        
        if self.table.ndim != 4 or self.table.shape[3] != 3:
            raise ValueError(f"Invalid table shape: {self.table.shape}. Expected (N, N, N, 3)")
        
        if size is None:
            size = self.table.shape[0]
            if not (self.table.shape[0] == self.table.shape[1] == self.table.shape[2] == size):
                raise ValueError(f"Table must be cubic, got shape {self.table.shape}")
        
        self.size = size
        
        if domain is None:
            self.domain = np.array([[0., 0., 0.], [1., 1., 1.]], dtype=np.float32)
        else:
            self.domain = np.asarray(domain, dtype=np.float32)
    
    def apply(self, RGB: np.ndarray, interpolation: str = 'trilinear') -> np.ndarray:
        """
        Apply 3D LUT to RGB values using trilinear interpolation.
        
        Parameters:
            RGB: array shape (..., 3) - input RGB values
            interpolation: 'trilinear' (only trilinear supported for now)
        
        Returns:
            RGB_out: array shape (..., 3) - transformed RGB values
        """
        RGB = np.asarray(RGB, dtype=np.float32)
        original_shape = RGB.shape
        
        # Flatten to 2D
        RGB_flat = RGB.reshape(-1, 3)
        
        # Normalize to [0, size-1] grid coordinates
        # RGB in domain → [0, 1] → [0, size-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        
        # Normalize to [0, 1]
        RGB_norm = (RGB_flat - domain_min) / (domain_max - domain_min)
        
        # Note: Do NOT clip here to preserve HDR values (>1.0)
        # Values outside [0, 1] will be extrapolated by the LUT
        
        # Scale to grid coordinates
        grid_coords = RGB_norm * (self.size - 1)
        
        # Trilinear interpolation
        RGB_out = self._trilinear_interpolation(grid_coords)
        
        return RGB_out.reshape(original_shape)
    
    def _trilinear_interpolation(self, coords: np.ndarray) -> np.ndarray:
        """
        Trilinear interpolation in 3D LUT with linear extrapolation.
        
        For values outside [0, size-1], uses linear extrapolation based on
        the gradient at the boundary.
        
        Parameters:
            coords: shape (N, 3) - coordinates, can be outside [0, size-1]
        
        Returns:
            values: shape (N, 3) - interpolated/extrapolated RGB values
        """
        # Clamp coordinates to valid indices
        # For extrapolation, we use the edge cells and adjust the fractional part
        coords_clamped = np.clip(coords, 0, self.size - 1)
        
        coords_floor = np.floor(coords_clamped).astype(np.int32)
        coords_floor = np.clip(coords_floor, 0, self.size - 2)
        coords_ceil = coords_floor + 1
        
        # Calculate fractional part for extrapolation
        # If coords > size-1, frac will be > 1, enabling extrapolation
        # If coords < 0, frac will be < 0, enabling extrapolation  
        coords_frac = coords - coords_floor
        
        # Extract coordinates
        r0, g0, b0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
        r1, g1, b1 = coords_ceil[:, 0], coords_ceil[:, 1], coords_ceil[:, 2]
        
        rd, gd, bd = coords_frac[:, 0:1], coords_frac[:, 1:2], coords_frac[:, 2:3]
        
        # Get 8 corner values
        c000 = self.table[r0, g0, b0]  # shape (N, 3)
        c001 = self.table[r0, g0, b1]
        c010 = self.table[r0, g1, b0]
        c011 = self.table[r0, g1, b1]
        c100 = self.table[r1, g0, b0]
        c101 = self.table[r1, g0, b1]
        c110 = self.table[r1, g1, b0]
        c111 = self.table[r1, g1, b1]
        
        # Trilinear interpolation/extrapolation
        # When rd>1 or rd<0, this extrapolates linearly
        c00 = c000 * (1 - rd) + c100 * rd
        c01 = c001 * (1 - rd) + c101 * rd
        c10 = c010 * (1 - rd) + c110 * rd
        c11 = c011 * (1 - rd) + c111 * rd
        
        # Interpolate along G
        c0 = c00 * (1 - gd) + c10 * gd
        c1 = c01 * (1 - gd) + c11 * gd
        
        # Interpolate along B
        result = c0 * (1 - bd) + c1 * bd
        
        return result


class LUT3x1D:
    """
    3x1D LUT (1D LUT for each RGB channel)
    
    Attributes:
        name: LUT name
        domain: array shape (2, 3) - [[min_r, min_g, min_b], [max_r, max_g, max_b]]
        size: LUT size (number of entries per channel)
        table: array shape (size, 3) - LUT data [R_lut, G_lut, B_lut]
    """
    
    def __init__(self, table: np.ndarray, name: str = "LUT3x1D",
                 domain: np.ndarray = None, size: int = None):
        """
        Initialize 3x1D LUT.
        
        Parameters:
            table: shape (size, 3)
            name: LUT name
            domain: shape (2, 3) - input range [[min], [max]]
            size: LUT size (derived from table if not provided)
        """
        self.table = np.asarray(table, dtype=np.float32)
        self.name = name
        
        if self.table.ndim != 2 or self.table.shape[1] != 3:
            raise ValueError(f"Invalid table shape: {self.table.shape}. Expected (N, 3)")
        
        if size is None:
            size = self.table.shape[0]
        
        self.size = size
        
        if domain is None:
            self.domain = np.array([[0., 0., 0.], [1., 1., 1.]], dtype=np.float32)
        else:
            self.domain = np.asarray(domain, dtype=np.float32)
    
    def apply(self, RGB: np.ndarray) -> np.ndarray:
        """
        Apply 3x1D LUT to RGB values using linear interpolation per channel.
        
        Parameters:
            RGB: array shape (..., 3) - input RGB values
        
        Returns:
            RGB_out: array shape (..., 3) - transformed RGB values
        """
        RGB = np.asarray(RGB, dtype=np.float32)
        original_shape = RGB.shape
        
        # Flatten to 2D
        RGB_flat = RGB.reshape(-1, 3)
        RGB_out = np.zeros_like(RGB_flat)
        
        # Process each channel independently
        for channel in range(3):
            # Normalize to [0, 1]
            domain_min = self.domain[0, channel]
            domain_max = self.domain[1, channel]
            
            values = (RGB_flat[:, channel] - domain_min) / (domain_max - domain_min)
            values = np.clip(values, 0, 1)
            
            # Scale to grid coordinates [0, size-1]
            coords = values * (self.size - 1)
            
            # Linear interpolation
            RGB_out[:, channel] = np.interp(coords, np.arange(self.size), self.table[:, channel])
        
        return RGB_out.reshape(original_shape)


def read_LUT_IridasCube(path: str) -> Union[LUT3D, LUT3x1D]:
    """
    Read Iridas .cube LUT file.
    
    Supports both 3D LUT and 3x1D LUT formats.
    
    Parameters:
        path: path to .cube file
    
    Returns:
        LUT3D or LUT3x1D instance
    
    Raises:
        ValueError: if file format is invalid
    
    Example .cube format:
        TITLE "My LUT"
        LUT_3D_SIZE 33
        DOMAIN_MIN 0.0 0.0 0.0
        DOMAIN_MAX 1.0 1.0 1.0
        
        0.0 0.0 0.0
        0.0 0.0 0.031373
        ...
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"LUT file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse header
    title = None
    lut_1d_size = None
    lut_3d_size = None
    domain_min = np.array([0., 0., 0.], dtype=np.float32)
    domain_max = np.array([1., 1., 1.], dtype=np.float32)
    
    data_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        # Parse keywords
        if line.startswith('TITLE'):
            title = line.split('"')[1] if '"' in line else line.split()[1]
        
        elif line.startswith('LUT_1D_SIZE'):
            lut_1d_size = int(line.split()[1])
        
        elif line.startswith('LUT_3D_SIZE'):
            lut_3d_size = int(line.split()[1])
        
        elif line.startswith('DOMAIN_MIN'):
            parts = line.split()[1:]
            domain_min = np.array([float(p) for p in parts], dtype=np.float32)
        
        elif line.startswith('DOMAIN_MAX'):
            parts = line.split()[1:]
            domain_max = np.array([float(p) for p in parts], dtype=np.float32)
        
        else:
            # Data line (should contain 3 float values)
            parts = line.split()
            if len(parts) == 3:
                try:
                    values = [float(p) for p in parts]
                    data_lines.append(values)
                except ValueError:
                    continue  # Skip invalid lines
    
    if not data_lines:
        raise ValueError(f"No valid LUT data found in {path}")
    
    # Convert to numpy array
    data = np.array(data_lines, dtype=np.float32)
    
    # Determine LUT type and create appropriate object
    if title is None:
        title = path.stem
    
    domain = np.array([domain_min, domain_max], dtype=np.float32)
    
    # 3D LUT
    if lut_3d_size is not None:
        expected_size = lut_3d_size ** 3
        if data.shape[0] != expected_size:
            raise ValueError(f"Expected {expected_size} entries for {lut_3d_size}³ LUT, got {data.shape[0]}")
        
        # Reshape to 3D
        table = data.reshape(lut_3d_size, lut_3d_size, lut_3d_size, 3)
        
        return LUT3D(table, name=title, domain=domain, size=lut_3d_size)
    
    # 1D LUT (3x1D)
    elif lut_1d_size is not None:
        if data.shape[0] != lut_1d_size:
            raise ValueError(f"Expected {lut_1d_size} entries for 1D LUT, got {data.shape[0]}")
        
        return LUT3x1D(data, name=title, domain=domain, size=lut_1d_size)
    
    else:
        # Try to infer from data size
        n = data.shape[0]
        
        # Check if it's a perfect cube
        cube_root = round(n ** (1/3))
        if cube_root ** 3 == n:
            # 3D LUT
            table = data.reshape(cube_root, cube_root, cube_root, 3)
            return LUT3D(table, name=title, domain=domain, size=cube_root)
        else:
            # 1D LUT
            return LUT3x1D(data, name=title, domain=domain, size=n)


# ============================================================================
# Your wrapper functions (colour library互換)
# ============================================================================

def read_lut(lut_path: str, clip: bool = False) -> Union[LUT3D, LUT3x1D]:
    """
    Reads a LUT from the specified path, returning instance of LUT3D or LUT3x1D
    
    Parameters:
        lut_path: the path to the file from which to read the LUT
        clip: flag indicating whether to apply clipping of LUT values
    
    Returns:
        LUT3D or LUT3x1D instance
    """
    import os
    
    lut = read_LUT_IridasCube(lut_path)
    lut.name = os.path.splitext(os.path.basename(lut_path))[0]
    
    if clip:
        if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
            lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
        else:
            if len(lut.table.shape) == 2:  # 3x1D
                for dim in range(3):
                    lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
            else:  # 3D
                for dim in range(3):
                    lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])
    
    return lut


def apply_lut(image: np.ndarray, lut: Union[LUT3D, LUT3x1D], log: bool = False) -> np.ndarray:
    """
    Apply LUT to image with optional log colorspace conversion.
    
    Parameters:
        image: input image array
        lut: LUT3D or LUT3x1D instance
        log: if True, transform to log colorspace before applying LUT
    
    Returns:
        transformed image array as float32
    """
    im_array = image
    is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
    dom_scale = None
    
    if is_non_default_domain:
        dom_scale = lut.domain[1] - lut.domain[0]
        im_array = im_array * dom_scale + lut.domain[0]
    
    if log:
        im_array = im_array ** (1/2.2)
    
    im_array = lut.apply(im_array)
    
    if log:
        im_array = im_array ** (2.2)
    
    if is_non_default_domain:
        im_array = (im_array - lut.domain[0]) / dom_scale
    
    return im_array.astype(np.float32)
