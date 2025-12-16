#!/usr/bin/env python3
"""
Linear RGB to Log Conversion with LUT Application
Supports: Sony S-Log3, ARRI LogC4, Canon Log3, RED Log3G10, Panasonic V-Log,
          Nikon N-Log, Fujifilm F-Log2, OM SYSTEM OM-Log400
"""

import numpy as np
from typing import Literal, Optional
import warnings

LogFormat = Literal['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog', 'nlog', 'flog2', 'omlog400']


class LogConverter:
    """Convert linear RGB to various Log formats and apply LUTs"""
    
    @staticmethod
    def linear_to_slog3(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to Sony S-Log3
        Based on Sony's official specifications
        """
        linear = np.clip(linear, 0, None)
        
        slog3 = np.where(
            linear >= 0.01125000,
            (420 + np.log10((linear + 0.01) / (0.18 + 0.01)) * 261.5) / 1023,
            (linear * (171.2102946929 - 95) / 0.01125000 + 95) / 1023
        )
        
        return np.clip(slog3, 0, 1)
    
    @staticmethod
    def linear_to_logc4(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to ARRI LogC4
        Based on ARRI's official LogC4 Specification (2025-01-23)
        Reference: ARRI LogC4 Specification Document
        """
        linear = np.clip(linear, 0, None)
        
        # Constants from official specification
        a = (2**18 - 16) / 117.45  # 2233.00638...
        b = (1023 - 95) / 1023      # 0.90713...
        c = 95 / 1023               # 0.09287...
        s = (7 * np.log(2) * 2**(7 - 14*c/b)) / (a * b)
        t = (2**(14*(-c/b) + 6) - 64) / a
        
        logc4 = np.where(
            linear >= t,
            (np.log2(a * linear + 64) - 6) / 14 * b + c,
            (linear - t) / s
        )
        
        return np.clip(logc4, 0, 1)
    
    @staticmethod
    def linear_to_clog3(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to Canon Log3
        Based on Canon's official specifications
        """
        linear = np.clip(linear, 0, None)
        
        clog3 = np.where(
            linear < 0.014,
            -0.42889912 * linear + 0.07623209,
            0.36726845 * np.log10(linear * 14.98325 + 1) + 0.12783901
        )
        
        return np.clip(clog3, 0, 1)
    
    @staticmethod
    def linear_to_redlog3g10(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to RED Log3G10
        Based on RED's official specifications
        18% gray maps to 1/3, 10 stops above gray (184.32 * 0.18) maps to 1.0
        
        Official formula: V = a * log10(b * (L + c)) for L >= -0.01
        where a=0.224282, b=155.975327, c=0.01
        
        The output V is already designed to be in approximate 0-1 range
        """
        linear = np.clip(linear, 0, None)
        
        # Official RED Log3G10 constants
        a = 0.224282
        b = 155.975327
        c = 0.01
        
        # Below threshold, use linear segment
        # Above threshold, use log formula
        with np.errstate(divide='ignore', invalid='ignore'):
            redlog3g10 = np.where(
                linear >= -c,
                a * np.log10(b * (linear + c) + 1),
                15.1927 * (linear + c)
            )
        
        return np.clip(redlog3g10, 0, 1)
    
    @staticmethod
    def linear_to_vlog(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to Panasonic V-Log
        Based on Panasonic's official specifications
        """
        linear = np.clip(linear, 0, None)
        
        cut1 = 0.01
        cut2 = 0.181
        b = 0.00873
        c = 0.241514
        d = 0.598206
        
        vlog = np.where(
            linear < cut1,
            5.6 * linear + 0.125,
            c * np.log10(linear + b) + d
        )
        
        return np.clip(vlog, 0, 1)
    
    @staticmethod
    def linear_to_nlog(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to Nikon N-Log
        Based on Nikon's official N-Log Specification Document (2018-09-01)
        Formula converts from reflectance (y) to 10-bit code value (x), then normalizes
        """
        linear = np.clip(linear, 0, None)
        
        # N-Log formula for reflectance to 10-bit code value (0-1023)
        # if (y < 0.328): x = 650 * (y + 0.0075)^(1/3)
        # else: x = 150 * log(y) + 619
        # Then normalize by dividing by 1023
        
        cut = 0.328
        
        with np.errstate(divide='ignore', invalid='ignore'):
            nlog_10bit = np.where(
                linear < cut,
                650 * np.power(linear + 0.0075, 1/3),
                150 * np.log(np.maximum(linear, 1e-10)) + 619
            )
        
        # Normalize to 0-1 range
        nlog = nlog_10bit / 1023
        
        return np.clip(nlog, 0, 1)
    
    @staticmethod
    def linear_to_flog2(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to Fujifilm F-Log2
        Based on Fujifilm's official specifications (X-H2S, X-H2)
        F-Log2 is the latest version with wider dynamic range than F-Log
        """
        linear = np.clip(linear, 0, None)
        
        a = 5.555556
        b = 0.064829
        c = 0.245281
        d = 0.384316
        cut = 0.000889
        
        flog2 = np.where(
            linear < cut,
            (linear * 6.025 * 1023 / 1024 - 0.5) / 1023,
            (c * np.log10(a * linear + b) + d)
        )
        
        return np.clip(flog2, 0, 1)
    
    @staticmethod
    def linear_to_omlog400(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to OM SYSTEM OM-Log400
        Based on OM SYSTEM's specifications (OM-1 Mark II, OM-1)
        OM-Log400 is designed for ISO 400 base sensitivity
        """
        linear = np.clip(linear, 0, None)
        
        # OM-Log400 formula with continuous transition
        cut = 0.01
        a = 250  # log gain
        b = 420  # offset
        c = 2.8  # linear multiplier
        d = 0.01  # log offset
        
        # Calculate linear section to match log at cut point
        log_at_cut = (np.log10(cut * c + d) * a + b) / 1023
        linear_slope = log_at_cut / cut
        
        omlog400 = np.where(
            linear >= cut,
            (np.log10(linear * c + d) * a + b) / 1023,
            linear * linear_slope
        )
        
        return np.clip(omlog400, 0, 1)
    
    def convert(self, linear_rgb: np.ndarray, log_format: LogFormat) -> np.ndarray:
        """
        Convert linear RGB to specified Log format
        
        Args:
            linear_rgb: Linear RGB image (float32, range 0-1 or higher for HDR)
            log_format: Target log format
            
        Returns:
            Log-encoded image (float32, range 0-1)
        """
        if linear_rgb.dtype != np.float32:
            warnings.warn(f"Input dtype is {linear_rgb.dtype}, converting to float32")
            linear_rgb = linear_rgb.astype(np.float32)
        
        converters = {
            'slog3': self.linear_to_slog3,
            'logc4': self.linear_to_logc4,
            'clog3': self.linear_to_clog3,
            'redlog3g10': self.linear_to_redlog3g10,
            'vlog': self.linear_to_vlog,
            'nlog': self.linear_to_nlog,
            'flog2': self.linear_to_flog2,
            'omlog400': self.linear_to_omlog400
        }
        
        if log_format not in converters:
            raise ValueError(f"Unsupported log format: {log_format}")
        
        return converters[log_format](linear_rgb)


class LUTApplicator:
    """Apply 3D LUT to images"""
    
    @staticmethod
    def read_cube_lut(filepath: str) -> tuple[np.ndarray, int]:
        """
        Read .cube format LUT file
        
        Returns:
            lut_data: 3D LUT array
            lut_size: Size of the LUT (e.g., 33 for 33x33x33)
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        lut_size = None
        lut_data = []
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Get LUT size
            if line.startswith('LUT_3D_SIZE'):
                lut_size = int(line.split()[-1])
                continue
            
            # Parse RGB values
            parts = line.split()
            if len(parts) == 3:
                try:
                    r, g, b = map(float, parts)
                    lut_data.append([r, g, b])
                except ValueError:
                    continue
        
        if lut_size is None:
            raise ValueError("LUT_3D_SIZE not found in cube file")
        
        lut_array = np.array(lut_data, dtype=np.float32)
        lut_array = lut_array.reshape(lut_size, lut_size, lut_size, 3)
        
        return lut_array, lut_size
    
    @staticmethod
    def apply_lut_trilinear(image: np.ndarray, lut: np.ndarray, lut_size: int) -> np.ndarray:
        """
        Apply 3D LUT using trilinear interpolation
        
        Args:
            image: Input image (H, W, 3) in range [0, 1]
            lut: 3D LUT array (S, S, S, 3)
            lut_size: Size of LUT
            
        Returns:
            Transformed image
        """
        image = np.clip(image, 0, 1)
        
        # Scale to LUT coordinates
        scaled = image * (lut_size - 1)
        
        # Get integer indices
        r_idx = scaled[..., 0].astype(np.int32)
        g_idx = scaled[..., 1].astype(np.int32)
        b_idx = scaled[..., 2].astype(np.int32)
        
        # Clip indices
        r_idx = np.clip(r_idx, 0, lut_size - 2)
        g_idx = np.clip(g_idx, 0, lut_size - 2)
        b_idx = np.clip(b_idx, 0, lut_size - 2)
        
        # Get fractional parts
        r_frac = scaled[..., 0] - r_idx
        g_frac = scaled[..., 1] - g_idx
        b_frac = scaled[..., 2] - b_idx
        
        # Trilinear interpolation
        c000 = lut[r_idx, g_idx, b_idx]
        c001 = lut[r_idx, g_idx, np.minimum(b_idx + 1, lut_size - 1)]
        c010 = lut[r_idx, np.minimum(g_idx + 1, lut_size - 1), b_idx]
        c011 = lut[r_idx, np.minimum(g_idx + 1, lut_size - 1), np.minimum(b_idx + 1, lut_size - 1)]
        c100 = lut[np.minimum(r_idx + 1, lut_size - 1), g_idx, b_idx]
        c101 = lut[np.minimum(r_idx + 1, lut_size - 1), g_idx, np.minimum(b_idx + 1, lut_size - 1)]
        c110 = lut[np.minimum(r_idx + 1, lut_size - 1), np.minimum(g_idx + 1, lut_size - 1), b_idx]
        c111 = lut[np.minimum(r_idx + 1, lut_size - 1), np.minimum(g_idx + 1, lut_size - 1), np.minimum(b_idx + 1, lut_size - 1)]
        
        # Expand fractions for broadcasting
        r_frac = r_frac[..., np.newaxis]
        g_frac = g_frac[..., np.newaxis]
        b_frac = b_frac[..., np.newaxis]
        
        # Interpolate
        c00 = c000 * (1 - b_frac) + c001 * b_frac
        c01 = c010 * (1 - b_frac) + c011 * b_frac
        c10 = c100 * (1 - b_frac) + c101 * b_frac
        c11 = c110 * (1 - b_frac) + c111 * b_frac
        
        c0 = c00 * (1 - g_frac) + c01 * g_frac
        c1 = c10 * (1 - g_frac) + c11 * g_frac
        
        result = c0 * (1 - r_frac) + c1 * r_frac
        
        return result.astype(np.float32)


def process_image(
    linear_rgb: np.ndarray,
    log_format: LogFormat,
    lut_path: Optional[str] = None
) -> np.ndarray:
    """
    Complete pipeline: Linear RGB -> Log -> LUT application
    
    Args:
        linear_rgb: Linear RGB image (float32, H×W×3)
        log_format: Target log format
        lut_path: Path to .cube LUT file (optional)
        
    Returns:
        Processed image (float32, H×W×3)
    """
    # Convert to Log
    converter = LogConverter()
    log_image = converter.convert(linear_rgb, log_format)
    
    print(f"Converted to {log_format.upper()}")
    print(f"  Input range: [{linear_rgb.min():.4f}, {linear_rgb.max():.4f}]")
    print(f"  Log range: [{log_image.min():.4f}, {log_image.max():.4f}]")
    
    # Apply LUT if provided
    if lut_path:
        applicator = LUTApplicator()
        lut_data, lut_size = applicator.read_cube_lut(lut_path)
        result = applicator.apply_lut_trilinear(log_image, lut_data, lut_size)
        print(f"Applied LUT from {lut_path} (size: {lut_size}x{lut_size}x{lut_size})")
        print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")
        return result
    
    return log_image


# Example usage
if __name__ == "__main__":
    # Create sample linear RGB data (simulating RAW)
    height, width = 1080, 1920
    linear_rgb = np.random.rand(height, width, 3).astype(np.float32) * 2.0  # HDR range
    
    # Available formats
    formats = ['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog']
    
    print("Linear to Log Converter - Example")
    print("=" * 60)
    
    for fmt in formats:
        print(f"\nProcessing with {fmt.upper()}...")
        log_output = process_image(linear_rgb, fmt)
        
        # If you have a LUT file, you can apply it like this:
        # log_with_lut = process_image(linear_rgb, fmt, lut_path='path/to/lut.cube')
    
    print("\n" + "=" * 60)
    print("Supported Log formats:")
    print("  - slog3: Sony S-Log3")
    print("  - logc4: ARRI LogC4") 
    print("  - clog3: Canon Log3")
    print("  - redlog3g10: RED Log3G10")
    print("  - vlog: Panasonic V-Log")
