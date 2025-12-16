#!/usr/bin/env python3
"""
Example: Load linear RGB, convert to Log, apply LUT, and save
"""

import numpy as np
from linear_to_log_lut import LogConverter, LUTApplicator, process_image
import argparse


def load_linear_exr(filepath: str) -> np.ndarray:
    """
    Load linear RGB from EXR file
    Requires: pip install OpenEXR Imath (or use imageio with freeimage plugin)
    """
    try:
        import imageio.v3 as iio
        image = iio.imread(filepath)
        return image.astype(np.float32)
    except ImportError:
        raise ImportError("Please install imageio: pip install imageio")


def load_linear_tiff(filepath: str) -> np.ndarray:
    """
    Load linear RGB from 16-bit or 32-bit TIFF
    """
    try:
        import imageio.v3 as iio
        image = iio.imread(filepath)
        
        if image.dtype == np.uint16:
            # Normalize 16-bit to float
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            # Normalize 8-bit to float
            image = image.astype(np.float32) / 255.0
        elif image.dtype in [np.float32, np.float64]:
            image = image.astype(np.float32)
        
        return image
    except ImportError:
        raise ImportError("Please install imageio: pip install imageio")


def save_image(image: np.ndarray, filepath: str, bit_depth: int = 16):
    """
    Save image to file
    
    Args:
        image: Float32 image in range [0, 1]
        filepath: Output path
        bit_depth: 8 or 16 for PNG/TIFF, 32 for float TIFF
    """
    try:
        import imageio.v3 as iio
        
        image = np.clip(image, 0, 1)
        
        if filepath.lower().endswith('.exr'):
            # Save as EXR (float32)
            iio.imwrite(filepath, image.astype(np.float32))
        elif bit_depth == 8:
            # Save as 8-bit
            image_8bit = (image * 255).astype(np.uint8)
            iio.imwrite(filepath, image_8bit)
        elif bit_depth == 16:
            # Save as 16-bit
            image_16bit = (image * 65535).astype(np.uint16)
            iio.imwrite(filepath, image_16bit)
        elif bit_depth == 32:
            # Save as 32-bit float TIFF
            iio.imwrite(filepath, image.astype(np.float32))
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        print(f"Saved: {filepath}")
    except ImportError:
        raise ImportError("Please install imageio: pip install imageio")


def main():
    parser = argparse.ArgumentParser(
        description='Convert linear RGB to Log format and apply LUT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to S-Log3 only
  python example_usage.py input.exr output.tiff --log slog3
  
  # Convert to LogC4 and apply LUT
  python example_usage.py input.exr output.tiff --log logc4 --lut my_lut.cube
  
  # Save as 8-bit PNG
  python example_usage.py input.tiff output.png --log clog3 --bit-depth 8

Supported Log formats:
  slog3       - Sony S-Log3
  logc4       - ARRI LogC4
  clog3       - Canon Log3
  redlog3g10  - RED Log3G10
  vlog        - Panasonic V-Log
  nlog        - Nikon N-Log
  flog2       - Fujifilm F-Log2
  omlog400    - OM SYSTEM OM-Log400
        """
    )
    
    parser.add_argument('input', help='Input linear RGB image (EXR, TIFF)')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--log', required=True, 
                       choices=['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog', 'nlog', 'flog2', 'omlog400'],
                       help='Target log format')
    parser.add_argument('--lut', help='Path to .cube LUT file (optional)')
    parser.add_argument('--bit-depth', type=int, default=16, choices=[8, 16, 32],
                       help='Output bit depth (default: 16)')
    
    args = parser.parse_args()
    
    print(f"Loading: {args.input}")
    
    # Load image based on extension
    if args.input.lower().endswith('.exr'):
        linear_rgb = load_linear_exr(args.input)
    else:
        linear_rgb = load_linear_tiff(args.input)
    
    print(f"  Shape: {linear_rgb.shape}")
    print(f"  Range: [{linear_rgb.min():.4f}, {linear_rgb.max():.4f}]")
    
    # Process
    result = process_image(linear_rgb, args.log, args.lut)
    
    # Save
    save_image(result, args.output, args.bit_depth)
    print("\nDone!")


if __name__ == "__main__":
    main()
