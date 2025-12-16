#!/usr/bin/env python3
"""
Test script for Log converters
Verifies that conversions produce expected results
"""

import numpy as np
from linear_to_log_lut import LogConverter
import sys


def test_conversion(converter, format_name, linear_value, expected_range=None):
    """Test a single conversion"""
    # Create test image with single value
    test_image = np.full((10, 10, 3), linear_value, dtype=np.float32)
    
    # Convert
    result = converter.convert(test_image, format_name)
    
    # Check result
    mean_value = result.mean()
    
    print(f"  Linear {linear_value:.4f} -> {format_name.upper()} {mean_value:.4f}", end="")
    
    if expected_range:
        if expected_range[0] <= mean_value <= expected_range[1]:
            print(" ✓")
            return True
        else:
            print(f" ✗ (expected {expected_range[0]:.4f}-{expected_range[1]:.4f})")
            return False
    else:
        print()
        return True


def run_tests():
    """Run all tests"""
    converter = LogConverter()
    formats = ['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog', 'nlog', 'flog2', 'omlog400']
    
    print("=" * 70)
    print("Testing Log Conversions")
    print("=" * 70)
    
    all_passed = True
    
    # Test critical values for each format
    test_cases = {
        'slog3': [
            (0.0, (0.0, 0.1)),      # Black
            (0.18, (0.35, 0.45)),   # 18% gray (middle gray)
            (1.0, (0.55, 0.65)),    # White
        ],
        'logc4': [
            (0.0, (0.09, 0.10)),     # Black
            (0.18, (0.27, 0.29)),    # 18% gray (should be 0.2784)
            (1.0, (0.42, 0.43)),     # White
        ],
        'clog3': [
            (0.0, (0.0, 0.1)),      # Black
            (0.18, (0.30, 0.40)),   # 18% gray
            (1.0, (0.55, 0.65)),    # White
        ],
        'redlog3g10': [
            (0.0, (0.09, 0.10)),    # Black
            (0.18, (0.33, 0.34)),   # 18% gray (should be exactly 1/3)
            (1.0, (0.49, 0.50)),    # White
        ],
        'vlog': [
            (0.0, (0.0, 0.15)),     # Black
            (0.18, (0.40, 0.50)),   # 18% gray
            (1.0, (0.55, 0.65)),    # White
        ],
        'nlog': [
            (0.0, (0.12, 0.13)),    # Black
            (0.18, (0.36, 0.37)),   # 18% gray
            (1.0, (0.60, 0.61)),    # White
        ],
        'flog2': [
            (0.0, (0.0, 0.05)),     # Black
            (0.18, (0.35, 0.42)),   # 18% gray
            (1.0, (0.50, 0.58)),    # White
        ],
        'omlog400': [
            (0.0, (0.0, 0.01)),     # Black
            (0.18, (0.32, 0.36)),   # 18% gray
            (1.0, (0.50, 0.54)),    # White
        ]
    }
    
    for fmt in formats:
        print(f"\n{fmt.upper()} Tests:")
        for linear_val, expected in test_cases[fmt]:
            passed = test_conversion(converter, fmt, linear_val, expected)
            if not passed:
                all_passed = False
    
    # Test HDR values (>1.0)
    print("\nHDR Tests (values > 1.0):")
    for fmt in formats:
        test_image = np.array([[[2.0, 4.0, 8.0]]], dtype=np.float32)
        result = converter.convert(test_image, fmt)
        print(f"  {fmt.upper():12s}: Linear [2.0, 4.0, 8.0] -> [{result[0,0,0]:.3f}, {result[0,0,1]:.3f}, {result[0,0,2]:.3f}]")
    
    # Test monotonicity (output should increase as input increases)
    print("\nMonotonicity Tests:")
    for fmt in formats:
        linear_values = np.linspace(0, 2, 20)
        test_images = linear_values[:, np.newaxis, np.newaxis, np.newaxis]
        test_images = np.repeat(test_images, 3, axis=-1).astype(np.float32)
        
        results = []
        for img in test_images:
            log_img = converter.convert(img, fmt)
            results.append(log_img[0, 0, 0])
        
        results = np.array(results)
        is_monotonic = np.all(results[1:] >= results[:-1])
        
        print(f"  {fmt.upper():12s}: {'✓ Monotonic' if is_monotonic else '✗ Not monotonic'}")
        if not is_monotonic:
            all_passed = False
    
    # Test output range
    print("\nOutput Range Tests (should be in [0, 1] or slightly above for HDR):")
    for fmt in formats:
        test_image = np.random.rand(100, 100, 3).astype(np.float32) * 2.0
        result = converter.convert(test_image, fmt)
        
        min_val, max_val = result.min(), result.max()
        in_range = (min_val >= 0 and max_val <= 1.5)  # Allow some HDR headroom
        
        print(f"  {fmt.upper():12s}: [{min_val:.4f}, {max_val:.4f}] {'✓' if in_range else '✗'}")
        if not in_range:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests PASSED")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1


def benchmark():
    """Quick performance benchmark"""
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    import time
    
    converter = LogConverter()
    formats = ['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog', 'nlog', 'flog2', 'omlog400']
    
    # Test with 4K image
    test_image = np.random.rand(2160, 3840, 3).astype(np.float32)
    
    print(f"\nTest image: 4K (2160x3840x3) = {test_image.size:,} values")
    print("\nConversion times:")
    
    for fmt in formats:
        start = time.time()
        result = converter.convert(test_image, fmt)
        elapsed = time.time() - start
        
        mpixels_per_sec = (2160 * 3840) / elapsed / 1e6
        print(f"  {fmt.upper():12s}: {elapsed*1000:6.2f} ms ({mpixels_per_sec:.1f} MP/s)")


if __name__ == "__main__":
    exit_code = run_tests()
    
    if exit_code == 0:
        benchmark()
    
    sys.exit(exit_code)
