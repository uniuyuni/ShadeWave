#!/usr/bin/env python3
"""
Demo: Create synthetic linear RGB and convert to various Log formats
"""

import numpy as np
from linear_to_log_lut import LogConverter, process_image
import matplotlib.pyplot as plt


def create_test_gradient():
    """Create a test gradient from black to white with some HDR content"""
    width = 1920
    height = 200
    
    # Create horizontal gradient
    gradient = np.linspace(0, 2.0, width)  # 0 to 2.0 for HDR
    gradient = np.tile(gradient, (height, 1))
    gradient = np.stack([gradient] * 3, axis=-1)
    
    return gradient.astype(np.float32)


def create_color_bars():
    """Create color bars test pattern"""
    width = 1920
    height = 1080
    
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    bar_width = width // 7
    
    # Standard color bars (white, yellow, cyan, green, magenta, red, blue)
    colors = [
        [1.0, 1.0, 1.0],  # White
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 1.0],  # Cyan
        [0.0, 1.0, 0.0],  # Green
        [1.0, 0.0, 1.0],  # Magenta
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
    ]
    
    for i, color in enumerate(colors):
        start_x = i * bar_width
        end_x = min((i + 1) * bar_width, width)
        image[:, start_x:end_x] = color
    
    return image


def visualize_conversions():
    """Visualize the difference between linear and various Log formats"""
    # Create test gradient
    linear = create_test_gradient()
    
    converter = LogConverter()
    formats = ['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog', 'nlog', 'flog2', 'omlog400']
    
    # Create figure
    fig, axes = plt.subplots(len(formats) + 1, 1, figsize=(15, 12))
    
    # Plot linear
    axes[0].imshow(np.clip(linear / 2.0, 0, 1))  # Normalize for display
    axes[0].set_title('Linear RGB (0.0 to 2.0)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot each log format
    for i, fmt in enumerate(formats, 1):
        log_image = converter.convert(linear, fmt)
        axes[i].imshow(log_image)
        axes[i].set_title(f'{fmt.upper()}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/log_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: log_comparison.png")
    plt.close()


def compare_value_curves():
    """Compare how different Log curves encode the same linear values"""
    # Create linear values from 0 to 2.0
    linear_values = np.linspace(0, 2.0, 1000)
    
    converter = LogConverter()
    formats = ['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog', 'nlog', 'flog2', 'omlog400']
    
    plt.figure(figsize=(14, 8))
    
    # Plot each curve
    for fmt in formats:
        # Create test array
        test_array = linear_values[:, np.newaxis, np.newaxis]
        test_array = np.repeat(test_array, 3, axis=-1).astype(np.float32)
        
        # Convert
        log_values = []
        for val in test_array:
            log_val = converter.convert(val, fmt)
            log_values.append(log_val[0, 0])
        
        log_values = np.array(log_values)
        
        plt.plot(linear_values, log_values, label=fmt.upper(), linewidth=2)
    
    # Add linear reference
    plt.plot(linear_values, linear_values / 2.0, '--', color='gray', 
             label='Linear (normalized)', linewidth=1, alpha=0.5)
    
    plt.xlabel('Linear Value', fontsize=12)
    plt.ylabel('Log Encoded Value', fontsize=12)
    plt.title('Log Encoding Curves Comparison (8 Formats)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2.0)
    plt.ylim(0, 1.0)
    
    # Add annotations
    plt.axvline(x=0.18, color='red', linestyle=':', alpha=0.5, linewidth=1)
    plt.text(0.19, 0.05, '18% gray (0.18)', fontsize=9, color='red')
    
    plt.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5, linewidth=1)
    plt.text(1.01, 0.05, 'White (1.0)', fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/log_curves_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: log_curves_comparison.png")
    plt.close()


def main():
    print("=" * 70)
    print("Log Converter Demo - 8 Formats Supported")
    print("=" * 70)
    
    print("\n1. Creating test patterns...")
    gradient = create_test_gradient()
    color_bars = create_color_bars()
    
    print(f"   Gradient: {gradient.shape}, range [{gradient.min():.2f}, {gradient.max():.2f}]")
    print(f"   Color bars: {color_bars.shape}, range [{color_bars.min():.2f}, {color_bars.max():.2f}]")
    
    print("\n2. Converting gradient to all Log formats...")
    converter = LogConverter()
    formats = ['slog3', 'logc4', 'clog3', 'redlog3g10', 'vlog', 'nlog', 'flog2', 'omlog400']
    
    for fmt in formats:
        log_image = converter.convert(gradient, fmt)
        print(f"   {fmt.upper():12s}: range [{log_image.min():.4f}, {log_image.max():.4f}]")
    
    print("\n3. Generating visualizations...")
    try:
        visualize_conversions()
        compare_value_curves()
        print("\n✓ Visualizations created successfully!")
    except Exception as e:
        print(f"\n⚠ Could not create visualizations: {e}")
        print("  (matplotlib may not be installed)")
    
    print("\n4. Example: Processing with LUT...")
    print("   To process with LUT:")
    print("   >>> result = process_image(linear_rgb, 'slog3', lut_path='my_lut.cube')")
    
    print("\n5. Command line usage:")
    print("   python example_usage.py input.exr output.tiff --log slog3")
    print("   python example_usage.py input.exr output.tiff --log nlog --lut rec709.cube")
    
    print("\n6. Supported formats:")
    print("   - slog3: Sony S-Log3")
    print("   - logc4: ARRI LogC4")
    print("   - clog3: Canon Log3")
    print("   - redlog3g10: RED Log3G10")
    print("   - vlog: Panasonic V-Log")
    print("   - nlog: Nikon N-Log")
    print("   - flog2: Fujifilm F-Log2")
    print("   - omlog400: OM SYSTEM OM-Log400")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
