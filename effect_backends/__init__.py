"""Optional effect backends for Platypus.

Each effect owns a small Python module that exposes the stable call surface
used by the current Python pipeline. CPU-native, GPU, or future external
runtime implementations can then be swapped in effect-by-effect without
changing the high-level Effect classes.
"""

__all__ = [
    "vignette_adapter",
    "color_separation_adapter",
    "film_grain_adapter",
    "cross_filter_adapter",
    "dehaze_adapter",
    "image_transform_adapter",
    "local_contrast_adapter",
    "colour_functions_adapter",
    "tone_adapter",
    "film_process_adapter",
    "subpixel_shift_adapter",
    "low_frequency_transfer_adapter",
    "lut_adapter",
    "lens_blur_adapter",
]
