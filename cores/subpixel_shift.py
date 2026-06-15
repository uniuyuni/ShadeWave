"""Compatibility shims for the Subpixel Shift backend."""

from __future__ import annotations

from effect_backends import subpixel_shift_adapter


def subpixel_shift(img_array, shift_x=0.5, shift_y=0.5):
    return subpixel_shift_adapter.subpixel_shift(img_array, shift_x, shift_y)


def create_enhanced_image(img_array):
    return subpixel_shift_adapter.create_enhanced_image(img_array)
