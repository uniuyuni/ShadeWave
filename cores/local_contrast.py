"""Compatibility shim for local contrast effects.

The implementations moved to ``effect_backends``. Keep this module as the old
import path for callers outside the effect pipeline.
"""

from effect_backends.local_contrast_adapter import (
    apply_clarity,
    apply_microcontrast,
    apply_texture,
)


__all__ = [
    "apply_clarity",
    "apply_texture",
    "apply_microcontrast",
]
