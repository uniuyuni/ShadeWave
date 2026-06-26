"""Backward-compatible shim.

The Film Process model moved to ``effect_backends`` (adapter + NumPy reference +
native CPU backend). This module keeps the old dict-based ``apply_film_process``
API alive for existing callers/tests and forwards to the adapter. Do not add new
compute logic here.
"""

from effect_backends import film_process_adapter
from effect_backends.film_process_reference import FILM_MODES  # re-export for compat

__all__ = ["FILM_MODES", "apply_film_process"]


def apply_film_process(image, params=None):
    p = params or {}
    return film_process_adapter.apply_film_process(
        image,
        mode=p.get("film_mode", "Off"),
        latitude=p.get("film_latitude", 55.0),
        contrast=p.get("film_contrast", 50.0),
        color_bias=p.get("film_color_bias", 0.0),
        color_drift=p.get("film_color_drift", 0.0),
        dye_purity=p.get("film_dye_purity", 75.0),
        layer_crosstalk=p.get("film_layer_crosstalk", 30.0),
        halation=p.get("film_halation", 0.0),
        aging=p.get("film_aging", 0.0),
    )
