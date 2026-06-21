"""Python-facing colour functions adapter.

This module exposes the historical colour_functions API and overrides the
display transform hot path with a native backend when available.
"""

from __future__ import annotations

import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    native_backend_enabled,
    optional_backend,
    strict_enabled,
)
from . import colour_functions_reference as reference
from .colour_functions_reference import *  # noqa: F401,F403


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_colour_functions_cpu")


_ENCODING_CODES = {
    "linear": 0,
    "srgb": 1,
    "rec709": 2,
    "rec2020": 3,
    "gamma-adobe-rgb": 4,
    "gamma-1.8": 5,
    "gamma-2.2": 6,
    "gamma-2.6": 7,
    "prophoto": 8,
}


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_COLOUR_FUNCTIONS_BACKEND")


def native_enabled() -> bool:
    return native_backend_enabled(_cpu_backend, _backend_preference())


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("colour_functions", "effect_backends._colour_functions_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "colour_functions",
            "effect_backends.colour_functions_reference",
            False,
            "cpu backend available; PLATYPUS_COLOUR_FUNCTIONS_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("colour_functions", "effect_backends.colour_functions_reference", False, detail)


def encoding_code(encoding: str) -> int:
    key = str(encoding).strip().lower()
    try:
        return _ENCODING_CODES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported display encoding: {encoding}") from exc


def apply_display_color_transform(
    image: np.ndarray,
    basis: np.ndarray,
    output_colourspace,
    luminance_weights=(0.2126, 0.7152, 0.0722),
    eps: float = 1e-12,
) -> np.ndarray:
    image32 = np.asarray(image, dtype=np.float32)
    basis32 = np.asarray(basis, dtype=np.float32)
    encoding = reference._get_encoding(output_colourspace)

    if native_enabled() and image32.ndim == 3 and image32.shape[-1] == 3 and basis32.shape == (3, 3):
        try:
            return _cpu_backend.apply_display_color_transform(
                np.ascontiguousarray(image32),
                np.ascontiguousarray(basis32),
                encoding_code(encoding),
                luminance_weights,
                float(eps),
            )
        except Exception:
            if _native_strict():
                raise

    return reference.apply_display_color_transform(image32, basis32, output_colourspace)


def display_color_transform_basis(
    input_colourspace,
    output_colourspace,
    chromatic_adaptation_transform: str = "CAT02",
    dtype=np.float32,
) -> np.ndarray:
    return reference.display_color_transform_basis(
        input_colourspace,
        output_colourspace,
        chromatic_adaptation_transform,
        dtype,
    )


def display_color_transform(
    image: np.ndarray,
    input_colourspace,
    output_colourspace,
    chromatic_adaptation_transform: str = "CAT02",
) -> np.ndarray:
    basis = display_color_transform_basis(
        input_colourspace,
        output_colourspace,
        chromatic_adaptation_transform,
    )
    return apply_display_color_transform(image, basis, output_colourspace)


def encode_display_output(rgb: np.ndarray, colourspace) -> np.ndarray:
    return reference.encode_display_output(rgb, colourspace)


def _native_strict() -> bool:
    return strict_enabled("PLATYPUS_COLOUR_FUNCTIONS_STRICT")


__all__ = [
    name for name in dir(reference)
    if not name.startswith("_")
] + [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "encoding_code",
    "encode_display_output",
    "display_color_transform_basis",
    "apply_display_color_transform",
    "display_color_transform",
]
