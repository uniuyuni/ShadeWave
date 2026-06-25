"""NumPy reference implementation of 3D LUT trilinear application.

This is the authoritative numerical baseline and the fallback used when the
compiled Metal backend is unavailable. The logic is a 1:1 port of the original
``cores.lut_functions.LUT3D.apply`` / ``_trilinear_interpolation`` so existing
output is preserved bit-for-bit (within float ordering).

BGR index convention (must match the kernel):
  Input RGB grid coords g = (gR, gG, gB) = norm * (size - 1).
  table[a, b, c] (C-order, flat = ((a*size + b)*size + c)*3) is addressed with
  a = floor(gB), b = floor(gG), c = floor(gR); interpolation weights are
  axis0 = frac(gB), axis1 = frac(gG), axis2 = frac(gR).
"""

from __future__ import annotations

import numpy as np


def _trilinear(table: np.ndarray, size: int, grid_coords: np.ndarray) -> np.ndarray:
    # Clamp to valid range (should already be in range due to domain clipping)
    coords = np.clip(grid_coords, 0, size - 1)

    coords_floor = np.floor(coords).astype(np.int32)
    coords_floor = np.clip(coords_floor, 0, size - 2)
    coords_ceil = coords_floor + 1

    coords_frac = coords - coords_floor

    # colour library stores .cube data with BGR indexing:
    # input RGB [R, G, B] maps to table indices [B, G, R].
    r0, g0, b0 = coords_floor[:, 2], coords_floor[:, 1], coords_floor[:, 0]
    r1, g1, b1 = coords_ceil[:, 2], coords_ceil[:, 1], coords_ceil[:, 0]

    rd, gd, bd = coords_frac[:, 2:3], coords_frac[:, 1:2], coords_frac[:, 0:1]

    c000 = table[r0, g0, b0]
    c001 = table[r0, g0, b1]
    c010 = table[r0, g1, b0]
    c011 = table[r0, g1, b1]
    c100 = table[r1, g0, b0]
    c101 = table[r1, g0, b1]
    c110 = table[r1, g1, b0]
    c111 = table[r1, g1, b1]

    c00 = c000 * (1 - rd) + c100 * rd
    c01 = c001 * (1 - rd) + c101 * rd
    c10 = c010 * (1 - rd) + c110 * rd
    c11 = c011 * (1 - rd) + c111 * rd

    c0 = c00 * (1 - gd) + c10 * gd
    c1 = c01 * (1 - gd) + c11 * gd

    return c0 * (1 - bd) + c1 * bd


def apply_lut3d(image: np.ndarray, table: np.ndarray, domain: np.ndarray, size: int) -> np.ndarray:
    """Apply a 3D LUT to ``image`` with trilinear interpolation.

    image:  (..., 3) float32, current pipeline color space.
    table:  (size, size, size, 3) float32.
    domain: (2, 3) float32 -> [[min_r, min_g, min_b], [max_r, max_g, max_b]].
    size:   LUT cube side length.
    Returns a new float32 array with the same shape as ``image``.
    """
    rgb = np.asarray(image, dtype=np.float32)
    table = np.asarray(table, dtype=np.float32)
    domain = np.asarray(domain, dtype=np.float32)
    original_shape = rgb.shape

    rgb_flat = rgb.reshape(-1, 3)

    domain_min = domain[0]
    domain_max = domain[1]

    # Clip to domain range (colour library behavior), then normalize to [0, 1].
    rgb_clipped = np.clip(rgb_flat, domain_min, domain_max)
    rgb_norm = (rgb_clipped - domain_min) / (domain_max - domain_min)
    grid_coords = rgb_norm * (size - 1)

    out = _trilinear(table, size, grid_coords)
    return out.reshape(original_shape).astype(np.float32)
