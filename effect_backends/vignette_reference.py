"""Reference Vignette implementation used for fallback and parity tests."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from threads import lock_numba


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def apply_vignette(image, intensity, radius_percent, disp_info, crop_rect, offset, gradient_softness=4.0):
    intensity = intensity / 100.0
    radius_percent = radius_percent / 100.0
    gradient_softness = max(0.1, gradient_softness)

    h, w = image.shape[:2]

    dx, dy, _, _, scale = disp_info

    x1, y1, x2, y2 = crop_rect
    offset_x, offset_y = offset

    center_x = (x1 + (x2 - x1) / 2 - dx) * scale + offset_x
    center_y = (y1 + (y2 - y1) / 2 - dy) * scale + offset_y

    mm = max((x2 - x1), (y2 - y1)) * scale
    max_radius = math.sqrt(mm**2 + mm**2) / 2

    radius = max_radius * radius_percent

    res = np.empty_like(image)

    c = 3
    if image.ndim == 2:
        c = 1

    for y in prange(h):
        for x in prange(w):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            val = dist / radius
            if val > 1.0:
                val = 1.0
            elif val < 0.0:
                val = 0.0

            mask = val ** gradient_softness
            mask = mask * mask * (3 - 2 * mask)

            if intensity < 0:
                vig = 1.0 + intensity * mask
                if c == 3:
                    for k in range(3):
                        res[y, x, k] = image[y, x, k] * vig
                else:
                    res[y, x] = image[y, x] * vig
            else:
                vig = 1.0 - intensity * mask
                if c == 3:
                    for k in range(3):
                        v = image[y, x, k]
                        res[y, x, k] = v + (1.0 - v) * (1.0 - vig)
                else:
                    v = image[y, x]
                    res[y, x] = v + (1.0 - v) * (1.0 - vig)
    return res
