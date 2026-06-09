"""
Draw Quick Select V2 entry point.

V2 is intentionally developed beside V1. It keeps the same public API so the UI
and matte composition can switch with an environment flag while the solver core
is rebuilt and compared against the corpus.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from cores.mask2 import draw_quick_select as _v1
from cores.mask2 import edge_refine as _er


DrawSupportResult = _v1.DrawSupportResult


def compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None,
        pixel_scale=1.0) -> DrawSupportResult:
    """Compute Draw Quick Select support with the V2 path.

    The first V2 milestone is a safe switchable entry point plus harness. Add-only
    strokes currently reuse the strongest V1 min-cut core while exposing V2 debug
    planes and timing. Erase stays on V1 until the separate erase profile is
    implemented.
    """
    t0 = time.perf_counter()
    strokes = _normalize_strokes(draw_strokes)
    has_erase = any(bool(getattr(stroke, "is_erasing", False)) for stroke in strokes)
    if has_erase:
        result = _v1.compute_draw_support(
            guide,
            mask,
            radius,
            strength,
            seed_mask=seed_mask,
            draw_strokes=draw_strokes,
            pixel_scale=pixel_scale,
        )
        return _tag_result(result, "v2_fallback_erase", t0)

    # V2 add-only MVP: keep the current min-cut result as the baseline while the
    # harness measures edge contact, zoom stability, and runtime. The next step is
    # to replace this call with V2's canonical edge-field/boundary solver without
    # touching UI or matte composition.
    result = _v1.compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=seed_mask,
        draw_strokes=draw_strokes,
        pixel_scale=pixel_scale,
    )
    return _tag_result(result, "v2_add_mvp", t0)


def _normalize_strokes(draw_strokes):
    if draw_strokes is None:
        return []
    try:
        return list(draw_strokes)
    except TypeError:
        return [draw_strokes]


def _tag_result(result: DrawSupportResult, mode: str, started_at: float) -> DrawSupportResult:
    if result.support.size:
        shape = result.support.shape
        elapsed = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        planes = list(result.debug_planes)
        planes.append(("v2_runtime_ms", np.full(shape, elapsed / 1000.0, dtype=np.float32)))
        planes.append(("v2_mode", np.full(shape, 1.0 if mode == "v2_add_mvp" else 0.0, dtype=np.float32)))
        logging.info("[DRAW_QS_V2] mode=%s runtime_ms=%.1f", mode, elapsed)
        return DrawSupportResult(result.seed, result.candidate, result.support, planes)
    return result


__all__ = ["DrawSupportResult", "compute_draw_support"]
