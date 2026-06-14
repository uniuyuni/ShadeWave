"""
Draw Quick Select V3.

V3 keeps V2's graph solve, then separates the two things that were fighting in
V2:

* binary support: what region belongs to the stroke
* support alpha: how hard the selected edge should appear on soft/weak edges

This lets Edge Bias affect edge feel without blindly dilating the support mask.
"""
from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np

from cores.mask2 import draw_quick_select as _v1
from cores.mask2 import draw_quick_select_v2 as _v2
from cores.mask2 import edge_refine as _er
from cores.mask2 import mask_rasters


DrawSupportResult = _v1.DrawSupportResult

# Guide-edge 1-slot cache: the guide image is stable across strokes on the
# same photo. Edge strength is ~40ms on 1M-pixel images; caching it avoids
# recomputing it 4-5× per stroke (trim, alpha, V4 trace, V4 trim, …).
_GUIDE_EDGE_CACHE: dict = {}  # {(buffer_ptr, shape): edge}


def _get_guide_edge(guide, shape):
    """Return cached edge-strength map for *guide* at *shape*, computing once."""
    try:
        key = (guide.ctypes.data, shape)
    except AttributeError:
        key = None
    if key is not None and key in _GUIDE_EDGE_CACHE:
        return _GUIDE_EDGE_CACHE[key]
    g = _er._prepare_guide_image(guide, shape)
    edge = _er._draw_snap_edge_strength(g) if g is not None else None
    if key is not None:
        _GUIDE_EDGE_CACHE.clear()  # keep only 1 entry
        _GUIDE_EDGE_CACHE[key] = edge
    return edge


def compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None,
        pixel_scale=1.0,
        edge_bias=0.0) -> DrawSupportResult:
    t0 = time.perf_counter()
    mask_f = _er._as_mask(mask)
    hint = mask_f > 0.02
    h, w = hint.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    strokes = _normalize_strokes(draw_strokes)
    add_strokes = [stroke for stroke in strokes if not bool(getattr(stroke, "is_erasing", False))]
    if not add_strokes:
        # Erase-only / empty stays on the V1 add+erase pipeline: there is no add
        # stroke to anchor a per-stroke solve against.
        base = _v2.compute_draw_support(
            guide,
            mask_f,
            radius,
            strength,
            seed_mask=seed_mask,
            draw_strokes=draw_strokes,
            pixel_scale=pixel_scale,
            edge_bias=edge_bias,
        )
        return _with_v3_alpha(guide, base, mask_f, draw_strokes, edge_bias, t0)

    _maybe_dump_v3_input(
        guide,
        mask_f,
        radius,
        strength,
        seed_mask,
        draw_strokes,
        pixel_scale,
        edge_bias,
    )

    all_erase = _erase_stroke_mask(hint.shape, strokes)
    seed_all = np.zeros((h, w), dtype=bool)
    candidate_all = np.zeros((h, w), dtype=bool)
    support_all = np.zeros((h, w), dtype=bool)
    gap_fill_all = np.zeros((h, w), dtype=bool)
    boundary_bias_all = np.zeros((h, w), dtype=bool)
    alpha_all = np.zeros((h, w), dtype=np.float32)
    plane_accum = {}
    plane_weight = np.zeros((h, w), dtype=bool)
    solved_count = 0

    for stroke_index, stroke in enumerate(strokes):
        if bool(getattr(stroke, "is_erasing", False)):
            continue
        # Only erases drawn *after* this add can undo it; an add drawn after an
        # erase must win in the overlap (draw -> erase -> draw again). So carve
        # and subtract by future erases only, preserving temporal stroke order.
        future_erases = [
            s for s in strokes[stroke_index + 1:]
            if bool(getattr(s, "is_erasing", False))
        ]
        erase = _erase_stroke_mask(hint.shape, future_erases)
        stroke_mask = _single_stroke_mask((w, h), stroke)
        stroke_mask_full = stroke_mask
        if np.any(erase):
            # Carve the future-erased region out of this add stroke's footprint so
            # the FG seed never overlaps the erase BG seed (same-colour overlap
            # would otherwise read the whole stroke as background).
            stroke_mask = np.where(erase, 0.0, stroke_mask).astype(np.float32, copy=False)
        if not np.any(stroke_mask > 0.02):
            continue
        stroke_seed = _er.make_confident_seed(stroke_mask)
        result = _v2._compute_add_only_support(
            guide,
            stroke_mask,
            radius,
            strength,
            seed_mask=stroke_seed,
            draw_strokes=[stroke] + future_erases,
            pixel_scale=pixel_scale,
            edge_bias=0.0,
            _dump_input=False,
        )
        if result.support.size == 0:
            continue
        support = np.asarray(result.support, dtype=bool) & ~erase
        if np.any(erase):
            support = _snap_kept_boundary_to_edges(
                guide, support, erase, strength, getattr(stroke, "size", 16.0))
        candidate = np.asarray(result.candidate, dtype=bool)
        seed = np.asarray(result.seed, dtype=bool)
        planes = {name: value for name, value in result.debug_planes}
        support, gap_fill = _fill_selected_color_voids(
            support,
            stroke_mask > 0.02,
            seed,
            erase,
            planes,
        )
        support, boundary_bias_delta = _apply_boundary_bias(
            support,
            candidate,
            seed,
            erase,
            planes,
            edge_bias,
        )
        if np.any(erase):
            # Erase is a *local* correction: carving the seed + feeding the erase
            # as a background seed re-routes the whole band-limited min-cut (and the
            # void-fill / boundary-bias that follow), so the boundary can move many
            # pixels away from where the user erased. Bound that: resolve+refine the
            # add *without* the erase (the region the user already sees) and only
            # keep the erase-affected result within a small reach of the erase
            # footprint. Outside that reach the boundary is the untouched no-erase
            # one, so an erase never affects pixels away from the brush.
            support = _localize_erase_effect(
                guide, support, stroke, stroke_mask_full, erase,
                radius, strength, pixel_scale, edge_bias)
        support_alpha = _support_alpha_from_edge_softness(
            guide,
            support,
            planes,
            edge_bias=edge_bias,
        )
        seed_all |= seed
        candidate_all |= candidate
        support_all |= support
        gap_fill_all |= gap_fill
        boundary_bias_all |= boundary_bias_delta
        alpha_all = np.maximum(alpha_all, support_alpha)
        _accumulate_debug_planes(plane_accum, plane_weight, result.debug_planes, support | candidate)
        solved_count += 1

    if solved_count == 0:
        return DrawSupportResult(empty, empty, empty.copy(), [])

    # Drop the 1px brush-circle rim left floating past object edges (before alpha
    # so the matte follows the trimmed boundary). Guide edge is already cached.
    _guide_edge = _get_guide_edge(guide, (h, w))
    support_all = _trim_offedge_brush_rim(support_all, mask_f, guide, _edge=_guide_edge)

    # No global erase subtract: temporal order is already handled per add stroke
    # via future-erase carving, so a later add wins over an earlier erase.
    net_erased = all_erase & ~support_all
    alpha_all = np.where(support_all & (alpha_all <= 1e-6), 1.0, alpha_all)
    alpha_all = np.where(support_all, alpha_all, 0.0).astype(np.float32, copy=False)
    out_planes = _finalize_debug_planes(plane_accum, support_all.shape)
    out_planes.extend([
        ("v3_same_color_void_fill", gap_fill_all),
        ("v3_boundary_bias_delta", boundary_bias_all),
        ("v3_erase_support", net_erased),
        ("boundary_bias_px", np.full(support_all.shape, float(edge_bias), dtype=np.float32)),
        ("support_alpha", alpha_all),
        ("v3_stroke_count", np.full(support_all.shape, float(solved_count), dtype=np.float32)),
        ("v3_runtime_ms", np.full(
            support_all.shape,
            max(0.0, (time.perf_counter() - t0) * 1000.0) / 1000.0,
            dtype=np.float32,
        )),
    ])
    logging.info(
        "[DRAW_QS_V3] support=%d gap_fill=%d alpha_soft=%d runtime_ms=%.1f",
        int(np.count_nonzero(support_all)),
        int(np.count_nonzero(gap_fill_all)),
        int(np.count_nonzero((alpha_all > 1e-4) & (alpha_all < 0.999))),
        (time.perf_counter() - t0) * 1000.0,
    )
    return DrawSupportResult(seed_all, candidate_all, support_all, out_planes)


def _with_v3_alpha(guide, base, mask, draw_strokes, edge_bias, started_at):
    if base.support.size == 0:
        return base
    hint = _er._as_mask(mask) > 0.02
    erase = _erase_stroke_mask(hint.shape, draw_strokes)
    planes = {name: value for name, value in base.debug_planes}
    support, gap_fill = _fill_selected_color_voids(
        np.asarray(base.support, dtype=bool),
        hint,
        np.asarray(base.seed, dtype=bool),
        erase,
        planes,
    )
    support, boundary_bias_delta = _apply_boundary_bias(
        support,
        np.asarray(base.candidate, dtype=bool),
        np.asarray(base.seed, dtype=bool),
        erase,
        planes,
        edge_bias,
    )
    support_alpha = _support_alpha_from_edge_softness(guide, support, planes, edge_bias=edge_bias)
    out_planes = list(base.debug_planes)
    out_planes.extend([
        ("v3_same_color_void_fill", gap_fill),
        ("v3_boundary_bias_delta", boundary_bias_delta),
        ("boundary_bias_px", np.full(support.shape, float(edge_bias), dtype=np.float32)),
        ("support_alpha", support_alpha),
        ("v3_stroke_count", np.full(support.shape, 0.0, dtype=np.float32)),
        ("v3_runtime_ms", np.full(
            support.shape,
            max(0.0, (time.perf_counter() - started_at) * 1000.0) / 1000.0,
            dtype=np.float32,
        )),
    ])
    return DrawSupportResult(base.seed, base.candidate, support, out_planes)


def _normalize_strokes(draw_strokes):
    if draw_strokes is None:
        return []
    try:
        return list(draw_strokes)
    except TypeError:
        return [draw_strokes]


def _maybe_dump_v3_input(
        guide,
        mask,
        radius,
        strength,
        seed_mask,
        draw_strokes,
        pixel_scale,
        edge_bias):
    if not (
            _v1.os.environ.get("QS_DUMP_INPUT")
            or _v1.os.environ.get("PLATYPUS_DEBUG_EDGE_REFINE", "").strip().lower()
            in {"1", "true", "yes", "on"}):
        return
    mask_f = _er._as_mask(mask)
    hint = mask_f > 0.02
    prepared = _er._prepare_guide_image(guide, hint.shape)
    edge = _er._draw_snap_edge_strength(prepared)
    if edge is None:
        edge = np.zeros(hint.shape, dtype=np.float32)
    effective, auto, offset, mode = _v2._resolve_edge_lock(strength, edge, hint)
    _v1._maybe_dump_input(
        prepared,
        mask_f,
        radius,
        strength,
        seed_mask,
        draw_strokes,
        pixel_scale,
        strength_mode=mode,
        edge_lock_auto=auto,
        edge_lock_effective=effective,
        edge_lock_offset=offset,
        edge_bias=edge_bias,
    )


def _single_stroke_mask(image_size, stroke):
    line = mask_rasters.Line(False, getattr(stroke, "size", 1.0), getattr(stroke, "soft", 100.0))
    _pts = getattr(stroke, "points", None)
    for x, y in (_pts if _pts is not None else []):
        line.add_point(float(x), float(y))
    if not line.points:
        return np.zeros((int(image_size[1]), int(image_size[0])), dtype=np.float32)
    return mask_rasters.draw_line_texture(
        image_size,
        [line],
        allow_over_one=False,
        allow_under_zero=False,
    )


def _accumulate_debug_planes(accum, _weight, debug_planes, scope):
    scope = np.asarray(scope, dtype=bool)
    for name, plane in debug_planes or []:
        arr = np.asarray(plane)
        if arr.ndim != 2 or arr.shape != scope.shape:
            continue
        current = accum.get(name)
        if current is None:
            dtype = bool if arr.dtype == np.bool_ else np.float32
            current = np.zeros(scope.shape, dtype=dtype)
            accum[name] = current
        if current.dtype == np.bool_:
            current |= scope & arr.astype(bool, copy=False)
        else:
            current[scope] = np.maximum(current[scope], arr.astype(np.float32, copy=False)[scope])


def _finalize_debug_planes(accum, shape):
    planes = []
    for name, plane in accum.items():
        arr = np.asarray(plane)
        if arr.shape == tuple(shape):
            planes.append((name, arr))
    return planes


def _trim_offedge_brush_rim(support, mask, guide, _edge=None):
    """Drop the 1px selection rim that merely traces the brush footprint.

    Where the add brush overhangs an object edge into the background, the min-cut
    cuts the bulk of the overhang but a 1px arc on the *brush circle* survives
    (the geometric FG prior keeps the outermost band pixel even though it is
    background) -- a circular sliver floating past the edge. Remove the outermost
    1px of support that (a) sits on the brush footprint boundary and (b) is NOT on
    any image edge: a real boundary lands on an edge and is kept, the floating
    brush arc is not. Purely geometric (no colour test) so it cannot misfire on
    low-contrast subjects where FG/BG colours are close.
    """
    support = np.asarray(support, dtype=bool)
    hint = _er._as_mask(mask) > 0.02
    if not np.any(support) or not np.any(hint):
        return support
    edge = _edge if _edge is not None else _get_guide_edge(guide, support.shape)
    if edge is None:
        return support
    k = np.ones((3, 3), dtype=np.uint8)
    # 0.08 separates the floating brush arc (sky/background, edge ~0 -> trimmed)
    # from genuine low-contrast boundaries (weak but nonzero edge -> spared).
    edge_t = float(os.environ.get("QS_RIM_EDGE_T", "0.08"))
    on_edge = cv2.dilate((np.asarray(edge, np.float32) > edge_t).astype(np.uint8),
                         k, iterations=2) > 0
    hint_b = hint & ~(cv2.erode(hint.astype(np.uint8), k, iterations=2) > 0)
    brush_zone = cv2.dilate(hint_b.astype(np.uint8), k, iterations=1) > 0
    ring = support & ~(cv2.erode(support.astype(np.uint8), k, iterations=1) > 0)
    residue = ring & brush_zone & ~on_edge
    return support & ~residue


def _erase_reach(erase, brush_size, radius):
    """How far (px) outside the erase footprint the erase is allowed to move the
    boundary: just enough for a local edge snap, never a band-wide reroute."""
    try:
        snap_r = float(np.clip(float(brush_size) * 0.30, 3.0, 10.0))
    except Exception:
        snap_r = 6.0
    try:
        rpx = max(0.0, float(radius))
    except Exception:
        rpx = 0.0
    return int(round(np.clip(snap_r + rpx, 3.0, 24.0)))


def _localize_erase_effect(
        guide, support_erase, stroke, stroke_mask_full, erase,
        radius, strength, pixel_scale, edge_bias):
    """Confine an erase's influence to a small reach around its footprint.

    ``support_erase`` is the (already refined) add solved *with* the erase as
    background -- it can reroute far away. We solve+refine the add *without* the
    erase to reproduce the boundary the user already sees, then keep the
    erase-affected result only within ``dilate(erase, reach)``; everywhere else we
    restore the no-erase boundary. The no-erase reference is run through the same
    void-fill / boundary-bias steps so the two agree away from the erase.
    """
    support_erase = np.asarray(support_erase, dtype=bool)
    erase = np.asarray(erase, dtype=bool)
    if not np.any(erase):
        return support_erase
    try:
        result_ne = _v2._compute_add_only_support(
            guide,
            stroke_mask_full,
            radius,
            strength,
            seed_mask=_er.make_confident_seed(stroke_mask_full),
            draw_strokes=[stroke],
            pixel_scale=pixel_scale,
            edge_bias=0.0,
            _dump_input=False,
        )
    except Exception:
        return support_erase
    if result_ne.support.size == 0:
        return support_erase
    empty = np.zeros_like(erase)
    support_ne = np.asarray(result_ne.support, dtype=bool)
    seed_ne = np.asarray(result_ne.seed, dtype=bool)
    candidate_ne = np.asarray(result_ne.candidate, dtype=bool)
    planes_ne = {name: value for name, value in result_ne.debug_planes}
    support_ne, _ = _fill_selected_color_voids(
        support_ne, stroke_mask_full > 0.02, seed_ne, empty, planes_ne)
    support_ne, _ = _apply_boundary_bias(
        support_ne, candidate_ne, seed_ne, empty, planes_ne, edge_bias)
    reach = _erase_reach(erase, getattr(stroke, "size", 16.0), radius)
    near = cv2.dilate(
        erase.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=reach) > 0
    return np.where(near, support_erase, support_ne)


def _snap_kept_boundary_to_edges(guide, support, erase, strength, brush_size):
    """Snap the kept/erased boundary out to a nearby strong edge.

    After an erase the kept boundary sits on the (rough) brush edge. If a strong
    image edge lies just inside the erased region, grow the kept region out to it
    -- but only when an edge is actually reached, so a sloppy erase over flat
    colour, or a deliberate erase well past an edge, is honoured exactly (no blind
    regrow). The snap reach is bounded by the brush size, so only small overshoots
    snap; large erases are kept as drawn.
    """
    support = np.asarray(support, dtype=bool)
    erase = np.asarray(erase, dtype=bool)
    if not np.any(support) or not np.any(erase):
        return support
    guide_img = _er._prepare_guide_image(guide, support.shape)
    if guide_img is None:
        return support
    barrier = _er._make_edge_stop_mask(guide_img, _er._draw_barrier_strength(strength))
    if barrier is None or not np.any(barrier):
        return support

    try:
        snap_r = int(round(float(np.clip(float(brush_size) * 0.30, 3.0, 10.0))))
    except Exception:
        snap_r = 6
    kernel = np.ones((3, 3), dtype=np.uint8)
    near = cv2.dilate(support.astype(np.uint8), kernel, iterations=snap_r) > 0
    band = near & erase & ~support
    if not np.any(band):
        return support

    free = (band & ~barrier) | support
    grown = _er._connected_to_seed(free, support) & band
    if not np.any(grown):
        return support

    # Keep only grown components that actually reach an edge (a real snap target);
    # drop blind growth over flat colour so a sloppy erase is honoured.
    barrier_near = cv2.dilate(barrier.astype(np.uint8), kernel, iterations=2) > 0
    n_labels, labels = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(grown)
    for i in range(1, n_labels):
        comp = labels == i
        if np.any(comp & barrier_near):
            keep |= comp
    if not np.any(keep):
        return support
    return support | keep


def _erase_stroke_mask(shape, draw_strokes):
    erase = np.zeros(tuple(shape[:2]), dtype=bool)
    for stroke in draw_strokes or []:
        if not bool(getattr(stroke, "is_erasing", False)):
            continue
        points = _er._stroke_points_array(stroke)
        if points.shape[0] == 0:
            continue
        try:
            size = float(max(1.0, getattr(stroke, "size", 1.0)))
        except Exception:
            size = 1.0
        erase |= _er._stroke_brush_mask(erase.shape, points, size)
    return erase


def _fill_selected_color_voids(
        support,
        hint,
        seed,
        erase,
        planes):
    """Include same-side sky/cloud gaps trapped inside a busy silhouette."""
    support = np.asarray(support, dtype=bool)
    hint = np.asarray(hint, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    erase = np.asarray(erase, dtype=bool)
    fill = np.zeros_like(support, dtype=bool)
    gap = hint & ~support & ~erase
    if not np.any(gap) or not np.any(support):
        return support, fill

    color_plane = planes.get("color_score")
    if color_plane is None:
        return support, fill
    color = np.asarray(color_plane, dtype=np.float32) * 2.0 - 1.0
    if color.shape != support.shape:
        return support, fill

    edge = np.asarray(planes.get("context_edge", np.zeros_like(color)), dtype=np.float32)
    if edge.shape != support.shape:
        edge = np.zeros_like(color, dtype=np.float32)

    edge_lock = _plane_max_percent(planes, "edge_lock_effective", fallback=60.0)
    lock = float(np.clip(edge_lock, 0.0, 100.0)) / 100.0
    color_min = float(np.clip(0.24 - 0.16 * lock, 0.06, 0.24))
    component_median_min = float(np.clip(0.30 - 0.18 * lock, 0.10, 0.30))
    max_dist = float(np.clip(8.0 + 18.0 * lock, 8.0, 28.0))
    edge_min = float(np.clip(0.22 - 0.11 * lock, 0.10, 0.22))

    dist_to_support = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 3)
    edge_near = edge >= edge_min
    if np.any(edge_near):
        edge_near = cv2.dilate(
            edge_near.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=2,
        ) > 0
    candidate = gap & (dist_to_support <= max_dist) & (color >= color_min) & edge_near
    if not np.any(candidate):
        return support, fill

    connected = _er._connected_to_seed(support | candidate, support) & candidate
    if not np.any(connected):
        return support, fill

    n_labels, labels = cv2.connectedComponents(connected.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return support, fill
    areas = np.bincount(labels.reshape(-1), minlength=n_labels)
    hint_area = max(1, int(np.count_nonzero(hint)))
    max_total = int(np.clip(hint_area * 0.045, 48, 6500))
    kept = []
    for label_id in range(1, n_labels):
        area = int(areas[label_id])
        if area <= 0:
            continue
        part = labels == label_id
        vals = color[part]
        if vals.size == 0:
            continue
        median = float(np.median(vals))
        p90 = float(np.percentile(vals, 90.0))
        edge_frac = float(np.mean(edge[part] >= edge_min))
        if median >= component_median_min or (median >= color_min and p90 >= 0.42 and edge_frac >= 0.18):
            kept.append((median + 0.25 * p90 + 0.10 * edge_frac, area, label_id))

    used = 0
    for _score, area, label_id in sorted(kept, reverse=True):
        if used and used + area > max_total:
            continue
        fill |= labels == label_id
        used += area
        if used >= max_total:
            break
    if not np.any(fill):
        return support, fill

    restored = support | fill
    if np.any(seed):
        restored = _er._connected_to_seed(restored, seed) | seed
        restored &= ~erase
        fill &= restored
    return restored, fill


def _apply_boundary_bias(
        support,
        candidate,
        seed,
        erase,
        planes,
        edge_bias=0.0):
    """Move support by a small px offset near already-accepted edge ridges.

    This is intentionally separate from colour membership. It only runs when the
    user asks for a non-zero bias, and it is limited to the solver candidate band
    near the local edge policy threshold.
    """
    support = np.asarray(support, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    erase = np.asarray(erase, dtype=bool)
    delta = np.zeros_like(support, dtype=bool)
    try:
        bias = float(edge_bias)
    except Exception:
        bias = 0.0
    steps = int(np.clip(round(abs(bias)), 0, 6))
    if steps <= 0 or not np.any(support) or not np.any(candidate):
        return support, delta

    edge = np.asarray(planes.get("context_edge", np.zeros_like(support, dtype=np.float32)), dtype=np.float32)
    if edge.shape != support.shape:
        return support, delta
    restore_threshold = _plane_max_value(
        planes,
        "edge_policy_restore_threshold",
        fallback=_v1._edge_restore_thresh_for_strength(
            _plane_max_percent(planes, "edge_lock_effective", fallback=60.0)),
    )
    edge_near = np.clip(edge, 0.0, 1.0) >= max(0.10, float(restore_threshold) - 0.05)
    if np.any(edge_near):
        edge_near = cv2.dilate(
            edge_near.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=max(2, steps),
        ) > 0
    else:
        return support, delta

    kernel = np.ones((3, 3), dtype=np.uint8)
    shifted = support.copy()
    if bias > 0.0:
        allowed = candidate & edge_near & ~erase
        for _ in range(steps):
            near_support = (
                cv2.dilate(shifted.astype(np.uint8), kernel, iterations=1) > 0
            ) & ~shifted
            add = near_support & allowed
            if not np.any(add):
                break
            shifted |= add
            delta |= add
    else:
        protected = seed | erase
        allowed = shifted & candidate & edge_near & ~protected
        background = ~shifted
        for _ in range(steps):
            near_bg = (
                cv2.dilate(background.astype(np.uint8), kernel, iterations=1) > 0
            ) & shifted
            remove = near_bg & allowed
            if not np.any(remove):
                break
            shifted &= ~remove
            background |= remove
            delta |= remove
        if np.any(seed):
            shifted = _er._connected_to_seed(shifted, seed) | seed
            shifted &= ~erase

    return shifted, delta


def _support_alpha_from_edge_softness(guide, support, planes, edge_bias=0.0):
    support = np.asarray(support, dtype=bool)
    alpha = support.astype(np.float32)
    if not np.any(support):
        return alpha

    edge = planes.get("context_edge")
    if edge is None:
        edge = _er._draw_snap_edge_strength(guide)
    if edge is None:
        return alpha
    edge = np.clip(np.asarray(edge, dtype=np.float32), 0.0, 1.0)
    if edge.shape != support.shape:
        return alpha

    try:
        bias = float(edge_bias)
    except Exception:
        bias = 0.0
    inside_dist = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
    width = float(np.clip(2.0 + max(0.0, bias) * 0.65 - max(0.0, -bias) * 0.30, 1.0, 5.0))
    edge_band = support & (inside_dist <= width)
    if not np.any(edge_band):
        return alpha

    local_peak = cv2.dilate(edge, np.ones((5, 5), dtype=np.uint8), iterations=1)
    local_mean = cv2.GaussianBlur(edge, (0, 0), 1.2)
    sharpness = np.clip((local_peak - local_mean) / 0.28, 0.0, 1.0)
    edge_conf = np.clip((local_peak - 0.10) / 0.55, 0.0, 1.0)
    soft_edge = 1.0 - np.clip(0.70 * edge_conf + 0.30 * sharpness, 0.0, 1.0)

    bias_softness = float(np.clip(0.20 + max(0.0, bias) * 0.12 - max(0.0, -bias) * 0.10, 0.0, 0.72))
    boundary_floor = 1.0 - bias_softness * soft_edge
    t = np.clip(inside_dist / max(width, 1e-3), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    edge_alpha = boundary_floor + (1.0 - boundary_floor) * smooth
    alpha[edge_band] = np.minimum(alpha[edge_band], edge_alpha[edge_band])
    return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)


def _plane_max_percent(planes, name, fallback):
    plane = planes.get(name)
    if plane is None:
        return float(fallback)
    arr = np.asarray(plane, dtype=np.float32)
    if arr.size == 0:
        return float(fallback)
    return float(np.nanmax(arr)) * 100.0


def _plane_max_value(planes, name, fallback):
    plane = planes.get(name)
    if plane is None:
        return float(fallback)
    arr = np.asarray(plane, dtype=np.float32)
    if arr.size == 0:
        return float(fallback)
    return float(np.nanmax(arr))


__all__ = ["DrawSupportResult", "compute_draw_support"]
