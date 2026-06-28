"""
Draw Quick Select V2 entry point.

V2 is intentionally developed beside V1. It keeps the same public API so the UI
and matte composition can switch with an environment flag while the solver core
is rebuilt and compared against the corpus.
"""
from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np

from cores.mask2 import draw_quick_select as _v1
from cores.mask2 import edge_refine as _er


DrawSupportResult = _v1.DrawSupportResult

# Edge-strength 1-slot cache: guide image is stable across strokes on the same
# photo. Full-resolution edge strength (~40ms) and stable-edge (~10ms) are
# deterministic functions of the guide; cache them keyed on a content
# fingerprint of the raw guide (captured before _prepare_guide_image reassigns
# the local var). A fingerprint -- not the buffer pointer -- so a recycled
# allocation address from a different crop can never return a stale edge.
_V2_EDGE_CACHE: dict = {}  # {(guide_fingerprint, shape): (edge_strength, stable_edge)}

# Edge-cost 1-slot cache: edge_cost_all (~20ms, incl. ridge skeletonize) depends
# only on solver_edge_strength (cached via _V2_EDGE_CACHE) + edge_lock policy.
# Cache it keyed on (guide_fingerprint, shape, raw_strength, edge_bias) so slider
# changes invalidate it while repeated painting strokes hit the cache.
_V2_COST_CACHE: dict = {}  # {(guide_fingerprint, shape, str_key, bias_key): (edge_cost_all, solver_es)}

# Strength/bias-INDEPENDENT prep cache: the stroke-local solve units (band geometry
# + distance transforms) and the per-unit colour analysis (_color_score_and_luma_delta,
# incl. the ~35ms shell-median) depend only on the guide, the stroke mask and the
# radius -- NOT on strength/bias. Caching them means a strength/bias slider change
# re-runs only the policy + min-cut, not the whole prep. This is what makes a param
# change cheap even with the per-stroke full-view (where each stroke is its own solve).
_V2_PREP_CACHE: dict = {}  # {prep_key: (solve_units, [ (color_roi, luma_delta) | None ])}

# These caches now hold MANY entries (one per stroke region) instead of 1 slot, so a
# param change finds every stroke's prep/edge already computed. Bounded by px budget.
_V2_CACHE_PX_BUDGET = 24_000_000


def _cache_px_size(value):
    total = 0
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, np.ndarray):
            total += v.size
        elif isinstance(v, (tuple, list)):
            stack.extend(v)
        elif hasattr(v, "component"):  # a solve unit
            for a in (getattr(v, "component", None), getattr(v, "core", None)):
                if isinstance(a, np.ndarray):
                    total += a.size
    return total


def _cache_put(cache, key, value, budget=_V2_CACHE_PX_BUDGET):
    if key is None or key in cache:
        return
    cache[key] = value
    # evict oldest (insertion order) until under the px budget
    while len(cache) > 1 and sum(_cache_px_size(v) for v in cache.values()) > budget:
        cache.pop(next(iter(cache)))


def _strokes_fingerprint(draw_strokes):
    out = []
    for s in draw_strokes or []:
        pts = getattr(s, "points", None)
        if pts is None:
            pts = []
        out.append((
            bool(getattr(s, "is_erasing", False)),
            round(float(getattr(s, "size", 0.0)), 2),
            round(float(getattr(s, "soft", 100.0)), 2),
            tuple((round(float(p[0]), 2), round(float(p[1]), 2)) for p in pts),
        ))
    return tuple(out)


# --- continuity helpers (Phase 1: de-cliff discrete control resolution) -------
def _smoothstep(x, lo, hi):
    """Hermite smoothstep: 0 at/below ``lo``, 1 at/above ``hi``, smooth between.

    Used to turn the solver's hard ``if value >= threshold`` regime switches into
    continuous ramps so a small change in the input cannot flip the output.
    """
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        return 1.0 if float(x) >= hi else 0.0
    t = float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _soft_band(x, lo0, lo1, hi0, hi1):
    """Smooth membership of ``x`` in an interval (rises lo0..lo1, falls hi0..hi1)."""
    return _smoothstep(x, lo0, lo1) * (1.0 - _smoothstep(x, hi0, hi1))


def _soft_or(a, b):
    """Probabilistic OR of two 0..1 memberships."""
    return 1.0 - (1.0 - float(a)) * (1.0 - float(b))


def compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None,
        pixel_scale=1.0,
        edge_bias=0.0) -> DrawSupportResult:
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
            edge_bias=edge_bias,
        )
        return _tag_result(result, "v2_fallback_erase", t0)

    result = _compute_add_only_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=seed_mask,
        draw_strokes=draw_strokes,
        pixel_scale=pixel_scale,
        edge_bias=edge_bias,
    )
    return _tag_result(result, "v2_add_local_edge", t0)


def _compute_add_only_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None,
        pixel_scale=1.0,
        edge_bias=0.0,
        _dump_input=True) -> DrawSupportResult:
    """Run the add-only V2 path without mutating V1 global state."""
    mask_f = _er._as_mask(mask)
    hint = mask_f > 0.02
    h, w = hint.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if not np.any(hint) or _v1.maximum_flow is None:
        return DrawSupportResult(empty, empty, empty.copy(), [])

    # Fingerprint the raw guide *before* _prepare_guide_image reassigns the local
    # variable; used as cache key for edge-strength computations. A content
    # fingerprint (not the buffer pointer) so a recycled allocation address from a
    # different crop can never return a stale edge (see _guide_fingerprint).
    try:
        _raw_guide_ptr = _er._guide_fingerprint(guide)
    except AttributeError:
        _raw_guide_ptr = None

    guide = _er._prepare_guide_image(guide, (h, w))
    if guide is None:
        return DrawSupportResult(empty, empty, empty.copy(), [])

    scale = _canonical_scale_factor(pixel_scale)
    if scale < 0.999:
        small = _downscale_problem(
            guide, mask_f, seed_mask, draw_strokes, radius, scale, edge_bias=edge_bias)
        small_result = _compute_add_only_support(
            small["guide"],
            small["mask"],
            small["radius"],
            strength,
            seed_mask=small["seed_mask"],
            draw_strokes=small["strokes"],
            pixel_scale=1.0,
            edge_bias=small["edge_bias"],
            _dump_input=False,
        )
        if _dump_input:
            _v1._maybe_dump_input(
                guide,
                mask_f,
                radius,
                strength,
                seed_mask,
                draw_strokes,
                pixel_scale,
                strength_mode=_debug_plane_mode(small_result),
                edge_lock_auto=_debug_plane_percent(small_result, "edge_lock_auto"),
                edge_lock_effective=_debug_plane_percent(small_result, "edge_lock_effective"),
                edge_lock_offset=_debug_plane_percent(small_result, "edge_lock_offset"),
                edge_bias=edge_bias,
            )
        return _upscale_result(small_result, (h, w), scale)

    scales = _v1._resolve_scales(radius, draw_strokes, hint)
    raw_strength = strength

    # The region min-cut reads the *perceptual* edge so its cut "wall" lands on the
    # boundary the user sees. On a linear scene-referred guide the raw-linear
    # gradient is dominated by highlights and the salient boundary collapses to ~0,
    # leaving no wall -> the cut overflows past the edge. Toned guides (ref >= 0.5)
    # are untouched by _to_perceptual_guide, so this is a no-op there. The
    # perceptual flag is part of the cache key so a linear and a toned guide never
    # share an entry. Set QS_REGION_PERCEPTUAL_EDGE=0 for the old raw-linear solve.
    region_perceptual = _region_perceptual_edge_enabled()
    region_ratio = _region_perceptual_shadow_ratio() if region_perceptual else 0.1
    _edge_cache_key = (
        (_raw_guide_ptr, (h, w), bool(region_perceptual), round(float(region_ratio), 3))
        if _raw_guide_ptr is not None else None)
    if _edge_cache_key is not None and _edge_cache_key in _V2_EDGE_CACHE:
        edge_strength, stable_edge_strength = _V2_EDGE_CACHE[_edge_cache_key]
    else:
        edge_strength = _er._draw_snap_edge_strength(
            guide, perceptual=region_perceptual, shadow_ratio=region_ratio)
        if edge_strength is None:
            edge_strength = np.zeros((h, w), dtype=np.float32)
        stable_edge_strength = _stable_edge_strength(
            guide, edge_strength, perceptual=region_perceptual, shadow_ratio=region_ratio)
        _cache_put(_V2_EDGE_CACHE, _edge_cache_key, (edge_strength, stable_edge_strength))
    use_stable_edge = os.environ.get("QS_V2_STABLE_EDGE", "").strip().lower() in {"1", "true", "yes", "on"}
    solve_edge_strength = stable_edge_strength if use_stable_edge else edge_strength
    strength, auto_strength, offset_strength, strength_mode = _resolve_edge_lock(
        raw_strength, solve_edge_strength, hint)
    if _dump_input:
        _v1._maybe_dump_input(
            guide,
            mask_f,
            radius,
            raw_strength,
            seed_mask,
            draw_strokes,
            pixel_scale,
            strength_mode=strength_mode,
            edge_lock_auto=auto_strength,
            edge_lock_effective=strength,
            edge_lock_offset=offset_strength,
            edge_bias=edge_bias,
        )
    # Cache edge_cost_all (includes ~20ms ridge+skeletonize) by guide + policy key.
    # raw_strength is the pre-resolve value that uniquely identifies the policy.
    _cost_key = (
        _raw_guide_ptr, (h, w),
        round(float(raw_strength), 3),
        round(float(edge_bias), 3),
    ) if _raw_guide_ptr is not None else None
    if _cost_key is not None and _cost_key in _V2_COST_CACHE:
        edge_cost_all, solver_edge_strength, resolved_policy = _V2_COST_CACHE[_cost_key]
    else:
        solver_edge_strength = _v1._solver_edge_strength(solve_edge_strength, pixel_scale)
        resolved_policy = _v1._edge_policy(strength, edge_bias=edge_bias)
        edge_cost_all = _v1._edge_cost_map(solver_edge_strength, policy=resolved_policy)
        _cache_put(_V2_COST_CACHE, _cost_key, (edge_cost_all, solver_edge_strength, resolved_policy))
    solver_edge_context_all = solver_edge_strength.copy()

    fg_stroke, bg_stroke, has_strokes = _er._draw_random_walker_stroke_seeds(
        hint.shape, draw_strokes, hint)

    hard_fg_core = _v1._seed_core(mask_f, fg_stroke, hint)
    erase_bg = bg_stroke & ~hard_fg_core

    support_all = np.zeros((h, w), dtype=bool)
    band_all = np.zeros((h, w), dtype=bool)
    fg_seed_all = np.zeros((h, w), dtype=bool)
    bg_seed_all = np.zeros((h, w), dtype=bool)
    prior_all = np.zeros((h, w), dtype=np.float32)
    cut_all = np.zeros((h, w), dtype=bool)
    color_all = np.zeros((h, w), dtype=np.float32)
    restore_color_min_all = np.full((h, w), float(_v1.EDGE_RESTORE_COLOR_MIN), dtype=np.float32)
    restore_steps_all = np.zeros((h, w), dtype=np.float32)
    edge_bias_auto_all = np.zeros((h, w), dtype=np.float32)
    edge_bias_effective_all = np.full((h, w), float(edge_bias), dtype=np.float32)
    restore_candidate_all = np.zeros((h, w), dtype=bool)
    neutral_edge_bias_candidate_all = np.zeros((h, w), dtype=bool)
    edge_restore_all = np.zeros((h, w), dtype=bool)
    neutral_edge_bias_all = np.zeros((h, w), dtype=bool)
    edge_bridge_all = np.zeros((h, w), dtype=bool)
    ridge_threshold_all = np.full((h, w), resolved_policy.ridge_threshold, dtype=np.float32)
    restore_threshold_all = np.full((h, w), resolved_policy.restore_threshold, dtype=np.float32)
    side_threshold_all = np.full((h, w), resolved_policy.side_threshold, dtype=np.float32)
    outside_threshold_all = np.full((h, w), resolved_policy.outside_keep_threshold, dtype=np.float32)
    boundary_bias_all = np.full((h, w), resolved_policy.boundary_bias_px, dtype=np.float32)
    resolved_auto_strength = float(auto_strength)
    resolved_effective_strength = float(strength)

    # Strength/bias-independent prep (solve units + per-unit colour): cached so a
    # strength/bias slider change reuses it and only the policy + min-cut re-run.
    # The colour data term separates FG/BG by colour where no edge ridge exists. On
    # a dark linear guide that separation collapses (values ~0.001), so feed it the
    # perceptual guide too (no-op on toned guides, same gate as the edge path).
    region_color_perceptual = region_perceptual and _region_perceptual_color_enabled()
    color_guide = (_er._to_perceptual_guide(guide, shadow_ratio=region_ratio)
                   if region_color_perceptual else guide)
    _prep_key = (
        _raw_guide_ptr, (h, w),
        round(float(radius), 3), round(float(pixel_scale), 4),
        bool(has_strokes),
        bool(region_color_perceptual),
        _strokes_fingerprint(draw_strokes),
        _er._guide_fingerprint(mask_f),
    ) if _raw_guide_ptr is not None else None
    _prep = _V2_PREP_CACHE.get(_prep_key) if _prep_key is not None else None
    if _prep is None:
        solve_units = _stroke_local_solve_units(
            mask_f, hint, hard_fg_core, draw_strokes, radius, has_strokes)
        _unit_colors = []
        for unit in solve_units:
            component = unit.component
            component_scales = unit.scales
            y0, y1, x0, x1 = _er._expanded_bbox(component, component_scales.roi_pad)
            if y1 <= y0 or x1 <= x0:
                _unit_colors.append(None)
                continue
            sl = np.s_[y0:y1, x0:x1]
            comp = component[sl]
            core_roi = unit.core[sl]
            _unit_colors.append(_v1._color_score_and_luma_delta(
                color_guide[sl], comp, core_roi & comp, hint[sl], component_scales.band_half_width,
                directional_bg=has_strokes))
        _prep = (solve_units, _unit_colors)
        _cache_put(_V2_PREP_CACHE, _prep_key, _prep)
    solve_units, _unit_colors = _prep
    total_band = 0
    total_flow = 0
    for unit, _unit_color in zip(solve_units, _unit_colors):
        if _unit_color is None:
            continue
        component = unit.component
        component_scales = unit.scales
        y0, y1, x0, x1 = _er._expanded_bbox(component, component_scales.roi_pad)
        if y1 <= y0 or x1 <= x0:
            continue
        sl = np.s_[y0:y1, x0:x1]
        comp = component[sl]
        core_roi = unit.core[sl]
        color_roi, selected_luma_delta = _unit_color
        unit_strength, unit_auto_strength = _v2_unit_edge_lock(
            strength,
            auto_strength,
            offset_strength,
            strength_mode,
            selected_luma_delta,
            component_scales,
        )
        resolved_auto_strength = max(resolved_auto_strength, unit_auto_strength)
        resolved_effective_strength = max(resolved_effective_strength, unit_strength)
        color_weight = _v1._color_weight_for_luma_delta(
            selected_luma_delta, strength=unit_strength)
        unit_edge_strength = _v1._contextual_edge_strength(
            solver_edge_strength[sl], color_roi, unit_strength)
        unit_policy = _v1._edge_policy(unit_strength, edge_bias=edge_bias)
        g_roi = _v1._edge_cost_map(unit_edge_strength, policy=unit_policy)
        restore_color_min = _v1._edge_restore_color_min_for_luma_delta(
            selected_luma_delta, strength=unit_strength)
        side_edge_thresh = _v2_side_edge_thresh(unit_strength, component_scales, selected_luma_delta)
        if _v2_is_thin_elongated_unit(comp, component_scales, unit_strength):
            side_edge_thresh = min(side_edge_thresh, _v2_thin_elongated_side_edge_thresh(unit_strength))
        out = _v1._solve_component(
            comp,
            hint[sl],
            g_roi,
            core_roi,
            erase_bg[sl],
            color_roi,
            component_scales,
            unit_edge_strength,
            color_weight=color_weight,
            side_edge_thresh=side_edge_thresh,
            side_relax_weight=_v1._side_edge_relax_weight_for_strength(unit_strength),
            prior_floor_in=_v2_prior_floor_in(selected_luma_delta, component_scales),
            strict_side_edge_thresh=side_edge_thresh,
            side_dilate=_v2_side_dilate(selected_luma_delta, component_scales),
            inside_color_bg_thresh=_v2_inside_color_bg_thresh(selected_luma_delta, component_scales),
            inside_color_bg_weight=_v2_inside_color_bg_weight(selected_luma_delta, component_scales),
        )
        color_all[sl] = np.where(out.band | comp, color_roi, color_all[sl])
        edge_cost_all[sl] = np.where(out.band | comp, g_roi, edge_cost_all[sl])
        solver_edge_context_all[sl] = np.where(
            out.band | comp, unit_edge_strength, solver_edge_context_all[sl])
        policy_scope = out.band | comp
        ridge_threshold_all[sl] = np.where(
            policy_scope, unit_policy.ridge_threshold, ridge_threshold_all[sl])
        restore_threshold_all[sl] = np.where(
            policy_scope, unit_policy.restore_threshold, restore_threshold_all[sl])
        side_threshold_all[sl] = np.where(
            policy_scope, side_edge_thresh, side_threshold_all[sl])
        outside_threshold_all[sl] = np.where(
            policy_scope, unit_policy.outside_keep_threshold, outside_threshold_all[sl])
        boundary_bias_all[sl] = np.where(
            policy_scope, unit_policy.boundary_bias_px, boundary_bias_all[sl])
        restore_candidate_roi = out.band & comp
        restore_candidate_all[sl] |= restore_candidate_roi
        if _v1._is_neutral_edge_bias_unit(selected_luma_delta):
            neutral_edge_bias_candidate_all[sl] |= restore_candidate_roi
        restore_color_min_all[sl] = np.where(
            restore_candidate_roi,
            np.minimum(restore_color_min_all[sl], restore_color_min),
            restore_color_min_all[sl],
        )
        auto_edge_bias = _v1._auto_edge_bias_for_unit(selected_luma_delta, component_scales)
        effective_edge_bias = float(auto_edge_bias) + float(edge_bias)
        restore_steps = _v1._edge_restore_steps_for_luma(
            selected_luma_delta, effective_edge_bias)
        restore_steps_all[sl] = np.where(
            restore_candidate_roi,
            np.maximum(restore_steps_all[sl], restore_steps),
            restore_steps_all[sl],
        )
        edge_bias_auto_all[sl] = np.where(
            restore_candidate_roi,
            auto_edge_bias,
            edge_bias_auto_all[sl],
        )
        edge_bias_effective_all[sl] = np.where(
            restore_candidate_roi,
            effective_edge_bias,
            edge_bias_effective_all[sl],
        )
        support_all[sl] |= out.support
        band_all[sl] |= out.band
        fg_seed_all[sl] |= out.hard_fg
        bg_seed_all[sl] |= out.hard_bg
        prior_all[sl] = np.where(out.band, out.prior, prior_all[sl])
        cut_all[sl] |= out.cut_boundary
        total_band += int(np.count_nonzero(out.band))
        total_flow = max(total_flow, out.flow_value)

    support_all = _v1._postprocess_support(support_all, hint, hard_fg_core, erase_bg)
    support_all, edge_restore_all = _v1._restore_selected_edge_rim(
        support_all,
        restore_candidate_all,
        solver_edge_context_all,
        color_all,
        hard_fg_core,
        erase_bg,
        color_min=restore_color_min_all,
        edge_thresh=_v1._edge_policy(resolved_effective_strength, edge_bias=edge_bias).restore_threshold,
        steps=restore_steps_all,
        edge_bias=edge_bias,
    )
    support_all, neutral_edge_bias_all = _v1._restore_neutral_edge_bias_rim(
        support_all,
        neutral_edge_bias_candidate_all,
        solver_edge_context_all,
        color_all,
        hard_fg_core,
        erase_bg,
        edge_bias=edge_bias,
        edge_thresh=_v1._edge_policy(resolved_effective_strength, edge_bias=edge_bias).restore_threshold,
    )
    support_all, edge_bridge_all = _v1._bridge_selected_edge_seams(
        support_all,
        restore_candidate_all,
        solver_edge_context_all,
        hard_fg_core,
        erase_bg,
    )
    support_all = _v1._limit_smooth_outside_growth(
        support_all, hint, solver_edge_context_all, hard_fg_core, erase_bg,
        strength=resolved_effective_strength)
    support_all, interior_fill_all = _v1._fill_selected_hint_holes(
        support_all, hint, hard_fg_core, erase_bg)
    support_all, same_side_gap_fill_all = _v2_fill_same_side_gaps(
        support_all,
        hint,
        hard_fg_core,
        erase_bg,
        solver_edge_context_all,
        color_all,
        resolved_effective_strength,
        scales,
    )
    support_all = _er._preserve_draw_component_separation(hint, support_all)

    debug_planes = [
        ("image_edge", edge_strength),
        ("stable_edge", stable_edge_strength),
        ("stable_edge_enabled", np.full((h, w), 1.0 if use_stable_edge else 0.0, dtype=np.float32)),
        ("context_edge", solver_edge_context_all),
        ("edge_cost", edge_cost_all),
        ("color_score", (color_all * 0.5 + 0.5).astype(np.float32)),
        ("seed_fg", fg_seed_all),
        ("seed_bg", bg_seed_all),
        ("prior", (prior_all * 0.5 + 0.5).astype(np.float32)),
        ("cut_boundary", cut_all),
        ("edge_restore", edge_restore_all),
        ("neutral_edge_bias", neutral_edge_bias_all),
        ("edge_bridge", edge_bridge_all),
        ("interior_fill", interior_fill_all),
        ("same_side_gap_fill", same_side_gap_fill_all),
        ("v2_graph_nodes", np.full((h, w), float(total_band), dtype=np.float32)),
        ("v2_flow_value", np.full((h, w), float(total_flow), dtype=np.float32)),
        ("edge_lock_auto", np.full((h, w), float(resolved_auto_strength) / 100.0, dtype=np.float32)),
        ("edge_lock_effective", np.full((h, w), float(resolved_effective_strength) / 100.0, dtype=np.float32)),
        ("edge_lock_offset", np.full((h, w), float(offset_strength) / 100.0, dtype=np.float32)),
        ("edge_lock_mode_offset", np.full((h, w), 1.0 if strength_mode == "offset" else 0.0, dtype=np.float32)),
        ("edge_bias_auto", edge_bias_auto_all),
        ("edge_bias_effective", edge_bias_effective_all),
        ("edge_bias_offset", np.full((h, w), float(edge_bias), dtype=np.float32)),
        ("edge_policy_ridge_threshold", ridge_threshold_all),
        ("edge_policy_restore_threshold", restore_threshold_all),
        ("edge_policy_side_threshold", side_threshold_all),
        ("edge_policy_outside_keep_threshold", outside_threshold_all),
        ("boundary_bias_px", boundary_bias_all),
    ]

    hint_area = int(np.count_nonzero(hint))
    support_area = int(np.count_nonzero(support_all))
    logging.debug(
        "[DRAW_QS_V2] hint=%d band=%d support=%d ratio=%.3f comps=%d max_flow=%d radius=%.1f",
        hint_area,
        total_band,
        support_area,
        (support_area / hint_area) if hint_area else 0.0,
        len(solve_units),
        total_flow,
        scales.band_half_width,
    )
    return DrawSupportResult(fg_seed_all, band_all, support_all, debug_planes)


def _resolve_edge_lock(raw_strength, edge_strength, hint):
    """Resolve V2 EdgeLock.

    Default UI semantics are offset-based: 0 = auto, positive = stricter,
    negative = looser. Corpus replay preserves old dumps by setting
    ``QS_V2_STRENGTH_MODE=internal`` from the dump metadata.
    """
    auto = _estimate_auto_edge_lock(edge_strength, hint)
    try:
        raw = float(raw_strength)
    except Exception:
        raw = 0.0
    mode = os.environ.get("QS_V2_STRENGTH_MODE", "").strip().lower()
    if not mode and os.environ.get("QS_DRAW_V2_OFFSET", "").strip().lower() in {"1", "true", "yes", "on"}:
        mode = "offset"
    if not mode:
        mode = "offset"
    if mode == "offset":
        offset = raw
        effective = _apply_edge_lock_offset(auto, offset)
        return effective, auto, offset, "offset"
    effective = float(np.clip(raw, 0.0, 100.0))
    return effective, auto, effective - auto, "internal"


def _v2_unit_edge_lock(
        base_strength,
        auto_strength,
        offset_strength,
        strength_mode,
        selected_luma_delta,
        scales):
    """Per-stroke auto correction for broad bright-side tree/cloud strokes."""
    base = float(np.clip(base_strength, 0.0, 100.0))
    auto = float(np.clip(auto_strength, 0.0, 100.0))
    if strength_mode != "offset":
        return base, auto

    delta = float(selected_luma_delta)
    stroke_hw = float(max(1.0, getattr(scales, "stroke_half_width", 1.0)))
    unit_auto = auto
    # Broad mid-tone bright dab: pull auto stricter (toward <=45) so it snaps to
    # the soft boundary. Smooth ramps replace the old hard delta/width gates so
    # a stroke just under the threshold is nudged, not switched.
    w_mid = _soft_band(delta, 0.03, 0.05, 0.19, 0.21) * _smoothstep(stroke_hw, 125.0, 140.0)
    if w_mid > 0.0:
        unit_auto = unit_auto * (1.0 - w_mid) + min(unit_auto, 45.0) * w_mid
    # Very bright background: floor auto looser (toward >=90).
    w_bright = _smoothstep(delta, 0.70, 0.78) * _smoothstep(stroke_hw, 108.0, 120.0)
    if w_bright > 0.0:
        unit_auto = unit_auto * (1.0 - w_bright) + max(unit_auto, 90.0) * w_bright

    effective = _apply_edge_lock_offset(unit_auto, offset_strength)
    return effective, unit_auto


def _apply_edge_lock_offset(auto_strength, offset_strength):
    """Map UI EdgeLock offset around auto without jumping across regimes.

    V2 internally still uses the old 0..100 "edge sensitivity" scale where larger
    accepts weaker edges. The UI is offset based: 0 = auto, + = stricter, - =
    looser. A direct one-to-one subtraction makes high-auto low-contrast strokes
    collapse with tiny positive moves and low-auto strong-edge strokes explode
    with large negative moves. Scale the offset by the amount of useful room on
    that side so +/- remains directional but more controllable near the extremes.
    """
    auto = float(np.clip(auto_strength, 0.0, 100.0))
    offset = float(offset_strength)
    if abs(offset) <= 1e-6:
        return auto
    if offset > 0.0:
        room = (100.0 - auto) / 100.0
    else:
        room = auto / 100.0
    scale = 0.25 + 0.75 * (float(np.clip(room, 0.0, 1.0)) ** 0.75)
    return float(np.clip(auto - offset * scale, 0.0, 100.0))


def _region_perceptual_edge_enabled():
    """Feed the region min-cut a perceptual (display-like) edge map.

    Default on. The toned-guide skip in ``_to_perceptual_guide`` makes it a no-op
    on already display-encoded guides, so only genuinely linear scene-referred
    captures change. Set ``QS_REGION_PERCEPTUAL_EDGE=0`` to restore the old
    raw-linear region solve.
    """
    value = os.environ.get("QS_REGION_PERCEPTUAL_EDGE")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _region_perceptual_shadow_ratio():
    """Gate-2 threshold for re-encoding the region edge to perceptual space.

    The V4 boundary trace keeps the strict 0.1 (deep shadow only). The region
    min-cut relaxes it to ~0.35 so a *moderately* shadow-weighted linear capture
    (the user's `leaf`, p50/ref=0.194) gets a real cut wall instead of overflowing
    past the edge, while balanced linear guides (`easy`, p50/ref=0.663) stay
    excluded so their strong boundary is not compressed. Tunable via
    ``QS_REGION_PERCEPTUAL_RATIO``.
    """
    try:
        return float(os.environ.get("QS_REGION_PERCEPTUAL_RATIO", "0.35"))
    except (TypeError, ValueError):
        return 0.35


def _region_perceptual_color_enabled():
    """Run the min-cut *colour* data term on the perceptual guide too (default on).

    On a dark linear capture the colour difference that the eye sees (e.g. a green
    leaf vs a violet background, hue 107 deg vs 261 deg) collapses to the 4th decimal
    in linear RGB (~0.001 values), so the solver's colour_score reads ~0 and the cut
    cannot separate them where no edge ridge exists. Re-encoding to perceptual space
    restores the separation (measured z~2.6-3.3 on the user's `leaf`). Same gate as the
    edge path (toned guides unchanged). Set ``QS_REGION_PERCEPTUAL_COLOR=0`` to keep the
    colour term on raw linear.
    """
    value = os.environ.get("QS_REGION_PERCEPTUAL_COLOR")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _stable_edge_strength(guide, edge_strength, perceptual=False, shadow_ratio=0.1):
    """Prefer edges that survive a 2x scale change.

    Fine snowy/tree texture often produces high single-scale gradients. Real
    object boundaries usually remain visible after downsampling. This gate keeps
    strong single-scale edges available, but discounts edges that disappear at
    half resolution.
    """
    edge = np.clip(np.asarray(edge_strength, dtype=np.float32), 0.0, 1.0)
    h, w = edge.shape[:2]
    if h < 16 or w < 16:
        return edge
    guide_arr = np.asarray(guide, dtype=np.float32)
    small = cv2.resize(guide_arr, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    small_edge = _er._draw_snap_edge_strength(small, perceptual=perceptual, shadow_ratio=shadow_ratio)
    if small_edge is None:
        return edge
    half = cv2.resize(np.asarray(small_edge, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    half = np.clip(half, 0.0, 1.0)
    # Keep a floor so thin but meaningful branches are not erased completely;
    # still let scale-stable boundaries dominate the min-cut.
    gate = 0.45 + 0.55 * np.sqrt(half)
    stable = edge * gate
    return np.maximum(stable, edge * 0.55).astype(np.float32, copy=False)


def _estimate_auto_edge_lock(edge_strength, hint):
    vals = _hint_boundary_edge_values(edge_strength, hint)
    if vals.size == 0:
        return 55.0
    p90 = float(np.percentile(vals, 90.0))
    p75 = float(np.percentile(vals, 75.0))
    strong_density = float(np.mean(vals >= 0.60))
    mid_density = float(np.mean(vals >= 0.35))
    return _auto_edge_lock_from_stats(p90, p75, strong_density, mid_density)


def _auto_edge_lock_from_stats(p90, p75, strong_density, mid_density):
    """Continuous auto EdgeLock from boundary edge statistics.

    This is a smooth (cliff-free) restatement of the original if/elif regime
    table. Each regime keeps its target value and its deciding condition, but the
    hard thresholds become smoothstep memberships and the elif precedence becomes
    a multiplicative ``remaining`` weight. At the interior of a regime the result
    equals the old discrete value (so corpus behavior is preserved); only near a
    regime boundary does it blend instead of jumping -- which is what made the
    same input at a different zoom or stroke position pick a different regime.
    """
    p90 = float(p90)
    p75 = float(p75)
    sd = float(strong_density)
    md = float(mid_density)
    # (membership, target) in the original elif order; targets are unchanged.
    regimes = (
        # crisp solid boundary -> strict / low sensitivity
        (_smoothstep(p90, 0.72, 0.78) * _smoothstep(sd, 0.13, 0.19), 34.0),
        # strong boundary (high peak OR many strong pixels)
        (_soft_or(_smoothstep(p90, 0.56, 0.64), _smoothstep(sd, 0.06, 0.10)), 44.0),
        # almost featureless -> very loose
        ((1.0 - _smoothstep(p90, 0.06, 0.10)) * (1.0 - _smoothstep(md, 0.015, 0.025)), 100.0),
        # sparse strong peaks (e.g. trees on sky) -> loose
        (_soft_band(p90, 0.43, 0.47, 0.53, 0.57)
         * _soft_band(sd, 0.03, 0.05, 0.07, 0.09)
         * (1.0 - _smoothstep(md, 0.18, 0.22)), 96.0),
        # diffuse mid-tone boundary -> strict
        (_soft_band(p90, 0.43, 0.47, 0.53, 0.57)
         * (1.0 - _smoothstep(sd, 0.05, 0.07))
         * _smoothstep(md, 0.18, 0.22), 20.0),
        # weak boundary -> loose-ish
        ((1.0 - _smoothstep(p75, 0.23, 0.27)) * (1.0 - _smoothstep(md, 0.07, 0.09)), 78.0),
        # moderate boundary
        (1.0 - _smoothstep(p90, 0.43, 0.47), 60.0),
    )
    remaining = 1.0
    auto = 0.0
    for raw, target in regimes:
        raw = float(np.clip(raw, 0.0, 1.0))
        weight = raw * remaining
        auto += weight * target
        remaining *= (1.0 - raw)
    auto += remaining * 55.0  # default / final else
    return float(np.clip(auto, 0.0, 100.0))


def _hint_boundary_edge_values(edge_strength, hint, width=8):
    edge = np.asarray(edge_strength, dtype=np.float32)
    hint = np.asarray(hint, dtype=bool)
    if edge.shape != hint.shape or not np.any(hint):
        return np.array([], dtype=np.float32)
    width = int(max(1, round(float(width))))
    dist_in = cv2.distanceTransform(hint.astype(np.uint8), cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform((~hint).astype(np.uint8), cv2.DIST_L2, 3)
    band = (hint & (dist_in <= width)) | ((~hint) & (dist_out <= width))
    vals = edge[band]
    if vals.size == 0:
        return np.array([], dtype=np.float32)
    return vals.astype(np.float32, copy=False)


def _v2_prior_floor_in(selected_luma_delta, scales):
    if float(selected_luma_delta) > 0.04:
        stroke_hw = float(getattr(scales, "stroke_half_width", 0.0))
        if stroke_hw >= 150.0:
            return 0.0
        if 45.0 <= stroke_hw <= 110.0:
            return 0.02
        return 0.05
    return None


def _v2_inside_color_bg_thresh(selected_luma_delta, scales):
    stroke_hw = float(getattr(scales, "stroke_half_width", 0.0))
    delta = float(selected_luma_delta)
    if 0.04 < delta <= 0.20 and stroke_hw >= 140.0:
        return -0.02
    return None


def _v2_inside_color_bg_weight(selected_luma_delta, scales):
    stroke_hw = float(getattr(scales, "stroke_half_width", 0.0))
    delta = float(selected_luma_delta)
    if 0.04 < delta <= 0.20 and stroke_hw >= 180.0:
        return 0.85
    if 0.04 < delta <= 0.20 and stroke_hw >= 140.0:
        return 0.65
    return 0.0


def _v2_side_edge_thresh(strength, scales, selected_luma_delta):
    base = _v1._side_edge_thresh_for_strength(strength)
    stroke_hw = float(getattr(scales, "stroke_half_width", 0.0))
    if float(selected_luma_delta) > 0.04 and 45.0 <= stroke_hw <= 110.0:
        return min(base, 0.45)
    return base


def _v2_is_thin_elongated_unit(comp, scales, strength):
    if float(strength) < 40.0:
        return False
    stroke_hw = float(getattr(scales, "stroke_half_width", 0.0))
    if stroke_hw > 40.0:
        return False
    comp = np.asarray(comp, dtype=bool)
    if not np.any(comp):
        return False
    ys, xs = np.where(comp)
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    short = max(1.0, min(width, height))
    long = max(width, height)
    if long / short < 7.0:
        return False
    return True


def _v2_thin_elongated_side_edge_thresh(strength):
    lock = float(np.clip(strength, 0.0, 100.0)) / 100.0
    return float(np.clip(0.34 - 0.27 * lock, 0.12, 0.34))


def _v2_side_dilate(selected_luma_delta, scales):
    stroke_hw = float(getattr(scales, "stroke_half_width", 0.0))
    delta = float(selected_luma_delta)
    if 0.02 < delta <= 0.04 and stroke_hw >= 120.0:
        return 1
    return None


def _v2_fill_same_side_gaps(
        support,
        hint,
        hard_fg_core,
        erase_bg,
        edge_strength,
        color_score,
        strength,
        scales):
    """Restore selected-side gaps without turning EdgeLock into blind dilation.

    The min-cut is conservative around busy silhouettes: narrow sky/cloud gaps
    between dark branches can be left as background because they stay connected
    to the hint boundary. This post-process only restores pixels that are
    connected to the existing support through FG-like colour, close to an image
    edge, and pass a per-component confidence check. EdgeLock controls how weak
    a same-side gap may be before we accept it.
    """
    support = np.asarray(support, dtype=bool)
    hint = np.asarray(hint, dtype=bool)
    erase = np.asarray(erase_bg, dtype=bool)
    fill = np.zeros_like(support, dtype=bool)
    gap = hint & ~support & ~erase
    if not np.any(support) or not np.any(gap):
        return support, fill

    color = np.nan_to_num(
        np.asarray(color_score, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    if color.shape != support.shape:
        return support, fill
    if float(np.max(color[gap], initial=-1.0)) <= 0.0:
        return support, fill

    edge = np.clip(np.asarray(edge_strength, dtype=np.float32), 0.0, 1.0)
    if edge.shape != support.shape:
        edge = np.zeros_like(color, dtype=np.float32)

    params = _v2_gap_fill_params(strength, scales)
    dist_to_support = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 3)
    edge_near = edge >= params.edge_min
    if np.any(edge_near):
        edge_near = cv2.dilate(
            edge_near.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=params.edge_near,
        ) > 0

    candidate = (
        gap
        & (dist_to_support <= params.max_distance)
        & (color >= params.pixel_min)
        & edge_near
    )
    if not np.any(candidate):
        return support, fill

    connected = _er._connected_to_seed(support | candidate, support) & candidate
    if not np.any(connected):
        return support, fill

    n_labels, labels = cv2.connectedComponents(connected.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return support, fill

    areas = np.bincount(labels.reshape(-1), minlength=n_labels)
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
        edge_frac = float(np.mean(edge[part] >= params.edge_min))
        keep = median >= params.component_median and area <= params.medium_area
        if not keep and area > params.medium_area:
            keep = median >= params.large_median and p90 >= params.large_p90
        if not keep and area <= params.small_area:
            keep = median >= params.small_median and p90 >= params.small_p90
        if not keep and area <= params.medium_area:
            keep = (
                median >= params.small_median
                and p90 >= params.small_p90
                and edge_frac >= params.edge_fraction
            )
        if keep:
            score = median + 0.35 * p90 + 0.10 * edge_frac
            kept.append((float(score), area, label_id))

    if not kept:
        return support, fill

    max_total = _v2_gap_fill_max_total(hint, support, params)
    used = 0
    for _score, area, label_id in sorted(kept, reverse=True):
        if used > 0 and used + area > max_total:
            continue
        if used == 0 or used + area <= max_total:
            fill |= labels == label_id
            used += area

    if not np.any(fill):
        return support, fill

    restored = support | fill
    if np.any(hard_fg_core):
        restored = _er._connected_to_seed(restored, hard_fg_core) | hard_fg_core
        restored &= ~erase
        fill &= restored
    return restored, fill


def _v2_gap_fill_params(strength, scales):
    lock = float(np.clip(strength, 0.0, 100.0)) / 100.0
    stroke_hw = float(max(1.0, getattr(scales, "stroke_half_width", 1.0)))
    max_distance = float(np.clip(stroke_hw * (0.08 + 0.16 * lock), 5.0, 28.0))
    small_area = int(round(np.clip(16.0 + stroke_hw * stroke_hw * 0.012, 24.0, 480.0)))
    medium_area = int(round(np.clip(32.0 + stroke_hw * stroke_hw * 0.035, 64.0, 900.0)))
    return SimpleNamespace(
        pixel_min=float(np.clip(0.20 - 0.30 * lock, -0.08, 0.22)),
        component_median=float(np.clip(0.22 - 0.14 * lock, 0.10, 0.24)),
        large_median=float(np.clip(0.24 - 0.10 * lock, 0.14, 0.26)),
        large_p90=float(np.clip(0.38 - 0.14 * lock, 0.24, 0.40)),
        small_median=float(np.clip(0.12 - 0.10 * lock, 0.035, 0.14)),
        small_p90=float(np.clip(0.34 - 0.16 * lock, 0.18, 0.36)),
        edge_min=float(np.clip(0.28 - 0.16 * lock, 0.10, 0.30)),
        edge_near=int(np.clip(round(1.0 + 2.0 * lock), 1, 3)),
        edge_fraction=float(np.clip(0.82 - 0.32 * lock, 0.50, 0.86)),
        max_distance=max_distance,
        small_area=small_area,
        medium_area=medium_area,
    )


def _v2_gap_fill_max_total(hint, support, params):
    hint_area = int(np.count_nonzero(hint))
    support_area = int(np.count_nonzero(support))
    geometric_cap = int(round(float(params.small_area) * 4.0))
    area_cap = int(round(min(
        max(32.0, float(hint_area) * 0.035),
        max(32.0, float(support_area) * 0.080),
        max(64.0, float(geometric_cap)),
    )))
    return max(1, area_cap)


def _canonical_scale_factor(pixel_scale):
    if os.environ.get("QS_V2_CANONICAL_SCALE", "").strip().lower() in {"0", "false", "no", "off"}:
        return 1.0
    try:
        scale = float(pixel_scale)
    except Exception:
        scale = 1.0
    if scale <= 1.01:
        return 1.0
    return float(np.clip(1.0 / scale, 0.25, 1.0))


def _downscale_problem(guide, mask, seed_mask, draw_strokes, radius, scale, edge_bias=0.0):
    h, w = mask.shape[:2]
    sw = max(1, int(round(w * float(scale))))
    sh = max(1, int(round(h * float(scale))))
    small_guide = cv2.resize(
        np.asarray(guide, dtype=np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(
        np.asarray(mask, dtype=np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small_seed = None
    if seed_mask is not None:
        small_seed = cv2.resize(
            np.asarray(seed_mask, dtype=np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    return {
        "guide": small_guide.astype(np.float32, copy=False),
        "mask": small_mask.astype(np.float32, copy=False),
        "seed_mask": small_seed,
        "radius": float(radius) * float(scale),
        "edge_bias": float(edge_bias) * float(scale),
        "strokes": _scale_strokes(draw_strokes, scale),
    }


def _scale_strokes(draw_strokes, scale):
    if not draw_strokes:
        return draw_strokes
    out = []
    for stroke in draw_strokes:
        points = np.asarray(getattr(stroke, "points", []), dtype=np.float32)
        if points.size:
            points = points * float(scale)
            pts = [(float(x), float(y)) for x, y in points[:, :2]]
        else:
            pts = []
        out.append(SimpleNamespace(
            points=pts,
            size=float(getattr(stroke, "size", 1.0)) * float(scale),
            soft=float(getattr(stroke, "soft", 100.0)),
            is_erasing=bool(getattr(stroke, "is_erasing", False)),
        ))
    return out


def _upscale_result(result, shape, scale):
    h, w = shape

    def resize_bool(arr):
        return cv2.resize(
            np.asarray(arr, dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    def resize_plane(arr):
        a = np.asarray(arr)
        interp = cv2.INTER_NEAREST if a.dtype == np.bool_ else cv2.INTER_LINEAR
        out = cv2.resize(a.astype(np.float32), (w, h), interpolation=interp)
        return out.astype(np.float32, copy=False)

    planes = []
    for name, plane in result.debug_planes:
        arr = np.asarray(plane)
        if arr.ndim == 2 and arr.shape[:2] != (h, w):
            planes.append((name, resize_plane(arr)))
        else:
            planes.append((name, plane))
    planes.append(("v2_canonical_scale", np.full((h, w), float(scale), dtype=np.float32)))
    return DrawSupportResult(
        resize_bool(result.seed),
        resize_bool(result.candidate),
        resize_bool(result.support),
        planes,
    )


def _debug_plane_percent(result, name):
    for plane_name, plane in getattr(result, "debug_planes", []) or []:
        if plane_name == name:
            arr = np.asarray(plane, dtype=np.float32)
            if arr.size:
                return float(np.nanmax(arr)) * 100.0
    return None


def _debug_plane_mode(result):
    value = _debug_plane_percent(result, "edge_lock_mode_offset")
    if value is None:
        return None
    return "offset" if value >= 50.0 else "internal"


def _stroke_local_solve_units(mask_f, hint, hard_fg_core, draw_strokes, radius, has_strokes):
    """V2 add-only solve units: local brush footprint, not whole component."""
    hint = np.asarray(hint, dtype=bool)
    units = []
    covered = np.zeros_like(hint, dtype=bool)

    if has_strokes and draw_strokes:
        shape = hint.shape
        for stroke in draw_strokes:
            if bool(getattr(stroke, "is_erasing", False)):
                continue
            points = _er._stroke_points_array(stroke)
            if points.shape[0] == 0:
                continue
            try:
                size = float(max(1.0, getattr(stroke, "size", 1.0)))
            except Exception:
                size = 1.0
            stroke_mask = _er._stroke_brush_mask(shape, points, size) & hint
            if not np.any(stroke_mask):
                continue
            center = _er._stroke_center_mask(shape, points, size) & stroke_mask
            stroke_core = _v1._seed_core(np.asarray(mask_f) * stroke_mask, center, stroke_mask)
            stroke_scales = _v1._resolve_scales(radius, [stroke], stroke_mask)
            covered |= stroke_mask
            _v1._append_connected_units(units, stroke_mask, stroke_core, stroke_scales)

    fallback = hint & ~covered if units else hint
    if units:
        fallback = _v1._drop_tiny_components(fallback, _v1.MIN_SOLVE_COMPONENT_AREA)
    if np.any(fallback):
        fallback_scales = _v1._resolve_scales(radius, draw_strokes, fallback)
        _v1._append_connected_units(units, fallback, hard_fg_core & fallback, fallback_scales)

    return units


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
        planes.append(("v2_mode", np.full(shape, 1.0 if mode.startswith("v2_add") else 0.0, dtype=np.float32)))
        logging.debug("[DRAW_QS_V2] mode=%s runtime_ms=%.1f", mode, elapsed)
        return DrawSupportResult(result.seed, result.candidate, result.support, planes)
    return result


__all__ = ["DrawSupportResult", "compute_draw_support"]
