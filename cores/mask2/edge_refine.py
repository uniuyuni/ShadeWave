"""
Quick-selection style mask refinement shared by AI, draw and parametric masks.
"""
from __future__ import annotations

import cv2
import logging
import numpy as np
import os
import time

try:
    from scipy import sparse as _sparse
    from scipy.sparse import linalg as _sparse_linalg
except Exception:
    _sparse = None
    _sparse_linalg = None


MODE_OFF = "Off"
MODE_QUICK_SELECT = "Quick Select"

MODE_VALUES = (MODE_OFF, MODE_QUICK_SELECT)
STRATEGY_REFINE = "refine"
STRATEGY_DRAW = "draw"
_DEBUG_DUMP_COUNTER = 0


def normalize_mode(mode):
    if mode is None:
        return MODE_OFF
    value = str(mode).strip().lower().replace("_", " ")
    aliases = {
        "off": MODE_OFF,
        "none": MODE_OFF,
        "quick": MODE_QUICK_SELECT,
        "quick select": MODE_QUICK_SELECT,
        "quick selection": MODE_QUICK_SELECT,
        # Legacy experimental modes are treated as the new single mode so old
        # in-memory params do not silently disable the user's test setup.
        "refine": MODE_QUICK_SELECT,
        "edge refine": MODE_QUICK_SELECT,
        "grow": MODE_QUICK_SELECT,
        "edge grow": MODE_QUICK_SELECT,
        "grow islands": MODE_QUICK_SELECT,
        "grow + islands": MODE_QUICK_SELECT,
        "edge grow + islands": MODE_QUICK_SELECT,
        "lock": MODE_QUICK_SELECT,
        "edge lock": MODE_QUICK_SELECT,
    }
    return aliases.get(value, mode if mode in MODE_VALUES else MODE_OFF)


def is_enabled(mode):
    return normalize_mode(mode) != MODE_OFF


def make_confident_seed(mask, threshold=0.05, shrink_ratio=0.55, min_shrink=1.5):
    seed = _as_mask(mask) > float(threshold)
    if not np.any(seed):
        return seed

    dist = cv2.distanceTransform(seed.astype(np.uint8), cv2.DIST_L2, 3)
    max_dist = float(dist.max(initial=0.0))
    if max_dist <= min_shrink:
        return seed

    cutoff = max(float(min_shrink), max_dist * float(shrink_ratio))
    cutoff = min(cutoff, max_dist - 0.5)
    confident = dist >= cutoff
    if not np.any(confident):
        confident = dist >= max_dist * 0.75
    return confident


def refine_mask_edge_aware(
        image_rgb,
        mask,
        guide_point=None,
        mode=MODE_OFF,
        radius=60,
        strength=60,
        fill_grown_region=True,
        seed_from_guide=False,
        seed_mask=None,
        debug_label=None,
        support_softness=0.0,
        selection_strategy=STRATEGY_REFINE,
        draw_strokes=None,
        draw_pixel_scale=1.0,
        return_support=False):
    mode = normalize_mode(mode)
    mask_f = _as_mask(mask)
    if mode == MODE_OFF or mask_f.size == 0 or float(np.nanmax(mask_f)) <= 0:
        return (mask_f, None) if return_support else mask_f

    h, w = mask_f.shape[:2]
    guide = _prepare_guide_image(image_rgb, (h, w))
    if guide is None:
        return (mask_f, None) if return_support else mask_f

    raw_radius = float(radius)
    radius = int(max(1, raw_radius))
    raw_strength = float(strength)
    strength = float(np.clip(raw_strength, 0, 100))
    if selection_strategy == STRATEGY_DRAW:
        draw_strength = raw_strength
        if os.environ.get("PLATYPUS_DRAW_QS_LEGACY"):
            # Legacy grabCut/target-edge path kept for one release as a fallback.
            seed, candidate, support, extra_debug_planes = _draw_grabcut_band_support(
                guide,
                mask_f,
                radius,
                strength,
                seed_mask=seed_mask,
                draw_strokes=draw_strokes,
                pixel_scale=draw_pixel_scale,
            )
            effective_edge_lock = strength
        else:
            from cores.mask2 import draw_quick_select as _draw_quick_select
            _draw_result = _draw_quick_select.compute_draw_support(
                guide,
                mask_f,
                raw_radius,
                draw_strength,
                seed_mask=seed_mask,
                draw_strokes=draw_strokes,
                pixel_scale=draw_pixel_scale,
            )
            seed = _draw_result.seed
            candidate = _draw_result.candidate
            support = _draw_result.support
            extra_debug_planes = _draw_result.debug_planes
            effective_edge_lock = float(getattr(_draw_result, "edge_lock", strength))
        if support is None:
            support = _fallback_support(mask_f, seed, candidate)
        refined = _compose_refined_mask(
            mask_f,
            support,
            fill_grown_region,
            support_softness=support_softness,
            guide=guide,
            natural_edge=True,
            edge_lock=effective_edge_lock,
        )
        _debug_dump_refine_state(
            guide,
            mask_f,
            refined,
            guide_point,
            seed,
            candidate,
            support,
            _draw_barrier_strength(effective_edge_lock),
            seed_from_guide,
            debug_label,
            "edge_snap",
            extra_planes=extra_debug_planes,
        )
        return (refined, support.astype(np.float32)) if return_support else refined

    gc_mask, seed, candidate = _build_grabcut_mask(
        guide,
        mask_f,
        guide_point,
        radius,
        strength,
        seed_from_guide=seed_from_guide,
        seed_mask=seed_mask,
    )
    if gc_mask is None:
        support = _fallback_support(mask_f, seed, candidate)
        refined = _compose_refined_mask(
            mask_f,
            support,
            fill_grown_region,
            support_softness=support_softness,
        )
        _debug_dump_refine_state(
            guide,
            mask_f,
            refined,
            guide_point,
            seed,
            candidate,
            support,
            strength,
            seed_from_guide,
            debug_label,
            "fallback",
        )
        return (refined, support.astype(np.float32)) if return_support else refined

    guide_u8 = _guide_to_grabcut_image(guide)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    iterations = _grabcut_iterations(strength)
    try:
        cv2.grabCut(
            guide_u8,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        support = _fallback_support(mask_f, seed, candidate)
        refined = _compose_refined_mask(
            mask_f,
            support,
            fill_grown_region,
            support_softness=support_softness,
        )
        _debug_dump_refine_state(
            guide,
            mask_f,
            refined,
            guide_point,
            seed,
            candidate,
            support,
            strength,
            seed_from_guide,
            debug_label,
            "grabcut_error",
        )
        return (refined, support.astype(np.float32)) if return_support else refined

    support = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)) & candidate
    support |= seed
    min_expected = max(int(np.count_nonzero(seed) * 2), int(np.count_nonzero(candidate) * 0.25))
    if np.count_nonzero(support) < min_expected:
        support = candidate | seed
    refined = _compose_refined_mask(
        mask_f,
        support,
        fill_grown_region,
        support_softness=support_softness,
    )
    _debug_dump_refine_state(
        guide,
        mask_f,
        refined,
        guide_point,
        seed,
        candidate,
        support,
        strength,
        seed_from_guide,
        debug_label,
        "grabcut",
    )

    return (refined, support.astype(np.float32)) if return_support else refined


def _build_grabcut_mask(
        guide,
        mask,
        guide_point,
        radius,
        strength,
        seed_from_guide=False,
        seed_mask=None):
    h, w = mask.shape[:2]
    seed = _make_foreground_seed(
        mask,
        guide_point,
        radius,
        seed_from_guide=seed_from_guide,
        seed_mask=seed_mask,
    )
    if not np.any(seed):
        return None, None, None

    hint = _make_foreground_hint(mask, guide_point, seed_from_guide=seed_from_guide)
    anchor = seed | hint
    expansion_radius = _expansion_radius(radius, strength)
    candidate = _make_quick_select_candidate(guide, seed, anchor, expansion_radius, strength)
    if not np.any(candidate):
        return None, None, None

    roi = candidate | (_distance_from(candidate) <= float(max(2, expansion_radius // 3)))

    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[roi] = cv2.GC_PR_BGD

    gc_mask[candidate] = cv2.GC_PR_FGD
    gc_mask[seed] = cv2.GC_FGD

    # Keep a local background ring. Without it GrabCut can learn only the
    # foreground colors and select the whole ROI on low-contrast images.
    sure_bg = (~roi) | ((mask <= 0.001) & (_distance_from(seed) > float(expansion_radius)))
    gc_mask[sure_bg] = cv2.GC_BGD

    if not _has_grabcut_samples(gc_mask):
        return None, seed, candidate
    return gc_mask, seed, candidate


def _draw_grabcut_band_support(guide, mask, radius, strength, seed_mask=None, draw_strokes=None):
    mask_f = _as_mask(mask)
    hint = mask_f > 0.02
    h, w = hint.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if not np.any(hint):
        return empty, empty, empty, []

    search_radius = int(max(1, round(float(radius))))
    edge_strength = _draw_snap_edge_strength(guide)
    image_edge = _draw_component_image_edge(edge_strength, strength) if edge_strength is not None else empty
    hint_boundary = _mask_boundary_strength(hint)
    seed_mask_bool = _resize_bool_mask(seed_mask, hint.shape) if seed_mask is not None else None
    fg_stroke_seed, bg_stroke_seed, has_strokes = _draw_random_walker_stroke_seeds(
        hint.shape,
        draw_strokes,
        hint,
    )
    stroke_half_width = _draw_strokes_half_width(draw_strokes)
    geometry_snap_all = np.zeros_like(hint, dtype=bool)
    if has_strokes:
        _geometry_seed, _geometry_candidate, geometry_support = _draw_stroke_geometry_snap_support(
            guide,
            hint.shape,
            draw_strokes,
            radius,
            strength,
        )
        if geometry_support is not None:
            geometry_snap_all = np.asarray(geometry_support, dtype=bool) & ~bg_stroke_seed
    guide_u8 = _draw_grabcut_band_guide_image(guide, strength)

    n_labels, labels = cv2.connectedComponents(hint.astype(np.uint8), connectivity=8)
    seed_all = np.zeros_like(hint, dtype=bool)
    bg_seed_all = np.zeros_like(hint, dtype=bool)
    candidate_all = np.zeros_like(hint, dtype=bool)
    support_all = np.zeros_like(hint, dtype=bool)
    target_edge_all = np.zeros_like(hint, dtype=bool)
    raw_target_edge_all = np.zeros_like(hint, dtype=bool)
    probable_fg_all = np.zeros_like(hint, dtype=bool)
    probable_bg_all = np.zeros_like(hint, dtype=bool)
    grabcut_result_all = np.zeros_like(hint, dtype=bool)

    for label_id in range(1, n_labels):
        component = labels == label_id
        if not np.any(component):
            continue

        roi = _expanded_bbox(component, search_radius + 8)
        y0, y1, x0, x1 = roi
        sl = np.s_[y0:y1, x0:x1]
        comp_roi = component[sl]
        guide_roi = guide_u8[sl]
        edge_strength_roi = edge_strength[sl] if edge_strength is not None else None
        image_edge_roi = image_edge[sl]
        dist_to_component = _distance_from(comp_roi)
        candidate_roi = comp_roi | (dist_to_component <= float(search_radius))
        if not np.any(candidate_roi):
            continue

        component_half_width = _component_half_width(comp_roi, stroke_half_width)
        target_edge_roi = _draw_component_target_edge(
            image_edge_roi,
            edge_strength_roi if edge_strength_roi is not None else np.zeros_like(comp_roi, dtype=np.float32),
            comp_roi,
            search_radius,
            component_half_width,
            seed=fg_stroke_seed[sl] if has_strokes else None,
            seed_from_stroke=has_strokes,
            strength=strength,
        )
        raw_target_edge_roi = target_edge_roi.copy()
        if (
                has_strokes
                and np.any(geometry_snap_all)
                and float(search_radius) >= max(80.0, float(component_half_width) * 5.0)):
            target_edge_roi = _draw_grabcut_band_geometry_target_edge(
                target_edge_roi,
                geometry_snap_all[sl],
                component_half_width,
            )

        fg_roi = _draw_grabcut_band_fg_seed(
            comp_roi,
            seed_mask_bool[sl] if seed_mask_bool is not None else None,
            fg_stroke_seed[sl] if has_strokes else None,
            has_strokes,
        )
        if not np.any(fg_roi):
            fg_roi = comp_roi.copy()
        target_edge_is_reliable = _draw_grabcut_band_target_edge_reliable(
            target_edge_roi,
            fg_roi,
            comp_roi,
            component_half_width,
            has_strokes,
        )
        if not target_edge_is_reliable:
            target_edge_roi = np.zeros_like(target_edge_roi, dtype=bool)
        if (
                np.any(target_edge_roi)
                and (
                    has_strokes
                    or (
                        search_radius <= 2
                        and int(np.count_nonzero(target_edge_roi)) <= 512
                    )
                )):
            original_fg_roi = fg_roi.copy()
            fg_roi, rejected_fg_roi = _draw_random_walker_filter_fg_seed_crossing(
                fg_roi,
                target_edge_roi,
                comp_roi,
            )
            if not np.any(fg_roi):
                fg_roi = original_fg_roi
                rejected_fg_roi = np.zeros_like(fg_roi, dtype=bool)
        else:
            rejected_fg_roi = np.zeros_like(fg_roi, dtype=bool)

        probable_fg_roi = _draw_grabcut_band_probable_fg(
            comp_roi,
            fg_roi,
            dist_to_component,
            component_half_width,
            search_radius,
        )
        probable_bg_roi = candidate_roi & ~probable_fg_roi
        bg_roi = _draw_grabcut_band_bg_seed(
            candidate_roi,
            comp_roi,
            dist_to_component,
            search_radius,
            fg_roi,
        )
        bg_roi |= rejected_fg_roi & candidate_roi
        bg_roi |= bg_stroke_seed[sl] & candidate_roi
        bg_roi &= ~fg_roi

        if edge_strength_roi is None or not _draw_grabcut_band_has_usable_edge(
                edge_strength_roi,
                candidate_roi,
                comp_roi,
                target_edge_roi):
            support_roi = comp_roi.copy()
            grabcut_roi = comp_roi.copy()
        else:
            support_roi, grabcut_roi = _solve_draw_grabcut_band_component(
                guide_roi,
                comp_roi,
                candidate_roi,
                fg_roi,
                bg_roi,
                probable_fg_roi,
                probable_bg_roi,
                image_edge_roi,
                target_edge_roi,
                search_radius,
                strength,
                component_half_width,
            )

        support_roi = _draw_grabcut_band_cleanup(
            support_roi,
            comp_roi,
            candidate_roi,
            fg_roi,
            bg_roi,
            target_edge_roi,
            component_half_width,
        )
        if has_strokes and np.any(geometry_snap_all):
            support_roi = _draw_grabcut_band_geometry_guard(
                support_roi,
                comp_roi,
                candidate_roi,
                fg_roi,
                bg_roi,
                target_edge_roi,
                geometry_snap_all[sl],
                search_radius,
                component_half_width,
            )

        seed_all[sl] |= fg_roi
        bg_seed_all[sl] |= bg_roi
        candidate_all[sl] |= candidate_roi
        support_all[sl] |= support_roi
        target_edge_all[sl] |= target_edge_roi
        raw_target_edge_all[sl] |= raw_target_edge_roi
        probable_fg_all[sl] |= probable_fg_roi
        probable_bg_all[sl] |= probable_bg_roi
        grabcut_result_all[sl] |= grabcut_roi

    support_all = _preserve_draw_component_separation(hint, support_all)
    support_all &= ~bg_stroke_seed
    extra_debug_planes = [
        ("image_edge", edge_strength if edge_strength is not None else empty.astype(np.float32)),
        ("hint_boundary", hint_boundary),
        ("raw_target_edge", raw_target_edge_all),
        ("target_edge", target_edge_all),
        ("hard_fg", seed_all),
        ("hard_bg", bg_seed_all),
        ("band", candidate_all),
        ("probable_fg", probable_fg_all),
        ("probable_bg", probable_bg_all),
        ("grabcut_result", grabcut_result_all),
    ]
    if np.any(geometry_snap_all):
        extra_debug_planes.append(("geometry_snap", geometry_snap_all))
    return seed_all, candidate_all, support_all, extra_debug_planes


def _draw_grabcut_band_guide_image(guide, strength):
    guide_f = _prepare_guide_image(guide, np.asarray(guide).shape[:2])
    if guide_f is None:
        return _guide_to_grabcut_image(guide)
    lock = float(np.clip(strength, 0, 100)) / 100.0
    # Light blur keeps snow/foliage texture from dominating the GMM while still
    # preserving the broad cloud/sky boundary the user is trying to catch.
    sigma = 1.10 - 0.35 * lock
    if sigma > 0.05:
        guide_f = cv2.GaussianBlur(guide_f, (0, 0), sigma)
    return _guide_to_grabcut_image(guide_f)


def _draw_grabcut_band_fg_seed(component, seed_mask, stroke_seed, has_strokes):
    component = np.asarray(component, dtype=bool)
    if not np.any(component):
        return component.copy()

    if has_strokes and stroke_seed is not None:
        seed = np.asarray(stroke_seed, dtype=bool) & component
        if np.any(seed):
            return _draw_grabcut_band_expand_seed(seed, component)

    if seed_mask is not None:
        seed = np.asarray(seed_mask, dtype=bool) & component
        if np.any(seed):
            return _draw_grabcut_band_expand_seed(seed, component)

    seed = _component_seed(component, None, (0, component.shape[0], 0, component.shape[1]))
    return _draw_grabcut_band_expand_seed(seed, component)


def _draw_grabcut_band_expand_seed(seed, component):
    seed = np.asarray(seed, dtype=bool) & np.asarray(component, dtype=bool)
    if not np.any(seed):
        return seed
    if np.count_nonzero(seed) >= 24:
        return seed
    seed = cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    return seed & np.asarray(component, dtype=bool)


def _draw_grabcut_band_probable_fg(component, fg_seed, dist_to_component, half_width, search_radius):
    component = np.asarray(component, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool)
    if not np.any(component):
        return component.copy()
    inside_dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
    core_floor = max(1.0, min(float(half_width) * 0.35, float(search_radius) * 0.45 + 1.0))
    core = component & (inside_dist >= core_floor)
    if not np.any(core):
        core = component.copy()
    outside_reach = max(3.0, min(float(search_radius), float(half_width) * 0.65 + 1.5))
    near_component = np.asarray(dist_to_component, dtype=np.float32) <= outside_reach
    return core | component | fg_seed | near_component


def _draw_grabcut_band_bg_seed(candidate, component, dist_to_component, search_radius, fg_seed):
    candidate = np.asarray(candidate, dtype=bool)
    component = np.asarray(component, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool)
    if not np.any(candidate):
        return candidate.copy()

    dist_to_component = np.asarray(dist_to_component, dtype=np.float32)
    if search_radius <= 2:
        shell = candidate & ~component
    else:
        shell = candidate & (dist_to_component >= max(1.0, float(search_radius) - 1.5))

    border = np.zeros_like(candidate, dtype=bool)
    border[0, :] = candidate[0, :]
    border[-1, :] = candidate[-1, :]
    border[:, 0] |= candidate[:, 0]
    border[:, -1] |= candidate[:, -1]
    border_distance = max(1.0, min(float(search_radius) - 1.5, 8.0))
    shell |= border & (dist_to_component >= border_distance)
    return shell & ~fg_seed


def _draw_grabcut_band_has_usable_edge(edge_strength, candidate, component, target_edge):
    if edge_strength is None:
        return False
    edge_strength = np.asarray(edge_strength, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=bool)
    component = np.asarray(component, dtype=bool)
    target_edge = np.asarray(target_edge, dtype=bool)
    if not np.any(candidate):
        return False
    if np.any(target_edge):
        return True
    band = candidate & ~component
    local = edge_strength[candidate | band]
    if local.size == 0:
        return False
    return float(np.percentile(local, 96.0)) >= 0.18


def _draw_grabcut_band_target_edge_reliable(target_edge, fg_seed, component, half_width, has_strokes):
    target_edge = np.asarray(target_edge, dtype=bool)
    if not np.any(target_edge):
        return False

    fg_seed = np.asarray(fg_seed, dtype=bool)
    component = np.asarray(component, dtype=bool)
    target_pixels = int(np.count_nonzero(target_edge))
    if target_pixels < max(8, int(round(float(half_width) * 0.75))):
        return False

    if np.any(fg_seed):
        fg_pixels = int(np.count_nonzero(fg_seed))
        # A tiny edge fragment near a long stroke is more dangerous than useful:
        # it splits the brush internally and creates the "hollow stripe" artifact.
        min_fraction = 0.55 if has_strokes else 0.30
        if target_pixels < max(12, int(round(fg_pixels * min_fraction))):
            return False

    boundary = _mask_boundary_bool(component)
    boundary_pixels = int(np.count_nonzero(boundary))
    if boundary_pixels > 0:
        if target_pixels < max(10, int(round(boundary_pixels * 0.035))):
            return False

    return True


def _solve_draw_grabcut_band_component(
        guide,
        component,
        candidate,
        fg_seed,
        bg_seed,
        probable_fg,
        probable_bg,
        image_edge,
        target_edge,
        search_radius,
        strength,
        half_width):
    component = np.asarray(component, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool) & candidate
    bg_seed = np.asarray(bg_seed, dtype=bool) & candidate & ~fg_seed
    probable_fg = np.asarray(probable_fg, dtype=bool) & candidate & ~fg_seed & ~bg_seed
    probable_bg = np.asarray(probable_bg, dtype=bool) & candidate & ~fg_seed & ~bg_seed

    if not np.any(candidate) or not np.any(fg_seed):
        return component.copy(), component.copy()

    gc_mask = np.full(candidate.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[candidate] = cv2.GC_PR_BGD
    gc_mask[probable_bg] = cv2.GC_PR_BGD
    gc_mask[probable_fg] = cv2.GC_PR_FGD
    gc_mask[fg_seed] = cv2.GC_FGD
    gc_mask[bg_seed] = cv2.GC_BGD

    if not _has_grabcut_samples(gc_mask):
        return component.copy(), component.copy()

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    iterations = int(np.clip(1 + round(float(strength) / 45.0), 1, 3))
    try:
        cv2.grabCut(
            np.ascontiguousarray(guide),
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return component.copy(), component.copy()

    grabcut_support = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)) & candidate
    support = grabcut_support | fg_seed
    used_barrier = False
    if np.any(target_edge):
        barrier_support = _draw_grabcut_band_barrier_support(
            guide,
            component,
            candidate,
            fg_seed,
            image_edge,
            target_edge,
            half_width,
        )
        if np.any(barrier_support):
            support = barrier_support
            used_barrier = True
    if (
            used_barrier
            and search_radius <= 2
            and int(np.count_nonzero(target_edge)) <= 512
            and np.array_equal(support, component)):
        compact_support = grabcut_support | fg_seed
        if np.any(compact_support):
            support = compact_support & candidate

    if np.count_nonzero(support) < max(1, int(np.count_nonzero(fg_seed) * 0.85)):
        support = component.copy()
    if np.count_nonzero(support) < max(1, int(np.count_nonzero(component) * 0.18)):
        support = component.copy()
    if np.count_nonzero(support) > max(np.count_nonzero(component) * 8, np.count_nonzero(candidate) * 0.95):
        support = component.copy()

    if (not used_barrier) or int(np.count_nonzero(target_edge)) <= 512:
        edge_surface = _restore_edge_pixels(candidate, support, target_edge)
        support |= edge_surface
    support &= candidate
    return support, grabcut_support


def _draw_grabcut_band_barrier_support(guide, component, candidate, fg_seed, image_edge, target_edge, half_width):
    component = np.asarray(component, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool) & candidate
    image_edge = np.asarray(image_edge, dtype=bool) & candidate
    target_edge = np.asarray(target_edge, dtype=bool) & candidate
    if not np.any(candidate) or not np.any(fg_seed) or not np.any(target_edge):
        return np.zeros_like(candidate, dtype=bool)
    if _draw_grabcut_band_should_keep_filled_component(component):
        return component.copy()

    barrier_edge = _draw_grabcut_band_barrier_edge(image_edge, target_edge)
    if not np.any(barrier_edge):
        barrier_edge = target_edge

    close_radius = int(max(1, min(7, round(float(half_width) * 0.38))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_radius * 2 + 1, close_radius * 2 + 1),
    )
    barrier = cv2.morphologyEx(barrier_edge.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    barrier = cv2.dilate(barrier, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0

    free = candidate & ~barrier
    seed = fg_seed & free
    if not np.any(seed):
        seed = _draw_grabcut_band_color_seed_side(guide, free, fg_seed, component)
    if not np.any(seed):
        return component.copy()

    selected = _connected_to_seed(free, seed)
    support = selected | fg_seed
    support |= _restore_edge_pixels(candidate, support, barrier_edge)
    return support & candidate


def _draw_grabcut_band_color_seed_side(guide, free, fg_seed, component):
    free = np.asarray(free, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool)
    component = np.asarray(component, dtype=bool)
    if not np.any(free) or not np.any(fg_seed):
        return np.zeros_like(free, dtype=bool)

    guide_arr = np.asarray(guide)
    if guide_arr.ndim == 2:
        guide_arr = np.repeat(guide_arr[..., None], 3, axis=2)
    if guide_arr.dtype != np.float32:
        guide_arr = guide_arr.astype(np.float32) / 255.0
    fg_samples = guide_arr[fg_seed]
    if fg_samples.size == 0:
        return np.zeros_like(free, dtype=bool)
    fg_color = np.median(fg_samples.reshape(-1, guide_arr.shape[-1]), axis=0)

    n_labels, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    if n_labels <= 1:
        return np.zeros_like(free, dtype=bool)

    near_component = component | (
        _distance_from(component) <= max(2.0, min(8.0, _hint_half_width(component) * 0.65))
    )
    scores = []
    for label_id in range(1, n_labels):
        label_mask = labels == label_id
        sample_mask = label_mask & near_component
        if np.count_nonzero(sample_mask) < 4:
            sample_mask = label_mask
        samples = guide_arr[sample_mask]
        if samples.size == 0:
            continue
        mean_color = np.mean(samples.reshape(-1, guide_arr.shape[-1]), axis=0)
        color_dist = float(np.linalg.norm(mean_color - fg_color))
        component_touch = int(np.count_nonzero(label_mask & near_component))
        area = int(np.count_nonzero(label_mask))
        scores.append((color_dist, -component_touch, -area, label_id))
    if not scores:
        return np.zeros_like(free, dtype=bool)

    scores.sort()
    best_dist = scores[0][0]
    keep = [
        label_id
        for color_dist, _touch, _area, label_id in scores
        if color_dist <= best_dist + 0.075
    ]
    if not keep:
        keep = [scores[0][3]]
    return np.isin(labels, np.asarray(keep, dtype=np.int32)) & free


def _draw_grabcut_band_barrier_edge(image_edge, target_edge):
    image_edge = np.asarray(image_edge, dtype=bool)
    target_edge = np.asarray(target_edge, dtype=bool)
    if not np.any(image_edge) or not np.any(target_edge):
        return target_edge.copy()
    touch_scope = cv2.dilate(
        target_edge.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ) > 0
    n_labels, labels = cv2.connectedComponents(image_edge.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return target_edge.copy()
    touched = np.unique(labels[touch_scope & image_edge])
    touched = touched[touched > 0]
    if touched.size == 0:
        return target_edge.copy()
    return np.isin(labels, touched)


def _draw_grabcut_band_should_keep_filled_component(component):
    component = np.asarray(component, dtype=bool)
    if not np.any(component):
        return False
    ys, xs = np.where(component)
    bbox_area = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    if bbox_area <= 0:
        return False
    fill_ratio = float(np.count_nonzero(component)) / float(bbox_area)
    # A thick freehand line can have a high fill ratio in its local bbox. Only
    # protect deliberately filled masks, such as rectangular/painted blocks.
    return fill_ratio >= 0.90 and int(np.count_nonzero(component)) >= 512


def _draw_grabcut_band_cleanup(
        support,
        component,
        candidate,
        fg_seed,
        bg_seed,
        target_edge,
        half_width):
    support = np.asarray(support, dtype=bool) & np.asarray(candidate, dtype=bool)
    component = np.asarray(component, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool)
    bg_seed = np.asarray(bg_seed, dtype=bool)
    target_edge = np.asarray(target_edge, dtype=bool)
    if not np.any(support):
        support = component.copy()

    support |= fg_seed
    support &= ~bg_seed
    if np.any(target_edge) and int(np.count_nonzero(target_edge)) <= 512:
        support |= _restore_edge_pixels(candidate, support, target_edge)
    if not np.any(target_edge):
        support = _fill_draw_support_pinholes(
            support,
            component,
            candidate,
            half_width,
        )
    if np.count_nonzero(support) < max(1, int(np.count_nonzero(fg_seed) * 0.75)):
        support = component.copy()
    return support & np.asarray(candidate, dtype=bool)


def _draw_grabcut_band_geometry_guard(
        support,
        component,
        candidate,
        fg_seed,
        bg_seed,
        target_edge,
        geometry_support,
        search_radius,
        half_width):
    support = np.asarray(support, dtype=bool) & np.asarray(candidate, dtype=bool)
    component = np.asarray(component, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    geometry_support = np.asarray(geometry_support, dtype=bool) & candidate
    if not np.any(support) or not np.any(component) or not np.any(geometry_support):
        return support

    component_pixels = int(np.count_nonzero(component))
    support_pixels = int(np.count_nonzero(support))
    geometry_pixels = int(np.count_nonzero(geometry_support))
    if component_pixels <= 0 or geometry_pixels <= 0:
        return support

    overlap = int(np.count_nonzero(geometry_support & component))
    overlap_floor = max(8, int(round(min(component_pixels, geometry_pixels) * 0.18)))
    if overlap < overlap_floor:
        return support

    radius_gate = float(search_radius) >= max(24.0, float(half_width) * 2.4)
    overgrown = support_pixels > max(
        int(round(component_pixels * 2.4)),
        int(round(geometry_pixels * 1.65)),
    )
    far_limit = max(
        float(half_width) * 1.3,
        min(float(search_radius) * 0.35, float(half_width) * 3.0),
    )
    far_growth = support & ~component & (_distance_from(component) > far_limit)
    far_growth_pixels = int(np.count_nonzero(far_growth))
    far_growth = far_growth_pixels > max(32, int(round(geometry_pixels * 0.10)))
    if not (radius_gate and (overgrown or far_growth)):
        return support

    near_geometry = _distance_from(geometry_support) <= max(1.5, min(4.0, float(half_width) * 0.35))
    guarded = geometry_support | (support & near_geometry) | (np.asarray(fg_seed, dtype=bool) & candidate)
    guarded &= candidate
    guarded &= ~np.asarray(bg_seed, dtype=bool)
    if int(np.count_nonzero(guarded)) < max(1, int(round(component_pixels * 0.35))):
        return support

    return _draw_grabcut_band_cleanup(
        guarded,
        component,
        candidate,
        fg_seed,
        bg_seed,
        target_edge,
        half_width,
    )


def _draw_grabcut_band_geometry_target_edge(target_edge, geometry_support, half_width):
    target_edge = np.asarray(target_edge, dtype=bool)
    geometry_support = np.asarray(geometry_support, dtype=bool)
    if not np.any(target_edge) or not np.any(geometry_support):
        return target_edge

    geometry_boundary = _mask_boundary_bool(geometry_support)
    if not np.any(geometry_boundary):
        return target_edge

    reach = max(2.5, min(7.0, float(half_width) * 0.45 + 2.0))
    near_geometry_boundary = _distance_from(geometry_boundary) <= reach
    filtered = target_edge & near_geometry_boundary
    if not np.any(filtered):
        return target_edge

    n_labels, labels = cv2.connectedComponents(filtered.astype(np.uint8), connectivity=8)
    if n_labels <= 2:
        return filtered

    geometry_touch = cv2.dilate(
        geometry_boundary.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ) > 0
    keep = np.zeros_like(filtered, dtype=bool)
    min_pixels = max(4, int(round(float(half_width) * 0.25)))
    for label_id in range(1, n_labels):
        part = labels == label_id
        if int(np.count_nonzero(part)) < min_pixels and not np.any(part & geometry_touch):
            continue
        keep |= part
    return keep if np.any(keep) else filtered


def _draw_random_walker_support(guide, mask, radius, strength, seed_mask=None, draw_strokes=None):
    if not draw_strokes:
        return _draw_component_edge_snap_support(
            guide,
            mask,
            radius,
            strength,
            seed_mask=seed_mask,
            draw_strokes=draw_strokes,
        )

    mask_f = _as_mask(mask)
    hint = mask_f > 0.02
    h, w = hint.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if not np.any(hint):
        return empty, empty, empty, []

    search_radius = int(max(1, round(float(radius))))
    edge_strength = _draw_snap_edge_strength(guide)
    if edge_strength is None or _sparse is None or _sparse_linalg is None:
        return hint.copy(), hint.copy(), hint.copy(), []

    fg_stroke_seed, bg_stroke_seed, has_strokes = _draw_random_walker_stroke_seeds(
        hint.shape,
        draw_strokes,
        hint,
    )
    stroke_half_width = _draw_strokes_half_width(draw_strokes)
    image_edge = _draw_component_image_edge(edge_strength, strength)
    hint_boundary = _mask_boundary_strength(hint)
    seed_mask_bool = _resize_bool_mask(seed_mask, hint.shape) if seed_mask is not None else None

    n_labels, labels = cv2.connectedComponents(hint.astype(np.uint8), connectivity=8)
    seed_all = np.zeros_like(hint, dtype=bool)
    bg_seed_all = np.zeros_like(hint, dtype=bool)
    candidate_all = np.zeros_like(hint, dtype=bool)
    support_all = np.zeros_like(hint, dtype=bool)
    probability_all = np.zeros_like(mask_f, dtype=np.float32)
    target_edge_all = np.zeros_like(hint, dtype=bool)

    for label_id in range(1, n_labels):
        component = labels == label_id
        if not np.any(component):
            continue

        roi = _expanded_bbox(component, search_radius + 8)
        y0, y1, x0, x1 = roi
        comp_roi = component[y0:y1, x0:x1]
        guide_roi = guide[y0:y1, x0:x1]
        edge_roi = edge_strength[y0:y1, x0:x1]
        image_edge_roi = image_edge[y0:y1, x0:x1]

        component_half_width = _component_half_width(comp_roi, stroke_half_width)
        if (not has_strokes) and _draw_random_walker_component_is_filled(comp_roi):
            sl = np.s_[y0:y1, x0:x1]
            seed_all[sl] |= comp_roi
            candidate_all[sl] |= comp_roi
            support_all[sl] |= comp_roi
            probability_all[sl] = np.maximum(probability_all[sl], comp_roi.astype(np.float32))
            continue

        dist_to_component = _distance_from(comp_roi)
        candidate_roi = comp_roi | (dist_to_component <= float(search_radius))

        if has_strokes:
            fg_roi = fg_stroke_seed[y0:y1, x0:x1] & comp_roi
        elif _draw_random_walker_component_is_filled(comp_roi):
            fg_roi = comp_roi.copy()
        elif seed_mask_bool is not None:
            fg_roi = seed_mask_bool[y0:y1, x0:x1] & comp_roi
        else:
            fg_roi = _component_seed(comp_roi, None, (0, comp_roi.shape[0], 0, comp_roi.shape[1]))
        if not np.any(fg_roi):
            fg_roi = _component_seed(comp_roi, None, (0, comp_roi.shape[0], 0, comp_roi.shape[1]))
        if not np.any(fg_roi):
            fg_roi = comp_roi

        bg_roi = bg_stroke_seed[y0:y1, x0:x1] & candidate_roi
        bg_roi |= _draw_random_walker_outer_bg(candidate_roi, comp_roi, dist_to_component, search_radius)
        bg_roi &= ~fg_roi

        target_edge_roi = _draw_component_target_edge(
            image_edge_roi,
            edge_roi,
            comp_roi,
            search_radius,
            component_half_width,
            seed=fg_roi if has_strokes else None,
            seed_from_stroke=has_strokes,
            strength=strength,
        )
        if has_strokes:
            original_fg_roi = fg_roi.copy()
            fg_roi, rejected_fg_roi = _draw_random_walker_filter_fg_seed_crossing(
                fg_roi,
                target_edge_roi,
                comp_roi,
            )
            if not np.any(fg_roi):
                fg_roi = original_fg_roi
                rejected_fg_roi = np.zeros_like(fg_roi, dtype=bool)
            bg_roi |= rejected_fg_roi & candidate_roi
            bg_roi &= ~fg_roi

        support_roi, probability_roi = _solve_draw_random_walker_component(
            guide_roi,
            comp_roi,
            candidate_roi,
            fg_roi,
            bg_roi,
            edge_roi,
            target_edge_roi,
            search_radius,
            strength,
            component_half_width,
            has_strokes,
        )

        sl = np.s_[y0:y1, x0:x1]
        seed_all[sl] |= fg_roi
        bg_seed_all[sl] |= bg_roi
        candidate_all[sl] |= candidate_roi
        support_all[sl] |= support_roi
        probability_all[sl] = np.maximum(probability_all[sl], probability_roi)
        target_edge_all[sl] |= target_edge_roi

    support_all = _preserve_draw_component_separation(hint, support_all)
    extra_debug_planes = [
        ("image_edge", edge_strength),
        ("hint_boundary", hint_boundary),
        ("target_edge", target_edge_all),
        ("bg_seed", bg_seed_all),
        ("probability", probability_all),
    ]
    return seed_all, candidate_all, support_all, extra_debug_planes


def _draw_random_walker_stroke_seeds(shape, draw_strokes, final_hint):
    h, w = int(shape[0]), int(shape[1])
    fg_seed = np.zeros((h, w), dtype=bool)
    bg_seed = np.zeros((h, w), dtype=bool)
    if not draw_strokes:
        return fg_seed, bg_seed, False

    saw_stroke = False
    for stroke in draw_strokes:
        points = _stroke_points_array(stroke)
        if points.shape[0] == 0:
            continue
        saw_stroke = True
        size = float(max(1.0, getattr(stroke, "size", 1.0)))
        brush = _stroke_brush_mask((h, w), points, size)
        if bool(getattr(stroke, "is_erasing", False)):
            fg_seed &= ~brush
            bg_seed |= brush
            continue

        center = _stroke_center_mask((h, w), points, size)
        fg_seed |= center
        bg_seed &= ~center

    if not saw_stroke:
        return fg_seed, bg_seed, False
    fg_seed &= np.asarray(final_hint, dtype=bool)
    bg_seed &= ~fg_seed
    return fg_seed, bg_seed, True


def _draw_random_walker_component_is_filled(component):
    component = np.asarray(component, dtype=bool)
    if not np.any(component):
        return False
    ys, xs = np.where(component)
    bbox_area = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    if bbox_area <= 0:
        return False
    fill_ratio = float(np.count_nonzero(component)) / float(bbox_area)
    return fill_ratio >= 0.65


def _draw_random_walker_filter_fg_seed_crossing(fg_seed, target_edge, component):
    fg_seed = np.asarray(fg_seed, dtype=bool)
    target_edge = np.asarray(target_edge, dtype=bool)
    component = np.asarray(component, dtype=bool)
    empty = np.zeros_like(fg_seed, dtype=bool)
    if not np.any(fg_seed) or not np.any(target_edge):
        return fg_seed, empty

    barrier = cv2.dilate(
        target_edge.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ) > 0
    free = component & ~barrier
    if not np.any(free):
        return fg_seed, empty

    n_labels, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    if n_labels <= 2:
        return fg_seed, empty

    seed_counts = np.bincount(labels[fg_seed & free], minlength=n_labels)
    hint_counts = np.bincount(labels[component & free], minlength=n_labels)
    max_count = int(seed_counts[1:].max(initial=0))
    if max_count <= 0:
        return fg_seed, empty

    seeded_labels = [
        label_id
        for label_id in range(1, n_labels)
        if int(seed_counts[label_id]) > 0
    ]
    if len(seeded_labels) > 1:
        max_seed_label = int(np.argmax(seed_counts[1:]) + 1)
        max_hint_count = int(hint_counts[1:].max(initial=0))
        if max_hint_count > 0 and int(hint_counts[max_seed_label]) < int(round(max_hint_count * 0.95)):
            area_keep = [
                label_id
                for label_id in seeded_labels
                if int(hint_counts[label_id]) >= int(round(max_hint_count * 0.95))
            ]
            if area_keep:
                kept = fg_seed & np.isin(labels, np.asarray(area_keep, dtype=np.int32))
                if np.any(kept):
                    rejected = fg_seed & ~kept
                    return kept, rejected

    keep_labels = []
    for label_id in range(1, n_labels):
        count = int(seed_counts[label_id])
        if count >= max(6, int(round(max_count * 0.35))):
            keep_labels.append(label_id)
    if not keep_labels:
        return fg_seed, empty

    kept = fg_seed & np.isin(labels, np.asarray(keep_labels, dtype=np.int32))
    if not np.any(kept):
        return fg_seed, empty
    rejected = fg_seed & ~kept
    return kept, rejected


def _draw_random_walker_outer_bg(candidate, component, dist_to_component, search_radius):
    candidate = np.asarray(candidate, dtype=bool)
    component = np.asarray(component, dtype=bool)
    if not np.any(candidate):
        return candidate.copy()

    if search_radius <= 2:
        shell = candidate & ~component
    else:
        shell_distance = max(1.0, float(search_radius) - 1.5)
        shell = candidate & (np.asarray(dist_to_component, dtype=np.float32) >= shell_distance)

    border = np.zeros_like(candidate, dtype=bool)
    border[0, :] = candidate[0, :]
    border[-1, :] = candidate[-1, :]
    border[:, 0] |= candidate[:, 0]
    border[:, -1] |= candidate[:, -1]
    border_distance = max(1.0, min(float(search_radius) - 1.5, 8.0))
    shell |= border & (np.asarray(dist_to_component, dtype=np.float32) >= border_distance)
    return shell


def _solve_draw_random_walker_component(
        guide,
        component,
        candidate,
        fg_seed,
        bg_seed,
        edge_strength,
        target_edge,
        search_radius,
        strength,
        half_width,
        has_strokes):
    component = np.asarray(component, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool) & candidate
    bg_seed = np.asarray(bg_seed, dtype=bool) & candidate & ~fg_seed

    if not np.any(candidate) or not np.any(fg_seed):
        return component.copy(), component.astype(np.float32)

    local_edge = np.asarray(target_edge, dtype=bool) & candidate
    if not np.any(local_edge):
        return component.copy(), component.astype(np.float32)

    if not np.any(bg_seed):
        dist_to_component = _distance_from(component)
        bg_seed = _draw_random_walker_outer_bg(candidate, component, dist_to_component, search_radius) & ~fg_seed
    if not np.any(bg_seed):
        return component.copy(), component.astype(np.float32)

    probability = _random_walker_probability(
        guide,
        candidate,
        fg_seed,
        bg_seed,
        edge_strength,
        strength,
    )
    if probability is None:
        return component.copy(), component.astype(np.float32)

    support = _compose_draw_random_walker_support(
        probability,
        component,
        candidate,
        fg_seed,
        bg_seed,
        local_edge,
        half_width,
        search_radius,
        strength,
        has_strokes,
    )

    if has_strokes:
        support = _draw_random_walker_limit_damage(support, component, fg_seed)

    support = _fill_draw_support_pinholes(
        support,
        component,
        candidate,
        half_width,
    )
    if not np.any(support):
        support = component.copy()
    return support, probability


def _random_walker_probability(guide, candidate, fg_seed, bg_seed, edge_strength, strength):
    candidate = np.asarray(candidate, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool) & candidate
    bg_seed = np.asarray(bg_seed, dtype=bool) & candidate & ~fg_seed
    unknown = candidate & ~fg_seed & ~bg_seed
    probability = np.zeros(candidate.shape, dtype=np.float32)
    probability[fg_seed] = 1.0
    if not np.any(unknown):
        return probability

    unknown_count = int(np.count_nonzero(unknown))
    if unknown_count > 140000:
        return None

    uid = -np.ones(candidate.shape, dtype=np.int32)
    uid[unknown] = np.arange(unknown_count, dtype=np.int32)

    diag = np.zeros(unknown_count, dtype=np.float64)
    b = np.zeros(unknown_count, dtype=np.float64)
    rows = []
    cols = []
    data = []

    guide_arr = _prepare_guide_image(guide, candidate.shape)
    if guide_arr is None:
        guide_arr = np.zeros((*candidate.shape, 3), dtype=np.float32)
    if guide_arr.ndim == 2:
        guide_arr = np.repeat(guide_arr[..., None], 3, axis=2)
    edge_arr = np.asarray(edge_strength, dtype=np.float32)
    if edge_arr.shape[:2] != candidate.shape[:2]:
        edge_arr = cv2.resize(
            edge_arr,
            (int(candidate.shape[1]), int(candidate.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )
    edge_arr = cv2.GaussianBlur(edge_arr, (0, 0), 0.8)

    def add_edges(valid, src_slice, dst_slice):
        if not np.any(valid):
            return
        src_unknown = uid[src_slice][valid]
        dst_unknown = uid[dst_slice][valid]
        src_fg = fg_seed[src_slice][valid]
        dst_fg = fg_seed[dst_slice][valid]
        src_bg = bg_seed[src_slice][valid]
        dst_bg = bg_seed[dst_slice][valid]
        src_color = guide_arr[src_slice][valid]
        dst_color = guide_arr[dst_slice][valid]
        src_edge = edge_arr[src_slice][valid]
        dst_edge = edge_arr[dst_slice][valid]
        weights = _random_walker_edge_weights(
            src_color,
            dst_color,
            np.maximum(src_edge, dst_edge),
            strength,
        )

        src_is_unknown = src_unknown >= 0
        dst_is_unknown = dst_unknown >= 0
        both_unknown = src_is_unknown & dst_is_unknown
        if np.any(both_unknown):
            su = src_unknown[both_unknown]
            du = dst_unknown[both_unknown]
            ww = weights[both_unknown]
            diag[su] += ww
            diag[du] += ww
            rows.extend(su.tolist())
            cols.extend(du.tolist())
            data.extend((-ww).tolist())
            rows.extend(du.tolist())
            cols.extend(su.tolist())
            data.extend((-ww).tolist())

        src_unknown_dst_seed = src_is_unknown & ~dst_is_unknown
        if np.any(src_unknown_dst_seed):
            su = src_unknown[src_unknown_dst_seed]
            ww = weights[src_unknown_dst_seed]
            diag[su] += ww
            b[su] += ww * dst_fg[src_unknown_dst_seed].astype(np.float64)

        dst_unknown_src_seed = dst_is_unknown & ~src_is_unknown
        if np.any(dst_unknown_src_seed):
            du = dst_unknown[dst_unknown_src_seed]
            ww = weights[dst_unknown_src_seed]
            diag[du] += ww
            b[du] += ww * src_fg[dst_unknown_src_seed].astype(np.float64)

    valid_h = candidate[:, :-1] & candidate[:, 1:]
    add_edges(valid_h, np.s_[:, :-1], np.s_[:, 1:])
    valid_v = candidate[:-1, :] & candidate[1:, :]
    add_edges(valid_v, np.s_[:-1, :], np.s_[1:, :])

    if np.count_nonzero(diag > 0.0) != unknown_count:
        diag = np.maximum(diag, 1e-6)
    rows.extend(np.arange(unknown_count, dtype=np.int32).tolist())
    cols.extend(np.arange(unknown_count, dtype=np.int32).tolist())
    data.extend(diag.tolist())

    matrix = _sparse.csr_matrix((data, (rows, cols)), shape=(unknown_count, unknown_count))
    try:
        if unknown_count <= 80000:
            values = _sparse_linalg.spsolve(matrix, b)
        else:
            values, info = _sparse_linalg.cg(matrix, b, rtol=1e-4, atol=0.0, maxiter=240)
            if info != 0:
                return None
    except TypeError:
        try:
            values, info = _sparse_linalg.cg(matrix, b, tol=1e-4, maxiter=240)
            if info != 0:
                return None
        except Exception:
            return None
    except Exception:
        return None

    probability[unknown] = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    probability[fg_seed] = 1.0
    probability[bg_seed] = 0.0
    return probability


def _random_walker_edge_weights(src_color, dst_color, edge, strength):
    color_delta = np.asarray(src_color, dtype=np.float32) - np.asarray(dst_color, dtype=np.float32)
    color_dist2 = np.sum(color_delta * color_delta, axis=1)
    edge = np.asarray(edge, dtype=np.float32)
    lock = float(np.clip(strength, 0, 100)) / 100.0
    color_beta = 18.0 + 28.0 * lock
    edge_beta = 5.0 + 24.0 * lock
    weights = np.exp(-(color_beta * color_dist2 + edge_beta * edge * edge))
    return np.clip(weights, 1e-5, 1.0).astype(np.float64, copy=False)


def _compose_draw_random_walker_support(
        probability,
        component,
        candidate,
        fg_seed,
        bg_seed,
        target_edge,
        half_width,
        search_radius,
        strength,
        has_strokes):
    probability = np.asarray(probability, dtype=np.float32)
    component = np.asarray(component, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool) & candidate
    bg_seed = np.asarray(bg_seed, dtype=bool) & candidate & ~fg_seed
    target_edge = np.asarray(target_edge, dtype=bool) & candidate

    lock = float(np.clip(strength, 0, 100)) / 100.0
    # The user's brush is the primary intent. Random Walker is used to clip
    # clearly wrong pixels and add nearby boundary pixels, not to redraw the
    # whole stroke from a thin center seed.
    keep_threshold = 0.018 + 0.060 * lock if has_strokes else 0.050 + 0.045 * lock
    near_fg = _distance_from(fg_seed) <= max(1.5, float(half_width) * 0.45)
    support = (component & ((probability >= keep_threshold) | near_fg)) | fg_seed

    if np.any(bg_seed & component):
        bg_reach = max(3.0, min(12.0, float(half_width) * 0.75 + 2.0))
        near_bg = _distance_from(bg_seed & component) <= bg_reach
        cut_threshold = max(0.32, keep_threshold + 0.18)
        support &= ~(component & near_bg & (probability < cut_threshold))

    dist_to_component = _distance_from(component)
    add_reach = max(2.0, min(float(search_radius), max(3.0, float(half_width) * 1.75)))
    near_component = dist_to_component <= add_reach
    if np.any(target_edge):
        near_target_edge = _distance_from(target_edge) <= max(2.5, min(10.0, float(half_width) * 0.55))
    else:
        near_target_edge = np.zeros_like(candidate, dtype=bool)
    scale_relax = max(0.0, min(0.16, (float(half_width) - 8.0) * 0.0125))
    add_threshold = 0.62 - 0.22 * lock - scale_relax
    support |= (
        (~component)
        & candidate
        & near_component
        & near_target_edge
        & (probability >= add_threshold)
    )

    edge_threshold = max(0.08, add_threshold - 0.28)
    edge_near_support = _distance_from(support) <= max(2.2, min(14.0, float(half_width) * 0.75))
    edge_band = (
        target_edge
        & candidate
        & near_component
        & edge_near_support
        & (probability >= edge_threshold)
    )
    if np.any(edge_band):
        edge_band = cv2.dilate(
            edge_band.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ) > 0
        support |= edge_band & candidate & near_component

    return support


def _draw_random_walker_edge_band(
        support,
        probability,
        target_edge,
        candidate,
        component,
        half_width,
        search_radius):
    target_edge = np.asarray(target_edge, dtype=bool)
    if not np.any(target_edge):
        return np.zeros_like(candidate, dtype=bool)
    support_reach = max(2.2, min(6.0, float(half_width) + 2.0))
    near_support = _distance_from(np.asarray(support, dtype=bool)) <= support_reach
    component_reach = max(2.5, min(float(search_radius), float(half_width) + 8.0))
    near_component = _distance_from(np.asarray(component, dtype=bool)) <= component_reach
    uncertain = np.asarray(probability, dtype=np.float32) >= 0.28
    edge_band = target_edge & near_support & near_component & uncertain & np.asarray(candidate, dtype=bool)
    if not np.any(edge_band):
        return edge_band
    edge_band = cv2.dilate(edge_band.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    return edge_band & np.asarray(candidate, dtype=bool) & near_component


def _draw_random_walker_limit_damage(support, component, fg_seed):
    support = np.asarray(support, dtype=bool)
    component = np.asarray(component, dtype=bool)
    fg_seed = np.asarray(fg_seed, dtype=bool)
    component_pixels = int(np.count_nonzero(component))
    if component_pixels <= 0:
        return support
    support_pixels = int(np.count_nonzero(support))
    if support_pixels >= component_pixels * 0.18:
        return support
    fallback = component & (
        _distance_from(fg_seed) <= max(2.0, _hint_half_width(component) * 1.15)
    )
    fallback |= fg_seed
    return fallback if np.any(fallback) else component.copy()


def _resize_bool_mask(mask, shape):
    arr = np.asarray(mask, dtype=bool)
    if arr.shape[:2] == tuple(shape):
        return arr
    return cv2.resize(
        arr.astype(np.uint8),
        (int(shape[1]), int(shape[0])),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _draw_component_edge_snap_support(guide, mask, radius, strength, seed_mask=None, draw_strokes=None):
    mask_f = _as_mask(mask)
    hint = mask_f > 0.02
    h, w = hint.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if not np.any(hint):
        return empty, empty, empty, []

    search_radius = int(max(1, round(float(radius))))
    edge_strength = _draw_snap_edge_strength(guide)
    if edge_strength is None:
        return hint.copy(), hint.copy(), hint.copy(), []

    image_edge = _draw_component_image_edge(edge_strength, strength)
    structure_edge = _make_edge_stop_mask(guide, _draw_barrier_strength(strength))
    if structure_edge is None:
        structure_edge = np.zeros_like(hint, dtype=bool)
    hint_boundary = _mask_boundary_strength(hint)
    stroke_seed = _draw_stroke_replay_seed(hint.shape, draw_strokes, hint)
    stroke_half_width = _draw_strokes_half_width(draw_strokes)
    n_labels, labels = cv2.connectedComponents(hint.astype(np.uint8), connectivity=8)

    seed_all = np.zeros_like(hint, dtype=bool)
    candidate_all = np.zeros_like(hint, dtype=bool)
    support_all = np.zeros_like(hint, dtype=bool)
    accepted_outside_all = np.zeros_like(hint, dtype=bool)
    rejected_radius_all = np.zeros_like(hint, dtype=bool)
    target_edge_all = np.zeros_like(hint, dtype=bool)

    for label_id in range(1, n_labels):
        component = labels == label_id
        if not np.any(component):
            continue
        roi = _expanded_bbox(component, search_radius + 6)
        y0, y1, x0, x1 = roi
        comp_roi = component[y0:y1, x0:x1]
        guide_roi = guide[y0:y1, x0:x1]
        edge_roi = image_edge[y0:y1, x0:x1]
        structure_roi = structure_edge[y0:y1, x0:x1]
        strength_roi = edge_strength[y0:y1, x0:x1]
        seed_roi = _component_seed(comp_roi, stroke_seed, roi)
        if not np.any(seed_roi):
            seed_roi = comp_roi
        component_half_width = _component_half_width(comp_roi, stroke_half_width)
        target_edge_roi = _draw_component_target_edge(
            edge_roi,
            strength_roi,
            comp_roi,
            search_radius,
            component_half_width,
            seed=seed_roi,
            seed_from_stroke=stroke_seed is not None,
            strength=strength,
        )

        dist_to_component = _distance_from(comp_roi)
        edge_scope = dist_to_component <= float(search_radius)
        search_band = edge_scope.copy()
        affinity = _make_color_affinity(
            guide_roi,
            comp_roi,
            search_radius,
            strength,
            sample_mask=seed_roi,
        )
        if affinity is not None:
            search_band &= affinity | comp_roi
        if not np.any(search_band):
            continue
        if stroke_seed is not None and not _draw_target_edge_reliable(
                target_edge_roi,
                comp_roi,
                component_half_width):
            sl = np.s_[y0:y1, x0:x1]
            seed_all[sl] |= seed_roi
            candidate_all[sl] |= search_band
            support_all[sl] |= comp_roi
            target_edge_all[sl] |= target_edge_roi
            continue

        raw_barrier = _draw_component_edge_barrier(
            structure_roi,
            comp_roi,
            search_radius,
            half_width=component_half_width,
            strong=stroke_seed is not None,
        )
        target_barrier = _draw_component_edge_barrier(
            target_edge_roi,
            comp_roi,
            search_radius,
            half_width=component_half_width,
            strong=stroke_seed is not None,
        )
        barrier = raw_barrier | target_barrier
        component_support, accepted_outside, rejected_radius = _partition_draw_component_support(
            comp_roi,
            seed_roi,
            search_band,
            barrier,
            target_edge_roi,
            dist_to_component,
            search_radius,
            edge_scope,
            half_width=component_half_width,
            seed_from_stroke=stroke_seed is not None,
        )

        sl = np.s_[y0:y1, x0:x1]
        seed_all[sl] |= seed_roi
        candidate_all[sl] |= search_band
        support_all[sl] |= component_support
        accepted_outside_all[sl] |= accepted_outside
        rejected_radius_all[sl] |= rejected_radius
        target_edge_all[sl] |= target_edge_roi

    support_all = _preserve_draw_component_separation(hint, support_all)
    extra_debug_planes = [
        ("image_edge", edge_strength),
        ("hint_boundary", hint_boundary),
        ("target_edge", target_edge_all),
        ("accepted", accepted_outside_all),
        ("radius_reject", rejected_radius_all),
    ]
    return seed_all, candidate_all, support_all, extra_debug_planes


def _partition_draw_component_support(
        component,
        seed,
        search_band,
        barrier,
        image_edge,
        dist_to_component,
        search_radius,
        edge_scope=None,
        half_width=None,
        seed_from_stroke=False):
    half_width = _component_half_width(component, half_width)
    if edge_scope is None:
        edge_scope = search_band
    free = search_band & ~barrier
    if not np.any(free):
        empty = np.zeros_like(component, dtype=bool)
        return component.copy(), empty, empty

    seed_in_free = seed & free
    if not np.any(seed_in_free):
        seed_in_free = component & free
    if not np.any(seed_in_free):
        seed_in_free = free & component
    if not np.any(seed_in_free):
        seed_in_free = free

    n_labels, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        empty = np.zeros_like(component, dtype=bool)
        return component.copy(), empty, empty

    label_ids = np.arange(1, n_labels)
    seed_counts = np.bincount(labels[seed_in_free], minlength=n_labels)
    hint_counts = np.bincount(labels[component & free], minlength=n_labels)
    if int(seed_counts[1:].max(initial=0)) > 0:
        selected_labels = _select_partition_labels(seed_counts, hint_counts)
    else:
        selected_labels = _select_partition_labels(hint_counts, hint_counts)

    selected = np.isin(labels, selected_labels)
    if not np.any(selected):
        selected = _connected_to_seed(free, seed_in_free)

    outside = selected & ~component
    if np.any(barrier & component):
        accepted_outside = np.zeros_like(outside, dtype=bool)
        rejected_radius = outside
    else:
        accepted_outside, rejected_radius = _accept_draw_outside_components(
            outside,
            image_edge,
            dist_to_component,
            search_radius,
            max_outside_distance=min(float(search_radius), half_width * 2.1 + 1.0),
        )
    edge_candidate = _draw_edge_surface_scope(
        component | accepted_outside,
        selected | accepted_outside,
        edge_scope,
        dist_to_component,
        half_width,
    )
    edge_surface = _restore_edge_pixels(edge_candidate, selected | accepted_outside, image_edge)
    edge_surface |= _draw_edge_ribbon(
        image_edge,
        selected | accepted_outside | edge_surface,
        edge_scope,
        dist_to_component,
        half_width,
        search_radius,
    )

    # Barrier pixels are allowed back only as the boundary surface adjacent to
    # the accepted side. The seed itself is not restored across the barrier.
    base_inside = component & (selected | edge_surface)
    component_support = (base_inside | accepted_outside | edge_surface) & edge_scope
    component_support = _fill_draw_support_pinholes(
        component_support,
        component,
        edge_scope,
        half_width,
    )
    if seed_from_stroke:
        outside_cut = _draw_outside_edge_cut(
            component,
            search_band,
            barrier,
            dist_to_component,
            half_width,
            search_radius,
        )
        if np.any(outside_cut):
            trimmed = (component_support & ~outside_cut) | edge_surface
            if np.count_nonzero(trimmed) >= np.count_nonzero(component_support) * 0.20:
                component_support = trimmed
    component_support |= _restore_edge_pixels(edge_scope, component_support, image_edge)
    if not np.any(component_support):
        component_support = component & selected
    if not np.any(component_support):
        component_support = component.copy()
    return component_support, accepted_outside, rejected_radius


def _draw_edge_surface_scope(base, selected, edge_scope, dist_to_component, half_width):
    base = np.asarray(base, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    edge_scope = np.asarray(edge_scope, dtype=bool)
    near_selected = cv2.dilate(
        selected.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ) > 0
    edge_distance = float(max(3.25, min(float(half_width) * 0.35 + 1.0, 3.75)))
    near_component = np.asarray(dist_to_component, dtype=np.float32) <= edge_distance
    return (base | near_component) & near_selected & edge_scope


def _draw_edge_ribbon(image_edge, selected, edge_scope, dist_to_component, half_width, search_radius):
    if image_edge is None or not np.any(image_edge) or not np.any(selected):
        return np.zeros_like(edge_scope, dtype=bool)
    edge_scope = np.asarray(edge_scope, dtype=bool)
    base_reach = float(max(3.25, min(float(half_width) * 0.35 + 1.0, 3.75)))
    reach = max(base_reach, _draw_edge_reach(search_radius, half_width))
    selected_distance = _distance_from(np.asarray(selected, dtype=bool))
    near_selected = selected_distance <= reach
    near_component = np.asarray(dist_to_component, dtype=np.float32) <= reach
    ribbon = np.asarray(image_edge, dtype=bool) & near_selected & near_component & edge_scope
    if not np.any(ribbon):
        return ribbon
    ribbon = cv2.dilate(ribbon.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    return ribbon & near_component & edge_scope


def _draw_edge_reach(search_radius, half_width):
    # User radius controls how far Draw Quick Select may pull the boundary.
    # Keep a small cap from the brush width so a huge radius does not make the
    # whole scene's texture edges part of the stroke.
    return float(max(1.0, min(float(search_radius), max(12.0, float(half_width) + 5.0))))


def _fill_draw_support_pinholes(support, component, edge_scope, half_width):
    support = np.asarray(support, dtype=bool)
    if not np.any(support):
        return support

    fill_area = np.asarray(edge_scope, dtype=bool)
    if not np.any(fill_area):
        return support

    holes = fill_area & ~support
    if not np.any(holes):
        return support

    n_labels, labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return support

    component = np.asarray(component, dtype=bool)
    max_hole_area = int(max(12, min(600, (float(half_width) ** 2) * 1.6)))
    filled = support.copy()
    for label_id in range(1, n_labels):
        part = labels == label_id
        area = int(np.count_nonzero(part))
        if area > max_hole_area:
            continue
        # Fill speckles inside the user's painted region. Gaps outside the
        # brush are usually intentional edge cuts or radius rejects.
        if np.count_nonzero(part & component) < area * 0.55:
            continue
        filled |= part
    return filled


def _draw_outside_edge_cut(component, search_band, barrier, dist_to_component, half_width, search_radius):
    component = np.asarray(component, dtype=bool)
    barrier = np.asarray(barrier, dtype=bool)
    if not np.any(component) or not np.any(barrier & component):
        return np.zeros_like(component, dtype=bool)

    cut_reach = float(max(float(search_radius), min(float(half_width), 12.0)))
    cut_scope = np.asarray(search_band, dtype=bool) | (_distance_from(component) <= cut_reach)
    free = cut_scope & ~barrier
    if not np.any(free):
        return np.zeros_like(component, dtype=bool)

    outside_seed = np.zeros_like(component, dtype=bool)
    outside_seed[0, :] = free[0, :]
    outside_seed[-1, :] = free[-1, :]
    outside_seed[:, 0] |= free[:, 0]
    outside_seed[:, -1] |= free[:, -1]
    outside_seed |= free & ~component & (
        np.asarray(dist_to_component, dtype=np.float32) >= float(max(1.0, search_radius - 1.0))
    )
    if not np.any(outside_seed):
        return np.zeros_like(component, dtype=bool)

    outside = _connected_to_seed(free, outside_seed)
    if not np.any(outside & component):
        return np.zeros_like(component, dtype=bool)

    cut_width = float(max(4.0, min(10.0, float(half_width) * 0.55 + 0.5)))
    barrier_distance = _distance_from(barrier)
    return outside & component & (barrier_distance <= cut_width)


def _select_partition_labels(primary_counts, hint_counts):
    primary_counts = np.asarray(primary_counts, dtype=np.int64)
    hint_counts = np.asarray(hint_counts, dtype=np.int64)
    if primary_counts.shape[0] <= 1:
        return np.asarray([], dtype=np.int32)

    max_primary = int(primary_counts[1:].max(initial=0))
    max_hint = int(hint_counts[1:].max(initial=0))
    selected = []
    for label_id in range(1, primary_counts.shape[0]):
        primary = int(primary_counts[label_id])
        hint = int(hint_counts[label_id]) if label_id < hint_counts.shape[0] else 0
        if primary <= 0 and hint <= 0:
            continue
        if primary >= max(8, int(round(max_primary * 0.22))):
            selected.append(label_id)
            continue
        if hint >= max(24, int(round(max_hint * 0.42))) and primary >= 2:
            selected.append(label_id)
    if not selected and max_hint > 0:
        selected.append(int(np.argmax(hint_counts[1:]) + 1))
    return np.asarray(selected, dtype=np.int32)


def _expanded_bbox(mask, pad):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    h, w = mask.shape[:2]
    pad = int(max(0, round(float(pad))))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def _component_seed(component, seed_mask, roi):
    y0, y1, x0, x1 = roi
    if seed_mask is not None:
        seed = np.asarray(seed_mask, dtype=bool)
        if seed.shape == component.shape:
            seed = seed & component
            if np.any(seed):
                return seed
        elif seed.ndim >= 2 and seed.shape[0] >= y1 and seed.shape[1] >= x1:
            seed = seed[y0:y1, x0:x1] & component
            if np.any(seed):
                return seed

    dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
    max_dist = float(dist.max(initial=0.0))
    if max_dist <= 0.0:
        return component.copy()

    local_max = cv2.dilate(dist, np.ones((3, 3), dtype=np.uint8), iterations=1)
    ridge_floor = max(1.0, max_dist * 0.18)
    seed = (dist >= local_max - 1e-4) & (dist >= ridge_floor) & component
    if np.any(seed):
        seed = cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        return seed & component

    cutoff = max(1.0, min(max_dist - 0.25, max_dist * 0.78))
    seed = (dist >= cutoff) & component
    return seed if np.any(seed) else component.copy()


def _draw_stroke_replay_seed(shape, draw_strokes, final_hint):
    if draw_strokes is None:
        return None
    h, w = int(shape[0]), int(shape[1])
    seed = np.zeros((h, w), dtype=bool)
    saw_stroke = False
    for stroke in draw_strokes:
        points = _stroke_points_array(stroke)
        if points.shape[0] == 0:
            continue
        saw_stroke = True
        size = float(max(1.0, getattr(stroke, "size", 1.0)))
        if bool(getattr(stroke, "is_erasing", False)):
            seed &= ~_stroke_brush_mask((h, w), points, size)
        else:
            seed |= _stroke_center_mask((h, w), points, size)
    if not saw_stroke:
        return None
    return seed & np.asarray(final_hint, dtype=bool)


def _draw_component_image_edge(edge_strength, strength):
    strength = float(np.clip(strength, 0, 100))
    threshold = float(np.clip(0.40 - strength * 0.0018, 0.22, 0.42))
    edge = np.asarray(edge_strength, dtype=np.float32) >= threshold
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(edge.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    return edge


def _draw_component_target_edge(
        edge,
        edge_strength,
        component,
        search_radius,
        half_width,
        seed=None,
        seed_from_stroke=False,
        strength=60.0):
    # edge is already thresholded upstream. Do not promote sub-threshold
    # edge_strength here: target_edge becomes a GrabCut barrier, so weak image
    # texture can split the user's stroke into hollow stripes.
    _ = edge_strength, strength
    edge = np.asarray(edge, dtype=bool)
    if not np.any(edge):
        return edge

    component = np.asarray(component, dtype=bool)
    boundary = _mask_boundary_bool(component)
    if not np.any(boundary):
        return edge

    half_width = float(max(1.0, half_width))
    search_radius = float(max(1.0, search_radius))
    boundary_reach = max(
        4.0,
        half_width * 1.35 + 3.0,
        min(search_radius, half_width * 1.85 + 8.0),
    )
    boundary_distance = _distance_from(boundary)
    near_boundary = boundary_distance <= boundary_reach
    boundary_surface_reach = max(3.0, min(search_radius, half_width * 0.75 + 6.0))
    boundary_surface = boundary_distance <= boundary_surface_reach

    local_edge = edge & near_boundary
    if seed is not None:
        seed = np.asarray(seed, dtype=bool)
        if seed.shape[:2] == local_edge.shape[:2] and np.any(seed):
            boundary_edge = local_edge & boundary_surface
            center_reach = _draw_target_edge_center_reach(
                search_radius,
                half_width,
                seed_from_stroke=seed_from_stroke,
            )
            center_edge = local_edge & (_distance_from(seed) <= center_reach)
            local_edge = center_edge | boundary_edge
    if not np.any(local_edge):
        return local_edge

    n_labels, labels = cv2.connectedComponents(local_edge.astype(np.uint8), connectivity=8)
    if n_labels <= 2:
        return local_edge

    min_area = max(8, int(round(half_width * 0.75)))
    min_surface_area = max(3, int(round(half_width * 0.22)))
    keep = np.zeros_like(local_edge, dtype=bool)
    for label_id in range(1, n_labels):
        part = labels == label_id
        area = int(np.count_nonzero(part))
        if area >= min_area or (area >= min_surface_area and np.any(part & boundary_surface)):
            keep |= labels == label_id

    if np.count_nonzero(keep) < max(3, min_area // 2):
        return local_edge
    return keep


def _draw_target_edge_reliable(target_edge, component, half_width):
    target_edge = np.asarray(target_edge, dtype=bool)
    if not np.any(target_edge):
        return False
    component = np.asarray(component, dtype=bool)
    edge_pixels = int(np.count_nonzero(target_edge))
    min_pixels = int(max(6, round(float(half_width) * 0.9)))
    if edge_pixels < min_pixels:
        return False

    boundary_pixels = int(np.count_nonzero(_mask_boundary_bool(component)))
    if boundary_pixels <= 0:
        return True
    min_boundary_fraction = 0.018
    return edge_pixels >= int(round(boundary_pixels * min_boundary_fraction))


def _draw_target_edge_center_reach(search_radius, half_width, seed_from_stroke=False):
    # The final mask boundary can be wide after drawing and erasing. Use the
    # center/ridge seed to keep the snap target near the user's stroke, while
    # still letting Radius pull a nearby edge into the painted band.
    if seed_from_stroke:
        cap = max(14.5, float(half_width) * 1.45 + 8.0)
    elif float(half_width) < 8.0:
        cap = float(search_radius)
    else:
        cap = max(6.0, 15.0 - float(half_width) * 0.75)
    return float(max(5.0, min(float(search_radius), cap)))


def _mask_boundary_bool(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.dilate(mask_u8, kernel, iterations=1) - cv2.erode(mask_u8, kernel, iterations=1)
    return boundary > 0


def _mask_boundary_strength(mask):
    boundary = _mask_boundary_bool(mask)
    boundary = cv2.GaussianBlur(boundary.astype(np.float32), (0, 0), 0.8)
    return np.clip(boundary, 0.0, 1.0).astype(np.float32, copy=False)


def _draw_component_edge_barrier(edge, component, search_radius, half_width=None, strong=False):
    if edge is None or not np.any(edge):
        return np.zeros_like(component, dtype=bool)
    half_width = _component_half_width(component, half_width)
    if strong:
        close_radius = int(max(1, min(11, round(half_width * 0.65))))
        dilate_iterations = 2
    else:
        close_radius = int(max(1, min(7, round(half_width * 0.45))))
        dilate_iterations = 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
    barrier = cv2.morphologyEx(edge.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1)
    barrier = cv2.dilate(barrier, np.ones((3, 3), dtype=np.uint8), iterations=dilate_iterations)
    return barrier > 0


def _accept_draw_outside_components(
        outside,
        image_edge,
        dist_to_component,
        search_radius,
        max_outside_distance=None):
    accepted = np.zeros_like(outside, dtype=bool)
    rejected_radius = np.zeros_like(outside, dtype=bool)
    if not np.any(outside):
        return accepted, rejected_radius

    if max_outside_distance is None:
        max_outside_distance = float(search_radius)
    outside_limit = dist_to_component <= float(max(1.0, max_outside_distance))
    edge_distance = _distance_from(image_edge.astype(bool))
    edge_band = edge_distance <= float(max(2.0, max_outside_distance))
    near_edge = cv2.dilate(image_edge.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=4) > 0
    radius_limit = dist_to_component >= max(1.0, float(search_radius) - 0.75)
    n_labels, labels = cv2.connectedComponents(outside.astype(np.uint8), connectivity=8)
    for label_id in range(1, n_labels):
        part = labels == label_id
        if not np.any(part):
            continue
        touches_edge = bool(np.any(part & near_edge))
        if touches_edge:
            part_accept = part & outside_limit & edge_band
            accepted |= part_accept
            rejected_radius |= part & ~part_accept
        else:
            rejected_radius |= part & radius_limit
    return accepted, rejected_radius


def _preserve_draw_component_separation(hint, support):
    if not np.any(support):
        return support
    n_hint, hint_labels = cv2.connectedComponents(hint.astype(np.uint8), connectivity=8)
    if n_hint <= 2:
        return support

    n_support, support_labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    if n_support <= 1:
        return support

    cleaned = support.copy()
    for label_id in range(1, n_support):
        part = support_labels == label_id
        touched = np.unique(hint_labels[part & hint])
        touched = touched[touched > 0]
        if touched.size <= 1:
            continue
        bridge = part & ~hint
        cleaned[bridge] = False
    return cleaned


def _draw_stroke_geometry_snap_support(guide, shape, draw_strokes, radius, strength):
    h, w = int(shape[0]), int(shape[1])
    support = np.zeros((h, w), dtype=bool)
    candidate = np.zeros((h, w), dtype=bool)
    seed = np.zeros((h, w), dtype=bool)
    edge_strength = _draw_snap_edge_strength(guide)
    if edge_strength is None:
        return seed, candidate, None

    for stroke in draw_strokes:
        points = _stroke_points_array(stroke)
        if points.shape[0] == 0:
            continue
        size = float(max(1.0, getattr(stroke, "size", 1.0)))
        brush = _stroke_brush_mask((h, w), points, size)
        center = _stroke_center_mask((h, w), points, size)
        candidate |= brush
        seed |= center

        if bool(getattr(stroke, "is_erasing", False)):
            support &= ~brush
            continue

        snapped = np.zeros((h, w), dtype=bool)
        margin = size * 0.5 + float(radius) + 8.0
        for segment in _visible_polyline_segments(points, (h, w), margin):
            snapped |= _snap_single_stroke_by_geometry(
                edge_strength,
                segment,
                size,
                brush,
                center,
                radius,
                strength,
            )
        if not np.any(snapped):
            snapped = brush
        candidate |= snapped
        support |= snapped

    return seed, candidate, support


def _stroke_points_array(stroke):
    points = getattr(stroke, "points", None)
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2]


def _stroke_brush_mask(shape, points, size):
    mask = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, int(round(float(size))))
    pts = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    if pts.shape[0] == 1:
        x, y = int(pts[0, 0, 0]), int(pts[0, 0, 1])
        cv2.circle(mask, (x, y), max(1, thickness // 2), 1, -1, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(mask, [pts], False, 1, thickness, cv2.LINE_AA)
    return mask > 0


def _stroke_center_mask(shape, points, size):
    mask = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, int(round(float(size) * 0.08)))
    pts = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    if pts.shape[0] == 1:
        x, y = int(pts[0, 0, 0]), int(pts[0, 0, 1])
        cv2.circle(mask, (x, y), max(1, thickness), 1, -1, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(mask, [pts], False, 1, thickness, cv2.LINE_AA)
    return mask > 0


def _visible_polyline_segments(points, shape, margin):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return []

    h, w = int(shape[0]), int(shape[1])
    margin = float(max(0.0, margin))
    x0, y0 = -margin, -margin
    x1, y1 = float(w - 1) + margin, float(h - 1) + margin

    def inside(point):
        x, y = float(point[0]), float(point[1])
        return x0 <= x <= x1 and y0 <= y <= y1

    segments = []
    current = []
    prev = points[0]
    prev_in = inside(prev)
    if prev_in:
        current.append(prev)

    for point in points[1:]:
        point_in = inside(point)
        intersects = prev_in or point_in or _segment_intersects_box(prev, point, x0, y0, x1, y1)
        if intersects:
            if not current:
                current.append(prev)
            current.append(point)
        elif current:
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=np.float32))
            current = []

        if current and not point_in:
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=np.float32))
            current = []

        prev = point
        prev_in = point_in

    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=np.float32))
    return segments


def _segment_intersects_box(p0, p1, x0, y0, x1, y1):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    t0, t1 = 0.0, 1.0
    for p, q in (
            (-dx, float(p0[0] - x0)),
            (dx, float(x1 - p0[0])),
            (-dy, float(p0[1] - y0)),
            (dy, float(y1 - p0[1]))):
        if abs(p) < 1e-6:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return True


def _snap_single_stroke_by_geometry(edge_strength, points, size, brush, center, radius, strength):
    if points.shape[0] < 2:
        return brush

    half_width = max(1.0, float(size) * 0.5)
    resampled = _resample_polyline(points, spacing=max(1.0, min(3.0, half_width * 0.25)))
    if resampled.shape[0] < 2:
        return brush

    tangents = _polyline_tangents(resampled)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    max_inward = max(1, int(round(min(half_width, 96.0))))
    outward_floor = half_width * 0.75
    max_outward = max(0, int(round(min(max(float(radius), outward_floor), half_width * 2.5, 96.0))))

    candidates = []
    for side in (1.0, -1.0):
        snap, score = _snap_curve_for_side(
            edge_strength,
            resampled,
            normals * side,
            half_width,
            max_inward,
            max_outward,
        )
        poly_mask = _stroke_polygon_from_snap(
            snap,
            resampled - normals * side * half_width,
            resampled,
            half_width,
            brush.shape,
        )
        poly_mask |= center
        candidates.append((poly_mask, score))

    best_mask, best_score = max(candidates, key=lambda item: item[1])
    if best_score < 0.08:
        return brush
    return best_mask


def _resample_polyline(points, spacing=2.0):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return points.copy()
    out = [points[0]]
    carried = 0.0
    last = points[0].astype(np.float32)
    spacing = max(float(spacing), 0.5)
    for target in points[1:]:
        target = target.astype(np.float32)
        segment = target - last
        seg_len = float(np.linalg.norm(segment))
        if seg_len <= 1e-6:
            continue
        direction = segment / seg_len
        dist = spacing - carried
        while dist <= seg_len:
            out.append(last + direction * dist)
            dist += spacing
        carried = max(0.0, seg_len - (dist - spacing))
        last = target
    if np.linalg.norm(out[-1] - points[-1]) > 0.5:
        out.append(points[-1])
    return np.asarray(out, dtype=np.float32)


def _polyline_tangents(points):
    tangents = np.zeros_like(points, dtype=np.float32)
    if points.shape[0] == 1:
        tangents[:, 0] = 1.0
        return tangents
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return tangents / norms


def _snap_curve_for_side(edge_strength, centers, outward_normals, half_width, max_inward, max_outward):
    n = centers.shape[0]
    offsets_values = np.arange(-int(max_outward), int(max_inward) + 1, dtype=np.float32)
    states = int(offsets_values.shape[0])
    samples = np.zeros((n, states), dtype=np.float32)
    boundary = centers + outward_normals * float(half_width)
    for i in range(n):
        points = boundary[i][None, :] - outward_normals[i][None, :] * offsets_values[:, None]
        samples[i] = _sample_bilinear_many(edge_strength, points[:, 0], points[:, 1])

    distance_penalty = (np.abs(offsets_values) / max(float(max_inward + max_outward), 1.0) * 0.10).astype(np.float32)
    dp = np.full((n, states), np.inf, dtype=np.float32)
    prev = np.zeros((n, states), dtype=np.int16)
    dp[0] = -samples[0] + distance_penalty
    smooth = 0.120
    for i in range(1, n):
        transition, argmins = _min_abs_transition(dp[i - 1], smooth)
        dp[i] = transition - samples[i] + distance_penalty
        prev[i] = argmins.astype(np.int16, copy=False)

    state_ids = np.zeros(n, dtype=np.int32)
    offset_ids = np.zeros(n, dtype=np.float32)
    state_ids[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 2, -1, -1):
        state_ids[i] = int(prev[i + 1, state_ids[i + 1]])
    offset_ids[:] = offsets_values[state_ids]
    smooth_offsets = _smooth_snap_offsets(offset_ids)

    chosen = samples[np.arange(n), state_ids]
    coverage = float(np.mean(chosen > 0.20))
    score = float(np.mean(chosen) * (0.5 + coverage))
    snap = boundary - outward_normals * smooth_offsets[:, None].astype(np.float32)
    return snap.astype(np.float32, copy=False), score


def _min_abs_transition(prev_cost, smooth):
    states = prev_cost.shape[0]
    left_val = np.empty(states, dtype=np.float32)
    left_idx = np.empty(states, dtype=np.int32)
    best_val = np.inf
    best_idx = 0
    for i in range(states):
        value = prev_cost[i] - smooth * float(i)
        if value < best_val:
            best_val = value
            best_idx = i
        left_val[i] = best_val + smooth * float(i)
        left_idx[i] = best_idx

    right_val = np.empty(states, dtype=np.float32)
    right_idx = np.empty(states, dtype=np.int32)
    best_val = np.inf
    best_idx = states - 1
    for i in range(states - 1, -1, -1):
        value = prev_cost[i] + smooth * float(i)
        if value < best_val:
            best_val = value
            best_idx = i
        right_val[i] = best_val - smooth * float(i)
        right_idx[i] = best_idx

    use_left = left_val <= right_val
    values = np.where(use_left, left_val, right_val)
    argmins = np.where(use_left, left_idx, right_idx)
    return values, argmins


def _smooth_snap_offsets(offsets):
    offsets = np.asarray(offsets, dtype=np.float32)
    if offsets.size < 7:
        return offsets
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(offsets, (3, 3), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32, copy=False)


def _stroke_polygon_from_snap(snap_curve, opposite_boundary, centers, half_width, shape):
    poly = np.vstack([snap_curve, opposite_boundary[::-1]])
    poly = np.rint(poly).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    if poly.shape[0] >= 3:
        cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], 1, lineType=cv2.LINE_AA)
    cap_floor = max(1.0, float(half_width))
    for snap, opposite, center in (
            (snap_curve[0], opposite_boundary[0], centers[0]),
            (snap_curve[-1], opposite_boundary[-1], centers[-1])):
        cap_center = (np.asarray(snap, dtype=np.float32) + np.asarray(opposite, dtype=np.float32)) * 0.5
        cap_radius = max(cap_floor, float(np.linalg.norm(np.asarray(snap) - np.asarray(opposite))) * 0.5)
        cap_radius = min(cap_radius, cap_floor * 1.75)
        cv2.circle(
            mask,
            (int(round(cap_center[0])), int(round(cap_center[1]))),
            int(round(cap_radius)),
            1,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            mask,
            (int(round(center[0])), int(round(center[1]))),
            int(round(cap_floor)),
            1,
            -1,
            lineType=cv2.LINE_AA,
        )
    return mask > 0


def _sample_bilinear(image, x, y):
    h, w = image.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = float(x - x0)
    dy = float(y - y0)
    v00 = float(image[y0, x0])
    v10 = float(image[y0, x0 + 1])
    v01 = float(image[y0 + 1, x0])
    v11 = float(image[y0 + 1, x0 + 1])
    return (v00 * (1.0 - dx) + v10 * dx) * (1.0 - dy) + (v01 * (1.0 - dx) + v11 * dx) * dy


def _sample_bilinear_many(image, xs, ys):
    h, w = image.shape[:2]
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    valid = (xs >= 0) & (ys >= 0) & (xs < w - 1) & (ys < h - 1)
    out = np.zeros(xs.shape, dtype=np.float32)
    if not np.any(valid):
        return out
    xv = xs[valid]
    yv = ys[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    dx = xv - x0.astype(np.float32)
    dy = yv - y0.astype(np.float32)
    v00 = image[y0, x0]
    v10 = image[y0, x0 + 1]
    v01 = image[y0 + 1, x0]
    v11 = image[y0 + 1, x0 + 1]
    out[valid] = (v00 * (1.0 - dx) + v10 * dx) * (1.0 - dy) + (v01 * (1.0 - dx) + v11 * dx) * dy
    return out


def _draw_snap_edge_strength(guide):
    guide = _prepare_guide_image(guide, np.asarray(guide).shape[:2])
    if guide is None:
        return None
    guide = cv2.GaussianBlur(guide, (0, 0), 1.0)
    if guide.ndim == 2:
        gray = guide.astype(np.float32, copy=False)
        color_grad = np.zeros_like(gray)
    else:
        gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        lab = _guide_to_lab(guide)
        color_grad = np.zeros(gray.shape, dtype=np.float32)
        for c in range(3):
            gx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3)
            color_grad += gx * gx + gy * gy
        color_grad = _normalize_by_percentile(np.sqrt(color_grad), 98.8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    luma_grad = _normalize_by_percentile(cv2.magnitude(gx, gy), 98.8)
    strength = np.maximum(luma_grad, color_grad)
    return cv2.GaussianBlur(strength, (0, 0), 0.6).astype(np.float32, copy=False)


def _draw_edge_snap_support(guide, mask, radius, strength, seed_mask=None):
    hint = _as_mask(mask) > 0.02
    seed = _draw_sure_foreground_seed(mask, seed_mask)
    if not np.any(seed):
        return seed, None, None

    search_radius = _draw_snap_search_radius(hint, seed, radius)
    candidate = (_distance_from(hint) <= float(search_radius)) | hint | seed
    if not np.any(candidate):
        return seed, candidate, None

    support = _edge_snapped_draw_support(
        guide,
        hint,
        seed,
        candidate,
        search_radius,
        strength,
    )
    return seed, candidate, support


def _edge_snapped_draw_support(guide, hint, seed, candidate, radius, strength):
    stop = _make_edge_stop_mask(guide, _draw_barrier_strength(strength))
    if stop is None or not np.any(stop):
        return hint | seed

    half_width = _hint_half_width(hint)
    edge_radius = max(2.0, min(float(radius), half_width * 1.25 + 2.0))
    local_edge = stop & (_distance_from(hint) <= edge_radius)
    if not np.any(local_edge):
        return hint | seed

    barrier = _draw_edge_barrier(local_edge, half_width)
    free = (hint | seed) & ~barrier

    seed_touch = (cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0) & free
    if not np.any(seed_touch):
        seed_touch = hint & free
    if not np.any(seed_touch):
        return hint | seed

    selected_side = _connected_to_seed(free, seed_touch)

    # Edge-snapped draw is primarily a clipped brush, not a region grow. The
    # radius decides which nearby edges can cut the brush; it does not inflate
    # the final mask away from the traced stroke.
    support = (hint & selected_side) | seed
    support |= _restore_edge_pixels(hint, support, local_edge)
    if np.count_nonzero(support) < np.count_nonzero(seed):
        support = seed | (hint & selected_side)
    return support


def _hint_half_width(hint):
    hint = np.asarray(hint, dtype=bool)
    if not np.any(hint):
        return 1.0
    dist_inside = cv2.distanceTransform(hint.astype(np.uint8), cv2.DIST_L2, 3)
    return max(1.0, float(dist_inside.max(initial=0.0)))


def _component_half_width(component, stroke_half_width=None):
    hint_half_width = _hint_half_width(component)
    if stroke_half_width is not None and np.isfinite(stroke_half_width):
        stroke_half_width = max(1.0, float(stroke_half_width))
        # Cropped zoom views can turn a brush stroke into a very large clipped
        # component. Use the stroke width as a ceiling there, but do not shrink
        # normal full-view strokes.
        return min(hint_half_width, stroke_half_width * 1.35 + 3.0)
    return hint_half_width


def _draw_strokes_half_width(draw_strokes):
    if not draw_strokes:
        return None
    sizes = []
    for stroke in draw_strokes:
        try:
            size = float(getattr(stroke, "size", 0.0))
        except Exception:
            size = 0.0
        if size > 0.0:
            sizes.append(size)
    if not sizes:
        return None
    return max(1.0, float(np.median(sizes)) * 0.5)


def _draw_edge_barrier(local_edge, half_width):
    radius = int(max(2, min(14, round(float(half_width) * 0.70))))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    barrier = cv2.morphologyEx(local_edge.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    barrier = cv2.dilate(barrier, kernel, iterations=1)
    return barrier > 0


def _draw_sure_foreground_seed(mask, seed_mask=None):
    mask_f = _as_mask(mask)
    if seed_mask is not None:
        seed = np.asarray(seed_mask, dtype=bool)
        if seed.shape != mask_f.shape:
            seed = cv2.resize(
                seed.astype(np.uint8),
                (int(mask_f.shape[1]), int(mask_f.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if np.any(seed):
            shrunk = make_confident_seed(seed.astype(np.float32), threshold=0.5, shrink_ratio=0.70, min_shrink=1.0)
            return shrunk if np.any(shrunk) else seed

    seed = make_confident_seed(mask_f, threshold=0.05, shrink_ratio=0.70, min_shrink=1.0)
    if np.any(seed):
        return seed
    return mask_f >= max(0.35, float(mask_f.max(initial=0.0)) * 0.6)


def _draw_snap_search_radius(hint, seed, radius):
    hint = np.asarray(hint, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    h, w = hint.shape[:2]
    base = float(max(1.0, round(float(radius))))
    if not np.any(hint):
        return int(base)

    dist_inside = cv2.distanceTransform(hint.astype(np.uint8), cv2.DIST_L2, 3)
    half_width = float(dist_inside.max(initial=0.0))
    min_dim = float(max(1, min(h, w)))
    floor = max(4.0, half_width * 1.8, min_dim * 0.055)
    return int(round(max(base, floor)))


def _draw_barrier_strength(strength):
    return float(max(60.0, np.clip(strength, 0, 100)))


def _restrict_candidate_by_edges(guide, seed, candidate, strength):
    stop = _make_edge_stop_mask(guide, strength)
    if stop is None or not np.any(stop):
        return candidate.astype(bool, copy=False)
    free = (candidate & ~stop) | seed
    selected = _connected_to_seed(free, seed)
    selected |= _restore_edge_pixels(candidate, selected, stop)
    selected |= seed
    return selected & candidate


def _make_quick_select_candidate(guide, seed, anchor, radius, strength):
    draw_like = np.count_nonzero(anchor) > np.count_nonzero(seed) * 1.2
    if draw_like:
        radius = _draw_like_search_radius(anchor, radius)
    dist = _distance_from(anchor)
    local = dist <= float(radius)
    affinity = _make_color_affinity(guide, seed, radius, strength, sample_mask=anchor if draw_like else None)
    if affinity is None:
        candidate = local | anchor
    else:
        candidate = (local | anchor) & affinity
        if draw_like and np.count_nonzero(candidate) < np.count_nonzero(anchor) * 1.35:
            loose_affinity = _make_color_affinity(
                guide,
                seed,
                radius,
                min(float(strength), 35.0),
                sample_mask=anchor,
                tolerance_gain=1.6,
            )
            if loose_affinity is not None:
                candidate = (local | anchor) & loose_affinity
        if draw_like and np.count_nonzero(candidate) < np.count_nonzero(anchor) * 1.20:
            candidate = local | anchor
    candidate |= seed
    edge_component, stop = _edge_connected_component_and_stop(guide, seed, strength)
    if edge_component is not None:
        selected = (candidate & edge_component) | seed
        selected = _connected_to_seed(selected, seed)
        selected |= _restore_edge_pixels(candidate, selected, stop)
        return selected
    return _connected_to_seed(candidate, seed)


def _draw_like_search_radius(anchor, radius):
    anchor = np.asarray(anchor, dtype=bool)
    if not np.any(anchor):
        return radius
    dist_inside = cv2.distanceTransform(anchor.astype(np.uint8), cv2.DIST_L2, 3)
    half_width = float(dist_inside.max(initial=0.0))
    return int(max(float(radius), 4.0, min(14.0, half_width * 1.4)))


def _fallback_support(mask, seed, candidate):
    if candidate is not None and np.any(candidate):
        return candidate.astype(bool, copy=False)
    if seed is not None and np.any(seed):
        return seed.astype(bool, copy=False)
    return _as_mask(mask) > 0.5


def _compose_refined_mask(
        mask,
        support,
        fill_grown_region,
        support_softness=0.0,
        guide=None,
        natural_edge=False,
        edge_lock=0.0):
    support = np.asarray(support, dtype=bool)
    if not np.any(support):
        return np.zeros_like(mask, dtype=np.float32)
    if fill_grown_region:
        result = support.astype(np.float32)
    else:
        result = np.where(support, mask, 0.0).astype(np.float32, copy=False)
    if natural_edge:
        result = _apply_natural_edge_matte(guide, result, support, edge_lock=edge_lock)
    support_softness = float(max(0.0, support_softness))
    if support_softness > 0.01:
        result = cv2.GaussianBlur(result, (0, 0), support_softness)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def _apply_natural_edge_matte(guide, mask, support, edge_lock=0.0):
    guide = _prepare_guide_image(guide, np.asarray(mask).shape[:2])
    if guide is None:
        return np.asarray(mask, dtype=np.float32)

    support = np.asarray(support, dtype=bool)
    if int(np.count_nonzero(support)) < 64:
        return np.asarray(mask, dtype=np.float32)

    if guide.ndim == 2:
        guide_rgb = np.repeat(guide[..., None], 3, axis=2)
    else:
        guide_rgb = guide[..., :3]
    guide_rgb = guide_rgb.astype(np.float32, copy=False)

    inside_dist = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
    outside_dist = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 3)
    edge_width = 2.25
    sample_width = 9.0
    edge_band = support & (inside_dist <= edge_width)
    image_edge = _draw_snap_edge_strength(guide_rgb)
    if image_edge is not None:
        edge_near = image_edge >= 0.25
        if np.any(edge_near):
            edge_near = cv2.dilate(
                edge_near.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
                iterations=2,
            ) > 0
            edge_band &= edge_near
    if int(np.count_nonzero(edge_band)) < 16:
        return np.asarray(mask, dtype=np.float32)

    fg_samples = support & (inside_dist >= max(2.0, edge_width * 0.75))
    bg_samples = (~support) & (outside_dist <= sample_width)
    if int(np.count_nonzero(fg_samples)) < 16 or int(np.count_nonzero(bg_samples)) < 16:
        return np.asarray(mask, dtype=np.float32)

    fg_global = np.median(guide_rgb[fg_samples].reshape(-1, guide_rgb.shape[-1]), axis=0)
    bg_global = np.median(guide_rgb[bg_samples].reshape(-1, guide_rgb.shape[-1]), axis=0)
    if float(np.linalg.norm(fg_global - bg_global)) < 0.018:
        return np.asarray(mask, dtype=np.float32)
    luma = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dim_fg_global = float(np.dot(fg_global[:3], luma)) < float(np.dot(bg_global[:3], luma)) + 0.015

    fg_local, fg_weight = _local_sample_mean(guide_rgb, fg_samples, sigma=5.0, fallback=fg_global)
    bg_local, bg_weight = _local_sample_mean(guide_rgb, bg_samples, sigma=5.0, fallback=bg_global)

    direction = fg_local - bg_local
    denom = np.sum(direction * direction, axis=2)
    color_alpha = np.sum((guide_rgb - bg_local) * direction, axis=2) / np.maximum(denom, 1e-5)
    color_alpha = np.clip(color_alpha, 0.0, 1.0)
    color_alpha = cv2.GaussianBlur(color_alpha.astype(np.float32), (0, 0), 0.45)

    contrast = np.sqrt(np.maximum(denom, 0.0))
    sample_confidence = np.minimum(fg_weight, bg_weight)
    color_weight = np.clip((contrast - 0.025) / 0.13, 0.0, 1.0)
    color_weight *= np.clip(sample_confidence / 0.020, 0.0, 1.0)

    t = np.clip((inside_dist - 0.35) / max(edge_width, 1e-3), 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)
    edge_floor = 0.52
    lower_floor = 0.34
    geometric_alpha = edge_floor + (1.0 - edge_floor) * smooth_t
    color_weight *= 0.82
    alpha = geometric_alpha * (1.0 - color_weight) + color_alpha * color_weight
    alpha = np.maximum(alpha, lower_floor + (1.0 - lower_floor) * smooth_t)
    if dim_fg_global:
        dim_outer_cap = 0.43 + 0.57 * np.clip((inside_dist - 0.95) / 1.20, 0.0, 1.0)
        alpha = np.where(edge_band, np.minimum(alpha, dim_outer_cap), alpha)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)

    out = np.asarray(mask, dtype=np.float32).copy()
    out[edge_band] = np.minimum(out[edge_band], alpha[edge_band])
    return out


def _local_sample_mean(image, sample_mask, sigma, fallback):
    sample = np.asarray(sample_mask, dtype=np.float32)
    sigma = float(max(0.1, sigma))
    weight = cv2.GaussianBlur(sample, (0, 0), sigma)
    weighted = cv2.GaussianBlur(
        np.asarray(image, dtype=np.float32) * sample[..., None],
        (0, 0),
        sigma,
    )
    out = weighted / np.maximum(weight[..., None], 1e-5)
    fallback = np.asarray(fallback, dtype=np.float32)
    out = np.where(weight[..., None] > 1e-5, out, fallback.reshape((1, 1, -1)))
    return out.astype(np.float32, copy=False), weight.astype(np.float32, copy=False)


def _debug_matte_drop(support, refined):
    if support is None or refined is None:
        return None
    support_f = _as_mask(support).astype(np.float32, copy=False)
    refined_f = _as_mask(refined).astype(np.float32, copy=False)
    if support_f.size == 0 or refined_f.size == 0:
        return None
    if refined_f.shape[:2] != support_f.shape[:2]:
        refined_f = cv2.resize(
            refined_f,
            (int(support_f.shape[1]), int(support_f.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.clip(support_f - refined_f, 0.0, 1.0).astype(np.float32, copy=False)


def _edge_connected_component_and_stop(guide, seed, strength):
    stop = _make_edge_stop_mask(guide, strength)
    if stop is None:
        return None, None
    if not np.any(stop):
        return np.ones(stop.shape, dtype=bool), stop
    # The user's current mask is a search hint, not a permission to tunnel
    # through edges. If anchor is added here, an edge already inside a draw
    # stroke becomes passable and inward snapping cannot happen.
    free = (~stop) | seed
    return _connected_to_seed(free, seed), stop


def _restore_edge_pixels(candidate, selected, stop):
    if stop is None or not np.any(stop) or not np.any(selected):
        return np.zeros_like(candidate, dtype=bool)
    near_selected = cv2.dilate(
        selected.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=2,
    ) > 0
    return candidate & stop & near_selected


def _make_edge_stop_mask(guide, strength):
    if guide is None:
        return None
    guide = _prepare_guide_image(guide, guide.shape[:2])
    if guide is None:
        return None
    strength = float(np.clip(strength, 0, 100))
    if strength <= 0.5:
        return np.zeros(guide.shape[:2], dtype=bool)

    # Work on a structure-scale guide. Raw Sobel/Canny on dark water, foliage,
    # skin, etc. promotes texture into walls and traps circular masks around
    # the tiny center seed. Higher Edge Lock lowers the blur a little, but still
    # avoids pixel-scale edges.
    lock = strength / 100.0
    guide_for_edges = cv2.GaussianBlur(guide, (0, 0), 2.4 - 0.9 * lock)
    if guide.ndim == 2:
        gray = guide_for_edges.astype(np.float32, copy=False)
        color_grad = np.zeros_like(gray)
    else:
        gray = cv2.cvtColor(guide_for_edges, cv2.COLOR_RGB2GRAY)
        lab = _guide_to_lab(guide_for_edges)
        color_grad = np.zeros(gray.shape, dtype=np.float32)
        for c in range(3):
            gx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3)
            color_grad += gx * gx + gy * gy
        color_grad = _normalize_by_percentile(
            np.sqrt(color_grad),
            _edge_normalize_percentile(strength),
        )

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    luma_grad = _normalize_by_percentile(
        cv2.magnitude(gx, gy),
        _edge_normalize_percentile(strength),
    )

    barrier = np.maximum(luma_grad, color_grad)
    barrier = cv2.GaussianBlur(barrier, (0, 0), 0.6)
    threshold = 0.92 - 0.0027 * strength
    threshold = float(np.clip(threshold, 0.62, 0.92))
    max_coverage = float(np.clip(0.015 + strength * 0.00055, 0.015, 0.07))
    if np.count_nonzero(barrier > 0.0) == 0:
        return np.zeros(barrier.shape, dtype=bool)

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(8):
        stop = barrier >= threshold
        stop = cv2.morphologyEx(stop.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        if float(np.mean(stop)) <= max_coverage or threshold >= 0.98:
            break
        threshold = min(0.98, threshold + 0.06)
    return stop


def _edge_normalize_percentile(strength):
    # Low strength keeps only very prominent boundaries. High strength lowers
    # the percentile so weaker edges start acting as barriers.
    return float(np.clip(99.6 - 0.010 * float(strength), 98.6, 99.6))


def _debug_dump_refine_state(
        guide,
        mask,
        refined,
        guide_point,
        seed,
        candidate,
        support,
        strength,
        seed_from_guide,
        debug_label,
        stage,
        extra_planes=None):
    prefix = _debug_dump_prefix(debug_label, stage)
    if prefix is None:
        return
    try:
        hint = _make_foreground_hint(mask, guide_point, seed_from_guide=seed_from_guide)
        stop = _make_edge_stop_mask(guide, strength)
        extra_map = {title: values for title, values in extra_planes or []}
        edge_comp_source = extra_map.get("target_edge", extra_map.get("image_edge"))
        planes = [
            ("guide", _debug_rgb_panel(guide)),
            ("mask", _debug_gray_panel(mask)),
            ("hint", _debug_gray_panel(hint)),
            ("seed", _debug_gray_panel(seed)),
            ("edge_stop", _debug_gray_panel(stop)),
            ("candidate", _debug_gray_panel(candidate)),
            ("support", _debug_gray_panel(support)),
            ("refined", _debug_gray_panel(refined)),
            ("mask_change", _debug_mask_change_panel(mask, refined)),
            ("edge_comp", _debug_edge_composite_panel(
                guide,
                seed,
                stop,
                refined,
                image_edge=edge_comp_source,
                accepted=extra_map.get("accepted"),
            )),
        ]
        matte_drop = _debug_matte_drop(support, refined)
        if matte_drop is not None and np.any(matte_drop > 0.001):
            planes.append(("matte_drop", _debug_gray_panel(matte_drop)))
        if extra_planes:
            for title, values in extra_planes:
                planes.append((title, _debug_gray_panel(values)))
        planes.append(("overlay", _debug_overlay_panel(guide, seed, stop, candidate, support)))
        mosaic = _debug_mosaic(planes)
        out_path = f"{prefix}_mosaic.png"
        cv2.imwrite(out_path, mosaic)
        logging.warning(
            "[EDGE_REFINE_DEBUG] wrote %s mask_sum=%.3f refined_sum=%.3f seed=%d candidate=%d support=%d stop=%d",
            out_path,
            float(np.sum(_as_mask(mask))),
            float(np.sum(_as_mask(refined))),
            int(np.count_nonzero(seed)) if seed is not None else 0,
            int(np.count_nonzero(candidate)) if candidate is not None else 0,
            int(np.count_nonzero(support)) if support is not None else 0,
            int(np.count_nonzero(stop)) if stop is not None else 0,
        )
    except Exception:
        logging.exception("[EDGE_REFINE_DEBUG] failed to write debug image")


def _debug_dump_prefix(debug_label, stage):
    global _DEBUG_DUMP_COUNTER
    if not _debug_dump_enabled():
        return None
    try:
        limit = int(os.getenv("PLATYPUS_DEBUG_EDGE_REFINE_LIMIT", "80"))
    except ValueError:
        limit = 80
    if limit >= 0 and _DEBUG_DUMP_COUNTER >= limit:
        if _DEBUG_DUMP_COUNTER == limit:
            logging.warning(
                "[EDGE_REFINE_DEBUG] skipped debug dump after reaching limit=%d; "
                "set PLATYPUS_DEBUG_EDGE_REFINE_LIMIT=-1 for unlimited dumps",
                limit,
            )
            _DEBUG_DUMP_COUNTER += 1
        return None

    dump_dir = os.getenv("PLATYPUS_DEBUG_EDGE_REFINE_DIR", "").strip()
    if not dump_dir:
        dump_dir = "/tmp/platypus_edge_refine"
    os.makedirs(dump_dir, exist_ok=True)
    _DEBUG_DUMP_COUNTER += 1
    label = _debug_safe_label(debug_label or "mask")
    stage = _debug_safe_label(stage or "refine")
    millis = int(time.time() * 1000)
    return os.path.join(dump_dir, f"{_DEBUG_DUMP_COUNTER:04d}_{label}_{stage}_{millis}")


def _debug_dump_enabled():
    flag = os.getenv("PLATYPUS_DEBUG_EDGE_REFINE", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _debug_safe_label(value):
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(value))[:80]


def _debug_rgb_panel(image):
    guide = _prepare_guide_image(image, np.asarray(image).shape[:2])
    if guide is None:
        guide = np.zeros((32, 32, 3), dtype=np.float32)
    if guide.ndim == 2:
        guide = np.repeat(guide[..., None], 3, axis=2)
    return cv2.cvtColor(_debug_to_u8(guide[..., :3]), cv2.COLOR_RGB2BGR)


def _debug_gray_panel(values):
    arr = _as_mask(values)
    if arr.size == 0:
        arr = np.zeros((32, 32), dtype=np.float32)
    gray = _debug_to_u8(arr)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _debug_overlay_panel(guide, seed, stop, candidate, support):
    panel = _debug_rgb_panel(guide)
    h, w = panel.shape[:2]
    for values, color, alpha in (
            (stop, (0, 0, 255), 0.70),
            (candidate, (255, 160, 0), 0.35),
            (support, (0, 220, 0), 0.45),
            (seed, (0, 255, 255), 0.85)):
        mask = _debug_bool_mask(values, (h, w))
        if not np.any(mask):
            continue
        color_arr = np.array(color, dtype=np.float32)
        panel_f = panel.astype(np.float32)
        panel_f[mask] = panel_f[mask] * (1.0 - alpha) + color_arr * alpha
        panel = np.clip(panel_f, 0, 255).astype(np.uint8)
    return panel


def _debug_mask_change_panel(mask, refined):
    before_f = _as_mask(mask)
    after_f = _as_mask(refined)
    if before_f.size == 0 and after_f.size == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    if before_f.size == 0:
        before_f = np.zeros(after_f.shape[:2], dtype=np.float32)
    if after_f.size == 0:
        after_f = np.zeros(before_f.shape[:2], dtype=np.float32)
    if after_f.shape[:2] != before_f.shape[:2]:
        after_f = cv2.resize(
            after_f.astype(np.float32),
            (int(before_f.shape[1]), int(before_f.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )

    before = before_f > 0.02
    after = after_f > 0.02
    kept = before & after
    added = ~before & after
    removed = before & ~after

    panel = np.zeros((*before.shape, 3), dtype=np.uint8)
    kept_level = np.clip(after_f * 205.0 + 50.0, 0, 255).astype(np.uint8)
    panel[kept] = np.repeat(kept_level[kept, None], 3, axis=1)
    panel[added] = (0, 220, 0)
    panel[removed] = (0, 0, 255)
    return panel


def _debug_edge_composite_panel(guide, seed, stop, refined, image_edge=None, accepted=None):
    panel = _debug_rgb_panel(guide)
    h, w = panel.shape[:2]
    final_mask = _debug_bool_mask(refined, (h, w))
    if image_edge is not None:
        edge_values = _debug_float_mask(image_edge, (h, w))
        edge_line = _debug_edge_line(edge_values)
    else:
        edge_line = _debug_bool_mask(stop, (h, w))

    if np.any(final_mask):
        visible_edge_scope = cv2.dilate(
            final_mask.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=3,
        ) > 0
        edge_line = edge_line & visible_edge_scope

    if np.any(edge_line):
        edge_band = cv2.dilate(
            edge_line.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=3,
        ) > 0
    else:
        edge_band = np.zeros((h, w), dtype=bool)

    accepted_mask = _debug_bool_mask(accepted, (h, w)) if accepted is not None else final_mask
    edge_mask_part = final_mask & edge_band
    if accepted is not None:
        edge_mask_part |= accepted_mask & edge_band

    panel = _debug_alpha_overlay(panel, edge_mask_part, (0, 220, 0), 0.58)
    panel = _debug_alpha_overlay(panel, edge_line, (0, 0, 255), 0.85)
    panel = _debug_alpha_overlay(panel, seed, (0, 255, 255), 0.75)
    return panel


def _debug_edge_line(edge_values):
    edge_values = np.asarray(edge_values, dtype=np.float32)
    if edge_values.size == 0:
        return np.zeros(edge_values.shape[:2], dtype=bool)
    nonzero = edge_values[edge_values > 0.0]
    if nonzero.size == 0:
        return np.zeros(edge_values.shape[:2], dtype=bool)
    threshold = max(0.30, float(np.percentile(nonzero, 88.0)))
    return edge_values >= threshold


def _debug_alpha_overlay(panel, values, color, alpha):
    h, w = panel.shape[:2]
    mask = _debug_bool_mask(values, (h, w))
    if not np.any(mask):
        return panel
    panel_f = panel.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    panel_f[mask] = panel_f[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(panel_f, 0, 255).astype(np.uint8)


def _debug_bool_mask(values, shape):
    if values is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(values)
    if arr.shape[:2] != tuple(shape):
        arr = cv2.resize(
            arr.astype(np.float32),
            (int(shape[1]), int(shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
    return arr.astype(bool)


def _debug_float_mask(values, shape):
    if values is None:
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[:2] != tuple(shape):
        arr = cv2.resize(
            arr,
            (int(shape[1]), int(shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _debug_to_u8(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _debug_mosaic(planes):
    panels = []
    for title, panel in planes:
        panel = _debug_resize_panel(panel)
        cv2.putText(
            panel,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)

    cell_h = max(p.shape[0] for p in panels)
    cell_w = max(p.shape[1] for p in panels)
    rows = []
    for row_start in range(0, len(panels), 3):
        row = []
        for panel in panels[row_start:row_start + 3]:
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cell[:panel.shape[0], :panel.shape[1]] = panel
            row.append(cell)
        while len(row) < 3:
            row.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _debug_resize_panel(panel, max_side=360):
    h, w = panel.shape[:2]
    if max(h, w) <= max_side:
        return panel.copy()
    scale = float(max_side) / float(max(h, w))
    return cv2.resize(
        panel,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _make_color_affinity(guide, seed, radius, strength, sample_mask=None, tolerance_gain=1.0):
    if guide is None or not np.any(seed):
        return None

    lab = _guide_to_lab(guide)
    if sample_mask is not None and np.any(sample_mask):
        sample_seed = np.asarray(sample_mask, dtype=bool)
    else:
        sample_seed = _expand_sparse_seed_for_color_sample(seed, radius)
    centers, tolerances = _seed_color_model(lab, sample_seed, strength)
    if centers is None:
        return None
    tolerances = tolerances * float(max(tolerance_gain, 0.01))

    flat = lab.reshape(-1, lab.shape[-1]).astype(np.float32, copy=False)
    best = np.full(flat.shape[0], np.inf, dtype=np.float32)
    for center, tolerance in zip(centers, tolerances):
        diff = flat - center.astype(np.float32, copy=False)
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        best = np.minimum(best, dist / max(float(tolerance), 1e-6))
    return best.reshape(lab.shape[:2]) <= 1.0


def _expand_sparse_seed_for_color_sample(seed, radius):
    if int(np.count_nonzero(seed)) >= 32:
        return seed
    iterations = int(max(1, min(6, radius // 8)))
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(seed.astype(np.uint8), kernel, iterations=iterations) > 0


def _seed_color_model(lab, seed, strength):
    samples = lab[seed].reshape(-1, lab.shape[-1]).astype(np.float32, copy=False)
    if samples.size == 0:
        return None, None
    if samples.shape[0] > 4096:
        stride = int(np.ceil(samples.shape[0] / 4096.0))
        samples = samples[::stride][:4096]

    mean = samples.mean(axis=0)
    global_dist = np.sqrt(np.sum((samples - mean) ** 2, axis=1))
    global_spread = float(np.percentile(global_dist, 85.0)) if global_dist.size else 0.0
    if samples.shape[0] < 48 or global_spread < 10.0:
        centers = mean.reshape(1, -1)
    else:
        k = 3 if samples.shape[0] >= 96 and global_spread >= 24.0 else 2
        centers = _farthest_color_centers(samples, k)

    for _ in range(2):
        dist = _color_distance_to_centers(samples, centers)
        labels = np.argmin(dist, axis=1)
        for i in range(centers.shape[0]):
            part = samples[labels == i]
            if part.size:
                centers[i] = part.mean(axis=0)

    dist = _color_distance_to_centers(samples, centers)
    labels = np.argmin(dist, axis=1)
    base_tol = 18.0 + (100.0 - float(strength)) * 0.20
    tolerances = []
    for i in range(centers.shape[0]):
        part_dist = dist[labels == i, i]
        spread = float(np.percentile(part_dist, 88.0)) if part_dist.size else 0.0
        tolerances.append(np.clip(base_tol + spread * 1.25, 14.0, 52.0))
    return centers.astype(np.float32, copy=False), np.asarray(tolerances, dtype=np.float32)


def _farthest_color_centers(samples, k):
    mean = samples.mean(axis=0)
    centers = [samples[int(np.argmax(np.sum((samples - mean) ** 2, axis=1)))]]
    while len(centers) < k:
        center_arr = np.asarray(centers, dtype=np.float32)
        dist = _color_distance_to_centers(samples, center_arr)
        nearest = np.min(dist, axis=1)
        centers.append(samples[int(np.argmax(nearest))])
    return np.asarray(centers, dtype=np.float32)


def _color_distance_to_centers(samples, centers):
    diff = samples[:, None, :] - centers[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _connected_to_seed(candidate, seed):
    n_labels, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
    if n_labels <= 1:
        return candidate
    selected = np.unique(labels[seed])
    selected = selected[selected > 0]
    if selected.size == 0:
        return seed.astype(bool, copy=True)
    return np.isin(labels, selected)


def _make_foreground_seed(mask, guide_point, radius, seed_from_guide=False, seed_mask=None):
    if seed_mask is not None:
        seed = np.asarray(seed_mask, dtype=bool)
        if seed.shape != mask.shape:
            seed = cv2.resize(
                seed.astype(np.uint8),
                (int(mask.shape[1]), int(mask.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if np.any(seed):
            return seed

    seed = np.zeros(mask.shape, dtype=bool)
    if seed_from_guide and guide_point is not None and _point_in_bounds(guide_point, mask.shape):
        x, y = _clip_point(guide_point, mask.shape)
        seed_radius = int(max(2, min(8, round(float(radius) * 0.08))))
        seed_u8 = seed.astype(np.uint8)
        cv2.circle(seed_u8, (x, y), seed_radius, 1, -1)
        return seed_u8.astype(bool)

    seed = make_confident_seed(mask)
    if np.any(seed):
        return seed

    max_v = float(np.nanmax(mask))
    threshold = max(0.25, min(0.75, max_v * 0.55))
    seed = mask >= threshold
    if not np.any(seed):
        seed = mask > 0.02
    return seed.astype(bool, copy=False)


def _make_foreground_hint(mask, guide_point, seed_from_guide=False):
    mask_f = _as_mask(mask)
    if seed_from_guide and guide_point is not None and _point_in_bounds(guide_point, mask_f.shape):
        x, y = _clip_point(guide_point, mask_f.shape)
        source = mask_f if mask_f[y, x] >= 0.5 else 1.0 - mask_f
    else:
        source = mask_f
    return source > 0.02


def _expansion_radius(radius, strength):
    # Radius controls how far Quick Select may look. Strength controls edge/color
    # sensitivity, not reach; tying the two together made the slider feel muddy.
    return int(max(1, round(float(radius))))


def _grabcut_iterations(strength):
    return int(np.clip(1 + round(float(strength) / 35.0), 1, 4))


def _has_grabcut_samples(gc_mask):
    fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    bg = (gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)
    return bool(np.any(fg) and np.any(bg))


def _distance_from(mask):
    return cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)


def _guide_to_grabcut_image(guide):
    guide = np.asarray(guide)
    if guide.ndim == 2:
        guide = np.repeat(guide[..., None], 3, axis=2)
    elif guide.shape[2] > 3:
        guide = guide[..., :3]
    if guide.dtype != np.uint8:
        guide = np.clip(guide, 0.0, 1.0)
        guide = (guide * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(guide)


def _guide_to_lab(guide):
    if guide.ndim == 2:
        return np.repeat(guide[..., None], 3, axis=2) * np.array([100.0, 0.0, 0.0], dtype=np.float32)
    return cv2.cvtColor(guide.astype(np.float32, copy=False), cv2.COLOR_RGB2LAB)


def _normalize_by_percentile(values, percentile):
    values = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    ref = float(np.percentile(values, percentile))
    if ref <= 1e-6:
        ref = float(values.max(initial=0.0))
    if ref <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / ref, 0.0, 1.0).astype(np.float32, copy=False)


def _as_mask(mask):
    if mask is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _prepare_guide_image(image, shape):
    if image is None:
        return None
    guide = np.asarray(image)
    if guide.size == 0:
        return None
    if guide.ndim == 3 and guide.shape[2] > 3:
        guide = guide[..., :3]
    if guide.shape[:2] != tuple(shape):
        guide = cv2.resize(guide, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
    guide = guide.astype(np.float32, copy=False)
    if guide.max(initial=0) > 2.0:
        guide = guide / 255.0
    return np.clip(guide, 0.0, 1.0)


def _clip_point(point, shape):
    h, w = int(shape[0]), int(shape[1])
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    return min(max(x, 0), max(w - 1, 0)), min(max(y, 0), max(h - 1, 0))


def _point_in_bounds(point, shape):
    h, w = int(shape[0]), int(shape[1])
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    return 0 <= x < w and 0 <= y < h
