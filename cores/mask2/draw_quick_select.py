"""
Draw Quick Select / edge snap via a band-limited 2D min-cut.

The drawn mask (FreeDraw / Polyline) is treated as a *boundary decision*
problem, not a region-estimation one: we build a thin search ``band`` around
the final mask boundary and solve a single binary foreground/background
min-cut inside it. Strong image edges become cheap to cut (the boundary snaps
to them); where there is no edge the cut falls back near the original mask
boundary, so the mask does not inflate.

Design notes live in ``docs/draw-quick-select-edge-refine-notes.md``.

Solver: ``scipy.sparse.csgraph.maximum_flow`` (Dinic). No compiled dependency
is added; the graph is built only over band pixels, per connected component,
inside a padded ROI, so it stays small and fast.
"""
from __future__ import annotations

import logging
import os
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np


def _envf(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow
except Exception:  # pragma: no cover - scipy is a hard dependency in practice
    csr_matrix = None
    maximum_flow = None
    breadth_first_order = None

try:
    from skimage.morphology import skeletonize as _skeletonize
except Exception:  # pragma: no cover - scikit-image is a hard dependency
    _skeletonize = None

from cores.mask2 import edge_refine as _er


# --- tuning constants --------------------------------------------------------
# Integer capacity scale for the graph. Capacities are int64.
CAP = 1024
# n-link floor: even fully smooth neighbours keep a small, uniform capacity so
# the min-cut prefers the shortest boundary (i.e. stays near the original mask
# boundary) instead of wandering. This is the doc's "weak default cut".
LAMBDA = 0.04
# Soft prior weight (pull each band pixel toward the side of the original mask
# boundary it started on). Kept well below the n-link scale so real image edges
# win, but enough to break ties and prevent inflation in smooth regions.
BETA = _envf("QS_BETA", 0.28)
# Constant floor on the prior magnitude at the original boundary. Without it the
# prior decays to 0 at the boundary and min-cut's shrinking bias eats the mask
# rim (uniform images shrink) or shortcuts concave boundaries (large radius
# bulges). The floor makes the original boundary "sticky"; image edges still win
# because snapping saves ~CAP per unit length, far more than BETA*FLOOR.
# The outside floor (anti outward-bulge/inflation) is stronger than the inside
# floor (which must stay low enough that the boundary can clip inward to an edge
# running through the brush body).
PRIOR_FLOOR_IN = _envf("QS_FLOOR_IN", 0.25)
# Outside the mask the prior is kept *light and flat* across most of the band so
# the boundary can reach outward and snap to any image edge inside `radius` (the
# edge is cheap to cut; a light flat prior lets the cut travel to it). It ramps
# hard to 1 only near the band rim, which hard-caps reach at `radius` and pins
# featureless regions so they do not bulge. Lower QS_FLOOR_OUT => longer reach.
PRIOR_FLOOR_OUT = _envf("QS_FLOOR_OUT", 0.12)
# Fraction of the band (from the boundary) over which the outward prior stays at
# the light floor; beyond it the prior ramps to 1 at the rim.
REACH_FRAC = _envf("QS_REACH_FRAC", 0.85)
# Weight of the colour data term (Boykov-Jolly style). A band pixel whose colour
# matches the foreground seed is pulled FG; one matching the background shell is
# pulled BG. This holds the brush body where it matches the drawn region and
# lets the boundary clip a same-geometry spill that is a *different* colour
# (e.g. a brush that overran a cloud edge into dark sky), which a purely
# geometric cut cannot do without a shrinking bias.
COLOR_W = _envf("QS_COLOR_W", 1.7)
BRIGHT_COLOR_W = _envf("QS_BRIGHT_COLOR_W", 0.6)
BRIGHT_COLOR_W_BASE = _envf("QS_BRIGHT_COLOR_W_BASE", 1.1)
BRIGHT_COLOR_W_START = _envf("QS_BRIGHT_COLOR_W_START", 70.0)
# Outside-the-mask FG-pull weight. 0 = conservative boundary snap (no explosion,
# rejects same-colour texture). >0 = grow through a same-colour region toward the
# surrounding edges ("fill the sky"), at the cost of grabbing same-colour texture
# and inflating at large radius. Off by default; QS_COLOR_W_OUT enables it.
COLOR_W_OUT = _envf("QS_COLOR_W_OUT", 0.7)
# Use the brush half-width as the base search radius (Photoshop-like "brush is
# the search area"); the UI radius then offsets it. This is the Draw Quick
# Select default. QS_BRUSH_AS_RADIUS=0 keeps the old experimental behaviour.
BRUSH_AS_RADIUS = bool(_envf("QS_BRUSH_AS_RADIUS", 1.0))
# Strong image edges inside the brush split the drawn component into seed-side
# and opposite-side regions. The opposite side gets a BG prior so a fat brush
# crossing a snow/sky edge snaps to the centerline side instead of preserving
# both sides just because they were inside the painted disk.
SIDE_EDGE_THRESH = _envf("QS_SIDE_EDGE_THRESH", 0.70)
SIDE_EDGE_LOOSE_THRESH = _envf("QS_SIDE_EDGE_LOOSE_THRESH", 0.25)
SIDE_EDGE_RELAX_START = _envf("QS_SIDE_EDGE_RELAX_START", 70.0)
SIDE_EDGE_SOFT_WINDOW = _envf("QS_SIDE_EDGE_SOFT_WINDOW", 0.10)
SIDE_EDGE_MIN_COMPONENT_AREA = _envf("QS_SIDE_EDGE_MIN_COMPONENT_AREA", 512.0)
SIDE_EDGE_MIN_COMPONENT_FRAC = _envf("QS_SIDE_EDGE_MIN_COMPONENT_FRAC", 0.01)
SIDE_DILATE = int(max(0, round(_envf("QS_SIDE_DILATE", 0.0))))
# At high EdgeLock, raw image edges inside snowy/leafy texture can be stronger
# than the actual object/cloud boundary. Blend in a colour-context gate so edges
# that do not separate FG-like and BG-like colours stop dominating the cut.
EDGE_CONTEXT_START = _envf("QS_EDGE_CONTEXT_START", 70.0)
EDGE_CONTEXT_FLOOR = _envf("QS_EDGE_CONTEXT_FLOOR", 0.35)
EDGE_CONTEXT_SPAN_SCALE = _envf("QS_EDGE_CONTEXT_SPAN_SCALE", 0.16)
EDGE_CONTEXT_SIGN_THRESH = _envf("QS_EDGE_CONTEXT_SIGN_THRESH", 0.035)
EDGE_CONTEXT_SIGN_BONUS = _envf("QS_EDGE_CONTEXT_SIGN_BONUS", 0.35)
EDGE_CONTEXT_MIN_SPAN = _envf("QS_EDGE_CONTEXT_MIN_SPAN", 0.035)
# Colour-separability confidence: colour contributes at weight
# clip((sep - COLOR_MIN_SEP) / COLOR_SEP_SCALE, 0, 1) where sep is the LAB
# distance between the FG-seed and BG-shell medians. Low (snow) scenes still get
# a partial colour signal instead of being hard-cut to zero.
COLOR_MIN_SEP = _envf("QS_COLOR_MIN_SEP", 1.5)
COLOR_SEP_SCALE = _envf("QS_COLOR_SEP_SCALE", 6.0)
DIRECTIONAL_BG_MAX_SEP_RATIO = _envf("QS_DIRECTIONAL_BG_MAX_SEP_RATIO", 2.0)
# After min-cut, restore a narrow selected-side edge rim when the cut lands on a
# strong image ridge. This fixes the snow/cloud side case where the graph cuts
# exactly on the ridge but the visible foreground wants the ridge pixels included.
EDGE_RESTORE_THRESH = _envf("QS_EDGE_RESTORE_THRESH", 0.40)
EDGE_RESTORE_COLOR_MIN = _envf("QS_EDGE_RESTORE_COLOR_MIN", 0.05)
BRIGHT_EDGE_RESTORE_COLOR_MIN = _envf("QS_BRIGHT_EDGE_RESTORE_COLOR_MIN", -0.10)
BRIGHT_EDGE_RESTORE_COLOR_MIN_LOCKED = _envf("QS_BRIGHT_EDGE_RESTORE_COLOR_MIN_LOCKED", 0.0)
BRIGHT_EDGE_RESTORE_COLOR_MIN_LOCK_END = _envf("QS_BRIGHT_EDGE_RESTORE_COLOR_MIN_LOCK_END", 60.0)
BRIGHT_EDGE_RESTORE_LUMA_DELTA = _envf("QS_BRIGHT_EDGE_RESTORE_LUMA_DELTA", 0.025)
EDGE_RESTORE_STEPS = int(max(1, round(_envf("QS_EDGE_RESTORE_STEPS", 4.0))))
EDGE_RESTORE_EDGE_NEAR = int(max(0, round(_envf("QS_EDGE_RESTORE_EDGE_NEAR", 2.0))))
EDGE_BIAS_NEUTRAL_LUMA_MAX = _envf("QS_EDGE_BIAS_NEUTRAL_LUMA_MAX", 0.045)
EDGE_BIAS_NEUTRAL_AUTO_PX = _envf("QS_EDGE_BIAS_NEUTRAL_AUTO_PX", 1.0)
EDGE_BIAS_MAX_STEPS = int(max(1, round(_envf("QS_EDGE_BIAS_MAX_STEPS", 8.0))))
# Edge Bias is a boundary-position/alpha control. Keep colour membership out of
# its default path so positive bias cannot turn into broad same-colour growth.
EDGE_BIAS_COLOR_RELAX = _envf("QS_EDGE_BIAS_COLOR_RELAX", 0.0)
EDGE_BIAS_COLOR_MIN = _envf("QS_EDGE_BIAS_COLOR_MIN", -0.18)
EDGE_BRIDGE_THRESH = _envf("QS_EDGE_BRIDGE_THRESH", 0.35)
EDGE_BRIDGE_MIN_NEIGHBORS = int(max(4, round(_envf("QS_EDGE_BRIDGE_MIN_NEIGHBORS", 4.0))))
OUTSIDE_KEEP_EDGE_THRESH = _envf("QS_OUTSIDE_KEEP_EDGE_THRESH", 0.60)
OUTSIDE_KEEP_EDGE_NEAR = int(max(0, round(_envf("QS_OUTSIDE_KEEP_EDGE_NEAR", 2.0))))
OUTSIDE_KEEP_EDGE_RELAX_DIST = _envf("QS_OUTSIDE_KEEP_EDGE_RELAX_DIST", 24.0)
# Shrink ratio for the hard-FG core. A thin core (skeleton-like) lets an edge
# that runs *through* the brush body clip it, while the prior floor keeps the
# body filled where there is no edge and preserves solid interiors.
CORE_SHRINK = _envf("QS_CORE_SHRINK", 0.80)
# Hard-seed capacity. Larger than any finite incident capacity at a node.
INF = CAP * 64
# radius=0 still leaves a thin band so a brush can snap to an edge crossing it.
MIN_BAND = 2.0
# Safety guard: a single component's band larger than this falls back to the
# raw hint to avoid pathological full-image strokes hanging the UI.
MAX_BAND_NODES = 700_000
# Anti-aliased brush rendering can leave many detached 1-7 px rim flecks around
# one real stroke island. With a large positive radius each fleck creates a huge
# overlapping graph; skip those only when a large sibling island exists.
MIN_SOLVE_COMPONENT_AREA = int(max(1, round(_envf("QS_MIN_SOLVE_COMPONENT_AREA", 32.0))))


class DrawSupportResult(NamedTuple):
    seed: np.ndarray            # bool HxW  hard-FG core (debug 'seed' slot)
    candidate: np.ndarray       # bool HxW  the search band (debug 'candidate')
    support: np.ndarray         # bool HxW  binary min-cut result (pre-matte)
    debug_planes: List[Tuple[str, np.ndarray]]


class _Scales(NamedTuple):
    band_half_width: float      # inward clip radius: brush half-width + UI offset
    grow_radius: float          # outward grow radius: max(UI offset, 0)
    roi_pad: int                # ROI padding around a component bbox
    stroke_half_width: float


class _SolveUnit(NamedTuple):
    component: np.ndarray       # bool HxW; one stroke-owned active island
    core: np.ndarray            # bool HxW; centerline/core for this unit only
    scales: _Scales             # resolved from this unit's stroke size


class _EdgePolicy(NamedTuple):
    # Effective EdgeLock after auto/offset resolution. Semantics: higher accepts
    # weaker/diffuse human-visible edges as valid cut locations.
    sensitivity: float
    # Threshold used to thin raw edge confidence into a 1px cut ridge.
    ridge_threshold: float
    # Width of the ridge falloff used after thinning.
    ridge_falloff_sigma: float
    # Graph edge-cost sigma; lower makes accepted ridges cheaper to cut.
    cut_sigma: float
    # Edge threshold for splitting a wide brush into seed-side / opposite-side.
    side_threshold: float
    side_relax_weight: float
    # Outside support can survive only near edges at this confidence.
    outside_keep_threshold: float
    # Post-cut selected-side rim restore threshold.
    restore_threshold: float
    # UI edge-bias offset in pixels. This must not affect edge sensitivity.
    boundary_bias_px: float


# --- public entry ------------------------------------------------------------
def compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None,
        pixel_scale=1.0,
        edge_bias=0.0) -> DrawSupportResult:
    """Compute the snapped foreground ``support`` for a drawn mask.

    Returns a :class:`DrawSupportResult`. ``support`` is a binary mask that the
    caller blends/mattes via ``edge_refine._compose_refined_mask``.
    """
    mask_f = _er._as_mask(mask)
    _maybe_dump_input(
        guide,
        mask_f,
        radius,
        strength,
        seed_mask,
        draw_strokes,
        pixel_scale,
        edge_bias=edge_bias,
    )
    hint = mask_f > 0.02
    h, w = hint.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if not np.any(hint) or maximum_flow is None:
        return DrawSupportResult(empty, empty, empty.copy(), [])

    guide = _er._prepare_guide_image(guide, (h, w))
    scales = _resolve_scales(radius, draw_strokes, hint)
    strength = float(np.clip(strength, 0.0, 100.0))
    policy = _edge_policy(strength, edge_bias=edge_bias)

    edge_strength = _er._draw_snap_edge_strength(guide)
    if edge_strength is None:
        edge_strength = np.zeros((h, w), dtype=np.float32)
    solver_edge_strength = _solver_edge_strength(edge_strength, pixel_scale)
    edge_cost_all = _edge_cost_map(solver_edge_strength, policy=policy)
    solver_edge_context_all = solver_edge_strength.copy()

    fg_stroke, bg_stroke, has_strokes = _er._draw_random_walker_stroke_seeds(
        hint.shape, draw_strokes, hint)

    hard_fg_core = _seed_core(mask_f, fg_stroke, hint)
    erase_bg = bg_stroke & ~hard_fg_core

    # Accumulators (global, for debug + final result).
    support_all = np.zeros((h, w), dtype=bool)
    band_all = np.zeros((h, w), dtype=bool)
    fg_seed_all = np.zeros((h, w), dtype=bool)
    bg_seed_all = np.zeros((h, w), dtype=bool)
    prior_all = np.zeros((h, w), dtype=np.float32)
    cut_all = np.zeros((h, w), dtype=bool)
    color_all = np.zeros((h, w), dtype=np.float32)
    restore_color_min_all = np.full((h, w), float(EDGE_RESTORE_COLOR_MIN), dtype=np.float32)
    restore_steps_all = np.zeros((h, w), dtype=np.float32)
    edge_bias_auto_all = np.zeros((h, w), dtype=np.float32)
    edge_bias_effective_all = np.full((h, w), float(edge_bias), dtype=np.float32)
    restore_candidate_all = np.zeros((h, w), dtype=bool)
    neutral_edge_bias_candidate_all = np.zeros((h, w), dtype=bool)
    edge_restore_all = np.zeros((h, w), dtype=bool)
    neutral_edge_bias_all = np.zeros((h, w), dtype=bool)
    edge_bridge_all = np.zeros((h, w), dtype=bool)

    solve_units = _draw_solve_units(mask_f, hint, hard_fg_core, draw_strokes, radius, has_strokes)
    total_band = 0
    total_flow = 0
    for unit in solve_units:
        component = unit.component
        component_scales = unit.scales
        y0, y1, x0, x1 = _er._expanded_bbox(component, component_scales.roi_pad)
        if y1 <= y0 or x1 <= x0:
            continue
        sl = np.s_[y0:y1, x0:x1]
        comp = component[sl]
        core_roi = unit.core[sl]
        # Colour model is LOCAL to this stroke: FG = this component's colour, BG =
        # its own surroundings. A global model blends differently-coloured strokes
        # (e.g. one on snow, one in sky) and makes some of them misbehave.
        color_roi = _color_score(
            guide[sl], comp, core_roi & comp, hint[sl], component_scales.band_half_width,
            directional_bg=has_strokes)
        color_weight = _color_weight_for_unit(
            guide[sl], comp, core_roi & comp, hint[sl], component_scales.band_half_width,
            directional_bg=has_strokes, strength=strength)
        selected_luma_delta = _selected_luma_delta(
            guide[sl], comp, core_roi & comp, hint[sl], component_scales.band_half_width,
            directional_bg=has_strokes)
        auto_edge_bias = _auto_edge_bias_for_unit(selected_luma_delta, component_scales)
        effective_edge_bias = float(auto_edge_bias) + float(edge_bias)
        unit_edge_strength = _contextual_edge_strength(
            solver_edge_strength[sl], color_roi, strength)
        unit_policy = _edge_policy(strength, edge_bias=edge_bias)
        g_roi = _edge_cost_map(unit_edge_strength, policy=unit_policy)
        restore_color_min = _edge_restore_color_min_for_unit(
            guide[sl], comp, core_roi & comp, hint[sl], component_scales.band_half_width,
            directional_bg=has_strokes,
            strength=strength)
        out = _solve_component(
            comp,
            hint[sl],
            g_roi,
            core_roi,
            erase_bg[sl],
            color_roi,
            component_scales,
            unit_edge_strength,
            color_weight=color_weight,
            side_edge_thresh=unit_policy.side_threshold,
            side_relax_weight=unit_policy.side_relax_weight,
        )
        color_all[sl] = np.where(out.band | comp, color_roi, color_all[sl])
        edge_cost_all[sl] = np.where(out.band | comp, g_roi, edge_cost_all[sl])
        solver_edge_context_all[sl] = np.where(
            out.band | comp, unit_edge_strength, solver_edge_context_all[sl])
        # Rim restore is for pixels the user actually brushed over. Positive UI
        # radius may expose a large outside search band; postprocess must not
        # invent brush-shaped growth there when no image edge was selected.
        restore_candidate_roi = out.band & comp
        restore_candidate_all[sl] |= restore_candidate_roi
        if _is_neutral_edge_bias_unit(selected_luma_delta):
            neutral_edge_bias_candidate_all[sl] |= restore_candidate_roi
        restore_color_min_all[sl] = np.where(
            restore_candidate_roi,
            np.minimum(restore_color_min_all[sl], restore_color_min),
            restore_color_min_all[sl],
        )
        restore_steps = _edge_restore_steps_for_luma(
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

    support_all = _postprocess_support(support_all, hint, hard_fg_core, erase_bg)
    support_all, edge_restore_all = _restore_selected_edge_rim(
        support_all,
        restore_candidate_all,
        solver_edge_context_all,
        color_all,
        hard_fg_core,
        erase_bg,
        color_min=restore_color_min_all,
        edge_thresh=policy.restore_threshold,
        steps=restore_steps_all,
        edge_bias=edge_bias,
    )
    support_all, neutral_edge_bias_all = _restore_neutral_edge_bias_rim(
        support_all,
        neutral_edge_bias_candidate_all,
        solver_edge_context_all,
        color_all,
        hard_fg_core,
        erase_bg,
        edge_bias=edge_bias,
        edge_thresh=policy.restore_threshold,
    )
    support_all, edge_bridge_all = _bridge_selected_edge_seams(
        support_all,
        restore_candidate_all,
        solver_edge_context_all,
        hard_fg_core,
        erase_bg,
    )
    support_all = _limit_smooth_outside_growth(
        support_all, hint, solver_edge_context_all, hard_fg_core, erase_bg, strength=strength)
    support_all, interior_fill_all = _fill_selected_hint_holes(
        support_all, hint, hard_fg_core, erase_bg)
    support_all = _er._preserve_draw_component_separation(hint, support_all)

    debug_planes = [
        ("image_edge", edge_strength),
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
        ("edge_bias_auto", edge_bias_auto_all),
        ("edge_bias_effective", edge_bias_effective_all),
        ("edge_bias_offset", np.full((h, w), float(edge_bias), dtype=np.float32)),
        ("edge_policy_ridge_threshold", np.full((h, w), policy.ridge_threshold, dtype=np.float32)),
        ("edge_policy_restore_threshold", np.full((h, w), policy.restore_threshold, dtype=np.float32)),
        ("edge_policy_side_threshold", np.full((h, w), policy.side_threshold, dtype=np.float32)),
        ("edge_policy_outside_keep_threshold", np.full((h, w), policy.outside_keep_threshold, dtype=np.float32)),
        ("boundary_bias_px", np.full((h, w), policy.boundary_bias_px, dtype=np.float32)),
    ]

    hint_area = int(np.count_nonzero(hint))
    support_area = int(np.count_nonzero(support_all))
    ratio = (support_area / hint_area) if hint_area else 0.0
    logging.debug(
        "[DRAW_QS] hint=%d band=%d edge_px_in_band=%d support=%d ratio=%.3f "
        "comps=%d max_flow=%d radius=%.1f",
        hint_area,
        total_band,
        int(np.count_nonzero(band_all & (solver_edge_context_all >= 0.4))),
        support_area,
        ratio,
        len(solve_units),
        total_flow,
        scales.band_half_width,
    )

    return DrawSupportResult(fg_seed_all, band_all, support_all, debug_planes)


_DUMP_COUNTER = 0


def _maybe_dump_input(
        guide,
        mask,
        radius,
        strength,
        seed_mask,
        draw_strokes,
        pixel_scale=1.0,
        strength_mode=None,
        edge_lock_auto=None,
        edge_lock_effective=None,
        edge_lock_offset=None,
        edge_bias=None):
    """When QS_DUMP_INPUT is set, save the *exact* inputs to an .npz so a
    production call (real resolution + colour space) can be reproduced offline.
    """
    global _DUMP_COUNTER
    d = os.environ.get("QS_DUMP_INPUT")
    if not d:
        # Piggyback on the debug-mosaic flag the user already uses, so the exact
        # production input (real resolution + colour space) is saved next to the
        # mosaic with no extra setup.
        if os.environ.get("PLATYPUS_DEBUG_EDGE_REFINE", "").strip().lower() in {"1", "true", "yes", "on"}:
            d = os.environ.get("PLATYPUS_DEBUG_EDGE_REFINE_DIR", "").strip() or "/tmp/platypus_edge_refine"
        else:
            return
    try:
        limit = int(os.environ.get("QS_DUMP_INPUT_LIMIT", "500"))
    except ValueError:
        limit = 500
    if limit >= 0 and _DUMP_COUNTER >= limit:
        return
    try:
        os.makedirs(d, exist_ok=True)
        strokes = []
        for s in (draw_strokes or []):
            pts = np.asarray(getattr(s, "points", []), dtype=np.float32)
            strokes.append({
                "points": pts,
                "size": float(getattr(s, "size", 1.0)),
                "soft": float(getattr(s, "soft", 100.0)),
                "is_erasing": bool(getattr(s, "is_erasing", False)),
            })
        path = os.path.join(d, f"qs_input_{_DUMP_COUNTER:03d}.npz")
        payload = {
            "guide": np.asarray(guide, dtype=np.float32),
            "mask": np.asarray(mask, dtype=np.float32),
            "seed_mask": (np.asarray(seed_mask) if seed_mask is not None else np.array([])),
            "radius": np.float32(radius),
            "strength": np.float32(strength),
            "pixel_scale": np.float32(pixel_scale),
            "strokes": np.array(strokes, dtype=object),
        }
        if strength_mode is not None:
            payload["strength_mode"] = np.array(str(strength_mode))
        if edge_lock_auto is not None:
            payload["edge_lock_auto"] = np.float32(edge_lock_auto)
        if edge_lock_effective is not None:
            payload["edge_lock_effective"] = np.float32(edge_lock_effective)
        if edge_lock_offset is not None:
            payload["edge_lock_offset"] = np.float32(edge_lock_offset)
        if edge_bias is not None:
            payload["edge_bias"] = np.float32(edge_bias)
        np.savez_compressed(path, **payload)
        _DUMP_COUNTER += 1
        logging.warning("[QS_DUMP_INPUT] wrote %s (guide %s, radius=%.1f, strokes=%d)",
                        path, np.asarray(guide).shape, float(radius), len(strokes))
    except Exception:
        logging.exception("[QS_DUMP_INPUT] failed")


# --- radius / scale resolution ----------------------------------------------
def _resolve_scales(radius, draw_strokes, hint) -> _Scales:
    stroke_hw = _er._draw_strokes_half_width(draw_strokes)
    hint_hw = _er._hint_half_width(hint)
    half_w = float(stroke_hw) if stroke_hw is not None else float(hint_hw)
    # Draw semantics:
    #   * brush half-width is the in-brush search radius (edge may cut the brush
    #     body inward even when UI radius is 0).
    #   * UI radius is an offset. Positive values grow the search outside the
    #     painted footprint; negative values reduce the in-brush clip depth.
    #
    # Legacy QS_BRUSH_AS_RADIUS=0 keeps the old "radius is a boundary band"
    # interpretation for emergency comparison.
    ui = float(radius)
    if BRUSH_AS_RADIUS:
        band_half_width = max(float(half_w) + ui, MIN_BAND)
        grow_radius = max(0.0, ui)
    else:
        band_half_width = max(ui, MIN_BAND)
        grow_radius = band_half_width
    roi_pad = int(round(max(band_half_width, grow_radius, 12.0))) + 8
    return _Scales(band_half_width, grow_radius, roi_pad, max(1.0, half_w))


def _draw_solve_units(mask_f, hint, hard_fg_core, draw_strokes, radius, has_strokes):
    """Split Draw Quick Select into stroke-owned islands when stroke geometry is
    available. Falling back to rendered-mask connected components keeps legacy
    and programmatic callers working.
    """
    hint = np.asarray(hint, dtype=bool)
    units: List[_SolveUnit] = []
    covered = np.zeros_like(hint, dtype=bool)

    if has_strokes and draw_strokes:
        shape = hint.shape
        has_erase = any(bool(getattr(stroke, "is_erasing", False)) for stroke in draw_strokes)
        hint_labels = None
        if not has_erase and np.any(hint):
            _, hint_labels = cv2.connectedComponents(hint.astype(np.uint8), connectivity=8)
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
            stroke_scales = _resolve_scales(radius, [stroke], stroke_mask)
            if hint_labels is not None:
                touched = np.unique(hint_labels[center]) if np.any(center) else np.array([], dtype=np.int32)
                if touched.size == 0:
                    touched = np.unique(hint_labels[stroke_mask])
                for label_id in touched:
                    label_id = int(label_id)
                    if label_id == 0:
                        continue
                    component = hint_labels == label_id
                    component_center = center & component
                    if not np.any(component_center):
                        component_center = stroke_mask & component
                    stroke_core = _seed_core(
                        np.asarray(mask_f) * component, component_center, component)
                    covered |= component
                    _append_connected_units(units, component, stroke_core, stroke_scales)
            else:
                stroke_core = _seed_core(np.asarray(mask_f) * stroke_mask, center, stroke_mask)
                covered |= stroke_mask
                _append_connected_units(units, stroke_mask, stroke_core, stroke_scales)

    fallback = hint & ~covered if units else hint
    if units:
        fallback = _drop_tiny_components(fallback, MIN_SOLVE_COMPONENT_AREA)
    if np.any(fallback):
        fallback_scales = _resolve_scales(radius, draw_strokes, fallback)
        _append_connected_units(units, fallback, hard_fg_core & fallback, fallback_scales)

    return units


def _drop_tiny_components(mask, min_area):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask
    areas = np.bincount(labels.reshape(-1), minlength=n_labels)
    keep_labels = np.flatnonzero(areas >= int(max(1, min_area)))
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == n_labels - 1:
        return mask
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep_labels)


def _append_connected_units(units, component_mask, core_mask, scales):
    component_mask = np.asarray(component_mask, dtype=bool)
    if not np.any(component_mask):
        return
    n_labels, labels = cv2.connectedComponents(component_mask.astype(np.uint8), connectivity=8)
    areas = np.bincount(labels.reshape(-1), minlength=n_labels)
    skip_tiny = n_labels > 2 and int(areas[1:].max(initial=0)) >= 1024
    for label_id in range(1, n_labels):
        if skip_tiny and int(areas[label_id]) < int(MIN_SOLVE_COMPONENT_AREA):
            continue
        component = labels == label_id
        core = np.asarray(core_mask, dtype=bool) & component
        if not np.any(core):
            core = _seed_core(component.astype(np.float32), None, component)
        units.append(_SolveUnit(component, core, scales))


# --- edge cost ---------------------------------------------------------------
def _solver_edge_strength(edge_strength, pixel_scale=1.0):
    edge = np.asarray(edge_strength, dtype=np.float32)
    try:
        scale = float(pixel_scale)
    except Exception:
        scale = 1.0
    if scale <= 1.01:
        return edge
    # Full-view zoom renders solve regions at source-image scale. The same
    # physical edge then spans more pixels, and the graph's edge term becomes
    # visibly stricter than the current-view solve. A gentle quarter-power
    # normalization keeps zoomed and unzoomed Draw QS behaviour aligned without
    # disabling the high-res edge information completely.
    factor = float(np.clip(scale ** -0.25, 0.55, 1.0))
    return (edge * factor).astype(np.float32, copy=False)


def _edge_policy(strength, edge_bias=0.0) -> _EdgePolicy:
    """Resolve the Draw QS control policy.

    Keep the public controls semantically separate:
      * EdgeLock/strength changes which image ridges count as boundaries.
      * Edge Bias changes the chosen side of an already accepted boundary.
      * Radius is resolved separately in _resolve_scales and only bounds search.
    """
    sensitivity = float(np.clip(strength, 0.0, 100.0))
    lock = sensitivity / 100.0
    return _EdgePolicy(
        sensitivity=sensitivity,
        ridge_threshold=float(0.58 - 0.30 * lock),
        ridge_falloff_sigma=float(0.48 + 0.42 * lock),
        cut_sigma=float(0.70 - 0.50 * lock),
        side_threshold=_side_edge_thresh_for_strength(strength),
        side_relax_weight=_side_edge_relax_weight_for_strength(strength),
        outside_keep_threshold=_outside_keep_edge_thresh_for_strength(strength),
        restore_threshold=_edge_restore_thresh_for_strength(strength),
        boundary_bias_px=float(edge_bias),
    )


def _edge_cost_map(edge_strength, strength=None, *, policy: Optional[_EdgePolicy] = None):
    # EdgeLock is used as edge sensitivity for Draw Quick Select:
    #   0   = strict; only strong ridges are attractive cut locations.
    #   100 = loose; weaker/diffuse ridges also become cheap to cut.
    if policy is None:
        policy = _edge_policy(0.0 if strength is None else strength)
    edge = np.clip(edge_strength.astype(np.float32, copy=False), 0.0, 1.0)
    # The raw edge map is blurred (a few px wide), so the cheap-to-cut band is
    # wide and the boundary stops on its inner ramp ~2-3px short of the true
    # peak when reaching outward. Thin it to a 1px ridge so the only cheap cut is
    # at the edge peak -> the boundary snaps precisely onto the edge.
    edge = _thin_edge_to_ridge(
        edge,
        thr=policy.ridge_threshold,
        falloff_sigma=policy.ridge_falloff_sigma,
    )
    g = np.exp(-(edge * edge) / (policy.cut_sigma * policy.cut_sigma))
    return g.astype(np.float32, copy=False)


def _edge_context_weight_for_strength(strength):
    start = float(np.clip(EDGE_CONTEXT_START, 0.0, 99.0))
    lock = float(np.clip(strength, 0.0, 100.0))
    return float(np.clip((lock - start) / max(100.0 - start, 1e-3), 0.0, 1.0))


def _contextual_edge_strength(edge_strength, color_score, strength):
    edge = np.asarray(edge_strength, dtype=np.float32)
    w = _edge_context_weight_for_strength(strength)
    if w <= 0.0:
        return edge

    color = np.asarray(color_score, dtype=np.float32)
    if color.shape != edge.shape:
        return edge
    color = np.nan_to_num(color, nan=0.0, posinf=1.0, neginf=-1.0)
    blurred = cv2.GaussianBlur(color, (0, 0), 1.0)
    kernel = np.ones((5, 5), dtype=np.uint8)
    c_hi = cv2.dilate(blurred, kernel)
    c_lo = cv2.erode(blurred, kernel)
    span = c_hi - c_lo
    if float(np.max(span, initial=0.0)) < float(EDGE_CONTEXT_MIN_SPAN):
        return edge

    transition = np.clip(span / max(float(EDGE_CONTEXT_SPAN_SCALE), 1e-6), 0.0, 1.0)
    sign_change = (c_hi > float(EDGE_CONTEXT_SIGN_THRESH)) & (c_lo < -float(EDGE_CONTEXT_SIGN_THRESH))
    gate_signal = np.maximum(
        transition,
        sign_change.astype(np.float32) * float(EDGE_CONTEXT_SIGN_BONUS),
    )
    gate = float(EDGE_CONTEXT_FLOOR) + (1.0 - float(EDGE_CONTEXT_FLOOR)) * gate_signal
    gate = np.clip(gate, 0.0, 1.0).astype(np.float32, copy=False)
    contextual = edge * gate
    return (edge * (1.0 - w) + contextual * w).astype(np.float32, copy=False)


def _thin_edge_to_ridge(edge, thr=0.35, falloff_sigma=_envf("QS_EDGE_FALLOFF", 0.8)):
    """Concentrate the (blurred) edge magnitude onto its 1px medial ridge so the
    min-cut snaps to the true edge peak instead of its inner ramp."""
    if _skeletonize is None:
        return edge
    strong = edge > thr
    if not np.any(strong):
        return edge
    try:
        ridge = np.asarray(_skeletonize(strong), dtype=bool)
    except Exception:
        return edge
    if not np.any(ridge):
        return edge
    dist = cv2.distanceTransform((~ridge).astype(np.uint8), cv2.DIST_L2, 3)
    falloff = np.exp(-(dist * dist) / (2.0 * falloff_sigma * falloff_sigma)).astype(np.float32)
    # Sharpen only the strong ridges; leave weak edges (below thr) untouched.
    # Weak/diffuse boundaries near a strong snow/tree ridge are still useful at
    # high EdgeLock, and suppressing them makes the slider feel ineffective.
    sharp = edge * falloff
    return np.where(strong, sharp, edge).astype(np.float32, copy=False)


def _color_score(guide, comp, core, all_hint, R_out, directional_bg=False):
    """Signed colour data term in [-1, 1] for one component: +1 looks like this
    stroke's FG colour, -1 looks like its local background. 0 when indistinct.
    The BG shell excludes *all* strokes (``all_hint``) so other components are
    not sampled as background."""
    score, _delta = _color_score_and_luma_delta(
        guide, comp, core, all_hint, R_out, directional_bg=directional_bg)
    return score


def _color_score_and_luma_delta(guide, comp, core, all_hint, R_out, directional_bg=False):
    """Return colour membership and FG-vs-BG luma relation from one shell sample."""
    score = np.zeros(comp.shape, dtype=np.float32)
    if guide is None or not np.any(core):
        return score, 0.0
    rgb = np.asarray(guide, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return score, 0.0
    rgb = rgb[..., :3]
    lab = _er._guide_to_lab(rgb)
    if lab.ndim != 3:
        return score, 0.0
    fg = core
    dist_out = _er._distance_from(comp)
    free = ~np.asarray(all_hint, dtype=bool)
    shell = free & (dist_out <= max(R_out, 12.0) + 4.0) & (dist_out >= 2.0)
    if not np.any(shell):
        shell = free & (dist_out <= max(R_out, 12.0) + 12.0)
    if not np.any(shell):
        return score, 0.0
    fg_rgb_med = np.median(rgb[fg].reshape(-1, 3), axis=0)
    bg_rgb_med = np.median(rgb[shell].reshape(-1, 3), axis=0)
    fg_med = np.median(lab[fg].reshape(-1, 3), axis=0)
    bg_med = np.median(lab[shell].reshape(-1, 3), axis=0)
    if directional_bg:
        bg_rgb_med = _directional_shell_median(rgb, shell, fg_rgb_med, bg_rgb_med)
        bg_med = _directional_shell_median(lab, shell, fg_med, bg_med)
    luma = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    selected_luma_delta = float(np.dot(fg_rgb_med[:3] - bg_rgb_med[:3], luma))
    # Scale the colour term by how separable FG/BG are, instead of a hard cutoff.
    # Low-contrast scenes (snow: cloud vs blue sky differ by only ~6 LAB) used to
    # fall under the cutoff and get NO colour help, leaving the boundary to a
    # weak edge alone; a graceful confidence keeps a (weaker) colour signal that
    # still pulls the boundary onto the cloud/sky luminance step.
    sep = float(np.linalg.norm(fg_med - bg_med))
    conf = float(np.clip((sep - COLOR_MIN_SEP) / COLOR_SEP_SCALE, 0.0, 1.0))
    if conf <= 0.0:
        return score, selected_luma_delta  # FG/BG colours genuinely indistinct
    d_fg = np.linalg.norm(lab - fg_med, axis=2)
    d_bg = np.linalg.norm(lab - bg_med, axis=2)
    score = conf * (d_bg - d_fg) / (d_bg + d_fg + 1e-3)
    return np.clip(score, -1.0, 1.0).astype(np.float32), selected_luma_delta


def _edge_restore_color_min_for_unit(
        guide,
        comp,
        core,
        all_hint,
        R_out,
        directional_bg=False,
        strength=0.0):
    delta = _selected_luma_delta(
        guide, comp, core, all_hint, R_out, directional_bg=directional_bg)
    if delta > float(BRIGHT_EDGE_RESTORE_LUMA_DELTA):
        return _bright_edge_restore_color_min_for_strength(strength)
    return float(EDGE_RESTORE_COLOR_MIN)


def _edge_restore_color_min_for_luma_delta(selected_luma_delta, strength=0.0):
    if float(selected_luma_delta) > float(BRIGHT_EDGE_RESTORE_LUMA_DELTA):
        return _bright_edge_restore_color_min_for_strength(strength)
    return float(EDGE_RESTORE_COLOR_MIN)


def _bright_edge_restore_color_min_for_strength(strength):
    lock = float(np.clip(strength, 0.0, 100.0))
    end = float(max(1.0, BRIGHT_EDGE_RESTORE_COLOR_MIN_LOCK_END))
    t = float(np.clip(lock / end, 0.0, 1.0))
    strict_edge = float(BRIGHT_EDGE_RESTORE_COLOR_MIN)
    loose_edge = float(BRIGHT_EDGE_RESTORE_COLOR_MIN_LOCKED)
    return float(strict_edge * (1.0 - t) + loose_edge * t)


def _is_neutral_edge_bias_unit(selected_luma_delta):
    return abs(float(selected_luma_delta)) <= float(EDGE_BIAS_NEUTRAL_LUMA_MAX)


def _auto_edge_bias_for_unit(selected_luma_delta, scales):
    delta = float(selected_luma_delta)
    stroke_hw = float(max(1.0, getattr(scales, "stroke_half_width", 1.0)))
    if _is_neutral_edge_bias_unit(delta):
        return 0.0
    if delta > 0.75 and stroke_hw >= 120.0:
        return -2.0
    if 0.30 <= delta <= 0.75 and stroke_hw >= 120.0:
        return 2.0
    if delta < -0.50 and stroke_hw >= 80.0:
        return -1.0
    return 0.0


def _edge_restore_steps_for_luma(selected_luma_delta, edge_bias=0.0):
    try:
        bias = float(edge_bias)
    except Exception:
        bias = 0.0
    if 0.0 < float(selected_luma_delta) <= float(EDGE_BIAS_NEUTRAL_LUMA_MAX):
        base = float(EDGE_BIAS_NEUTRAL_AUTO_PX)
    else:
        base = float(EDGE_RESTORE_STEPS)
    return float(np.clip(round(base + bias), 0, EDGE_BIAS_MAX_STEPS))


def _neutral_edge_bias_steps(edge_bias=0.0):
    try:
        bias = float(edge_bias)
    except Exception:
        bias = 0.0
    return int(np.clip(round(float(EDGE_BIAS_NEUTRAL_AUTO_PX) + bias), 0, EDGE_BIAS_MAX_STEPS))


def _color_weight_for_unit(guide, comp, core, all_hint, R_out, directional_bg=False, strength=100.0):
    delta = _selected_luma_delta(
        guide, comp, core, all_hint, R_out, directional_bg=directional_bg)
    return _color_weight_for_luma_delta(delta, strength=strength)


def _color_weight_for_luma_delta(selected_luma_delta, strength=100.0):
    delta = float(selected_luma_delta)
    if delta > float(BRIGHT_EDGE_RESTORE_LUMA_DELTA):
        start = float(np.clip(BRIGHT_COLOR_W_START, 0.0, 99.0))
        lock = float(np.clip(strength, 0.0, 100.0))
        t = float(np.clip((lock - start) / max(100.0 - start, 1e-3), 0.0, 1.0))
        base = float(min(COLOR_W, BRIGHT_COLOR_W_BASE))
        target = float(min(base, BRIGHT_COLOR_W))
        return float(base * (1.0 - t) + target * t)
    return float(COLOR_W)


def _selected_luma_delta(guide, comp, core, all_hint, R_out, directional_bg=False):
    if guide is None or not np.any(core):
        return 0.0
    rgb = np.asarray(guide, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return 0.0
    rgb = rgb[..., :3]
    dist_out = _er._distance_from(comp)
    free = ~np.asarray(all_hint, dtype=bool)
    shell = free & (dist_out <= max(R_out, 12.0) + 4.0) & (dist_out >= 2.0)
    if not np.any(shell):
        shell = free & (dist_out <= max(R_out, 12.0) + 12.0)
    if not np.any(shell):
        return 0.0
    fg_med = np.median(rgb[core].reshape(-1, 3), axis=0)
    bg_med = np.median(rgb[shell].reshape(-1, 3), axis=0)
    if directional_bg:
        bg_med = _directional_shell_median(rgb, shell, fg_med, bg_med)
    luma = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float(np.dot(fg_med[:3] - bg_med[:3], luma))


def _directional_shell_median(lab, shell, fg_med, fallback_med):
    candidate = _most_separated_shell_median(lab, shell, fg_med, fallback_med)
    base_sep = float(np.linalg.norm(np.asarray(fallback_med, dtype=np.float32) - fg_med))
    candidate_sep = float(np.linalg.norm(np.asarray(candidate, dtype=np.float32) - fg_med))
    # A directional shell helps when the local shell mixes several plausible
    # backgrounds. If the farthest sector is wildly farther from the seed than
    # the typical shell, it is usually a different foreground object inside the
    # brush (snow next to a dark branch), and using it flips FG/BG.
    if candidate_sep > max(base_sep * float(DIRECTIONAL_BG_MAX_SEP_RATIO), base_sep + 1e-3):
        return fallback_med
    return candidate


def _most_separated_shell_median(lab, shell, fg_med, fallback_med):
    ys, xs = np.where(shell)
    if ys.size < 64:
        return fallback_med
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    candidates = [np.asarray(fallback_med, dtype=np.float32)]
    min_n = max(32, int(round(float(ys.size) * 0.10)))
    yy, xx = np.indices(shell.shape)
    sectors = (
        shell & (xx <= cx),
        shell & (xx > cx),
        shell & (yy <= cy),
        shell & (yy > cy),
        shell & (xx <= cx) & (yy <= cy),
        shell & (xx > cx) & (yy <= cy),
        shell & (xx <= cx) & (yy > cy),
        shell & (xx > cx) & (yy > cy),
    )
    for sector in sectors:
        if int(np.count_nonzero(sector)) < min_n:
            continue
        candidates.append(np.median(lab[sector].reshape(-1, 3), axis=0))
    if len(candidates) <= 1:
        return fallback_med
    dists = [float(np.linalg.norm(np.asarray(c) - fg_med)) for c in candidates]
    return np.asarray(candidates[int(np.argmax(dists))], dtype=np.float32)


def _seed_core(mask_f, fg_stroke, hint):
    # Hard-FG anchor. When the user drew strokes, the *stroke centerline* is the
    # intent: it sits on one side of the target edge, so the opposite-side brush
    # body stays negotiable and snaps. The medial-axis skeleton of a wide/curved
    # stroke instead zig-zags and straddles the edge, anchoring the wrong side as
    # FG (the body can no longer be clipped -> long strokes "fail"); so the
    # skeleton is used ONLY when there is no stroke geometry (e.g. a programmatic
    # solid mask, where the medial curve correctly preserves the interior).
    if fg_stroke is not None and np.any(fg_stroke):
        core = fg_stroke & hint
        if np.any(core):
            return core
    if _skeletonize is not None and np.any(hint):
        try:
            core = np.asarray(_skeletonize(hint), dtype=bool)
            if np.any(core):
                return core
        except Exception:
            pass
    return _er.make_confident_seed(mask_f, shrink_ratio=CORE_SHRINK, min_shrink=1.0) & hint


# --- per-component solve -----------------------------------------------------
class _ComponentSolve(NamedTuple):
    support: np.ndarray
    band: np.ndarray
    hard_fg: np.ndarray
    hard_bg: np.ndarray
    prior: np.ndarray
    cut_boundary: np.ndarray
    flow_value: int


def _solve_component(
        comp,
        hint_roi,
        g_roi,
        core_roi,
        erase_roi,
        color_roi,
        scales,
        side_edge_roi=None,
        color_weight=COLOR_W,
        side_edge_thresh=None,
        side_relax_weight=1.0,
        prior_floor_in=None,
        strict_side_edge_thresh=None,
        side_dilate=None,
        inside_color_bg_thresh=None,
        inside_color_bg_weight=1.0) -> _ComponentSolve:
    zeros = np.zeros_like(comp, dtype=bool)
    fzeros = np.zeros(comp.shape, dtype=np.float32)
    if not np.any(comp):
        return _ComponentSolve(zeros, zeros, zeros, zeros, fzeros, zeros, 0)

    R_in = float(scales.band_half_width)
    R_out = float(scales.grow_radius)
    other_hint = hint_roi & ~comp

    dist_in = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform((~comp).astype(np.uint8), cv2.DIST_L2, 3)
    half_w = max(1.0, float(dist_in.max(initial=0.0)))

    core = core_roi & comp
    # Bound the *inward* clip to `radius`: brush interior deeper than R_out from
    # the boundary is hard FG, so a low radius keeps the mask ~as drawn and only
    # a rim of width R_out can snap. (The centerline core keeps strokes thinner
    # than R_out fully snap-able.) This makes radius behave predictably on both
    # sides and stops long strokes from clipping deep notches at radius~0.
    core = core | (comp & (dist_in > R_in))
    if not np.any(core):
        ridge = max(0.5, float(dist_in.max(initial=0.0)) - 0.5)
        core = comp & (dist_in >= ridge)

    dilated = comp | (dist_out <= R_out)
    outer_bg = ~dilated
    band = dilated & ~core & ~other_hint
    if not np.any(band):
        # Nothing to decide; keep the hint as-is.
        support = comp.copy()
        return _ComponentSolve(support, zeros, core, outer_bg, fzeros, zeros, 0)

    kernel = np.ones((3, 3), dtype=np.uint8)
    src_inf = band & (cv2.dilate(core.astype(np.uint8), kernel, iterations=1) > 0)
    sink_inf = band & ((cv2.dilate(outer_bg.astype(np.uint8), kernel, iterations=1) > 0) | erase_roi)
    sink_inf &= ~src_inf

    # Soft prior: + inside the original hint, - outside, decaying toward the
    # boundary so the cut can move freely near it but resists big jumps.
    prior = np.zeros(comp.shape, dtype=np.float32)
    inside = band & comp
    outside = band & ~comp
    # Geometric prior: + inside the drawn mask, - outside, both growing away from
    # the boundary. Inside it prevents min-cut's shrinking bias (the body does
    # not collapse toward the skeleton in smooth regions); outside it prevents
    # outward bulge / inflation. The colour term (added below) is what actually
    # clips a same-colour-as-nothing spill across an edge.
    floor_in = float(PRIOR_FLOOR_IN if prior_floor_in is None else prior_floor_in)
    mag_in = np.clip(floor_in + (1.0 - floor_in) * dist_in / max(half_w, 1.0), 0.0, 1.0)
    f_out = dist_out / max(R_out, 1.0)
    rim_ramp = np.clip((f_out - REACH_FRAC) / max(1.0 - REACH_FRAC, 1e-3), 0.0, 1.0)
    mag_out = np.clip(PRIOR_FLOOR_OUT + (1.0 - PRIOR_FLOOR_OUT) * rim_ramp, 0.0, 1.0)
    prior[inside] = mag_in[inside]
    prior[outside] = -mag_out[outside]
    strict_seed_side = _seed_side_through_smooth_interior(
        comp,
        core,
        side_edge_roi,
        edge_thresh=SIDE_EDGE_THRESH if strict_side_edge_thresh is None else strict_side_edge_thresh,
        side_dilate=side_dilate,
    )
    strict_opposite = inside & ~strict_seed_side
    if np.any(strict_opposite):
        prior[strict_opposite] = -np.maximum(mag_in[strict_opposite], floor_in)
    if side_edge_thresh is not None and float(side_edge_thresh) < float(SIDE_EDGE_THRESH):
        w = float(np.clip(side_relax_weight, 0.0, 1.0))
        side_thresh = float(side_edge_thresh)
        window = float(max(0.0, SIDE_EDGE_SOFT_WINDOW))
        if window > 1e-6:
            high_thresh = float(min(SIDE_EDGE_THRESH, side_thresh + window))
            low_thresh = float(max(SIDE_EDGE_LOOSE_THRESH, side_thresh - window))
            high_seed_side = _seed_side_through_smooth_interior(
                comp, core, side_edge_roi, edge_thresh=high_thresh, side_dilate=side_dilate)
            low_seed_side = _seed_side_through_smooth_interior(
                comp, core, side_edge_roi, edge_thresh=low_thresh, side_dilate=side_dilate)
            firm_opposite = inside & strict_seed_side & ~high_seed_side
            soft_opposite = inside & strict_seed_side & high_seed_side & ~low_seed_side
            if np.any(firm_opposite):
                firm_w = w * w
                bg_prior = -np.maximum(mag_in[firm_opposite], floor_in)
                prior[firm_opposite] = (
                    prior[firm_opposite] * (1.0 - firm_w) + bg_prior * firm_w
                )
            if np.any(soft_opposite):
                soft_w = w * w
                bg_prior = -np.maximum(mag_in[soft_opposite], floor_in)
                prior[soft_opposite] = (
                    prior[soft_opposite] * (1.0 - soft_w) + bg_prior * soft_w
                )
        else:
            loose_seed_side = _seed_side_through_smooth_interior(
                comp, core, side_edge_roi, edge_thresh=side_thresh, side_dilate=side_dilate)
            weak_opposite = inside & strict_seed_side & ~loose_seed_side
            if np.any(weak_opposite):
                bg_prior = -np.maximum(mag_in[weak_opposite], floor_in)
                prior[weak_opposite] = prior[weak_opposite] * (1.0 - w) + bg_prior * w

    if inside_color_bg_thresh is not None:
        try:
            color_thresh = float(inside_color_bg_thresh)
        except Exception:
            color_thresh = None
        if color_thresh is not None:
            color_opposite = inside & (np.asarray(color_roi, dtype=np.float32) < color_thresh) & ~core
            if np.any(color_opposite):
                w = float(np.clip(inside_color_bg_weight, 0.0, 1.0))
                bg_prior = -np.maximum(mag_in[color_opposite], floor_in)
                prior[color_opposite] = prior[color_opposite] * (1.0 - w) + bg_prior * w

    # Colour data term. Inside the mask it may pull either way (a same-geometry
    # spill of a *different* colour is pushed to BG and clipped). Outside the mask
    # colour may only push toward BG, never pull FG -- otherwise a large radius
    # grabs every same-coloured pixel (the grabCut explosion / busy-texture grab).
    # QS_COLOR_W_OUT>0 re-enables outward same-colour growth ("fill the sky up to
    # the edges"); default 0 keeps the conservative no-explosion behaviour.
    color_eff = color_roi.astype(np.float32, copy=True)
    color_w = float(max(0.0, color_weight))
    if COLOR_W_OUT > 0.0:
        pos_out = outside & (color_eff > 0.0)
        color_eff[pos_out] *= (COLOR_W_OUT / max(color_w, 1e-3))
    else:
        color_eff[outside] = np.minimum(color_eff[outside], 0.0)
    score = prior + color_w * color_eff
    score[src_inf] = 0.0
    score[sink_inf] = 0.0

    free = band & ~src_inf & ~sink_inf
    src_cap = np.zeros(comp.shape, dtype=np.int64)
    sink_cap = np.zeros(comp.shape, dtype=np.int64)
    src_cap[src_inf] = INF
    sink_cap[sink_inf] = INF
    pos = free & (score > 0.0)
    neg = free & (score < 0.0)
    src_cap[pos] = np.maximum(1, np.rint(CAP * BETA * score[pos]).astype(np.int64))
    sink_cap[neg] = np.maximum(1, np.rint(CAP * BETA * (-score[neg])).astype(np.int64))

    fg_band, flow_value = _build_and_solve(band, g_roi, src_cap, sink_cap)
    if fg_band is None:  # band too large -> fall back to raw hint
        support = comp.copy()
        return _ComponentSolve(support, band, core, sink_inf, prior, zeros, 0)

    support = core | fg_band
    support &= dilated  # never escape the band
    cut_boundary = band & (cv2.morphologyEx(
        support.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0)
    return _ComponentSolve(support, band, src_inf | core, sink_inf, prior, cut_boundary, flow_value)


def _side_edge_thresh_for_strength(strength):
    start = float(np.clip(SIDE_EDGE_RELAX_START, 0.0, 99.0))
    lock = float(np.clip(strength, 0.0, 100.0))
    t = float(np.clip((lock - start) / max(100.0 - start, 1e-3), 0.0, 1.0))
    strict = float(SIDE_EDGE_THRESH)
    loose = float(SIDE_EDGE_LOOSE_THRESH)
    return float(strict * (1.0 - t) + loose * t)


def _side_edge_relax_weight_for_strength(strength):
    start = float(np.clip(SIDE_EDGE_RELAX_START, 0.0, 99.0))
    lock = float(np.clip(strength, 0.0, 100.0))
    return float(np.clip((lock - start) / max(100.0 - start, 1e-3), 0.0, 1.0))


def _seed_side_through_smooth_interior(comp, core, edge_roi, edge_thresh=None, side_dilate=None):
    comp = np.asarray(comp, dtype=bool)
    core = np.asarray(core, dtype=bool) & comp
    if not np.any(comp) or not np.any(core):
        return comp.copy()
    if edge_roi is None:
        return comp.copy()
    # Side splitting should be stable while the user moves EdgeLock. Use the raw
    # strong image ridge map here; EdgeLock only changes the cut attraction, not
    # which side of the brush body becomes an opposite-side BG prior.
    thresh = float(SIDE_EDGE_THRESH if edge_thresh is None else edge_thresh)
    barrier = comp & (np.asarray(edge_roi, dtype=np.float32) >= thresh)
    if not np.any(barrier):
        return comp.copy()
    barrier = _filter_side_edge_barrier_components(barrier, comp)
    if not np.any(barrier):
        return comp.copy()
    dilate_steps = SIDE_DILATE if side_dilate is None else int(max(0, round(float(side_dilate))))
    if dilate_steps > 0:
        barrier = comp & (cv2.dilate(
            barrier.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=dilate_steps,
        ) > 0)
    walkable = comp & ~barrier
    seed = core & walkable
    if not np.any(seed):
        return comp.copy()
    seed_side = _er._connected_to_seed(walkable, seed)
    # Keep the core itself source-side even if a ridge overlaps it.
    seed_side |= core
    return seed_side & comp


def _filter_side_edge_barrier_components(barrier, comp):
    barrier = np.asarray(barrier, dtype=bool)
    if not np.any(barrier):
        return barrier
    comp_area = int(np.count_nonzero(comp))
    if comp_area <= 0:
        return barrier
    min_area = min(
        float(SIDE_EDGE_MIN_COMPONENT_AREA),
        max(1.0, float(comp_area) * float(SIDE_EDGE_MIN_COMPONENT_FRAC)),
    )
    if min_area <= 1.0:
        return barrier
    n_labels, labels = cv2.connectedComponents(barrier.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return barrier
    areas = np.bincount(labels.reshape(-1), minlength=n_labels)
    keep = np.flatnonzero(areas >= float(min_area))
    keep = keep[keep != 0]
    if keep.size == n_labels - 1:
        return barrier
    if keep.size == 0:
        return np.zeros_like(barrier, dtype=bool)
    return np.isin(labels, keep)


def _build_and_solve(band, g, src_cap, sink_cap):
    ys, xs = np.where(band)
    n = int(ys.size)
    if n == 0:
        return np.zeros_like(band), 0
    if n > MAX_BAND_NODES:
        return None, 0

    node_id = np.full(band.shape, -1, dtype=np.int64)
    node_id[ys, xs] = np.arange(n, dtype=np.int64)
    src = n
    sink = n + 1

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []

    def _nlinks(b_from, b_to, g_from, g_to, id_from, id_to):
        if id_from.size == 0:
            return
        w = np.rint(CAP * (np.minimum(g_from, g_to) + LAMBDA)).astype(np.int64)
        rows.append(id_from); cols.append(id_to); data.append(w)
        rows.append(id_to); cols.append(id_from); data.append(w)

    # right neighbour
    pair = band[:, :-1] & band[:, 1:]
    yy, xx = np.where(pair)
    if yy.size:
        _nlinks(None, None, g[yy, xx], g[yy, xx + 1], node_id[yy, xx], node_id[yy, xx + 1])
    # down neighbour
    pair = band[:-1, :] & band[1:, :]
    yy, xx = np.where(pair)
    if yy.size:
        _nlinks(None, None, g[yy, xx], g[yy + 1, xx], node_id[yy, xx], node_id[yy + 1, xx])

    # t-links
    sc = src_cap[ys, xs]
    tc = sink_cap[ys, xs]
    s_nodes = np.where(sc > 0)[0]
    if s_nodes.size:
        rows.append(np.full(s_nodes.size, src, dtype=np.int64))
        cols.append(s_nodes.astype(np.int64))
        data.append(sc[s_nodes].astype(np.int64))
    t_nodes = np.where(tc > 0)[0]
    if t_nodes.size:
        rows.append(t_nodes.astype(np.int64))
        cols.append(np.full(t_nodes.size, sink, dtype=np.int64))
        data.append(tc[t_nodes].astype(np.int64))

    if not rows:
        return np.zeros_like(band), 0

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    dat = np.concatenate(data)
    graph = csr_matrix((dat, (row, col)), shape=(n + 2, n + 2))

    result = maximum_flow(graph, src, sink, method="dinic")
    # Source side of the min-cut = nodes reachable from the source in the
    # residual graph (positive residual capacity).
    residual = graph - result.flow
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    order = breadth_first_order(
        residual, src, directed=True, return_predecessors=False)
    reachable = np.zeros(n + 2, dtype=bool)
    reachable[order] = True

    fg = np.zeros(band.shape, dtype=bool)
    fg[ys, xs] = reachable[:n]
    return fg, int(result.flow_value)


# --- post-process ------------------------------------------------------------
def _postprocess_support(support, hint, hard_fg_core, erase_bg):
    support = support | hard_fg_core
    support &= ~erase_bg
    # Drop FG islands not connected to any hard-FG seed (noise safety net).
    if np.any(hard_fg_core):
        support = _er._connected_to_seed(support, hard_fg_core) | hard_fg_core
        support &= ~erase_bg
    return support


def _limit_smooth_outside_growth(
        support,
        hint,
        edge_strength,
        hard_fg_core,
        erase_bg,
        strength=0.0):
    support = np.asarray(support, dtype=bool)
    hint = np.asarray(hint, dtype=bool)
    outside = support & ~hint
    if not np.any(outside):
        return support

    edge = np.asarray(edge_strength, dtype=np.float32)
    strict_thresh = _outside_keep_edge_strict_thresh()
    relaxed_thresh = _outside_keep_edge_thresh_for_strength(strength)
    dist_from_hint = _er._distance_from(hint)
    near_hint = dist_from_hint <= float(max(1.0, OUTSIDE_KEEP_EDGE_RELAX_DIST))
    edge_near = (edge >= strict_thresh) | (near_hint & (edge >= relaxed_thresh))
    if OUTSIDE_KEEP_EDGE_NEAR > 0 and np.any(edge_near):
        edge_near = cv2.dilate(
            edge_near.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=OUTSIDE_KEEP_EDGE_NEAR,
        ) > 0
    limited = (support & hint) | (outside & edge_near)
    limited &= ~np.asarray(erase_bg, dtype=bool)
    if np.any(hard_fg_core):
        limited = _er._connected_to_seed(limited, hard_fg_core) | hard_fg_core
        limited &= ~np.asarray(erase_bg, dtype=bool)
    return limited


def _outside_keep_edge_thresh_for_strength(strength):
    lock = float(np.clip(strength, 0.0, 100.0)) / 100.0
    strict = _outside_keep_edge_strict_thresh()
    loose = float(max(0.24, min(OUTSIDE_KEEP_EDGE_THRESH, 0.34)))
    return float(strict * (1.0 - lock) + loose * lock)


def _outside_keep_edge_strict_thresh():
    return float(min(0.72, max(OUTSIDE_KEEP_EDGE_THRESH, 0.66)))


def _fill_selected_hint_holes(support, hint, hard_fg_core, erase_bg):
    support = np.asarray(support, dtype=bool)
    hint = np.asarray(hint, dtype=bool)
    erase = np.asarray(erase_bg, dtype=bool)
    bg_inside_hint = hint & ~support & ~erase
    if not np.any(bg_inside_hint):
        return support, np.zeros_like(support, dtype=bool)

    kernel = np.ones((3, 3), dtype=np.uint8)
    hint_boundary = hint & (cv2.dilate((~hint).astype(np.uint8), kernel, iterations=1) > 0)
    seed = bg_inside_hint & hint_boundary
    if not np.any(seed):
        fill = bg_inside_hint
    else:
        keep_bg = _er._connected_to_seed(bg_inside_hint, seed)
        fill = bg_inside_hint & ~keep_bg
    if not np.any(fill):
        return support, fill

    restored = support | fill
    if np.any(hard_fg_core):
        restored = _er._connected_to_seed(restored, hard_fg_core) | hard_fg_core
        restored &= ~erase
        fill &= restored
    return restored, fill


def _edge_restore_thresh_for_strength(strength):
    lock = float(np.clip(strength, 0.0, 100.0)) / 100.0
    return float(np.clip(0.70 - 0.45 * lock, 0.20, 0.75))


def _restore_selected_edge_rim(
        support,
        candidate,
        edge_strength,
        color_score,
        hard_fg_core,
        erase_bg,
        color_min=EDGE_RESTORE_COLOR_MIN,
        edge_thresh=EDGE_RESTORE_THRESH,
        steps=EDGE_RESTORE_STEPS,
        edge_bias=0.0):
    support = np.asarray(support, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if not np.any(support) or not np.any(candidate):
        return support, np.zeros_like(support, dtype=bool)

    edge = np.asarray(edge_strength, dtype=np.float32)
    color = np.asarray(color_score, dtype=np.float32)
    restored = support.copy()
    restore = np.zeros_like(support, dtype=bool)
    erase = np.asarray(erase_bg, dtype=bool)
    steps_arr = np.asarray(steps, dtype=np.float32)
    if steps_arr.shape:
        max_steps = int(np.clip(np.max(np.rint(steps_arr), initial=0), 0, EDGE_BIAS_MAX_STEPS))
    else:
        max_steps = int(np.clip(round(float(steps_arr)), 0, EDGE_BIAS_MAX_STEPS))
    if max_steps <= 0:
        return support, restore
    edge_near = _edge_near_for_restore(edge, edge_thresh, max_steps)
    edge_band = (
        candidate
        & edge_near
        & ~erase
    )
    color_min_arr = np.asarray(color_min, dtype=np.float32)
    color_min_arr = _edge_bias_adjusted_color_min(color_min_arr, edge_bias)
    if color_min_arr.shape:
        edge_band &= color >= color_min_arr
    else:
        edge_band &= color >= float(color_min_arr)
    if not np.any(edge_band):
        return support, restore

    step_limit = None
    if steps_arr.shape:
        step_limit = np.rint(steps_arr).astype(np.int16, copy=False)
        max_steps = int(np.clip(np.max(step_limit[edge_band], initial=0), 0, EDGE_BIAS_MAX_STEPS))

    kernel = np.ones((3, 3), dtype=np.uint8)
    for step_index in range(max_steps):
        near_support = (cv2.dilate(restored.astype(np.uint8), kernel, iterations=1) > 0) & ~restored
        add = edge_band & near_support
        if step_limit is not None:
            add &= step_limit > step_index
        if not np.any(add):
            break
        restore |= add
        restored |= add
    if not np.any(restore):
        return support, restore

    if np.any(hard_fg_core):
        restored = _er._connected_to_seed(restored, hard_fg_core) | hard_fg_core
        restored &= ~erase
        restore &= restored
    return restored, restore


def _edge_bias_adjusted_color_min(color_min, edge_bias):
    """Positive UI Edge Bias may cross a faint edge-side colour ramp.

    Auto bias changes the default edge inclusion width, but the explicit UI
    offset should feel directional. Keep bias=0 exactly as before; only relax the
    colour gate when the user pushes the boundary outward.
    """
    try:
        bias = float(edge_bias)
    except Exception:
        bias = 0.0
    if bias <= 0.0:
        return color_min
    relax = float(EDGE_BIAS_COLOR_RELAX) * bias
    min_allowed = float(EDGE_BIAS_COLOR_MIN)
    return np.maximum(np.asarray(color_min, dtype=np.float32) - relax, min_allowed)


def _edge_near_for_restore(edge, edge_thresh, steps):
    edge_near = np.asarray(edge, dtype=np.float32) >= float(edge_thresh)
    near = int(max(0, round(max(float(EDGE_RESTORE_EDGE_NEAR), float(steps)))))
    if near > 0 and np.any(edge_near):
        edge_near = cv2.dilate(
            edge_near.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=near,
        ) > 0
    return edge_near


def _restore_neutral_edge_bias_rim(
        support,
        candidate,
        edge_strength,
        color_score,
        hard_fg_core,
        erase_bg,
        edge_bias=0.0,
        edge_thresh=EDGE_RESTORE_THRESH):
    steps = _neutral_edge_bias_steps(edge_bias)
    if steps <= 0:
        return np.asarray(support, dtype=bool), np.zeros_like(support, dtype=bool)
    support = np.asarray(support, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if not np.any(support) or not np.any(candidate):
        return support, np.zeros_like(support, dtype=bool)

    edge = np.asarray(edge_strength, dtype=np.float32)
    color = np.asarray(color_score, dtype=np.float32)
    erase = np.asarray(erase_bg, dtype=bool)
    edge_near = _edge_near_for_restore(edge, edge_thresh, steps)
    edge_band = candidate & edge_near & ~erase & (color >= 0.0)
    if not np.any(edge_band):
        return support, np.zeros_like(support, dtype=bool)

    restored = support.copy()
    restore = np.zeros_like(support, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(steps):
        near_support = (cv2.dilate(restored.astype(np.uint8), kernel, iterations=1) > 0) & ~restored
        add = edge_band & near_support
        if not np.any(add):
            break
        restore |= add
        restored |= add

    if not np.any(restore):
        return support, restore
    if np.any(hard_fg_core):
        restored = _er._connected_to_seed(restored, hard_fg_core) | hard_fg_core
        restored &= ~erase
        restore &= restored
    return restored, restore


def _bridge_selected_edge_seams(
        support,
        candidate,
        edge_strength,
        hard_fg_core,
        erase_bg,
        edge_thresh=EDGE_BRIDGE_THRESH):
    support = np.asarray(support, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if not np.any(support) or not np.any(candidate):
        return support, np.zeros_like(support, dtype=bool)

    erase = np.asarray(erase_bg, dtype=bool)
    edge = np.asarray(edge_strength, dtype=np.float32)
    gap = candidate & ~support & ~erase & (edge >= float(edge_thresh))
    if not np.any(gap):
        return support, np.zeros_like(support, dtype=bool)

    # When two strokes select both sides of the same boundary, the graph can
    # leave only the edge ridge itself unselected. Fill just those bracketed
    # one-pixel seams; count neighbours from the original support so the bridge
    # does not crawl along an edge that has support on only one side.
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbours = cv2.filter2D(
        support.astype(np.uint8),
        cv2.CV_16U,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    bridge = gap & (neighbours >= int(EDGE_BRIDGE_MIN_NEIGHBORS))
    if not np.any(bridge):
        return support, bridge

    restored = support | bridge
    if np.any(hard_fg_core):
        restored = _er._connected_to_seed(restored, hard_fg_core) | hard_fg_core
        restored &= ~erase
        bridge &= restored
    return restored, bridge
