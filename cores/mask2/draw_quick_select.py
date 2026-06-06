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
from typing import List, NamedTuple, Tuple

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
COLOR_W = _envf("QS_COLOR_W", 1.1)
# Outside-the-mask FG-pull weight. 0 = conservative boundary snap (no explosion,
# rejects same-colour texture). >0 = grow through a same-colour region toward the
# surrounding edges ("fill the sky"), at the cost of grabbing same-colour texture
# and inflating at large radius. Off by default; QS_COLOR_W_OUT enables it.
COLOR_W_OUT = _envf("QS_COLOR_W_OUT", 0.0)
# Use the brush half-width as the base search radius (Photoshop-like "brush is
# the search area"); the UI radius then offsets it. This is the Draw Quick
# Select default. QS_BRUSH_AS_RADIUS=0 keeps the old experimental behaviour.
BRUSH_AS_RADIUS = bool(_envf("QS_BRUSH_AS_RADIUS", 1.0))
# Strong image edges inside the brush split the drawn component into seed-side
# and opposite-side regions. The opposite side gets a BG prior so a fat brush
# crossing a snow/sky edge snaps to the centerline side instead of preserving
# both sides just because they were inside the painted disk.
SIDE_G_THRESH = _envf("QS_SIDE_G_THRESH", 0.35)
SIDE_DILATE = int(max(0, round(_envf("QS_SIDE_DILATE", 0.0))))
# Colour-separability confidence: colour contributes at weight
# clip((sep - COLOR_MIN_SEP) / COLOR_SEP_SCALE, 0, 1) where sep is the LAB
# distance between the FG-seed and BG-shell medians. Low (snow) scenes still get
# a partial colour signal instead of being hard-cut to zero.
COLOR_MIN_SEP = _envf("QS_COLOR_MIN_SEP", 1.5)
COLOR_SEP_SCALE = _envf("QS_COLOR_SEP_SCALE", 6.0)
# After min-cut, restore a narrow selected-side edge rim when the cut lands on a
# strong image ridge. This fixes the snow/cloud side case where the graph cuts
# exactly on the ridge but the visible foreground wants the ridge pixels included.
EDGE_RESTORE_THRESH = _envf("QS_EDGE_RESTORE_THRESH", 0.40)
EDGE_RESTORE_COLOR_MIN = _envf("QS_EDGE_RESTORE_COLOR_MIN", 0.05)
BRIGHT_EDGE_RESTORE_COLOR_MIN = _envf("QS_BRIGHT_EDGE_RESTORE_COLOR_MIN", -0.70)
BRIGHT_EDGE_RESTORE_LUMA_DELTA = _envf("QS_BRIGHT_EDGE_RESTORE_LUMA_DELTA", 0.025)
EDGE_RESTORE_STEPS = int(max(1, round(_envf("QS_EDGE_RESTORE_STEPS", 4.0))))
EDGE_RESTORE_EDGE_NEAR = int(max(0, round(_envf("QS_EDGE_RESTORE_EDGE_NEAR", 2.0))))
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


# --- public entry ------------------------------------------------------------
def compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None,
        pixel_scale=1.0) -> DrawSupportResult:
    """Compute the snapped foreground ``support`` for a drawn mask.

    Returns a :class:`DrawSupportResult`. ``support`` is a binary mask that the
    caller blends/mattes via ``edge_refine._compose_refined_mask``.
    """
    mask_f = _er._as_mask(mask)
    _maybe_dump_input(guide, mask_f, radius, strength, seed_mask, draw_strokes, pixel_scale)
    hint = mask_f > 0.02
    h, w = hint.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if not np.any(hint) or maximum_flow is None:
        return DrawSupportResult(empty, empty, empty.copy(), [])

    guide = _er._prepare_guide_image(guide, (h, w))
    scales = _resolve_scales(radius, draw_strokes, hint)
    strength = float(np.clip(strength, 0.0, 100.0))

    edge_strength = _er._draw_snap_edge_strength(guide)
    if edge_strength is None:
        edge_strength = np.zeros((h, w), dtype=np.float32)
    solver_edge_strength = _solver_edge_strength(edge_strength, pixel_scale)
    g_smooth = _edge_cost_map(solver_edge_strength, strength)

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
    edge_restore_all = np.zeros((h, w), dtype=bool)

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
        restore_color_min = _edge_restore_color_min_for_unit(
            guide[sl], comp, core_roi & comp, hint[sl], component_scales.band_half_width,
            directional_bg=has_strokes)
        out = _solve_component(
            comp,
            hint[sl],
            g_smooth[sl],
            core_roi,
            erase_bg[sl],
            color_roi,
            component_scales,
        )
        color_all[sl] = np.where(out.band | comp, color_roi, color_all[sl])
        restore_color_min_all[sl] = np.where(
            out.band,
            np.minimum(restore_color_min_all[sl], restore_color_min),
            restore_color_min_all[sl],
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
        band_all,
        solver_edge_strength,
        color_all,
        hard_fg_core,
        erase_bg,
        color_min=restore_color_min_all,
    )
    support_all = _er._preserve_draw_component_separation(hint, support_all)

    debug_planes = [
        ("image_edge", edge_strength),
        ("edge_cost", g_smooth),
        ("color_score", (color_all * 0.5 + 0.5).astype(np.float32)),
        ("seed_fg", fg_seed_all),
        ("seed_bg", bg_seed_all),
        ("prior", (prior_all * 0.5 + 0.5).astype(np.float32)),
        ("cut_boundary", cut_all),
        ("edge_restore", edge_restore_all),
    ]

    hint_area = int(np.count_nonzero(hint))
    support_area = int(np.count_nonzero(support_all))
    ratio = (support_area / hint_area) if hint_area else 0.0
    logging.info(
        "[DRAW_QS] hint=%d band=%d edge_px_in_band=%d support=%d ratio=%.3f "
        "comps=%d max_flow=%d radius=%.1f",
        hint_area,
        total_band,
        int(np.count_nonzero(band_all & (solver_edge_strength >= 0.4))),
        support_area,
        ratio,
        len(solve_units),
        total_flow,
        scales.band_half_width,
    )

    return DrawSupportResult(fg_seed_all, band_all, support_all, debug_planes)


_DUMP_COUNTER = 0


def _maybe_dump_input(guide, mask, radius, strength, seed_mask, draw_strokes, pixel_scale=1.0):
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
        np.savez_compressed(
            path,
            guide=np.asarray(guide, dtype=np.float32),
            mask=np.asarray(mask, dtype=np.float32),
            seed_mask=(np.asarray(seed_mask) if seed_mask is not None else np.array([])),
            radius=np.float32(radius),
            strength=np.float32(strength),
            pixel_scale=np.float32(pixel_scale),
            strokes=np.array(strokes, dtype=object),
        )
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
            stroke_core = _seed_core(np.asarray(mask_f) * stroke_mask, center, stroke_mask)
            stroke_scales = _resolve_scales(radius, [stroke], stroke_mask)
            covered |= stroke_mask
            _append_connected_units(units, stroke_mask, stroke_core, stroke_scales)

    fallback = hint & ~covered if units else hint
    if np.any(fallback):
        fallback_scales = _resolve_scales(radius, draw_strokes, fallback)
        _append_connected_units(units, fallback, hard_fg_core & fallback, fallback_scales)

    return units


def _append_connected_units(units, component_mask, core_mask, scales):
    component_mask = np.asarray(component_mask, dtype=bool)
    if not np.any(component_mask):
        return
    n_labels, labels = cv2.connectedComponents(component_mask.astype(np.uint8), connectivity=8)
    for label_id in range(1, n_labels):
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


def _edge_cost_map(edge_strength, strength):
    # EdgeLock is used as edge sensitivity for Draw Quick Select:
    #   0   = strict; only strong ridges are attractive cut locations.
    #   100 = loose; weaker/diffuse ridges also become cheap to cut.
    # The previous range made strength=0 already permissive enough that moving
    # the slider barely changed snow/sky strokes. Widen both ridge threshold and
    # smoothness response so the UI has a visible range.
    lock = float(np.clip(strength, 0.0, 100.0)) / 100.0
    ridge_thr = 0.58 - 0.30 * lock
    falloff_sigma = 0.48 + 0.42 * lock
    sigma = 0.70 - 0.50 * lock
    edge = np.clip(edge_strength.astype(np.float32, copy=False), 0.0, 1.0)
    # The raw edge map is blurred (a few px wide), so the cheap-to-cut band is
    # wide and the boundary stops on its inner ramp ~2-3px short of the true
    # peak when reaching outward. Thin it to a 1px ridge so the only cheap cut is
    # at the edge peak -> the boundary snaps precisely onto the edge.
    edge = _thin_edge_to_ridge(edge, thr=ridge_thr, falloff_sigma=falloff_sigma)
    g = np.exp(-(edge * edge) / (sigma * sigma))
    return g.astype(np.float32, copy=False)


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
    sharp = edge * falloff
    return np.where(strong | (dist <= 2.0), sharp, edge).astype(np.float32, copy=False)


def _color_score(guide, comp, core, all_hint, R_out, directional_bg=False):
    """Signed colour data term in [-1, 1] for one component: +1 looks like this
    stroke's FG colour, -1 looks like its local background. 0 when indistinct.
    The BG shell excludes *all* strokes (``all_hint``) so other components are
    not sampled as background."""
    score = np.zeros(comp.shape, dtype=np.float32)
    if guide is None or not np.any(core):
        return score
    lab = _er._guide_to_lab(guide)
    if lab.ndim != 3:
        return score
    fg = core
    dist_out = _er._distance_from(comp)
    free = ~np.asarray(all_hint, dtype=bool)
    shell = free & (dist_out <= max(R_out, 12.0) + 4.0) & (dist_out >= 2.0)
    if not np.any(shell):
        shell = free & (dist_out <= max(R_out, 12.0) + 12.0)
    if not np.any(shell):
        return score
    fg_med = np.median(lab[fg].reshape(-1, 3), axis=0)
    bg_med = np.median(lab[shell].reshape(-1, 3), axis=0)
    if directional_bg:
        bg_med = _most_separated_shell_median(lab, shell, fg_med, bg_med)
    # Scale the colour term by how separable FG/BG are, instead of a hard cutoff.
    # Low-contrast scenes (snow: cloud vs blue sky differ by only ~6 LAB) used to
    # fall under the cutoff and get NO colour help, leaving the boundary to a
    # weak edge alone; a graceful confidence keeps a (weaker) colour signal that
    # still pulls the boundary onto the cloud/sky luminance step.
    sep = float(np.linalg.norm(fg_med - bg_med))
    conf = float(np.clip((sep - COLOR_MIN_SEP) / COLOR_SEP_SCALE, 0.0, 1.0))
    if conf <= 0.0:
        return score  # FG/BG colours genuinely indistinct
    d_fg = np.linalg.norm(lab - fg_med, axis=2)
    d_bg = np.linalg.norm(lab - bg_med, axis=2)
    score = conf * (d_bg - d_fg) / (d_bg + d_fg + 1e-3)
    return np.clip(score, -1.0, 1.0).astype(np.float32)


def _edge_restore_color_min_for_unit(guide, comp, core, all_hint, R_out, directional_bg=False):
    delta = _selected_luma_delta(
        guide, comp, core, all_hint, R_out, directional_bg=directional_bg)
    if delta > float(BRIGHT_EDGE_RESTORE_LUMA_DELTA):
        return float(BRIGHT_EDGE_RESTORE_COLOR_MIN)
    return float(EDGE_RESTORE_COLOR_MIN)


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
        bg_med = _most_separated_shell_median(rgb, shell, fg_med, bg_med)
    luma = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float(np.dot(fg_med[:3] - bg_med[:3], luma))


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


def _solve_component(comp, hint_roi, g_roi, core_roi, erase_roi, color_roi, scales) -> _ComponentSolve:
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
    mag_in = np.clip(PRIOR_FLOOR_IN + (1.0 - PRIOR_FLOOR_IN) * dist_in / max(half_w, 1.0), 0.0, 1.0)
    f_out = dist_out / max(R_out, 1.0)
    rim_ramp = np.clip((f_out - REACH_FRAC) / max(1.0 - REACH_FRAC, 1e-3), 0.0, 1.0)
    mag_out = np.clip(PRIOR_FLOOR_OUT + (1.0 - PRIOR_FLOOR_OUT) * rim_ramp, 0.0, 1.0)
    prior[inside] = mag_in[inside]
    prior[outside] = -mag_out[outside]
    seed_side = _seed_side_through_smooth_interior(comp, core, g_roi)
    opposite_inside = inside & ~seed_side
    if np.any(opposite_inside):
        prior[opposite_inside] = -np.maximum(mag_in[opposite_inside], PRIOR_FLOOR_IN)

    # Colour data term. Inside the mask it may pull either way (a same-geometry
    # spill of a *different* colour is pushed to BG and clipped). Outside the mask
    # colour may only push toward BG, never pull FG -- otherwise a large radius
    # grabs every same-coloured pixel (the grabCut explosion / busy-texture grab).
    # QS_COLOR_W_OUT>0 re-enables outward same-colour growth ("fill the sky up to
    # the edges"); default 0 keeps the conservative no-explosion behaviour.
    color_eff = color_roi.astype(np.float32, copy=True)
    if COLOR_W_OUT > 0.0:
        pos_out = outside & (color_eff > 0.0)
        color_eff[pos_out] *= (COLOR_W_OUT / max(COLOR_W, 1e-3))
    else:
        color_eff[outside] = np.minimum(color_eff[outside], 0.0)
    score = prior + COLOR_W * color_eff
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


def _seed_side_through_smooth_interior(comp, core, g_roi):
    comp = np.asarray(comp, dtype=bool)
    core = np.asarray(core, dtype=bool) & comp
    if not np.any(comp) or not np.any(core):
        return comp.copy()
    # g is smoothness: low values are cheap-to-cut image edges. Treat strong
    # edge ridges as barriers only for the inside prior; the min-cut graph still
    # decides the exact boundary.
    barrier = comp & (np.asarray(g_roi, dtype=np.float32) <= float(SIDE_G_THRESH))
    if not np.any(barrier):
        return comp.copy()
    if SIDE_DILATE > 0:
        barrier = comp & (cv2.dilate(
            barrier.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=SIDE_DILATE,
        ) > 0)
    walkable = comp & ~barrier
    seed = core & walkable
    if not np.any(seed):
        return comp.copy()
    seed_side = _er._connected_to_seed(walkable, seed)
    # Keep the core itself source-side even if a ridge overlaps it.
    seed_side |= core
    return seed_side & comp


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


def _restore_selected_edge_rim(
        support,
        candidate,
        edge_strength,
        color_score,
        hard_fg_core,
        erase_bg,
        color_min=EDGE_RESTORE_COLOR_MIN):
    support = np.asarray(support, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if not np.any(support) or not np.any(candidate):
        return support, np.zeros_like(support, dtype=bool)

    edge = np.asarray(edge_strength, dtype=np.float32)
    color = np.asarray(color_score, dtype=np.float32)
    restored = support.copy()
    restore = np.zeros_like(support, dtype=bool)
    erase = np.asarray(erase_bg, dtype=bool)
    edge_near = edge >= float(EDGE_RESTORE_THRESH)
    if EDGE_RESTORE_EDGE_NEAR > 0 and np.any(edge_near):
        edge_near = cv2.dilate(
            edge_near.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=EDGE_RESTORE_EDGE_NEAR,
        ) > 0
    edge_band = (
        candidate
        & edge_near
        & ~erase
    )
    color_min_arr = np.asarray(color_min, dtype=np.float32)
    if color_min_arr.shape:
        edge_band &= color >= color_min_arr
    else:
        edge_band &= color >= float(color_min_arr)
    steps = int(max(1, EDGE_RESTORE_STEPS))
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
