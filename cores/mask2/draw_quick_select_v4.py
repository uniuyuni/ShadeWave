"""Draw Quick Select V4.

V4 keeps V3's per-stroke region solve and replaces the image edge's role: it is
no longer a *wall* inside the min-cut, it becomes a *trace target*. V3 decides
the region (colour / prior); V4 then re-traces the region boundary along the
strongest nearby image edges with a global edge-following pass (a livewire-style
shortest contour), so the boundary lands on the real silhouette even where it
runs many pixels away from V3's smooth cut.

The trace is a dynamic-programming shortest path on the boundary "ribbon": the V3
boundary is sampled into an ordered closed contour, a perpendicular window of
+-W pixels is laid down at each contour point, and DP finds the offset sequence
that hugs strong (coherent) edges while staying smooth and close to the
reference. Smoothness + a distance prior keep it from jumping onto disconnected
texture clutter (foliage / roof tiles) or drifting where there is no edge.

Default OFF (opt-in via ``QS_V4_EDGE_SNAP=1``) until validated against real
hand-drawn ground truth; default keeps V4 == V3 everywhere. See
``docs/draw-quick-select-v4-design.md``.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from cores.mask2 import draw_quick_select as _v1
from cores.mask2 import draw_quick_select_v3 as _v3
from cores.mask2 import edge_refine as _er


DrawSupportResult = _v1.DrawSupportResult

_EDGE_SNAP_DEFAULT = False

# Coherence 1-slot cache: structure tensor is expensive (~20ms) and only
# depends on the guide image, so cache it alongside the edge.
_GUIDE_COH_CACHE: dict = {}  # {(guide_fingerprint, shape): coh}


def _snap_enabled():
    value = os.environ.get("QS_V4_EDGE_SNAP")
    if value is None:
        return bool(_EDGE_SNAP_DEFAULT)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None,
        pixel_scale=1.0,
        edge_bias=0.0) -> DrawSupportResult:
    base = _v3.compute_draw_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=seed_mask,
        draw_strokes=draw_strokes,
        pixel_scale=pixel_scale,
        edge_bias=edge_bias,
    )
    support = np.asarray(base.support, dtype=bool)
    if support.size == 0 or not np.any(support) or not _snap_enabled():
        return base

    planes = {name: value for name, value in base.debug_planes}
    # Derive the trace edge straight from the guide image, NOT from the V3
    # "context_edge" debug plane. That plane is accumulated per add stroke and
    # weighted by each stroke's candidate region, so an *unrelated* stroke (e.g. an
    # erase far from this add) perturbs it even when the V3 support is byte
    # identical -- which used to shift the trace tens of pixels near the add
    # (breaking stroke independence: "erasing elsewhere moved my selection"). The
    # guide is stroke independent, so the trace edge is too. The result is cached
    # per guide buffer across strokes (~40ms on 1M-pixel images).
    # Perceptual edge for the trace: the app feeds a linear scene-referred guide,
    # so on a deep-shadow subject the linear gradient is dominated by highlights
    # and the boundary's shadow side reads ~0 -- the trace then snaps to some of
    # the silhouette and ignores the rest (the "Edge Lock only moves part of the
    # edge" symptom). Re-encoding to a perceptual space before the edge map gives
    # the DP a consistent signal all the way round. Scoped to the V4 trace only;
    # the V3 region solver and trim keep the validated linear edge.
    edge = _v3._get_guide_edge(guide, support.shape, perceptual=True)
    if edge is None:
        edge = planes.get("context_edge")
    if edge is None:
        return base
    edge = np.clip(np.asarray(edge, dtype=np.float32), 0.0, 1.0)
    if edge.shape != support.shape:
        return base

    # Coherence gate: prefer oriented edges (silhouettes) over isotropic texture.
    # Cached per guide buffer; structure tensor is ~20ms on 1M-pixel images.
    coh = _get_guide_coh(guide, support.shape)
    trace_edge = edge * coh if (coh is not None and coh.shape == edge.shape) else edge

    seed = np.asarray(base.seed, dtype=bool)
    # Snap strength rides on existing controls: Edge Lock -> how hard to snap
    # (dist_prior), Quick Radius -> how far to look (band). No new slider. The
    # Edge Lock is read *locally per contour point* from the edge_lock_effective
    # plane, so one stroke's strength never bleeds onto another stroke's
    # boundary (preserves per-stroke independence for add/erase correction).
    lock_plane = planes.get("edge_lock_effective")
    snapped, _delta = _trace_boundary_to_edges(
        support, trace_edge, seed, _band_w(radius), lock_plane)
    # Keep the trace a *local* edit: only change near the painted/erased footprint
    # and never grow outward past Quick Radius. So Radius=0 never affects pixels
    # outside the brush, and an erase only moves boundary near where it was drawn.
    snapped = _localize_to_brush(snapped, support, mask, draw_strokes, radius)
    # The DP trace fills a polygon, which can re-add the 1px brush-circle rim that
    # V3 trimmed. Re-trim it here so the final boundary never traces the brush
    # footprint past an object edge. The trim keys off the *linear* edge (same as
    # V3's trim): the perceptual trace edge would mark deep-shadow texture as a
    # real edge and spare background arcs the trim is meant to drop. Cached, so
    # this is the region solver's edge (a cache hit, no extra recompute).
    edge_linear = _v3._get_guide_edge(guide, support.shape, perceptual=False)
    snapped = _v3._trim_offedge_brush_rim(
        snapped, mask, guide, _edge=edge_linear if edge_linear is not None else edge)
    delta = snapped ^ support
    if not np.any(delta):
        return base

    out_planes = list(base.debug_planes)
    out_planes.append(("v4_boundary_snap_delta", delta))
    logging.info("[DRAW_QS_V4] edge-trace moved=%d px", int(np.count_nonzero(delta)))
    return DrawSupportResult(base.seed, base.candidate, snapped, out_planes)


def _localize_to_brush(snapped, support, mask, draw_strokes, radius):
    """Confine the trace edit to the painted/erased footprint (+Quick Radius).

    Two interactive-editing invariants the global trace would otherwise break:
      * Radius=0 must not affect anything outside the brush -> outward growth is
        clipped to ``dilate(footprint, radius)`` (none at Radius=0; inward snap to
        edges inside the brush is still allowed).
      * a stroke must only move boundary where it was drawn -> changes outside
        ``dilate(footprint, radius+2)`` are reverted to the V3 support.
    """
    support = np.asarray(support, dtype=bool)
    snapped = np.asarray(snapped, dtype=bool)
    hint = _er._as_mask(mask) > 0.02
    footprint = hint | _v3._erase_stroke_mask(support.shape, draw_strokes)
    if not np.any(footprint):
        return support
    try:
        rpx = max(0, int(round(float(radius))))
    except Exception:
        rpx = 0
    k3 = np.ones((3, 3), dtype=np.uint8)
    acted = cv2.dilate(footprint.astype(np.uint8), k3, iterations=rpx + 2) > 0
    result = np.where(acted, snapped, support)
    # forbid outward growth beyond Quick Radius
    grow_ok = (cv2.dilate(footprint.astype(np.uint8), k3, iterations=rpx) > 0
               if rpx > 0 else footprint)
    grown = result & ~support
    result = result & ~(grown & ~grow_ok)
    return result


def _band_w(radius):
    try:
        ui = float(radius)
    except Exception:
        ui = 0.0
    base = float(_v1._envf("QS_V4_SNAP_BAND", 32.0))
    return int(np.clip(round(base + max(0.0, ui)), 8, 48))


def _edge_coherence(guide, shape):
    """Structure-tensor coherence in [0,1]: ~1 on oriented edges, ~0 on texture."""
    g = _er._prepare_guide_image(guide, shape)
    if g is None:
        return None
    # Match _draw_snap_edge_strength: re-encode the linear guide to a perceptual
    # space so the structure tensor is not dominated by highlights (see
    # edge_refine._to_perceptual_guide).
    g = _er._to_perceptual_guide(g)
    gray = cv2.cvtColor(g[..., :3], cv2.COLOR_RGB2GRAY) if g.ndim == 3 else g
    gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    win = (9, 9)
    jxx = cv2.blur(gx * gx, win)
    jyy = cv2.blur(gy * gy, win)
    jxy = cv2.blur(gx * gy, win)
    trace = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (trace + 1e-6)
    return np.clip(coh, 0.0, 1.0).astype(np.float32, copy=False)


def _get_guide_coh(guide, shape):
    """Return cached coherence for *guide* at *shape*, computing once per guide."""
    try:
        key = (_er._guide_fingerprint(guide), shape)
    except AttributeError:
        key = None
    if key is not None and key in _GUIDE_COH_CACHE:
        return _GUIDE_COH_CACHE[key]
    coh = _edge_coherence(guide, shape)
    if key is not None:
        _GUIDE_COH_CACHE.clear()  # keep only 1 entry
        _GUIDE_COH_CACHE[key] = coh
    return coh


def _bilinear(img, px, py):
    h, w = img.shape
    px = np.clip(px, 0.0, w - 1.0)
    py = np.clip(py, 0.0, h - 1.0)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = px - x0
    wy = py - y0
    return (img[y0, x0] * (1 - wx) * (1 - wy)
            + img[y0, x1] * wx * (1 - wy)
            + img[y1, x0] * (1 - wx) * wy
            + img[y1, x1] * wx * wy)


def _lock_to_dist_prior(lock):
    """Edge Lock (0..100) -> distance prior. Low lock => high prior (stay on the
    brush, very light snap); high lock => lower prior (snap harder).

    The floor is *moderate* (~0.55), not aggressive: a measured sweep against the
    hand-drawn GT showed a strong snap (prior <= ~0.2) drifts the boundary onto
    texture clutter and makes every case worse (simple 0.83 -> 0.65 b_f1, roof
    0.51 -> 0.37). The useful band is prior in [~0.55, 1.05]: light snap is best
    on clean edges (simple/lowcontrast), moderate snap helps same-colour edges
    (roof). So Edge Lock spans light..moderate, never the destructive over-snap
    that made the top of the slider feel like it "went wrong"."""
    return np.clip(1.05 - 0.0045 * np.asarray(lock, dtype=np.float32), 0.55, 1.05)


def _trace_one_contour(cnt, edge, lock_plane, W, smooth):
    """DP ribbon trace for one ordered contour. Returns new boundary points, or
    None to keep the contour unchanged (too short).

    The former ``e.max() < 0.30`` early-exit is intentionally removed: when no
    strong edge exists in the ribbon the DP cost function naturally keeps offset=0
    for every point (dist_prior makes any other offset more expensive than staying
    put), so the boundary doesn't move. Removing the guard makes Edge Lock affect
    the *whole* contour uniformly -- the old guard caused the slider to visibly
    "stop working" on parts of the boundary that happened to have weaker edges.
    """
    if len(cnt) < 16:
        return None
    step = max(1, len(cnt) // 480)
    cnt = cnt[::step]
    n = len(cnt)
    nxt = np.roll(cnt, -1, axis=0)
    prv = np.roll(cnt, 1, axis=0)
    tang = nxt - prv
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    offsets = np.arange(-W, W + 1)
    px = cnt[:, 0][:, None] + offsets[None, :] * normal[:, 0][:, None]
    py = cnt[:, 1][:, None] + offsets[None, :] * normal[:, 1][:, None]
    e = _bilinear(edge, px, py)

    # Per-contour-point Edge Lock read locally from the plane => each stroke's
    # boundary uses its own snap strength (no cross-stroke bleed).
    if lock_plane is not None:
        h, w = lock_plane.shape
        lx = np.clip(np.round(cnt[:, 0]).astype(np.int64), 0, w - 1)
        ly = np.clip(np.round(cnt[:, 1]).astype(np.int64), 0, h - 1)
        lock = np.asarray(lock_plane, dtype=np.float32)[ly, lx] * 100.0
    else:
        lock = np.full(n, 60.0, dtype=np.float32)
    override = os.environ.get("QS_V4_TRACE_DISTPRIOR")
    if override is not None:
        dpri = np.full(n, float(override), dtype=np.float32)
    else:
        dpri = _lock_to_dist_prior(lock)
    cost = (1.0 - e) + dpri[:, None] * (np.abs(offsets)[None, :].astype(np.float32) / max(W, 1))

    k = 2
    inf = 1e9
    m = 2 * W + 1
    dp = cost[0].astype(np.float64).copy()
    back = np.zeros((n, m), dtype=np.int32)
    # Pre-allocate loop temporaries once (avoids 480 × 7 np.full/zeros calls).
    _d_vals = np.arange(-k, k + 1, dtype=np.int32)   # [-2,-1,0,1,2]
    _pens = smooth * np.abs(_d_vals).astype(np.float64)
    _shifted = np.empty((2 * k + 1, m), dtype=np.float64)
    _idx = np.arange(m, dtype=np.int32)
    for i in range(1, n):
        _shifted.fill(inf)
        for ki, (d, pen) in enumerate(zip(_d_vals, _pens)):
            if d >= 0:
                _shifted[ki, d:] = dp[:m - d] + pen
            else:
                _shifted[ki, :d] = dp[-d:] + pen
        best_ki = np.argmin(_shifted, axis=0)   # (m,) index into _d_vals
        dp = cost[i] + _shifted[best_ki, _idx]
        back[i] = _idx - _d_vals[best_ki]       # source index = dest - d

    o = int(np.argmin(dp))
    path = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = offsets[o]
        o = int(back[i][o]) if i > 0 else o
    return cnt + path[:, None] * normal


def _trace_boundary_to_edges(support, edge, seed, band_w, lock_plane):
    """Re-trace each support contour onto nearby strong edges (DP ribbon path).

    Every contour is traced *independently* with its own locally-read Edge Lock,
    so adding/removing one stroke does not move another stroke's boundary. Only
    pixels within +-band_w of the original boundary change.
    """
    support = np.asarray(support, dtype=bool)
    empty = np.zeros_like(support)
    contours, _ = cv2.findContours(
        support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return support, empty
    W = int(band_w)
    smooth = float(_v1._envf("QS_V4_TRACE_SMOOTH", 0.12))
    filled = np.zeros_like(support, dtype=np.uint8)
    any_change = False
    for c in contours:
        cnt = c.reshape(-1, 2).astype(np.float32)
        new_pts = _trace_one_contour(cnt, edge, lock_plane, W, smooth)
        if new_pts is None:
            cv2.fillPoly(filled, [np.round(cnt).astype(np.int32)], 1)
        else:
            cv2.fillPoly(filled, [np.round(new_pts).astype(np.int32)], 1)
            any_change = True
    if not any_change:
        return support, empty
    new_support = filled.astype(bool)
    ribbon = cv2.dilate(
        (cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_GRADIENT,
                          np.ones((3, 3), np.uint8)) > 0).astype(np.uint8),
        np.ones((3, 3), np.uint8), iterations=W + 1) > 0
    result = np.where(ribbon, new_support, support)
    if np.any(seed):
        result = _er._connected_to_seed(result | seed, seed) | seed
    delta = result ^ support
    return result, delta


__all__ = ["DrawSupportResult", "compute_draw_support"]
