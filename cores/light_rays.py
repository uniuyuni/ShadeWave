"""Procedural volumetric light rays for fog, forest beams, and window light.

The functions in this module are intentionally UI-free.  Callers pass guide
coordinates in the current image pixel space; Kivy/TCG conversion stays in the
effect or editor layer.
"""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


_EPS = 1e-6

# Previews (resolution_scale < 1.0) larger than this on their longest side have
# their volumetric overlay rendered smaller and upsampled.  Exports always run
# at native resolution, so their output is unaffected by this cap.
_PREVIEW_MAX_RENDER_PX = 960


def apply_light_rays(
    img_rgb: np.ndarray,
    guides: Iterable[dict],
    *,
    intensity: float = 50.0,
    length: float = 65.0,
    decay: float = 55.0,
    width: float = 45.0,
    softness: float = 65.0,
    edge_bias: float = 0.0,
    spread: float = 35.0,
    count: float = 6.0,
    density: float = 45.0,
    variation: float = 35.0,
    fog: float = 35.0,
    occlusion: float = 55.0,
    color_rgb=(1.0, 0.95, 0.82),
    seed: int = 0,
    resolution_scale: float = 1.0,
    scene_size_px=None,
) -> np.ndarray:
    """Return ``img_rgb`` plus HDR light-ray overlay.

    Supported guide dictionaries:

    - ``{"type": "line", "mode": "parallel", "p1": (x, y), "p2": (x, y)}``
    - ``{"type": "line", "mode": "directional", "p1": ..., "p2": ...}``
    - ``{"type": "point", "mode": "radial", "p": (x, y)}``
    - ``{"type": "point", "mode": "directional", "p": ..., "p2": ...}``

    The result is float32 and is not clipped.  This is important for the HDR
    pipeline: downstream tone mapping/export should decide how over-range light
    is handled.

    ``resolution_scale`` only acts as a quality/speed hint.  For full renders
    (``>= 1.0``, i.e. export) the overlay is always computed at native
    resolution, so the output is independent of this value.  For interactive
    previews (``< 1.0``) a large overlay may be rendered at reduced resolution
    and upsampled; the base image keeps full detail and only the smooth glow is
    approximated.  Geometry stays in current pixel space regardless.
    """

    image = np.ascontiguousarray(img_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    amount = float(intensity) / 100.0
    if amount <= 0.0:
        return image.copy()

    norm_guides = _normalize_guides(guides)
    if not norm_guides:
        return image.copy()

    h, w = image.shape[:2]
    params = dict(
        amount=amount,
        length=length,
        decay=decay,
        width=width,
        softness=softness,
        edge_bias=edge_bias,
        spread=spread,
        count=count,
        density=density,
        variation=variation,
        fog=fog,
        occlusion=occlusion,
        color_rgb=color_rgb,
        seed=seed,
    )

    render_scale = _preview_render_scale(resolution_scale, h, w)
    if render_scale < 1.0:
        # Preview only: volumetric rays are low frequency, so the additive
        # overlay is rendered at a reduced resolution and upsampled.  The base
        # image keeps full detail; only the smooth glow is approximated.  At
        # export (resolution_scale >= 1.0) this branch is skipped, so the
        # full-resolution render stays byte-for-byte unchanged.
        h2 = max(1, int(round(h * render_scale)))
        w2 = max(1, int(round(w * render_scale)))
        small = cv2.resize(image, (w2, h2), interpolation=cv2.INTER_AREA)
        add_small = _compute_additive(
            small,
            _scale_guides(norm_guides, render_scale),
            scene_size_px=_scale_scene(scene_size_px, render_scale),
            **params,
        )
        if add_small is None:
            return image.copy()
        add = cv2.resize(add_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return (image + add).astype(np.float32, copy=False)

    add = _compute_additive(image, norm_guides, scene_size_px=scene_size_px, **params)
    if add is None:
        return image.copy()
    return (image + add).astype(np.float32, copy=False)


def _compute_additive(
    image,
    norm_guides,
    *,
    amount,
    length,
    decay,
    width,
    softness,
    edge_bias,
    spread,
    count,
    density,
    variation,
    fog,
    occlusion,
    color_rgb,
    seed,
    scene_size_px,
):
    """Return the additive HDR ray overlay (h, w, 3), or ``None`` if empty."""

    h, w = image.shape[:2]
    scene_w, scene_h = _scene_size_px(scene_size_px, w, h)
    scene = max(1.0, float(min(scene_w, scene_h)))
    diagonal = max(1.0, float(np.hypot(scene_w, scene_h)))
    length_n = float(np.clip(float(length) / 100.0, 0.0, 2.0))
    length_px = max(4.0, length_n * diagonal)
    width_n = float(np.clip(float(width) / 100.0, 0.0, 2.0))
    # Volumetric rays should read as shafts of lit air, not as drawn strokes.
    # The slider therefore maps to a broad scene-scale aperture.
    width_px = max(2.0, (0.018 + 0.300 * (width_n ** 1.15)) * scene)
    decay_px = length_px * (0.20 + 1.80 * (float(decay) / 100.0))
    spread_deg = 8.0 + 84.0 * (float(spread) / 100.0)

    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    overlay = np.zeros((h, w), dtype=np.float32)
    for index, guide in enumerate(norm_guides[:16]):
        guide_seed = int(seed) + index * 1009
        if guide["type"] == "line":
            p1 = guide["p1"]
            p2 = guide["p2"]
            direction = _unit(np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32))
            if direction is None:
                continue
            mode = guide.get("mode", "parallel")
            if mode == "directional":
                render_origin = np.array(p1, dtype=np.float32)
                line_reach_px = max(4.0, float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
                line_decay_px = line_reach_px * (0.20 + 1.80 * (float(decay) / 100.0))
                layer = _directional_ray(
                    xx,
                    yy,
                    render_origin,
                    direction,
                    width_px,
                    line_decay_px,
                    line_reach_px,
                    spread_deg,
                    count,
                    density,
                    variation,
                    softness,
                    edge_bias,
                    guide_seed,
                )
            else:
                render_origin = _line_start_outside_image(
                    np.array(p1, dtype=np.float32),
                    np.array(p2, dtype=np.float32),
                    direction,
                    w,
                    h,
                )
                line_reach_px = max(4.0, float(np.hypot(p2[0] - render_origin[0], p2[1] - render_origin[1])))
                line_decay_px = line_reach_px * (0.20 + 1.80 * (float(decay) / 100.0))
                layer = _parallel_ray(
                    xx,
                    yy,
                    render_origin,
                    direction,
                    width_px,
                    line_decay_px,
                    line_reach_px,
                    count,
                    density,
                    variation,
                    softness,
                    edge_bias,
                    guide_seed,
                )
            layer *= _cast_shadow_map(image, direction, line_reach_px, occlusion)
        else:
            origin = np.array(guide["p"], dtype=np.float32)
            mode = guide.get("mode", "radial")
            if mode == "directional":
                p2 = guide.get("p2")
                if p2 is None:
                    angle = math.radians(float(guide.get("angle", 0.0)))
                    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                else:
                    direction = _unit(np.array([p2[0] - origin[0], p2[1] - origin[1]], dtype=np.float32))
                    if direction is None:
                        continue
                layer = _point_directional_ray(
                    xx,
                    yy,
                    origin,
                    direction,
                    width_px,
                    decay_px,
                    length_px,
                    spread_deg,
                    count,
                    density,
                    variation,
                    softness,
                    edge_bias,
                    guide_seed,
                )
                layer *= _cast_shadow_map(image, direction, length_px, occlusion)
            else:
                p2 = _point(guide.get("p2"))
                proj_len_px = _length_px_from_slider(guide.get("projection_length", length), diagonal)
                # Crossfade classic <-> projected radial by how far the direction
                # handle has been pulled out.  A hard switch made the short
                # projection (where the projected solver degenerates into a muddy,
                # dim, omni-directional burst that lacks the classic centre fill)
                # snap to a much brighter classic radial.  Blending over a
                # scale-relative window keeps the start brightness continuous and
                # lets the clean classic centre dominate while the handle is short.
                lo = max(2.0, float(width_px) * 0.04)
                hi = max(lo * 2.0, proj_len_px * 0.16)
                if p2 is None:
                    w_proj = 0.0
                else:
                    axis_len_raw = float(np.hypot(p2[0] - origin[0], p2[1] - origin[1]))
                    t = float(np.clip((axis_len_raw - lo) / max(hi - lo, 1e-3), 0.0, 1.0))
                    w_proj = t * t * (3.0 - 2.0 * t)

                def _classic():
                    return _radial_ray(
                        xx, yy, origin, width_n, width_px, decay_px, length_px,
                        count, density, variation, softness, edge_bias, guide_seed,
                    )

                def _projected():
                    return _projected_radial_ray(
                        xx, yy, origin, np.array(p2, dtype=np.float32),
                        width_n, width_px, decay_px, length_px,
                        proj_len_px, proj_len_px * (0.20 + 1.80 * (float(decay) / 100.0)),
                        count, density, variation, softness, edge_bias, spread_deg, guide_seed,
                    )

                if w_proj <= 0.0:
                    layer = _classic()
                elif w_proj >= 1.0:
                    layer = _projected()
                else:
                    layer = (
                        _classic() * np.float32(1.0 - w_proj)
                        + _projected() * np.float32(w_proj)
                    )
        layer = _harden_low_softness(layer, softness)
        overlay += layer.astype(np.float32, copy=False)

    if np.max(overlay, initial=0.0) <= 0.0:
        return None

    overlay *= _occlusion_map(image, occlusion)
    fog_gain = 0.55 + 0.90 * (float(fog) / 100.0)
    color = np.asarray(color_rgb, dtype=np.float32)
    if color.size < 3:
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    color = np.clip(color[:3], 0.0, None)
    if float(color.max(initial=0.0)) <= 0.0:
        color[:] = 1.0
    color = color / max(float(color.max(initial=1.0)), _EPS)

    rays = overlay[..., np.newaxis] * color.reshape(1, 1, 3)
    return (rays * np.float32(amount * fog_gain)).astype(np.float32, copy=False)


def _preview_render_scale(resolution_scale, h, w):
    # Export / full renders (resolution_scale >= 1.0) always render at native
    # resolution, so their output is byte-for-byte identical to before.  Only
    # interactive previews (resolution_scale < 1.0) larger than the cap are
    # rendered smaller and upsampled.
    if float(resolution_scale) >= 1.0:
        return 1.0
    longest = max(int(h), int(w))
    if longest <= _PREVIEW_MAX_RENDER_PX:
        return 1.0
    return max(_PREVIEW_MAX_RENDER_PX / float(longest), 0.25)


def _scale_guides(norm_guides, f):
    scaled = []
    for guide in norm_guides:
        item = dict(guide)
        if guide.get("type") == "line":
            item["p1"] = (guide["p1"][0] * f, guide["p1"][1] * f)
            item["p2"] = (guide["p2"][0] * f, guide["p2"][1] * f)
        else:
            item["p"] = (guide["p"][0] * f, guide["p"][1] * f)
            if "p2" in guide:
                item["p2"] = (guide["p2"][0] * f, guide["p2"][1] * f)
        scaled.append(item)
    return scaled


def _scale_scene(scene_size_px, f):
    if scene_size_px is None:
        return None
    try:
        return (float(scene_size_px[0]) * f, float(scene_size_px[1]) * f)
    except Exception:
        return None


def _normalize_guides(guides: Iterable[dict]) -> list[dict]:
    out = []
    for guide in guides or []:
        if not isinstance(guide, dict):
            continue
        gtype = str(guide.get("type", "")).lower()
        mode = str(guide.get("mode", "")).lower()
        if gtype == "line":
            p1 = _point(guide.get("p1"))
            p2 = _point(guide.get("p2"))
            if p1 is None or p2 is None:
                continue
            out.append({"type": "line", "mode": mode or "parallel", "p1": p1, "p2": p2})
        elif gtype == "point":
            p = _point(guide.get("p"))
            if p is None:
                continue
            item = {"type": "point", "mode": mode or "radial", "p": p}
            p2 = _point(guide.get("p2"))
            if p2 is not None:
                item["p2"] = p2
            if "projection_length" in guide:
                try:
                    item["projection_length"] = float(guide["projection_length"])
                except Exception:
                    pass
            if "angle" in guide:
                item["angle"] = float(guide["angle"])
            out.append(item)
    return out


def _scene_size_px(scene_size_px, fallback_w, fallback_h):
    try:
        scene_w = float(scene_size_px[0])
        scene_h = float(scene_size_px[1])
    except Exception:
        scene_w = float(fallback_w)
        scene_h = float(fallback_h)
    if scene_w <= 0.0 or scene_h <= 0.0:
        scene_w = float(fallback_w)
        scene_h = float(fallback_h)
    return scene_w, scene_h


def _length_px_from_slider(length, diagonal):
    length_n = float(np.clip(float(length) / 100.0, 0.0, 2.0))
    return max(4.0, length_n * max(1.0, float(diagonal)))


def _point(value):
    try:
        return (float(value[0]), float(value[1]))
    except Exception:
        return None


def _unit(v: np.ndarray):
    n = float(np.hypot(float(v[0]), float(v[1])))
    if n < _EPS:
        return None
    return (v / np.float32(n)).astype(np.float32)


def _line_start_outside_image(p1, p2, direction, w, h):
    if p1[0] < 0.0 or p1[0] >= float(w) or p1[1] < 0.0 or p1[1] >= float(h):
        return p1.astype(np.float32, copy=False)
    margin = max(2.0, float(min(w, h)) * 0.03)
    candidates = []
    bounds = (
        (0, -margin),
        (0, float(w - 1) + margin),
        (1, -margin),
        (1, float(h - 1) + margin),
    )
    for axis, boundary in bounds:
        d = float(direction[axis])
        if abs(d) <= _EPS:
            continue
        t = (float(p1[axis]) - boundary) / d
        if t > 0.0:
            point = p1 - direction * np.float32(t + 1.0)
            if (
                point[0] < -margin * 0.5
                or point[0] > float(w - 1) + margin * 0.5
                or point[1] < -margin * 0.5
                or point[1] > float(h - 1) + margin * 0.5
            ):
                candidates.append((t, point.astype(np.float32, copy=False)))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    diagonal = float(np.hypot(w, h))
    return (p1 - direction * np.float32(diagonal + margin)).astype(np.float32, copy=False)


def _parallel_ray(
    xx,
    yy,
    origin,
    direction,
    width_px,
    decay_px,
    length_px,
    count,
    density,
    variation,
    softness,
    edge_bias,
    seed,
):
    dx = xx - np.float32(origin[0])
    dy = yy - np.float32(origin[1])
    along = dx * direction[0] + dy * direction[1]
    perp = -dx * direction[1] + dy * direction[0]
    soft = float(softness) / 100.0
    beam_width = width_px * (2.2 + 4.4 * soft)
    band = _edge_harden(_asymmetric_gaussian(perp, beam_width, edge_bias), softness)
    frame_span = float(np.max(np.abs(perp), initial=0.0))
    shafts = _parallel_shaft_field(
        perp, along, width_px, beam_width, frame_span, count, density, variation, softness, edge_bias, seed)
    band_gain = 0.24 - 0.10 * float(np.clip(float(count) / 32.0, 0.0, 1.0))
    falloff = _distance_falloff(along, decay_px)
    feather = max(float(width_px) * (1.5 + 2.0 * soft), float(length_px) * 0.04, 2.0)
    start = np.clip(along / np.float32(feather), 0.0, 1.0)
    end = _end_tail(along, length_px, feather)
    texture = _ray_texture(perp, along, width_px, density, variation, seed)
    return (np.float32(band_gain) * band + shafts) * texture * falloff * start * end


def _directional_ray(
    xx,
    yy,
    origin,
    direction,
    width_px,
    decay_px,
    length_px,
    spread_deg,
    count,
    density,
    variation,
    softness,
    edge_bias,
    seed,
):
    dx = xx - np.float32(origin[0])
    dy = yy - np.float32(origin[1])
    along = dx * direction[0] + dy * direction[1]
    perp = -dx * direction[1] + dy * direction[0]
    spread = math.tan(math.radians(spread_deg) * 0.5)
    soft = float(softness) / 100.0
    ahead = np.maximum(along, 0.0)
    cone = (
        width_px * np.float32(1.4 + 2.8 * soft)
        + ahead * np.float32(spread)
    )
    band = _edge_harden(_asymmetric_gaussian(perp, cone * np.float32(1.20), edge_bias), softness)
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    start_stagger_px = max(float(width_px) * (0.85 + 1.30 * soft), float(length_px) * 0.14, 1.0) * variation_n
    shafts = _directional_shaft_field(
        perp, along, cone, count, density, variation, softness, edge_bias, seed, start_stagger_px)
    band_gain = 0.20 - 0.08 * float(np.clip(float(count) / 32.0, 0.0, 1.0))
    fade = _distance_falloff(along, decay_px)
    start_feather = max(float(width_px) * (0.90 + 1.40 * soft), float(length_px) * 0.075, 6.0)
    start = np.clip(along / np.float32(start_feather), 0.0, 1.0)
    start = (start * start * (np.float32(3.0) - np.float32(2.0) * start)).astype(np.float32, copy=False)
    end_feather = max(float(width_px) * (1.2 + 1.8 * soft), float(length_px) * 0.04, 2.0)
    end = _end_tail(along, length_px, end_feather)
    texture = _ray_texture(perp, along, width_px, density, variation, seed)
    return (np.float32(band_gain) * band + shafts) * texture * fade * start * end


def _point_directional_ray(
    xx,
    yy,
    origin,
    direction,
    width_px,
    decay_px,
    length_px,
    spread_deg,
    count,
    density,
    variation,
    softness,
    edge_bias,
    seed,
):
    dx = xx - np.float32(origin[0])
    dy = yy - np.float32(origin[1])
    along = dx * direction[0] + dy * direction[1]
    perp = -dx * direction[1] + dy * direction[0]
    spread = math.tan(math.radians(spread_deg) * 0.5)
    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    source_radius = max(min(float(length_px) * 0.012, 6.0), 1.8)
    ahead = np.maximum(along, 0.0)
    width_spread = np.float32(
        float(width_px) / max(float(length_px) * 0.72, 1.0) * (0.38 + 0.95 * soft))
    cone = (
        np.float32(source_radius * (0.95 + 0.35 * soft))
        + ahead * (np.float32(spread) + width_spread)
    )
    band = _edge_harden(_asymmetric_gaussian(perp, cone * np.float32(1.10), edge_bias), softness)
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    start_stagger_px = max(float(width_px) * (0.55 + 1.10 * soft), float(length_px) * 0.10, 1.0) * variation_n
    shafts = _directional_shaft_field(
        perp, along, cone, count, density, variation, softness, edge_bias, seed + 0x1357, start_stagger_px)
    texture = _ray_texture(perp, along, width_px, density, variation, seed + 0x2468)
    fade = _distance_falloff(along, decay_px * 0.84)
    gate_width = np.float32(max(source_radius * (1.15 + 0.85 * soft), 1.0))
    forward_gate = np.clip((along + gate_width * np.float32(0.20)) / gate_width, 0.0, 1.0)
    forward_gate = (forward_gate * forward_gate * (np.float32(3.0) - np.float32(2.0) * forward_gate)).astype(np.float32, copy=False)
    end = _end_tail(along, length_px, max(float(width_px) * (1.8 + 2.4 * soft), float(length_px) * 0.05, 2.0))
    body = (np.float32(0.15) * band + shafts) * forward_gate
    return body * texture * fade * end


def _radial_ray(xx, yy, origin, width_n, width_px, decay_px, length_px, count, density, variation, softness, edge_bias, seed):
    dx = xx - np.float32(origin[0])
    dy = yy - np.float32(origin[1])
    r = np.sqrt(dx * dx + dy * dy) + np.float32(1e-3)
    theta = np.arctan2(dy, dx)
    count_i = float(np.clip(round(float(count)), 1, 32))
    density_n = float(np.clip(float(density) / 100.0, 0.0, 1.0))
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    rng = np.random.default_rng((int(seed) + 0x52A3) & 0xFFFFFFFF)
    count_i_int = int(count_i)
    n = max(1.0, count_i)
    phase = np.float32((int(seed) % 997) / 997.0 * 2.0 * math.pi)
    spacing = np.float32(2.0 * math.pi / max(n, 1.0))
    width_ratio = float(np.clip(float(width_n) / 2.0, 0.0, 1.0))
    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    base_sigma = spacing * np.float32(0.014 + 0.42 * width_ratio + 0.060 * soft)
    root_width_px = np.float32(max(float(width_px) * (0.18 + 0.62 * width_ratio + 0.12 * soft), 1.5))
    detail_rng = np.random.default_rng((int(seed) + 0xB4C3) & 0xFFFFFFFF)
    angle_jitter_scale = 0.04 * min(1.0, max(0.0, n - 4.0) / 4.0)
    rays = np.zeros_like(theta, dtype=np.float32)
    for i in range(count_i_int):
        center_jitter = np.float32(detail_rng.uniform(-angle_jitter_scale, angle_jitter_scale) * float(spacing) * variation_n)
        center = phase + np.float32(i) * spacing + center_jitter
        ray_cos = np.float32(math.cos(float(center)))
        ray_sin = np.float32(math.sin(float(center)))
        along_ray = dx * ray_cos + dy * ray_sin
        cross_ray = -dx * ray_sin + dy * ray_cos
        width_jitter = 1.0 + float(rng.uniform(-0.56, 0.72)) * variation_n + float(rng.uniform(-0.20, 0.20)) * density_n
        amp = 1.0 + float(rng.uniform(-0.40, 0.30)) * variation_n
        start_offset = float(rng.uniform(0.0, max(float(root_width_px) * 2.2, float(length_px) * 0.12))) * variation_n
        ray_phase = np.float32(detail_rng.uniform(0.0, 2.0 * math.pi))
        angular_sigma = max(float(base_sigma) * max(width_jitter, 0.35), 1e-3)
        width_along = root_width_px + np.maximum(along_ray, 0.0) * np.float32(angular_sigma)
        ray = _asymmetric_gaussian(cross_ray, width_along, edge_bias)
        ray *= _rounded_ray_start_gate(
            along_ray,
            cross_ray,
            start_offset,
            root_width_px,
            width_along,
            length_px,
            softness,
            variation,
        )
        ray *= _single_ray_texture(along_ray, cross_ray, width_px, length_px, density_n, variation_n, ray_phase)
        bias = float(np.clip(float(edge_bias) / 100.0, -1.0, 1.0))
        if abs(bias) > 1e-4:
            delta = np.arctan2(np.sin(theta - center), np.cos(theta - center))
            hard_side = (delta * np.float32(bias)) > 0.0
            hard = np.power(np.clip(ray, 0.0, 1.0), np.float32(1.0 + 4.2 * abs(bias)))
            soft_side = np.power(np.clip(ray, 0.0, 1.0), np.float32(max(0.42, 1.0 - 0.42 * abs(bias))))
            ray = np.where(hard_side, hard, soft_side).astype(np.float32, copy=False)
        rays = np.maximum(rays, np.float32(max(0.25, amp)) * ray)
    detail_freq = max(1, int(round(n * (2.0 + 1.0 * variation_n + 4.0 * density_n))))
    fine_freq = max(1, int(round(n * (4.0 + 6.0 * density_n))))
    detail = (
        0.72
        + (0.18 + 0.14 * density_n) * (0.5 + 0.5 * np.cos(theta * np.float32(detail_freq) + phase * 1.7))
        + (0.04 + 0.18 * density_n) * (0.5 + 0.5 * np.cos(theta * np.float32(fine_freq) + r / np.float32(max(width_px * 1.4, 1.0)) + phase * 2.3))
    )
    detail = 1.0 + (1.0 - 0.45 * variation_n) * (detail - 1.0)
    rays *= detail.astype(np.float32, copy=False)
    sharp = 0.65 + 3.25 * (1.0 - soft)
    rays = np.power(np.clip(rays, 0.0, 1.0), np.float32(sharp))
    rays = _edge_harden(rays, softness)
    rays *= np.float32(1.0 / (1.0 + 0.55 * width_ratio))
    # Radial sources need a stronger distance rolloff than parallel beams;
    # otherwise the frame edge can look brighter than the area around the
    # source on small previews.
    fade = _distance_falloff(r, decay_px * 1.45)
    end = _end_tail(r, length_px, max(length_px * 0.22, 2.0))
    core_guard = np.clip(r / np.float32(max(2.0, length_px * 0.03)), 0.0, 1.0)
    near_scatter = np.float32(0.035 + 0.020 * soft) * np.exp(-r / np.float32(max(length_px * 0.18, 1.0)))
    root_lift = np.float32(0.34 + 0.10 * soft) * np.exp(-r / np.float32(max(width_px * 3.4, 1.0))) * rays
    core_radius = max(float(width_px) * (0.20 + 0.42 * width_ratio + 0.08 * soft), 3.0)
    core_x = r / np.float32(max(core_radius, 1.0))
    source_core = np.float32(0.42 + 0.04 * soft) / (
        np.float32(1.0) + core_x ** np.float32(4.0))
    return (rays + near_scatter + root_lift) * fade * end * core_guard + source_core * end


def _projected_radial_ray(
    xx,
    yy,
    origin,
    target,
    width_n,
    width_px,
    decay_px,
    length_px,
    projection_length_px,
    projection_decay_px,
    count,
    density,
    variation,
    softness,
    edge_bias,
    spread_deg,
    seed,
):
    dx0 = xx - np.float32(origin[0])
    dy0 = yy - np.float32(origin[1])
    axis = target - origin
    raw_axis_len = max(float(np.hypot(float(axis[0]), float(axis[1]))), 1.0)
    axis_dir = axis / np.float32(raw_axis_len)
    projection_span_px = max(float(projection_length_px) * 4.0, 1.0)
    projection_decay_span_px = max(float(projection_decay_px) * 4.0, 1.0)
    axis_len = max(min(raw_axis_len * 8.0, projection_span_px * 0.50), 1.0)
    target = origin + axis_dir * np.float32(axis_len)
    r0 = np.sqrt(dx0 * dx0 + dy0 * dy0) + np.float32(1e-3)
    theta0 = np.arctan2(dy0, dx0)
    count_i = int(np.clip(round(float(count)), 1, 32))
    density_n = float(np.clip(float(density) / 100.0, 0.0, 1.0))
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    width_ratio = float(np.clip(float(width_n) / 2.0, 0.0, 1.0))
    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    spread = math.tan(math.radians(spread_deg) * 0.5)
    n = max(float(count_i), 1.0)
    phase = np.float32((int(seed) % 997) / 997.0 * 2.0 * math.pi)
    spacing = np.float32(2.0 * math.pi / n)
    base_sigma = float(spacing) * (0.010 + 0.16 * width_ratio + 0.035 * soft)
    root_radius = max(min(axis_len * 0.12, float(projection_length_px) * 0.035) * (0.75 + 0.25 * soft), 2.0)
    front_radius = max(
        axis_len * (0.40 + 1.15 * spread),
        projection_span_px * (0.08 + 0.58 * spread),
        root_radius * 2.5,
    )
    shaft_root_width = np.float32(max(root_radius * (0.75 + 0.35 * soft), 1.4))
    shaft_tip_width = np.float32(max(float(width_px) * (0.42 + 1.65 * width_ratio + 0.36 * soft), 1.6))
    rng = np.random.default_rng((int(seed) + 0x79B1) & 0xFFFFFFFF)
    detail_rng = np.random.default_rng((int(seed) + 0xA6D7) & 0xFFFFFFFF)
    angle_jitter_scale = 0.035 * min(1.0, max(0.0, n - 4.0) / 4.0)
    rays = np.zeros_like(theta0, dtype=np.float32)
    for i in range(count_i):
        center_jitter = np.float32(detail_rng.uniform(-angle_jitter_scale, angle_jitter_scale) * float(spacing) * variation_n)
        center = phase + np.float32(i) * spacing + center_jitter
        ray_cos = np.float32(math.cos(float(center)))
        ray_sin = np.float32(math.sin(float(center)))
        width_jitter = 1.0 + float(rng.uniform(-0.50, 0.70)) * variation_n + float(rng.uniform(-0.18, 0.18)) * density_n
        amp = 1.0 + float(rng.uniform(-0.38, 0.30)) * variation_n
        start_jitter = float(rng.uniform(0.0, max(root_radius * 1.8, projection_length_px * 0.035))) * variation_n
        ray_phase = np.float32(detail_rng.uniform(0.0, 2.0 * math.pi))
        p0x = np.float32(float(origin[0]) + root_radius * ray_cos)
        p0y = np.float32(float(origin[1]) + root_radius * ray_sin)
        p1x = np.float32(float(target[0]) + front_radius * ray_cos)
        p1y = np.float32(float(target[1]) + front_radius * ray_sin)
        vx = p1x - p0x
        vy = p1y - p0y
        seg_len = max(float(np.hypot(float(vx), float(vy))), 1.0)
        visible_len = max(float(length_px), 1.0)
        ux = np.float32(float(vx) / seg_len)
        uy = np.float32(float(vy) / seg_len)
        dx = xx - p0x
        dy = yy - p0y
        along = dx * ux + dy * uy
        cross = -dx * uy + dy * ux
        t = np.clip(along / np.float32(seg_len), 0.0, 1.0)
        # Widen the shaft as a straight cone from apex (source) to the projection
        # front.  The old ``((t-0.08)/0.92)**0.62`` held the beam pinched in a
        # flat dead-zone and then lifted off steeply, so with a large
        # ``shaft_tip_width`` the beam stayed a thin line and then ballooned at a
        # visible elbow (worse the wider the slider).  A linear ramp grows the
        # width evenly with no kink while still letting width thicken the front.
        width_t = t
        width_along = shaft_root_width + width_t * shaft_tip_width * np.float32(max(width_jitter, 0.35))
        # Width must not touch the start: the brightness normalisation that keeps
        # wide shafts from overpowering is applied where the shaft actually
        # widens (downstream, via ``width_t``), so the near-source root stays at
        # full strength regardless of the width slider.
        width_norm = np.float32(1.0) / (np.float32(1.0) + np.float32(0.42 * width_ratio) * width_t)
        ray = _asymmetric_gaussian(cross, width_along, edge_bias)
        start_gate = _rounded_ray_start_gate(
            along,
            cross,
            start_jitter,
            shaft_root_width,
            width_along,
            projection_length_px,
            softness,
            variation,
        )
        projection_end = _end_tail(
            along,
            seg_len,
            max(float(width_px) * (1.4 + 2.2 * soft), seg_len * 0.08, 2.0),
        )
        visible_end = _end_tail(along, visible_len, max(visible_len * 0.22, 2.0))
        fade = _distance_falloff(along, projection_decay_span_px * 0.80)
        delta = np.arctan2(np.sin(theta0 - center), np.cos(theta0 - center))
        bias = float(np.clip(float(edge_bias) / 100.0, -1.0, 1.0))
        if abs(bias) > 1e-4:
            hard_side = (delta * np.float32(bias)) > 0.0
            hard = np.power(np.clip(ray, 0.0, 1.0), np.float32(1.0 + 3.8 * abs(bias)))
            soft_side = np.power(np.clip(ray, 0.0, 1.0), np.float32(max(0.45, 1.0 - 0.38 * abs(bias))))
            ray = np.where(hard_side, hard, soft_side).astype(np.float32, copy=False)
        ray *= _single_ray_texture(along, cross, width_px, projection_span_px, density_n, variation_n, ray_phase)
        rays = np.maximum(rays, np.float32(max(0.25, amp)) * ray * start_gate * projection_end * visible_end * fade * width_norm)
    detail_freq = n * (1.6 + 1.2 * variation_n + 4.2 * density_n)
    detail = (
        0.74
        + (0.16 + 0.12 * density_n) * (0.5 + 0.5 * np.cos(theta0 * np.float32(detail_freq) + phase * np.float32(1.41)))
        + (0.04 + 0.14 * density_n) * (0.5 + 0.5 * np.cos(r0 / np.float32(max(width_px * 1.3, 1.0)) + phase * np.float32(2.1)))
    )
    detail = 1.0 + (1.0 - 0.45 * variation_n) * (detail - 1.0)
    rays *= detail.astype(np.float32, copy=False)
    sharp = 0.72 + 2.80 * (1.0 - soft)
    rays = _edge_harden(np.power(np.clip(rays, 0.0, 1.0), np.float32(sharp)), softness)
    source_radius = max(root_radius * (1.35 + 0.65 * soft), 2.0)
    source_core = np.float32(0.24 + 0.08 * soft) / (
        np.float32(1.0) + (r0 / np.float32(max(source_radius, 1.0))) ** np.float32(4.0))
    return rays + source_core


def _smoothstep01(value):
    v = np.clip(value, 0.0, 1.0).astype(np.float32, copy=False)
    return (v * v * (np.float32(3.0) - np.float32(2.0) * v)).astype(np.float32, copy=False)


def _rounded_ray_start_gate(along, cross, start_offset, root_width_px, width_along, length_px, softness, variation):
    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    root = float(max(float(root_width_px), 1.0))
    feather = np.float32(max(
        root * (1.90 + 1.10 * soft + 0.90 * variation_n),
        float(length_px) * (0.035 + 0.040 * variation_n),
        3.0,
    ))
    cap_radius = np.maximum(
        np.float32(root * (1.10 + 0.70 * soft)),
        width_along * np.float32(0.36 + 0.20 * soft),
    )
    cross_n = cross / np.maximum(cap_radius, np.float32(1e-3))
    rounded_lead = cap_radius * np.float32(0.78 + 0.34 * soft) / (np.float32(1.0) + cross_n * cross_n)
    return _smoothstep01((along - np.float32(start_offset) + rounded_lead + feather * np.float32(0.10)) / feather)


def _single_ray_texture(along, cross, width_px, length_px, density_n, variation_n, phase):
    if density_n <= 1e-6 and variation_n <= 1e-6:
        return np.float32(1.0)
    width = float(max(float(width_px), 1.0))
    long_scale = np.float32(max(float(length_px) * (0.20 + 0.16 * variation_n), width * 4.0, 4.0))
    cross_scale = np.float32(max(width * (0.85 + 0.50 * (1.0 - density_n)), 1.0))
    long = np.cos(along / long_scale + phase)
    diagonal = np.cos((along * np.float32(0.38 + 0.18 * variation_n) + cross) / cross_scale + phase * np.float32(1.73))
    texture = (
        np.float32(0.96)
        + np.float32(0.07 + 0.10 * variation_n + 0.06 * density_n) * long
        + np.float32(0.03 + 0.08 * density_n) * diagonal
    )
    return np.clip(texture, np.float32(0.55), np.float32(1.24)).astype(np.float32, copy=False)


def _distance_falloff(distance, decay_px):
    d = np.maximum(distance, 0.0).astype(np.float32, copy=False)
    scale = np.float32(max(float(decay_px) * 0.42, 1.0))
    x = d / scale
    return (np.float32(1.0) / (np.float32(1.0) + x * x)).astype(np.float32, copy=False)


def _end_tail(distance, length_px, feather_px):
    tail = np.maximum(distance - np.float32(length_px), 0.0)
    x = tail / np.float32(max(float(feather_px), 1.0))
    return (np.float32(1.0) / (np.float32(1.0) + x * x * np.float32(1.6))).astype(np.float32, copy=False)


def _ray_texture(perp, along, width_px, density, variation, seed):
    density_n = float(np.clip(float(density) / 100.0, 0.0, 1.0))
    if density_n <= 1e-4:
        return np.float32(1.0)
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    width = max(float(width_px), 1.0)
    phase = np.float32((int(seed) % 65521) / 65521.0 * 2.0 * math.pi)
    cross_scale = np.float32(max(width * (2.1 - 1.45 * density_n), 1.2))
    fine_scale = np.float32(max(width * (0.92 - 0.58 * density_n), 1.0))
    long_scale = np.float32(max(width * (8.5 - 5.8 * density_n + 2.2 * variation_n), 4.0))
    soft_streaks = np.cos(perp / cross_scale + along / long_scale + phase)
    fine_streaks = np.cos((perp + along * np.float32(0.22 + 0.22 * variation_n)) / fine_scale + phase * np.float32(1.73))
    longitudinal = np.cos(along / np.float32(max(width * (1.55 - 0.95 * density_n), 1.4)) + phase * np.float32(0.61))
    grain = 0.36 * soft_streaks + 0.34 * fine_streaks + 0.30 * longitudinal
    amount = np.float32(0.16 + 0.68 * density_n)
    return np.clip(1.0 + amount * grain, 0.30, 1.76).astype(np.float32, copy=False)


def _parallel_shaft_field(perp, along, width_px, beam_width, frame_span, count, density, variation, softness, edge_bias, seed):
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    count_n = float(np.clip(float(count) / 32.0, 0.0, 1.0))
    density_n = float(np.clip(float(density) / 100.0, 0.0, 1.0))
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    count_i = int(np.clip(round(float(count)), 1, 32))
    offset_span = max(float(beam_width) * (0.55 + 0.85 * variation_n), float(width_px) * 2.5)
    if count_i >= 3:
        offset_span = max(offset_span, float(frame_span) * (0.30 + 0.70 * count_n))
    field = np.zeros_like(perp, dtype=np.float32)
    if count_i <= 1:
        offsets = [0.0]
    else:
        offsets = list(np.linspace(-offset_span, offset_span, count_i, dtype=np.float32))
        jitter_strength = 0.08 + 0.25 * variation_n + 0.12 * density_n
        jitter = rng.uniform(-jitter_strength, jitter_strength, size=count_i).astype(np.float32) * np.float32(offset_span / max(1, count_i - 1))
        offsets = [float(o + j) for o, j in zip(offsets, jitter)]
        if count_i % 2 == 1:
            offsets[count_i // 2] = 0.0
    center_index = min(range(len(offsets)), key=lambda idx: abs(offsets[idx]))
    spacing = (2.0 * offset_span / max(1, count_i - 1)) if count_i > 1 else offset_span
    for idx, offset in enumerate(offsets):
        if idx == center_index:
            shaft_w = min(float(width_px) * 1.45 * (0.85 + 0.55 * soft), spacing * (0.42 + 0.56 * soft))
            amp = 1.30 - 0.35 * count_n
        else:
            shaft_w = float(width_px) * float(rng.uniform(0.48, 1.45)) * (0.72 + 0.58 * soft + 0.18 * density_n)
            shaft_w = min(shaft_w, spacing * (0.30 + 0.48 * soft))
            amp = float(rng.uniform(0.34, 0.82))
        shaft_w = max(1.5, shaft_w)
        shaft = np.float32(amp) * _edge_harden(_asymmetric_gaussian(perp - np.float32(offset), shaft_w, edge_bias), softness)
        field = np.maximum(field, shaft.astype(np.float32, copy=False))
    veil_w = max(float(beam_width) * 1.35, float(width_px) * 5.0)
    veil = np.exp(-(perp * perp) / np.float32(2.0 * veil_w * veil_w))
    field += np.float32(0.05 + 0.10 * soft + 0.05 * count_n) * veil
    phase = np.float32((int(seed) % 4099) / 4099.0 * 2.0 * math.pi)
    longitudinal = 0.88 + 0.12 * np.cos(along / np.float32(max(width_px * 5.5, 1.0)) + phase)
    return np.clip(field * longitudinal, 0.0, None).astype(np.float32)


def _directional_shaft_field(perp, along, cone, count, density, variation, softness, edge_bias, seed, start_stagger_px=0.0):
    rng = np.random.default_rng((int(seed) + 0x516B) & 0xFFFFFFFF)
    count_n = float(np.clip(float(count) / 32.0, 0.0, 1.0))
    density_n = float(np.clip(float(density) / 100.0, 0.0, 1.0))
    variation_n = float(np.clip(float(variation) / 100.0, 0.0, 1.0))
    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    count_i = int(np.clip(round(float(count)), 1, 32))
    field = np.zeros_like(perp, dtype=np.float32)
    if count_i <= 1:
        fractions = [0.0]
    else:
        fractions = list(np.linspace(-0.92, 0.92, count_i, dtype=np.float32))
        jitter = rng.uniform(-(0.04 + 0.16 * variation_n), 0.04 + 0.16 * variation_n, size=count_i).astype(np.float32)
        fractions = [float(np.clip(f + j, -0.96, 0.96)) for f, j in zip(fractions, jitter)]
        if count_i % 2 == 1:
            fractions[count_i // 2] = 0.0
    for frac in fractions:
        center = cone * np.float32(frac)
        frac_width = min(float(rng.uniform(0.12, 0.34)) * (0.65 + soft + 0.18 * density_n), 1.90 / max(count_i, 1))
        shaft_w = np.maximum(cone * np.float32(frac_width), np.float32(2.0))
        amp = float(rng.uniform(0.45, 1.0))
        shaft = np.float32(amp) * _edge_harden(_asymmetric_gaussian(perp - center, shaft_w, edge_bias), softness)
        if start_stagger_px > _EPS:
            start_offset = float(rng.uniform(0.0, start_stagger_px))
            start_feather = max(float(start_stagger_px) * 0.35, 3.0)
            start_gate = np.clip((along - np.float32(start_offset)) / np.float32(start_feather), 0.0, 1.0)
            start_gate = start_gate * start_gate * (np.float32(3.0) - np.float32(2.0) * start_gate)
            shaft *= start_gate.astype(np.float32, copy=False)
        field = np.maximum(field, shaft.astype(np.float32, copy=False))
    veil = np.exp(-(perp * perp) / np.maximum(np.float32(2.0) * (cone * np.float32(1.65)) ** 2, np.float32(1.0)))
    field += np.float32(0.05 + 0.12 * soft + 0.05 * count_n) * veil
    phase = np.float32((int(seed) % 8191) / 8191.0 * 2.0 * math.pi)
    longitudinal = 0.86 + 0.14 * np.cos(along / np.float32(80.0 + 140.0 * variation_n) + phase)
    return np.clip(field * longitudinal, 0.0, None).astype(np.float32)


def _asymmetric_gaussian(delta, width, edge_bias):
    bias = float(np.clip(float(edge_bias) / 100.0, -1.0, 1.0))
    width = np.maximum(np.asarray(width, dtype=np.float32), np.float32(1.0))
    if abs(bias) < 1e-4:
        sigma = width
    else:
        amount = abs(bias)
        hard = np.maximum(width * np.float32(1.0 - 0.62 * amount), np.float32(1.0))
        soft = width * np.float32(1.0 + 1.55 * amount)
        if bias > 0.0:
            neg_width, pos_width = soft, hard
        else:
            neg_width, pos_width = hard, soft
        sigma = np.where(delta < 0.0, neg_width, pos_width).astype(np.float32, copy=False)
    return np.exp(-(delta * delta) / np.maximum(np.float32(2.0) * sigma * sigma, np.float32(1.0))).astype(np.float32)


def _edge_harden(values, softness):
    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    power = np.float32(1.0 + 6.0 * ((1.0 - soft) ** 2.2))
    return np.power(np.clip(values, 0.0, None), power).astype(np.float32, copy=False)


def _harden_low_softness(layer, softness):
    """Turn the low end of the Edge Softness slider into crisp, thick shafts.

    ``_edge_harden`` only narrows the per-component gaussians, so by itself a low
    softness produced a thin spike, not a hard-edged beam (every mode also lays a
    broad soft band/veil underneath, which keeps the composite edge soft).  Below
    30 we additionally remap the *composited* layer through a flat-top contrast:
    the threshold drops (so the bright core spreads into a wider plateau) and the
    slope steepens (so the boundary turns crisp).  At/above 30 the layer is
    returned untouched, so the rest of the slider behaves exactly as before.
    """

    soft = float(np.clip(float(softness) / 100.0, 0.0, 1.0))
    if soft >= 0.30:
        return layer
    pk = float(np.max(layer, initial=0.0))
    if pk <= 0.0:
        return layer
    h = float((0.30 - soft) / 0.30)
    x = (layer / np.float32(pk)).astype(np.float32, copy=False)
    threshold = np.float32(0.5 - 0.45 * h)
    steepness = np.float32(1.0 + 26.0 * h)
    hardened = np.clip((x - threshold) * steepness + np.float32(0.5), 0.0, 1.0)
    return (
        layer * np.float32(1.0 - h) + hardened.astype(np.float32, copy=False) * np.float32(pk) * np.float32(h)
    ).astype(np.float32, copy=False)


def _occlusion_map(img, occlusion):
    strength = float(occlusion) / 100.0
    if strength <= 0.0:
        return np.float32(1.0)
    rgb = img[..., :3]
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    lum = cv2.GaussianBlur(lum, (0, 0), 1.6)
    lo, hi = np.percentile(lum, [5, 95])
    norm = np.clip((lum - np.float32(lo)) / np.float32(max(float(hi - lo), 1e-4)), 0.0, 1.0)
    transmit = 0.08 + 0.92 * np.power(norm, np.float32(0.82))
    return (1.0 - strength) + strength * transmit


def _cast_shadow_map(img, direction, length_px, occlusion):
    strength = float(occlusion) / 100.0
    if strength <= 0.0:
        return np.float32(1.0)
    rgb = img[..., :3]
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    lo, hi = np.percentile(lum, [4, 94])
    norm = np.clip((lum - np.float32(lo)) / np.float32(max(float(hi - lo), 1e-4)), 0.0, 1.0)
    blocker = np.power(np.clip(1.0 - norm, 0.0, 1.0), np.float32(0.78)).astype(np.float32)
    local = lum - cv2.GaussianBlur(lum, (0, 0), 5.0)
    detail = np.clip(-local / np.float32(max(float(np.percentile(np.abs(local), 92)), 1e-4)), 0.0, 1.0)
    blocker = np.maximum(blocker, detail * np.float32(0.72))
    blocker = cv2.GaussianBlur(blocker, (0, 0), 0.9)
    h, w = blocker.shape[:2]
    max_len = min(float(length_px), float(max(h, w)) * 1.35)
    samples = int(np.clip(max_len / max(8.0, min(h, w) / 28.0), 8, 28))
    shadow = np.zeros_like(blocker, dtype=np.float32)
    weight_sum = 0.0
    for i in range(samples):
        t = max_len * (i / max(1, samples - 1))
        weight = math.exp(-0.8 * i / max(1, samples - 1))
        matrix = np.array(
            [[1.0, 0.0, float(direction[0]) * t],
             [0.0, 1.0, float(direction[1]) * t]],
            dtype=np.float32,
        )
        shifted = cv2.warpAffine(
            blocker,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        shadow = np.maximum(shadow, shifted.astype(np.float32, copy=False) * np.float32(weight))
        weight_sum = max(weight_sum, weight)
    if weight_sum > 0:
        shadow /= np.float32(weight_sum)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(0.8, min(h, w) * 0.010))
    shadow = np.power(np.clip(shadow, 0.0, 1.0), np.float32(0.86))
    return np.clip(1.0 - np.float32(0.98 * strength) * shadow, 0.03, 1.0).astype(np.float32)
