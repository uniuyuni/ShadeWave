"""
Mask2 の拡張パラメータ（ぼかし・HLS 等）をマスク画像に適用。BaseMask._apply_extened_params と同等。
"""
from __future__ import annotations

import numpy as np
import cv2
import logging
import os

import cores.core as core
import cores.expand_mask as expand_mask
import effects
import params
from cores.mask2 import edge_refine
from cores.mask2 import mask_rasters
from cores.mask2.coordinate_context import Mask2CoordinateContext

from cores.mask2.mask_mesh import mesh_cps_hash_key as _mesh_cps_hash_key


def get_mask_hash_tuple(effects_param):
    return (
        effects.Mask2Effect.get_param(effects_param, "switch_mask2_settings"),
        effects.Mask2Effect.get_param(effects_param, "mask2_invert"),
        effects.Mask2Effect.get_param(effects_param, "mask2_allow_over_one"),
        effects.Mask2Effect.get_param(effects_param, "mask2_allow_under_zero"),
        effects.Mask2Effect.get_param(effects_param, "switch_mask2_depth"),
        effects.Mask2Effect.get_param(effects_param, "mask2_depth_min"),
        effects.Mask2Effect.get_param(effects_param, "mask2_depth_max"),
        effects.Mask2Effect.get_param(effects_param, "switch_mask2_hue"),
        effects.Mask2Effect.get_param(effects_param, "mask2_hue_distance"),
        effects.Mask2Effect.get_param(effects_param, "mask2_hue_min"),
        effects.Mask2Effect.get_param(effects_param, "mask2_hue_max"),
        effects.Mask2Effect.get_param(effects_param, "switch_mask2_lum"),
        effects.Mask2Effect.get_param(effects_param, "mask2_lum_distance"),
        effects.Mask2Effect.get_param(effects_param, "mask2_lum_min"),
        effects.Mask2Effect.get_param(effects_param, "mask2_lum_max"),
        effects.Mask2Effect.get_param(effects_param, "switch_mask2_sat"),
        effects.Mask2Effect.get_param(effects_param, "mask2_sat_distance"),
        effects.Mask2Effect.get_param(effects_param, "mask2_sat_min"),
        effects.Mask2Effect.get_param(effects_param, "mask2_sat_max"),
        effects.Mask2Effect.get_param(effects_param, "switch_mask2_options"),
        effects.Mask2Effect.get_param(effects_param, "mask2_blur"),
        effects.Mask2Effect.get_param(effects_param, "mask2_open_space"),
        effects.Mask2Effect.get_param(effects_param, "mask2_close_space"),
        effects.Mask2Effect.get_param(effects_param, "mask2_freedraw_brush_hardness"),
        effects.Mask2Effect.get_param(effects_param, "mask2_polyline_fill"),
        effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_mode"),
        effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_radius"),
        effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_strength"),
        effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_bias"),
        # mask Mesh warp 関連 (Composit のみ実効、子マスクは placeholder default)
        tuple(effects.Mask2Effect.get_param(effects_param, "mask_mesh_size") or ()),
        _mesh_cps_hash_key(effects.Mask2Effect.get_param(effects_param, "mask_mesh_control_points")),
        bool(effects.Mask2Effect.get_param(effects_param, "mask_mesh_link_to_image")),
    )


def apply_extended_params(
    ctx,
    effects_param,
    image,
    center_tcg,
    fill_grown_region=True,
    seed_from_guide=False,
    seed_mask=None,
    edge_refine_enabled=True,
    edge_refine_support_softness=0.0,
    edge_refine_debug_label=None,
    edge_refine_selection_strategy=edge_refine.STRATEGY_REFINE,
    edge_refine_draw_strokes=None,
):
    """center_tcg: マスクの中心（TCG）。HLS 範囲の参照点に使う。"""
    simg = _apply_mask_space(ctx, effects_param, image)
    simg, edge_support = _apply_edge_refine(
        ctx,
        effects_param,
        simg,
        center_tcg,
        fill_grown_region=fill_grown_region,
        seed_from_guide=seed_from_guide,
        seed_mask=seed_mask,
        edge_refine_enabled=edge_refine_enabled,
        edge_refine_support_softness=edge_refine_support_softness,
        edge_refine_debug_label=edge_refine_debug_label,
        edge_refine_selection_strategy=edge_refine_selection_strategy,
        edge_refine_draw_strokes=edge_refine_draw_strokes,
    )
    return apply_post_edge_params(ctx, effects_param, simg, center_tcg, edge_support=edge_support)


def apply_post_edge_params(ctx, effects_param, image, center_tcg, edge_support=None):
    dimg = _apply_depth_mask(effects_param, image)
    himg = _draw_hue_mask(ctx, effects_param, dimg, center_tcg)
    limg = _draw_lum_mask(ctx, effects_param, himg, center_tcg)
    simg = _draw_sat_mask(ctx, effects_param, limg, center_tcg)
    bimg = _apply_mask_blur(effects_param, simg)
    if edge_support is not None:
        bimg = np.where(edge_support > 0.001, bimg, 0.0)
    return bimg


def render_freedraw_edge_refine_full_view(
    ctx,
    effects_param,
    source_lines,
    center_tcg,
    mask_shape,
    debug_label=None,
):
    if effects.Mask2Effect.get_param(effects_param, "switch_mask2_options") is not True:
        return None
    mode = effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_mode")
    if not edge_refine.is_enabled(mode):
        return None
    # Default ON (disable via PLATYPUS_DRAW_QS_FULL_VIEW=0). The guide is now built
    # geometry-consistently with the strokes: _warp_original_to_render_region warps
    # the original into the rotated render-rect space that ctx.tcg_to_full_image
    # targets (see below), so straighten/rotation is followed. Runs only when the
    # view is cropped/zoomed (_should_render_draw_refine_full_view); at full display
    # it defers to the regular crop path, which already has the whole image.
    full_view_flag = os.getenv("PLATYPUS_DRAW_QS_FULL_VIEW", "").strip().lower()
    if full_view_flag in {"0", "false", "no", "off"}:
        return None
    original = ctx.get_original_image_rgb()
    if original is None or getattr(original, "size", 0) == 0:
        return None
    if not _should_render_draw_refine_full_view(ctx, original):
        return None

    orig_h, orig_w = original.shape[:2]
    if orig_w <= 0 or orig_h <= 0:
        return None

    disp_info = params.get_disp_info(ctx.tcg_info)
    render_rect = _freedraw_refine_render_rect(
        ctx,
        original,
        disp_info,
        effects_param,
        source_lines,
    )
    if render_rect is None:
        return None
    rx0, ry0, rx1, ry1 = render_rect
    # Geometry-correct guide: warp the original directly into the rotated render
    # region (the space tcg_to_full_image / the strokes live in), region-sized so
    # there is no full-image rotation. valid_region marks real image content.
    source_region, valid_region = _warp_original_to_render_region(ctx, original, render_rect)
    if source_region is None:
        return None
    region_h, region_w = source_region.shape[:2]
    if region_w <= 0 or region_h <= 0:
        return None
    render_image, render_scale = _scale_freedraw_refine_region(source_region)
    render_h, render_w = render_image.shape[:2]
    total_scale = float(render_scale)
    # Validity at render resolution (eroded a few px so the trace stays off the
    # synthetic border beyond the image edge).
    valid_render = None
    if valid_region is not None and bool(np.any(valid_region)) and not bool(np.all(valid_region)):
        vr = cv2.resize(valid_region.astype(np.float32), (render_w, render_h),
                        interpolation=cv2.INTER_NEAREST) > 0.5
        vr = cv2.erode(vr.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        valid_render = vr

    render_ctx = _make_region_view_context(
        ctx,
        render_image,
        (rx1 - rx0, ry1 - ry0),
        total_scale,
    )
    render_lines = _freedraw_lines_to_region_texture(ctx, source_lines, (rx0, ry0), total_scale)
    if not render_lines:
        return None

    render_mask = mask_rasters.draw_line_texture(
        (render_w, render_h),
        render_lines,
        allow_over_one=False,
        allow_under_zero=False,
    )
    if effects.Mask2Effect.get_param(effects_param, "switch_mask2_settings") is True:
        if effects.Mask2Effect.get_param(effects_param, "mask2_invert") is True:
            render_mask = 1.0 - render_mask

    render_mask = _apply_mask_space(render_ctx, effects_param, render_mask)
    refined, render_support = edge_refine.refine_mask_edge_aware(
        render_image,
        render_mask,
        guide_point=_safe_tcg_to_region_texture(ctx, center_tcg, (rx0, ry0), total_scale),
        mode=mode,
        radius=_edge_refine_radius_to_texture(
            render_ctx,
            effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_radius"),
        ),
        strength=effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_strength"),
        edge_bias=_edge_refine_edge_bias_to_texture(
            render_ctx,
            effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_bias"),
        ),
        fill_grown_region=True,
        seed_from_guide=False,
        seed_mask=edge_refine.make_confident_seed(render_mask),
        support_softness=0.0,
        debug_label=debug_label,
        selection_strategy=edge_refine.STRATEGY_DRAW,
        draw_strokes=render_lines,
        draw_pixel_scale=total_scale,
        return_support=True,
    )

    # Keep the selection off the synthetic border beyond the image edge: clip the
    # refined mask / support to where real image content exists.
    if valid_render is not None and os.environ.get("QS_FULLVIEW_VALIDITY", "1").strip().lower() not in {"0", "false", "no", "off"}:
        vmask = valid_render.astype(np.float32)
        refined = np.asarray(refined, dtype=np.float32) * vmask
        if render_support is not None:
            render_support = np.asarray(render_support, dtype=np.float32) * vmask

    out = _crop_full_view_to_texture(
        ctx,
        refined,
        tuple(mask_shape),
        source_origin=(rx0, ry0),
        source_scale=total_scale,
    )
    support = (
        _crop_full_view_to_texture(
            ctx,
            render_support,
            tuple(mask_shape),
            source_origin=(rx0, ry0),
            source_scale=total_scale,
        )
        if render_support is not None
        else None
    )
    _debug_freedraw_refine_current_view(
        ctx,
        effects_param,
        source_lines,
        center_tcg,
        tuple(mask_shape),
        out,
        support,
        debug_label,
    )
    return apply_post_edge_params(ctx, effects_param, out, center_tcg, edge_support=support)


def _should_render_draw_refine_full_view(ctx, original):
    try:
        disp_info = params.get_disp_info(ctx.tcg_info)
    except Exception:
        return False
    if disp_info is None:
        return False
    # Skip when the whole image is displayed: the regular crop path already has
    # the full image as its guide, so full-view adds no beyond-viewport context.
    # _disp_is_initial_full_rect is the correct "whole image" test (the previous
    # dx==0,dy==0 check missed the centered letterbox of non-square images, e.g.
    # a 160x100 image whose initial rect is (0,30,160,130) -> dy=30).
    if _disp_is_initial_full_rect(ctx, original, disp_info):
        return False
    return True


def _disp_is_initial_full_rect(ctx, original, disp_info):
    if disp_info is None:
        return False
    try:
        orig_w, orig_h = getattr(ctx, "tcg_info", {}).get(
            "original_img_size",
            (original.shape[1], original.shape[0]),
        )
        x0, y0, x1, y1 = core.get_initial_crop_rect(int(orig_w), int(orig_h))
        return _rect_close(disp_info[:4], (x0, y0, x1 - x0, y1 - y0), tolerance=2.0)
    except Exception:
        return False


def _rect_close(a, b, tolerance=1.0):
    try:
        return all(abs(float(av) - float(bv)) <= float(tolerance) for av, bv in zip(a, b))
    except Exception:
        return False


def _freedraw_refine_render_rect(ctx, original, disp_info, effects_param, source_lines):
    if disp_info is None:
        return None
    orig_h, orig_w = original.shape[:2]
    coord_w = coord_h = max(int(orig_w), int(orig_h))
    dx, dy, dw, dh = [float(v) for v in disp_info[:4]]
    if dw <= 0.0 or dh <= 0.0:
        return None

    margin = _freedraw_refine_margin(effects_param, source_lines)
    base = (
        max(0.0, dx),
        max(0.0, dy),
        min(float(coord_w), dx + dw),
        min(float(coord_h), dy + dh),
    )
    expanded_base = _expand_rect(base, margin, coord_w, coord_h)
    render_base = None
    for line_rect in _freedraw_line_full_image_rects(ctx, source_lines, coord_w, coord_h):
        if _rects_intersect(line_rect, expanded_base):
            expanded_line = _expand_rect(line_rect, margin, coord_w, coord_h)
            render_base = expanded_line if render_base is None else _union_rect(render_base, expanded_line)
    if render_base is None:
        render_base = expanded_base

    x0 = max(0, int(np.floor(render_base[0])))
    y0 = max(0, int(np.floor(render_base[1])))
    x1 = min(coord_w, int(np.ceil(render_base[2])))
    y1 = min(coord_h, int(np.ceil(render_base[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _warp_original_to_render_region(ctx, original, render_rect):
    """Warp the *pre-rotation* original directly into the render rect (which is in
    the rotated "full image" canvas space that ctx.tcg_to_full_image targets).

    This makes the full-view guide geometry-consistent with the strokes (which are
    positioned via tcg_to_full_image) without rotating the whole image: a single
    cv2 warp whose cost is proportional to the (small) output region, reusing the
    exact transform the geometry effect uses (core.combined_rotation_canvas_matrix).

    Returns (region_rgb_f32, valid_bool) where valid marks where real image content
    exists (False over the synthetic border beyond the image, so the edge trace can
    be kept off it).
    """
    rx0, ry0, rx1, ry1 = [int(round(float(v))) for v in render_rect]
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw <= 0 or rh <= 0:
        return None, None
    tcg = getattr(ctx, "tcg_info", {}) or {}
    angle_deg = float(np.degrees(float(tcg.get("rotation", 0.0)) + float(tcg.get("rotation2", 0.0))))
    flip = int(tcg.get("flip_mode", 0))
    matrix = tcg.get("matrix", None)
    if matrix is not None and np.allclose(np.asarray(matrix, dtype=np.float64), np.eye(3), atol=1e-9):
        matrix = None

    # original -> rotated canvas (same transform that produced imgc / the display).
    trans, _size, ttype = core.combined_rotation_canvas_matrix(
        np.asarray(original).shape, angle_deg, flip, matrix)
    trans3 = np.eye(3, dtype=np.float64)
    if ttype == "perspective":
        trans3 = np.asarray(trans, dtype=np.float64)
    else:
        trans3[:2, :] = np.asarray(trans, dtype=np.float64)
    # compose original -> render-region output (crop translate by the rect origin)
    crop_t = np.array([[1.0, 0.0, -rx0], [0.0, 1.0, -ry0], [0.0, 0.0, 1.0]], dtype=np.float64)
    m = crop_t @ trans3

    src = np.asarray(original, dtype=np.float32)
    ones = np.ones(src.shape[:2], dtype=np.float32)
    # Zero (constant black) border beyond the image: the photo edge is a real
    # boundary, so the wall there matches the in-crop behaviour (and the validity
    # mask below stops the selection from leaking into the synthetic void).
    if ttype == "perspective":
        region = cv2.warpPerspective(src, m, (rw, rh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpPerspective(ones, m, (rw, rh), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        m2 = m[:2, :]
        region = cv2.warpAffine(src, m2, (rw, rh), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpAffine(ones, m2, (rw, rh), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return region.astype(np.float32, copy=False), (valid > 0.5)


def _crop_padded_image_region(image, rect, coordinate_scale=1.0):
    arr = np.asarray(image)
    img_h, img_w = arr.shape[:2]
    scale = float(coordinate_scale)
    x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in rect]
    out_w = max(0, x1 - x0)
    out_h = max(0, y1 - y0)
    out = np.zeros((out_h, out_w) + arr.shape[2:], dtype=arr.dtype)
    if out_w <= 0 or out_h <= 0:
        return out

    px0, py0, _px1, _py1 = core.get_initial_crop_rect(img_w, img_h)
    img_rect = (int(px0), int(py0), int(px0) + img_w, int(py0) + img_h)
    ix0 = max(x0, img_rect[0])
    iy0 = max(y0, img_rect[1])
    ix1 = min(x1, img_rect[2])
    iy1 = min(y1, img_rect[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return out

    src_x0 = ix0 - img_rect[0]
    src_y0 = iy0 - img_rect[1]
    src_x1 = ix1 - img_rect[0]
    src_y1 = iy1 - img_rect[1]
    dst_x0 = ix0 - x0
    dst_y0 = iy0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def _freedraw_refine_max_pixels():
    try:
        return max(120_000, int(os.getenv("PLATYPUS_DRAW_REFINE_MAX_PIXELS", "1200000")))
    except ValueError:
        return 1_200_000


def _scale_freedraw_refine_region(image):
    h, w = image.shape[:2]
    pixels = max(1, int(w) * int(h))
    max_pixels = _freedraw_refine_max_pixels()
    if pixels <= max_pixels:
        return image, 1.0
    scale = float(np.sqrt(float(max_pixels) / float(pixels)))
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _freedraw_refine_margin(effects_param, source_lines):
    try:
        radius = float(effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_radius"))
    except Exception:
        radius = 0.0
    stroke_sizes = []
    for line in source_lines or []:
        try:
            stroke_sizes.append(float(getattr(line, "size", 0.0)))
        except Exception:
            pass
    brush = max(stroke_sizes) if stroke_sizes else 0.0
    return max(32.0, radius * 2.0 + 16.0, brush * 2.0 + 16.0)


def _freedraw_line_full_image_rects(ctx, source_lines, orig_w, orig_h):
    rects = []
    for line in source_lines or []:
        points = getattr(line, "points", None)
        if not points:
            continue
        full_points = []
        for point in points:
            try:
                full_points.append(ctx.tcg_to_full_image(*point))
            except Exception:
                continue
        if not full_points:
            continue
        pts = np.asarray(full_points, dtype=np.float32)
        try:
            pad = max(1.0, float(getattr(line, "size", 1.0)) * 0.5 + 2.0)
        except Exception:
            pad = 3.0
        rects.append(_expand_rect((
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
        ), pad, orig_w, orig_h))
    return rects


def _expand_rect(rect, pad, max_w, max_h):
    x0, y0, x1, y1 = [float(v) for v in rect]
    pad = float(max(0.0, pad))
    return (
        max(0.0, x0 - pad),
        max(0.0, y0 - pad),
        min(float(max_w), x1 + pad),
        min(float(max_h), y1 + pad),
    )


def _union_rect(a, b):
    return (
        min(float(a[0]), float(b[0])),
        min(float(a[1]), float(b[1])),
        max(float(a[2]), float(b[2])),
        max(float(a[3]), float(b[3])),
    )


def _rects_intersect(a, b):
    return (
        float(a[0]) < float(b[2])
        and float(b[0]) < float(a[2])
        and float(a[1]) < float(b[3])
        and float(b[1]) < float(a[3])
    )


def _make_region_view_context(ctx, image, source_size, source_scale):
    source_w, source_h = int(source_size[0]), int(source_size[1])
    render_h, render_w = image.shape[:2]
    source = getattr(ctx, "primary_param", None)
    primary = dict(source) if isinstance(source, dict) else {}
    primary["original_img_size"] = (source_w, source_h)
    primary["img_size"] = (source_w, source_h)
    primary.setdefault("rotation", 0)
    primary.setdefault("rotation2", 0)
    primary.setdefault("flip_mode", 0)
    primary["matrix"] = np.eye(3)
    region_disp = (0, 0, source_w, source_h, float(source_scale))
    primary["disp_info"] = region_disp

    region_ctx = Mask2CoordinateContext()
    region_ctx.set_texture_size(render_w, render_h)
    region_ctx.set_primary_param(primary, region_disp)
    region_ctx.set_ref_image(image, image)
    return region_ctx


def _freedraw_lines_to_region_texture(ctx, source_lines, origin, scale=1.0):
    result = []
    ox, oy = float(origin[0]), float(origin[1])
    scale = float(scale)
    for src in source_lines or []:
        points = getattr(src, "points", None)
        if not points:
            continue
        line = mask_rasters.Line(
            bool(getattr(src, "is_erasing", False)),
            float(max(1.0, getattr(src, "size", 1.0))) * scale,
            getattr(src, "soft", 100),
        )
        for point in points:
            try:
                px, py = ctx.tcg_to_full_image(*point)
                line.add_point((px - ox) * scale, (py - oy) * scale)
            except Exception:
                continue
        if line.points:
            result.append(line)
    return result


def _freedraw_lines_to_current_texture(ctx, source_lines):
    result = []
    for src in source_lines or []:
        points = getattr(src, "points", None)
        if not points:
            continue
        try:
            size = ctx.tcg_to_image_scale(float(max(1.0, getattr(src, "size", 1.0))), 0)[0]
        except Exception:
            size = float(max(1.0, getattr(src, "size", 1.0)))
        line = mask_rasters.Line(
            bool(getattr(src, "is_erasing", False)),
            size,
            getattr(src, "soft", 100),
        )
        for point in points:
            try:
                line.add_point(*ctx.tcg_to_texture(*point))
            except Exception:
                continue
        if line.points:
            result.append(line)
    return result


def _debug_freedraw_refine_current_view(
        ctx,
        effects_param,
        source_lines,
        center_tcg,
        mask_shape,
        refined,
        support,
        debug_label):
    if not getattr(edge_refine, "_debug_dump_enabled", lambda: False)():
        return
    texture_h, texture_w = int(mask_shape[0]), int(mask_shape[1])
    if texture_w <= 0 or texture_h <= 0:
        return
    try:
        view_lines = _freedraw_lines_to_current_texture(ctx, source_lines)
        view_mask = mask_rasters.draw_line_texture(
            (texture_w, texture_h),
            view_lines,
            allow_over_one=False,
            allow_under_zero=False,
        )
        if effects.Mask2Effect.get_param(effects_param, "switch_mask2_settings") is True:
            if effects.Mask2Effect.get_param(effects_param, "mask2_invert") is True:
                view_mask = 1.0 - view_mask
        view_mask = _apply_mask_space(ctx, effects_param, view_mask)
        guide = _get_edge_refine_guide_image(ctx, view_mask.shape[:2])
        if guide is None:
            return
        refined_mask = _as_float_mask_for_debug(refined, view_mask.shape)
        support_mask = (
            _as_float_mask_for_debug(support, view_mask.shape)
            if support is not None
            else refined_mask
        )
        seed = edge_refine.make_confident_seed(view_mask)
        candidate = (view_mask > 0.02) | (refined_mask > 0.02) | (support_mask > 0.02)
        label = f"{debug_label or 'FreeDrawMaskFull'}Crop"
        edge_refine._debug_dump_refine_state(
            guide,
            view_mask,
            refined_mask,
            _get_edge_refine_guide_point(ctx, center_tcg),
            seed,
            candidate,
            support_mask > 0.02,
            _draw_barrier_strength_for_debug(effects_param),
            False,
            label,
            "edge_snap",
            extra_planes=[("full_support_crop", support_mask)],
        )
    except Exception:
        logging.exception("[EDGE_REFINE_DEBUG] failed to write full-view crop debug")


def _as_float_mask_for_debug(image, shape):
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape[:2] != tuple(shape[:2]):
        arr = cv2.resize(
            arr,
            (int(shape[1]), int(shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _draw_barrier_strength_for_debug(effects_param):
    try:
        return float(effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_strength"))
    except Exception:
        return 60.0


def _safe_tcg_to_region_texture(ctx, center_tcg, origin, scale=1.0):
    if center_tcg is None:
        return None
    try:
        px, py = ctx.tcg_to_full_image(*center_tcg)
        scale = float(scale)
        return ((px - float(origin[0])) * scale, (py - float(origin[1])) * scale)
    except Exception:
        return None


def _crop_full_view_to_texture(ctx, image, mask_shape, source_origin=(0, 0), source_scale=1.0):
    texture_h, texture_w = int(mask_shape[0]), int(mask_shape[1])
    disp_info = params.get_disp_info(ctx.tcg_info)
    if image is None or disp_info is None or texture_w <= 0 or texture_h <= 0:
        return np.zeros((texture_h, texture_w), dtype=np.float32)

    nw, nh, ox, oy = core.crop_size_and_offset_from_texture(texture_w, texture_h, disp_info)
    if nw <= 0 or nh <= 0:
        return np.zeros((texture_h, texture_w) + np.asarray(image).shape[2:], dtype=np.asarray(image).dtype)

    cx, cy, cw, ch, _scale = disp_info
    source_scale = float(source_scale)
    cx = (float(cx) - float(source_origin[0])) * source_scale
    cy = (float(cy) - float(source_origin[1])) * source_scale
    cw, ch = float(cw) * source_scale, float(ch) * source_scale
    if cw <= 0 or ch <= 0:
        return np.zeros((texture_h, texture_w) + np.asarray(image).shape[2:], dtype=np.asarray(image).dtype)

    arr = np.asarray(image)
    src_h, src_w = arr.shape[:2]
    x0 = int(round(cx))
    y0 = int(round(cy))
    x1 = int(round(cx + cw))
    y1 = int(round(cy + ch))
    if 0 <= x0 < x1 <= src_w and 0 <= y0 < y1 <= src_h:
        content = cv2.resize(arr[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        sx = float(cw) / float(nw)
        sy = float(ch) / float(nh)
        matrix = np.array([
            [sx, 0.0, cx + sx * 0.5 - 0.5],
            [0.0, sy, cy + sy * 0.5 - 0.5],
        ], dtype=np.float32)
        content = cv2.warpAffine(
            arr,
            matrix,
            (nw, nh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    out = np.zeros((texture_h, texture_w) + arr.shape[2:], dtype=content.dtype)
    dst_x0 = max(0, int(ox))
    dst_y0 = max(0, int(oy))
    dst_x1 = min(texture_w, int(ox) + nw)
    dst_y1 = min(texture_h, int(oy) + nh)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return out
    src_x0 = dst_x0 - int(ox)
    src_y0 = dst_y0 - int(oy)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = content[src_y0:src_y1, src_x0:src_x1]
    return out


def _apply_mask_space(ctx, effects_param, image):
    switch_mask2_options = effects.Mask2Effect.get_param(effects_param, "switch_mask2_options")
    if switch_mask2_options is True:
        open_space = effects.Mask2Effect.get_param(effects_param, "mask2_open_space")
        image = expand_mask.adjust_foreground_only(
            image, open_space * params.get_disp_info(ctx.tcg_info)[4], False
        )

        close_space = effects.Mask2Effect.get_param(effects_param, "mask2_close_space")
        image = expand_mask.adjust_holes_only(
            image, close_space * params.get_disp_info(ctx.tcg_info)[4], False
        )

    return image


def _apply_edge_refine(
    ctx,
    effects_param,
    image,
    center_tcg,
    fill_grown_region=True,
    seed_from_guide=False,
    seed_mask=None,
    edge_refine_enabled=True,
    edge_refine_support_softness=0.0,
    edge_refine_debug_label=None,
    edge_refine_selection_strategy=edge_refine.STRATEGY_REFINE,
    edge_refine_draw_strokes=None,
):
    if not edge_refine_enabled:
        return image, None
    if effects.Mask2Effect.get_param(effects_param, "switch_mask2_options") is not True:
        return image, None
    mode = effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_mode")
    if not edge_refine.is_enabled(mode):
        return image, None
    guide = _get_edge_refine_guide_image(ctx, image.shape[:2])
    guide_point = _get_edge_refine_guide_point(ctx, center_tcg)
    return edge_refine.refine_mask_edge_aware(
        guide,
        image,
        guide_point=guide_point,
        mode=mode,
        radius=_edge_refine_radius_to_texture(
            ctx,
            effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_radius"),
        ),
        strength=effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_strength"),
        edge_bias=_edge_refine_edge_bias_to_texture(
            ctx,
            effects.Mask2Effect.get_param(effects_param, "mask2_edge_refine_bias"),
        ),
        fill_grown_region=fill_grown_region,
        seed_from_guide=seed_from_guide,
        seed_mask=seed_mask,
        support_softness=edge_refine_support_softness,
        debug_label=edge_refine_debug_label,
        selection_strategy=edge_refine_selection_strategy,
        draw_strokes=edge_refine_draw_strokes,
        return_support=True,
    )


def _edge_refine_radius_to_texture(ctx, radius):
    try:
        disp_scale = float(params.get_disp_info(ctx.tcg_info)[4])
    except Exception:
        disp_scale = 1.0
    return float(radius) * disp_scale


def _edge_refine_edge_bias_to_texture(ctx, edge_bias):
    try:
        disp_scale = float(params.get_disp_info(ctx.tcg_info)[4])
    except Exception:
        disp_scale = 1.0
    return float(edge_bias) * disp_scale


def _get_edge_refine_guide_image(ctx, mask_shape):
    crop = getattr(ctx, "crop_image_rgb", None)
    if crop is not None and getattr(crop, "shape", (None, None))[:2] == tuple(mask_shape):
        return crop

    original = ctx.get_original_image_rgb()
    if original is not None:
        guide = _fit_image_to_texture(ctx, original, mask_shape)
        if getattr(guide, "shape", (None, None))[:2] != tuple(mask_shape):
            guide = cv2.resize(
                guide,
                (int(mask_shape[1]), int(mask_shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )
        return guide

    hls = getattr(ctx, "crop_image_hls", None)
    if hls is not None:
        return hls[..., 1]
    return None


def _get_edge_refine_guide_point(ctx, center_tcg):
    if center_tcg is None:
        return None
    try:
        return ctx.tcg_to_texture(*center_tcg)
    except Exception:
        return None


def _fit_image_to_texture(ctx, image, mask_shape):
    texture_h, texture_w = int(mask_shape[0]), int(mask_shape[1])
    disp_info = params.get_disp_info(ctx.tcg_info)
    if image is None or disp_info is None or texture_w <= 0 or texture_h <= 0:
        return None

    nw, nh, ox, oy = core.crop_size_and_offset_from_texture(texture_w, texture_h, disp_info)
    if nw <= 0 or nh <= 0:
        return np.zeros((texture_h, texture_w) + image.shape[2:], dtype=image.dtype)

    cx, cy, cw, ch, _scale = disp_info
    cx, cy, cw, ch = int(cx), int(cy), int(cw), int(ch)
    if cw <= 0 or ch <= 0:
        return np.zeros((texture_h, texture_w) + image.shape[2:], dtype=image.dtype)

    src_h, src_w = image.shape[:2]
    orig_w, orig_h = ctx.tcg_info.get("original_img_size", (src_w, src_h))
    maxsize = max(int(orig_w), int(orig_h))
    if (src_w, src_h) == (int(orig_w), int(orig_h)) and (src_w, src_h) != (maxsize, maxsize):
        cx = float(cx) - (maxsize - int(orig_w)) / 2.0
        cy = float(cy) - (maxsize - int(orig_h)) / 2.0

    in_bounds = 0 <= cx and 0 <= cy and cx + cw <= src_w and cy + ch <= src_h
    integer_rect = abs(float(cx) - round(float(cx))) < 1e-6 and abs(float(cy) - round(float(cy))) < 1e-6
    if in_bounds and integer_rect:
        x0 = int(round(cx))
        y0 = int(round(cy))
        content = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], (nw, nh))
    else:
        sx = float(cw) / float(nw)
        sy = float(ch) / float(nh)
        matrix = np.array([
            [sx, 0.0, float(cx) + sx * 0.5 - 0.5],
            [0.0, sy, float(cy) + sy * 0.5 - 0.5],
        ], dtype=np.float32)
        content = cv2.warpAffine(
            image,
            matrix,
            (nw, nh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    out = np.zeros((texture_h, texture_w) + image.shape[2:], dtype=content.dtype)
    dst_x0 = max(0, int(ox))
    dst_y0 = max(0, int(oy))
    dst_x1 = min(texture_w, int(ox) + nw)
    dst_y1 = min(texture_h, int(oy) + nh)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return out
    src_x0 = dst_x0 - int(ox)
    src_y0 = dst_y0 - int(oy)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = content[src_y0:src_y1, src_x0:src_x1]
    return out


def _apply_depth_mask(effects_param, image):
    switch_mask2_depth = effects.Mask2Effect.get_param(effects_param, "switch_mask2_depth")
    if switch_mask2_depth is True:
        dmin = effects.Mask2Effect.get_param(effects_param, "mask2_depth_min") / 255
        dmax = effects.Mask2Effect.get_param(effects_param, "mask2_depth_max") / 255
        if (dmin != 0) or (1 != dmax):
            image = np.where((image < dmin) | (dmax < image), 0, image)

    return image


def _apply_mask_blur(effects_param, image):
    switch_mask2_options = effects.Mask2Effect.get_param(effects_param, "switch_mask2_options")
    blur = effects.Mask2Effect.get_param(effects_param, "mask2_blur")
    if switch_mask2_options is True and blur != 0:
        ksize = int(max(0, blur * 2 - 1))
        image = core.gaussian_blur_cv(image, (ksize, ksize))

    return image


def _draw_hls_mask(ctx, effects_param, mask, hls_str, center_tcg):
    HLS_NUM = {"hue": 0, "lum": 1, "sat": 2}
    HLS_DIS_MAX = {"hue": 179, "lum": 127, "sat": 127}
    HLS_MAX = {"hue": 359, "lum": 255, "sat": 255}

    crop_image_hls = ctx.get_crop_image_hls()
    if crop_image_hls is not None:
        cimg = crop_image_hls[..., HLS_NUM[hls_str]]
        dmax = HLS_DIS_MAX[hls_str]
        mmax = HLS_MAX[hls_str]

        ndis = effects.Mask2Effect.get_param(effects_param, f"mask2_{hls_str}_distance", dmax)
        if ndis != dmax:
            cx, cy = ctx.tcg_to_crop_image(*center_tcg)
            center_n = cimg[int(cy), int(cx)]

            if hls_str == "hue":
                _min = (center_n - ndis) % 360
                _max = (center_n + ndis) % 360
            else:
                ndis = ndis / 255
                _min = (((center_n - ndis) * 65535) % 65536) / 65535
                _max = (((center_n + ndis) * 65535) % 65536) / 65535

            if _min <= _max:
                nimg = np.where((cimg < _min) | (_max < cimg), 0, mask)
            else:
                nimg = np.where(((cimg < _min) & (_max < cimg)), 0, mask)
        else:
            nimg = mask

        _min = effects.Mask2Effect.get_param(effects_param, f"mask2_{hls_str}_min")
        _max = effects.Mask2Effect.get_param(effects_param, f"mask2_{hls_str}_max", mmax)
        if _min != 0 or _max != mmax:
            if hls_str != "hue":
                _min = _min / mmax
                _max = _max / mmax

            if _min <= _max:
                nimg = np.where((cimg < _min) | (_max < cimg), 0, nimg)
            else:
                nimg = np.where(((cimg < _min) & (_max < cimg)), 0, nimg)

        return nimg

    return mask


def _draw_hue_mask(ctx, effects_param, mask, center_tcg):
    if effects.Mask2Effect.get_param(effects_param, "switch_mask2_hue") is True:
        return _draw_hls_mask(ctx, effects_param, mask, "hue", center_tcg)
    return mask


def _draw_lum_mask(ctx, effects_param, mask, center_tcg):
    if effects.Mask2Effect.get_param(effects_param, "switch_mask2_lum") is True:
        return _draw_hls_mask(ctx, effects_param, mask, "lum", center_tcg)
    return mask


def _draw_sat_mask(ctx, effects_param, mask, center_tcg):
    if effects.Mask2Effect.get_param(effects_param, "switch_mask2_sat") is True:
        return _draw_hls_mask(ctx, effects_param, mask, "sat", center_tcg)
    return mask
