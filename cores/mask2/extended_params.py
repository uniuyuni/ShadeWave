"""
Mask2 の拡張パラメータ（ぼかし・HLS 等）をマスク画像に適用。BaseMask._apply_extened_params と同等。
"""
from __future__ import annotations

import numpy as np
import cv2

import cores.core as core
import cores.expand_mask as expand_mask
import effects
import params
from cores.mask2 import edge_refine

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
    dimg = _apply_depth_mask(effects_param, simg)
    himg = _draw_hue_mask(ctx, effects_param, dimg, center_tcg)
    limg = _draw_lum_mask(ctx, effects_param, himg, center_tcg)
    simg = _draw_sat_mask(ctx, effects_param, limg, center_tcg)
    bimg = _apply_mask_blur(effects_param, simg)
    if edge_support is not None:
        bimg = np.where(edge_support > 0.001, bimg, 0.0)
    return bimg


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
    return max(1.0, float(radius) * disp_scale)


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
