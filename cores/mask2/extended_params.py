"""
Mask2 の拡張パラメータ（ぼかし・HLS 等）をマスク画像に適用。BaseMask._apply_extened_params と同等。
"""
from __future__ import annotations

import numpy as np

import cores.core as core
import cores.expand_mask as expand_mask
import effects
import params


def get_mask_hash_tuple(effects_param):
    return (
        effects.Mask2Effect.get_param(effects_param, "switch_mask2_settings"),
        effects.Mask2Effect.get_param(effects_param, "mask2_invert"),
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
    )


def apply_extended_params(ctx, effects_param, image, center_tcg):
    """center_tcg: マスクの中心（TCG）。HLS 範囲の参照点に使う。"""
    simg = _apply_mask_space(ctx, effects_param, image)
    dimg = _apply_depth_mask(effects_param, simg)
    himg = _draw_hue_mask(ctx, effects_param, dimg, center_tcg)
    limg = _draw_lum_mask(ctx, effects_param, himg, center_tcg)
    simg = _draw_sat_mask(ctx, effects_param, limg, center_tcg)
    return _apply_mask_blur(effects_param, simg)


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
