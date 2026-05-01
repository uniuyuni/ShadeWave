"""
Kivy なしでマスクの pmck 復元と get_mask_image を提供する実装。
"""
from __future__ import annotations

import logging

import numpy as np

import cores.core as core
import effects
import params

from cores.mask2 import elliptical_raster, extended_params, gradient_raster
from cores.mask2.exceptions import HeadlessMaskNotSupported
from cores.mask2.mask_types import MaskTypeStr


def _clip_mask_range(image, allow_over_one=False, allow_under_zero=False):
    min_value = None if allow_under_zero else 0
    max_value = None if allow_over_one else 1
    if min_value is None and max_value is None:
        return image
    return np.clip(image, min_value, max_value)


class HeadlessCompositMask:
    def __init__(self, ctx, pipeline):
        self.ctx = ctx
        self.pipeline = pipeline
        self.name = "Composit"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.mask_list = []
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return True

    def deserialize(self, d):
        self.name = d["name"]
        self.effects_param.update(d["effects_param"])
        self.mask_list.clear()
        for i, mask_info in enumerate(d["mask_list"]):
            subd = mask_info[0]
            op = mask_info[1]
            child = self.pipeline.instantiate_mask_from_dict(subd)
            self.add_mask(child, op, i)

    def add_mask(self, mask, maskop="Add", index=0):
        self.mask_list.insert(index, (mask, maskop))

    def get_mask_list(self):
        return self.mask_list

    def get_mask_image(self):
        composit = np.zeros(
            (int(self.ctx.texture_size[1]), int(self.ctx.texture_size[0])),
            dtype=np.float32,
        )
        allow_over_one = False
        allow_under_zero = False
        for mask, maskop in reversed(self.mask_list):
            mimage = mask.get_mask_image()
            mask_allow_over_one = False
            mask_allow_under_zero = False
            match maskop:
                case "Add":
                    composit = _clip_mask_range(composit + mimage, mask_allow_over_one, mask_allow_under_zero)
                case "Subtract":
                    composit = _clip_mask_range(composit - mimage, mask_allow_over_one, mask_allow_under_zero)
                case _:
                    logging.error("Unknown mask operation: %s", maskop)
                    raise ValueError(maskop)
        return composit


class HeadlessFullMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Full"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.center = (0.0, 0.0)
        self.initializing = False
        self.image_mask_cache = None
        self.image_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def deserialize(self, d):
        self.initializing = False
        cx, cy = d["center"]
        self.name = d["name"]
        self.effects_param.update(d["effects_param"])
        self.center = params.denorm_param(self.effects_param, (cx, cy))

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        center = self.ctx.tcg_to_texture(*self.center)
        newhash = hash(
            (
                self.get_hash_items(),
                self.ctx.get_hash_items(),
                image_size,
                center,
            )
        )
        if (
            self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        ) and self.initializing is False:
            gradient_image = np.ones((image_size[1], image_size[0]), dtype=np.float32)
            gradient_image = extended_params.apply_extended_params(
                self.ctx, self.effects_param, gradient_image, self.center
            )
            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
        return (
            self.image_mask_cache
            if self.image_mask_cache is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


class HeadlessCircularGradientMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Circle"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.center = (0.0, 0.0)
        self.inner_radius_x = self.inner_radius_y = 0.0
        self.outer_radius_x = self.outer_radius_y = 0.0
        self.rotate_rad = 0.0
        self.initializing = False
        self.image_mask_cache = None
        self.image_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def deserialize(self, d):
        self.initializing = False
        self.name = d["name"]
        cx, cy = d["center"]
        ix, iy = d["inner_radius"]
        ox, oy = d["outer_radius"]
        self.rotate_rad = d["rotate_rad"]
        self.effects_param.update(d["effects_param"])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        self.inner_radius_x, self.inner_radius_y = params.denorm_param(
            self.effects_param, (ix, iy)
        )
        self.outer_radius_x, self.outer_radius_y = params.denorm_param(
            self.effects_param, (ox, oy)
        )

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        center = self.ctx.tcg_to_texture(*self.center)
        inner_axes = self.ctx.tcg_to_image_scale(self.inner_radius_x, self.inner_radius_y)
        outer_axes = self.ctx.tcg_to_image_scale(self.outer_radius_x, self.outer_radius_y)
        rotate_rad = self.ctx.get_rotate_rad(self.rotate_rad)
        if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_settings") is True:
            invert = not effects.Mask2Effect.get_param(self.effects_param, "mask2_invert")
        else:
            invert = False

        newhash = hash(
            (
                self.get_hash_items(),
                self.ctx.get_hash_items(),
                image_size,
                center,
                inner_axes,
                outer_axes,
                rotate_rad,
                invert,
            )
        )
        if (
            self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        ) and self.initializing is False:
            gradient_image = elliptical_raster.draw_elliptical_gradient(
                image_size,
                center,
                inner_axes,
                outer_axes,
                rotate_rad,
                invert,
                1.5,
            )
            gradient_image = extended_params.apply_extended_params(
                self.ctx, self.effects_param, gradient_image, self.center
            )
            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return (
            self.image_mask_cache
            if self.image_mask_cache is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


class HeadlessGradientMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Line"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.start_point = [0.0, 0.0]
        self.end_point = [0.0, 0.0]
        self.center = [0.0, 0.0]
        self.initializing = False
        self.image_mask_cache = None
        self.image_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def deserialize(self, d):
        self.initializing = False
        self.name = d["name"]
        sx, sy = d["start_point"]
        ex, ey = d["end_point"]
        self.effects_param.update(d["effects_param"])
        self.start_point = list(params.denorm_param(self.effects_param, (sx, sy)))
        self.end_point = list(params.denorm_param(self.effects_param, (ex, ey)))
        self.center = [
            (self.start_point[0] + self.end_point[0]) / 2,
            (self.start_point[1] + self.end_point[1]) / 2,
        ]

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        center = self.ctx.tcg_to_texture(*self.center)
        start_point = self.ctx.tcg_to_texture(*self.start_point)
        end_point = self.ctx.tcg_to_texture(*self.end_point)
        if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_settings") is True:
            if effects.Mask2Effect.get_param(self.effects_param, "mask2_invert") is True:
                start_point, end_point = end_point, start_point

        newhash = hash(
            (
                self.get_hash_items(),
                self.ctx.get_hash_items(),
                image_size,
                center,
                start_point,
                end_point,
            )
        )
        if (
            self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        ) and self.initializing is False:
            gradient_image = gradient_raster.draw_linear_gradient(
                image_size, center, start_point, end_point, 1
            )
            gradient_image = extended_params.apply_extended_params(
                self.ctx, self.effects_param, gradient_image, self.center
            )
            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return (
            self.image_mask_cache
            if self.image_mask_cache is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


def instantiate_mask_from_type(ctx, pipeline, mask_type: str):
    mt = str(mask_type)
    if mt == MaskTypeStr.COMPOSIT:
        return HeadlessCompositMask(ctx, pipeline)
    if mt == MaskTypeStr.CIRCULAR:
        return HeadlessCircularGradientMask(ctx)
    if mt == MaskTypeStr.GRADIENT:
        return HeadlessGradientMask(ctx)
    if mt == MaskTypeStr.FULL:
        return HeadlessFullMask(ctx)
    if mt in (
        MaskTypeStr.FREEDRAW,
        MaskTypeStr.SEGMENT,
        MaskTypeStr.DEPTHMAP,
        MaskTypeStr.FACE,
        MaskTypeStr.TARGET_TEXT,
    ):
        from cores.mask2.headless_inference_masks import instantiate_inference_mask

        return instantiate_inference_mask(ctx, mt)
    raise HeadlessMaskNotSupported(f"mask type not implemented for headless: {mt!r}")
