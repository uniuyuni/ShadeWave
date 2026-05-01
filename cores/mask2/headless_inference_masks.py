"""
推論・自由描画マスクのヘッドレス実装（mask_editor2 の get_mask_image と同じ計算経路）。
"""
from __future__ import annotations

import cv2
import numpy as np

import cores.core as core
import effects
import params
import utils.utils as utils

from cores.mask2 import extended_params, inference_runtime
from cores.mask2.freedraw_raster import Line, draw_line_texture
from cores.mask2.mask_types import MaskTypeStr


class HeadlessFreeDrawMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Draw"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.lines: list = []
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
        self.name = d["name"]
        cx, cy = d["center"]
        self.effects_param.update(d["effects_param"])
        lines = []
        for line in d["lines"]:
            lineobj = Line(
                is_erasing=line["is_erasing"],
                size=line["size"],
                soft=line["soft"],
            )
            for point in line["points"]:
                lineobj.add_point(*point)
            lines.append(lineobj)
        self.lines = lines
        self.center = params.denorm_param(self.effects_param, (cx, cy))

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        copy_lines = []
        for src_line in self.lines:
            copy_line = Line(
                src_line.is_erasing,
                self.ctx.tcg_to_image_scale(src_line.size, 0)[0],
                src_line.soft,
            )
            for point in src_line.points:
                copy_line.add_point(*self.ctx.tcg_to_texture(*point))
            copy_lines.append(copy_line)

        line_hash = tuple(
            (line.is_erasing, line.size, line.soft, tuple(line.points))
            for line in self.lines
        )
        newhash = hash(
            (self.get_hash_items(), self.ctx.get_hash_items(), image_size, line_hash)
        )
        if (
            self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        ) and not self.initializing:
            allow_over_one = False
            allow_under_zero = False
            mask = draw_line_texture(
                image_size,
                copy_lines,
                allow_over_one=allow_over_one,
                allow_under_zero=allow_under_zero,
            )
            mask = extended_params.apply_extended_params(
                self.ctx, self.effects_param, mask, self.center
            )
            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return (
            self.image_mask_cache
            if self.image_mask_cache is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


class HeadlessSegmentMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Segment"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.center = (0.0, 0.0)
        self.corner = (0.0, 0.0)
        self.initializing = False
        self.image_mask_cache = None
        self.image_mask_cache_hash = None
        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def deserialize(self, d):
        self.initializing = False
        cx, cy = d["center"]
        crx, cry = d.get("corner", (cx, cy))
        self.name = d["name"]
        self.effects_param.update(d["effects_param"])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        self.corner = params.denorm_param(self.effects_param, (crx, cry))
        self.image_mask_cache = d.get("image_mask_cache", None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = d.get("image_mask_cache_hash", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        center = self.ctx.tcg_to_original_image(*self.center)
        corner = self.ctx.tcg_to_original_image(*self.corner)
        if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_settings") is True:
            invert = effects.Mask2Effect.get_param(self.effects_param, "mask2_invert")
        else:
            invert = False
        segment_mask = None

        newhash = hash((image_size, center, corner))
        if self.image_mask_cache_hash != newhash and not self.initializing:
            self.image_mask_cache_hash = newhash
            cx, cy = center
            crx, cry = corner
            min_x = min(cx, crx)
            min_y = min(cy, cry)
            w = abs(cx - crx)
            h = abs(cy - cry)
            img = self.ctx.get_original_image_rgb()
            segment_mask = inference_runtime.predict_sam3_bbox(
                img, [min_x, min_y, w, h], invert
            )
            self.image_mask_cache = segment_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items()))
        if (
            self.image_mask_cache is not None
            and (
                self.image_mask_cache is segment_mask
                or self.segment_mask_cache is None
                or self.segment_mask_cache_hash != newhash2
            )
            and not self.initializing
        ):
            self.segment_mask_cache_hash = newhash2
            segment_mask = self.image_mask_cache
            disp_info, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            segment_mask = core.rotation(
                segment_mask,
                np.rad2deg(rotate_rad),
                flip,
                np.array(matrix).reshape(3, 3),
            )
            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(
                *self.ctx.texture_size, disp_info
            )
            cx2, cy2, cw, ch, scale = disp_info
            segment_mask = cv2.resize(
                segment_mask[cy2 : cy2 + ch, cx2 : cx2 + cw], (nw, nh)
            )
            segment_mask = np.pad(
                segment_mask,
                (
                    (oy, self.ctx.texture_size[1] - (oy + nh)),
                    (ox, self.ctx.texture_size[0] - (ox + nw)),
                ),
                constant_values=0,
            )
            segment_mask = extended_params.apply_extended_params(
                self.ctx, self.effects_param, segment_mask, self.center
            )
            self.segment_mask_cache = segment_mask

        if segment_mask is None:
            segment_mask = self.segment_mask_cache

        return (
            segment_mask
            if segment_mask is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


class HeadlessDepthMapMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Depth Map"
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
        self.depth_map_mask_cache = None
        self.depth_map_mask_cache_hash = None
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
        self.image_mask_cache = d.get("image_mask_cache", None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = d.get("image_mask_cache_hash", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        depth_map_mask = None

        newhash = hash((image_size,))
        if (
            self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        ) and not self.initializing:
            self.image_mask_cache_hash = newhash
            img = self.ctx.get_original_image_rgb()
            depth_map_mask = inference_runtime.predict_depth_map(img)
            self.image_mask_cache = depth_map_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items()))
        if (
            self.image_mask_cache is not None
            and (
                self.image_mask_cache is depth_map_mask
                or self.depth_map_mask_cache is None
                or self.depth_map_mask_cache_hash != newhash2
            )
            and not self.initializing
        ):
            self.depth_map_mask_cache_hash = newhash2
            depth_map_mask = self.image_mask_cache
            disp_info, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            depth_map_mask = core.rotation(
                depth_map_mask, rotate_rad, flip, np.array(matrix).reshape(3, 3)
            )
            depth_map_mask = core.crop_image_with_disp_info(depth_map_mask, disp_info)
            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(
                self.ctx.texture_size[0], self.ctx.texture_size[1], disp_info
            )
            cx2, cy2, cw, ch, scale = disp_info
            cx2, cy2, cw, ch = int(cx2 * scale), int(cy2 * scale), int(cw * scale), int(ch * scale)
            depth_map_mask = cv2.resize(
                depth_map_mask[cy2 : cy2 + ch, cx2 : cx2 + cw], (nw, nh)
            )
            depth_map_mask = np.pad(
                depth_map_mask,
                (
                    (oy, self.ctx.texture_size[0] - (oy + nh)),
                    (ox, self.ctx.texture_size[1] - (ox + nw)),
                ),
                constant_values=0,
            )
            depth_map_mask = extended_params.apply_extended_params(
                self.ctx, self.effects_param, depth_map_mask, self.center
            )
            self.depth_map_mask_cache = depth_map_mask

        if depth_map_mask is None:
            depth_map_mask = self.depth_map_mask_cache

        return (
            depth_map_mask
            if depth_map_mask is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


class HeadlessFaceMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Face"
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
        self.faces_mask_cache = None
        self.faces_mask_cache_hash = None
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
        self.image_mask_cache = d.get("image_mask_cache", None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = d.get("image_mask_cache_hash", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        exclude_names = []
        if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_face") is True:
            if effects.Mask2Effect.get_param(self.effects_param, "mask2_face_face") is False:
                exclude_names.append("face")
            if effects.Mask2Effect.get_param(self.effects_param, "mask2_face_brows") is False:
                exclude_names.extend(["rb", "lb"])
            if effects.Mask2Effect.get_param(self.effects_param, "mask2_face_eyes") is False:
                exclude_names.extend(["re", "le"])
            if effects.Mask2Effect.get_param(self.effects_param, "mask2_face_nose") is False:
                exclude_names.append("nose")
            if effects.Mask2Effect.get_param(self.effects_param, "mask2_face_mouth") is False:
                exclude_names.append("imouth")
            if effects.Mask2Effect.get_param(self.effects_param, "mask2_face_lips") is False:
                exclude_names.extend(["ulip", "llip"])
        faces_mask = None

        newhash = hash((image_size, tuple(exclude_names)))
        if (
            self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        ) and not self.initializing:
            self.image_mask_cache_hash = newhash
            img = self.ctx.get_original_image_rgb()
            faces_mask = inference_runtime.predict_face_mask(img, exclude_names)
            self.image_mask_cache = faces_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items()))
        if (
            self.image_mask_cache is not None
            and (
                self.image_mask_cache is faces_mask
                or self.faces_mask_cache is None
                or self.faces_mask_cache_hash != newhash2
            )
            and not self.initializing
        ):
            self.faces_mask_cache_hash = newhash2
            faces_mask = self.image_mask_cache
            disp_info, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            faces_mask = core.rotation(
                faces_mask,
                np.rad2deg(rotate_rad),
                flip,
                np.array(matrix).reshape(3, 3),
            )
            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(
                *self.ctx.texture_size, disp_info
            )
            cx2, cy2, cw, ch, scale = disp_info
            faces_mask = cv2.resize(
                faces_mask[cy2 : cy2 + ch, cx2 : cx2 + cw], (nw, nh)
            )
            faces_mask = np.pad(
                faces_mask,
                (
                    (oy, self.ctx.texture_size[1] - (oy + nh)),
                    (ox, self.ctx.texture_size[0] - (ox + nw)),
                ),
                constant_values=0,
            )
            faces_mask = extended_params.apply_extended_params(
                self.ctx, self.effects_param, faces_mask, self.center
            )
            self.faces_mask_cache = faces_mask

        if faces_mask is None:
            faces_mask = self.faces_mask_cache

        return (
            faces_mask
            if faces_mask is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


class HeadlessTargetTextMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Target Text"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.center = (0.0, 0.0)
        self.target_text = ""
        self.initializing = False
        self.image_mask_cache = None
        self.image_mask_cache_hash = None
        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def deserialize(self, d):
        self.initializing = False
        cx, cy = d["center"]
        self.name = d["name"]
        self.target_text = d.get("target_text", "All")
        self.effects_param.update(d["effects_param"])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        self.image_mask_cache = d.get("image_mask_cache", None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = d.get("image_mask_cache_hash", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_settings") is True:
            invert = effects.Mask2Effect.get_param(self.effects_param, "mask2_invert")
        else:
            invert = False
        text = self.target_text
        segment_mask = None

        newhash = hash((image_size, text))
        if self.image_mask_cache_hash != newhash and not self.initializing:
            self.image_mask_cache_hash = newhash
            img = self.ctx.get_original_image_rgb()
            segment_mask = inference_runtime.predict_sam3_text(img, text, invert)
            self.image_mask_cache = segment_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items()))
        if (
            self.image_mask_cache is not None
            and (
                self.image_mask_cache is segment_mask
                or self.segment_mask_cache is None
                or self.segment_mask_cache_hash != newhash2
            )
            and not self.initializing
        ):
            self.segment_mask_cache_hash = newhash2
            segment_mask = self.image_mask_cache
            disp_info, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            segment_mask = core.rotation(
                segment_mask,
                np.rad2deg(rotate_rad),
                flip,
                np.array(matrix).reshape(3, 3),
            )
            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(
                *self.ctx.texture_size, disp_info
            )
            cx2, cy2, cw, ch, scale = disp_info
            segment_mask = cv2.resize(
                segment_mask[cy2 : cy2 + ch, cx2 : cx2 + cw], (nw, nh)
            )
            segment_mask = np.pad(
                segment_mask,
                (
                    (oy, self.ctx.texture_size[1] - (oy + nh)),
                    (ox, self.ctx.texture_size[0] - (ox + nw)),
                ),
                constant_values=0,
            )
            segment_mask = extended_params.apply_extended_params(
                self.ctx, self.effects_param, segment_mask, self.center
            )
            self.segment_mask_cache = segment_mask

        if segment_mask is None:
            segment_mask = self.segment_mask_cache

        return (
            segment_mask
            if segment_mask is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


def instantiate_inference_mask(ctx, mask_type: str):
    mt = str(mask_type)
    if mt == MaskTypeStr.FREEDRAW:
        return HeadlessFreeDrawMask(ctx)
    if mt == MaskTypeStr.SEGMENT:
        return HeadlessSegmentMask(ctx)
    if mt == MaskTypeStr.DEPTHMAP:
        return HeadlessDepthMapMask(ctx)
    if mt == MaskTypeStr.FACE:
        return HeadlessFaceMask(ctx)
    if mt == MaskTypeStr.TARGET_TEXT:
        return HeadlessTargetTextMask(ctx)
    raise ValueError(f"unknown inference mask type: {mt!r}")
