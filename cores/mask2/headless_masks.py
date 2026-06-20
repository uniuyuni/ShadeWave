"""
Kivy なしでマスクの pmck 復元と get_mask_image を提供する実装。
"""
from __future__ import annotations

import logging

import numpy as np

import cores.core as core
import effects
import params
import utils.utils as utils

from cores.mask2 import mask_rasters
from cores.mask2.mask_rasters import Line, draw_line_texture
from cores.mask2.mask_rasters import Polyline as RasterPolyline, draw_polyline_texture
from cores.mask2.exceptions import HeadlessMaskNotSupported
from cores.mask2 import cache_keys, edge_refine, extended_params, inference_runtime
from cores.mask2.mask_types import MaskTypeStr
from cores.mask2 import mask_geometry as mask_geometry_mod
from cores.mask2.mask_mesh import apply_mask_mesh_warp as _apply_mask_mesh_warp_shared


def _clip_mask_range(image, allow_over_one=False, allow_under_zero=False):
    min_value = None if allow_under_zero else 0
    max_value = None if allow_over_one else 1
    if min_value is None and max_value is None:
        return image
    return np.clip(image, min_value, max_value)


def _fit_image_mask_to_texture(ctx, image):
    texture_w = int(ctx.texture_size[0])
    texture_h = int(ctx.texture_size[1])
    if image is None or texture_w <= 0 or texture_h <= 0:
        return np.zeros((max(texture_h, 0), max(texture_w, 0)), dtype=np.float32)

    fitted = extended_params._fit_image_to_texture(ctx, image, (texture_h, texture_w))
    if fitted is None:
        return np.zeros((texture_h, texture_w) + image.shape[2:], dtype=image.dtype)
    return fitted


def _apply_mask_mesh_warp(composit, ctx, effects_param):
    """共通ヘルパ cores.mask2.mask_mesh.apply_mask_mesh_warp への薄いラッパ。
    mask_mesh_link_to_image=True の Composit は画像 mesh の CP を都度参照する。"""
    orig = ctx.tcg_info.get('original_img_size') if isinstance(ctx.tcg_info, dict) else None
    # t2t=texture px (disp_info込み), t2f=フル画像px (F=MLS構築空間)。マスク warp の
    # 共役 (射影込みでズーム位置がズレないため) に両方必要。
    t2t = getattr(ctx, 'tcg_to_texture', None)
    t2f = getattr(ctx, 'tcg_to_full_image', None)
    linked = effects.Mask2Effect.get_param(effects_param, 'mask_mesh_link_to_image')
    if linked and getattr(ctx, 'primary_param', None) is not None:
        merged = dict(effects_param)
        merged['mask_mesh_control_points'] = ctx.primary_param.get('control_points', {})
        merged['mask_mesh_size'] = ctx.primary_param.get('mesh_size', [4, 4])
        return _apply_mask_mesh_warp_shared(composit, merged, orig, t2t, t2f)
    return _apply_mask_mesh_warp_shared(composit, effects_param, orig, t2t, t2f)


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
        # 古いファイル互換: mask_mesh_link_to_image が未設定なら、自前 CP の有無で
        # 判定 (空 → linked, あり → local 保持)。
        if 'mask_mesh_link_to_image' not in d.get('effects_param', {}):
            self.effects_param['mask_mesh_link_to_image'] = \
                not bool(d.get('effects_param', {}).get('mask_mesh_control_points'))
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
        # mask Geometry: この Composit の mask Geom matrix を tcg_info['matrix'] に
        # 一時的に乗せて、子マスクの座標変換に含めるよう差し替え。finally で必ず復元。
        # widgets/mask_editor2.py CompositMask.get_mask_image (line 729-813) と同じ流儀。
        ctx = self.ctx
        saved_matrix = ctx.tcg_info["matrix"] if (ctx.tcg_info is not None) else None
        base = getattr(ctx, "_image_only_matrix", None)
        if base is not None and ctx.tcg_info is not None:
            if mask_geometry_mod.is_enabled(self.effects_param):
                M_mask = mask_geometry_mod.build_matrix_tcg(
                    self.effects_param, ctx.tcg_info["original_img_size"]
                )
                ctx.tcg_info["matrix"] = M_mask @ base
            else:
                ctx.tcg_info["matrix"] = base.copy()
        try:
            composit = np.zeros(
                (int(ctx.texture_size[1]), int(ctx.texture_size[0])),
                dtype=np.float32,
            )
            for mask, maskop in reversed(self.mask_list):
                # follows_mask_geometry()==False のマスク (Segment/Depth/Face/Text 等
                # 推論系) は image-only matrix で実行する
                if getattr(mask, "follows_mask_geometry", lambda: True)():
                    mimage = mask.get_mask_image()
                else:
                    mimage = ctx._call_with_image_only_matrix(mask.get_mask_image)
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
            # mask Mesh warp: 合成済 composit に非線形 TPS 変形を適用 (空なら no-op)
            composit = _apply_mask_mesh_warp(composit, ctx, self.effects_param)
            return composit
        finally:
            if saved_matrix is not None and ctx.tcg_info is not None:
                ctx.tcg_info["matrix"] = saved_matrix


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
                self.ctx,
                self.effects_param,
                gradient_image,
                self.center,
                fill_grown_region=True,
                seed_from_guide=True,
                edge_refine_enabled=False,
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
            gradient_image = mask_rasters.draw_elliptical_gradient(
                image_size,
                center,
                inner_axes,
                outer_axes,
                rotate_rad,
                invert,
                1.5,
            )
            gradient_image = extended_params.apply_extended_params(
                self.ctx,
                self.effects_param,
                gradient_image,
                self.center,
                fill_grown_region=False,
                seed_from_guide=True,
                edge_refine_enabled=False,
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
            gradient_image = mask_rasters.draw_linear_gradient(
                image_size, center, start_point, end_point, 1
            )
            gradient_image = extended_params.apply_extended_params(
                self.ctx,
                self.effects_param,
                gradient_image,
                self.center,
                fill_grown_region=False,
                seed_from_guide=True,
                edge_refine_enabled=False,
            )
            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return (
            self.image_mask_cache
            if self.image_mask_cache is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


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
            full_refined = extended_params.render_freedraw_edge_refine_full_view(
                self.ctx,
                self.effects_param,
                self.lines,
                self.center,
                mask.shape,
                debug_label="HeadlessFreeDrawMaskFull",
            )
            if full_refined is None:
                mask = extended_params.apply_extended_params(
                    self.ctx,
                    self.effects_param,
                    mask,
                    self.center,
                    fill_grown_region=True,
                    seed_mask=edge_refine.make_confident_seed(mask),
                    edge_refine_debug_label="HeadlessFreeDrawMask",
                    edge_refine_selection_strategy=edge_refine.STRATEGY_DRAW,
                    edge_refine_draw_strokes=copy_lines,
                )
            else:
                mask = full_refined
            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return (
            self.image_mask_cache
            if self.image_mask_cache is not None
            else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        )


class HeadlessPolylineMask:
    def __init__(self, ctx):
        self.ctx = ctx
        self.name = "Polyline"
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, ctx.get_image_size())
        params.set_temperature_to_param(
            self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0))
        )
        self.polylines: list = []
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
        polys = []
        for p in d.get("polylines", []):
            polyobj = RasterPolyline(
                is_erasing=p.get("is_erasing", False),
                size=p.get("size", 10),
                soft=p.get("soft", 100),
                is_closed=p.get("is_closed", False),
                is_filled=p.get("is_filled", True),
            )
            for point in p.get("points", []):
                polyobj.add_point(*point)
            polys.append(polyobj)
        self.polylines = polys
        self.center = params.denorm_param(self.effects_param, (cx, cy))

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        copy_polys = []
        for src in self.polylines:
            tex_poly = RasterPolyline(
                is_erasing=src.is_erasing,
                size=self.ctx.tcg_to_image_scale(src.size, 0)[0],
                soft=src.soft,
                is_closed=src.is_closed,
                is_filled=src.is_filled and src.is_closed,
            )
            for point in src.points:
                tex_poly.add_point(*self.ctx.tcg_to_texture(*point))
            copy_polys.append(tex_poly)

        poly_hash = tuple(
            (p.is_erasing, p.size, p.soft, p.is_closed, p.is_filled, tuple(p.points))
            for p in self.polylines
        )
        newhash = hash(
            (self.get_hash_items(), self.ctx.get_hash_items(), image_size, poly_hash)
        )
        if (
            self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        ) and not self.initializing:
            mask = draw_polyline_texture(
                image_size,
                copy_polys,
                allow_over_one=False,
                allow_under_zero=False,
            )
            mask = extended_params.apply_extended_params(
                self.ctx,
                self.effects_param,
                mask,
                self.center,
                fill_grown_region=True,
                seed_mask=edge_refine.make_confident_seed(mask),
                edge_refine_debug_label="HeadlessPolylineMask",
                edge_refine_selection_strategy=edge_refine.STRATEGY_DRAW,
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
        self.image_mask_cache_key = None
        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def follows_mask_geometry(self):
        # Segment は SAM 推論を original_image 空間で実行するため、
        # mask Geometry の matrix swap には追従しない (image-only matrix で動作)。
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
            self.image_mask_cache_key = d.get("image_mask_cache_key", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        original_image_size = tuple(self.ctx.get_image_size())
        center = self.ctx.tcg_to_original_image(*self.center)
        corner = self.ctx.tcg_to_original_image(*self.corner)
        if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_settings") is True:
            invert = effects.Mask2Effect.get_param(self.effects_param, "mask2_invert")
        else:
            invert = False
        segment_mask = None

        cache_key = cache_keys.segment_cache_key(original_image_size, center, corner, False)
        if (
            self.image_mask_cache is None or self.image_mask_cache_key != cache_key
        ) and not self.initializing:
            self.image_mask_cache_key = cache_key
            cx, cy = center
            crx, cry = corner
            min_x = min(cx, crx)
            min_y = min(cy, cry)
            w = abs(cx - crx)
            h = abs(cy - cry)
            img = self.ctx.get_original_image_rgb()
            segment_mask = inference_runtime.predict_sam3_bbox(
                img, [min_x, min_y, w, h], False
            )
            self.image_mask_cache = segment_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items(), image_size))
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
            if invert:
                segment_mask = 1.0 - segment_mask
            _, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            segment_mask = core.rotation(
                segment_mask,
                np.rad2deg(rotate_rad),
                flip,
                np.array(matrix).reshape(3, 3),
            )
            segment_mask = _fit_image_mask_to_texture(self.ctx, segment_mask)
            segment_mask = extended_params.apply_extended_params(
                self.ctx,
                self.effects_param,
                segment_mask,
                self.center,
                edge_refine_support_softness=1.0,
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
        self.image_mask_cache_key = None
        self.depth_map_mask_cache = None
        self.depth_map_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def follows_mask_geometry(self):
        # DepthMap は深度推論を original_image 空間で実行し、結果を core.rotation/crop
        # で配置する。mask Geometry の matrix swap には追従しない。
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
            self.image_mask_cache_key = d.get("image_mask_cache_key", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        original_image_size = tuple(self.ctx.get_image_size())
        depth_map_mask = None

        cache_key = cache_keys.depth_cache_key(
            original_image_size, inference_runtime.DEPTH_MAP_ALGORITHM_VERSION
        )
        if (
            self.image_mask_cache is None or self.image_mask_cache_key != cache_key
        ) and not self.initializing:
            self.image_mask_cache_key = cache_key
            img = self.ctx.get_original_image_rgb()
            depth_map_mask = inference_runtime.predict_depth_map(img)
            self.image_mask_cache = depth_map_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items(), image_size))
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
            _, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            depth_map_mask = core.rotation(
                depth_map_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3)
            )
            depth_map_mask = _fit_image_mask_to_texture(self.ctx, depth_map_mask)
            if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_settings") is True:
                if effects.Mask2Effect.get_param(self.effects_param, "mask2_invert") is True:
                    depth_map_mask = 1.0 - depth_map_mask
            depth_map_mask = extended_params.apply_extended_params(
                self.ctx,
                self.effects_param,
                depth_map_mask,
                self.center,
                edge_refine_support_softness=1.0,
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
        self.image_mask_cache_key = None
        self.faces_mask_cache = None
        self.faces_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def follows_mask_geometry(self):
        # Face は顔検出を original_image 空間で実行し、結果を core.rotation/crop
        # で配置する。mask Geometry の matrix swap には追従しない。
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
            self.image_mask_cache_key = d.get("image_mask_cache_key", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        original_image_size = tuple(self.ctx.get_image_size())
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

        cache_key = cache_keys.face_cache_key(original_image_size, exclude_names)
        if (
            self.image_mask_cache is None or self.image_mask_cache_key != cache_key
        ) and not self.initializing:
            self.image_mask_cache_key = cache_key
            img = self.ctx.get_original_image_rgb()
            faces_mask = inference_runtime.predict_face_mask(img, exclude_names)
            self.image_mask_cache = faces_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items(), image_size))
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
            _, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            faces_mask = core.rotation(
                faces_mask,
                np.rad2deg(rotate_rad),
                flip,
                np.array(matrix).reshape(3, 3),
            )
            faces_mask = _fit_image_mask_to_texture(self.ctx, faces_mask)
            faces_mask = extended_params.apply_extended_params(
                self.ctx,
                self.effects_param,
                faces_mask,
                self.center,
                edge_refine_support_softness=1.0,
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
        self.image_mask_cache_key = None
        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None
        self.is_draw_mask = True
        self.do_draw_composit_mask = True

    def is_composit(self):
        return False

    def follows_mask_geometry(self):
        # TargetText は SAM3 テキスト推論を original_image 空間で実行する。
        # mask Geometry の matrix swap には追従しない。
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
            self.image_mask_cache_key = d.get("image_mask_cache_key", None)

    def get_hash_items(self):
        return extended_params.get_mask_hash_tuple(self.effects_param)

    def get_mask_image(self):
        image_size = (int(self.ctx.texture_size[0]), int(self.ctx.texture_size[1]))
        original_image_size = tuple(self.ctx.get_image_size())
        if effects.Mask2Effect.get_param(self.effects_param, "switch_mask2_settings") is True:
            invert = effects.Mask2Effect.get_param(self.effects_param, "mask2_invert")
        else:
            invert = False
        text = self.target_text
        segment_mask = None

        cache_key = cache_keys.target_text_cache_key(original_image_size, text, False)
        if (
            self.image_mask_cache is None or self.image_mask_cache_key != cache_key
        ) and not self.initializing:
            self.image_mask_cache_key = cache_key
            img = self.ctx.get_original_image_rgb()
            segment_mask = inference_runtime.predict_sam3_text(img, text, False)
            self.image_mask_cache = segment_mask

        newhash2 = hash((self.get_hash_items(), self.ctx.get_hash_items(), image_size))
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
            if invert:
                segment_mask = 1.0 - segment_mask
            _, rotate_rad, flip, matrix = self.ctx.get_hash_items()
            segment_mask = core.rotation(
                segment_mask,
                np.rad2deg(rotate_rad),
                flip,
                np.array(matrix).reshape(3, 3),
            )
            segment_mask = _fit_image_mask_to_texture(self.ctx, segment_mask)
            segment_mask = extended_params.apply_extended_params(
                self.ctx,
                self.effects_param,
                segment_mask,
                self.center,
                edge_refine_support_softness=1.0,
            )
            self.segment_mask_cache = segment_mask

        if segment_mask is None:
            segment_mask = self.segment_mask_cache

        return (
            segment_mask
            if segment_mask is not None
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
    if mt == MaskTypeStr.FREEDRAW:
        return HeadlessFreeDrawMask(ctx)
    if mt == MaskTypeStr.POLYLINE:
        return HeadlessPolylineMask(ctx)
    if mt == MaskTypeStr.SEGMENT:
        return HeadlessSegmentMask(ctx)
    if mt == MaskTypeStr.DEPTHMAP:
        return HeadlessDepthMapMask(ctx)
    if mt == MaskTypeStr.FACE:
        return HeadlessFaceMask(ctx)
    if mt == MaskTypeStr.TARGET_TEXT:
        return HeadlessTargetTextMask(ctx)
    
    raise HeadlessMaskNotSupported(f"mask type not implemented for headless: {mt!r}")
