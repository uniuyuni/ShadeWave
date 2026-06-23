"""
Kivy MaskEditor2 の代替。export / 別プロセス向け。
"""
from __future__ import annotations

from cores.mask2.coordinate_context import Mask2CoordinateContext
from cores.mask2.headless_masks import instantiate_mask_from_type
from cores.ai_image_cache import AIImageCache


def _normalize_mask_type(t) -> str:
    if isinstance(t, str):
        return t
    return getattr(t, "value", str(t))


class Mask2HeadlessPipeline:
    """params.deserialize / pipeline.export_pipeline が期待する API に合わせる。"""

    def __init__(self):
        self.ctx = Mask2CoordinateContext()
        self.mask_list = []
        self.ai_image_cache = AIImageCache()

    def set_ai_image_cache(self, cache):
        self.ai_image_cache = cache if cache is not None else AIImageCache()

    def set_serialized_ai_image_cache(self, serialized):
        self.ai_image_cache.deserialize(serialized)

    def serialize_ai_image_cache(self):
        return self.ai_image_cache.serialize()

    def get_ai_depth_map(self, cache_key, compute_func):
        return self.ai_image_cache.get_depth_map(cache_key, compute_func)

    def set_texture_size(self, tx, ty):
        self.ctx.set_texture_size(tx, ty)

    def set_primary_param(self, primary_param, disp_info):
        self.ctx.set_primary_param(primary_param, disp_info)

    def set_ref_image(self, crop_image, original_image=None):
        self.ctx.set_ref_image(crop_image, original_image)

    def update(self):
        pass

    def clear_mask(self):
        self.mask_list.clear()

    def deserialize(self, d):
        self.set_serialized_ai_image_cache(d.get("ai_image_cache"))
        ml = d.get("mask2")
        if not ml:
            self.clear_mask()
            return
        self.clear_mask()
        for raw in ml:
            m = self.instantiate_mask_from_dict(raw)
            self.mask_list.append(m)

    def instantiate_mask_from_dict(self, raw):
        t = _normalize_mask_type(raw.get("type"))
        m = instantiate_mask_from_type(self.ctx, self, t)
        m.deserialize(raw)
        return m

    def get_mask_list(self):
        return self.mask_list

    def serialize(self):
        return None
