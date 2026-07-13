"""AI マスクビットマップ共有ストア(cores/ai_image_cache.py AIImageCache 拡張)のテスト。

widgets/mask_editor2.py(GUI 側)は kivy Window/SDL に依存しヘッドレスで import できない
(tests/test_draw_brush_cursor_rotation_flow.py 等、既存テストの前例のとおり)ため、GUI の
serialize()/deserialize() の実装内容は AST ソース検証で構造的に確認する
(tests/test_ai_image_cache_flow.py と同じ手法)。それ以外(ストア本体・headless マスク・
params.py のサイドカー往復)は実際にインポート/実行して検証する。
"""
import ast
import os
import pathlib
import sys
import unittest

import msgpack
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import params
from cores.ai_image_cache import AIImageCache
from cores.mask2 import cache_keys, inference_runtime
from cores.mask2.coordinate_context import Mask2CoordinateContext
from cores.mask2.headless_masks import HeadlessFaceMask, HeadlessSegmentMask
from cores.mask2.headless_pipeline import Mask2HeadlessPipeline

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MASK_EDITOR_PATH = PROJECT_ROOT / "widgets" / "mask_editor2.py"


def _class_method_source(path, class_name, method_name):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(source, item)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _make_ctx(image, texture_size=None):
    h, w = image.shape[:2]
    ctx = Mask2CoordinateContext()
    ctx.set_texture_size(*(texture_size or (w, h)))
    primary = {
        "original_img_size": (w, h),
        "img_size": (w, h),
        "disp_info": (0, 0, w, h, 1.0),
        "rotation": 0,
        "rotation2": 0,
        "flip_mode": 0,
        "matrix": np.eye(3),
    }
    ctx.set_primary_param(primary, primary["disp_info"])
    ctx.set_ref_image(image, image)
    return ctx


# --------------------------------------------------------------------------
# 1. AIImageCache: put/get/sweep の基本動作
# --------------------------------------------------------------------------
class AIImageCacheMaskBitmapTest(unittest.TestCase):
    def test_put_get_and_sweep_removes_only_unreferenced_keys(self):
        cache = AIImageCache()
        key_a = cache_keys.segment_cache_key((10, 10), [1.0, 1.0], [2.0, 2.0], False)
        key_b = cache_keys.segment_cache_key((10, 10), [3.0, 3.0], [4.0, 4.0], False)
        img_a = np.ones((2, 2), dtype=np.float32)
        img_b = np.zeros((2, 2), dtype=np.float32)

        cache.put_mask_bitmap(key_a, img_a)
        cache.put_mask_bitmap(key_b, img_b)

        self.assertIs(cache.get_mask_bitmap(key_a), img_a)
        self.assertIs(cache.get_mask_bitmap(key_b), img_b)
        self.assertIsNone(cache.get_mask_bitmap(["not-a-key"]))

        # key_a と等価な「別のリストインスタンス」でも同じエントリを引ける(値で比較)こと
        self.assertIs(cache.get_mask_bitmap(list(key_a)), img_a)

        result = cache.sweep_mask_bitmaps([key_a])

        self.assertEqual(1, result["mask_bitmap_entries"])
        self.assertIs(cache.get_mask_bitmap(key_a), img_a)
        self.assertIsNone(cache.get_mask_bitmap(key_b))

    def test_get_serialized_mask_bitmap_is_lazy_and_not_recompressed(self):
        cache = AIImageCache()
        key = cache_keys.segment_cache_key((4, 4), [0.0, 0.0], [1.0, 1.0], False)
        self.assertIsNone(cache.get_serialized_mask_bitmap(key))

        cache.put_mask_bitmap(key, np.array([[0.1, 0.5], [0.9, 0.2]], dtype=np.float32))

        first = cache.get_serialized_mask_bitmap(key)
        second = cache.get_serialized_mask_bitmap(key)
        self.assertIsNotNone(first)
        self.assertIs(first, second, "同一キーへの2回目の呼び出しは再圧縮せず同一オブジェクトを返すこと")

        # put で上書きすると圧縮キャッシュは破棄され、再生成される
        cache.put_mask_bitmap(key, np.array([[0.7, 0.7], [0.7, 0.7]], dtype=np.float32))
        third = cache.get_serialized_mask_bitmap(key)
        self.assertIsNotNone(third)
        self.assertIsNot(third, first)

    def test_clear_removes_mask_bitmaps_clear_transient_keeps_them(self):
        cache = AIImageCache()
        key = cache_keys.segment_cache_key((4, 4), [0.0, 0.0], [1.0, 1.0], False)
        cache.put_mask_bitmap(key, np.zeros((2, 2), dtype=np.float32))
        cache.get_depth_map(["depth"], lambda: np.zeros((2, 2), dtype=np.float32))

        cache.clear_transient()

        self.assertIsNotNone(cache.get_mask_bitmap(key), "clear_transient はマスクビットマップを残すこと")
        self.assertIsNone(cache.peek_depth_map(["depth"]), "clear_transient は深度マップを消すこと")

        cache.clear()

        self.assertIsNone(cache.get_mask_bitmap(key), "clear はマスクビットマップも全消去すること")


# --------------------------------------------------------------------------
# 2. headless Segment/Face マスク: ストア経由の推論スキップ
# --------------------------------------------------------------------------
class HeadlessMaskStoreReuseTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 32, 3), dtype=np.float32)
        self.image[:, :16] = (0.2, 0.6, 0.2)
        self.image[:, 16:] = (0.6, 0.2, 0.6)
        self.ctx = _make_ctx(self.image)
        self.pipeline = Mask2HeadlessPipeline()
        self.pipeline.ctx = self.ctx

    def test_second_segment_mask_with_same_key_skips_inference(self):
        calls = []
        original = inference_runtime.predict_sam3_bbox

        def fake_predict(img, bbox, invert):
            calls.append(bbox)
            return np.full(img.shape[:2], 0.5, dtype=np.float32)

        inference_runtime.predict_sam3_bbox = fake_predict
        try:
            m1 = HeadlessSegmentMask(self.ctx, self.pipeline)
            m1.initializing = False
            m1.center = (-8.0, 0.0)
            m1.corner = (8.0, 8.0)

            m2 = HeadlessSegmentMask(self.ctx, self.pipeline)
            m2.initializing = False
            m2.center = (-8.0, 0.0)
            m2.corner = (8.0, 8.0)

            img1 = m1.get_mask_image()
            img2 = m2.get_mask_image()
        finally:
            inference_runtime.predict_sam3_bbox = original

        self.assertEqual(1, len(calls), "同一キーの2マスク目はストア参照で推論をスキップすること")
        self.assertIsNotNone(img1)
        self.assertIsNotNone(img2)
        # ストアには1エントリだけ入っていること
        self.assertEqual(m1.image_mask_cache_key, m2.image_mask_cache_key)
        self.assertIsNotNone(self.pipeline.get_ai_mask_bitmap(m1.image_mask_cache_key))

    def test_deserialize_legacy_inline_bitmap_migrates_into_store(self):
        legacy_image = np.array([[0.25, 0.75], [0.1, 0.9]], dtype=np.float32)
        import utils.utils as utils

        legacy_dict = {
            "center": [0.0, 0.0],
            "name": "Face",
            "effects_param": {},
            "image_mask_cache": utils.convert_image_to_list(legacy_image),
            # キーは意図的に欠損させ、sha1 代替キー生成パスを通す
        }

        mask = HeadlessFaceMask(self.ctx, self.pipeline)
        mask.deserialize(legacy_dict)

        self.assertIsNotNone(mask.image_mask_cache_key)
        np.testing.assert_array_equal(legacy_image, mask.image_mask_cache)

        stored = self.pipeline.get_ai_mask_bitmap(mask.image_mask_cache_key)
        self.assertIsNotNone(stored, "旧形式インラインはストアへ移行されること")
        np.testing.assert_array_equal(legacy_image, stored)


# --------------------------------------------------------------------------
# 3. GUI 側(widgets/mask_editor2.py)の構造検証: インライン出力が消えていること
#    (Kivy 依存のため import できず、AST ソース検証で確認する)
# --------------------------------------------------------------------------
class GuiMaskSerializeStructureTest(unittest.TestCase):
    def test_ai_masks_no_longer_serialize_inline_bitmap(self):
        for cls in ("SegmentMask", "FaceMask", "TargetTextMask"):
            with self.subTest(cls=cls):
                serialize_source = _class_method_source(MASK_EDITOR_PATH, cls, "serialize")
                deserialize_source = _class_method_source(MASK_EDITOR_PATH, cls, "deserialize")

                self.assertNotIn("convert_image_to_list(self.image_mask_cache)", serialize_source)
                self.assertIn("_serialize_image_mask_cache", serialize_source)
                self.assertIn("_deserialize_image_mask_cache", deserialize_source)

    def test_base_mask_routes_through_shared_store(self):
        get_or_compute_source = _class_method_source(
            MASK_EDITOR_PATH, "BaseMask", "_get_or_compute_image_mask_cache"
        )
        serialize_helper_source = _class_method_source(
            MASK_EDITOR_PATH, "BaseMask", "_serialize_image_mask_cache"
        )
        deserialize_helper_source = _class_method_source(
            MASK_EDITOR_PATH, "BaseMask", "_deserialize_image_mask_cache"
        )

        self.assertIn("get_ai_mask_bitmap", get_or_compute_source)
        self.assertIn("put_ai_mask_bitmap", get_or_compute_source)
        self.assertIn("image_mask_cache_key", serialize_helper_source)
        self.assertNotIn("convert_image_to_list", serialize_helper_source)
        self.assertIn("image_mask_cache_key", deserialize_helper_source)
        self.assertIn("put_ai_mask_bitmap", deserialize_helper_source)
        self.assertIn("get_ai_mask_bitmap", deserialize_helper_source)

    def test_memory_pressure_path_uses_clear_transient(self):
        source = _class_method_source(MASK_EDITOR_PATH, "MaskEditor2", "clear_ai_intermediate_caches")

        self.assertIn("self.ai_image_cache.clear_transient()", source)
        self.assertNotIn("self.ai_image_cache.clear()", source)


# --------------------------------------------------------------------------
# 4. params.py: mask2_bitmaps の serialize/deserialize round-trip
# --------------------------------------------------------------------------
class _StubMaskEditor:
    """widgets/mask_editor2.MaskEditor2 の代わりに params.serialize/deserialize を
    駆動する最小スタブ(Kivy 非依存)。tests/test_params_msgpack_safe.py の
    _EmptyMaskEditor パターンに倣う。"""

    def __init__(self, mask_dict=None, watch_key=None):
        self._mask_dict = mask_dict
        self.ai_image_cache = AIImageCache()
        self.watch_key = watch_key
        self.bitmap_present_at_deserialize = None
        self.deserialize_calls = []

    def serialize(self):
        return self._mask_dict

    def clear_mask(self):
        pass

    def deserialize(self, ser):
        self.deserialize_calls.append(ser)
        if self.watch_key is not None:
            # mask2_bitmaps のマージは mask_editor2.deserialize() より前に
            # 行われる仕様なので、ここで既にストアへ入っているはず。
            self.bitmap_present_at_deserialize = (
                self.ai_image_cache.get_mask_bitmap(self.watch_key) is not None
            )

    def serialize_ai_image_cache(self):
        return None

    def set_serialized_ai_image_cache(self, serialized):
        pass


class ParamsMask2BitmapsRoundTripTest(unittest.TestCase):
    def test_round_trip_includes_shared_bitmap_once_and_restores_into_store(self):
        key = cache_keys.segment_cache_key((100, 80), [10.0, 10.0], [50.0, 50.0], False)
        bitmap = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)

        mask_dict = {
            "mask2": [
                {
                    "type": "composit",
                    "name": "C",
                    "effects_param": {},
                    "mask_list": [
                        (
                            {
                                "type": "segment",
                                "name": "S1",
                                "center": [0.0, 0.0],
                                "corner": [0.1, 0.1],
                                "effects_param": {},
                                "image_mask_cache_key": key,
                            },
                            "Add",
                        ),
                        (
                            {
                                "type": "segment",
                                "name": "S2 copy",
                                "center": [0.0, 0.0],
                                "corner": [0.1, 0.1],
                                "effects_param": {},
                                # コピー等で同じキーを共有しているケースを模す
                                "image_mask_cache_key": key,
                            },
                            "Subtract",
                        ),
                    ],
                }
            ]
        }

        editor = _StubMaskEditor(mask_dict)
        editor.ai_image_cache.put_mask_bitmap(key, bitmap)

        param = {"original_img_size": (100, 80), "crop_rect": (0, 0, 100, 80)}
        ser = params.serialize(param, editor)

        self.assertIsNotNone(ser)
        self.assertIn("mask2_bitmaps", ser)
        self.assertEqual(1, len(ser["mask2_bitmaps"]), "共有キーは1回だけ含まれること")

        # msgpack セーフであること(実ファイル保存経路と同じ制約)
        packed = msgpack.packb(ser, use_bin_type=True)
        ser2 = msgpack.unpackb(packed, raw=False)

        new_editor = _StubMaskEditor(mask_dict, watch_key=key)
        # 実運用では params.deserialize は「画像ロード済みの param」に上書きマージされる
        # (original_img_size/crop_rect は読み込み時点で既に入っている)。それを模す。
        new_param = {"original_img_size": (100, 80), "crop_rect": (0, 0, 100, 80)}
        params.deserialize(ser2, new_param, new_editor)

        self.assertTrue(
            new_editor.bitmap_present_at_deserialize,
            "mask2_bitmaps のマージは mask_editor2.deserialize() より前に完了していること",
        )
        restored = new_editor.ai_image_cache.get_mask_bitmap(key)
        self.assertIsNotNone(restored)
        np.testing.assert_allclose(bitmap, restored)

    def test_no_masks_means_no_mask2_bitmaps_key(self):
        editor = _StubMaskEditor(None)
        param = {"original_img_size": (10, 10), "crop_rect": (0, 0, 10, 10), "exposure": 0.5}

        ser = params.serialize(param, editor)

        if ser is not None:
            self.assertNotIn("mask2_bitmaps", ser)


if __name__ == "__main__":
    unittest.main()
