"""マスク作成モード「コピー」(widgets/mask_editor2.py の
MaskEditor2.copy_mask_into / copy_composit_children_into)のテスト。

widgets/mask_editor2.py は通常どおり import 可能。ただし `root`(kv の `root: root`
バインドでのみ設定される ObjectProperty 相当の動的属性)は素の Python インスタンス化では
未設定のため、`editor.root = None` を明示しておく(コード側は `if self.root is not None:`
のガードのみなので None なら安全にスキップされる)。これにより GUI クラスを実際に
インスタンス化してマスクを作成し、コピー処理を素通しで検証できる。
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import effects
import params
import widgets.mask_editor2 as me2
from cores.mask2 import cache_keys, inference_runtime


def _make_editor(size=(64, 64)):
    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
    editor = me2.MaskEditor2()
    editor.root = None  # kv 経由でのみ設定されるため素の Python インスタンス化では補う
    editor.set_ref_image(img, img)
    editor.set_texture_size(*size)
    primary = {}
    params.set_image_param(primary, img)
    editor.set_primary_param(primary, primary.get('disp_info'))
    return editor


def _add_circle(editor, parent_composit, maskop, center, inner, outer):
    index = len(editor.get_mask_list())
    mask = editor._create_mask(me2.MaskType.CIRCULAR, index)
    mask.initializing = False
    mask.center = center
    mask.inner_radius_x, mask.inner_radius_y = inner
    mask.outer_radius_x, mask.outer_radius_y = outer
    mask.create_control_points()
    parent_composit.add_mask(mask, maskop, 0)
    return mask


class Mask2CopyCircularShapeTest(unittest.TestCase):
    def test_copy_preserves_shape_new_id_and_name_suffix(self):
        editor = _make_editor()
        comp = editor._create_mask(me2.MaskType.COMPOSIT, 0)
        circ = _add_circle(editor, comp, 'Add', (5.0, -3.0), (2.0, 3.0), (6.0, 8.0))
        circ.rotate_rad = 0.25
        # ホワイトリスト外(調整系)の非デフォルト値を混入 -> 引き継がれないことを確認する
        circ.effects_param['mask2_blend_mode'] = 'Multiply'

        new_mask = editor.copy_mask_into(circ, comp, 'Add')

        self.assertIsInstance(new_mask, me2.CircularGradientMask)
        self.assertNotEqual(new_mask.mask_id, circ.mask_id)
        self.assertEqual(new_mask.name, "Circle copy")

        np.testing.assert_allclose(tuple(new_mask.center), tuple(circ.center), atol=1e-4)
        self.assertAlmostEqual(new_mask.inner_radius_x, circ.inner_radius_x, places=3)
        self.assertAlmostEqual(new_mask.inner_radius_y, circ.inner_radius_y, places=3)
        self.assertAlmostEqual(new_mask.outer_radius_x, circ.outer_radius_x, places=3)
        self.assertAlmostEqual(new_mask.outer_radius_y, circ.outer_radius_y, places=3)
        self.assertAlmostEqual(new_mask.rotate_rad, circ.rotate_rad, places=6)

        # ホワイトリスト外(調整・描画系)は新規デフォルトのまま
        self.assertEqual(
            effects.Mask2Effect.get_param(new_mask.effects_param, 'mask2_blend_mode'),
            'Normal',
        )
        # コピー元の状態は変更されていない(non-mutating)こと
        self.assertEqual(circ.effects_param.get('mask2_blend_mode'), 'Multiply')
        self.assertEqual(circ.name, "Circle")

    def test_whitelisted_shape_params_are_inherited_non_whitelisted_are_not(self):
        editor = _make_editor()
        comp = editor._create_mask(me2.MaskType.COMPOSIT, 0)
        circ = _add_circle(editor, comp, 'Add', (1.0, 1.0), (1.0, 1.0), (2.0, 2.0))

        circ.effects_param['mask2_invert'] = True
        circ.effects_param['mask2_blur'] = 42
        circ.effects_param['mask2_depth_min'] = 17
        circ.effects_param['mask2_skin_smooth_amount'] = 88  # ホワイトリスト外

        new_mask = editor.copy_mask_into(circ, comp, 'Add')

        self.assertEqual(effects.Mask2Effect.get_param(new_mask.effects_param, 'mask2_invert'), True)
        self.assertEqual(effects.Mask2Effect.get_param(new_mask.effects_param, 'mask2_blur'), 42)
        self.assertEqual(effects.Mask2Effect.get_param(new_mask.effects_param, 'mask2_depth_min'), 17)

        default_dict = effects.Mask2Effect.get_param_dict({})
        self.assertEqual(
            effects.Mask2Effect.get_param(new_mask.effects_param, 'mask2_skin_smooth_amount'),
            default_dict['mask2_skin_smooth_amount'],
        )


class Mask2CopyCompositExpansionTest(unittest.TestCase):
    def test_composit_children_expand_into_target_preserving_maskop_and_order(self):
        editor = _make_editor()
        src_comp = editor._create_mask(me2.MaskType.COMPOSIT, 0)
        target_comp = editor._create_mask(me2.MaskType.COMPOSIT, 1)

        a = _add_circle(editor, src_comp, 'Add', (0.0, 0.0), (0.5, 0.5), (1.0, 1.0))
        b = _add_circle(editor, src_comp, 'Subtract', (2.0, 2.0), (1.5, 1.5), (3.0, 3.0))

        src_ops_before = [op for _, op in src_comp.get_mask_list()]

        created = editor.copy_composit_children_into(src_comp, target_comp)

        self.assertEqual(2, len(created))
        self.assertEqual(2, len(target_comp.get_mask_list()))
        # 元のコンポジットは変更されない
        self.assertEqual(src_ops_before, [op for _, op in src_comp.get_mask_list()])
        self.assertIn(a, [m for m, _ in src_comp.get_mask_list()])
        self.assertIn(b, [m for m, _ in src_comp.get_mask_list()])

        # 各子の maskop がコピー元の値のまま(順序も含め)維持されていること
        target_ops = [op for _, op in target_comp.get_mask_list()]
        self.assertEqual(src_ops_before, target_ops)

        # コピー先の子はコピー元と別オブジェクト(新規 mask_id)
        target_masks = [m for m, _ in target_comp.get_mask_list()]
        self.assertNotIn(a, target_masks)
        self.assertNotIn(b, target_masks)
        for m in target_masks:
            self.assertTrue(m.name.endswith(" copy"))


class Mask2CopySelfCompositTest(unittest.TestCase):
    def test_copying_composit_into_itself_doubles_children_without_infinite_loop(self):
        editor = _make_editor()
        comp = editor._create_mask(me2.MaskType.COMPOSIT, 0)
        _add_circle(editor, comp, 'Add', (0.0, 0.0), (0.5, 0.5), (1.0, 1.0))

        self.assertEqual(1, len(comp.get_mask_list()))

        created = editor.copy_composit_children_into(comp, comp)

        self.assertEqual(1, len(created))
        self.assertEqual(2, len(comp.get_mask_list()))


class Mask2CopySegmentMaskSharesStoreTest(unittest.TestCase):
    def test_segment_mask_copy_shares_store_entry_without_extra_inference(self):
        editor = _make_editor()
        comp = editor._create_mask(me2.MaskType.COMPOSIT, 0)

        index = len(editor.get_mask_list())
        seg = editor._create_mask(me2.MaskType.SEGMENT, index)
        seg.initializing = False
        seg.center = (0.0, 0.0)
        seg.corner = (10.0, 10.0)
        seg.create_control_points()
        comp.add_mask(seg, 'Add', 0)

        key = cache_keys.segment_cache_key((64, 64), (0.0, 0.0), (10.0, 10.0), False)
        bitmap = np.full((8, 8), 0.5, dtype=np.float32)
        editor.put_ai_mask_bitmap(key, bitmap)
        seg.image_mask_cache_key = key
        seg.image_mask_cache = bitmap

        self.assertEqual(1, len(editor.ai_image_cache._mask_bitmaps))

        calls = []
        original = inference_runtime.predict_sam3_bbox

        def _fake_predict(*args, **kwargs):
            calls.append(args)
            return np.zeros((1, 1), dtype=np.float32)

        inference_runtime.predict_sam3_bbox = _fake_predict
        try:
            new_mask = editor.copy_mask_into(seg, comp, 'Add')
        finally:
            inference_runtime.predict_sam3_bbox = original

        self.assertEqual(0, len(calls), "コピー処理は推論を再実行しないこと")
        self.assertIsInstance(new_mask, me2.SegmentMask)
        self.assertEqual(new_mask.image_mask_cache_key, seg.image_mask_cache_key)
        self.assertEqual(
            1, len(editor.ai_image_cache._mask_bitmaps),
            "ストアのエントリ数は増えない(ビットマップは共有される)",
        )
        stored = editor.get_ai_mask_bitmap(new_mask.image_mask_cache_key)
        self.assertIsNotNone(stored)
        np.testing.assert_array_equal(bitmap, stored)


if __name__ == "__main__":
    unittest.main()
