import os
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GeometryDeferredCacheTest(unittest.TestCase):
    """deferred preview transform のフレーム間キャッシュ検証。

    pipeline_lv0 は毎フレーム GeometryEffect.make_diff を呼ぶため、param 不変時に
    _build_deferred_preview_transform（特に calculate_mesh_mls_coarse_map）が
    スキップされることを、関数呼び出し回数を数えて確認する。
    """

    def setUp(self):
        import effects
        from effect_backends import image_transform_adapter

        self.effects = effects
        self.adapter = image_transform_adapter
        self._previous_native_available = image_transform_adapter.native_available
        image_transform_adapter.native_available = lambda: True

        self._original_mesh_fn = effects.calculate_mesh_mls_coarse_map
        self.mesh_calls = 0

        def counting_mesh_fn(*args, **kwargs):
            self.mesh_calls += 1
            return self._original_mesh_fn(*args, **kwargs)

        effects.calculate_mesh_mls_coarse_map = counting_mesh_fn

    def tearDown(self):
        self.adapter.native_available = self._previous_native_available
        self.effects.calculate_mesh_mls_coarse_map = self._original_mesh_fn
        os.environ.pop("PLATYPUS_GEOMETRY_DEFERRED_CACHE", None)

    def _make_param(self, geometry, size=(72, 48)):
        import params

        param = geometry.get_param_dict({"original_img_size": size})
        param["original_img_size"] = size
        param["rotation"] = 7
        param["mesh_size"] = [4, 4]
        param["control_points"] = {(1, 1): (0.03, -0.02), (2, 2): (-0.04, 0.03)}
        params.set_crop_rect(param, (0, 0, size[0], size[1]))
        params.set_disp_info(param, (0, 0, size[0], size[0], 1.0))
        return param

    def _make_efconfig(self):
        from effects import EffectConfig

        return EffectConfig()

    def test_same_params_build_once(self):
        from effects import GeometryEffect

        geometry = GeometryEffect()
        param = self._make_param(geometry)
        image = np.zeros((48, 72, 3), dtype=np.float32)

        efconfig1 = self._make_efconfig()
        geometry.make_diff(image, param, efconfig1)
        self.assertEqual(self.mesh_calls, 1)
        matrix_after_build = np.array(param["matrix"], copy=True)
        payload1 = efconfig1.deferred_geometry_transform

        efconfig2 = self._make_efconfig()
        geometry.make_diff(image, param, efconfig2)
        self.assertEqual(self.mesh_calls, 1)  # キャッシュヒットで再構築なし
        payload2 = efconfig2.deferred_geometry_transform
        self.assertIsNotNone(payload2)

        # payload の等価性（mesh 配列は同一オブジェクト = zero-copy 前提）
        np.testing.assert_array_equal(payload1["matrix"], payload2["matrix"])
        self.assertIs(payload1["mesh_map_x"], payload2["mesh_map_x"])
        self.assertIs(payload1["mesh_map_y"], payload2["mesh_map_y"])
        # ヒット時に param['matrix'] が build 後と同じ値へ復元される
        np.testing.assert_array_equal(param["matrix"], matrix_after_build)

    def test_param_change_rebuilds(self):
        from effects import GeometryEffect

        geometry = GeometryEffect()
        param = self._make_param(geometry)
        image = np.zeros((48, 72, 3), dtype=np.float32)

        geometry.make_diff(image, param, self._make_efconfig())
        self.assertEqual(self.mesh_calls, 1)

        param["rotation"] = 12
        geometry.make_diff(image, param, self._make_efconfig())
        self.assertEqual(self.mesh_calls, 2)

    def test_image_shape_change_rebuilds(self):
        from effects import GeometryEffect

        geometry = GeometryEffect()
        param = self._make_param(geometry)

        geometry.make_diff(np.zeros((48, 72, 3), dtype=np.float32), param, self._make_efconfig())
        self.assertEqual(self.mesh_calls, 1)

        geometry.make_diff(np.zeros((96, 144, 3), dtype=np.float32), param, self._make_efconfig())
        self.assertEqual(self.mesh_calls, 2)

    def test_fresh_instance_rebuilds(self):
        from effects import GeometryEffect

        geometry1 = GeometryEffect()
        param = self._make_param(geometry1)
        image = np.zeros((48, 72, 3), dtype=np.float32)

        geometry1.make_diff(image, param, self._make_efconfig())
        self.assertEqual(self.mesh_calls, 1)

        # depth-map 経路 (_apply_primary_geometry_to_depth_map) と同様の fresh
        # インスタンスはキャッシュを共有せず、単に再構築される（挙動不変）。
        geometry2 = GeometryEffect()
        geometry2.make_diff(image, param, self._make_efconfig())
        self.assertEqual(self.mesh_calls, 2)

    def test_kill_switch_disables_cache(self):
        from effects import GeometryEffect

        os.environ["PLATYPUS_GEOMETRY_DEFERRED_CACHE"] = "0"
        geometry = GeometryEffect()
        param = self._make_param(geometry)
        image = np.zeros((48, 72, 3), dtype=np.float32)

        geometry.make_diff(image, param, self._make_efconfig())
        geometry.make_diff(image, param, self._make_efconfig())
        self.assertEqual(self.mesh_calls, 2)


if __name__ == "__main__":
    unittest.main()
