import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import effects
import pipeline


PREVIEW_SIZE = 384


class MaskEditorStub:
    def __init__(self):
        self.texture_size = None
        self.primary = None
        self.ref_image = None

    def set_texture_size(self, width, height):
        self.texture_size = (width, height)

    def set_primary_param(self, param, disp_info, redraw_mask=True):
        self.primary = (param, disp_info, redraw_mask)

    def set_ref_image(self, imgc, pre_rotation_img):
        self.ref_image = (imgc, pre_rotation_img)

    def get_mask_list(self):
        return []

    def update(self):
        pass


def install_headless_config(preview_size=PREVIEW_SIZE):
    config._config = {
        "import_path": os.getcwd(),
        "lut_path": os.getcwd() + "/lut",
        "preview_size": preview_size,
        "ai_demosaic": False,
        "raw_auto_exposure": False,
        "scale_threshold": 0.5,
        "inpaint_resize_limit": 1024,
        "inpaint_use_realesrgan": False,
        "display_color_gamut": "sRGB",
        "gpu_device": "mps",
        "cat": "cat16",
        "base_resolution_scale": [4096, 4096],
        "display_output_dither": False,
        "display_output_downscale": True,
        "debug_nan_inf_check": False,
        "mesh_rbf_function": "mls",
    }
    config._preview_texture_size = (preview_size, preview_size)


def make_image():
    # 構造のあるグラデーション+ノイズ(縮小/拡大の劣化が出るが決定的な画)
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:600, 0:800].astype(np.float32)
    base = np.stack(
        (
            xx / 800.0,
            yy / 600.0,
            (xx + yy) / 1400.0,
        ),
        axis=-1,
    )
    return np.clip(base + rng.normal(0.0, 0.02, base.shape).astype(np.float32), 0.0, 1.0).astype(np.float32)


def make_param():
    return {
        "original_img_size": (800, 600),
        "rgb_or_raw": "rgb",
        "exif_data": {},
        "color_temperature_reset": 5000,
        "color_tint_reset": 0,
    }


def run_pipeline(img, crop, primary_effects, param, mask_editor, version, drag_quality):
    return pipeline.process_pipeline(
        img,
        crop,
        False,
        1.0,
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        0,
        0,
        primary_effects,
        param,
        mask_editor,
        None,
        version,
        "Ed",
        loading_flag=-1,
        is_drag=False,
        drag_quality=drag_quality,
    )


class DragHalfResFlowTest(unittest.TestCase):
    """スライダードラッグ中の lv1-lv2 half-res プレビューの回帰テスト。

    不変条件:
      - ドラッグフレームでも出力/テクスチャ/disp_info(スケール)は非ドラッグ時と同一
      - half-res→full の切り替えで別解像度の stale diff を再利用しない
      - キルスイッチ PLATYPUS_DRAG_HALF_RES=0 で完全に旧挙動
    """

    def setUp(self):
        self.old_config = config._config
        self.old_preview_texture_size = config._preview_texture_size
        install_headless_config()
        self.old_env = os.environ.get("PLATYPUS_DRAG_HALF_RES")
        os.environ.pop("PLATYPUS_DRAG_HALF_RES", None)
        # 前のテストの half-res 状態を持ち越さない
        pipeline._drag_half_res_prev_state["primary"] = False

    def tearDown(self):
        config._config = self.old_config
        config._preview_texture_size = self.old_preview_texture_size
        if self.old_env is None:
            os.environ.pop("PLATYPUS_DRAG_HALF_RES", None)
        else:
            os.environ["PLATYPUS_DRAG_HALF_RES"] = self.old_env
        pipeline._drag_half_res_prev_state["primary"] = False

    def test_drag_frame_keeps_output_size_and_disp_info(self):
        img = make_image()
        param = make_param()
        me = MaskEditorStub()
        pe = effects.create_effects()

        out_full, crop = run_pipeline(img, None, pe, param, me, 1, drag_quality=False)
        self.assertEqual(out_full.shape[:2], (PREVIEW_SIZE, PREVIEW_SIZE))
        disp_after_full = tuple(param["disp_info"])

        out_drag, crop = run_pipeline(img, crop, pe, param, me, 2, drag_quality=True)
        # 外から見える解像度・座標系は一切変わらない
        self.assertEqual(out_drag.shape, out_full.shape)
        self.assertEqual(me.texture_size, (PREVIEW_SIZE, PREVIEW_SIZE))
        self.assertEqual(tuple(param["disp_info"]), disp_after_full)
        self.assertTrue(np.isfinite(np.asarray(out_drag)).all())
        # 内容は半解像度なりに近い(構造画像なので大きくは離れない)
        diff = np.abs(np.asarray(out_drag, dtype=np.float32) - np.asarray(out_full, dtype=np.float32))
        self.assertLess(float(diff.mean()), 0.05)

    def test_release_recovers_bitwise_full_resolution(self):
        img = make_image()
        param = make_param()
        # UnsharpMask は param_hash に解像度/上流を含めないため、
        # 境界リセット(lv1reset)が無いと half-res の diff が漏れて出力サイズが壊れる。
        param["unsharp_mask_amount"] = 50
        me = MaskEditorStub()
        pe = effects.create_effects()

        out_full, crop = run_pipeline(img, None, pe, param, me, 1, drag_quality=False)
        out_drag, crop = run_pipeline(img, crop, pe, param, me, 2, drag_quality=True)
        self.assertEqual(out_drag.shape, out_full.shape)
        out_back, crop = run_pipeline(img, crop, pe, param, me, 3, drag_quality=False)
        self.assertEqual(out_back.shape, out_full.shape)
        np.testing.assert_allclose(
            np.asarray(out_back, dtype=np.float32),
            np.asarray(out_full, dtype=np.float32),
            atol=1e-6,
            err_msg="release 後のフル解像度出力が drag 前と一致しない(stale diff 混入の疑い)",
        )

    def test_param_change_during_drag_settles_to_fresh_full_result(self):
        img = make_image()
        param = make_param()
        me = MaskEditorStub()
        pe = effects.create_effects()

        _, crop = run_pipeline(img, None, pe, param, me, 1, drag_quality=False)
        # ドラッグ中に露出を動かす(実際のスライダードラッグ相当)
        param["exposure"] = 0.5
        _, crop = run_pipeline(img, crop, pe, param, me, 2, drag_quality=True)
        param["exposure"] = 1.0
        _, crop = run_pipeline(img, crop, pe, param, me, 3, drag_quality=True)
        out_settled, crop = run_pipeline(img, crop, pe, param, me, 4, drag_quality=False)

        # まっさらな effects で同じパラメータをフル解像度計算した結果と一致すること
        param_ref = make_param()
        param_ref["exposure"] = 1.0
        me_ref = MaskEditorStub()
        pe_ref = effects.create_effects()
        out_ref, _ = run_pipeline(img, None, pe_ref, param_ref, me_ref, 1, drag_quality=False)
        np.testing.assert_allclose(
            np.asarray(out_settled, dtype=np.float32),
            np.asarray(out_ref, dtype=np.float32),
            atol=1e-5,
            err_msg="ドラッグ後の確定結果がフル解像度の素の計算結果と一致しない",
        )

    def test_kill_switch_restores_legacy_behavior(self):
        os.environ["PLATYPUS_DRAG_HALF_RES"] = "0"
        img = make_image()
        param = make_param()
        me = MaskEditorStub()
        pe = effects.create_effects()

        out_full, crop = run_pipeline(img, None, pe, param, me, 1, drag_quality=False)
        out_drag, crop = run_pipeline(img, crop, pe, param, me, 2, drag_quality=True)
        np.testing.assert_allclose(
            np.asarray(out_drag, dtype=np.float32),
            np.asarray(out_full, dtype=np.float32),
            atol=1e-6,
            err_msg="キルスイッチ有効時に drag_quality が出力へ影響している",
        )

    def test_consecutive_drag_frames_are_stable(self):
        img = make_image()
        param = make_param()
        me = MaskEditorStub()
        pe = effects.create_effects()

        _, crop = run_pipeline(img, None, pe, param, me, 1, drag_quality=False)
        out_a, crop = run_pipeline(img, crop, pe, param, me, 2, drag_quality=True)
        out_b, crop = run_pipeline(img, crop, pe, param, me, 3, drag_quality=True)
        np.testing.assert_allclose(
            np.asarray(out_a, dtype=np.float32),
            np.asarray(out_b, dtype=np.float32),
            atol=1e-6,
            err_msg="同一パラメータの連続ドラッグフレームで出力が揺れる",
        )


if __name__ == "__main__":
    unittest.main()
