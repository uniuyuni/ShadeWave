"""mask1 (inpaint) full-preview モードの enter/exit 時のズーム・pan アンカー回帰テスト。

以前の実装の不具合:

1. 拡大表示のまま mask1 (Make mask) を開くと、pan アンカー(click_x/y)が常に全体
   画像中心へ飛ばされ、ジオメトリ(回転/クロップ)の状態によって「違う場所」に移動した。
   → 現在は、ズーム窓中心を回転/行列の逆変換でバイパス空間へ写像し、同じ画像内容を
   指す click アンカーへ引き直す。
2. exit 時に突入時スナップショットの disp_info(=突入時のビューポート)を復元していた
   ため、編集中にズームを変えても「突入時の拡大表示(最初の位置)」へ戻された。
   → 現在は is_zoomed/zoom_ratio/disp_info を復元せず、復元した crop_rect から
   非ズーム基準 disp_info を再計算し、ズーム継続時はアンカーを復元後空間へ逆写像する。

MainWidget は kivy App に依存しヘッドレスで import できないため、関連メソッドを
AST でソース抽出し、スタブオブジェクトへバインドして検証する。
"""

import ast
import copy
import math
import pathlib
import sys
import textwrap
import types
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAIN_PATH = PROJECT_ROOT / "main.py"

import config
import params
import cores.core as core


def _find_class_node(tree, class_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _load_method_source(source_text, class_node, function_name):
    for child in class_node.body:
        if isinstance(child, ast.FunctionDef) and child.name == function_name:
            return ast.get_source_segment(source_text, child)
    return None


def _load_class_attr_source(source_text, class_node, attr_name):
    for child in class_node.body:
        if isinstance(child, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == attr_name for t in child.targets
        ):
            return ast.get_source_segment(source_text, child)
    return None


class _StubEffects:
    @staticmethod
    def reeffect_all(effects, lv=0):
        pass


def _build_stub():
    source_text = MAIN_PATH.read_text()
    tree = ast.parse(source_text)
    class_node = _find_class_node(tree, "MainWidget")
    assert class_node is not None

    ns = {
        "params": params,
        "core": core,
        "config": config,
        "effects": _StubEffects,
        "copy": copy,
        "np": np,
    }
    bypass_keys_src = _load_class_attr_source(source_text, class_node, "_MASK1_GEOMETRY_BYPASS_KEYS")
    assert bypass_keys_src is not None
    exec(textwrap.dedent(bypass_keys_src), ns)

    static_methods = ("_mask1_identity_tcg_info",)
    bound_methods = (
        "_mask1_map_view_center",
        "_mask1_view_anchor_click",
        "_mask1_full_preview_disp_info",
        "_backup_mask1_geometry_params",
        "_restore_mask1_geometry_params",
        "_apply_mask1_geometry_bypass",
        "enter_mask1_full_preview_mode",
        "exit_mask1_full_preview_mode",
    )

    funcs = {}
    for name in static_methods + bound_methods:
        src = _load_method_source(source_text, class_node, name)
        assert src is not None, name
        exec(textwrap.dedent(src), ns)
        funcs[name] = ns[name]

    class _Stub:
        pass

    stub = _Stub()
    stub._MASK1_GEOMETRY_BYPASS_KEYS = ns["_MASK1_GEOMETRY_BYPASS_KEYS"]
    for name in static_methods:
        setattr(stub, name, funcs[name])  # instance attr: self なしで呼べる
    for name in bound_methods:
        setattr(stub, name, types.MethodType(funcs[name], stub))

    stub.primary_param = {}
    stub.is_zoomed = False
    stub.zoom_ratio = 1.0
    stub.click_x = 0
    stub.click_y = 0
    stub.drag_center_start = None
    stub.crop_image = None
    stub.crop_image_view_key = None
    stub._mask1_full_preview_backup = None
    stub._mask1_full_preview_sources = set()
    stub.primary_effects = [{}]
    stub._image_interaction_ready = lambda: True
    stub.start_draw_image_and_crop = lambda imgset: None
    return stub


def _init_param(stub, width=200, height=120, *, crop_rect=None, rotation=0, zoom_disp=None):
    stub.primary_param['original_img_size'] = (width, height)
    stub.primary_param['img_size'] = (width, height)
    params.set_crop_rect(
        stub.primary_param,
        crop_rect if crop_rect is not None else core.get_initial_crop_rect(width, height),
    )
    stub.primary_param['rotation'] = rotation
    stub.primary_param['rotation2'] = 0
    stub.primary_param['matrix'] = np.eye(3)
    stub.primary_param['flip_mode'] = 0
    if zoom_disp is not None:
        params.set_disp_info(stub.primary_param, zoom_disp)
    else:
        params.set_disp_info(
            stub.primary_param,
            core.convert_rect_to_info(params.get_crop_rect(stub.primary_param), 1.0),
        )


def _designated_center(stub):
    """現在の param disp_info(非ズーム基準)と click アンカーから、
    zoom_crop_source_info が作るズーム窓の中心(padded-square 座標)を求める。"""
    disp = params.get_disp_info(stub.primary_param)
    tex_w = config.get_config('preview_width')
    tex_h = config.get_config('preview_height')
    crop_rect = params.get_crop_rect(stub.primary_param)
    (crop_x, crop_y, crop_w, crop_h, _), _dbg = core.zoom_crop_source_info(
        disp, crop_rect, tex_w, tex_h,
        stub.click_x, stub.click_y, None, stub.zoom_ratio,
    )
    return (crop_x + crop_w / 2.0, crop_y + crop_h / 2.0)


class Mask1ZoomFramingTest(unittest.TestCase):
    def test_enter_preserves_content_center_without_geometry(self):
        # 素のジオメトリ: ズーム窓中心(70, 60)がバイパス後も同じ内容位置を指すこと。
        stub = _build_stub()
        zoom_center = (70.0, 60.0)
        _init_param(stub, zoom_disp=(zoom_center[0] - 40, zoom_center[1] - 30, 80, 60, 2.0))
        stub.is_zoomed = True
        stub.zoom_ratio = 2.0

        stub.enter_mask1_full_preview_mode('inpaint')

        cx, cy = _designated_center(stub)
        self.assertAlmostEqual(cx, zoom_center[0], delta=1.5)
        self.assertAlmostEqual(cy, zoom_center[1], delta=1.5)

    def test_enter_maps_content_center_through_rotation(self):
        # 回転 30° のジオメトリ空間で見ていた点が、バイパス(オリジナル)空間の
        # 対応点(center_rotate_invert による写像先)を指すこと。
        stub = _build_stub()
        zoom_center = (70.0, 60.0)
        _init_param(stub, rotation=30,
                    zoom_disp=(zoom_center[0] - 40, zoom_center[1] - 30, 80, 60, 2.0))
        stub.is_zoomed = True
        stub.zoom_ratio = 2.0
        tcg_from = params.param_to_tcg_info(stub.primary_param)

        stub.enter_mask1_full_preview_mode('inpaint')

        imax = 100.0  # max(200,120)/2
        ex, ey = params.center_rotate_invert(zoom_center[0] - imax, zoom_center[1] - imax, tcg_from)
        expected = (ex + imax, ey + imax)
        cx, cy = _designated_center(stub)
        self.assertAlmostEqual(cx, expected[0], delta=1.5)
        self.assertAlmostEqual(cy, expected[1], delta=1.5)

    def test_exit_does_not_restore_entry_viewport(self):
        # 編集中に縮小表示へ変えたら、exit 後も縮小のまま。disp_info も突入時の
        # ズーム窓ではなく、復元した crop_rect の非ズーム基準になっていること。
        stub = _build_stub()
        crop_rect = (20, 10, 100, 90)
        _init_param(stub, crop_rect=crop_rect, rotation=30,
                    zoom_disp=(30.0, 30.0, 80, 60, 2.0))
        stub.is_zoomed = True
        stub.zoom_ratio = 3.0

        stub.enter_mask1_full_preview_mode('inpaint')
        # 編集中: ユーザーが縮小表示へ切り替え、pipeline が全体表示の disp を書き戻す。
        stub.is_zoomed = False
        stub.zoom_ratio = 1.0
        bypass_disp = stub._mask1_full_preview_disp_info()
        params.set_disp_info(stub.primary_param, bypass_disp)

        stub.exit_mask1_full_preview_mode('inpaint')

        self.assertFalse(stub.is_zoomed)
        self.assertEqual(stub.zoom_ratio, 1.0)
        expected_disp = core.convert_rect_to_info(
            params.get_crop_rect(stub.primary_param),
            config.get_preview_texture_side() / 200,
        )
        got = params.get_disp_info(stub.primary_param)
        for g, e in zip(got[:4], expected_disp[:4]):
            self.assertAlmostEqual(g, e, delta=1.0)

    def test_enter_exit_roundtrip_returns_to_same_content(self):
        # ズームを維持したまま enter → exit したら、元のズーム窓中心へ戻ること。
        stub = _build_stub()
        zoom_center = (70.0, 60.0)
        crop_rect = (20, 10, 100, 90)
        _init_param(stub, crop_rect=crop_rect, rotation=30,
                    zoom_disp=(zoom_center[0] - 40, zoom_center[1] - 30, 80, 60, 2.0))
        stub.is_zoomed = True
        stub.zoom_ratio = 2.0

        stub.enter_mask1_full_preview_mode('inpaint')
        # 編集中に pipeline が(クリックアンカーに基づく)ズーム窓 disp を書き戻す状況を模擬:
        # enter が設定したアンカーの指す中心でズーム窓 disp を作る。
        bypass_disp = stub._mask1_full_preview_disp_info()
        tex_w = config.get_config('preview_width')
        tex_h = config.get_config('preview_height')
        (zx, zy, zw, zh, zr), _ = core.zoom_crop_source_info(
            bypass_disp, params.get_crop_rect(stub.primary_param), tex_w, tex_h,
            stub.click_x, stub.click_y, None, stub.zoom_ratio,
        )
        params.set_disp_info(stub.primary_param, (zx, zy, zw, zh, zr))

        stub.exit_mask1_full_preview_mode('inpaint')

        self.assertTrue(stub.is_zoomed)
        cx, cy = _designated_center(stub)
        self.assertAlmostEqual(cx, zoom_center[0], delta=2.5)
        self.assertAlmostEqual(cy, zoom_center[1], delta=2.5)

    def test_exit_restores_original_geometry(self):
        stub = _build_stub()
        original_crop_rect = (20, 10, 100, 90)
        _init_param(stub, crop_rect=original_crop_rect, rotation=30)

        stub.enter_mask1_full_preview_mode('inpaint')
        self.assertEqual(stub.primary_param.get('rotation'), 0)
        self.assertNotEqual(tuple(params.get_crop_rect(stub.primary_param)), original_crop_rect)

        stub.exit_mask1_full_preview_mode('inpaint')

        self.assertEqual(stub.primary_param.get('rotation'), 30)
        self.assertEqual(tuple(params.get_crop_rect(stub.primary_param)), original_crop_rect)

    def test_not_zoomed_enter_resets_anchor(self):
        stub = _build_stub()
        _init_param(stub)
        stub.is_zoomed = False
        stub.click_x, stub.click_y = 37.0, 41.0

        stub.enter_mask1_full_preview_mode('inpaint')

        self.assertEqual((stub.click_x, stub.click_y), (0, 0))


if __name__ == "__main__":
    unittest.main()
