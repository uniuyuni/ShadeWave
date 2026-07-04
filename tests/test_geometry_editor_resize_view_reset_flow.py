"""Geometry エディタ (Mesh/Lines/FourPoints/Trapezoid/Lens) の resize / cmd+F 表示リセット回帰テスト。

以前の不具合:
    Ge タブ表示中にウィンドウリサイズや cmd+F (フォーカスモード) を行うと
    プレビューテクスチャサイズが変わり param 側の disp_info が再計算されるが、
    各エディタが生成時にキャッシュした tcg_info は古いままのため、
    グリッド/CP などのオーバーレイ表示が画像とズレたままになった (画像自体は正常)。

修正後のゴール:
    refresh_preview_overlays → GeometryEffect.update_geometry_editor_texture_size(param)
    の経路でエディタの view 座標系 (tcg_info) が現在の param から再同期され、
    TCG→window 座標が新しい表示座標系と一致する (=表示がリセットされる)。

kivy App はヘッドレスで import できないため、widget 側は set_view_param メソッドを
AST でソース抽出してスタブへバインドし、params.tcg_to_window の実座標計算で検証する。
"""

import ast
import copy
import pathlib
import sys
import types
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import params
import cores.core as core

WIDGET_DIR = PROJECT_ROOT / "widgets" / "distortion_correction"

# 各エディタ: (ファイル, クラス名, set_view_param が呼ぶ再描画メソッド名)
EDITOR_SPECS = [
    ("mesh_warp_widget.py", "MeshWarpWidget", "_redraw_mesh"),
    ("line_guide_correction_widget.py", "LineGuideCorrectionWidget", "_redraw_lines"),
    ("four_point_correction_widget.py", "FourPointCorrectionWidget", "_sync_tcg_to_kivy"),
    ("trapezoid_correction_widget.py", "TrapezoidCorrectionWidget", "update_preview"),
    ("lens_distortion_widget.py", "LensDistortionWidget", "update_preview"),
]


def _load_methods(path, class_name, method_names):
    source = path.read_text()
    tree = ast.parse(source)
    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_node = node
            break
    if class_node is None:
        raise AssertionError(f"{class_name} not found in {path.name}")

    methods = {}
    for child in class_node.body:
        if isinstance(child, ast.FunctionDef) and child.name in method_names:
            segment = ast.get_source_segment(source, child)
            namespace = {"params": params, "np": np, "copy": copy}
            exec(compile("import textwrap\n", "<none>", "exec"), namespace)
            import textwrap
            exec(textwrap.dedent(segment), namespace)
            methods[child.name] = namespace[child.name]
    return methods


def _make_param(original_img_size, texture_side):
    crop_rect = core.get_initial_crop_rect(*original_img_size)
    disp_info = core.convert_rect_to_info(
        crop_rect, texture_side / max(original_img_size))
    return {
        "original_img_size": original_img_size,
        "disp_info": disp_info,
        "rotation": 0,
        "rotation2": 0,
        "flip_mode": 0,
        "matrix": np.eye(3),
    }


class _StubEditor:
    """set_view_param をバインドする最小スタブ (widget geometry のみ保持)"""

    def __init__(self, size, pos, texture_size, tcg_info):
        self.size = size
        self.pos = pos
        self.texture_size = texture_size
        self.tcg_info = tcg_info
        self.redraw_calls = 0

    def to_window(self, x, y):
        return (x, y)

    def _redraw(self, *args, **kwargs):
        self.redraw_calls += 1


class GeometryEditorResizeViewResetFlowTest(unittest.TestCase):
    ORIGINAL_IMG_SIZE = (4000, 3000)
    OLD_TEXTURE = (1024, 1024)
    NEW_TEXTURE = (1400, 1400)

    def setUp(self):
        # 座標検証を DPI 非依存にする
        self._orig_dpi_scale = params.device.dpi_scale
        params.device.dpi_scale = lambda: 1.0

    def tearDown(self):
        params.device.dpi_scale = self._orig_dpi_scale

    def _window_pos(self, editor, tcg_xy):
        return params.tcg_to_window(
            tcg_xy[0], tcg_xy[1], editor, editor.texture_size, editor.tcg_info)

    def test_set_view_param_resets_overlay_to_new_display_space(self):
        """リサイズで disp_info が変わった後、set_view_param で TCG→window 座標が
        新しい表示座標系の期待位置へ戻る (=ズレがリセットされる) こと"""
        param_old = _make_param(self.ORIGINAL_IMG_SIZE, max(self.OLD_TEXTURE))
        param_new = _make_param(self.ORIGINAL_IMG_SIZE, max(self.NEW_TEXTURE))
        probe_tcg = (0.5, 0.5)  # 画像右下隅の CP を想定

        for filename, class_name, redraw_name in EDITOR_SPECS:
            with self.subTest(editor=class_name):
                method_names = {"set_view_param"}
                if class_name == "MeshWarpWidget":
                    # Mesh は set_view_param -> set_tcg_info -> _copy_tcg_info 経由
                    method_names |= {"set_tcg_info", "_copy_tcg_info"}
                methods = _load_methods(WIDGET_DIR / filename, class_name, method_names)
                self.assertIn(
                    "set_view_param", methods,
                    f"{class_name} に表示リセット用の set_view_param がない")

                # エディタ生成時: 旧テクスチャの座標系をキャッシュ
                editor = _StubEditor(
                    size=list(self.NEW_TEXTURE), pos=(0, 0),
                    texture_size=self.OLD_TEXTURE,
                    tcg_info=params.param_to_tcg_info(param_old))
                setattr(editor, redraw_name, editor._redraw)
                if class_name == "MeshWarpWidget":
                    editor._view_context = None
                    editor._view_context_image_only_matrix = False
                    editor._copy_tcg_info = staticmethod(methods["_copy_tcg_info"])
                    editor.set_tcg_info = types.MethodType(methods["set_tcg_info"], editor)

                # リサイズ相当: texture_size だけ更新された状態 (旧実装の到達点)
                editor.texture_size = self.NEW_TEXTURE

                # 新しい表示座標系での期待位置 (画像は param_new の disp_info で描画される)
                expected_editor = _StubEditor(
                    size=list(self.NEW_TEXTURE), pos=(0, 0),
                    texture_size=self.NEW_TEXTURE,
                    tcg_info=params.param_to_tcg_info(param_new))
                expected = self._window_pos(expected_editor, probe_tcg)

                # 修正前の状態: 古い tcg_info のままだとズレている (バグの再現条件)
                stale = self._window_pos(editor, probe_tcg)
                self.assertGreater(
                    np.hypot(stale[0] - expected[0], stale[1] - expected[1]), 1.0,
                    "前提条件: disp_info が変わればズレが発生するはず")

                # 表示リセット: 現在の param から view を再同期
                methods["set_view_param"](editor, param_new)
                synced = self._window_pos(editor, probe_tcg)
                self.assertAlmostEqual(synced[0], expected[0], places=5)
                self.assertAlmostEqual(synced[1], expected[1], places=5)
                self.assertGreaterEqual(
                    editor.redraw_calls, 1,
                    f"{class_name}.set_view_param は再描画を伴うこと")

    def test_geometry_effect_resyncs_editor_view_on_texture_refresh(self):
        """GeometryEffect.update_geometry_editor_texture_size(param) が
        texture_size と view (param) の両方をエディタへ再同期すること"""
        import config
        import effects

        effect = effects.GeometryEffect()

        calls = []

        class _FakeEditor:
            def set_texture_size(self, texture_size):
                calls.append(("texture", tuple(texture_size)))

            def set_view_param(self, param):
                calls.append(("view", param))

        effect.geometry_editor = _FakeEditor()
        param = _make_param(self.ORIGINAL_IMG_SIZE, max(self.NEW_TEXTURE))
        effect.update_geometry_editor_texture_size(param)

        self.assertIn(("texture", tuple(config.get_preview_texture_size())), calls)
        view_calls = [c for c in calls if c[0] == "view"]
        self.assertEqual(len(view_calls), 1)
        self.assertIs(view_calls[0][1], param)

    def test_main_refresh_passes_primary_param_to_geometry_editor(self):
        """main.refresh_preview_overlays が現在の primary_param を渡して
        geometry エディタの表示をリセットしていること"""
        source = (PROJECT_ROOT / "main.py").read_text()
        self.assertIn(
            "geometry_effect.update_geometry_editor_texture_size(self.primary_param)",
            source)


class _Vec:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class _Rotate:
    def __init__(self):
        self.angle = 0.0


class _Line:
    def __init__(self, rectangle):
        self.rectangle = rectangle


class CropEditorFullImageFrameResizeTest(unittest.TestCase):
    """画像全体枠 (input_line) の resize 追従回帰テスト。

    以前の不具合: input_line は CropEditor.__init__ 時の scale で一度だけ矩形が
    設定され、resize / cmd+F で scale が変わってもクロップ枠 (white/black line) だけ
    更新されて画像全体のグレー枠がズレたままになった。
    ゴール: scale 変更後に update_centering を通すと input_line が新しい
    スケール済み画像サイズと一致する。
    """

    def test_update_centering_resyncs_full_image_frame_line(self):
        path = PROJECT_ROOT / "widgets" / "crop_editor.py"
        methods = _load_methods(path, "CropEditor", {"update_centering"})
        self.assertIn("update_centering", methods)

        input_width, input_height = 4000, 3000
        old_scale = 1024 / 4000
        new_scale = 1400 / 4000

        editor = types.SimpleNamespace()
        editor.input_width = input_width
        editor.input_height = input_height
        editor.scale = old_scale
        editor.input_angle = 0
        editor.width = 1500
        editor.height = 1500
        editor.parent = types.SimpleNamespace(pos=(0, 0))
        editor.to_window = lambda x, y: (x, y)
        editor.translate = _Vec()
        editor.input_translate = _Vec()
        editor.input_rotate = _Rotate()
        editor.update_rect = lambda *a, **k: None
        # __init__ 相当: 旧スケールで全体枠を設定
        editor.input_line = _Line((
            -input_width * old_scale / 2, -input_height * old_scale / 2,
            input_width * old_scale, input_height * old_scale))

        # resize 相当: scale だけ更新された状態
        editor.scale = new_scale

        expected = (
            -input_width * new_scale / 2, -input_height * new_scale / 2,
            input_width * new_scale, input_height * new_scale)
        # 前提条件: 更新前はズレている
        self.assertNotEqual(tuple(editor.input_line.rectangle), expected)

        methods["update_centering"](editor)

        for actual, exp in zip(editor.input_line.rectangle, expected):
            self.assertAlmostEqual(actual, exp, places=5)


if __name__ == "__main__":
    unittest.main()
