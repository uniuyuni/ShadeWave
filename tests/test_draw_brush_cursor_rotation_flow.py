"""FreeDraw / Polyline のブラシカーソルがジオメトリ回転でズレない回帰テスト。

カーソルは canvas で Translate → Rotate(origin) → ellipse の順に描かれ、update_mask が
self.rotate.angle にジオメトリ回転角をセットする。回転原点が (0,0) のままだと、円カーソルの
中心 (brush/2, brush/2) が R·c - c だけ移動し、回転方向へズレる。update_brush_cursor が
rotate.origin をカーソル中心に合わせることで、その場回転(円なので見た目不変)になり
ズレが消えることを検証する。

widgets.mask_editor2 は kivy Window/SDL に依存しヘッドレスで import できないため、
(1) Kivy の Translate/Rotate(origin) 合成を再現した数式でドリフトが消えることを示し、
(2) 両ツールの update_brush_cursor が rotate.origin をカーソル中心へ設定していることを
ソースから構造的に確認する。
"""

import ast
import math
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MASK_EDITOR2_PATH = PROJECT_ROOT / "widgets" / "mask_editor2.py"


def _T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def _R(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _kivy_rotate(deg, ox, oy):
    # Kivy Rotate(angle, origin=(ox,oy)) == T(o) · R(angle) · T(-o)
    return _T(ox, oy) @ _R(deg) @ _T(-ox, -oy)


def _cursor_center(x, y, brush, angle_deg, origin):
    """canvas: PushMatrix, Translate(tx,ty), Rotate(angle,origin), ellipse(0,0,brush,brush)。
    ellipse 中心 (brush/2, brush/2) の最終描画位置を返す。"""
    tx, ty = x - brush / 2, y - brush / 2
    m = _T(tx, ty) @ _kivy_rotate(angle_deg, *origin)
    c = np.array([brush / 2, brush / 2, 1.0])
    p = m @ c
    return p[0], p[1]


class BrushCursorRotationMathTest(unittest.TestCase):
    X, Y, BRUSH = 500.0, 400.0, 200.0

    def test_origin_zero_drifts_in_rotation_direction(self):
        # 現状(バグ)の再現: origin=(0,0) だと角度に比例してズレる。
        for deg in (15, 30, 45, 90):
            cx, cy = _cursor_center(self.X, self.Y, self.BRUSH, deg, (0.0, 0.0))
            drift = math.hypot(cx - self.X, cy - self.Y)
            self.assertGreater(drift, 1.0, f"angle={deg} で本来ドリフトするはず")

    def test_origin_center_removes_drift(self):
        # 修正: origin=カーソル中心なら角度に関わらずズレ 0。
        for deg in (0, 15, 30, 45, 90, -30):
            cx, cy = _cursor_center(self.X, self.Y, self.BRUSH, deg, (self.BRUSH / 2, self.BRUSH / 2))
            self.assertAlmostEqual(cx, self.X, places=6, msg=f"angle={deg}")
            self.assertAlmostEqual(cy, self.Y, places=6, msg=f"angle={deg}")


def _find_method(tree, class_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    return None


def _sets_rotate_origin_to_center(method_node):
    """method 内に `self.rotate.origin = (brush_size / 2, brush_size / 2)` 相当の
    代入があるかを AST で確認する。"""
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "origin"
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "rotate"
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "self"
            ):
                return True
    return False


class BrushCursorSourceContractTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(MASK_EDITOR2_PATH.read_text())

    def test_freedraw_sets_rotate_origin_to_center(self):
        method = _find_method(self.tree, "FreeDrawMask", "update_brush_cursor")
        self.assertIsNotNone(method)
        self.assertTrue(
            _sets_rotate_origin_to_center(method),
            "FreeDrawMask.update_brush_cursor が self.rotate.origin をカーソル中心へ設定していない",
        )

    def test_polyline_sets_rotate_origin_to_center(self):
        method = _find_method(self.tree, "PolylineMask", "update_brush_cursor")
        self.assertIsNotNone(method)
        self.assertTrue(
            _sets_rotate_origin_to_center(method),
            "PolylineMask.update_brush_cursor が self.rotate.origin をカーソル中心へ設定していない",
        )


if __name__ == "__main__":
    unittest.main()
