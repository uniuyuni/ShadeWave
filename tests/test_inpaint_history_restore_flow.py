"""Inpaint マスクの履歴 undo/redo 回帰テスト。

- MaskEditor.add_mask が存在すること(redo 時の after_set2widget が呼ぶ。
  未実装だと AttributeError で redo が失敗し Make mask ボタン状態も崩れる)。
- add_mask の貼り戻しロジック(クリップ・最大値合成)。
- InpaintEffect / PatchmatchInpaintEffect のリストデフォルトが共有されないこと。

MaskEditor は kivy Window に依存しヘッドレスで import できないため、
add_mask 本体は AST でソース抽出して exec で検証する。
"""

import ast
import pathlib
import sys
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MASK_EDITOR_PATH = PROJECT_ROOT / "widgets" / "mask_editor.py"

import effects


def _load_method_source(path, class_name, function_name):
    source_text = path.read_text()
    tree = ast.parse(source_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source_text, child)
    return None


def _make_add_mask():
    src = _load_method_source(MASK_EDITOR_PATH, "MaskEditor", "add_mask")
    assert src is not None
    ns = {"np": np}
    exec(textwrap.dedent(src), ns)
    return ns["add_mask"]


class MaskEditorAddMaskTest(unittest.TestCase):
    def test_add_mask_exists(self):
        # effects.py の after_set2widget(履歴復元)が呼ぶ API。無いと redo で AttributeError。
        self.assertIsNotNone(_load_method_source(MASK_EDITOR_PATH, "MaskEditor", "add_mask"))

    def test_add_mask_pastes_region(self):
        add_mask = _make_add_mask()
        editor = SimpleNamespace(mask=np.zeros((100, 200), dtype=np.uint8))
        image = np.full((20, 30), 255, dtype=np.uint8)
        add_mask(editor, (40, 10, 30, 20), image)
        self.assertTrue(np.all(editor.mask[10:30, 40:70] == 255))
        self.assertEqual(int(editor.mask.sum()), 255 * 20 * 30)

    def test_add_mask_clips_out_of_bounds(self):
        add_mask = _make_add_mask()
        editor = SimpleNamespace(mask=np.zeros((50, 50), dtype=np.uint8))
        image = np.full((20, 20), 255, dtype=np.uint8)
        add_mask(editor, (-10, 40, 20, 20), image)  # 左と下にはみ出す
        self.assertTrue(np.all(editor.mask[40:50, 0:10] == 255))
        self.assertEqual(int(editor.mask.sum()), 255 * 10 * 10)
        # 完全に領域外は no-op
        add_mask(editor, (100, 100, 20, 20), image)
        self.assertEqual(int(editor.mask.sum()), 255 * 10 * 10)

    def test_add_mask_merges_with_maximum(self):
        add_mask = _make_add_mask()
        editor = SimpleNamespace(mask=np.full((10, 10), 200, dtype=np.uint8))
        add_mask(editor, (0, 0, 10, 10), np.full((10, 10), 100, dtype=np.uint8))
        self.assertTrue(np.all(editor.mask == 200))
        add_mask(editor, (0, 0, 10, 10), np.full((10, 10), 255, dtype=np.uint8))
        self.assertTrue(np.all(editor.mask == 255))


class InpaintDefaultListIsolationTest(unittest.TestCase):
    def _assert_default_lists_isolated(self, effect, keys):
        for key in keys:
            lst = effect._get_param({}, key)
            self.assertEqual(lst, [], key)
            lst.append("polluted")  # make_diff 等の in-place append を模擬
            self.assertEqual(effect._get_param({}, key), [], key)

    def test_inpaint_effect_defaults_not_shared(self):
        self._assert_default_lists_isolated(
            effects.InpaintEffect(),
            ("inpaint_diff_list", "inpaint_mask_list"),
        )

    def test_patchmatch_inpaint_effect_defaults_not_shared(self):
        self._assert_default_lists_isolated(
            effects.PatchmatchInpaintEffect(),
            ("patchmatch_inpaint_diff_list", "patchmatch_inpaint_mask_list"),
        )


if __name__ == "__main__":
    unittest.main()
