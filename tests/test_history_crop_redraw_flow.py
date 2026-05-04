import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_function_node(name):
    tree = ast.parse(MAIN_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


class HistoryCropRedrawFlowTest(unittest.TestCase):
    def _assert_uses_crop_redraw(self, function_name):
        node = _load_function_node(function_name)
        call_names = [
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        ]

        self.assertIn("start_draw_image_and_crop", call_names)
        self.assertNotIn(
            "start_draw_image",
            call_names,
            f"{function_name} must clear crop_image cache before redraw.",
        )

    def test_undo_clears_crop_image_cache_before_redraw(self):
        self._assert_uses_crop_redraw("_undo")

    def test_redo_clears_crop_image_cache_before_redraw(self):
        self._assert_uses_crop_redraw("_redo")

    def test_history_selection_clears_crop_image_cache_before_redraw(self):
        self._assert_uses_crop_redraw("_on_history_selected")


if __name__ == "__main__":
    unittest.main()
