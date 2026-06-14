import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_function(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


class DisplayTransformFlowTest(unittest.TestCase):
    def test_settled_preview_uses_colour_functions_adapter(self):
        draw_image_core = _load_function(MAIN_PATH, "draw_image_core")
        source = ast.get_source_segment(MAIN_PATH.read_text(), draw_image_core)

        self.assertIn("_fast_display_color_transform(img, src_space, dst_space, cat)", source)
        self.assertIn("colour_functions.display_color_transform(img, src_space, dst_space, cat)", source)
        self.assertNotIn("apply_gamut_mapping=True", source)


if __name__ == "__main__":
    unittest.main()
