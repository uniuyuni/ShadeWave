import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECT_SELECTOR_PATH = PROJECT_ROOT / "widgets" / "effect_selector.py"
EFFECT_SELECTOR_KV_PATH = PROJECT_ROOT / "widgets" / "effect_selector.kv"


def _load_class_function(class_name, function_name):
    source = EFFECT_SELECTOR_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class EffectSelectorCheckboxStyleTest(unittest.TestCase):
    def test_checkboxes_use_shared_navy_accent(self):
        py_source = EFFECT_SELECTOR_PATH.read_text()
        kv_source = EFFECT_SELECTOR_KV_PATH.read_text()
        create_checkbox_source = _load_class_function("EffectSelector", "_create_checkbox")

        self.assertIn("_ACCENT = (0.13, 0.23, 0.74, 1)", py_source)
        self.assertIn("border_color_active=_ACCENT", create_checkbox_source)
        self.assertIn("check_color=_ACCENT", create_checkbox_source)
        self.assertIn("border_color_active: [0.13, 0.23, 0.74, 1]", kv_source)
        self.assertIn("check_color: [0.13, 0.23, 0.74, 1]", kv_source)
        self.assertNotIn("0.45, 0.72, 0.98", py_source)
        self.assertNotIn("0.45, 0.72, 0.98", kv_source)


if __name__ == "__main__":
    unittest.main()
