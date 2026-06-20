import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKBOX_PATH = PROJECT_ROOT / "widgets" / "modern_checkbox.py"
CHECKBOX_KV_PATH = PROJECT_ROOT / "widgets" / "modern_checkbox.kv"


def _load_class_function(class_name, function_name):
    source = CHECKBOX_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class ModernCheckboxLowResStyleTest(unittest.TestCase):
    def test_default_widget_size_is_compact_for_low_resolution(self):
        kv_source = CHECKBOX_KV_PATH.read_text()
        py_source = CHECKBOX_PATH.read_text()

        self.assertIn("ref_width: 16", kv_source)
        self.assertIn("ref_height: 16", kv_source)
        self.assertIn("kvutils.dpi_scale_width(root.ref_width)", kv_source)
        self.assertIn("kvutils.dpi_scale_height(root.ref_height)", kv_source)
        self.assertIn("ref_width = NumericProperty(16)", py_source)
        self.assertIn("ref_height = NumericProperty(16)", py_source)
        self.assertNotIn("dp(", kv_source)
        self.assertNotIn("from kivy.metrics import dp", py_source)
        self.assertNotIn("size: dp(18), dp(18)", kv_source)

    def test_box_and_focus_are_tighter_with_subtle_corner_radius(self):
        box_size_source = _load_class_function("ModernCheckBox", "_box_size")
        geometry_source = _load_class_function("ModernCheckBox", "_update_geometry")
        box_radius_source = _load_class_function("ModernCheckBox", "_box_radius")
        box_fill_radius_source = _load_class_function("ModernCheckBox", "_box_fill_radius")
        focus_radius_source = _load_class_function("ModernCheckBox", "_focus_radius")
        kv_source = CHECKBOX_KV_PATH.read_text()

        self.assertIn("side * 0.64", box_size_source)
        self.assertNotIn("dp(", box_size_source)
        self.assertIn("box_side + side * 0.06", geometry_source)
        self.assertIn("self._box_side * 0.18", box_radius_source)
        self.assertIn("self._box_side * 0.12", box_fill_radius_source)
        self.assertIn("radius: root._box_fill_radius()", kv_source)
        self.assertIn("root._box_radius()[0]", kv_source)
        self.assertIn("self._focus_side * 0.22", focus_radius_source)

    def test_line_widths_do_not_force_one_pixel_minimum(self):
        border_source = _load_class_function("ModernCheckBox", "_border_line_width")
        check_source = _load_class_function("ModernCheckBox", "_check_line_width")

        self.assertIn("max(0.65", border_source)
        self.assertIn("0.9", border_source)
        self.assertIn("max(0.8", check_source)
        self.assertIn("1.05", check_source)
        self.assertNotIn("max(1.0", border_source)
        self.assertNotIn("max(1.0", check_source)


if __name__ == "__main__":
    unittest.main()
