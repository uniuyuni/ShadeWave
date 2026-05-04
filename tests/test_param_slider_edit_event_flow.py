import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAM_SLIDER_PATH = PROJECT_ROOT / "widgets" / "param_slider.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class ParamSliderEditEventFlowTest(unittest.TestCase):
    def test_edit_events_are_counters_not_value_assignments(self):
        source = PARAM_SLIDER_PATH.read_text()

        self.assertIn("before_edit = KVNumericProperty(0)", source)
        self.assertIn("after_edit = KVNumericProperty(0)", source)
        self.assertIn("def _notify_before_edit", source)
        self.assertIn("self.before_edit += 1", source)
        self.assertIn("def _notify_after_edit", source)
        self.assertIn("self.after_edit += 1", source)

    def test_slider_touch_down_always_emits_before_edit(self):
        on_slider_touch_down = _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "on_slider_touch_down")
        source = ast.get_source_segment(PARAM_SLIDER_PATH.read_text(), on_slider_touch_down)

        self.assertIn("self._notify_before_edit()", source)
        self.assertNotIn("self.before_edit = self.value", source)


if __name__ == "__main__":
    unittest.main()
