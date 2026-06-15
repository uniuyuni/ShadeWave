import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAM_SLIDER_PATH = PROJECT_ROOT / "widgets" / "param_slider.py"
PARAM_SLIDER_KV_PATH = PROJECT_ROOT / "widgets" / "param_slider.kv"
MULTI_SLIDER_PATH = PROJECT_ROOT / "widgets" / "multi_slider.py"


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

    def test_multi_slider_starts_edit_before_touch_value_update(self):
        source_text = MULTI_SLIDER_PATH.read_text()
        on_touch_down = _load_class_function(MULTI_SLIDER_PATH, "MultiSlider", "on_touch_down")
        source = ast.get_source_segment(source_text, on_touch_down)

        self.assertIn("interaction_start_callback", source)
        self.assertLess(
            source.index("self.interaction_start_callback()"),
            source.index("self._update_value_from_touch_x(touch.x)"),
        )

    def test_param_slider_wires_multi_slider_interaction_callbacks(self):
        kv_source = PARAM_SLIDER_KV_PATH.read_text()
        before_source = ast.get_source_segment(
            PARAM_SLIDER_PATH.read_text(),
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "_notify_before_edit"),
        )

        self.assertIn("interaction_start_callback: root.on_slider_interaction_start", kv_source)
        self.assertIn("interaction_end_callback: root.on_slider_interaction_end", kv_source)
        self.assertIn("if self._editing:", before_source)


if __name__ == "__main__":
    unittest.main()
