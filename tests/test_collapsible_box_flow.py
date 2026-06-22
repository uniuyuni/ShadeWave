import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PY_PATH = PROJECT_ROOT / "widgets" / "collapsible_box.py"
KV_PATH = PROJECT_ROOT / "widgets" / "collapsible_box.kv"


def _load_class_function(function_name):
    source = PY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CollapsibleBox":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"CollapsibleBox.{function_name} was not found")


class CollapsibleBoxFlowTest(unittest.TestCase):
    def test_collapsible_box_uses_event_driven_height_updates(self):
        source = PY_PATH.read_text(encoding="utf-8")
        init_source = _load_class_function("__init__")
        kv_source = KV_PATH.read_text(encoding="utf-8")

        self.assertIn("self.bind(is_expanded=self._schedule_content_height_update)", init_source)
        self.assertIn("content.bind(", source)
        self.assertIn("minimum_height=self._schedule_content_height_update", source)
        self.assertIn("self.content_height = content.minimum_height", source)
        self.assertIn("KVClock.create_trigger(self._update_content_height, 0)", source)
        self.assertNotIn("schedule_interval(self._update_content_height", source)
        self.assertIn("height: btn_header.height + scroll_view.height", kv_source)
        self.assertNotIn("disabled: not root.is_expanded", kv_source)
        self.assertIn("opacity: 1 if root.is_expanded else 0", kv_source)

    def test_header_uses_button_behavior_with_fixed_canvas_background(self):
        py_source = PY_PATH.read_text(encoding="utf-8")
        kv_source = KV_PATH.read_text(encoding="utf-8")
        header_block = kv_source.split("CollapsibleHeader:", 1)[1].split("ScrollView:", 1)[0]

        self.assertIn("class CollapsibleHeader(KVButtonBehavior, KVBoxLayout):", py_source)
        self.assertIn("canvas.before:", header_block)
        self.assertIn("rgba: 0.12, 0.12, 0.12, 1", header_block)
        self.assertNotIn("background_normal", header_block)
        self.assertNotIn("background_down", header_block)

    def test_dynamic_content_addition_schedules_height_update(self):
        source = _load_class_function("add_widget")

        self.assertIn("self.ids.content.add_widget(widget", source)
        self.assertIn("self._bind_child_height(widget)", source)
        self.assertIn("self._schedule_content_height_update()", source)

    def test_toggle_applies_layout_without_disabling_children(self):
        source = _load_class_function("toggle")
        layout_source = _load_class_function("_apply_layout_state")
        parent_source = _load_class_function("_schedule_parent_layout_update")
        local_parent_source = _load_class_function("_update_local_parent_layout")
        mask2_kv_source = (PROJECT_ROOT / "widgets" / "mask2_content.kv").read_text(encoding="utf-8")

        self.assertIn("self._apply_layout_state()", source)
        self.assertIn("self._schedule_parent_layout_update()", source)
        self.assertIn("scroll_view.height = self.content_height if self.is_expanded else 0", layout_source)
        self.assertIn("self.height = header.height + scroll_view.height", layout_source)
        self.assertIn("self._update_local_parent_layout()", parent_source)
        self.assertIn("parent = getattr(self, \"parent\", None)", local_parent_source)
        self.assertIn("parent.height = parent.minimum_height", local_parent_source)
        self.assertNotIn("while widget is not None", local_parent_source)
        self.assertNotIn("disabled: True if root.disabled == True else False", mask2_kv_source)


if __name__ == "__main__":
    unittest.main()
