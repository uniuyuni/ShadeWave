import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class Mask2LiTabDisableFlowTest(unittest.TestCase):
    def test_li_tab_has_stable_id_and_basic_fallback(self):
        kv_source = MAIN_KV_PATH.read_text()

        self.assertIn("id: tab_basic", kv_source)
        self.assertIn("text: 'Ba'", kv_source)
        self.assertIn("id: tab_mask2", kv_source)
        self.assertIn("text: 'M2'", kv_source)
        self.assertIn("id: tab_li", kv_source)
        self.assertIn("text: 'Li'", kv_source)

    def test_mask2_state_keeps_li_tab_available_but_editor_requires_composit(self):
        source_text = MAIN_PATH.read_text()
        helper_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_set_li_tab_disabled_for_mask2"),
        )
        block_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_is_li_tab_blocked_for_mask2"),
        )
        can_open_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "can_open_liquify_editor"),
        )
        update_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "update_mask2_options_enabled"),
        )
        enable_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_enable_mask2"),
        )
        disable_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_disable_mask2"),
        )
        tab_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "on_current_tab"),
        )

        self.assertIn('self.ids.get("tab_li")', helper_source)
        self.assertIn('self._find_effect_tab("Li")', helper_source)
        self.assertIn("disabled = self._is_li_tab_blocked_for_mask2()", helper_source)
        self.assertIn("li_tab.disabled = disabled", helper_source)
        self.assertIn("return False", block_source)
        self.assertIn("if not self._is_mask2_enabled():", can_open_source)
        self.assertIn("active = editor.get_active_mask()", can_open_source)
        self.assertIn("active is None or not active.is_composit()", can_open_source)
        self.assertIn("self.is_mask_mesh_editor_active()", can_open_source)
        self.assertIn("self._has_initializing_mask()", can_open_source)
        self.assertIn("editor.get_created_mask()", can_open_source)
        self.assertIn("self._close_inactive_distortion_painters(None)", helper_source)
        self.assertNotIn("switch_to", helper_source)
        self.assertIn("self._set_li_tab_disabled_for_mask2()", update_source)
        self.assertIn("self._set_li_tab_disabled_for_mask2()", enable_source)
        self.assertIn("self._set_li_tab_disabled_for_mask2()", disable_source)
        self.assertNotIn("switch_to", tab_source)


if __name__ == "__main__":
    unittest.main()
