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


class LoadDependentPanelsFlowTest(unittest.TestCase):
    def test_presets_and_history_panels_follow_full_decode_wait_flag(self):
        source = MAIN_PATH.read_text()
        update = _load_function_node("update_load_dependent_panels_enabled")
        update_source = ast.get_source_segment(source, update)

        self.assertIn("disabled = bool(self.mask2_wait_full_load)", update_source)
        self.assertIn('"preset_panel"', update_source)
        self.assertIn('"history_panel"', update_source)
        self.assertIn("panel.disabled = disabled", update_source)

    def test_load_stage_changes_refresh_panel_enabled_state(self):
        on_fcs_get_file = _load_function_node("on_fcs_get_file")
        update_mask2_options_enabled = _load_function_node("update_mask2_options_enabled")
        on_kv_post = _load_function_node("on_kv_post")

        for node in (on_fcs_get_file, update_mask2_options_enabled, on_kv_post):
            call_names = [
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            ]
            self.assertIn("update_load_dependent_panels_enabled", call_names)

    def test_user_preset_and_history_actions_are_ignored_while_waiting_full_decode(self):
        for function_name in ("_on_history_selected", "start_add_preset", "apply_preset_path", "confirm_delete_preset"):
            node = _load_function_node(function_name)
            source = ast.get_source_segment(MAIN_PATH.read_text(), node)
            self.assertIn("if self.mask2_wait_full_load:", source)
            self.assertIn("return", source)

    def test_save_preset_name_uses_native_macos_prompt(self):
        source = MAIN_PATH.read_text()
        node = _load_function_node("_open_preset_name_dialog")
        dialog_source = ast.get_source_segment(source, node)

        self.assertIn("device.prompt_native(", dialog_source)
        self.assertIn('title="Save Preset"', dialog_source)
        self.assertIn("show_cancel=True", dialog_source)
        self.assertIn("if not preset_name or not preset_name.strip():", dialog_source)
        self.assertNotIn("KVTextInput", dialog_source)
        self.assertNotIn("KVPopup", dialog_source)


if __name__ == "__main__":
    unittest.main()
