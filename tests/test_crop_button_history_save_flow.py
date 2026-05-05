import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"


def _load_function(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class CropButtonHistorySaveFlowTest(unittest.TestCase):
    def test_crop_buttons_use_explicit_history_action_not_button_state_polling(self):
        source = MAIN_KV_PATH.read_text()

        self.assertIn("on_release: root.apply_crop_button_action('reset')", source)
        self.assertIn("on_release: root.apply_crop_button_action('auto')", source)

    def test_crop_button_action_records_history_redraws_and_saves_sidecar(self):
        node = _load_function(MAIN_PATH, "apply_crop_button_action")
        call_names = [
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        ]

        self.assertIn("begin_history_effect_ctrl", call_names)
        self.assertIn("apply_crop_button_action", call_names)
        self.assertIn("start_draw_image_and_crop", call_names)
        self.assertIn("end_history_effect_ctrl", call_names)
        self.assertIn("save_current_sidecar", call_names)

    def test_crop_effect_button_action_commits_crop_rect_and_disp_info(self):
        node = _load_class_function(EFFECTS_PATH, "CropEffect", "apply_crop_button_action")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertIn('if action == "reset":', source)
        self.assertIn('if action == "auto":', source)
        self.assertIn("params.set_crop_rect(param, self.crop_editor.get_crop_rect(enforce_bounds=enforce_bounds))", source)
        self.assertIn("params.set_disp_info(param, self.crop_editor.get_disp_info(enforce_bounds=enforce_bounds))", source)

    def test_crop_reset_syncs_aspect_ratio_before_resetting_editor(self):
        for function_name in ("set2param", "apply_crop_button_action"):
            node = _load_class_function(EFFECTS_PATH, "CropEffect", function_name)
            source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

            self.assertLess(
                source.index("self.reset2_crop_editor(param)"),
                source.index("self.reset_crop_editor()"),
            )


if __name__ == "__main__":
    unittest.main()
