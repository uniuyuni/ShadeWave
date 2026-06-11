import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class Mask2OverlayPolicyFlowTest(unittest.TestCase):
    def test_overlay_policy_centralizes_mask2_show_hide_preserve(self):
        source_text = MAIN_PATH.read_text()
        node = _load_class_function(MAIN_PATH, "MainWidget", "_mask_overlay_policy")
        source = ast.get_source_segment(source_text, node)

        self.assertIn('reason == "tab_sync"', source)
        self.assertIn('return "preserve"', source)
        self.assertIn('mask2_group in ("mask2", "mask_geometry")', source)
        self.assertIn('return "show"', source)
        self.assertIn('mask2_group == "mask2_draw_effects"', source)
        self.assertIn('return "hide"', source)

    def test_tab_switch_preserves_existing_mask_overlay_state(self):
        source_text = MAIN_PATH.read_text()
        node = _load_class_function(MAIN_PATH, "MainWidget", "on_current_tab")
        source = ast.get_source_segment(source_text, node)

        self.assertIn('self.apply_effects_lv(0, "geometry", overlay_reason="tab_sync")', source)
        self.assertIn('self.apply_effects_lv(0, "crop", overlay_reason="tab_sync")', source)
        self.assertIn('self.apply_effects_lv(1, "distortion", overlay_reason="tab_sync")', source)

    def test_apply_effects_uses_overlay_policy_helper(self):
        source_text = MAIN_PATH.read_text()
        node = _load_class_function(MAIN_PATH, "MainWidget", "apply_effects_lv")
        source = ast.get_source_segment(source_text, node)

        self.assertIn("overlay_reason=\"param_change\"", source)
        self.assertIn("self._apply_mask_overlay_policy(", source)
        self.assertIn("reason=overlay_reason", source)
        self.assertNotIn("set_draw_mask(", source)


if __name__ == "__main__":
    unittest.main()
