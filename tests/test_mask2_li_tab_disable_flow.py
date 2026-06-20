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

    def test_mask2_state_disables_li_tab_and_switches_away(self):
        source_text = MAIN_PATH.read_text()
        helper_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_set_li_tab_disabled_for_mask2"),
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
        self.assertIn("disabled = self._is_mask2_enabled()", helper_source)
        self.assertIn("li_tab.disabled = disabled", helper_source)
        self.assertIn('getattr(current_tab, "text", None) != "Li"', helper_source)
        self.assertIn('self.ids.get("tab_mask2")', helper_source)
        self.assertIn('self._find_effect_tab("M2")', helper_source)
        self.assertIn('self.ids.get("tab_basic")', helper_source)
        self.assertIn("effects_panel.switch_to(fallback)", helper_source)
        self.assertIn("self._set_li_tab_disabled_for_mask2()", update_source)
        self.assertIn("self._set_li_tab_disabled_for_mask2()", enable_source)
        self.assertIn("self._set_li_tab_disabled_for_mask2()", disable_source)
        self.assertIn('getattr(current, "text", None) == "Li"', tab_source)
        self.assertIn("self._is_mask2_enabled()", tab_source)


if __name__ == "__main__":
    unittest.main()
