import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER_PATH = PROJECT_ROOT / "widgets" / "viewer.py"


def _load_class_function(class_name, function_name):
    source = VIEWER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class ViewerSelectionFlowTest(unittest.TestCase):
    def test_reselecting_single_selected_card_does_not_notify_reload(self):
        source = _load_class_function("ViewerWidget", "handle_selection")

        self.assertIn("already_single_selected = (", source)
        self.assertIn("index in self.selected_indices", source)
        self.assertIn("len(self.selected_indices) == 1", source)
        self.assertIn("if already_single_selected:", source)
        self.assertIn("self.last_selected_index = index", source)
        self.assertIn("return", source)
        plain_click_branch = source.split("if 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers:", 1)[1]
        self.assertLess(
            plain_click_branch.index("if already_single_selected:"),
            plain_click_branch.index("self.clear_selection()"),
        )

    def test_ai_job_state_never_initializes_recycle_data_as_none(self):
        new_item_source = _load_class_function("ViewerWidget", "_new_image_item")
        set_path_source = _load_class_function("ViewerWidget", "set_path")
        set_state_source = _load_class_function("ViewerWidget", "set_ai_job_state_for_path")

        self.assertIn("'ai_job_state': \"\"", new_item_source)
        self.assertIn("'ai_job_state': \"\"", set_path_source)
        self.assertIn('else ""', set_state_source)
        self.assertNotIn("'ai_job_state': None", new_item_source)
        self.assertNotIn("'ai_job_state': None", set_path_source)

    def test_ai_job_indicator_uses_loading_spinner_size(self):
        source = _load_class_function("ThumbnailCard", "__init__")
        ai_icon_block = source.split("self.ai_job_icon = KVImage(", 1)[1].split("self.image_box.add_widget(self.ai_job_icon)", 1)[0]

        self.assertIn("source=rel(\"assets\", \"spinner.gif\")", ai_icon_block)
        self.assertIn("size_hint=(1, 1)", ai_icon_block)
        self.assertIn("pos_hint={\"x\": 0, \"y\": 0}", ai_icon_block)
        self.assertNotIn("_PMCK_ICON_REF_SIZE", ai_icon_block)


if __name__ == "__main__":
    unittest.main()
