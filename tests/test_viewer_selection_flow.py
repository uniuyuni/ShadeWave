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


if __name__ == "__main__":
    unittest.main()
