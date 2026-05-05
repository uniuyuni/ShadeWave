import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "history.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class HistoryDiffOrderFlowTest(unittest.TestCase):
    def test_effect_diff_preserves_effect_param_order_for_history_label(self):
        node = _load_class_function(HISTORY_PATH, "Operation", "set_update")
        source = ast.get_source_segment(HISTORY_PATH.read_text(), node)

        self.assertIn("keys = list(self.backup.keys())", source)
        self.assertIn("keys.extend(key for key in self.update.keys() if key not in self.backup)", source)
        self.assertIn("for key in keys", source)
        self.assertNotIn("self.backup.keys() | self.update.keys()", source)


if __name__ == "__main__":
    unittest.main()
