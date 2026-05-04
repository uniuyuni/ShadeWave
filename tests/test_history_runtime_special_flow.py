import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "history.py"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"


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


class HistoryRuntimeSpecialFlowTest(unittest.TestCase):
    def test_all_history_restore_preserves_missing_runtime_special_params(self):
        undo = _load_function(HISTORY_PATH, "undo")
        redo = _load_function(HISTORY_PATH, "redo")
        source = HISTORY_PATH.read_text()

        self.assertIn("def _restore_missing_runtime_special", source)
        self.assertIn("params.copy_special_param(runtime_special, widget.primary_param)", source)
        self.assertGreaterEqual(
            source.count("self._restore_missing_runtime_special(widget.primary_param, runtime_special)"),
            4,
        )

        for node in (undo, redo):
            names = [
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            ]
            self.assertIn("_restore_missing_runtime_special", names)

    def test_color_temperature_still_requires_real_reset_basis(self):
        make_diff = _load_class_function(EFFECTS_PATH, "ColorTemperatureEffect", "make_diff")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), make_diff)

        self.assertIn("param['color_temperature_reset']", source)
        self.assertIn("param['color_tint_reset']", source)

    def test_effect_history_snapshots_mutable_values(self):
        source = HISTORY_PATH.read_text()

        self.assertIn("def _copy_history_value", source)
        self.assertIn("value.copy()", source)
        self.assertIn("copy.deepcopy(value)", source)
        self.assertIn("self.backup[key] = self._copy_history_value(val)", source)
        self.assertIn("self.update[key] = self._copy_history_value(val)", source)


if __name__ == "__main__":
    unittest.main()
