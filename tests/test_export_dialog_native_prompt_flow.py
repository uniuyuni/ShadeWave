import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORT_DIALOG_PATH = PROJECT_ROOT / "widgets" / "export_dialog.py"


def _source():
    return EXPORT_DIALOG_PATH.read_text(encoding="utf-8")


def _load_class_node(name):
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


def _load_class_function(class_name, function_name):
    source = _source()
    cls = _load_class_node(class_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class ExportDialogNativePromptFlowTest(unittest.TestCase):
    def test_export_preset_name_uses_native_prompt(self):
        source = _source()
        save_source = _load_class_function("ExportDialog", "save_preset")

        self.assertNotIn("class PresetNameDialog", source)
        self.assertNotIn("KVTextInput", source)
        self.assertIn("device.prompt_native(", save_source)
        self.assertIn('title="Save Preset"', save_source)
        self.assertIn("show_cancel=True", save_source)
        self.assertIn("self._save_preset_with_name(preset_name)", save_source)

    def test_export_confirm_dialog_stays_kivy_popup(self):
        source = _source()

        self.assertIn("class ExportConfirmDialog(KVPopup):", source)
        self.assertIn("KVButton(text='Rename')", source)
        self.assertIn("KVButton(text='Overwrite')", source)


if __name__ == "__main__":
    unittest.main()
