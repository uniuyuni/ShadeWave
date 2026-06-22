import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MACOS_PATH = PROJECT_ROOT / "macos.py"


def _source():
    return MACOS_PATH.read_text(encoding="utf-8")


def _load_function(name):
    source = _source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} was not found")


class MacOSAlertIconFlowTest(unittest.TestCase):
    def test_legacy_icon_names_are_mapped_to_nsalert_styles(self):
        source = _load_function("_alert_style")

        self.assertIn('"note": AppKit.NSAlertStyleInformational', source)
        self.assertIn('"caution": AppKit.NSAlertStyleWarning', source)
        self.assertIn('"stop": AppKit.NSAlertStyleCritical', source)
        self.assertIn('"informational": AppKit.NSAlertStyleInformational', source)
        self.assertIn('"warning": AppKit.NSAlertStyleWarning', source)
        self.assertIn('"critical": AppKit.NSAlertStyleCritical', source)

    def test_alert_and_confirm_use_app_modal_window(self):
        native_source = _load_function("_run_native_alert")
        alert_source = _load_function("alert")
        confirm_source = _load_function("confirm")

        self.assertIn("AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_", native_source)
        self.assertIn("app.runModalForWindow_(win)", native_source)
        self.assertIn("if not _center_window_on_app(win):", native_source)
        self.assertIn("_restore_app_window_focus()", native_source)
        self.assertIn("_run_native_alert(text, title, icon, buttons)", alert_source)
        self.assertIn("_run_native_alert(", confirm_source)
        self.assertNotIn("display alert", alert_source)
        self.assertNotIn("display alert", confirm_source)

    def test_dialog_windows_center_on_app_and_restore_focus(self):
        center_source = _load_function("_center_window_on_app")
        restore_source = _load_function("_restore_app_window_focus")
        prompt_source = _load_function("prompt_native")

        self.assertIn("parent_frame = parent.frame()", center_source)
        self.assertIn("win.setFrameOrigin_", center_source)
        self.assertIn("win.makeMainWindow()", restore_source)
        self.assertIn("win.makeKeyAndOrderFront_(None)", restore_source)
        self.assertIn("if not _center_window_on_app(win):", prompt_source)
        self.assertIn("_restore_app_window_focus()", prompt_source)
        self.assertNotIn("activateIgnoringOtherApps_", prompt_source)


if __name__ == "__main__":
    unittest.main()
