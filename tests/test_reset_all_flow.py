import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_reset_all_node():
    tree = ast.parse(MAIN_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "reset_all":
            return node
    raise AssertionError("reset_all was not found")


def _call_name(call):
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _call_owner_name(call):
    if isinstance(call.func, ast.Attribute):
        if isinstance(call.func.value, ast.Name):
            return call.func.value.id
        if isinstance(call.func.value, ast.Attribute):
            return call.func.value.attr
    return None


class ResetAllFlowTest(unittest.TestCase):
    def test_reset_all_does_not_preserve_crop_rect_remain_param(self):
        reset_all = _load_reset_all_node()
        calls = [node for node in ast.walk(reset_all) if isinstance(node, ast.Call)]

        self.assertFalse(
            any(_call_name(call) == "copy_remain_param" for call in calls),
            "All Reset must not preserve REMAIN_PARAM such as crop_rect.",
        )

    def test_reset_all_syncs_widgets_before_crop_apply_reads_widget_values(self):
        reset_all = _load_reset_all_node()
        calls = [node for node in ast.walk(reset_all) if isinstance(node, ast.Call)]

        set2widget_lines = [
            call.lineno
            for call in calls
            if _call_name(call) == "set2widget_all" and _call_owner_name(call) == "self"
        ]
        crop_apply_lines = [
            call.lineno
            for call in calls
            if _call_name(call) == "apply_effects_lv"
            and len(call.args) >= 2
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == 0
            and isinstance(call.args[1], ast.Constant)
            and call.args[1].value == "crop"
        ]

        self.assertTrue(set2widget_lines, "reset_all should sync widgets through MainWidget.set2widget_all.")
        self.assertTrue(crop_apply_lines, "reset_all should apply the crop effect after resetting.")
        self.assertLess(
            min(set2widget_lines),
            min(crop_apply_lines),
            "Widgets must be synced before crop apply reads spinner_acpect_ratio.text.",
        )

    def test_reset_all_uses_guarded_widget_sync_not_direct_effect_sync(self):
        reset_all = _load_reset_all_node()
        calls = [node for node in ast.walk(reset_all) if isinstance(node, ast.Call)]

        self.assertFalse(
            any(_call_name(call) == "set2widget_all" and _call_owner_name(call) == "effects" for call in calls),
            "reset_all should use self.set2widget_all so widget event handlers are guarded.",
        )


if __name__ == "__main__":
    unittest.main()
