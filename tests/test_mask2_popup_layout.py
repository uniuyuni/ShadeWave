import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
MASK2_CONTENT_PATH = ROOT / "widgets" / "mask2_content.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def _function_source(path, name):
    source = _source(path)
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


class Mask2PopupLayoutTest(unittest.TestCase):
    def test_add_mask_popup_height_is_content_driven(self):
        source = _source(MASK2_CONTENT_PATH)
        show_popup = _function_source(MASK2_CONTENT_PATH, "show_add_mask_popup")
        height_helper = _function_source(MASK2_CONTENT_PATH, "_add_mask_popup_ref_height")

        self.assertIn("_ADD_MASK_POPUP_BUTTON_HEIGHT_REF * item_count", height_helper)
        self.assertIn("_ADD_MASK_POPUP_CHROME_HEIGHT_REF", height_helper)
        self.assertIn("KVWindow.height", height_helper)
        self.assertIn("popup.ref_height = _add_mask_popup_ref_height(len(types))", show_popup)
        self.assertIn("dialogutils.install_ref_scaling(popup, on_rescale=_fit_popup_height)", show_popup)
        self.assertNotIn("popup.ref_height = 420", source)


if __name__ == "__main__":
    unittest.main()
