import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
MASK_EDITOR_PATH = ROOT / "widgets" / "mask_editor2.py"
HEADLESS_MASKS_PATH = ROOT / "cores" / "mask2" / "headless_masks.py"


def _class_method_source(path, class_name, method_name):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(source, item)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


class AIMaskRotationUnitsTest(unittest.TestCase):
    def test_kivy_depth_map_converts_tcg_rotation_to_degrees(self):
        source = _class_method_source(MASK_EDITOR_PATH, "DepthMapMask", "get_mask_image")

        self.assertIn("rotate_rad, flip, matrix", source)
        self.assertIn("np.rad2deg(rotate_rad)", source)
        self.assertNotIn("core.rotation(depth_map_mask, rotate_rad", source)

    def test_headless_depth_map_converts_tcg_rotation_to_degrees(self):
        source = _class_method_source(HEADLESS_MASKS_PATH, "HeadlessDepthMapMask", "get_mask_image")

        self.assertIn("_, rotate_rad, flip, matrix", source)
        self.assertIn("np.rad2deg(rotate_rad)", source)
        self.assertNotIn("core.rotation(\n                depth_map_mask, rotate_rad", source)


if __name__ == "__main__":
    unittest.main()
