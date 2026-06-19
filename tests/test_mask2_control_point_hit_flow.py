import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
MASK_EDITOR_PATH = ROOT / "widgets" / "mask_editor2.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def _class_source(path, name):
    source = _source(path)
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"class not found: {name}")


class Mask2ControlPointHitFlowTest(unittest.TestCase):
    def test_control_point_hit_area_matches_visible_circle_not_widget_size(self):
        source = _class_source(MASK_EDITOR_PATH, "ControlPoint")
        tree = ast.parse(source)
        assigns_self_size = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assigns_self_size.extend(
                    target for target in node.targets
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "size"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    )
                )

        self.assertIn("HIT_RADIUS_PX = 10.0", source)
        self.assertIn("def collide_point(self, x, y):", source)
        self.assertIn("tcg_to_window_for_overlay", source)
        self.assertIn("self.HIT_RADIUS_PX * self.HIT_RADIUS_PX", source)
        self.assertEqual(assigns_self_size, [])


if __name__ == "__main__":
    unittest.main()
