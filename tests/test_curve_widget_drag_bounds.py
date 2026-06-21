import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CURVE_PATH = PROJECT_ROOT / "widgets" / "curve.py"


def _load_class_function(class_name, function_name):
    source = CURVE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class CurveWidgetDragBoundsTest(unittest.TestCase):
    def test_dragged_control_point_can_clamp_to_widget_edges_outside_bounds(self):
        down_source = _load_class_function("CurveWidget", "on_touch_down")
        move_source = _load_class_function("CurveWidget", "on_touch_move")
        up_source = _load_class_function("CurveWidget", "on_touch_up")

        self.assertIn("touch.grab(self)", down_source)
        self.assertIn("dragging_self = self.selected_point is not None", move_source)
        self.assertIn("getattr(touch, \"grab_current\", None) is self", move_source)
        self.assertIn("if not dragging_self and not self.collide_point(*touch.pos):", move_source)
        self.assertIn("np.clip(local_x, 0.0, 1.0)", move_source)
        self.assertIn("np.clip(local_y, 0.0, 1.0)", move_source)
        self.assertIn("touch.ungrab(self)", up_source)


if __name__ == "__main__":
    unittest.main()
