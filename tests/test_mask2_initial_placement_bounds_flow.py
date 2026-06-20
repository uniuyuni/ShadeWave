import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
MASK_EDITOR_PATH = ROOT / "widgets" / "mask_editor2.py"


def _class_function_source(class_name, function_name):
    source = MASK_EDITOR_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return ast.get_source_segment(source, item)
    raise AssertionError(f"{class_name}.{function_name} not found")


class Mask2InitialPlacementBoundsFlowTest(unittest.TestCase):
    def test_initial_placement_area_allows_preview_padding(self):
        source = _class_function_source("BaseMask", "_touch_in_initial_placement_area")

        self.assertIn("self.editor.collide_point(*touch.pos)", source)
        self.assertNotIn("window_point_in_image_rect", source)

    def test_cp_based_masks_ignore_initial_down_outside_preview(self):
        for class_name in (
            "CircularGradientMask",
            "GradientMask",
            "FullMask",
            "SegmentMask",
            "DepthMapMask",
            "FaceMask",
            "TargetTextMask",
        ):
            with self.subTest(class_name=class_name):
                source = _class_function_source(class_name, "on_touch_down")
                self.assertIn("if self.initializing:", source)
                self.assertIn("self._begin_initial_touch_if_in_placement_area(touch)", source)
                self.assertIn("return False", source)

    def test_cp_based_masks_do_not_finish_without_initial_down(self):
        for class_name in (
            "CircularGradientMask",
            "GradientMask",
            "FullMask",
            "SegmentMask",
            "DepthMapMask",
            "FaceMask",
            "TargetTextMask",
        ):
            with self.subTest(class_name=class_name):
                source = _class_function_source(class_name, "on_touch_up")
                self.assertIn("if self.initializing:", source)
                self.assertIn("self._initial_touch_can_finish()", source)

    def test_polyline_first_point_uses_same_initial_placement_guard(self):
        source = _class_function_source("PolylineMask", "on_touch_down")

        self.assertIn("was_initializing = self.initializing", source)
        self.assertIn("self._begin_initial_touch_if_in_placement_area(touch)", source)
        self.assertIn("self._initial_touch_started = False", source)

    def test_mask_editor_does_not_finish_created_mask_without_initial_down(self):
        source = _class_function_source("MaskEditor2", "on_touch_up")

        self.assertIn("self.created_mask is not None", source)
        self.assertIn("getattr(self.created_mask, 'initializing', False)", source)
        self.assertIn("getattr(self.created_mask, '_initial_touch_started', False)", source)
        self.assertIn("return False", source)

    def test_segment_cp_drag_defers_rerender_until_touch_up(self):
        move_source = _class_function_source("SegmentMask", "on_touch_move")
        up_source = _class_function_source("SegmentMask", "on_touch_up")
        down_source = _class_function_source("SegmentMask", "on_touch_down")

        self.assertIn("self.editor.draw_mask_image(None)", down_source)
        self.assertIn("cp.on_touch_move(touch)", move_source)
        self.assertIn("self.update_mask()", move_source)
        self.assertNotIn("request_mask_render_update", move_source)
        self.assertNotIn("update_draw_mask()", move_source)
        self.assertIn("reason=\"segment_control_point_touch_up\"", up_source)
        self.assertIn("redraw_pipeline=True", up_source)


if __name__ == "__main__":
    unittest.main()
