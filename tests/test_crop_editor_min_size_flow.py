import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CROP_EDITOR_PATH = PROJECT_ROOT / "widgets" / "crop_editor.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class CropEditorMinSizeFlowTest(unittest.TestCase):
    def test_min_crop_size_constants_are_128_pixels(self):
        source = CROP_EDITOR_PATH.read_text()

        self.assertIn("_MIN_CROP_WIDTH = 128", source)
        self.assertIn("_MIN_CROP_HEIGHT = 128", source)
        self.assertIn("_CROP_MOVE_SLIDE_SOFTNESS = kvdp(4)", source)

    def test_crop_rect_export_paths_enforce_min_size(self):
        for function_name in ("get_crop_rect", "get_disp_info"):
            node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", function_name)
            source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)
            self.assertIn("enforce_bounds=True", source)
            self.assertIn("if enforce_bounds:", source)
            self.assertIn("self.crop_rect = self._enforce_min_crop_rect(self.crop_rect)", source)

    def test_set_to_local_crop_rect_can_skip_editor_rebounding_for_auto_crop(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "set_to_local_crop_rect")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("def set_to_local_crop_rect(self, crop_rect, enforce_bounds=True):", source)
        self.assertIn("if enforce_bounds:", source)
        self.assertIn("crop_rect = self._enforce_min_crop_rect(crop_rect)", source)

    def test_reset_zero_crop_with_aspect_ratio_uses_max_area_at_image_center(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "set_to_local_crop_rect")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("if self.aspect_ratio > 0:", source)
        self.assertIn("self.crop_rect = self._max_aspect_crop_rect_at_image_center()", source)
        self.assertIn("return", source)

    def test_aspect_reset_maximizes_area_around_image_center(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_max_aspect_crop_rect_at_image_center")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("center_x, center_y = self._crop_image_center()", source)
        self.assertIn("for _ in range(24):", source)
        self.assertIn("width = height * aspect_ratio", source)
        self.assertIn("if self._crop_rect_inside_image(rect):", source)
        self.assertIn("best_rect = rect", source)

    def test_drag_resize_paths_enforce_min_size(self):
        for function_name in ("__resize_by_corner2", "__resize_by_edge", "__resize_crop", "on_touch_up"):
            node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", function_name)
            source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)
            self.assertIn("_enforce_min_crop_rect", source)

    def test_label_size_string_cannot_divide_by_zero(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "update_rect")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("gcd = max(gcd, 1)", source)

    def test_min_size_enforcement_keeps_rect_inside_rotated_image_and_crop_square(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_enforce_min_crop_rect")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("self._keep_crop_rect_inside_image(crop_rect)", source)
        self.assertIn("return self._keep_crop_rect_inside_image((x1, y1, x2, y2))", source)

    def test_boundary_clamp_checks_rotated_image_then_crop_square(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_keep_crop_rect_inside_image")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("for cx, cy in self._crop_rect_corners(rect):", source)
        self.assertIn("rotate_and_correct_point(", source)
        self.assertIn("self.input_angle", source)
        self.assertIn("rect = self._translate_crop_rect(rect, dx, dy)", source)
        self.assertIn("rect = self._keep_crop_rect_inside_square(rect)", source)

    def test_move_drag_clamps_delta_against_image_before_applying(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "__move_rect")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("requested_dx, requested_dy = dx, dy", source)
        self.assertIn("dx, dy = self._clamp_move_delta(dx, dy)", source)
        self.assertIn("dx, dy = self._unlock_move_delta(requested_dx, requested_dy)", source)
        self.assertIn("self._translate_crop_rect(self.crop_rect, dx, dy)", source)

    def test_move_delta_uses_binary_search_for_largest_inside_move(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_clamp_move_delta")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("target_rect = self._translate_crop_rect(self.crop_rect, dx, dy)", source)
        self.assertIn("if self._crop_rect_inside_image(target_rect):", source)
        self.assertIn("slide_rect = self._keep_crop_rect_inside_image(target_rect)", source)
        self.assertIn("self._same_crop_size(self.crop_rect, slide_rect)", source)
        self.assertIn("self._crop_rect_shift_distance(target_rect, slide_rect) <= self._crop_move_slide_softness()", source)
        self.assertIn("for _ in range(12):", source)
        self.assertIn("dx * mid", source)
        self.assertIn("return dx * low, dy * low", source)

    def test_exact_max_crop_rect_unlocks_on_first_move_only_when_fully_locked(self):
        source = CROP_EDITOR_PATH.read_text()

        self.assertIn("_CROP_MOVE_UNLOCK_MARGIN = kvdp(0.5)", source)
        self.assertIn("def _unlock_move_delta", source)
        self.assertIn("if self._crop_rect_can_move(self.crop_rect):", source)
        self.assertIn("unlocked = self._inset_crop_rect_for_move(self.crop_rect)", source)
        self.assertIn("self.crop_rect = unlocked", source)
        self.assertIn("def _crop_rect_can_move", source)
        self.assertIn("def _inset_crop_rect_for_move", source)

    def test_square_clamp_limits_rect_against_zero_and_max_square_bounds(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_keep_crop_rect_inside_square")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("max_side = self._crop_bounds_local()", source)
        self.assertIn("dx = max(0, -min_x) + min(0, max_side - max_x)", source)
        self.assertIn("dy = max(0, -min_y) + min(0, max_side - max_y)", source)

    def test_inside_check_uses_square_bounds_and_rotated_image_geometry(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_point_inside_image")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("if not self._point_inside_square(x, y):", source)
        self.assertIn("get_point_position_in_rotated_rectangle(", source)
        self.assertIn("self.input_angle", source)
        self.assertIn("return position != PointPosition.OUTSIDE", source)

    def test_square_inside_check_uses_hard_crop_square_bounds(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_point_inside_square")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("return 0 <= x <= max_side and 0 <= y <= max_side", source)

    def test_point_correction_uses_hard_boundary(self):
        source = CROP_EDITOR_PATH.read_text()

        self.assertIn("def rotate_and_correct_point(point_x, point_y, old_px, old_py, rect_width, rect_height, angle_degrees):", source)
        self.assertNotIn("tolerance=_CROP_BOUNDARY_SOFTNESS", source)

    def test_crop_image_center_keeps_reset_anchor_explicit(self):
        node = _load_class_function(CROP_EDITOR_PATH, "CropEditor", "_crop_image_center")
        source = ast.get_source_segment(CROP_EDITOR_PATH.read_text(), node)

        self.assertIn("center = self._crop_bounds_local() / 2", source)
        self.assertIn("return center, center", source)


if __name__ == "__main__":
    unittest.main()
