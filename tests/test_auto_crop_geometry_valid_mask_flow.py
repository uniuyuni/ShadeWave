import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
FIND_BOUNDING_BOX_PATH = PROJECT_ROOT / "cores" / "find_bounding_box.py"


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


class AutoCropGeometryValidMaskFlowTest(unittest.TestCase):
    def test_auto_crop_uses_geometry_valid_mask_when_param_is_available(self):
        node = _load_class_function(EFFECTS_PATH, "CropEffect", "auto_crop_editor")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertIn("valid_mask = _build_geometry_valid_mask(param)", source)
        self.assertIn("find_largest_inscribed_rectangle_in_mask(", source)
        self.assertIn("threshold=0.999", source)
        self.assertIn("aspect_ratio=aspect_ratio", source)
        self.assertIn("self.crop_editor.set_to_local_crop_rect(bbox, enforce_bounds=param is None)", source)

    def test_crop_auto_button_passes_primary_param_to_auto_crop(self):
        source = EFFECTS_PATH.read_text()

        self.assertIn("self.auto_crop_editor(self.backup_img, param)", source)

    def test_geometry_valid_mask_replays_all_geometry_black_border_sources(self):
        node = _load_function(EFFECTS_PATH, "_build_geometry_valid_mask")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertIn("np.ones((height, width, 3), dtype=np.float32)", source)
        self.assertIn("rotation_limit_mask = core.rotation(", source)
        self.assertIn("correct_lens_distortion(", source)
        self.assertIn("core.rotation(", source)
        self.assertIn('border_mode="constant"', source)
        self.assertIn("correct_trapezoid(", source)
        self.assertIn("correct_four_points(", source)
        self.assertIn("correct_with_lines(", source)
        self.assertIn("warp_mesh(", source)
        self.assertIn("return np.minimum(mask, rotation_limit_mask)", source)

    def test_geometry_valid_mask_is_clipped_to_rotation_frame_after_distortion(self):
        node = _load_function(EFFECTS_PATH, "_build_geometry_valid_mask")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertLess(source.index("rotation_limit_mask = core.rotation("), source.index("mask = core.rotation("))
        self.assertLess(source.index("correct_trapezoid("), source.index("return np.minimum(mask, rotation_limit_mask)"))
        self.assertLess(source.index("correct_with_lines("), source.index("return np.minimum(mask, rotation_limit_mask)"))

    def test_mask_rectangle_search_does_not_fill_or_close_invalid_black_areas(self):
        node = _load_function(FIND_BOUNDING_BOX_PATH, "find_largest_inscribed_rectangle_in_mask")
        source = ast.get_source_segment(FIND_BOUNDING_BOX_PATH.read_text(), node)

        self.assertIn("valid_mask = (mask >= threshold).astype(np.uint8) * 255", source)
        self.assertIn("_find_largest_inscribed_rectangle_with_aspect", source)
        self.assertIn("_find_largest_inscribed_rectangle(valid_mask)", source)
        self.assertNotIn("morphologyEx", source)
        self.assertNotIn("drawContours", source)

    def test_auto_crop_result_is_not_rebounded_by_rotation_only_editor_logic(self):
        node = _load_class_function(EFFECTS_PATH, "CropEffect", "auto_crop_editor")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertIn("enforce_bounds=param is None", source)

    def test_auto_crop_commit_does_not_rebound_valid_mask_bbox(self):
        for function_name in ("set2param", "apply_crop_button_action"):
            node = _load_class_function(EFFECTS_PATH, "CropEffect", function_name)
            source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

            self.assertIn("enforce_bounds", source)
            self.assertIn("get_crop_rect(enforce_bounds=enforce_bounds)", source)


if __name__ == "__main__":
    unittest.main()
