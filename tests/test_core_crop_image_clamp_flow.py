import ast
from pathlib import Path
import unittest


CORE_PATH = Path(__file__).resolve().parents[1] / "cores" / "core.py"


def _function_source(name):
    source = CORE_PATH.read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found")


class CoreCropImageClampFlowTest(unittest.TestCase):
    def test_crop_image_clamps_crop_rect_and_disp_info_to_image_before_resize(self):
        source = _function_source("crop_image")

        self.assertLess(
            source.index("crop_rect = _clamp_crop_rect_to_image"),
            source.index("new_width, new_height, offset_x, offset_y"),
        )
        self.assertLess(
            source.index("disp_info = _clamp_disp_info_to_image"),
            source.index("new_width, new_height, offset_x, offset_y"),
        )
        self.assertLess(
            source.index("disp_info = _clamp_disp_info_to_image"),
            source.index("image_transform_adapter.fit_crop_to_canvas"),
        )
        self.assertNotIn(
            "disp_info = _clamp_disp_info_to_crop_rect(disp_info, crop_rect)",
            source,
        )

    def test_crop_image_info_clamps_crop_rect_and_disp_info_before_slice(self):
        source = _function_source("crop_image_info")

        self.assertLess(
            source.index("crop_rect = _clamp_crop_rect_to_image"),
            source.index("disp_x, disp_y, disp_width, disp_height, scale"),
        )
        self.assertLess(
            source.index("disp_info = _clamp_disp_info_to_crop_rect"),
            source.index("disp_x, disp_y, disp_width, disp_height, scale"),
        )
        self.assertLess(
            source.index("disp_info = _clamp_disp_info_to_crop_rect"),
            source.index("cropped_img = image"),
        )

    def test_crop_rect_clamp_never_returns_empty_rect(self):
        source = _function_source("_clamp_crop_rect_to_image")

        self.assertIn("x2 = max(x1 + 1, min(x2, image_width))", source)
        self.assertIn("y2 = max(y1 + 1, min(y2, image_height))", source)

    def test_disp_info_clamp_never_returns_empty_size(self):
        source = _function_source("_clamp_disp_info_to_crop_rect")

        self.assertIn("disp_width = int(max(1, min(round(disp_width), crop_width)))", source)
        self.assertIn("disp_height = int(max(1, min(round(disp_height), crop_height)))", source)

    def test_preview_disp_info_clamp_is_against_image_not_crop_rect(self):
        source = _function_source("_clamp_disp_info_to_image")

        self.assertIn("_clamp_disp_info_to_crop_rect(", source)
        self.assertIn("(0, 0, max(1, image_width), max(1, image_height))", source)


if __name__ == "__main__":
    unittest.main()
