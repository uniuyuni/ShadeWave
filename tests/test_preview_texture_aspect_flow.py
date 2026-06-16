import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"
MAIN_KV_PATH = ROOT / "main.kv"


def _function_source(name):
    source = MAIN_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


class PreviewTextureAspectFlowTest(unittest.TestCase):
    def test_preview_source_image_size_prefers_crop_rect(self):
        source = _function_source("_preview_source_image_size")

        self.assertIn("if not self._preview_uses_full_image_size():", source)
        self.assertIn("crop_rect = params.get_crop_rect(self.primary_param)", source)
        self.assertIn("crop_width = max(1.0, float(x2) - float(x1))", source)
        self.assertIn("crop_height = max(1.0, float(y2) - float(y1))", source)
        self.assertIn("return (crop_width, crop_height)", source)
        self.assertLess(
            source.index("return (crop_width, crop_height)"),
            source.index("return (float(original_img_size[0]), float(original_img_size[1]))"),
        )

    def test_full_preview_modes_use_original_image_size(self):
        source = _function_source("_preview_uses_full_image_size")

        self.assertIn('"_mask1_full_preview_sources"', source)
        self.assertIn("return self._is_image_geometry_mode()", source)

    def test_preview_texture_side_uses_image_aspect_when_unzoomed(self):
        source = _function_source("_preview_texture_side_for_widget")

        self.assertIn("image_long = max(image_width, image_height)", source)
        self.assertIn("image_short = min(image_width, image_height)", source)
        self.assertIn("display_long = max(widget_width, widget_height)", source)
        self.assertIn("display_short = min(widget_width, widget_height)", source)
        self.assertIn("image_aspect = image_long / image_short", source)
        self.assertIn("same_orientation = (image_width >= image_height) == (widget_width >= widget_height)", source)
        self.assertIn("side = min(display_long, display_short * image_aspect)", source)
        self.assertIn("side = display_short", source)
        self.assertIn("math.isclose(image_long, image_short)", source)

    def test_preview_widget_size_keeps_click_margin_for_image(self):
        source = _function_source("_preview_widget_logical_size")
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        kv_source = MAIN_KV_PATH.read_text(encoding="utf-8")

        self.assertIn("PREVIEW_CLICK_MARGIN_DP = 6.0", main_source)
        self.assertIn("kvdp(self.PREVIEW_CLICK_MARGIN_DP)", source)
        self.assertIn("usable_width = max(1.0, float(preview_widget.width) - margin * 2.0)", source)
        self.assertIn("usable_height = max(1.0, float(preview_widget.height) - margin * 2.0)", source)
        self.assertIn("id: preview_widget", kv_source)
        self.assertNotIn("id: preview_margin", kv_source)

    def test_zoom_preview_texture_uses_rectangular_widget_size(self):
        source = _function_source("get_preview_texture_size")

        self.assertIn("if self.is_zoomed:", source)
        self.assertIn("widget_width, widget_height = self._preview_widget_logical_size(preview_widget)", source)
        self.assertIn("max(1, int(round(widget_width)))", source)
        self.assertIn("max(1, int(round(widget_height)))", source)
        self.assertIn("return (side, side)", source)
        self.assertLess(
            source.index("if self.is_zoomed:"),
            source.index("side = self._preview_texture_side_for_widget(preview_widget)"),
        )

    def test_zoom_start_uses_image_center_before_rectangular_resize(self):
        mapper_source = _function_source("_preview_texture_pos_to_image_pos")
        touch_source = _function_source("on_image_touch_down")
        keyboard_source = _function_source("_zoom_preview_from_keyboard")

        self.assertIn("core.crop_size_and_offset_from_texture(", mapper_source)
        self.assertIn("image_x = disp_info[0] + (tex_pos[0] - offset_x) / scale", mapper_source)
        self.assertIn("image_y = disp_info[1] + (tex_pos[1] - offset_y) / scale", mapper_source)
        self.assertIn("center_pos=self._preview_texture_pos_to_image_pos(tex_pos)", touch_source)
        self.assertIn("center_pos=self._preview_texture_pos_to_image_pos(tex_pos)", keyboard_source)

    def test_preview_texture_size_change_invalidates_crop_cache(self):
        source = _function_source("draw_image_core")

        self.assertIn("if self.update_preview_texture_size():", source)
        self.assertIn("self.crop_image = None", source)

    def test_zoom_reset_recomputes_unzoomed_texture_size_before_disp_info(self):
        source = _function_source("_reset_preview_zoom")

        self.assertLess(
            source.index("self.update_preview_texture_size()"),
            source.index("core.convert_rect_to_info("),
        )

    def test_tab_switch_recomputes_texture_size_before_reopening_crop_editor(self):
        source = _function_source("on_current_tab")

        self.assertIn("if self.update_preview_texture_size():", source)
        self.assertLess(
            source.index("if self.update_preview_texture_size():"),
            source.index('self.apply_effects_lv(0, "crop", overlay_reason="tab_sync")'),
        )


if __name__ == "__main__":
    unittest.main()
