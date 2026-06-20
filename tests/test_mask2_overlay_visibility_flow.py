import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MASK_EDITOR2_PATH = ROOT / "widgets" / "mask_editor2.py"
MAIN_PATH = ROOT / "main.py"


def _function_source(name):
    source = MASK_EDITOR2_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


class Mask2OverlayVisibilityFlowTest(unittest.TestCase):
    def test_overlay_target_uses_composit_for_child_masks(self):
        source = _function_source("overlay_mask_for_active")

        self.assertIn("return self._mask_parent_for_visibility(active)", source)
        self.assertIn("active.is_composit()", source)
        self.assertIn("return active", source)
        self.assertNotIn('maskop == "Subtract"', source)

    def test_visibility_policy_shows_all_same_composit_control_points(self):
        source = _function_source("mask_visibility_policy_for")

        self.assertIn('{"control_points": "all", "overlay": True}', source)
        self.assertNotIn('{"control_points": "all", "overlay": mask is active}', source)
        self.assertIn("mask_parent is not active_parent", source)

    def test_refresh_and_render_use_overlay_target_helper(self):
        refresh_source = _function_source("refresh_mask_visibility")
        request_source = _function_source("request_mask_render_update")
        active_overlay_source = _function_source("refresh_active_mask_overlay")

        self.assertIn("overlay_mask_for_active(active)", refresh_source)
        self.assertIn("overlay_mask_for_active() or mask", request_source)
        self.assertIn("overlay_mask_for_active()", active_overlay_source)

    def test_created_mask_is_visibility_reference_while_initializing(self):
        reference_source = _function_source("visibility_reference_mask")
        parent_source = _function_source("_mask_parent_for_visibility")
        refresh_source = _function_source("refresh_mask_visibility")
        create_source = _function_source("_create_start_new_mask")
        end_source = _function_source("_create_end_new_mask")

        self.assertIn("self.created_mask if self.created_mask is not None else self.get_active_mask()", reference_source)
        self.assertIn("mask is self.created_mask", parent_source)
        self.assertIn("self.created_mask_index", parent_source)
        self.assertIn("active = self.visibility_reference_mask()", refresh_source)
        self.assertIn("self.created_mask_index = index", create_source)
        self.assertIn("self._mask_overlay_enabled = True", create_source)
        self.assertLess(
            end_source.index("self.created_mask = None"),
            end_source.index("self.request_mask_render_update("),
        )

    def test_child_overlay_refreshes_child_cache_before_composit_redraw(self):
        prepare_source = _function_source("_refresh_child_mask_cache_for_overlay")
        request_source = _function_source("request_mask_render_update")
        draw_source = _function_source("_draw_overlay_mask")

        self.assertIn("overlay_mask.is_composit()", prepare_source)
        self.assertIn("_mask_parent_for_visibility(mask) is not overlay_mask", prepare_source)
        self.assertIn("mask.get_mask_image()", prepare_source)
        self.assertIn("_refresh_child_mask_cache_for_overlay(mask, overlay_mask)", request_source)
        self.assertIn("_draw_overlay_mask(overlay_mask)", request_source)
        self.assertIn("overlay_mask.draw_mask_to_fbo(True)", draw_source)
        self.assertLess(
            request_source.index("_refresh_child_mask_cache_for_overlay(mask, overlay_mask)"),
            request_source.index("_draw_overlay_mask(overlay_mask)"),
        )

    def test_m_key_temporarily_hides_overlay_and_control_points(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        set_hidden_source = _function_source("set_overlay_control_points_hidden")
        hide_source = _function_source("_hide_overlay_and_control_points")
        draw_source = _function_source("draw_mask_image")
        refresh_source = _function_source("refresh_mask_visibility")

        self.assertIn("set_overlay_control_points_hidden(True)", main_source)
        self.assertIn("set_overlay_control_points_hidden(False)", main_source)
        self.assertIn("self._overlay_control_points_hidden = hidden", set_hidden_source)
        self.assertIn("self._hide_overlay_and_control_points()", set_hidden_source)
        self.assertIn("self.refresh_mask_visibility()", set_hidden_source)
        self.assertIn("cp.opacity = 0", hide_source)
        self.assertIn("self.draw_mask_image(None)", hide_source)
        self.assertIn("self.clear_mask_geom_axes()", hide_source)
        self.assertIn("self._overlay_control_points_hidden and glayimg is not None", draw_source)
        self.assertIn("if self._overlay_control_points_hidden:", refresh_source)

    def test_hidden_overlay_does_not_revive_after_refresh(self):
        mask_editor_source = MASK_EDITOR2_PATH.read_text(encoding="utf-8")
        set_draw_source = _function_source("set_draw_mask")
        active_overlay_source = _function_source("refresh_active_mask_overlay")
        refresh_source = _function_source("refresh_mask_visibility")
        request_source = _function_source("request_mask_render_update")
        set_active_source = _function_source("set_active_mask")

        self.assertIn("self._mask_overlay_enabled = False", mask_editor_source)
        self.assertIn("self._mask_overlay_enabled = bool(is_draw_mask)", set_draw_source)
        self.assertIn("self.draw_mask_image(None)", set_draw_source)
        self.assertIn("or not self._mask_overlay_enabled", active_overlay_source)
        self.assertIn("if not self._mask_overlay_enabled:", refresh_source)
        self.assertIn("mask.is_draw_mask = False", refresh_source)
        self.assertIn("elif redraw_overlay and self._mask_overlay_enabled:", request_source)
        self.assertIn("elif redraw_overlay:", request_source)
        self.assertIn("self._mask_overlay_enabled = mask is not None", set_active_source)


if __name__ == "__main__":
    unittest.main()
