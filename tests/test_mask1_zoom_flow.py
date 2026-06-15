import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]


class Mask1ZoomFlowTest(unittest.TestCase):
    def test_mask_editor_avoids_double_applying_disp_info_for_touch_coords(self):
        source = (ROOT / "widgets" / "mask_editor.py").read_text(encoding="utf-8")
        method = source.split("def _window_to_mask_coords", 1)[1].split("def on_touch_down", 1)[0]

        self.assertIn("params.window_to_tcg(wx, wy, self, self.texture_size", method)
        self.assertIn("apply_disp_info=False", method)
        self.assertNotIn("/ self.tcg_info['disp_info'][4]", method)

    def test_mask_editor_refreshes_preview_texture_size(self):
        source = (ROOT / "widgets" / "mask_editor.py").read_text(encoding="utf-8")

        self.assertIn("def _sync_texture_size", source)
        self.assertGreaterEqual(source.count("self._sync_texture_size()"), 3)

    def test_mask_editor_reuses_overlay_texture_and_coalesces_updates(self):
        source = (ROOT / "widgets" / "mask_editor.py").read_text(encoding="utf-8")

        self.assertIn("self._update_canvas_event = None", source)
        self.assertIn("if self._update_canvas_event is None", source)
        self.assertIn("def _ensure_canvas_texture", source)
        self.assertIn("self._canvas_texture", source)
        self.assertIn("colorfmt='luminance_alpha'", source)
        self.assertIn("self._la_buffer = bytearray", source)

    def test_mask_editor_uses_original_image_size_for_temporary_mask(self):
        source = (ROOT / "widgets" / "mask_editor.py").read_text(encoding="utf-8")

        self.assertIn("param.get('original_img_size', param.get('img_size'", source)

    def test_inpaint_mask_mode_enters_and_exits_full_preview(self):
        effects_source = (ROOT / "effects.py").read_text(encoding="utf-8")
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn("enter_mask1_full_preview_mode('inpaint')", effects_source)
        self.assertIn("exit_mask1_full_preview_mode('inpaint')", effects_source)
        self.assertIn("def enter_mask1_full_preview_mode", main_source)
        self.assertIn("def exit_mask1_full_preview_mode", main_source)
        self.assertIn("def _finish_ai_inpaint_mask_mode", main_source)
        self.assertIn("def _restore_mask1_view_after_submit", main_source)
        self.assertIn("_mask1_restore_view_after_submit", main_source)
        self.assertIn("core.get_initial_disp_info(width, height, scale)", main_source)

    def test_inpaint_mask_mode_temporarily_bypasses_geometry_params(self):
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn("_MASK1_GEOMETRY_BYPASS_KEYS", main_source)
        self.assertIn("'rotation2'", main_source)
        self.assertIn("'control_points'", main_source)
        self.assertIn("'matrix'", main_source)
        self.assertIn("'switch_distortion'", main_source)
        self.assertIn("core.get_initial_crop_rect(width, height)", main_source)
        self.assertIn("self.primary_param['matrix'] = np.eye(3)", main_source)
        self.assertIn("self._restore_mask1_geometry_params", main_source)
        self.assertIn("effects.reeffect_all(self.primary_effects, 0)", main_source)

    def test_mask1_mode_is_cancelled_on_image_and_tab_switch_and_is_exclusive(self):
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn("def _cancel_mask1_mode", main_source)
        self.assertIn("def _apply_mask1_exclusive_buttons", main_source)
        self.assertIn("self._cancel_mask1_mode(redraw=False)", main_source)
        on_select = main_source.split("def on_select(self, card):", 1)[1].split("@kvmainthread", 1)[0]
        self.assertLess(
            on_select.index("self._cancel_mask1_mode(redraw=False)"),
            on_select.index("self.save_current_sidecar()"),
        )
        self.assertIn("self._apply_mask1_exclusive_buttons(e)", main_source)
        self.assertIn("self.ids['switch_patchmatch_inpaint'].state = \"normal\"", main_source)
        self.assertIn("self.ids['switch_inpaint'].state = \"normal\"", main_source)

    def test_mask_editor_does_not_draw_while_space_panning(self):
        source = (ROOT / "widgets" / "mask_editor.py").read_text(encoding="utf-8")

        self.assertIn("def _is_space_panning", source)
        self.assertIn("return super(MaskEditor, self).on_touch_down(touch)", source)
        self.assertIn("return super(MaskEditor, self).on_touch_move(touch)", source)
        self.assertIn("self._hide_cursor()", source)


if __name__ == "__main__":
    unittest.main()
