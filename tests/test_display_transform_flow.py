import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_function(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


class DisplayTransformFlowTest(unittest.TestCase):
    def test_settled_preview_uses_colour_functions_adapter(self):
        draw_image_core = _load_function(MAIN_PATH, "draw_image_core")
        source = ast.get_source_segment(MAIN_PATH.read_text(), draw_image_core)

        self.assertIn("_fast_display_color_transform(img, src_space, dst_space, cat)", source)
        self.assertIn("colour_functions.display_color_transform(img, src_space, dst_space, cat)", source)
        self.assertNotIn("apply_gamut_mapping=True", source)

    def test_forced_full_preview_can_allow_stale_display_frames(self):
        draw_image_core = _load_function(MAIN_PATH, "draw_image_core")
        source = ast.get_source_segment(MAIN_PATH.read_text(), draw_image_core)

        self.assertIn("force_full_preview_render = pipeline.preview_full_render_enabled(current_tab)", source)
        self.assertIn("full_preview_allow_stale = force_full_preview_render and pipeline.preview_allow_stale_enabled(current_tab)", source)
        self.assertIn("effective_allow_stale = effective_fast_display or full_preview_allow_stale", source)
        self.assertIn("if stale_frame and not effective_allow_stale:", source)
        self.assertIn("allow_stale=effective_allow_stale", source)

    def test_forced_full_preview_drains_all_versions_only_when_requested(self):
        draw_image = _load_function(MAIN_PATH, "draw_image")
        source = ast.get_source_segment(MAIN_PATH.read_text(), draw_image)

        self.assertIn("full_preview_render = pipeline.preview_full_render_enabled(current_tab)", source)
        self.assertIn("drain_all_preview = pipeline.preview_drain_all_enabled(current_tab)", source)
        self.assertIn("full_preview_render and drain_all_preview and last_processed_version >= 0", source)
        # 積み残しラグに上限を設け、超えたら最新版へ追いつく(ドラッグ停止後の UI 固着を防ぐ)。
        self.assertIn("current_version - last_processed_version <= _PREVIEW_DRAIN_MAX_LAG", source)
        self.assertIn("frame_version_override=target_version", source)

    def test_geometry_interaction_invalidates_crop_cache_for_preview_interpolation_change(self):
        geometry_callback = _load_function(MAIN_PATH, "geometry_callback")
        source = ast.get_source_segment(MAIN_PATH.read_text(), geometry_callback)

        self.assertIn("self.crop_image = None", source)
        self.assertIn("self.start_draw_image(invalidate_crop=True)", source)

    def test_preview_overlay_sync_can_defer_to_blit_resize_path(self):
        draw_image_core = _load_function(MAIN_PATH, "draw_image_core")
        source = ast.get_source_segment(MAIN_PATH.read_text(), draw_image_core)
        blit_image = _load_function(MAIN_PATH, "blit_image")
        blit_source = ast.get_source_segment(MAIN_PATH.read_text(), blit_image)
        helper = _load_function(MAIN_PATH, "_preview_overlay_after_blit_enabled")
        helper_source = ast.get_source_segment(MAIN_PATH.read_text(), helper)

        self.assertIn('PLATYPUS_PREVIEW_OVERLAY_AFTER_BLIT", "0"', helper_source)
        self.assertIn("overlay_after_blit = self._preview_overlay_after_blit_enabled()", source)
        self.assertIn("if not overlay_after_blit:", source)
        self.assertLess(
            source.index("if not overlay_after_blit:"),
            source.index("pipeline.process_pipeline("),
        )
        self.assertIn("self.resize()", blit_source)


if __name__ == "__main__":
    unittest.main()
