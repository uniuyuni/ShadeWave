import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER_PATH = PROJECT_ROOT / "widgets" / "viewer.py"


def _load_class_function(class_name, function_name):
    source = VIEWER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class ViewerSelectionFlowTest(unittest.TestCase):
    def test_reselecting_single_selected_card_does_not_notify_reload(self):
        source = _load_class_function("ViewerWidget", "handle_selection")

        self.assertIn("already_single_selected = (", source)
        self.assertIn("index in self.selected_indices", source)
        self.assertIn("len(self.selected_indices) == 1", source)
        self.assertIn("if already_single_selected:", source)
        self.assertIn("self.last_selected_index = index", source)
        self.assertIn("return", source)
        plain_click_branch = source.split("if 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers:", 1)[1]
        self.assertLess(
            plain_click_branch.index("if already_single_selected:"),
            plain_click_branch.index("self.clear_selection()"),
        )

    def test_ai_job_state_never_initializes_recycle_data_as_none(self):
        new_item_source = _load_class_function("ViewerWidget", "_new_image_item")
        set_path_source = _load_class_function("ViewerWidget", "set_path")
        set_state_source = _load_class_function("ViewerWidget", "set_ai_job_state_for_path")

        self.assertIn("'ai_job_state': \"\"", new_item_source)
        self.assertIn("'ai_job_state': \"\"", set_path_source)
        self.assertIn('else ""', set_state_source)
        self.assertNotIn("'ai_job_state': None", new_item_source)
        self.assertNotIn("'ai_job_state': None", set_path_source)

    def test_ai_job_indicator_uses_loading_spinner_size(self):
        source = _load_class_function("ThumbnailCard", "__init__")
        ai_icon_block = source.split("self.ai_job_icon = KVImage(", 1)[1].split("self.image_box.add_widget(self.ai_job_icon)", 1)[0]

        self.assertIn("source=rel(\"assets\", \"spinner.gif\")", ai_icon_block)
        self.assertIn("size_hint=(1, 1)", ai_icon_block)
        self.assertIn("pos_hint={\"x\": 0, \"y\": 0}", ai_icon_block)
        self.assertNotIn("_PMCK_ICON_REF_SIZE", ai_icon_block)

    def test_thumbnail_image_layout_is_reset_for_recycled_cards(self):
        init_source = _load_class_function("ThumbnailCard", "__init__")
        configure_source = _load_class_function("ThumbnailCard", "_configure_thumbnail_image_widget")
        refresh_source = _load_class_function("ThumbnailCard", "refresh_view_attrs")
        thumb_source = _load_class_function("ThumbnailCard", "on_thumb_source")

        self.assertIn("self.loading_spinner = KVImage(", init_source)
        self.assertIn("fit_mode=\"scale-down\"", init_source)
        self.assertIn("self.image = ThumbnailImage(", init_source)
        self.assertIn("max_display_side=_THUMBNAIL_DISPLAY_MAX_SIDE", init_source)
        self.assertIn("self.loading_spinner, self.image", configure_source)
        self.assertIn("widget.size_hint = (1, 1)", configure_source)
        self.assertIn("widget.pos_hint = {\"x\": 0, \"y\": 0}", configure_source)
        self.assertIn("widget.allow_stretch = False", configure_source)
        self.assertIn("widget.keep_ratio = True", configure_source)
        self.assertIn("widget.fit_mode = \"scale-down\"", configure_source)
        self.assertIn("self.image.max_display_side = _THUMBNAIL_DISPLAY_MAX_SIDE", configure_source)
        self.assertIn("self._configure_thumbnail_image_widget()", refresh_source)
        self.assertIn("self._configure_thumbnail_image_widget()", thumb_source)
        self.assertIn("self._schedule_thumbnail_geometry_refresh()", refresh_source)
        self.assertIn("self._schedule_thumbnail_geometry_refresh()", thumb_source)

    def test_thumbnail_geometry_refresh_is_deferred_after_recycle_layout(self):
        init_source = _load_class_function("ThumbnailCard", "__init__")
        schedule_source = _load_class_function("ThumbnailCard", "_schedule_thumbnail_geometry_refresh")
        refresh_source = _load_class_function("ThumbnailCard", "_refresh_thumbnail_geometry")
        layout_source = _load_class_function("ThumbnailCard", "refresh_view_layout")

        self.assertIn("self._thumbnail_geometry_event = None", init_source)
        self.assertIn("self._thumbnail_geometry_late_event = None", init_source)
        self.assertIn("KVClock.schedule_once(", schedule_source)
        self.assertIn("self._refresh_thumbnail_geometry, 0.05", schedule_source)
        self.assertIn("self.image._update_rect()", refresh_source)
        self.assertIn("self._update_pmck_icon_layout()", refresh_source)
        self.assertIn("self._schedule_thumbnail_geometry_refresh()", layout_source)

    def test_thumbnail_image_draws_texture_with_own_capped_scale_down_rect(self):
        source = _load_class_function("ThumbnailImage", "_update_rect")
        init_source = _load_class_function("ThumbnailImage", "__init__")
        layout_source = _load_class_function("ThumbnailCard", "refresh_view_layout")
        viewer_source = VIEWER_PATH.read_text(encoding="utf-8")

        self.assertIn("_THUMBNAIL_DISPLAY_MAX_SIDE = 240", viewer_source)
        self.assertIn("max_display_side = KVNumericProperty(_THUMBNAIL_DISPLAY_MAX_SIDE)", viewer_source)
        self.assertIn("max_display_side=self._update_rect", init_source)
        self.assertIn("scale = min(1.0, self.width / tex_w, self.height / tex_h)", source)
        self.assertIn("self.max_display_side / max(tex_w, tex_h)", source)
        self.assertIn("self._rect.texture = texture", source)
        self.assertIn("self._rect.size = (draw_w, draw_h)", source)
        self.assertIn("self.x + (self.width - draw_w) / 2", source)
        self.assertIn("self.y + (self.height - draw_h) / 2", source)
        self.assertIn("self.norm_image_size = [draw_w, draw_h]", source)
        self.assertIn("self.image._update_rect()", layout_source)
        self.assertIn("self._update_pmck_icon_layout()", layout_source)

    def test_thumbnail_resize_never_upscales_small_embedded_previews(self):
        source = _load_class_function("ViewerWidget", "_calc_resize_image")

        self.assertIn("scale_factor = min(1.0, max_length / max(width, height))", source)
        self.assertIn("max(1, int(round(width * scale_factor)))", source)
        self.assertIn("max(1, int(round(height * scale_factor)))", source)
        self.assertNotIn("scale_factor = max_length / width", source)
        self.assertNotIn("scale_factor = max_length / height", source)

    def test_thumbnail_texture_buffer_format_matches_uploaded_float_data(self):
        source = _load_class_function("ThumbnailCard", "on_thumb_source")

        self.assertIn("bufferfmt='float'", source)
        self.assertNotIn("bufferfmt='ushort'", source)
        self.assertIn("self.texture = None", source)
        self.assertNotIn("self.image.source = ''", source)
        self.assertIn("self.loading_spinner.opacity = 1.0", source)
        self.assertIn("self.loading_spinner.opacity = 0.0", source)

    def test_thumbnail_card_uses_plain_card_background(self):
        source = VIEWER_PATH.read_text(encoding="utf-8")
        init_source = _load_class_function("ThumbnailCard", "__init__")
        selected_source = _load_class_function("ThumbnailCard", "on_selected")

        self.assertIn("from widgets.plain_card import PlainCard", source)
        self.assertIn("class ThumbnailCard(RecycleDataViewBehavior, PlainCard):", source)
        self.assertIn("self.bg_color = [0.1, 0.1, 0.1, 1]", init_source)
        self.assertIn("self.shadow_color = [0, 0, 0, 0.5]", init_source)
        self.assertIn("self.shadow_offset = [0, -3]", init_source)
        self.assertIn("self.shadow_spread = [2, 2]", init_source)
        self.assertIn("self.bg_color = [0.8, 0.8, 0.8, 1] if value else [0.1, 0.1, 0.1, 1]", selected_source)
        self.assertNotIn("MDCard", source)
        self.assertNotIn("md_bg_color", source)


if __name__ == "__main__":
    unittest.main()
