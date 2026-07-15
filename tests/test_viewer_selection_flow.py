import ast
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils import viewer_query


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


def _load_viewer_method(function_name):
    namespace = {}
    exec(
        _load_class_function("ViewerWidget", function_name),
        {"viewer_query": viewer_query},
        namespace,
    )
    return namespace[function_name]


class ViewerSelectionFlowTest(unittest.TestCase):
    def test_reselecting_single_selected_card_does_not_notify_reload(self):
        source = _load_class_function("ViewerWidget", "handle_selection")

        self.assertIn("already_single_selected = (", source)
        self.assertIn("index in self.selected_indices", source)
        self.assertIn("len(self.selected_indices) == 1", source)
        self.assertIn("if already_single_selected:", source)
        self.assertIn("self._set_last_selected(index)", source)
        self.assertIn("return", source)
        plain_click_branch = source.split("if 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers:", 1)[1]
        self.assertLess(
            plain_click_branch.index("if already_single_selected:"),
            plain_click_branch.index("self.clear_selection(notify=False)"),
        )

    def test_modifier_selection_keeps_current_until_it_is_deselected(self):
        handle_source = _load_class_function("ViewerWidget", "handle_selection")
        reconcile_source = _load_class_function("ViewerWidget", "_reconcile_current_selection")

        self.assertIn("current_index = self._current_view_index()", handle_source)
        self.assertIn("self._reconcile_current_selection(reference_index)", handle_source)
        self.assertIn("self._current_path in self.selected_paths", reconcile_source)
        self.assertIn("viewer_query.nearest_selected_index", reconcile_source)

    def test_nearest_remaining_selection_is_used_for_current(self):
        self.assertEqual(4, viewer_query.nearest_selected_index({1, 4, 9}, 5))
        self.assertEqual(9, viewer_query.nearest_selected_index({1, 4, 9}, 8))
        self.assertEqual(4, viewer_query.nearest_selected_index({4, 6}, 5))
        self.assertIsNone(viewer_query.nearest_selected_index(set(), 5))

    def test_reconcile_current_selection_behavior(self):
        reconcile = _load_viewer_method("_reconcile_current_selection")

        class StubViewer:
            def __init__(self, current_path, selected_paths, selected_indices):
                self._current_path = current_path
                self.selected_paths = set(selected_paths)
                self.selected_indices = set(selected_indices)
                self.notifications = []

            def notify_selection_change(self, index):
                self.notifications.append(index)

        retained = StubViewer("current", {"current", "other"}, {2, 6})
        reconcile(retained, 2)
        self.assertEqual([], retained.notifications)

        replaced = StubViewer("current", {"near", "far"}, {4, 9})
        reconcile(replaced, 5)
        self.assertEqual([4], replaced.notifications)

        emptied = StubViewer("current", set(), set())
        reconcile(emptied, 5)
        self.assertEqual([None], emptied.notifications)

        initially_empty = StubViewer(None, set(), set())
        reconcile(initially_empty, None)
        self.assertEqual([], initially_empty.notifications)

    def test_empty_selection_notifies_main_widget_with_none(self):
        reconcile_source = _load_class_function("ViewerWidget", "_reconcile_current_selection")
        notify_source = _load_class_function("ViewerWidget", "notify_selection_change")
        clear_source = _load_class_function("ViewerWidget", "clear_selection")

        self.assertIn("self.notify_selection_change(None)", reconcile_source)
        self.assertIn("app.main_widget.on_select(None)", notify_source)
        self.assertIn("self._reconcile_current_selection(current_index)", clear_source)

    def test_select_all_does_not_replace_an_existing_current_image(self):
        source = _load_class_function("ViewerWidget", "on_key_down")

        self.assertIn("current_index = self._current_view_index()", source)
        self.assertIn("self.clear_selection(notify=False)", source)
        self.assertIn("self._reconcile_current_selection(current_index)", source)
        self.assertNotIn("self.notify_selection_change(len(self.data)-1)", source)

    def test_ai_job_state_never_initializes_recycle_data_as_none(self):
        new_item_source = _load_class_function("ViewerWidget", "_new_image_item")
        set_path_source = _load_class_function("ViewerWidget", "set_path")
        set_state_source = _load_class_function("ViewerWidget", "set_ai_job_state_for_path")

        self.assertIn("'ai_job_state': \"\"", new_item_source)
        # set_path は _new_image_item 経由で item dict を生成する（初期値の一元化）。
        self.assertIn("self._add_item_if_missing(file_path)", set_path_source)
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

        self.assertIn("self.content_box = KVBoxLayout(orientation='vertical')", init_source)
        self.assertIn("self._sync_content_box_layout_metrics()", init_source)
        self.assertIn("self._thumbnail_geometry_event = None", init_source)
        self.assertIn("self._thumbnail_geometry_late_event = None", init_source)
        self.assertIn("KVClock.schedule_once(", schedule_source)
        self.assertIn("self._refresh_thumbnail_geometry, 0.05", schedule_source)
        self.assertIn("def _sync_content_box_layout_metrics", _load_class_function("ThumbnailCard", "_sync_content_box_layout_metrics"))
        self.assertIn("self.content_box.padding = kvutils.dpi_scale_width(self.content_box.ref_layout_padding)", _load_class_function("ThumbnailCard", "_sync_content_box_layout_metrics"))
        self.assertIn("self._sync_content_box_layout_metrics()", refresh_source)
        self.assertIn("self.do_layout()", refresh_source)
        self.assertIn("self.content_box.do_layout()", refresh_source)
        self.assertIn("self.image_box.do_layout()", refresh_source)
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

    def test_thumbnail_texture_buffer_format_matches_uploaded_uint8_data(self):
        source = _load_class_function("ThumbnailCard", "on_thumb_source")

        # thumb_source は float32[0,1] だが、GPU 転送は ubyte に変換してから行う
        # （float だと 1px あたり4倍のデータ量になり転送が無駄に重くなるため）。
        self.assertIn("astype(np.uint8)", source)
        self.assertIn("bufferfmt='ubyte'", source)
        self.assertNotIn("bufferfmt='float'", source)
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
        self.assertIn("self.bg_color = [0.32, 0.32, 0.32, 1] if value else [0.1, 0.1, 0.1, 1]", selected_source)
        self.assertNotIn("MDCard", source)
        self.assertNotIn("md_bg_color", source)


if __name__ == "__main__":
    unittest.main()
