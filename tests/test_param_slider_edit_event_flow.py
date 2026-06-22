import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAM_SLIDER_PATH = PROJECT_ROOT / "widgets" / "param_slider.py"
PARAM_SLIDER_KV_PATH = PROJECT_ROOT / "widgets" / "param_slider.kv"
MULTI_SLIDER_PATH = PROJECT_ROOT / "widgets" / "multi_slider.py"
FLOAT_INPUT_PATH = PROJECT_ROOT / "widgets" / "float_input.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


def _load_class(path, class_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} was not found")


class ParamSliderEditEventFlowTest(unittest.TestCase):
    def test_edit_events_are_counters_not_value_assignments(self):
        source = PARAM_SLIDER_PATH.read_text()

        self.assertIn("before_edit = KVNumericProperty(0)", source)
        self.assertIn("after_edit = KVNumericProperty(0)", source)
        self.assertIn("def _notify_before_edit", source)
        self.assertIn("self.before_edit += 1", source)
        self.assertIn("def _notify_after_edit", source)
        self.assertIn("self.after_edit += 1", source)

    def test_slider_touch_down_always_emits_before_edit(self):
        on_slider_touch_down = _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "on_slider_touch_down")
        source = ast.get_source_segment(PARAM_SLIDER_PATH.read_text(), on_slider_touch_down)

        self.assertIn("self._notify_before_edit()", source)
        self.assertNotIn("self.before_edit = self.value", source)

    def test_param_slider_label_double_tap_resets_slider(self):
        source = PARAM_SLIDER_PATH.read_text()
        kv_source = PARAM_SLIDER_KV_PATH.read_text()
        label_source = ast.get_source_segment(
            source,
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "on_label_touch_down"),
        )
        reset_source = ast.get_source_segment(
            source,
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "_reset_slider_to_default"),
        )
        set_reset_source = ast.get_source_segment(
            source,
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "set_slider_reset"),
        )

        self.assertIn("on_touch_down: root.on_label_touch_down(args[1])", kv_source)
        self.assertIn("if not touch.is_double_tap:", label_source)
        self.assertIn("self._reset_slider_to_default()", label_source)
        self.assertIn("self._notify_before_edit()", reset_source)
        self.assertIn("slider.values = list(self.reset_values)", reset_source)
        self.assertIn("self._set_slider_value_at(self._active_slider_index(), self.reset_value)", reset_source)
        self.assertIn("self._notify_after_edit()", reset_source)
        self.assertIn("self.reset_values = list(value)", set_reset_source)
        self.assertIn("self.reset_value = self.reset_values[0]", set_reset_source)

    def test_multi_slider_starts_edit_before_touch_value_update(self):
        source_text = MULTI_SLIDER_PATH.read_text()
        on_touch_down = _load_class_function(MULTI_SLIDER_PATH, "MultiSlider", "on_touch_down")
        source = ast.get_source_segment(source_text, on_touch_down)

        self.assertIn("interaction_start_callback", source)
        self.assertLess(
            source.index("self.interaction_start_callback()"),
            source.index("self._update_value_from_touch_x(touch.x)"),
        )

    def test_param_slider_wires_multi_slider_interaction_callbacks(self):
        kv_source = PARAM_SLIDER_KV_PATH.read_text()
        before_source = ast.get_source_segment(
            PARAM_SLIDER_PATH.read_text(),
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "_notify_before_edit"),
        )

        self.assertIn("interaction_start_callback: root.on_slider_interaction_start", kv_source)
        self.assertIn("interaction_end_callback: root.on_slider_interaction_end", kv_source)
        self.assertIn("if self._editing:", before_source)

    def test_multi_slider_tracks_active_thumb_for_multi_point_editing(self):
        source = MULTI_SLIDER_PATH.read_text()

        self.assertIn("active_index = NumericProperty(0)", source)
        self.assertIn("active_thumb_color = ColorProperty([0.72, 0.30, 0.28, 1])", source)
        self.assertIn("self.active_index = closest_idx", source)
        self.assertIn("def _closest_thumb_index_for_x", source)
        self.assertNotIn("hit_padding_y = NumericProperty(0)", source)
        self.assertNotIn("def _collide_touch_area", source)
        self.assertNotIn("def _touch_near_thumb", source)
        self.assertNotIn("def _point_in_extended_touch_area", source)
        self.assertNotIn("def collide_point(self, x, y):", source)
        self.assertIn("touch.grab(self)", source)
        self.assertIn("touch.ungrab(self)", source)
        self.assertIn("elif len(self.values) > 1 and i == self.active_index:", source)
        self.assertIn("c_thumb = self.active_thumb_color", source)

    def test_param_slider_multi_point_controls_edit_values_not_scalar_value(self):
        input_class_source = ast.get_source_segment(
            PARAM_SLIDER_PATH.read_text(),
            _load_class(PARAM_SLIDER_PATH, "ParamFloatInput"),
        )
        input_source = ast.get_source_segment(
            PARAM_SLIDER_PATH.read_text(),
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "on_input_text_validate"),
        )
        button_source = ast.get_source_segment(
            PARAM_SLIDER_PATH.read_text(),
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "on_button_press"),
        )
        scrub_source = ast.get_source_segment(
            PARAM_SLIDER_PATH.read_text(),
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "_apply_input_scrub_step"),
        )
        kv_source = PARAM_SLIDER_KV_PATH.read_text()

        self.assertIn("multi_value_edit_mode = KVStringProperty(\"active\")", PARAM_SLIDER_PATH.read_text())
        self.assertIn("def _scrub_owner(self):", input_class_source)
        self.assertIn("parent = getattr(parent, 'parent', None)", input_class_source)
        self.assertIn("owner.on_input_scrub_pixels(dx)", input_class_source)
        self.assertIn("self._set_slider_value_at(self._active_slider_index(), val)", input_source)
        self.assertIn("self._set_slider_value_at(self._active_slider_index(), self._active_slider_value() + step)", button_source)
        self.assertIn("self._set_slider_value_at(self._active_slider_index(), self._active_slider_value() + delta)", scrub_source)
        self.assertNotIn("self.ids['slider'].value = self.value", input_source)
        self.assertIn("on_values: root.on_multi_slider_values()", kv_source)
        self.assertIn("on_active_index: root.on_slider_active_index()", kv_source)
        self.assertIn("allow_overlap: root.allow_overlap", kv_source)
        self.assertNotIn("hit_padding_y:", kv_source)
        self.assertIn("allow_overlap = KVBooleanProperty(False)", PARAM_SLIDER_PATH.read_text())
        self.assertIn("self.ids['slider'].allow_overlap = self.allow_overlap", PARAM_SLIDER_PATH.read_text())
        self.assertIn("def on_slider_values(self, *args):", PARAM_SLIDER_PATH.read_text())
        self.assertIn("def on_multi_slider_values(self):", PARAM_SLIDER_PATH.read_text())

    def test_param_slider_does_not_extend_touch_area_outside_own_row(self):
        source = PARAM_SLIDER_PATH.read_text()

        self.assertNotIn("def _slider_accepts_touch", source)
        self.assertNotIn("def _slider_prefers_touch", source)
        self.assertNotIn("def _slider_drag_active", source)
        self.assertNotIn('getattr(slider, "_collide_touch_area", None)', source)
        self.assertNotIn("return bool(slider.collide_point(x, y))", source)
        self.assertNotIn("return self.ids['slider'].on_touch_down(touch)", source)
        self.assertNotIn("return self.ids['slider'].on_touch_move(touch)", source)
        self.assertNotIn("return self.ids['slider'].on_touch_up(touch)", source)

    def test_float_input_allows_leading_minus_for_signed_sliders(self):
        source = FLOAT_INPUT_PATH.read_text()

        self.assertIn("if '-' in substring and old_cursor_pos == 0", source)
        self.assertIn("not self._internal_value.startswith('-')", source)
        self.assertIn("s = sign + s", source)

    def test_split_mode_places_up_to_three_value_boxes_under_slider(self):
        source = PARAM_SLIDER_KV_PATH.read_text()
        param_source = PARAM_SLIDER_PATH.read_text()
        set_slider_value_source = ast.get_source_segment(
            PARAM_SLIDER_PATH.read_text(),
            _load_class_function(PARAM_SLIDER_PATH, "ParamSlider", "set_slider_value"),
        )

        self.assertIn("ref_height: 48 if root.show_multi_value_boxes else 24", source)
        self.assertIn("height: kvutils.dpi_scale_height(48) if root.show_multi_value_boxes else kvutils.dpi_scale_height(24)", source)
        self.assertIn("height: kvutils.dpi_scale_height(24)", source)
        self.assertIn("height: kvutils.dpi_scale_height(24) if root.show_multi_value_boxes else 0", source)
        self.assertIn("ref_width: 40 if root.show_right_value_controls else 0", source)
        self.assertIn("width: kvutils.dpi_scale_width(40) if root.show_right_value_controls else 0", source)
        self.assertIn("width: kvutils.dpi_scale_width(13) if root.show_right_value_controls else 0", source)
        self.assertIn("id: input_multi_0", source)
        self.assertIn("id: input_multi_1", source)
        self.assertIn("id: input_multi_2", source)
        self.assertIn("x: self.parent.x if self.parent else 0", source)
        self.assertIn("x: self.parent.center_x - self.width / 2 if self.parent else 0", source)
        self.assertIn("x: self.parent.right - self.width if self.parent else 0", source)
        self.assertNotIn("x: parent.", source)
        self.assertNotIn("center_y: parent.", source)
        self.assertIn("opacity: 1 if root.multi_point_count == 3 else 0", source)
        self.assertIn("opacity: 1 if root.multi_point_count >= 2 else 0", source)
        self.assertIn("def _visual_value_indices", param_source)
        self.assertIn("slider._get_x_from_value(values[i])", param_source)
        self.assertIn("return [visual[0], visual[count // 2], visual[-1]]", param_source)
        self.assertIn("def _multi_slot_value_index", param_source)
        self.assertIn("return visual[0]", param_source)
        self.assertIn("return visual[1]", param_source)
        self.assertIn("return visual[-1]", param_source)
        self.assertIn("values = list(value)", set_slider_value_source)
        self.assertNotIn("values = [min(self.max, max(self.min, v)) for v in value]", set_slider_value_source)


if __name__ == "__main__":
    unittest.main()
