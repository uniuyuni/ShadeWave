import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DISTORTION_PAINTER_PATH = PROJECT_ROOT / "widgets" / "distortion_painter.py"
MAIN_PATH = PROJECT_ROOT / "main.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class DistortionPainterScrollFlowTest(unittest.TestCase):
    def test_scroll_changes_brush_size_before_recording_stroke(self):
        source_text = DISTORTION_PAINTER_PATH.read_text()
        node = _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_down")
        source = ast.get_source_segment(source_text, node)

        self.assertIn("touch.is_mouse_scrolling", source)
        self.assertIn("self._adjust_brush_size_from_scroll(touch)", source)
        self.assertLess(source.index("touch.is_mouse_scrolling"), source.index("self.recorded.append(record)"))

    def test_scroll_does_not_close_distortion_history(self):
        source_text = DISTORTION_PAINTER_PATH.read_text()
        move = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_move"),
        )
        up = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_up"),
        )

        self.assertIn("self._paint_touch_uid != touch.uid", move)
        self.assertIn("self._paint_touch_uid != touch.uid", up)
        self.assertIn("self.callback('end', self)", up)

    def test_brush_size_callback_updates_ui_without_history(self):
        source_text = MAIN_PATH.read_text()
        node = _load_class_function(MAIN_PATH, "MainWidget", "distortion_callback")
        source = ast.get_source_segment(source_text, node)

        self.assertIn("case 'brush_size'", source)
        self.assertIn("self.primary_param['distortion_brush_size'] = widget.brush_size", source)
        self.assertIn('self.ids["slider_distortion_brush_size"].set_slider_value(widget.brush_size)', source)

    def test_distortion_preview_touch_clears_text_focus(self):
        painter_source_text = DISTORTION_PAINTER_PATH.read_text()
        painter_source = ast.get_source_segment(
            painter_source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_down"),
        )
        main_source_text = MAIN_PATH.read_text()
        callback_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "distortion_callback"),
        )
        preview_touch_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "on_image_touch_down"),
        )

        self.assertIn("self.callback('focus', self)", painter_source)
        self.assertLess(painter_source.index("self.callback('focus', self)"), painter_source.index("touch.is_mouse_scrolling"))
        self.assertIn("case 'focus'", callback_source)
        self.assertIn("self._clear_text_input_focus()", callback_source)
        self.assertIn("self._clear_text_input_focus()", preview_touch_source)

    def test_liquify_painter_resyncs_on_preview_geometry_change(self):
        source_text = EFFECTS_PATH.read_text()
        make_key_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_make_painter_ref_key"),
        )
        sync_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_sync_distortion_painter_ref"),
        )
        make_diff_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "make_diff"),
        )

        self.assertIn("params.get_disp_info(param)", make_key_source)
        self.assertIn("getattr(efconfig, 'upstream_hash', None)", make_key_source)
        self.assertIn("ref_key == self._painter_ref_key", sync_source)
        self.assertIn("self.distortion_painter.set_ref_image(img, True)", sync_source)
        self.assertIn("self._sync_distortion_painter_ref(img, param, efconfig)", make_diff_source)
        self.assertIn("self._painter_ref_key", make_diff_source)

    def test_liquify_reset_button_calls_painter_directly(self):
        kv_source = MAIN_KV_PATH.read_text()
        main_source_text = MAIN_PATH.read_text()
        reset_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "reset_distortion_painter_action"),
        )
        set2param_source = ast.get_source_segment(
            EFFECTS_PATH.read_text(),
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "set2param"),
        )

        self.assertIn("id: button_distortion_reset", kv_source)
        self.assertIn("on_press: root.reset_distortion_painter_action()", kv_source)
        self.assertIn("painter.reset_image()", reset_source)
        self.assertNotIn('button_distortion_reset"].state == "down"', set2param_source)


if __name__ == "__main__":
    unittest.main()
