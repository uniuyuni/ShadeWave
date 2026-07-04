import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class CropEditorSyncFlowTest(unittest.TestCase):
    def test_crop_set2widget_syncs_open_editor_from_param(self):
        set2widget = _load_class_function(EFFECTS_PATH, "CropEffect", "set2widget")
        call_names = [
            call.func.attr
            for call in ast.walk(set2widget)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        ]

        self.assertIn("set_text", call_names)
        self.assertIn("sync_crop_editor_from_param", call_names)

    def test_crop_editor_sync_updates_geometry_and_restores_param_rect_last(self):
        sync = _load_class_function(EFFECTS_PATH, "CropEffect", "sync_crop_editor_from_param")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), sync)

        for snippet in (
            "self.crop_editor.input_width = input_width",
            "self.crop_editor.input_height = input_height",
            "self.crop_editor.scale =",
            "self.crop_editor.input_angle =",
            "self.crop_editor.set_aspect_ratio(self._param_to_aspect_ratio(param))",
            "self.crop_editor.update_rect()",
            "self.crop_editor.update_centering()",
        ):
            self.assertIn(snippet, source)

        normal_sync = source.split("# set_aspect_ratio may resize the current editor rect; restore the saved param rect last.", 1)[1]
        first_rect = normal_sync.find("self.crop_editor.set_to_local_crop_rect(crop_rect)")
        aspect = normal_sync.find("self.crop_editor.set_aspect_ratio")
        last_rect = normal_sync.rfind("self.crop_editor.set_to_local_crop_rect(crop_rect)")
        self.assertLess(first_rect, aspect)
        self.assertLess(aspect, last_rect)

    def test_full_decode_resyncs_open_crop_editor_after_size_change(self):
        source = MAIN_PATH.read_text()
        self.assertIn("sync_crop_editor_mode_from_widget(self, self.primary_param)", source)

    def test_crop_callback_invalidates_crop_cache_and_redraws_like_geometry_callback(self):
        # crop_callback は geometry_callback/distortion_callback と違い、以前は
        # crop_rect を書き換えるだけで redraw を一切トリガーしていなかった。
        # そのため、ドラッグ中もドラッグ終了後も（タブ切替など別操作が起きるまで）
        # クロップ枠の移動が通常プレビューへ反映されなかった。
        crop_callback = _load_class_function(MAIN_PATH, "MainWidget", "crop_callback")
        source = ast.get_source_segment(MAIN_PATH.read_text(), crop_callback)

        self.assertIn("self.crop_image = None", source)
        self.assertIn("self.apply_effects_lv(0, 'crop', sync=True)", source)
        self.assertIn("self.start_draw_image(invalidate_crop=True)", source)


if __name__ == "__main__":
    unittest.main()
