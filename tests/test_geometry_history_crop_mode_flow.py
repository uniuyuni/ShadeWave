import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
PIPELINE_PATH = PROJECT_ROOT / "pipeline.py"
MAIN_PATH_TEXT = MAIN_PATH.read_text()
CORE_PATH = PROJECT_ROOT / "cores" / "core.py"
PARAMS_PATH = PROJECT_ROOT / "params.py"


def _load_function(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class GeometryHistoryCropModeFlowTest(unittest.TestCase):
    def test_history_redraw_syncs_crop_mode_before_rebuilding_crop_image(self):
        for function_name in ("_undo", "_redo", "_on_history_selected"):
            node = _load_function(MAIN_PATH, function_name)
            call_names = [
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            ]

            self.assertIn("_sync_editor_modes_after_history", call_names)
            self.assertIn("start_draw_image_and_crop", call_names)

        source = MAIN_PATH.read_text()
        self.assertIn(
            "self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)",
            source,
        )

    def test_crop_editor_mode_sync_does_not_write_crop_enable(self):
        sync = _load_class_function(EFFECTS_PATH, "CropEffect", "sync_crop_editor_mode_from_widget")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), sync)

        self.assertIn('widget.ids["effects"].current_tab.text == "Ge"', source)
        self.assertIn("self._open_crop_editor(param, widget)", source)
        self.assertIn("self._close_crop_editor(param, widget)", source)
        self.assertIn("self.sync_crop_editor_from_param(param)", source)
        self.assertNotIn("crop_enable", source)

    def test_crop_enable_is_not_saved_by_geometry_crop_or_vignette_effects(self):
        for class_name in ("GeometryEffect", "CropEffect", "VignetteEffect"):
            get_param_dict = _load_class_function(EFFECTS_PATH, class_name, "get_param_dict")
            source = ast.get_source_segment(EFFECTS_PATH.read_text(), get_param_dict)
            self.assertNotIn("crop_enable", source)

    def test_preview_crop_editing_is_runtime_pipeline_state(self):
        process_pipeline = _load_function(PIPELINE_PATH, "process_pipeline")
        source = ast.get_source_segment(PIPELINE_PATH.read_text(), process_pipeline)

        self.assertIn('efconfig.current_tab = current_tab', source)
        self.assertIn('efconfig.crop_editing = current_tab == "Ge"', source)

    def test_export_never_uses_geometry_editing_mode(self):
        export_pipeline = _load_function(PIPELINE_PATH, "export_pipeline")
        source = ast.get_source_segment(PIPELINE_PATH.read_text(), export_pipeline)

        self.assertIn("efconfig.crop_editing = False", source)

    def test_crop_dependent_effects_read_runtime_crop_editing(self):
        for class_name in ("GeometryEffect", "CropEffect", "VignetteEffect"):
            make_diff = _load_class_function(EFFECTS_PATH, class_name, "make_diff")
            source = ast.get_source_segment(EFFECTS_PATH.read_text(), make_diff)
            self.assertIn("getattr(efconfig, 'crop_editing', False)", source)
            self.assertNotIn("'crop_enable'", source)

    def test_zero_wrap_uses_runtime_crop_editing_not_param_crop_enable(self):
        apply_zero_wrap = _load_function(CORE_PATH, "apply_zero_wrap")
        source = ast.get_source_segment(CORE_PATH.read_text(), apply_zero_wrap)

        self.assertIn("crop_editing=False", source)
        self.assertIn("if not crop_editing:", source)
        self.assertNotIn("crop_enable", source)
        self.assertIn("crop_editing = self.ids[\"effects\"].current_tab.text == \"Ge\"", MAIN_PATH_TEXT)
        self.assertIn("crop_editing=crop_editing", MAIN_PATH_TEXT)

    def test_crop_enable_is_not_copied_into_history_runtime_special(self):
        source = PARAMS_PATH.read_text()

        self.assertIn("'crop_enable'", source)
        self.assertIn("DO_NOT_COPY_SPECIAL_PARAM", source)
        self.assertIn("if key in DO_NOT_COPY_SPECIAL_PARAM:", source)

    def test_geometry_history_captures_crop_state_changed_by_rotation_redraw(self):
        get_param_dict = _load_class_function(EFFECTS_PATH, "GeometryEffect", "get_param_dict")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), get_param_dict)

        self.assertIn("'crop_rect':", source)
        self.assertIn("'disp_info':", source)


if __name__ == "__main__":
    unittest.main()
