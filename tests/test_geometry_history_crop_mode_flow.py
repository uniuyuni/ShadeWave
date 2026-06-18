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
MASK_EDITOR2_PATH = PROJECT_ROOT / "widgets" / "mask_editor2.py"


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


def _load_class(path, class_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} was not found")


def _node_source(path, node):
    return ast.get_source_segment(path.read_text(), node)


def _attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _assigned_sources(path, function_node, target_name):
    sources = []
    for node in ast.walk(function_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(_attribute_name(target) == target_name for target in targets):
            sources.append(_node_source(path, node.value))
    return sources


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
            class_node = _load_class(EFFECTS_PATH, class_name)
            source = ast.get_source_segment(EFFECTS_PATH.read_text(), class_node)
            self.assertNotIn("crop_enable", source)

    def test_preview_crop_editing_is_runtime_pipeline_state(self):
        config_state = _load_function(PIPELINE_PATH, "_configure_preview_effect_config")
        source = ast.get_source_segment(PIPELINE_PATH.read_text(), config_state)
        crop_editing_sources = _assigned_sources(PIPELINE_PATH, config_state, "efconfig.crop_editing")

        self.assertIn('efconfig.current_tab = current_tab', source)
        self.assertIn('is_geometry_tab = current_tab == "Ge"', source)
        self.assertTrue(any("is_geometry_tab" in value for value in crop_editing_sources))
        self.assertNotIn("crop_enable", "".join(crop_editing_sources))

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
        self.assertNotIn("count_nonzero", source)
        self.assertNotIn("crop_enable", source)
        draw_image_core = _load_function(MAIN_PATH, "draw_image_core")
        draw_source = ast.get_source_segment(MAIN_PATH.read_text(), draw_image_core)
        self.assertIn('crop_editing = current_tab == "Ge"', draw_source)
        self.assertIn("crop_editing=crop_editing", draw_source)

    def test_mask_overlay_is_clipped_to_zero_wrap_image_area(self):
        draw_mask_image = _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "draw_mask_image")
        draw_source = ast.get_source_segment(MASK_EDITOR2_PATH.read_text(), draw_mask_image)
        clip_overlay = _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "_clip_mask_overlay_to_image_area")
        clip_source = ast.get_source_segment(MASK_EDITOR2_PATH.read_text(), clip_overlay)

        self.assertIn("_clip_mask_overlay_to_image_area(glayimg, disp_info)", draw_source)
        self.assertIn("core.crop_size_and_offset_from_texture", clip_source)
        self.assertIn("np.zeros_like(glayimg)", clip_source)
        self.assertNotIn("control_points", clip_source)

    def test_crop_enable_is_not_copied_into_history_runtime_special(self):
        source = PARAMS_PATH.read_text()

        self.assertIn("'crop_enable'", source)
        self.assertIn("DO_NOT_COPY_SPECIAL_PARAM", source)
        self.assertIn("if key in DO_NOT_COPY_SPECIAL_PARAM:", source)

    def test_geometry_history_captures_crop_state_changed_by_rotation_redraw(self):
        get_param_dict = _load_class_function(EFFECTS_PATH, "GeometryEffect", "get_param_dict")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), get_param_dict)

        self.assertIn("default_param['crop_rect']", source)
        self.assertIn("default_param['disp_info']", source)


if __name__ == "__main__":
    unittest.main()
