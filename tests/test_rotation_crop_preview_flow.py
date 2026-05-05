import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
MAIN_PATH = PROJECT_ROOT / "main.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class RotationCropPreviewFlowTest(unittest.TestCase):
    def test_rotation_slider_wraps_crop_preview_between_history_begin_and_end(self):
        source = MAIN_KV_PATH.read_text()

        self.assertIn("root.begin_history_effect_ctrl(0, 'geometry')", source)
        self.assertIn("root.begin_rotation_crop_preview()", source)
        self.assertIn("root.end_rotation_crop_preview()", source)
        self.assertIn("root.end_history_effect_ctrl(0, 'geometry')", source)
        self.assertLess(
            source.index("root.begin_history_effect_ctrl(0, 'geometry')"),
            source.index("root.begin_rotation_crop_preview()"),
        )
        self.assertLess(
            source.index("root.end_rotation_crop_preview()"),
            source.index("root.end_history_effect_ctrl(0, 'geometry')"),
        )

    def test_crop_effect_uses_saved_crop_rect_during_rotation_preview_without_committing(self):
        node = _load_class_function(EFFECTS_PATH, "CropEffect", "set2param")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertIn("self._rotation_preview_crop_rect is not None", source)
        self.assertIn("self.crop_editor.set_to_local_crop_rect(self._rotation_preview_crop_rect)", source)
        self.assertIn("self.crop_editor.update_crop_size()", source)
        self.assertIn("if self._rotation_preview_crop_rect is None:", source)
        self.assertIn("params.set_crop_rect(param, self.crop_editor.get_crop_rect", source)

    def test_crop_effect_commits_preview_crop_only_when_rotation_edit_ends(self):
        begin_node = _load_class_function(EFFECTS_PATH, "CropEffect", "begin_rotation_preview")
        end_node = _load_class_function(EFFECTS_PATH, "CropEffect", "end_rotation_preview")
        source = EFFECTS_PATH.read_text()
        begin_source = ast.get_source_segment(source, begin_node)
        end_source = ast.get_source_segment(source, end_node)

        self.assertIn("self._rotation_preview_crop_rect = params.get_crop_rect(param)", begin_source)
        self.assertIn("self.crop_editor.set_to_local_crop_rect(self._rotation_preview_crop_rect)", end_source)
        self.assertIn("self.crop_editor.update_crop_size()", end_source)
        self.assertIn("params.set_crop_rect(param, self.crop_editor.get_crop_rect())", end_source)
        self.assertIn("self._rotation_preview_crop_rect = None", end_source)

    def test_crop_editor_resync_keeps_rotation_preview_instead_of_param_rect(self):
        node = _load_class_function(EFFECTS_PATH, "CropEffect", "sync_crop_editor_from_param")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertIn("self._rotation_preview_crop_rect", source)
        self.assertIn("if self._rotation_preview_crop_rect is not None:", source)
        self.assertIn("self.crop_editor.set_to_local_crop_rect(crop_rect)", source)
        self.assertIn("self.crop_editor.update_crop_size()", source)
        preview_block = source.split("if self._rotation_preview_crop_rect is not None:", 1)[1].split("# set_aspect_ratio", 1)[0]
        self.assertIn("return", preview_block)

    def test_main_exposes_rotation_crop_preview_hooks(self):
        source = MAIN_PATH.read_text()

        self.assertIn("def begin_rotation_crop_preview(self):", source)
        self.assertIn("begin_rotation_preview(self.primary_param)", source)
        self.assertIn("def end_rotation_crop_preview(self):", source)
        self.assertIn("end_rotation_preview(self.primary_param)", source)


if __name__ == "__main__":
    unittest.main()
