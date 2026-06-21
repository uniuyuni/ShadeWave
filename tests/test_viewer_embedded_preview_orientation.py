import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER_PATH = PROJECT_ROOT / "widgets" / "viewer.py"


def _source():
    return VIEWER_PATH.read_text(encoding="utf-8")


def _load_class_function(class_name, function_name):
    source = _source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class ViewerEmbeddedPreviewOrientationTest(unittest.TestCase):
    def test_preview_sources_use_imageset_style_exif_transpose(self):
        source = _source()
        decode_source = _load_class_function("ViewerWidget", "_decode_embedded_thumbnail")
        preview_source = _load_class_function("ViewerWidget", "_decode_embedded_preview")

        self.assertIn('_EMBEDDED_PREVIEW_KEYS = ("PreviewImage", "JpgFromRaw", "PreviewTIFF")', source)
        self.assertIn("for key in _EMBEDDED_PREVIEW_KEYS:", decode_source)
        self.assertIn("PILImageOps.exif_transpose(img)", preview_source)
        self.assertIn("return self._decode_embedded_preview(encoded), key", decode_source)

    def test_parent_orientation_is_only_applied_to_thumbnail_or_fallback_sources(self):
        process_source = _load_class_function("ViewerWidget", "process_exif_data")
        should_source = _load_class_function("ViewerWidget", "_should_apply_parent_orientation")

        self.assertIn("thumb, thumb_source_key = self._decode_embedded_thumbnail(exif_data)", process_source)
        self.assertIn("self._should_apply_parent_orientation(thumb_source_key)", process_source)
        self.assertIn("return embedded_key not in _EMBEDDED_PREVIEW_KEYS", should_source)


if __name__ == "__main__":
    unittest.main()
