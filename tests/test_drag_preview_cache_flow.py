import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "pipeline.py"


def _load_function(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


class DragPreviewCacheFlowTest(unittest.TestCase):
    def test_drag_preview_does_not_reuse_crop_cache_for_normal_pipeline(self):
        process_pipeline = _load_function(PIPELINE_PATH, "process_pipeline")
        source = ast.get_source_segment(PIPELINE_PATH.read_text(), process_pipeline)

        self.assertIn("pipeline_drag_preview = bool(is_drag) and not preview_full_render_enabled(current_tab)", source)
        self.assertIn("if not pipeline_drag_preview:", source)
        self.assertIn("crop_image = None", source)
        self.assertIn("return img2, crop_image if pipeline_drag_preview else imgc", source)


if __name__ == "__main__":
    unittest.main()
