import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"
VIEWER_PATH = ROOT / "widgets" / "viewer.py"


def _function_source(path, name):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


class ExportViewerRefreshFlowTest(unittest.TestCase):
    def test_export_finish_syncs_exported_paths_into_viewer(self):
        finish_source = _function_source(MAIN_PATH, "_export_finish_ui")
        retry_source = _function_source(MAIN_PATH, "_export_retry_viewer_exif")

        self.assertIn("viewer.refresh_exported_paths(exported_ok or [])", finish_source)
        self.assertIn("v.refresh_exported_paths(paths or [])", retry_source)
        self.assertNotIn("refresh_exif_for_exported_path(p)", finish_source)
        self.assertNotIn("refresh_exif_for_exported_path(p)", retry_source)

    def test_viewer_export_refresh_adds_missing_current_directory_files(self):
        source = _function_source(VIEWER_PATH, "refresh_exported_paths")

        self.assertIn("self.is_supported_image(file_path)", source)
        self.assertIn("self._is_in_current_watch_directory(file_path)", source)
        self.assertIn("self._insert_image_item_sorted(file_path)", source)
        self.assertIn("file_path_dict[self.data[idx][\"file_path\"]] = idx", source)
        self.assertIn("self.load_images(file_path_dict)", source)

    def test_watch_add_and_modify_share_export_refresh_path(self):
        added_source = _function_source(VIEWER_PATH, "_added_file")
        modified_source = _function_source(VIEWER_PATH, "_modified_file")
        load_source = _function_source(VIEWER_PATH, "load_images_thread")
        pending_source = _function_source(VIEWER_PATH, "_set_load_pending")

        self.assertIn("self.refresh_exported_paths([file_path])", added_source)
        self.assertIn("self.refresh_exported_paths([file_path])", modified_source)
        self.assertNotIn("self.data.insert(idx, new_item)", added_source)
        self.assertNotIn("self.load_images({file_path: idx})", added_source)
        self.assertIn("self._mapped_or_current_index(file_path_dict, file_path)", load_source)
        self.assertIn("self._mapped_or_current_index(file_path_dict, file_path)", pending_source)


if __name__ == "__main__":
    unittest.main()
