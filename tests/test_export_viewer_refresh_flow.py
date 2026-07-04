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

        self.assertIn("self.is_visible_image(file_path)", source)
        self.assertIn("self._is_in_current_watch_directory(file_path)", source)
        self.assertIn("self._add_item_if_missing(file_path)", source)
        self.assertIn("self._rebuild_view()", source)
        self.assertIn("self.load_images(load_paths)", source)

    def test_watch_add_and_modify_share_export_refresh_path(self):
        added_source = _function_source(VIEWER_PATH, "_added_file")
        modified_source = _function_source(VIEWER_PATH, "_modified_file")
        apply_meta_source = _function_source(VIEWER_PATH, "_apply_metadata")
        apply_thumb_source = _function_source(VIEWER_PATH, "_apply_thumbnail")
        pending_source = _function_source(VIEWER_PATH, "_set_load_pending")

        self.assertIn("self.refresh_exported_paths([file_path])", added_source)
        self.assertIn("self.refresh_exported_paths([file_path])", modified_source)
        self.assertNotIn("self.data.insert(idx, new_item)", added_source)
        self.assertNotIn("self.load_images({file_path: idx})", added_source)
        # 反映時は path で現在の item を引き直す（リネーム/削除に安全）。
        self.assertIn("self._item_for_path(file_path)", apply_meta_source)
        self.assertIn("self._item_for_path(file_path)", apply_thumb_source)
        self.assertIn("self._item_for_path(file_path)", pending_source)

    def test_viewer_ignores_hidden_temp_image_paths(self):
        refresh_source = _function_source(VIEWER_PATH, "refresh_exported_paths")
        added_source = _function_source(VIEWER_PATH, "_added_file")
        modified_source = _function_source(VIEWER_PATH, "_modified_file")
        set_path_source = _function_source(VIEWER_PATH, "set_path")
        visible_source = _function_source(VIEWER_PATH, "is_visible_image")

        self.assertIn("not self.is_visible_image(file_path)", refresh_source)
        self.assertIn("self.is_visible_image(file_path)", added_source)
        self.assertIn("not self.is_visible_image(file_path)", modified_source)
        self.assertIn("self.is_visible_image(file_name)", set_path_source)
        self.assertIn('os.path.basename(str(file_name or ""))', visible_source)
        self.assertIn('not basename.startswith(".")', visible_source)
        self.assertIn("self.is_supported_image(file_name)", visible_source)

    def test_watch_directory_changes_restart_watchfiles(self):
        viewer_source = VIEWER_PATH.read_text(encoding="utf-8")
        watch_source = _function_source(VIEWER_PATH, "_watchfiles_thread")
        set_watch_source = _function_source(VIEWER_PATH, "_set_watch_directory")
        set_path_source = _function_source(VIEWER_PATH, "set_path")
        deleted_source = _function_source(VIEWER_PATH, "_deleted_file")

        self.assertIn("self._watch_directory_lock = threading.Lock()", viewer_source)
        self.assertIn("self._watch_stop_event = None", viewer_source)
        self.assertIn("watch(watch_directory, stop_event=stop_event)", watch_source)
        self.assertIn("self._watch_stop_event.set()", set_watch_source)
        self.assertIn("self._watch_stop_event = threading.Event() if directory else None", set_watch_source)
        self.assertIn("self._set_watch_directory(directory)", set_path_source)
        self.assertNotIn("self.watch_directory = directory", set_path_source)
        self.assertIn("self._is_in_current_watch_directory(file_path)", deleted_source)


if __name__ == "__main__":
    unittest.main()
