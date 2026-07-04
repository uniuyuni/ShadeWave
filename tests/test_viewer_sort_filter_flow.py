import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import viewer_query


ROOT = Path(__file__).resolve().parents[1]
VIEWER_PATH = ROOT / "widgets" / "viewer.py"
MAIN_PATH = ROOT / "main.py"
MAIN_KV_PATH = ROOT / "main.kv"
DIALOG_PATH = ROOT / "widgets" / "sort_filter_dialog.py"
DIALOG_KV_PATH = ROOT / "widgets" / "sort_filter_dialog.kv"


def _function_source(path, name):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


def _item(file_path, rating=0, pmck=False, exif=None):
    return {
        "file_path": file_path,
        "thumb_source": None,
        "exif_data": exif,
        "load_pending": False,
        "selected": False,
        "ctx": None,
        "rating": rating,
        "pmck_exists": pmck,
        "ai_job_state": "",
        "ai_job_progress": "",
    }


class ViewerQueryPureLogicTest(unittest.TestCase):
    def test_default_settings_sort_by_filename_ascending(self):
        items = [_item("/d/b.jpg"), _item("/d/a.jpg"), _item("/d/c.jpg")]
        view = viewer_query.build_view(items, dict(viewer_query.DEFAULT_SETTINGS))
        self.assertEqual(
            [os.path.basename(i["file_path"]) for i in view],
            ["a.jpg", "b.jpg", "c.jpg"],
        )

    def test_sort_descending_reverses_order(self):
        items = [_item("/d/a.jpg"), _item("/d/b.jpg")]
        settings = dict(viewer_query.DEFAULT_SETTINGS, sort_descending=True)
        view = viewer_query.build_view(items, settings)
        self.assertEqual(
            [os.path.basename(i["file_path"]) for i in view], ["b.jpg", "a.jpg"]
        )

    def test_sort_by_rating_with_path_tiebreak(self):
        items = [
            _item("/d/b.jpg", rating=2),
            _item("/d/a.jpg", rating=5),
            _item("/d/c.jpg", rating=2),
        ]
        settings = dict(viewer_query.DEFAULT_SETTINGS, sort_key="rating")
        view = viewer_query.build_view(items, settings)
        self.assertEqual(
            [os.path.basename(i["file_path"]) for i in view],
            ["b.jpg", "c.jpg", "a.jpg"],
        )

    def test_sort_by_date_uses_exif_create_date(self):
        items = [
            _item("/d/new.jpg", exif={"CreateDate": "2026:07:01 10:00:00"}),
            _item("/d/old.jpg", exif={"CreateDate": "2020:01:01 10:00:00"}),
        ]
        settings = dict(viewer_query.DEFAULT_SETTINGS, sort_key="date")
        view = viewer_query.build_view(items, settings)
        self.assertEqual(
            [os.path.basename(i["file_path"]) for i in view],
            ["old.jpg", "new.jpg"],
        )

    def test_sort_by_edited_puts_pmck_last_ascending(self):
        items = [_item("/d/a.jpg", pmck=True), _item("/d/b.jpg", pmck=False)]
        settings = dict(viewer_query.DEFAULT_SETTINGS, sort_key="edited")
        view = viewer_query.build_view(items, settings)
        self.assertEqual(
            [os.path.basename(i["file_path"]) for i in view],
            ["b.jpg", "a.jpg"],
        )

    def test_filter_rating_min_hides_low_rated_but_keeps_unloaded(self):
        loaded_low = _item("/d/low.jpg", rating=1, exif={})
        loaded_high = _item("/d/high.jpg", rating=4, exif={})
        pending = _item("/d/pending.jpg", rating=0, exif=None)
        settings = dict(viewer_query.DEFAULT_SETTINGS, filter_rating_min=3)
        view = viewer_query.build_view([loaded_low, loaded_high, pending], settings)
        names = {os.path.basename(i["file_path"]) for i in view}
        self.assertEqual(names, {"high.jpg", "pending.jpg"})

    def test_filter_edited_and_unedited(self):
        edited = _item("/d/e.jpg", pmck=True)
        unedited = _item("/d/u.jpg", pmck=False)
        s_edited = dict(viewer_query.DEFAULT_SETTINGS, filter_edited="edited")
        s_unedited = dict(viewer_query.DEFAULT_SETTINGS, filter_edited="unedited")
        self.assertEqual(
            [i["file_path"] for i in viewer_query.build_view([edited, unedited], s_edited)],
            ["/d/e.jpg"],
        )
        self.assertEqual(
            [i["file_path"] for i in viewer_query.build_view([edited, unedited], s_unedited)],
            ["/d/u.jpg"],
        )

    def test_filter_type_raw_vs_rgb(self):
        raw = _item("/d/a.arw")
        jpg = _item("/d/b.jpg")
        s_raw = dict(viewer_query.DEFAULT_SETTINGS, filter_type="raw")
        s_rgb = dict(viewer_query.DEFAULT_SETTINGS, filter_type="rgb")
        self.assertEqual(
            [i["file_path"] for i in viewer_query.build_view([raw, jpg], s_raw)],
            ["/d/a.arw"],
        )
        self.assertEqual(
            [i["file_path"] for i in viewer_query.build_view([raw, jpg], s_rgb)],
            ["/d/b.jpg"],
        )

    def test_filter_text_matches_basename_case_insensitive(self):
        items = [_item("/d/IMG_100.jpg"), _item("/d/DSC_200.jpg")]
        settings = dict(viewer_query.DEFAULT_SETTINGS, filter_text="img")
        view = viewer_query.build_view(items, settings)
        self.assertEqual(
            [os.path.basename(i["file_path"]) for i in view], ["IMG_100.jpg"]
        )

    def test_is_default_settings(self):
        self.assertTrue(viewer_query.is_default_settings(dict(viewer_query.DEFAULT_SETTINGS)))
        self.assertFalse(
            viewer_query.is_default_settings(
                dict(viewer_query.DEFAULT_SETTINGS, filter_rating_min=1)
            )
        )


class ViewerSortFilterSourceFlowTest(unittest.TestCase):
    def test_view_rebuild_is_single_choke_point(self):
        rebuild_source = _function_source(VIEWER_PATH, "_rebuild_view")
        coalesced_source = _function_source(VIEWER_PATH, "_do_coalesced_refresh")
        set_settings_source = _function_source(VIEWER_PATH, "set_view_settings")

        self.assertIn("viewer_query.build_view(self._all_items, self.view_settings)", rebuild_source)
        self.assertIn("self.selected_indices = {", rebuild_source)
        self.assertIn("self.refresh_from_data()", rebuild_source)
        # メタデータ到着（星/日付/pmck 変化）でソート/フィルタ結果が変わるため rebuild を通す。
        self.assertIn("self._rebuild_view()", coalesced_source)
        self.assertIn("self._rebuild_view()", set_settings_source)

    def test_selection_is_tracked_by_path_across_rebuilds(self):
        rebuild_source = _function_source(VIEWER_PATH, "_rebuild_view")
        select_source = _function_source(VIEWER_PATH, "select_at")
        toggle_source = _function_source(VIEWER_PATH, "toggle_at")
        clear_source = _function_source(VIEWER_PATH, "clear_selection")
        silent_source = _function_source(VIEWER_PATH, "set_selection_silent")

        self.assertIn("self.selected_paths", rebuild_source)
        self.assertIn("self.selected_paths.add", select_source)
        self.assertIn("self.selected_paths.discard", toggle_source)
        self.assertIn("self.selected_paths.clear()", clear_source)
        self.assertIn("self._rebuild_view()", silent_source)

    def test_mutations_target_all_items_not_view(self):
        rating_source = _function_source(VIEWER_PATH, "set_rating_for_path")
        pmck_source = _function_source(VIEWER_PATH, "set_pmck_indicator_for_path")
        deleted_source = _function_source(VIEWER_PATH, "_deleted_file")

        self.assertIn("self._item_for_path(file_path)", rating_source)
        self.assertIn("self._rebuild_view()", rating_source)
        self.assertIn("self._item_for_path(file_path)", pmck_source)
        self.assertIn("self._rebuild_view()", pmck_source)
        self.assertIn("self._remove_item(file_path)", deleted_source)

    def test_main_widget_reads_viewer_via_path_helpers(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        snapshot_source = _function_source(MAIN_PATH, "_viewer_snapshot_rating")
        exif_source = _function_source(MAIN_PATH, "_viewer_exif_for_path")

        self.assertIn("get_rating_for_path(file_path)", snapshot_source)
        self.assertIn("get_exif_for_path(file_path)", exif_source)
        self.assertIn("set_exif_for_path(fp, exif)", main_source)
        # viewer.data の直接走査は残さない（フィルタ中は一部しか入っていないため）。
        self.assertNotIn('for d in self.ids["viewer"].data', main_source)

    def test_sort_filter_dialog_is_wired_from_main(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        main_kv = MAIN_KV_PATH.read_text(encoding="utf-8")
        dialog_source = DIALOG_PATH.read_text(encoding="utf-8")
        dialog_kv = DIALOG_KV_PATH.read_text(encoding="utf-8")

        self.assertIn("from widgets.sort_filter_dialog import SortFilterDialog", main_source)
        self.assertIn("SortFilterDialog(viewer=self.ids['viewer'])", main_source)
        self.assertIn("#:include widgets/sort_filter_dialog.kv", main_kv)
        self.assertIn("root.on_sort_filter_press()", main_kv)
        self.assertIn("set_view_settings", dialog_source)
        self.assertIn("viewer_query.DEFAULT_SETTINGS", dialog_source)
        for widget_id in (
            "sort_key_spinner",
            "sort_order_spinner",
            "rating_spinner",
            "edited_spinner",
            "type_spinner",
            "filter_text_input",
        ):
            self.assertIn(widget_id, dialog_kv)


if __name__ == "__main__":
    unittest.main()
