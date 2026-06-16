import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
VIEWER_PATH = ROOT / "widgets" / "viewer.py"


def _function_source(name):
    source = VIEWER_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


def _class_source(name):
    source = VIEWER_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"class not found: {name}")


class ViewerHoverHintFlowTest(unittest.TestCase):
    def test_hint_text_contains_file_identity_and_photo_metadata(self):
        source = _function_source("_build_file_hint_text")

        self.assertIn("os.path.basename(file_path", source)
        self.assertIn("os.path.dirname(file_path", source)
        self.assertIn('"CreateDate"', source)
        self.assertIn("_format_image_size(exif_data)", source)
        self.assertIn("_format_file_size(file_path)", source)
        self.assertIn('"Make"', source)
        self.assertIn('"Model"', source)
        self.assertIn('"LensModel"', source)
        self.assertIn('"ISO"', source)
        self.assertIn('"FocalLength"', source)

    def test_image_size_includes_megapixels(self):
        source = _function_source("_format_image_size")

        self.assertIn("mp = width * height / 1_000_000.0", source)
        self.assertIn('f"{width} x {height} · {mp:.1f} MP"', source)

    def test_thumbnail_card_does_not_own_window_hover_state(self):
        source = _class_source("ThumbnailCard")

        self.assertIn("_HOVER_HINT_DELAY = 0.7", VIEWER_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("KVWindow.bind(mouse_pos=", source)
        self.assertNotIn("def _on_window_mouse_pos", source)
        self.assertNotIn("_hovering", source)
        self.assertNotIn("_hover_event", source)

    def test_viewer_owns_single_window_hint_widget(self):
        source = _class_source("ViewerWidget")

        self.assertIn("self._file_hint = None", source)
        self.assertIn("KVWindow.add_widget(self._file_hint)", source)
        self.assertIn("_build_file_hint_text(file_path, exif_data)", source)
        self.assertIn("self._file_hint.hide()", source)

    def test_scroll_resyncs_hover_under_stationary_mouse(self):
        source = _class_source("ViewerWidget")

        self.assertIn("self._hover_recheck_event = None", source)
        self.assertIn("self._hover_hint_event = None", source)
        self.assertIn("self._hover_index = None", source)
        self.assertIn("KVWindow.bind(mouse_pos=self._on_window_mouse_pos)", source)
        self.assertIn("def _visible_thumbnail_cards", source)
        self.assertIn("def hover_index_at_window_pos", source)
        self.assertIn("def hover_card_at_window_pos", source)
        self.assertIn("card = self.hover_card_at_window_pos(pos)", source)
        self.assertIn('getattr(card, "index", -1)', source)
        self.assertIn("if not self.collide_point(*pos):", source)
        self.assertIn("local_pos = self.to_local(pos[0], pos[1])", source)
        self.assertIn("card.collide_point(*local_pos)", source)
        self.assertNotIn("scroll_offset =", source)
        self.assertIn("def _schedule_hover_hint", source)
        self.assertIn("_show_hover_hint(index, expected_path, mouse_pos)", source)
        self.assertIn('item.get("file_path") != expected_path', source)
        self.assertIn('self.show_file_hint(item.get("file_path"), item.get("exif_data")', source)
        self.assertIn("def _on_window_mouse_pos", source)
        self.assertIn("def _schedule_hover_recheck", source)
        self.assertIn("def _recheck_hover_cards", source)
        self.assertIn("self._on_window_mouse_pos(KVWindow, KVWindow.mouse_pos, force=True)", source)
        self.assertIn("self.bind(scroll_x=self._on_viewer_scroll_position)", source)
        self.assertIn("def _on_viewer_scroll_position", source)
        self.assertIn("def on_scroll_start", source)
        self.assertIn("def on_touch_move", source)
        self.assertIn("and self.get_drag_files()", source)
        self.assertIn("self.start_drag(touch)", source)
        self.assertIn("def on_touch_up", source)
        self.assertIn("self.dragging = False", source)
        self.assertNotIn("def on_scroll_move", source)
        self.assertNotIn("def on_scroll_stop", source)
        self.assertGreaterEqual(source.count("self._schedule_hover_recheck()"), 2)


if __name__ == "__main__":
    unittest.main()
