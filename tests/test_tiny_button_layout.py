import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]


class TinyButtonLayoutTest(unittest.TestCase):
    def test_tiny_button_background_fills_cell_without_separate_border(self):
        source = (ROOT / "widgets" / "tiny_button.kv").read_text(encoding="utf-8")

        self.assertIn("pos: self.x, self.y", source)
        self.assertIn("size: self.width, self.height", source)
        self.assertNotIn("pos: int(self.x) + 1, int(self.y)", source)
        self.assertNotIn("background_color:", source)
        self.assertNotIn("background_normal:", source)

    def test_tiny_button_uses_input_colored_background_and_own_thin_border(self):
        kv_source = (ROOT / "widgets" / "tiny_button.kv").read_text(encoding="utf-8")
        py_source = (ROOT / "widgets" / "tiny_button.py").read_text(encoding="utf-8")

        self.assertIn("Line:", kv_source)
        self.assertIn("width: 1", kv_source)
        self.assertIn("Triangle:", kv_source)
        self.assertIn("class TinyButton(ButtonBehavior, Widget):", py_source)
        self.assertIn("bg_color_normal = ColorProperty([0.18, 0.18, 0.18, 1])", py_source)
        self.assertNotIn("from kivy.uix.button import Button", py_source)
        self.assertNotIn("border_width", py_source)


if __name__ == "__main__":
    unittest.main()
