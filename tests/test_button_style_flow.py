from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ButtonStyleFlowTest(unittest.TestCase):
    def test_icon_toggle_and_reset_buttons_are_dark_only(self):
        main_kv = (ROOT / "main.kv").read_text(encoding="utf-8")
        scaled_button = (ROOT / "widgets" / "scaled_button.py").read_text(encoding="utf-8")
        distortion_painter_kv = (ROOT / "widgets" / "distortion_painter.kv").read_text(encoding="utf-8")

        self.assertIn("background_color: dark_blue if self.state == 'down' else [0.58, 0.58, 0.58, 1]", main_kv)
        self.assertIn("<IconToggleButton@ScaledToggleButton>:", main_kv)
        self.assertIn("bg_color_normal: 0.12, 0.13, 0.16, 0.92", main_kv)
        self.assertNotIn("text: \"Reset\"\n                        background_color: dark_blue if self.state == 'down' else [0.12, 0.13, 0.16, 0.92]", main_kv)
        self.assertIn("bg_color_normal = ListProperty([0.38, 0.5, 0.54, 0.92])", scaled_button)
        self.assertIn("bg_color_reset_normal = ListProperty([0.12, 0.13, 0.16, 0.92])", scaled_button)
        self.assertIn('getattr(self, "text", "") == "Reset"', scaled_button)
        self.assertIn("state=self._schedule_update_bg", scaled_button)
        self.assertIn("bg_color_reset_normal=self._schedule_update_bg", scaled_button)
        self.assertIn("Clock.schedule_once(self._update_bg, 0)", scaled_button)
        self.assertIn("background_color: [0.12, 0.13, 0.16, 0.92]", distortion_painter_kv)


if __name__ == "__main__":
    unittest.main()
