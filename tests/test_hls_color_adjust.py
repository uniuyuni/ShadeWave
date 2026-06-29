import ast
import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORE_PATH = PROJECT_ROOT / "cores" / "core.py"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"
HLS_COLORS = ("red", "skin", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")

import cores.core as core
import effects


def _load_hls_color_setting():
    tree = ast.parse(CORE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "HLS_COLOR_SETTING":
                    return ast.literal_eval(node.value)
    raise AssertionError("HLS_COLOR_SETTING was not found")


def _assert_hue_close(testcase, actual, expected, atol=1e-6):
    diff = ((float(actual) - float(expected) + 180.0) % 360.0) - 180.0
    testcase.assertLessEqual(abs(diff), atol)


def _apply_full_weight_hue_adjust(hue, hue_adjust):
    return (hue + _normalize_hue_adjust(hue_adjust)) % 360.0


def _normalize_hue_adjust(hue_adjust):
    if hue_adjust >= 180.0:
        hue_adjust -= 360.0
    elif hue_adjust < -180.0:
        hue_adjust += 360.0
    return hue_adjust


def _apply_weighted_hue_adjust(hue, hue_adjust, weight):
    return (hue + weight * _normalize_hue_adjust(hue_adjust)) % 360.0


def _single_hls_pixel(hue):
    return np.array([[[hue % 360.0, 0.5, 0.5, 1.0]]], dtype=np.float32)


def _hue_adjust_delta(actual, source):
    return abs(((float(actual) - float(source) + 180.0) % 360.0) - 180.0)


class HLSColorAdjustTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.color_settings = _load_hls_color_setting()

    def test_each_hls_color_can_shift_to_any_hue_adjustment(self):
        for color_name in HLS_COLORS:
            center = self.color_settings[color_name]["center"]
            for hue_adjust in (-180.0, -120.0, -90.0, -45.0, 0.0, 45.0, 90.0, 120.0, 180.0):
                with self.subTest(color=color_name, hue_adjust=hue_adjust):
                    actual = _apply_full_weight_hue_adjust(center, hue_adjust)
                    _assert_hue_close(self, actual, (center + hue_adjust) % 360.0)

    def test_hls_color_plateau_uses_full_hue_adjustment_on_both_sides(self):
        for color_name in HLS_COLORS:
            setting = self.color_settings[color_name]
            center = setting["center"]
            hue_values = [
                center - setting["width"][0] * 0.5,
                center,
                center + setting["width"][1] * 0.5,
            ]
            hue_adjust = 37.5
            for hue in hue_values:
                with self.subTest(color=color_name, hue=hue):
                    actual = _apply_full_weight_hue_adjust(hue, hue_adjust)
                    _assert_hue_close(self, actual, (hue + hue_adjust) % 360.0)

    def test_runtime_hue_adjustment_uses_circular_addition(self):
        source = CORE_PATH.read_text()
        self.assertIn("new_h = (hls_img[i, j, 0] + adj_h) % 360.0", source)

    def test_plus_180_and_minus_180_match_in_weighted_falloff(self):
        for color_name in HLS_COLORS:
            center = self.color_settings[color_name]["center"]
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                with self.subTest(color=color_name, weight=weight):
                    plus = _apply_weighted_hue_adjust(center, 180.0, weight)
                    minus = _apply_weighted_hue_adjust(center, -180.0, weight)
                    _assert_hue_close(self, plus, minus)

    def test_runtime_normalizes_positive_180_before_weighting(self):
        source = CORE_PATH.read_text()
        self.assertIn("if adjust[0] >= 180.0:", source)
        self.assertIn("adjust[0] -= 360.0", source)

    def test_hls_effect_tracks_skin_color(self):
        source = EFFECTS_PATH.read_text()
        self.assertIn('"red", "skin", "orange"', source)

    def test_skin_range_extends_slightly_toward_orange_side(self):
        skin = self.color_settings["skin"]

        self.assertEqual([15.0, 16.0], skin["width"])
        self.assertEqual([20.0, 16.3], skin["fade_width"])

    def test_hls_hue_range_defaults_preserve_current_asymmetric_ranges(self):
        for color_name in HLS_COLORS:
            with self.subTest(color=color_name):
                original = self.color_settings[color_name]
                scaled = effects.HLSEffect._setting_with_hue_range(original, [100, 100])

                self.assertEqual(original["width"], scaled["width"])
                self.assertEqual(original["fade_width"], scaled["fade_width"])

    def test_hls_hue_range_scales_left_and_right_sides_independently(self):
        red = self.color_settings["red"]
        scaled = effects.HLSEffect._setting_with_hue_range(red, [50, 150])

        self.assertEqual([red["width"][0] * 0.5, red["width"][1] * 1.5], scaled["width"])
        self.assertEqual([red["fade_width"][0] * 0.5, red["fade_width"][1] * 1.5], scaled["fade_width"])

    def test_left_hls_hue_range_changes_adjusted_color_boundary(self):
        base = core.HLS_COLOR_SETTING["red"]
        probe_hue = (base["center"] - base["width"][0] * 1.5) % 360.0
        hue_adjust = 30.0

        narrow = effects.HLSEffect._setting_with_hue_range(base, [50, 100])
        narrow["adjust"] = [hue_adjust, 0.0, 0.0]
        narrow["kernel_size"] = 3
        narrow_out = core.adjust_hls_colors(_single_hls_pixel(probe_hue), [narrow], 1.0)

        expanded = effects.HLSEffect._setting_with_hue_range(base, [200, 100])
        expanded["adjust"] = [hue_adjust, 0.0, 0.0]
        expanded["kernel_size"] = 3
        expanded_out = core.adjust_hls_colors(_single_hls_pixel(probe_hue), [expanded], 1.0)

        self.assertLess(_hue_adjust_delta(narrow_out[0, 0, 0], probe_hue), 1e-4)
        self.assertGreater(_hue_adjust_delta(expanded_out[0, 0, 0], probe_hue), 25.0)

    def test_hls_hue_range_ui_uses_active_multi_slider(self):
        source = MAIN_KV_PATH.read_text()

        for color_name in HLS_COLORS:
            with self.subTest(color=color_name):
                self.assertIn(f"id: slider_hls_{color_name}_hue_range", source)

        self.assertIn('text: "Hue Range"', source)
        self.assertIn('slider_values: [100, 100]', source)
        self.assertIn('allow_overlap: True', source)
        self.assertIn('multi_value_edit_mode: "active"', source)
        self.assertIn('bar_show_active_overlay: False', source)


if __name__ == "__main__":
    unittest.main()
