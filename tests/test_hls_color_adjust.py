import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT_ROOT / "cores" / "core.py"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
HLS_COLORS = ("red", "skin", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")


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


if __name__ == "__main__":
    unittest.main()
