import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
EFFECTS_PATH = ROOT / "effects.py"
MAIN_PATH = ROOT / "main.py"
MAIN_KV_PATH = ROOT / "main.kv"


def _function_source(path, name):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


class Mask2RangeSliderFlowTest(unittest.TestCase):
    def test_mask2_hls_uses_single_two_point_sliders_in_kv(self):
        kv = MAIN_KV_PATH.read_text(encoding="utf-8")

        self.assertIn("id: slider_mask2_hue_range", kv)
        self.assertIn("slider_values: [0, 359]", kv)
        self.assertIn("allow_overlap: True", kv)
        self.assertIn('multi_value_edit_mode:', kv)
        self.assertNotIn("id: slider_mask2_hue_min", kv)
        self.assertNotIn("id: slider_mask2_hue_max", kv)
        self.assertIn("id: slider_mask2_lum_range", kv)
        self.assertIn("id: slider_mask2_sat_range", kv)
        self.assertIn("slider_values: [0, 255]", kv)
        self.assertNotIn("id: slider_mask2_lum_min", kv)
        self.assertNotIn("id: slider_mask2_lum_max", kv)
        self.assertNotIn("id: slider_mask2_sat_min", kv)
        self.assertNotIn("id: slider_mask2_sat_max", kv)

    def test_mask2_effect_reads_and_writes_hls_range_values(self):
        source = EFFECTS_PATH.read_text(encoding="utf-8")

        self.assertIn('widget.ids["slider_mask2_hue_range"].set_slider_value([', source)
        self.assertIn("self._get_param(param, 'mask2_hue_min')", source)
        self.assertIn("self._get_param(param, 'mask2_hue_max')", source)
        self.assertIn('hue_values = list(widget.ids["slider_mask2_hue_range"].ids["slider"].values)', source)
        self.assertIn("param['mask2_hue_min'] = hue_values[0]", source)
        self.assertIn("param['mask2_hue_max'] = hue_values[-1]", source)
        self.assertNotIn('widget.ids["slider_mask2_hue_min"].value', source)
        self.assertNotIn('widget.ids["slider_mask2_hue_max"].value', source)
        self.assertIn('widget.ids["slider_mask2_lum_range"].set_slider_value([', source)
        self.assertIn('lum_values = list(widget.ids["slider_mask2_lum_range"].ids["slider"].values)', source)
        self.assertIn("param['mask2_lum_min'] = lum_values[0]", source)
        self.assertIn("param['mask2_lum_max'] = lum_values[-1]", source)
        self.assertNotIn('widget.ids["slider_mask2_lum_min"].value', source)
        self.assertNotIn('widget.ids["slider_mask2_lum_max"].value', source)
        self.assertIn('widget.ids["slider_mask2_sat_range"].set_slider_value([', source)
        self.assertIn('sat_values = list(widget.ids["slider_mask2_sat_range"].ids["slider"].values)', source)
        self.assertIn("param['mask2_sat_min'] = sat_values[0]", source)
        self.assertIn("param['mask2_sat_max'] = sat_values[-1]", source)
        self.assertNotIn('widget.ids["slider_mask2_sat_min"].value', source)
        self.assertNotIn('widget.ids["slider_mask2_sat_max"].value', source)

    def test_color_shortcut_updates_range_values_with_events(self):
        source = _function_source(MAIN_PATH, "set_mask2_hue_range")

        self.assertIn("slider = self.ids['slider_mask2_hue_range'].ids['slider']", source)
        self.assertIn("slider.active_index = 0", source)
        self.assertIn("slider.values = [hmin, hmax]", source)
        self.assertNotIn("slider_mask2_hue_min", source)
        self.assertNotIn("slider_mask2_hue_max", source)


if __name__ == "__main__":
    unittest.main()
