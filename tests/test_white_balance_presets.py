import ast
import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import effects
from cores import core


MAIN_PATH = PROJECT_ROOT / "main.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"
HOVER_SPINNER_PATH = PROJECT_ROOT / "widgets" / "hover_spinner.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class WhiteBalancePresetTest(unittest.TestCase):
    def test_color_temperature_preset_values_include_common_wb_options(self):
        options = effects.ColorTemperatureEffect.preset_options()

        self.assertEqual(options[0], "As Shot")
        self.assertIn("Daylight", options)
        self.assertIn("Cloudy", options)
        self.assertIn("Shade", options)
        self.assertIn("Tungsten", options)
        self.assertIn("Fluorescent", options)
        self.assertIn("Flash", options)
        self.assertEqual(options[-1], "Custom")
        self.assertEqual(
            effects.ColorTemperatureEffect.preset_values("Tungsten", {}),
            (2850, 0),
        )
        self.assertEqual(
            effects.ColorTemperatureEffect.preset_values(
                "As Shot",
                {"color_temperature_reset": 6120, "color_tint_reset": -4},
            ),
            (6120, -4),
        )

    def test_missing_preset_key_infers_as_shot_or_custom(self):
        self.assertEqual(
            effects.ColorTemperatureEffect.infer_preset({
                "color_temperature_reset": 5300,
                "color_tint_reset": 3,
                "color_temperature": 5300,
                "color_tint": 3,
            }),
            "As Shot",
        )
        self.assertEqual(
            effects.ColorTemperatureEffect.infer_preset({
                "color_temperature_reset": 5300,
                "color_tint_reset": 3,
                "color_temperature": 5600,
                "color_tint": 3,
            }),
            "Custom",
        )

    def test_white_balance_preset_ui_is_connected(self):
        kv_source = MAIN_KV_PATH.read_text()
        main_source = MAIN_PATH.read_text()
        preset_handler = ast.get_source_segment(
            main_source,
            _load_class_function(MAIN_PATH, "MainWidget", "on_color_temperature_preset_value"),
        )
        custom_handler = ast.get_source_segment(
            main_source,
            _load_class_function(MAIN_PATH, "MainWidget", "on_color_temperature_slider_changed"),
        )

        self.assertIn("id: spinner_color_temperature_preset", kv_source)
        self.assertIn("values: [ 'As Shot', 'Daylight', 'Cloudy', 'Shade', 'Tungsten', 'Fluorescent', 'Flash', 'Custom' ]", kv_source)
        self.assertIn("root.on_color_temperature_preset_value(self.text)", kv_source)
        self.assertIn("root.on_color_temperature_slider_changed()", kv_source)
        self.assertIn("effects.ColorTemperatureEffect.preset_values", preset_handler)
        self.assertIn("effects.ColorTemperatureEffect.PRESET_AS_SHOT", preset_handler)
        self.assertIn('self.ids["slider_color_temperature"].reset_value', preset_handler)
        self.assertIn('self.ids["slider_color_tint"].reset_value', preset_handler)
        self.assertIn("effects.ColorTemperatureEffect.PRESET_CUSTOM", custom_handler)

    def test_as_shot_param_write_uses_widget_reset_values(self):
        source = pathlib.Path(effects.__file__).read_text()
        set2param = ast.get_source_segment(
            source,
            _load_class_function(pathlib.Path(effects.__file__), "ColorTemperatureEffect", "set2param"),
        )

        self.assertIn("preset == self.PRESET_AS_SHOT", set2param)
        self.assertIn('widget.ids["slider_color_temperature"].reset_value', set2param)
        self.assertIn('widget.ids["slider_color_tint"].reset_value', set2param)

    def test_invert_temp_tint_rgb_reflects_temp_and_negates_tint_around_reference(self):
        temp, tint, Y, reference_temp = 5500.0, 4.0, 1.0, 5000.0

        mired_temp = 1e6 / temp
        mired_ref = 1e6 / reference_temp
        expected_inverted_temp = 1e6 / (mired_ref - (mired_temp - mired_ref))
        expected_inverted_tint = -tint
        expected_rgb = core.convert_TempTint2RGB(expected_inverted_temp, expected_inverted_tint, Y)

        actual_rgb = core.invert_TempTint2RGB(temp, tint, Y, reference_temp=reference_temp)

        np.testing.assert_allclose(actual_rgb, expected_rgb, rtol=1e-6, atol=1e-6)

    def test_hover_spinner_dispatches_same_user_value(self):
        source = HOVER_SPINNER_PATH.read_text()
        on_text = ast.get_source_segment(
            source,
            _load_class_function(HOVER_SPINNER_PATH, "HoverSpinner", "on_text"),
        )

        self.assertIn("if self.value == value", on_text)
        self.assertIn("self.property('value').dispatch(self)", on_text)


if __name__ == "__main__":
    unittest.main()
