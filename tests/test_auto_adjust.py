import ast
import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import auto_adjust


MAIN_PATH = PROJECT_ROOT / "main.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class AutoAdjustTest(unittest.TestCase):
    def test_dark_flat_image_gets_exposure_and_contrast(self):
        x = np.linspace(0.05, 0.18, 128, dtype=np.float32)
        image = np.repeat(x[None, :, None], 96, axis=0)
        image = np.repeat(image, 3, axis=2)

        adjustment = auto_adjust.compute_basic_auto_adjustment(image)

        self.assertGreater(adjustment["exposure"], 0.0)
        self.assertGreater(adjustment["contrast"], 0)
        self.assertGreaterEqual(adjustment["clarity"], 0)
        self.assertGreaterEqual(adjustment["vibrance"], 0)
        self.assertTrue(adjustment["switch_exposure_contrast"])
        self.assertTrue(adjustment["switch_tone"])
        self.assertTrue(adjustment["switch_precence"])
        self.assertTrue(adjustment["switch_saturation"])

    def test_bright_highlights_are_protected(self):
        image = np.ones((96, 128, 3), dtype=np.float32) * 0.22
        image[:, 96:, :] = 1.8

        adjustment = auto_adjust.compute_basic_auto_adjustment(image)

        self.assertLessEqual(adjustment["highlight"], 0)
        self.assertLessEqual(adjustment["white"], 0)

    def test_desaturated_image_gets_color_and_presence_boost(self):
        x = np.linspace(0.12, 0.62, 128, dtype=np.float32)
        image = np.repeat(x[None, :, None], 96, axis=0)
        image = np.repeat(image, 3, axis=2)
        image[..., 0] *= 1.02
        image[..., 2] *= 0.98

        adjustment = auto_adjust.compute_basic_auto_adjustment(image)

        self.assertGreater(adjustment["vibrance"], 0)
        self.assertGreater(adjustment["saturation"], 0)
        self.assertGreater(adjustment["color_density"], 0)
        self.assertGreater(adjustment["color_separation"], 0)
        self.assertGreater(adjustment["texture"], 0)
        self.assertGreater(adjustment["microcontrast"], 0)

    def test_already_saturated_image_avoids_overcooking_color(self):
        x = np.linspace(0.12, 0.72, 128, dtype=np.float32)
        image = np.zeros((96, 128, 3), dtype=np.float32)
        image[..., 0] = x[None, :] * 1.25
        image[..., 1] = x[None, :] * 0.18
        image[..., 2] = x[None, :] * 0.08

        adjustment = auto_adjust.compute_basic_auto_adjustment(image)

        self.assertLessEqual(adjustment["saturation"], 5)
        self.assertLessEqual(adjustment["vibrance"], 8)
        self.assertLessEqual(adjustment["color_density"], 5)

    def test_high_dynamic_dark_interior_keeps_tone_controls_moderate(self):
        x = np.linspace(0.002, 0.10, 220, dtype=np.float32)
        image = np.ones((160, 220, 3), dtype=np.float32) * x[None, :, None]
        image[20:80, 150:210, :] = 0.75
        image[80:150, 120:200, :] = 0.18

        adjustment = auto_adjust.compute_basic_auto_adjustment(image)

        self.assertLessEqual(adjustment["exposure"], 0.8)
        self.assertLessEqual(abs(adjustment["highlight"]), 22)
        self.assertLessEqual(adjustment["shadow"], 18)
        self.assertLessEqual(abs(adjustment["contrast"]), 16)
        self.assertLessEqual(adjustment["detail_tonemap"], 4)

    def test_crop_rect_limits_analysis_region(self):
        image = np.ones((80, 120, 3), dtype=np.float32) * 0.8
        image[:, :60, :] = 0.06

        full = auto_adjust.compute_basic_auto_adjustment(image)
        cropped = auto_adjust.compute_basic_auto_adjustment(image, crop_rect=(0, 0, 60, 80))

        self.assertGreater(cropped["exposure"], full["exposure"])

    def test_auto_button_is_connected_to_main_handler(self):
        kv_source = MAIN_KV_PATH.read_text()
        main_source = MAIN_PATH.read_text()
        handler_source = ast.get_source_segment(
            main_source,
            _load_class_function(MAIN_PATH, "MainWidget", "on_auto_adjust_press"),
        )

        self.assertIn("id: auto_adjust", kv_source)
        self.assertIn("disabled: root.image_loaded == False or root.mask2_wait_full_load == True", kv_source)
        self.assertIn("on_release: root.on_auto_adjust_press()", kv_source)
        self.assertIn("if self.mask2_wait_full_load or getattr(self, \"_actively_loading\", False):", handler_source)
        self.assertIn("auto_adjust.compute_basic_auto_adjustment", handler_source)
        self.assertIn("self.begin_history_effect_ctrl(2, effect_list)", handler_source)
        self.assertIn("self.end_history_effect_ctrl(2, effect_list)", handler_source)
        self.assertIn("'vs_and_saturation'", handler_source)
        self.assertIn("'color_separation'", handler_source)
        self.assertIn("'microcontrast'", handler_source)


if __name__ == "__main__":
    unittest.main()
