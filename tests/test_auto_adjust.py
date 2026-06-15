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
        self.assertTrue(adjustment["switch_exposure_contrast"])
        self.assertTrue(adjustment["switch_tone"])

    def test_bright_highlights_are_protected(self):
        image = np.ones((96, 128, 3), dtype=np.float32) * 0.22
        image[:, 96:, :] = 1.8

        adjustment = auto_adjust.compute_basic_auto_adjustment(image)

        self.assertLessEqual(adjustment["highlight"], 0)
        self.assertLessEqual(adjustment["white"], 0)

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
        self.assertIn("on_release: root.on_auto_adjust_press()", kv_source)
        self.assertIn("auto_adjust.compute_basic_auto_adjustment", handler_source)
        self.assertIn("self.begin_history_effect_ctrl(2, effect_list)", handler_source)
        self.assertIn("self.end_history_effect_ctrl(2, effect_list)", handler_source)


if __name__ == "__main__":
    unittest.main()
