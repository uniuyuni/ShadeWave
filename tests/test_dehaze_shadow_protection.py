import unittest
import os
import sys

import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effect_backends import dehaze_adapter


def _linear_luma(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _shadow_ramp_with_haze_patch():
    h, w = 96, 256
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    shadow = 0.002 + (x ** 2.2) * 0.16
    image = np.tile(shadow, (h, 1))
    image = np.dstack((image * 0.95, image, image * 1.03)).astype(np.float32)
    image[:, 190:, :] += np.array([0.16, 0.17, 0.18], dtype=np.float32)
    return np.ascontiguousarray(image, dtype=np.float32)


class DehazeShadowProtectionTest(unittest.TestCase):
    def test_positive_dehaze_40_does_not_crush_positive_shadow_tones(self):
        image = _shadow_ramp_with_haze_patch()
        result = dehaze_adapter.dehaze_image(image, 40.0 / 200.0)

        source_luma = _linear_luma(image)
        result_luma = _linear_luma(result)
        shadow = source_luma < 0.04

        self.assertTrue(np.all(result_luma[shadow] > 0.0))
        self.assertGreaterEqual(
            len(np.unique(np.round(result_luma[shadow] * 4096).astype(np.int32))),
            len(np.unique(np.round(source_luma[shadow] * 4096).astype(np.int32))) - 2,
        )

    def test_positive_dehaze_still_changes_midtones(self):
        image = _shadow_ramp_with_haze_patch()
        result = dehaze_adapter.dehaze_image(image, 40.0 / 200.0)

        source_luma = _linear_luma(image)
        midtone = (source_luma > 0.08) & (source_luma < 0.18)
        delta = np.mean(np.abs(result[midtone] - image[midtone]))

        self.assertGreater(float(delta), 0.001)

    def test_positive_dehaze_does_not_make_shadow_texture_dirtier(self):
        h, w = 160, 240
        rng = np.random.default_rng(2)
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        luma = np.tile(0.006 + (x ** 1.8) * 0.09, (h, 1))
        luma += rng.normal(0.0, 0.0025, (h, w)).astype(np.float32)
        luma += np.sin(np.arange(w, dtype=np.float32) * 1.7)[np.newaxis, :] * 0.0015
        luma = np.clip(luma, 0.0005, 1.0)
        image = np.dstack((luma * 0.96, luma, luma * 1.04)).astype(np.float32)
        image[:, 170:, :] += np.array([0.16, 0.17, 0.18], dtype=np.float32)

        result = dehaze_adapter.dehaze_image(np.ascontiguousarray(image, dtype=np.float32), 40.0 / 200.0)

        def shadow_high_frequency_std(rgb):
            y = _linear_luma(rgb)[:, :120].astype(np.float32)
            low = cv2.GaussianBlur(y, (0, 0), 2.0)
            return float(np.std(y - low))

        self.assertLessEqual(
            shadow_high_frequency_std(result),
            shadow_high_frequency_std(image) * 1.1,
        )


if __name__ == "__main__":
    unittest.main()
