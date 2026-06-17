import unittest
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cores import core


class LuminanceContrastTest(unittest.TestCase):
    def test_positive_contrast_expands_around_midpoint(self):
        ramp = np.linspace(0.0, 1.0, 11, dtype=np.float32)
        img = np.repeat(ramp[:, None, None], 3, axis=2)

        result = core.adjust_luminance_contrast(img, 50)
        out = result[:, 0, 0]

        self.assertLess(out[2], ramp[2])
        self.assertAlmostEqual(float(out[5]), float(ramp[5]), places=5)
        self.assertGreater(out[8], ramp[8])

    def test_positive_contrast_does_not_shift_midgray_right(self):
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)

        result = core.adjust_luminance_contrast(img, 50)

        self.assertTrue(np.allclose(result, img, atol=1e-5))

    def test_positive_contrast_expands_dark_image_around_image_pivot(self):
        ramp = np.linspace(0.0, 0.4, 11, dtype=np.float32)
        img = np.repeat(ramp[:, None, None], 3, axis=2)

        result = core.adjust_luminance_contrast(img, 50)
        out = result[:, 0, 0]

        self.assertLess(out[2], ramp[2])
        self.assertGreater(out[8], ramp[8])

    def test_positive_contrast_softens_shadow_side_only(self):
        ramp = np.linspace(0.0, 1.0, 11, dtype=np.float32)
        img = np.repeat(ramp[:, None, None], 3, axis=2)

        result = core.adjust_luminance_contrast(img, 20, c=0.5)
        out = result[:, 0, 0]
        old_linear = np.maximum(0.5 + (ramp - 0.5) * 1.2, 0.0)

        self.assertGreater(out[1], old_linear[1])
        self.assertGreater(out[2], old_linear[2])
        self.assertAlmostEqual(float(out[5]), float(ramp[5]), places=5)
        self.assertAlmostEqual(float(out[8]), float(old_linear[8]), places=5)

    def test_luminance_contrast_preserves_channel_ratios(self):
        img = np.array([[[0.2, 0.4, 0.8]]], dtype=np.float32)

        result = core.adjust_luminance_contrast(img, 50)

        self.assertAlmostEqual(float(result[0, 0, 0] / result[0, 0, 1]), 0.5, places=5)
        self.assertAlmostEqual(float(result[0, 0, 2] / result[0, 0, 1]), 2.0, places=5)

    def test_negative_contrast_lifts_dark_color_without_washing_out(self):
        img = np.array([[[0.12, 0.02, 0.02]]], dtype=np.float32)

        result = core.adjust_luminance_contrast(img, -50, c=0.5)

        self.assertGreater(result[0, 0, 0], img[0, 0, 0])
        self.assertGreater(result[0, 0, 1], img[0, 0, 1])
        self.assertGreater(result[0, 0, 0] - result[0, 0, 1], 0.05)

    def test_negative_contrast_protects_deep_black_from_lifting(self):
        img = np.full((1, 1, 3), 0.01, dtype=np.float32)

        result = core.adjust_luminance_contrast(img, -50, c=0.5)

        self.assertLess(float(result[0, 0, 0]), 0.05)

    def test_negative_contrast_keeps_black_point_fixed(self):
        img = np.zeros((1, 1, 3), dtype=np.float32)

        result = core.adjust_luminance_contrast(img, -100, c=0.5)

        self.assertTrue(np.allclose(result, img, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
