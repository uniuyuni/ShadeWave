import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cores.mask2 import hls_mask


class Mask2HLSMaskSelectionTest(unittest.TestCase):
    def test_selection_luminance_uses_perceptual_encoded_brightness(self):
        rgb = np.array(
            [
                [[0.1, 0.1, 0.1], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )

        hls = hls_mask.rgb_to_selection_hls(rgb)

        self.assertGreater(float(hls[0, 1, 1]), float(hls[0, 0, 1]))
        self.assertLess(float(hls[0, 0, 1]), 1.0)
        self.assertAlmostEqual(float(hls[0, 1, 1]), 1.0, places=6)

    def test_luminance_distance_clamps_instead_of_wrapping_to_dark_values(self):
        hls = np.zeros((1, 4, 4), dtype=np.float32)
        hls[0, :, 1] = [1.0, 0.998, 0.5, 0.0]
        mask = np.ones((1, 4), dtype=np.float32)

        out = hls_mask.apply_channel_mask(
            hls,
            mask,
            "lum",
            center_xy=(0, 0),
            distance=1,
            range_min=0,
            range_max=255,
        )

        np.testing.assert_array_equal(out, np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32))

    def test_saturation_range_does_not_wrap_when_min_exceeds_max(self):
        hls = np.zeros((1, 4, 4), dtype=np.float32)
        hls[0, :, 2] = [50 / 255, 120 / 255, 180 / 255, 230 / 255]
        mask = np.ones((1, 4), dtype=np.float32)

        out = hls_mask.apply_channel_mask(
            hls,
            mask,
            "sat",
            center_xy=(0, 0),
            distance=255,
            range_min=200,
            range_max=100,
        )

        np.testing.assert_array_equal(out, np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32))

    def test_luminance_distance_255_is_full_range(self):
        hls = np.zeros((1, 4, 4), dtype=np.float32)
        hls[0, :, 1] = [1.0, 0.66, 0.33, 0.0]
        mask = np.ones((1, 4), dtype=np.float32)

        out = hls_mask.apply_channel_mask(
            hls,
            mask,
            "lum",
            center_xy=(0, 0),
            distance=255,
            range_min=0,
            range_max=255,
        )

        np.testing.assert_array_equal(out, mask)

    def test_negative_full_luminance_distance_excludes_full_range(self):
        hls = np.zeros((1, 4, 4), dtype=np.float32)
        hls[0, :, 1] = [1.0, 0.66, 0.33, 0.0]
        mask = np.ones((1, 4), dtype=np.float32)

        out = hls_mask.apply_channel_mask(
            hls,
            mask,
            "lum",
            center_xy=(0, 0),
            distance=-255,
            range_min=0,
            range_max=255,
        )

        np.testing.assert_array_equal(out, np.zeros_like(mask))

    def test_negative_luminance_distance_excludes_center_neighborhood(self):
        hls = np.zeros((1, 4, 4), dtype=np.float32)
        hls[0, :, 1] = [1.0, 0.998, 0.5, 0.0]
        mask = np.ones((1, 4), dtype=np.float32)

        out = hls_mask.apply_channel_mask(
            hls,
            mask,
            "lum",
            center_xy=(0, 0),
            distance=-1,
            range_min=0,
            range_max=255,
        )

        np.testing.assert_array_equal(out, np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32))

    def test_hue_distance_keeps_circular_wraparound(self):
        hls = np.zeros((1, 4, 4), dtype=np.float32)
        hls[0, :, 0] = [0.0, 358.0, 2.0, 180.0]
        mask = np.ones((1, 4), dtype=np.float32)

        out = hls_mask.apply_channel_mask(
            hls,
            mask,
            "hue",
            center_xy=(0, 0),
            distance=5,
            range_min=0,
            range_max=359,
        )

        np.testing.assert_array_equal(out, np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32))

    def test_negative_hue_distance_excludes_circular_center_neighborhood(self):
        hls = np.zeros((1, 4, 4), dtype=np.float32)
        hls[0, :, 0] = [0.0, 358.0, 2.0, 180.0]
        mask = np.ones((1, 4), dtype=np.float32)

        out = hls_mask.apply_channel_mask(
            hls,
            mask,
            "hue",
            center_xy=(0, 0),
            distance=-5,
            range_min=0,
            range_max=359,
        )

        np.testing.assert_array_equal(out, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
