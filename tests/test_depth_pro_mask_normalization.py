import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import depth_pro_helper


class DepthProMaskNormalizationTest(unittest.TestCase):
    def test_depth_is_converted_to_near_white_mask(self):
        depth = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)

        mask = depth_pro_helper.normalize_depth_for_mask(
            depth,
            lower_percentile=0.0,
            upper_percentile=100.0,
        )

        self.assertGreater(mask[0, 0], mask[0, 1])
        self.assertGreater(mask[0, 1], mask[0, 2])
        np.testing.assert_allclose(mask[[0], [0, 2]], [1.0, 0.0], atol=1e-6)

    def test_depth_normalization_ignores_invalid_values(self):
        depth = np.array([[0.0, np.nan, 2.0, 8.0]], dtype=np.float32)

        mask = depth_pro_helper.normalize_depth_for_mask(
            depth,
            lower_percentile=0.0,
            upper_percentile=100.0,
        )

        self.assertEqual(0.0, float(mask[0, 0]))
        self.assertEqual(0.0, float(mask[0, 1]))
        self.assertGreater(mask[0, 2], mask[0, 3])

    def test_depth_normalization_clips_percentile_outliers(self):
        depth = np.full((101,), 10.0, dtype=np.float32)
        depth[0] = 0.01
        depth[1] = 2.0
        depth[-1] = 100.0

        mask = depth_pro_helper.normalize_depth_for_mask(depth)

        self.assertEqual(1.0, float(mask[0]))
        self.assertGreater(mask[1], mask[-1])
        self.assertTrue(np.isfinite(mask).all())


if __name__ == "__main__":
    unittest.main()
