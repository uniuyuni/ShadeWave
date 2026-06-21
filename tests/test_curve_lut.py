import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cores import core
from effects import TonecurveEffect, TonecurveRedEffect, TonecurveGreenEffect, TonecurveBlueEffect


class CurveLutTest(unittest.TestCase):
    def test_curve_lut_keeps_endpoints_for_near_end_dip(self):
        point_list = [(0.0, 0.0), (0.65, 0.9), (0.92, 0.05), (1.0, 1.0)]

        lut = core.calc_point_list_to_lut(point_list)

        self.assertAlmostEqual(float(lut[0]), 0.0, places=6)
        self.assertAlmostEqual(float(lut[-1]), 1.0, places=6)
        self.assertGreaterEqual(float(np.min(lut)), 0.0)
        self.assertLessEqual(float(np.max(lut)), 1.0)

    def test_curve_lut_does_not_go_negative_from_parametric_loop(self):
        point_list = [(0.0, 0.0), (0.087, 0.162), (0.134, 0.128), (1.0, 1.0)]

        lut = core.calc_point_list_to_lut(point_list)

        self.assertTrue(np.isfinite(lut).all())
        self.assertGreaterEqual(float(np.min(lut)), 0.0)
        self.assertAlmostEqual(float(lut[-1]), 1.0, places=6)

    def test_curve_lut_accepts_duplicate_x_positions(self):
        point_list = [(0.0, 0.0), (0.5, 0.2), (0.5, 0.8), (1.0, 1.0)]

        lut = core.calc_point_list_to_lut(point_list)

        self.assertTrue(np.isfinite(lut).all())
        self.assertAlmostEqual(float(lut[0]), 0.0, places=6)
        self.assertAlmostEqual(float(lut[-1]), 1.0, places=6)

    def test_apply_lut_default_clips_overrange_values(self):
        lut = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
        values = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)

        result = core.apply_lut(values, lut)

        self.assertAlmostEqual(float(result[0, 0]), 0.5, delta=1.0 / 65535)
        self.assertAlmostEqual(float(result[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(result[0, 2]), 1.0, places=6)

    def test_apply_lut_preserve_overrange_values(self):
        lut = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
        values = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)

        result = core.apply_lut(values, lut, overrange="preserve")

        self.assertAlmostEqual(float(result[0, 0]), 0.5, delta=1.0 / 65535)
        self.assertAlmostEqual(float(result[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(result[0, 2]), 1.5, places=6)

    def test_apply_lut_preserve_overrange_uses_endpoint_offset(self):
        lut = np.linspace(0.0, 0.8, 65536, dtype=np.float32)
        values = np.array([[1.25]], dtype=np.float32)

        result = core.apply_lut(values, lut, overrange="preserve")

        self.assertAlmostEqual(float(result[0, 0]), 1.05, places=6)

    def test_apply_lut_scale_overrange_uses_endpoint_as_gain(self):
        lut = np.linspace(0.0, 0.5, 65536, dtype=np.float32)
        values = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)

        result = core.apply_lut(values, lut, overrange="scale")

        self.assertAlmostEqual(float(result[0, 0]), 0.25, delta=1.0 / 65535)
        self.assertAlmostEqual(float(result[0, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(result[0, 2]), 0.75, places=6)
        self.assertAlmostEqual(float(result[0, 3]), 1.0, places=6)

    def test_apply_lut_scale_overrange_can_black_out_hdr(self):
        lut = np.zeros(65536, dtype=np.float32)
        values = np.array([[1.0, 1.25, 3.0]], dtype=np.float32)

        result = core.apply_lut(values, lut, overrange="scale")

        np.testing.assert_array_equal(result, np.zeros_like(values))

    def test_tonecurve_effects_use_scale_overrange_mode(self):
        for cls in (TonecurveEffect, TonecurveRedEffect, TonecurveGreenEffect, TonecurveBlueEffect):
            with self.subTest(effect=cls.__name__):
                effect = cls()
                effect.diff = np.linspace(0.0, 0.5, 65536, dtype=np.float32)
                result = effect.apply_diff(np.array([[1.5]], dtype=np.float32))
                self.assertAlmostEqual(float(result[0, 0]), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
