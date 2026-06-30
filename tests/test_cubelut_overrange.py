import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cores import cubelut
from cores.lut_functions import LUT3D, LUT3x1D


def make_identity_3d_lut(size=4, domain=None):
    if domain is None:
        domain = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    domain = np.asarray(domain, dtype=np.float32)
    table = np.zeros((size, size, size, 3), dtype=np.float32)
    for r in range(size):
        for g in range(size):
            for b in range(size):
                table[r, g, b] = [
                    np.interp(b, [0, size - 1], [domain[0, 0], domain[1, 0]]),
                    np.interp(g, [0, size - 1], [domain[0, 1], domain[1, 1]]),
                    np.interp(r, [0, size - 1], [domain[0, 2], domain[1, 2]]),
                ]
    return LUT3D(table, size=size, domain=domain)


class CubeLutOverrangeTest(unittest.TestCase):
    def test_default_3d_lut_clips_overrange_values(self):
        lut = make_identity_3d_lut()
        rgb = np.array([[[-0.25, 0.5, 1.5]]], dtype=np.float32)

        result = cubelut.apply_lut(rgb, lut)

        np.testing.assert_allclose(result, [[[0.0, 0.5, 1.0]]], atol=1e-6)

    def test_preserve_3d_lut_keeps_overrange_values_continuous(self):
        lut = make_identity_3d_lut()
        rgb = np.array([[[-0.25, 0.5, 1.5]]], dtype=np.float32)

        result = cubelut.apply_lut(rgb, lut, overrange="preserve")

        np.testing.assert_allclose(result, rgb, atol=1e-6)

    def test_3d_lut_reuses_backend_ready_table(self):
        lut = make_identity_3d_lut()
        rgb = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)

        first = cubelut.apply_lut(rgb, lut)
        cached_table = lut._backend_table
        cached_domain = lut._backend_domain
        second = cubelut.apply_lut(rgb, lut)

        self.assertIs(lut._backend_table, cached_table)
        self.assertIs(lut._backend_domain, cached_domain)
        np.testing.assert_allclose(first, second, rtol=0, atol=0)

    def test_preserve_1d_lut_extends_endpoint_offset(self):
        table = np.linspace(0.0, 0.8, 4, dtype=np.float32)[:, None].repeat(3, axis=1)
        lut = LUT3x1D(table, size=4)
        rgb = np.array([[[1.25, 1.5, 2.0]]], dtype=np.float32)

        result = cubelut.apply_lut(rgb, lut, overrange="preserve")

        np.testing.assert_allclose(result, [[[1.05, 1.3, 1.8]]], atol=1e-6)

    def test_non_default_domain_is_not_rescaled_by_wrapper(self):
        domain = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        lut = make_identity_3d_lut(domain=domain)
        rgb = np.array([[[0.5, 1.0, 1.5], [2.0, 0.25, 3.0]]], dtype=np.float32)

        result = cubelut.apply_lut(rgb, lut)

        np.testing.assert_allclose(result, [[[0.5, 1.0, 1.5], [2.0, 0.25, 2.0]]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
