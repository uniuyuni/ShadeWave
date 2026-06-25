import os
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import lut_adapter, lut_reference


def _random_lut(size, rng):
    return rng.random((size, size, size, 3)).astype(np.float32)


def _domains():
    return [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[0.1, 0.0, -0.2], [0.9, 1.2, 1.0]], dtype=np.float32),
    ]


class LutReferenceTest(unittest.TestCase):
    def test_identity_lut_is_near_passthrough(self):
        size = 33
        axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
        table = np.empty((size, size, size, 3), dtype=np.float32)
        for a in range(size):
            for b in range(size):
                for c in range(size):
                    table[a, b, c] = (axis[c], axis[b], axis[a])
        domain = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        rng = np.random.default_rng(1)
        img = rng.random((20, 24, 3)).astype(np.float32)
        out = lut_reference.apply_lut3d(img, table, domain, size)
        self.assertLess(float(np.max(np.abs(out - img))), 1e-3)

    def test_preserves_shape_and_dtype(self):
        size = 8
        rng = np.random.default_rng(2)
        table = _random_lut(size, rng)
        domain = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        img = rng.random((5, 7, 3)).astype(np.float32)
        out = lut_reference.apply_lut3d(img, table, domain, size)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.float32)


class LutMetalBackendTest(unittest.TestCase):
    def _require_metal(self):
        os.environ["PLATYPUS_LUT_BACKEND"] = "metal"
        status = lut_adapter.backend_status()
        if status.backend != "effect_backends._lut_metal":
            self.skipTest(f"Metal backend is unavailable: {status.detail}")

    def test_metal_matches_reference(self):
        self._require_metal()
        rng = np.random.default_rng(3)
        max_diff = 0.0
        for size in (4, 17, 33):
            for domain in _domains():
                table = _random_lut(size, rng)
                # include out-of-domain values to exercise clipping
                img = (rng.random((37, 53, 3)).astype(np.float32) * 1.6 - 0.3)
                ref = lut_reference.apply_lut3d(img, table, domain, size)
                got = lut_adapter.apply_lut3d(img, table, domain, size)
                max_diff = max(max_diff, float(np.max(np.abs(got - ref))))
        self.assertLess(max_diff, 1e-4, f"metal/reference max diff too large: {max_diff}")


if __name__ == "__main__":
    unittest.main()
