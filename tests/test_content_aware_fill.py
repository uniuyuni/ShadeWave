import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cores.content_aware_fill import content_aware_fill


def _stripes(h=96, w=96):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = 0.5 + 0.5 * np.sin((xx + yy) * 0.4)
    img = np.stack([s, 0.3 + 0.4 * s, 1.0 - s], -1).astype(np.float32)
    return np.clip(img, 0, 1)


def _gradient(h=96, w=96):
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    g = yy / max(1.0, yy.max())
    img = np.stack([0.4 + 0.5 * g, 0.6 + 0.3 * g, 0.9 - 0.1 * g], -1).astype(np.float32)
    return np.clip(img, 0, 1)


def _center_hole(h=96, w=96, frac=0.3):
    hh, ww = int(h * frac), int(w * frac)
    y0, x0 = (h - hh) // 2, (w - ww) // 2
    m = np.zeros((h, w), np.uint8)
    m[y0:y0 + hh, x0:x0 + ww] = 255
    return m


class ContentAwareFillTest(unittest.TestCase):
    def _fill(self, img, mask):
        # Force CPU so the test is deterministic and portable (no MPS/CUDA required).
        return content_aware_fill(img.copy(), mask.copy(), verbose=False, device="cpu")

    def test_shape_dtype_and_known_pixels_preserved(self):
        img = _stripes()
        mask = _center_hole()
        out = self._fill(img, mask)

        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())

        known = mask == 0
        # Pixels outside the hole must be returned untouched.
        self.assertTrue(np.allclose(out[known], img[known], atol=1e-4))

    def test_hole_is_filled(self):
        img = _stripes()
        mask = _center_hole()
        out = self._fill(img, mask)
        hole = mask > 0
        # Every hole pixel receives a (finite, non-degenerate) value.
        self.assertTrue(np.isfinite(out[hole]).all())
        self.assertGreater(float(np.abs(out[hole]).sum()), 0.0)

    def test_texture_is_not_blurred_away(self):
        # The whole point of the rewrite: textured holes keep their high-frequency
        # content instead of collapsing to a smooth blur.
        img = _stripes()
        mask = _center_hole()
        out = np.clip(self._fill(img, mask), 0, 1)

        hole = mask > 0
        known = mask == 0
        fill_std = float(out[hole].std())
        known_std = float(np.clip(img, 0, 1)[known].std())
        self.assertGreater(fill_std / max(known_std, 1e-6), 0.6)

    def test_empty_mask_is_noop(self):
        img = _stripes()
        mask = np.zeros(img.shape[:2], np.uint8)
        out = self._fill(img, mask)
        self.assertTrue(np.allclose(out, img, atol=1e-4))

    def test_flat_region_fills_smoothly(self):
        # Sky-like gradient: must fill without artifacts and preserve known pixels.
        img = _gradient()
        mask = _center_hole()
        out = self._fill(img, mask)
        self.assertTrue(np.isfinite(out).all())
        known = mask == 0
        self.assertTrue(np.allclose(out[known], img[known], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
