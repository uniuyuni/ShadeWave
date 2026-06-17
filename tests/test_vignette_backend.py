import unittest
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import vignette_adapter, vignette_reference


class VignetteBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = vignette_adapter.backend_status()

        self.assertEqual(status.effect, "vignette")
        self.assertIn(status.backend, {"effect_backends._vignette_cpu", "effect_backends.vignette_reference"})

    def test_apply_vignette_matches_reference_implementation(self):
        rng = np.random.default_rng(123)
        image = rng.random((64, 96, 3), dtype=np.float32)
        disp_info = (3.0, -2.0, 96.0, 64.0, 1.25)
        crop_rect = (4.0, 6.0, 88.0, 58.0)
        offset = (1.5, -3.5)

        expected = vignette_reference.apply_vignette(image, -45.0, 82.0, disp_info, crop_rect, offset, 2.35)
        actual = vignette_adapter.apply_vignette(image, -45.0, 82.0, disp_info, crop_rect, offset, 2.35)

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
