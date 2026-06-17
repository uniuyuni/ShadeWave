import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import cross_filter_adapter, cross_filter_reference


class CrossFilterBackendTest(unittest.TestCase):
    def _peak_image(self):
        image = np.zeros((72, 96, 3), dtype=np.float32)
        image[20, 24] = (4.0, 3.0, 2.0)
        image[50, 70] = (3.0, 4.0, 2.5)
        return image

    def test_backend_status_is_reported(self):
        status = cross_filter_adapter.backend_status()

        self.assertEqual(status.effect, "cross_filter")
        self.assertIn(status.backend, {"effect_backends._cross_filter_cpu", "effect_backends.cross_filter_reference"})

    def test_adapter_output_shape_dtype_and_effect(self):
        image = self._peak_image()

        actual = cross_filter_adapter.apply_cross_filter(
            image,
            num_points=4,
            length=80,
            angle_deg=15.0,
            threshold=1.0,
            intensity=0.35,
            spectral_strength=0.15,
            line_thickness=1.0,
            min_distance=5,
            randomness=0.0,
            speed_factor=4,
        )

        self.assertEqual(actual.shape, image.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertGreater(float(np.max(actual)), float(np.max(image)))

    def test_debug_mode_marks_reference_peaks(self):
        image = self._peak_image()

        actual = cross_filter_adapter.apply_cross_filter(
            image,
            num_points=4,
            length=40,
            threshold=1.0,
            min_distance=5,
            debug_mode=True,
        )

        self.assertEqual(actual.shape, image.shape)
        self.assertGreaterEqual(float(actual[20, 24, 2]), 9.0)
        self.assertGreaterEqual(float(actual[50, 70, 2]), 9.0)

    def test_reference_remains_available(self):
        image = self._peak_image()
        expected = cross_filter_reference.apply_cross_filter(
            image,
            num_points=4,
            length=40,
            threshold=1.0,
            intensity=0.2,
            min_distance=5,
            randomness=0.0,
            speed_factor=4,
        )

        self.assertEqual(expected.shape, image.shape)
        self.assertEqual(expected.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
