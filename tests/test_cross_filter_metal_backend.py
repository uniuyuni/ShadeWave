import os
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CrossFilterMetalBackendTest(unittest.TestCase):
    def test_metal_backend_runs_when_available(self):
        from effect_backends import cross_filter_adapter

        previous = os.environ.get("PLATYPUS_CROSS_FILTER_BACKEND")
        os.environ["PLATYPUS_CROSS_FILTER_BACKEND"] = "metal"
        try:
            status = cross_filter_adapter.backend_status()
            if status.backend != "effect_backends._cross_filter_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            image = np.zeros((96, 128, 3), dtype=np.float32)
            image[32, 40] = (4.0, 3.5, 3.0)
            image[70, 95] = (3.0, 4.0, 3.2)

            actual = cross_filter_adapter.apply_cross_filter(
                image,
                num_points=4,
                length=160,
                angle_deg=20.0,
                threshold=1.0,
                intensity=0.25,
                spectral_strength=0.2,
                line_thickness=1.0,
                min_distance=8,
                randomness=0.0,
                speed_factor=4,
            )

            self.assertEqual(actual.shape, image.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertGreater(float(np.max(actual)), float(np.max(image)))
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_CROSS_FILTER_BACKEND", None)
            else:
                os.environ["PLATYPUS_CROSS_FILTER_BACKEND"] = previous


if __name__ == "__main__":
    unittest.main()
