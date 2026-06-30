import os
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import coating_adapter, coating_reference


def _hdr_image(height=48, width=64):
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    img = np.empty((height, width, 3), dtype=np.float32)
    img[..., 0] = x * np.float32(2.0)
    img[..., 1] = y * np.float32(1.5)
    img[..., 2] = (x + y) * np.float32(1.2)
    img[height // 2 - 1:height // 2 + 2, width // 2 - 1:width // 2 + 2] = np.float32(6.0)
    return img


class CoatingBackendTest(unittest.TestCase):
    def test_presets_are_exposed_through_adapter(self):
        presets = coating_adapter.presets()
        self.assertIn("VINTAGE_NO_COAT", presets)
        self.assertEqual(presets["VINTAGE_NO_COAT"]["name"], "Vintage No-Coat")

    def test_forced_reference_matches_reference_module_and_preserves_hdr(self):
        img = _hdr_image()
        old_backend = os.environ.get("PLATYPUS_COATING_BACKEND")
        os.environ["PLATYPUS_COATING_BACKEND"] = "reference"
        try:
            out = coating_adapter.apply_preset(
                img,
                "VINTAGE_NO_COAT",
                light_source_intensity=2.0,
                resolution_scale=0.5,
            )
            expected = coating_reference.apply_preset(
                img,
                "VINTAGE_NO_COAT",
                light_source_intensity=2.0,
                resolution_scale=0.5,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_COATING_BACKEND", None)
            else:
                os.environ["PLATYPUS_COATING_BACKEND"] = old_backend

        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 1.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    @unittest.skipUnless(coating_adapter.native_available(), "coating Metal backend is not built")
    def test_metal_matches_reference_closely(self):
        img = _hdr_image(36, 44)
        old_backend = os.environ.get("PLATYPUS_COATING_BACKEND")
        try:
            os.environ["PLATYPUS_COATING_BACKEND"] = "reference"
            reference = coating_adapter.apply_preset(
                img,
                "VINTAGE_NO_COAT",
                light_source_intensity=2.0,
                resolution_scale=0.35,
            )
            os.environ["PLATYPUS_COATING_BACKEND"] = "metal"
            status = coating_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = coating_adapter.apply_preset(
                img,
                "VINTAGE_NO_COAT",
                light_source_intensity=2.0,
                resolution_scale=0.35,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_COATING_BACKEND", None)
            else:
                os.environ["PLATYPUS_COATING_BACKEND"] = old_backend

        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
