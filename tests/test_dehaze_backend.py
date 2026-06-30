import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import dehaze_adapter, dehaze_reference
from effects import DehazeEffect


class DehazeBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = dehaze_adapter.backend_status()

        self.assertEqual(status.effect, "dehaze")
        self.assertIn(
            status.backend,
            {
                "effect_backends._dehaze_metal",
                "effect_backends.dehaze_reference",
            },
        )

    def test_reference_can_be_forced(self):
        old_value = os.environ.get("PLATYPUS_DEHAZE_BACKEND")
        os.environ["PLATYPUS_DEHAZE_BACKEND"] = "reference"
        try:
            status = dehaze_adapter.backend_status()

            self.assertEqual(status.backend, "effect_backends.dehaze_reference")
            self.assertFalse(dehaze_adapter.native_enabled())
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_DEHAZE_BACKEND", None)
            else:
                os.environ["PLATYPUS_DEHAZE_BACKEND"] = old_value

    def test_adapter_matches_reference_for_positive_dehaze_on_hdr_input(self):
        old_value = os.environ.get("PLATYPUS_DEHAZE_BACKEND")
        os.environ["PLATYPUS_DEHAZE_BACKEND"] = "reference"
        x = np.linspace(0.0, 1.0, 96, dtype=np.float32)
        y = np.linspace(0.0, 1.0, 80, dtype=np.float32)[:, np.newaxis]
        image = np.empty((80, 96, 3), dtype=np.float32)
        image[..., 0] = 0.08 + 1.7 * x
        image[..., 1] = 0.10 + 1.3 * y
        image[..., 2] = 0.12 + 0.9 * (x + y)

        try:
            expected = dehaze_reference.dehaze_image(image, 0.18)
            actual = dehaze_adapter.dehaze_image(image, 0.18)

            self.assertEqual(actual.shape, image.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertGreater(float(np.max(actual)), 1.0)
            self.assertGreaterEqual(float(np.min(actual)), 0.0)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_DEHAZE_BACKEND", None)
            else:
                os.environ["PLATYPUS_DEHAZE_BACKEND"] = old_value

    def test_adapter_matches_reference_for_negative_haze_addition(self):
        old_value = os.environ.get("PLATYPUS_DEHAZE_BACKEND")
        os.environ["PLATYPUS_DEHAZE_BACKEND"] = "reference"
        image = np.linspace(0.0, 1.5, 32 * 40 * 3, dtype=np.float32).reshape(32, 40, 3)

        try:
            expected = dehaze_reference.dehaze_image(image, -0.35)
            actual = dehaze_adapter.dehaze_image(image, -0.35)

            self.assertEqual(actual.dtype, np.float32)
            self.assertGreater(float(np.min(actual)), float(np.min(image)))
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_DEHAZE_BACKEND", None)
            else:
                os.environ["PLATYPUS_DEHAZE_BACKEND"] = old_value

    def test_dehaze_effect_dispatches_to_adapter(self):
        image = np.ones((8, 10, 3), dtype=np.float32) * np.float32(0.5)
        expected = np.ones_like(image) * np.float32(0.75)
        effect = DehazeEffect()
        param = {"switch_precence": True, "dehaze": 40}
        efconfig = SimpleNamespace()

        with mock.patch.object(dehaze_adapter, "dehaze_image", return_value=expected) as patched:
            actual = effect.make_diff(image, param, efconfig)

        self.assertIs(actual, expected)
        patched.assert_called_once_with(image, 40 / 400)


class DehazeMetalBackendTest(unittest.TestCase):
    def _require_metal(self):
        previous = os.environ.get("PLATYPUS_DEHAZE_BACKEND")
        os.environ["PLATYPUS_DEHAZE_BACKEND"] = "metal"
        status = dehaze_adapter.backend_status()
        if status.backend != "effect_backends._dehaze_metal":
            if previous is None:
                os.environ.pop("PLATYPUS_DEHAZE_BACKEND", None)
            else:
                os.environ["PLATYPUS_DEHAZE_BACKEND"] = previous
            self.skipTest(f"Metal backend is unavailable: {status.detail}")
        return previous

    def _restore_backend(self, previous):
        if previous is None:
            os.environ.pop("PLATYPUS_DEHAZE_BACKEND", None)
        else:
            os.environ["PLATYPUS_DEHAZE_BACKEND"] = previous

    def _smooth_hdr_image(self, h=96, w=128):
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        ripple = np.sin(x * np.float32(18.0))[np.newaxis, :] * np.float32(0.035)
        image = np.empty((h, w, 3), dtype=np.float32)
        image[..., 0] = 0.06 + 1.55 * x + ripple
        image[..., 1] = 0.08 + 1.15 * y - ripple * np.float32(0.6)
        image[..., 2] = 0.10 + 0.8 * (x + y)
        return np.ascontiguousarray(np.maximum(image, 0.0), dtype=np.float32)

    def test_metal_preserves_hdr_shape_and_finite_values(self):
        previous = self._require_metal()
        try:
            image = self._smooth_hdr_image()

            for strength in (0.18, -0.35):
                with self.subTest(strength=strength):
                    actual = dehaze_adapter.dehaze_image(image, strength)

                    self.assertEqual(actual.shape, image.shape)
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertTrue(np.all(np.isfinite(actual)))
                    self.assertGreater(float(np.max(actual)), 1.0)
                    self.assertGreaterEqual(float(np.min(actual)), 0.0)
        finally:
            self._restore_backend(previous)

    def test_metal_approximates_reference_for_positive_dehaze(self):
        previous = self._require_metal()
        try:
            image = self._smooth_hdr_image(112, 144)

            expected = dehaze_reference.dehaze_image(image, 0.18)
            actual = dehaze_adapter.dehaze_image(image, 0.18)
            delta = np.abs(actual - expected)

            self.assertLess(float(np.mean(delta)), 0.012)
            self.assertLess(float(np.percentile(delta, 99)), 0.055)
            self.assertLess(float(np.max(delta)), 0.18)
            self.assertGreater(float(np.max(actual)), 1.0)
        finally:
            self._restore_backend(previous)

    def test_metal_matches_reference_for_negative_haze_addition(self):
        previous = self._require_metal()
        try:
            image = self._smooth_hdr_image(64, 72)

            expected = dehaze_reference.dehaze_image(image, -0.35)
            actual = dehaze_adapter.dehaze_image(image, -0.35)

            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        finally:
            self._restore_backend(previous)


if __name__ == "__main__":
    unittest.main()
