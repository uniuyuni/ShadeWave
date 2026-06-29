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

from effect_backends import local_contrast_adapter, local_contrast_reference
import cores.local_contrast as local_contrast_shim
from effects import ClarityEffect, MicroContrastEffect, TextureEffect


class LocalContrastBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = local_contrast_adapter.backend_status()

        self.assertEqual(status.effect, "local_contrast")
        self.assertIn(
            status.backend,
            {
                "effect_backends._local_contrast_metal",
                "effect_backends.local_contrast_reference",
            },
        )

    def test_reference_can_be_forced(self):
        old_value = os.environ.get("PLATYPUS_LOCAL_CONTRAST_BACKEND")
        os.environ["PLATYPUS_LOCAL_CONTRAST_BACKEND"] = "reference"
        try:
            status = local_contrast_adapter.backend_status()

            self.assertEqual(status.backend, "effect_backends.local_contrast_reference")
            self.assertFalse(local_contrast_adapter.native_enabled())
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_LOCAL_CONTRAST_BACKEND", None)
            else:
                os.environ["PLATYPUS_LOCAL_CONTRAST_BACKEND"] = old_value

    def test_adapter_matches_reference_for_all_three_effects_on_hdr_input(self):
        old_value = os.environ.get("PLATYPUS_LOCAL_CONTRAST_BACKEND")
        os.environ["PLATYPUS_LOCAL_CONTRAST_BACKEND"] = "reference"
        rng = np.random.default_rng(123)
        image = rng.random((36, 48, 3), dtype=np.float32) * np.float32(2.4)

        try:
            cases = [
                (
                    local_contrast_reference.apply_clarity,
                    local_contrast_adapter.apply_clarity,
                    0.35,
                ),
                (
                    local_contrast_reference.apply_texture,
                    local_contrast_adapter.apply_texture,
                    -0.25,
                ),
                (
                    local_contrast_reference.apply_microcontrast,
                    local_contrast_adapter.apply_microcontrast,
                    0.65,
                ),
            ]

            for reference_func, adapter_func, strength in cases:
                with self.subTest(effect=adapter_func.__name__):
                    expected = reference_func(image, strength)
                    actual = adapter_func(image, strength)

                    self.assertEqual(actual.dtype, np.float32)
                    self.assertEqual(actual.shape, image.shape)
                    self.assertTrue(np.all(np.isfinite(actual)))
                    self.assertGreater(float(np.max(actual)), 1.0)
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_LOCAL_CONTRAST_BACKEND", None)
            else:
                os.environ["PLATYPUS_LOCAL_CONTRAST_BACKEND"] = old_value

    def test_zero_strength_returns_equal_copy(self):
        image = np.linspace(0.0, 1.5, 12 * 14 * 3, dtype=np.float32).reshape(12, 14, 3)

        for apply_func in (
            local_contrast_adapter.apply_clarity,
            local_contrast_adapter.apply_texture,
            local_contrast_adapter.apply_microcontrast,
        ):
            with self.subTest(effect=apply_func.__name__):
                actual = apply_func(image, 0)

                self.assertIsNot(actual, image)
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(actual, image)

    def test_microcontrast_is_deterministic_for_same_hdr_input(self):
        rng = np.random.default_rng(234)
        image = rng.random((36, 48, 3), dtype=np.float32) * np.float32(2.0)

        first = local_contrast_adapter.apply_microcontrast(image, 0.5)
        second = local_contrast_adapter.apply_microcontrast(image, 0.5)

        np.testing.assert_array_equal(second, first)

    def test_cores_local_contrast_is_compatibility_shim(self):
        image = np.linspace(0.0, 1.2, 18 * 20 * 3, dtype=np.float32).reshape(18, 20, 3)

        expected = local_contrast_adapter.apply_microcontrast(image, 0.4)
        actual = local_contrast_shim.apply_microcontrast(image, 0.4)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_effects_dispatch_to_adapter(self):
        image = np.ones((8, 10, 3), dtype=np.float32) * np.float32(0.5)
        expected = np.ones_like(image) * np.float32(0.75)
        param = {
            "switch_precence": True,
            "clarity": 30,
            "texture": 20,
            "microcontrast": 40,
        }
        efconfig = SimpleNamespace()

        cases = [
            (ClarityEffect(), "apply_clarity"),
            (TextureEffect(), "apply_texture"),
            (MicroContrastEffect(), "apply_microcontrast"),
        ]

        for effect, adapter_name in cases:
            with self.subTest(effect=effect.__class__.__name__):
                with mock.patch.object(local_contrast_adapter, adapter_name, return_value=expected) as patched:
                    actual = effect.make_diff(image, param, efconfig)

                self.assertIs(actual, expected)
                patched.assert_called_once()


class LocalContrastMetalBackendTest(unittest.TestCase):
    def _require_metal(self):
        previous = os.environ.get("PLATYPUS_LOCAL_CONTRAST_BACKEND")
        os.environ["PLATYPUS_LOCAL_CONTRAST_BACKEND"] = "metal"
        status = local_contrast_adapter.backend_status()
        if status.backend != "effect_backends._local_contrast_metal":
            if previous is None:
                os.environ.pop("PLATYPUS_LOCAL_CONTRAST_BACKEND", None)
            else:
                os.environ["PLATYPUS_LOCAL_CONTRAST_BACKEND"] = previous
            self.skipTest(f"Metal backend is unavailable: {status.detail}")
        return previous

    def _restore_backend(self, previous):
        if previous is None:
            os.environ.pop("PLATYPUS_LOCAL_CONTRAST_BACKEND", None)
        else:
            os.environ["PLATYPUS_LOCAL_CONTRAST_BACKEND"] = previous

    def test_metal_preserves_hdr_shape_and_finite_values(self):
        previous = self._require_metal()
        try:
            x = np.linspace(0.0, 1.0, 72, dtype=np.float32)
            y = np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, np.newaxis]
            image = np.empty((64, 72, 3), dtype=np.float32)
            image[..., 0] = 0.25 + 1.8 * x
            image[..., 1] = 0.15 + 1.6 * y
            image[..., 2] = 0.2 + 0.7 * (x + y)

            for apply_func, strength in (
                (local_contrast_adapter.apply_clarity, 0.35),
                (local_contrast_adapter.apply_texture, 0.25),
                (local_contrast_adapter.apply_microcontrast, 0.5),
            ):
                with self.subTest(effect=apply_func.__name__):
                    actual = apply_func(image, strength)

                    self.assertEqual(actual.shape, image.shape)
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertTrue(np.all(np.isfinite(actual)))
                    self.assertGreater(float(np.max(actual)), 1.0)
        finally:
            self._restore_backend(previous)

    def test_metal_approximates_reference_on_smooth_hdr_image(self):
        previous = self._require_metal()
        try:
            x = np.linspace(0.0, 1.0, 96, dtype=np.float32)
            y = np.linspace(0.0, 1.0, 80, dtype=np.float32)[:, np.newaxis]
            ripple = np.sin(x * np.float32(18.0))[np.newaxis, :] * np.float32(0.06)
            image = np.empty((80, 96, 3), dtype=np.float32)
            image[..., 0] = 0.15 + 1.6 * x + ripple
            image[..., 1] = 0.20 + 1.2 * y - ripple
            image[..., 2] = 0.10 + 0.9 * (x + y)

            cases = [
                (local_contrast_reference.apply_clarity, local_contrast_adapter.apply_clarity, 0.25, 0.03, 0.08),
                (local_contrast_reference.apply_texture, local_contrast_adapter.apply_texture, 0.25, 0.02, 0.06),
                (local_contrast_reference.apply_microcontrast, local_contrast_adapter.apply_microcontrast, 0.45, 0.04, 0.10),
            ]
            for ref_func, metal_func, strength, mean_gate, p99_gate in cases:
                with self.subTest(effect=metal_func.__name__):
                    expected = ref_func(image, strength)
                    actual = metal_func(image, strength)
                    delta = np.abs(actual - expected)

                    self.assertLess(float(np.mean(delta)), mean_gate)
                    self.assertLess(float(np.percentile(delta, 99)), p99_gate)
                    self.assertGreater(float(np.max(actual)), 1.0)
        finally:
            self._restore_backend(previous)


if __name__ == "__main__":
    unittest.main()
