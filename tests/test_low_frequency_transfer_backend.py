import os
import pathlib
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import low_frequency_transfer_adapter, low_frequency_transfer_reference


class LowFrequencyTransferBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = low_frequency_transfer_adapter.backend_status()

        self.assertEqual(status.effect, "low_frequency_transfer")
        self.assertIn(
            status.backend,
            {
                "effect_backends._low_frequency_transfer_metal",
                "effect_backends._low_frequency_transfer_cpu",
                "effect_backends.low_frequency_transfer_reference",
            },
        )

    def test_reference_can_be_forced(self):
        old_value = os.environ.get("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND")
        os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND"] = "reference"
        try:
            self.assertFalse(low_frequency_transfer_adapter.native_enabled())
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND", None)
            else:
                os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND"] = old_value

    def test_reference_resizes_to_restored_shape(self):
        rng = np.random.default_rng(456)
        restored = rng.random((25, 31, 3), dtype=np.float32)
        reference = rng.random((13, 17, 3), dtype=np.float32)

        actual = low_frequency_transfer_adapter.apply_low_frequency_transfer(
            restored,
            reference,
            sigma=3,
            highlight_threshold=None,
        )

        self.assertEqual(actual.shape, restored.shape)
        self.assertEqual(actual.dtype, np.float32)

    def test_luminance_transfer_can_be_disabled_explicitly(self):
        restored = np.full((80, 120, 3), 0.2, dtype=np.float32)
        restored[:, 60:, :] = 0.8
        reference = restored.copy()
        reference[:, 60:, :] = 1.0

        actual = low_frequency_transfer_adapter.apply_low_frequency_transfer(
            restored,
            reference,
            sigma=12,
            highlight_threshold=None,
            luminance_transfer_strength=0.0,
        )
        default_transfer = low_frequency_transfer_adapter.apply_low_frequency_transfer(
            restored,
            reference,
            sigma=12,
            highlight_threshold=None,
        )

        self.assertLess(float(np.max(np.abs(actual - restored))), 1.0e-5)
        self.assertGreater(float(np.max(np.abs(default_transfer - restored))), 0.05)

    @unittest.skipUnless(low_frequency_transfer_adapter.native_available(), "low frequency transfer native backend is not built")
    def test_default_native_uses_exact_path(self):
        rng = np.random.default_rng(654)
        restored = rng.random((72, 96, 3), dtype=np.float32)
        reference = rng.random((72, 96, 3), dtype=np.float32)
        old_backend = os.environ.get("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND")
        old_downsample = os.environ.get("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE")
        os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND", None)
        os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE", None)
        try:
            default = low_frequency_transfer_adapter.apply_low_frequency_transfer(
                restored,
                reference,
                sigma=75,
                highlight_threshold=0.7,
                highlight_transition=0.4,
                highlight_detail_strength=0.2,
            )
            os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND"] = "exact"
            exact = low_frequency_transfer_adapter.apply_low_frequency_transfer(
                restored,
                reference,
                sigma=75,
                highlight_threshold=0.7,
                highlight_transition=0.4,
                highlight_detail_strength=0.2,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND", None)
            else:
                os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND"] = old_backend
            if old_downsample is None:
                os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE", None)
            else:
                os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE"] = old_downsample

        np.testing.assert_allclose(default, exact, rtol=0.0, atol=0.0)

    @unittest.skipUnless(
        low_frequency_transfer_adapter._cpu_backend is not None,
        "low frequency transfer lowres composer is not built",
    )
    def test_downsample_argument_uses_approximate_path(self):
        rng = np.random.default_rng(246)
        restored = rng.random((128, 160, 3), dtype=np.float32)
        reference = cv2.GaussianBlur(restored, (0, 0), 3)
        reference = np.asarray(reference + rng.normal(0.0, 0.05, restored.shape), dtype=np.float32)

        old_backend = os.environ.get("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND")
        old_downsample = os.environ.get("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE")
        try:
            os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND"] = "exact"
            os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE", None)
            exact = low_frequency_transfer_adapter.apply_low_frequency_transfer(
                restored,
                reference,
                sigma=32,
                highlight_threshold=0.7,
                highlight_transition=0.3,
                highlight_detail_strength=0.1,
            )

            os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND"] = "auto"
            os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE", None)
            approx = low_frequency_transfer_adapter.apply_low_frequency_transfer(
                restored,
                reference,
                sigma=32,
                highlight_threshold=0.7,
                highlight_transition=0.3,
                highlight_detail_strength=0.1,
                downsample=4,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND", None)
            else:
                os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND"] = old_backend
            if old_downsample is None:
                os.environ.pop("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE", None)
            else:
                os.environ["PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE"] = old_downsample

        self.assertEqual(approx.shape, restored.shape)
        self.assertEqual(approx.dtype, np.float32)
        self.assertGreater(float(np.max(np.abs(approx - exact))), 1.0e-5)
        self.assertLess(float(np.mean(np.abs(approx - exact))), 0.02)

    @unittest.skipUnless(low_frequency_transfer_adapter.native_available(), "low frequency transfer native backend is not built")
    def test_native_approximates_reference(self):
        rng = np.random.default_rng(789)
        restored = rng.random((96, 128, 3), dtype=np.float32)
        reference = cv2.GaussianBlur(restored, (0, 0), 3)
        reference = np.asarray(reference + rng.normal(0.0, 0.03, restored.shape), dtype=np.float32)

        expected = low_frequency_transfer_reference.apply_low_frequency_transfer(
            restored,
            reference,
            sigma=8,
            highlight_threshold=0.7,
            highlight_transition=0.4,
            highlight_detail_strength=0.2,
        )
        actual = low_frequency_transfer_adapter.apply_low_frequency_transfer(
            restored,
            reference,
            sigma=8,
            highlight_threshold=0.7,
            highlight_transition=0.4,
            highlight_detail_strength=0.2,
        )

        self.assertEqual(actual.dtype, np.float32)
        self.assertLess(float(np.mean(np.abs(actual - expected))), 0.006)
        self.assertLess(float(np.percentile(np.abs(actual - expected), 99)), 0.025)


if __name__ == "__main__":
    unittest.main()
