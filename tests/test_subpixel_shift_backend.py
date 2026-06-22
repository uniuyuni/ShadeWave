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

from effect_backends import subpixel_shift_adapter, subpixel_shift_reference
from effects import SubpixelShiftEffect


class SubpixelShiftBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = subpixel_shift_adapter.backend_status()

        self.assertEqual(status.effect, "subpixel_shift")
        self.assertIn(
            status.backend,
            {"effect_backends._subpixel_shift_cpu", "effect_backends.subpixel_shift_reference"},
        )

    @unittest.skipUnless(subpixel_shift_adapter.native_available(), "subpixel shift native backend is not built")
    def test_native_subpixel_shift_matches_reference(self):
        rng = np.random.default_rng(123)
        image = rng.random((17, 23, 3), dtype=np.float32)

        expected = subpixel_shift_reference.subpixel_shift(image, shift_x=0.35, shift_y=-0.65)
        actual = subpixel_shift_adapter.subpixel_shift(image, shift_x=0.35, shift_y=-0.65)

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    @unittest.skipUnless(subpixel_shift_adapter.native_available(), "subpixel shift native backend is not built")
    def test_native_enhanced_image_matches_reference(self):
        rng = np.random.default_rng(456)
        image = rng.random((19, 29, 3), dtype=np.float32)

        expected = subpixel_shift_reference.create_enhanced_image(image)
        actual = subpixel_shift_adapter.create_enhanced_image(image)

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_effect_dispatches_to_adapter(self):
        image = np.ones((6, 8, 3), dtype=np.float32)
        expected = np.ones_like(image) * np.float32(0.25)
        effect = SubpixelShiftEffect()
        param = {
            "switch_details": True,
            "subpixel_shift": True,
        }
        efconfig = SimpleNamespace(
            loading_flag=0,
            processor=None,
            effect_hash_snapshot=None,
            upstream_hash=1,
            effect_cache={},
        )

        with mock.patch.object(subpixel_shift_adapter, "create_enhanced_image", return_value=expected) as patched:
            actual = effect.make_diff(image, param, efconfig)

        self.assertIs(actual, expected)
        patched.assert_called_once_with(image)

    def test_native_effect_bypasses_async_worker_copy(self):
        image = np.ones((6, 8, 3), dtype=np.float32)
        expected = np.ones_like(image) * np.float32(0.5)
        effect = SubpixelShiftEffect()
        param = {
            "switch_details": True,
            "subpixel_shift": True,
        }
        efconfig = SimpleNamespace(
            loading_flag=0,
            processor=object(),
            effect_hash_snapshot=None,
            upstream_hash=1,
            effect_cache={},
        )

        with (
            mock.patch.object(subpixel_shift_adapter, "native_enabled", return_value=True),
            mock.patch.object(subpixel_shift_adapter, "create_enhanced_image", return_value=expected),
            mock.patch.object(effect, "try_async_execution") as async_mock,
        ):
            actual = effect.make_diff(image, param, efconfig)

        self.assertIs(actual, expected)
        async_mock.assert_not_called()

    def test_effect_recomputes_when_hash_matches_but_diff_was_cleared(self):
        image = np.ones((6, 8, 3), dtype=np.float32)
        expected = np.ones_like(image) * np.float32(0.75)
        effect = SubpixelShiftEffect()
        param = {
            "switch_details": True,
            "subpixel_shift": True,
        }
        efconfig = SimpleNamespace(
            loading_flag=0,
            processor=None,
            upstream_hash=7,
        )
        effect.hash = hash((hash((True)), efconfig.upstream_hash))
        effect.diff = None

        with mock.patch.object(subpixel_shift_adapter, "create_enhanced_image", return_value=expected) as patched:
            actual = effect.make_diff(image, param, efconfig)

        self.assertIs(actual, expected)
        patched.assert_called_once_with(image)

    def test_reference_can_be_forced(self):
        old_value = os.environ.get("PLATYPUS_SUBPIXEL_SHIFT_BACKEND")
        os.environ["PLATYPUS_SUBPIXEL_SHIFT_BACKEND"] = "reference"
        try:
            self.assertFalse(subpixel_shift_adapter.native_enabled())
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_SUBPIXEL_SHIFT_BACKEND", None)
            else:
                os.environ["PLATYPUS_SUBPIXEL_SHIFT_BACKEND"] = old_value


if __name__ == "__main__":
    unittest.main()
