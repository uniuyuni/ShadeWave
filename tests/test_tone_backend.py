import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import tone_adapter, tone_reference
from effects import ToneEffect


class ToneBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = tone_adapter.backend_status()

        self.assertEqual(status.effect, "tone")
        self.assertIn(status.backend, {"effect_backends._tone_cpu", "effect_backends.tone_reference"})

    @unittest.skipUnless(tone_adapter.native_available(), "tone native backend is not built")
    def test_native_matches_reference_without_blur_path(self):
        rng = np.random.default_rng(123)
        image = rng.random((24, 32, 3), dtype=np.float32) * np.float32(2.0)

        expected = tone_reference.adjust_tone(
            image,
            highlights=35.0,
            shadows=22.0,
            midtone=-18.0,
            white_level=16.0,
            black_level=-12.0,
            resolution_scale=1.0,
        )
        actual = tone_adapter.adjust_tone(
            image,
            highlights=35.0,
            shadows=22.0,
            midtone=-18.0,
            white_level=16.0,
            black_level=-12.0,
            resolution_scale=1.0,
        )

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)

    @unittest.skipUnless(tone_adapter.native_available(), "tone native backend is not built")
    def test_native_matches_reference_with_negative_highlight_and_white(self):
        rng = np.random.default_rng(456)
        image = rng.random((28, 36, 3), dtype=np.float32) * np.float32(2.4)

        expected = tone_reference.adjust_tone(
            image,
            highlights=-52.0,
            shadows=-18.0,
            midtone=24.0,
            white_level=-38.0,
            black_level=20.0,
            resolution_scale=1.0,
        )
        actual = tone_adapter.adjust_tone(
            image,
            highlights=-52.0,
            shadows=-18.0,
            midtone=24.0,
            white_level=-38.0,
            black_level=20.0,
            resolution_scale=1.0,
        )

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)

    def test_tone_effect_dispatches_to_adapter(self):
        image = np.ones((6, 8, 3), dtype=np.float32) * np.float32(0.5)
        expected = np.ones_like(image) * np.float32(0.25)
        effect = ToneEffect()
        param = {
            "switch_tone": True,
            "shadow": 10,
            "highlight": 0,
            "midtone": 0,
            "white": 0,
            "black": 0,
        }
        efconfig = SimpleNamespace(disp_info=(0, 0, 8, 6, 1.0), resolution_scale=1.0)

        with mock.patch.object(tone_adapter, "adjust_tone", return_value=expected) as patched:
            actual = effect.make_diff(image, param, efconfig)

        self.assertIs(actual, expected)
        patched.assert_called_once()


if __name__ == "__main__":
    unittest.main()
