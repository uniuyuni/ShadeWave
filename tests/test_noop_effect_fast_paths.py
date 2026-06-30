import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import effects
from cores import core


class NoopEffectFastPathTest(unittest.TestCase):
    def test_vignette_zero_intensity_skips_diff_even_with_default_radius(self):
        image = np.full((8, 10, 3), 1.8, dtype=np.float32)
        param = {
            "switch_vignette": True,
            "vignette_intensity": 0,
            "vignette_radius_percent": 80,
            "vignette_softness": 80,
        }
        efconfig = SimpleNamespace(crop_editing=False)

        effect = effects.VignetteEffect()

        self.assertIsNone(effect.make_diff(image, param, efconfig))
        self.assertIsNone(effect.diff)
        self.assertIsNone(effect.hash)

    def test_glow_zero_opacity_skips_diff_even_when_source_controls_are_nonzero(self):
        rng = np.random.default_rng(123)
        image = rng.random((8, 10, 3), dtype=np.float32) * np.float32(2.5)
        param = {
            "switch_glow_effect": True,
            "glow_black": 30,
            "glow_gauss": 40,
            "glow_opacity": 0,
        }
        efconfig = SimpleNamespace(resolution_scale=1.0)

        effect = effects.GlowEffect()

        self.assertIsNone(effect.make_diff(image, param, efconfig))
        self.assertIsNone(effect.diff)
        self.assertIsNone(effect.hash)

    def test_solid_color_blends_hdr_linearly_without_clipping(self):
        image = np.array(
            [
                [[0.25, 1.5, 2.25], [3.0, 0.5, 1.25]],
                [[1.0, 2.0, 4.0], [0.0, 0.75, 1.5]],
            ],
            dtype=np.float32,
        )
        color = (0.2, 1.4, 2.6)
        opacity = 0.35

        actual = core.apply_solid_color(image, color, opacity)
        expected = image * np.float32(1.0 - opacity) + np.asarray(color, dtype=np.float32) * np.float32(opacity)

        self.assertEqual(np.float32, actual.dtype)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        self.assertGreater(float(actual.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
