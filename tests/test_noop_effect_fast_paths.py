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
from effect_backends import vignette_adapter


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

    def test_vignette_active_path_matches_backend_and_handles_missing_crop_rect(self):
        rng = np.random.default_rng(321)
        image = rng.random((16, 20, 3), dtype=np.float32) * np.float32(2.0)
        param = {
            "switch_vignette": True,
            "vignette_intensity": -35,
            "vignette_radius_percent": 82,
            "vignette_softness": 55,
            "original_img_size": (20, 16),
        }
        efconfig = SimpleNamespace(
            crop_editing=False,
            mode=effects.EffectMode.EXPORT,
            disp_info=(1.0, -2.0, 20.0, 16.0, 1.1),
        )

        actual = effects.VignetteEffect().make_diff(image, param, efconfig)
        expected = vignette_adapter.apply_vignette(
            image,
            -35,
            82,
            efconfig.disp_info,
            (0, 0, 20, 16),
            (0, 0),
            (100 - 55) / 100.0 * 3.0 + 1.0,
        )

        self.assertEqual(np.float32, actual.dtype)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_vignette_reuses_mask_without_reusing_stale_output(self):
        rng = np.random.default_rng(322)
        image_a = rng.random((16, 20, 3), dtype=np.float32)
        image_b = rng.random((16, 20, 3), dtype=np.float32) * np.float32(1.5)
        param = {
            "switch_vignette": True,
            "vignette_intensity": -35,
            "vignette_radius_percent": 82,
            "vignette_softness": 55,
            "original_img_size": (20, 16),
        }
        efconfig = SimpleNamespace(
            crop_editing=False,
            mode=effects.EffectMode.EXPORT,
            disp_info=(1.0, -2.0, 20.0, 16.0, 1.1),
        )
        effect = effects.VignetteEffect()

        first = effect.make_diff(image_a, param, efconfig)
        cached_mask = effect._vignette_mask
        second = effect.make_diff(image_b, param, efconfig)
        expected_second = vignette_adapter.apply_vignette_mask(image_b, cached_mask, -35)

        self.assertIs(effect._vignette_mask, cached_mask)
        self.assertFalse(np.allclose(first, second))
        np.testing.assert_allclose(second, expected_second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
