import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores import core
import effects


class ColorSeparationEffectTest(unittest.TestCase):
    def test_zero_parameters_are_identity_object(self):
        image = np.random.default_rng(1).random((16, 12, 3), dtype=np.float32)

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=0.0,
            shadow_threshold=0.2,
            color_separation=0.0,
        )

        self.assertIs(out, image)

    def test_shadow_clean_reduces_shadow_chroma(self):
        image = np.array(
            [
                [[0.08, 0.055, 0.035], [0.30, 0.28, 0.26]],
                [[0.75, 0.50, 0.35], [1.00, 1.00, 0.00]],
            ],
            dtype=np.float32,
        )
        ycbcr_before = core.hlsrgb.linear_rgb_to_ycbcr(image)

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=1.0,
            shadow_threshold=0.2,
            color_separation=0.0,
        )
        ycbcr_after = core.hlsrgb.linear_rgb_to_ycbcr(out)

        chroma_before = np.linalg.norm(ycbcr_before[..., 1:3], axis=-1)
        chroma_after = np.linalg.norm(ycbcr_after[..., 1:3], axis=-1)

        self.assertLess(float(chroma_after[0, 0]), float(chroma_before[0, 0]))
        self.assertGreater(float(chroma_after[1, 1]), float(chroma_before[1, 1]) * 0.95)

    def test_color_separation_does_not_introduce_negative_values(self):
        image = np.array(
            [
                [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.2, 0.3, 0.8]],
            ],
            dtype=np.float32,
        )

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=0.0,
            shadow_threshold=0.2,
            color_separation=1.0,
        )

        self.assertGreaterEqual(float(np.min(out)), 0.0)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_color_density_increases_mid_chroma(self):
        image = np.array([[[0.45, 0.32, 0.22], [0.22, 0.32, 0.45]]], dtype=np.float32)
        ycbcr_before = core.hlsrgb.linear_rgb_to_ycbcr(image)

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=0.0,
            shadow_threshold=0.2,
            color_separation=0.0,
            color_density=1.0,
        )
        ycbcr_after = core.hlsrgb.linear_rgb_to_ycbcr(out)

        chroma_before = np.linalg.norm(ycbcr_before[..., 1:3], axis=-1)
        chroma_after = np.linalg.norm(ycbcr_after[..., 1:3], axis=-1)
        self.assertTrue(np.all(chroma_after > chroma_before))

    def test_negative_color_density_decreases_mid_chroma(self):
        image = np.array([[[0.45, 0.32, 0.22], [0.22, 0.32, 0.45]]], dtype=np.float32)
        ycbcr_before = core.hlsrgb.linear_rgb_to_ycbcr(image)

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=0.0,
            shadow_threshold=0.2,
            color_separation=0.0,
            color_density=-1.0,
        )
        ycbcr_after = core.hlsrgb.linear_rgb_to_ycbcr(out)

        chroma_before = np.linalg.norm(ycbcr_before[..., 1:3], axis=-1)
        chroma_after = np.linalg.norm(ycbcr_after[..., 1:3], axis=-1)
        self.assertTrue(np.all(chroma_after < chroma_before))

    def test_chroma_clarity_enhances_chroma_edges(self):
        image = np.zeros((24, 24, 3), dtype=np.float32)
        image[:, :12] = (0.45, 0.30, 0.20)
        image[:, 12:] = (0.20, 0.32, 0.45)
        ycbcr_before = core.hlsrgb.linear_rgb_to_ycbcr(image)

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=0.0,
            shadow_threshold=0.2,
            color_separation=0.0,
            chroma_clarity=1.0,
        )
        ycbcr_after = core.hlsrgb.linear_rgb_to_ycbcr(out)

        edge_before = abs(float(ycbcr_before[12, 11, 1] - ycbcr_before[12, 12, 1]))
        edge_after = abs(float(ycbcr_after[12, 11, 1] - ycbcr_after[12, 12, 1]))
        self.assertGreater(edge_after, edge_before)

    def test_negative_chroma_clarity_smooths_chroma_edges(self):
        image = np.zeros((24, 24, 3), dtype=np.float32)
        image[:, :12] = (0.45, 0.30, 0.20)
        image[:, 12:] = (0.20, 0.32, 0.45)
        ycbcr_before = core.hlsrgb.linear_rgb_to_ycbcr(image)

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=0.0,
            shadow_threshold=0.2,
            color_separation=0.0,
            chroma_clarity=-1.0,
        )
        ycbcr_after = core.hlsrgb.linear_rgb_to_ycbcr(out)

        edge_before = abs(float(ycbcr_before[12, 11, 1] - ycbcr_before[12, 12, 1]))
        edge_after = abs(float(ycbcr_after[12, 11, 1] - ycbcr_after[12, 12, 1]))
        self.assertLess(edge_after, edge_before)

    def test_subtractive_saturation_increases_chroma_with_density(self):
        image = np.array([[[0.55, 0.30, 0.16], [0.18, 0.32, 0.54]]], dtype=np.float32)
        ycbcr_before = core.hlsrgb.linear_rgb_to_ycbcr(image)
        luma_before = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]

        out = core.apply_color_separation(
            image,
            shadow_chroma_clean=0.0,
            shadow_threshold=0.2,
            color_separation=0.0,
            subtractive_saturation=1.0,
        )
        ycbcr_after = core.hlsrgb.linear_rgb_to_ycbcr(out)
        luma_after = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]

        chroma_before = np.linalg.norm(ycbcr_before[..., 1:3], axis=-1)
        chroma_after = np.linalg.norm(ycbcr_after[..., 1:3], axis=-1)
        self.assertTrue(np.all(chroma_after > chroma_before))
        self.assertTrue(np.all(luma_after < luma_before))

    def test_effect_normalizes_percent_sliders_for_core(self):
        image = np.array([[[0.55, 0.30, 0.16], [0.18, 0.32, 0.54]]], dtype=np.float32)
        effect = effects.ColorSeparationEffect()
        efconfig = effects.EffectConfig()
        param = {
            "switch_global": True,
            "shadow_chroma_clean": 100.0,
            "shadow_chroma_threshold": 0.2,
            "color_separation": 100.0,
            "chroma_clarity": -100.0,
            "color_density": 50.0,
            "subtractive_saturation": 25.0,
            "detail_tonemap": 0.0,
        }

        out = effect.make_diff(image, param, efconfig)
        expected = core.apply_color_separation(
            image,
            shadow_chroma_clean=1.0,
            shadow_threshold=0.2,
            color_separation=1.0,
            chroma_clarity=-1.0,
            color_density=0.5,
            subtractive_saturation=0.25,
        )

        np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_effect_normalizes_detail_tonemap_slider_for_core(self):
        image = np.array([[[0.55, 0.30, 0.16], [0.18, 0.32, 0.54]]], dtype=np.float32)
        effect = effects.ColorSeparationEffect()
        efconfig = effects.EffectConfig()
        param = {
            "switch_global": True,
            "shadow_chroma_clean": 0.0,
            "shadow_chroma_threshold": 0.2,
            "color_separation": 0.0,
            "chroma_clarity": 0.0,
            "color_density": 0.0,
            "subtractive_saturation": 0.0,
            "detail_tonemap": 50.0,
        }

        original = core.detail_preserving_tonemap
        calls = []

        def fake_detail_tonemap(img, strength=1.0):
            calls.append(strength)
            return img + strength

        try:
            core.detail_preserving_tonemap = fake_detail_tonemap
            out = effect.make_diff(image, param, efconfig)
        finally:
            core.detail_preserving_tonemap = original

        self.assertEqual(calls, [0.5])
        np.testing.assert_allclose(out, image + 0.5, rtol=1.0e-6, atol=1.0e-6)

    def test_effect_is_registered_in_lv2(self):
        effect_sets = effects.create_effects()

        self.assertIn("color_separation", effect_sets[2])


if __name__ == "__main__":
    unittest.main()
