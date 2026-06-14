import unittest

import numpy as np

from cores import colour_functions as cf


class ColourFunctionsRGBToRGBTest(unittest.TestCase):
    def test_matrix_rgb_to_rgb_matches_basis_conversion(self):
        matrix = cf.matrix_RGB_to_RGB("ProPhoto RGB", "sRGB", "Bradford")
        basis = cf.RGB_to_RGB(
            np.eye(3),
            "ProPhoto RGB",
            "sRGB",
            "Bradford",
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        )

        np.testing.assert_allclose(basis, matrix.T, rtol=1e-12, atol=1e-12)

    def test_legacy_apply_gamut_mapping_matches_explicit_two_step_call(self):
        rgb = np.array(
            [
                [0.6, 0.2, 0.1],
                [1.5, 0.5, -0.2],
                [-0.1, 0.3, 0.8],
            ],
            dtype=np.float64,
        )

        legacy = cf.RGB_to_RGB(
            rgb,
            "ProPhoto RGB",
            "sRGB",
            "Bradford",
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=True,
        )
        explicit = cf.RGB_to_RGB(
            rgb,
            "ProPhoto RGB",
            "sRGB",
            "Bradford",
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        )
        explicit = cf.apply_RGB_gamut_mapping(explicit)

        np.testing.assert_allclose(legacy, explicit, rtol=1e-12, atol=1e-12)

    def test_linear_colourspace_names_do_not_apply_cctf(self):
        rgb = np.array([[0.25, 0.5, 2.0]], dtype=np.float64)

        out = cf.RGB_to_RGB(
            rgb,
            "Linear sRGB",
            "Linear sRGB",
            apply_cctf_decoding=True,
            apply_cctf_encoding=True,
        )

        np.testing.assert_allclose(out, rgb, rtol=1e-12, atol=1e-12)

    def test_float32_image_stays_float32_through_rgb_to_rgb(self):
        rgb = np.array([[[0.25, 0.5, 2.0], [0.01, 0.02, 0.03]]], dtype=np.float32)

        out = cf.RGB_to_RGB(
            rgb,
            "ProPhoto RGB",
            "sRGB",
            "Bradford",
            apply_cctf_decoding=False,
            apply_cctf_encoding=True,
            apply_gamut_mapping=False,
        )

        self.assertEqual(out.dtype, np.float32)

    def test_non_array_input_keeps_float64_compatibility(self):
        out = cf.RGB_to_RGB(
            [[0.25, 0.5, 1.0]],
            "sRGB",
            "ProPhoto RGB",
            "Bradford",
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        )

        self.assertEqual(out.dtype, np.float64)

    def test_negative_display_gamut_compression_preserves_luminance(self):
        rgb = np.array(
            [
                [-0.02, 0.03, 0.04],
                [0.20, -0.04, 0.25],
                [1.50, 0.20, -0.10],
            ],
            dtype=np.float32,
        )
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        before_luma = rgb @ weights

        out = cf.compress_negative_display_gamut(rgb)
        after_luma = out @ weights

        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(out >= 0))
        np.testing.assert_allclose(after_luma, before_luma, rtol=1e-5, atol=1e-6)

    def test_negative_display_gamut_compression_leaves_positive_pixels_unchanged(self):
        rgb = np.array([[0.2, 0.3, 1.4]], dtype=np.float32)

        out = cf.compress_negative_display_gamut(rgb)

        np.testing.assert_array_equal(out, rgb)

    def test_encode_display_output_uses_colourspace_encoding(self):
        rgb = np.array([[0.25, 0.5, 1.0]], dtype=np.float32)

        srgb = cf.encode_display_output(rgb, "sRGB")
        adobe = cf.encode_display_output(rgb, "Adobe RGB")

        self.assertEqual(srgb.dtype, np.float32)
        self.assertEqual(adobe.dtype, np.float32)
        self.assertFalse(np.allclose(srgb, adobe))

    def test_encode_display_output_keeps_linear_colourspace_linear(self):
        rgb = np.array([[0.25, 0.5, 1.0]], dtype=np.float32)

        out = cf.encode_display_output(rgb, "Linear sRGB")

        np.testing.assert_array_equal(out, rgb)


if __name__ == "__main__":
    unittest.main()
