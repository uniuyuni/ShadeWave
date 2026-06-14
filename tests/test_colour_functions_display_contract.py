import unittest
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import colour_functions_reference as cf


def _display_encode_prophoto_to_srgb(rgb):
    linear = cf.RGB_to_RGB(
        rgb,
        "ProPhoto RGB",
        "sRGB",
        "CAT16",
        apply_cctf_decoding=False,
        apply_cctf_encoding=False,
        apply_gamut_mapping=False,
    )
    compressed = cf.compress_negative_display_gamut(linear)
    return cf.encode_display_output(compressed, "sRGB")


class ColourFunctionsDisplayContractTest(unittest.TestCase):
    def test_srgb_output_encoding_contract(self):
        linear = np.array([-0.01, 0.0, 0.0031308, 0.18, 1.0, 2.0], dtype=np.float32)

        encoded = cf.linear_to_sRGB(linear)

        expected = np.array(
            [-0.1292, 0.0, 0.040449936, 0.4613561, 0.99999994, 1.353256],
            dtype=np.float32,
        )
        np.testing.assert_allclose(encoded, expected, rtol=0.0, atol=2e-7)

    def test_prophoto_to_srgb_cat16_display_contract_known_values(self):
        samples = np.array(
            [
                [-0.04, 0.02, 0.10],
                [0.0, 0.0, 0.0],
                [0.003, 0.010, 0.050],
                [0.18, 0.18, 0.18],
                [0.50, 0.25, 0.10],
                [1.00, 1.00, 1.00],
                [1.70, 0.80, 0.20],
            ],
            dtype=np.float32,
        )

        encoded = _display_encode_prophoto_to_srgb(samples)

        expected = np.array(
            [
                [0.0, 0.042930827, 0.06095542],
                [0.0, 0.0, 0.0],
                [0.0, 0.091661565, 0.16421070],
                [0.46134216, 0.46136785, 0.46128303],
                [0.91485894, 0.48351353, 0.30621287],
                [0.99997145, 1.00002380, 0.99985070],
                [1.57971990, 0.80839425, 0.36001572],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(encoded, expected, rtol=0.0, atol=2e-6)

    def test_prophoto_to_srgb_cat16_matrix_contract(self):
        matrix = cf.matrix_RGB_to_RGB("ProPhoto RGB", "sRGB", "CAT16")

        expected = np.array(
            [
                [2.0795340327, -0.7644689998, -0.3151299751],
                [-0.2177368010, 1.2412457245, -0.0234545351],
                [-0.0106919237, -0.1289087891, 1.1392610874],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=5e-10)

    def test_neutral_ramp_remains_monotonic_after_display_transform(self):
        ramp = np.linspace(0.0, 4.0, 2048, dtype=np.float32)
        rgb = np.repeat(ramp[:, None], 3, axis=1)

        encoded = _display_encode_prophoto_to_srgb(rgb)

        self.assertEqual(encoded.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(encoded)))
        for channel in range(3):
            deltas = np.diff(encoded[:, channel])
            self.assertGreaterEqual(float(np.min(deltas)), -2e-6)

    def test_display_contract_preserves_hdr_values_above_one(self):
        rgb = np.array([[1.7, 0.8, 0.2]], dtype=np.float32)

        encoded = _display_encode_prophoto_to_srgb(rgb)

        self.assertGreater(float(encoded[0, 0]), 1.0)
        self.assertGreater(float(np.max(encoded)), 1.0)


if __name__ == "__main__":
    unittest.main()
