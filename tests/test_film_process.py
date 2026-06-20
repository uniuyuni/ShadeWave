import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores import film_process


def _chroma_amount(image):
    luma = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    return float(np.mean(np.abs(image[..., :3] - luma[..., np.newaxis])))


class FilmProcessTest(unittest.TestCase):
    def test_off_mode_returns_original_float_image(self):
        image = np.array([[[0.2, 0.4, 0.8], [1.2, 0.5, 0.1]]], dtype=np.float32)

        result = film_process.apply_film_process(image, {"film_mode": "Off"})

        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.array_equal(result, image))

    def test_process_returns_finite_float_rgb(self):
        image = np.array(
            [[[0.0, 0.2, 0.5], [1.0, 2.0, 4.0]], [[np.nan, np.inf, -1.0], [0.8, 0.6, 0.2]]],
            dtype=np.float32,
        )

        result = film_process.apply_film_process(image, {
            "film_mode": "Negative",
            "film_latitude": 65,
            "film_contrast": 55,
            "film_color_bias": 10,
            "film_dye_purity": 70,
            "film_layer_crosstalk": 25,
            "film_halation": 20,
            "film_aging": 10,
        })

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_dye_purity_preserves_more_chroma(self):
        image = np.array(
            [[[1.0, 0.05, 0.05], [0.05, 1.0, 0.05], [0.05, 0.05, 1.0]]],
            dtype=np.float32,
        )

        low = film_process.apply_film_process(image, {"film_mode": "Negative", "film_dye_purity": 0})
        high = film_process.apply_film_process(image, {"film_mode": "Negative", "film_dye_purity": 100})

        self.assertGreater(_chroma_amount(high), _chroma_amount(low))

    def test_layer_crosstalk_mixes_color_layers(self):
        image = np.array(
            [[[1.0, 0.02, 0.02], [0.02, 1.0, 0.02], [0.02, 0.02, 1.0]]],
            dtype=np.float32,
        )

        clean = film_process.apply_film_process(
            image,
            {"film_mode": "Negative", "film_layer_crosstalk": 0, "film_dye_purity": 100},
        )
        mixed = film_process.apply_film_process(
            image,
            {"film_mode": "Negative", "film_layer_crosstalk": 100, "film_dye_purity": 100},
        )

        self.assertLess(_chroma_amount(mixed), _chroma_amount(clean))

    def test_color_drift_creates_tonal_color_turn(self):
        image = np.array(
            [[[0.10, 0.10, 0.10], [0.50, 0.50, 0.50], [0.95, 0.95, 0.95]]],
            dtype=np.float32,
        )

        neutral = film_process.apply_film_process(image, {"film_mode": "Negative", "film_color_drift": 0})
        drifted = film_process.apply_film_process(image, {"film_mode": "Negative", "film_color_drift": 100})

        shadow_delta = drifted[0, 0] - neutral[0, 0]
        highlight_delta = drifted[0, 2] - neutral[0, 2]
        self.assertGreater(float(shadow_delta[2] - shadow_delta[0]), 0.01)
        self.assertGreater(float(highlight_delta[0] - highlight_delta[2]), 0.01)

    def test_halation_affects_bright_neighbors(self):
        image = np.zeros((31, 31, 3), dtype=np.float32)
        image[15, 15] = 4.0

        no_halo = film_process.apply_film_process(image, {"film_mode": "Negative", "film_halation": 0})
        halo = film_process.apply_film_process(image, {"film_mode": "Negative", "film_halation": 100})

        self.assertGreater(float(halo[15, 16, 0]), float(no_halo[15, 16, 0]))

    def test_black_and_white_mode_outputs_neutral_channels(self):
        image = np.array([[[1.0, 0.1, 0.05], [0.2, 0.8, 0.1]]], dtype=np.float32)

        result = film_process.apply_film_process(image, {"film_mode": "B&W"})

        self.assertTrue(np.allclose(result[..., 0], result[..., 1], atol=1e-6))
        self.assertTrue(np.allclose(result[..., 1], result[..., 2], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
