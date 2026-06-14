import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import colour_functions_reference as ref
from effect_backends import colour_functions_adapter


class ColourFunctionsBackendTest(unittest.TestCase):
    def setUp(self):
        if not colour_functions_adapter.native_available():
            self.skipTest("native colour functions backend is not built")

    def test_native_matches_reference_display_transform(self):
        rng = np.random.default_rng(123)
        image = rng.normal(0.25, 0.55, (48, 64, 3)).astype(np.float32)
        image[:8] *= 3.0
        image[8:16] -= 0.35
        basis = ref.display_color_transform_basis("ProPhoto RGB", "sRGB", "CAT16")

        native = colour_functions_adapter.apply_display_color_transform(
            image,
            basis,
            "sRGB",
        )
        expected = ref.apply_display_color_transform(image, basis, "sRGB")

        self.assertEqual(native.dtype, np.float32)
        np.testing.assert_allclose(native, expected, rtol=3e-5, atol=3e-5)

    def test_native_matches_reference_supported_display_encodings(self):
        image = np.array(
            [
                [[0.0, 0.001, 0.01], [0.18, 0.5, 1.0]],
                [[1.5, 0.25, 0.05], [3.0, 1.2, 0.4]],
            ],
            dtype=np.float32,
        )
        basis = np.eye(3, dtype=np.float32)
        colourspaces = [
            "Linear sRGB",
            "sRGB",
            "Display P3",
            "Rec.709",
            "Rec.2020",
            "Adobe RGB (1998)",
            "DCI-P3",
            "ProPhoto RGB",
        ]

        for colourspace in colourspaces:
            with self.subTest(colourspace=colourspace):
                native = colour_functions_adapter.apply_display_color_transform(
                    image,
                    basis,
                    colourspace,
                )
                expected = ref.apply_display_color_transform(image, basis, colourspace)
                np.testing.assert_allclose(native, expected, rtol=3e-5, atol=3e-5)

    def test_facade_dispatches_to_native_backend(self):
        image = np.ones((4, 5, 3), dtype=np.float32) * 0.18
        basis = ref.display_color_transform_basis("ProPhoto RGB", "sRGB", "CAT16")

        with mock.patch.object(colour_functions_adapter, "native_enabled", return_value=True):
            with mock.patch.object(
                colour_functions_adapter,
                "apply_display_color_transform",
                wraps=colour_functions_adapter.apply_display_color_transform,
            ) as spy:
                out = colour_functions_adapter.apply_display_color_transform(image, basis, "sRGB")

        self.assertEqual(out.shape, image.shape)
        self.assertTrue(spy.called)

    def test_facade_keeps_reference_fallback_for_non_image_shapes(self):
        samples = np.array([[0.18, 0.18, 0.18], [1.0, 0.5, 0.1]], dtype=np.float32)
        basis = ref.display_color_transform_basis("ProPhoto RGB", "sRGB", "CAT16")

        out = colour_functions_adapter.apply_display_color_transform(samples, basis, "sRGB")
        expected = ref.apply_display_color_transform(samples, basis, "sRGB")

        np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
