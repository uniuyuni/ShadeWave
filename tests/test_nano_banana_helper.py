import os
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import nano_banana_helper


class NanoBananaHelperTest(unittest.TestCase):
    def test_default_model_uses_current_nano_banana_2(self):
        self.assertEqual("gemini-3.1-flash-image", nano_banana_helper.DEFAULT_MODEL)
        self.assertEqual("imagen-3.0-capability-001", nano_banana_helper.DEFAULT_EDIT_MODEL)
        self.assertIn("gemini-2.5-flash-image", nano_banana_helper.FALLBACK_MODELS)

    def test_mask_to_pil_uses_grayscale_mask(self):
        mask = np.array([[0.0, 1.0]], dtype=np.float32)

        pil = nano_banana_helper._mask_to_pil(mask)

        self.assertEqual("L", pil.mode)
        np.testing.assert_array_equal(np.asarray(pil), [[0, 255]])

    def test_build_prompt_mentions_red_mask(self):
        prompt = nano_banana_helper._build_prompt("continue the same wall texture")

        self.assertIn("pure red masked area", prompt)
        self.assertIn("#FF0000", prompt)
        self.assertIn("continue the same wall texture", prompt)

    def test_make_red_marker_image_fills_masked_pixels(self):
        image = np.zeros((1, 2, 3), dtype=np.float32)
        image[0, 0] = [0.2, 0.3, 0.4]
        image[0, 1] = [0.5, 0.6, 0.7]
        mask = np.array([[0.0, 1.0]], dtype=np.float32)

        marked = nano_banana_helper._make_red_marker_image(image, mask)

        np.testing.assert_allclose(marked[0, 0], [0.2, 0.3, 0.4])
        np.testing.assert_allclose(marked[0, 1], [1.0, 0.0, 0.0])

    def test_extract_image_supports_response_parts(self):
        image = Image.new("RGB", (2, 1), (255, 0, 0))
        part = SimpleNamespace(text=None, as_image=lambda: image)
        response = SimpleNamespace(parts=[part])

        result = nano_banana_helper._extract_image(response)

        self.assertEqual((2, 1), result.size)
        self.assertEqual("RGB", result.mode)

    def test_extract_image_supports_genai_image_from_as_image(self):
        image = Image.new("RGB", (2, 1), (0, 0, 255))
        genai_image = SimpleNamespace(_pil_image=image)
        part = SimpleNamespace(text=None, as_image=lambda: genai_image)
        response = SimpleNamespace(parts=[part])

        result = nano_banana_helper._extract_image(response)

        self.assertEqual((2, 1), result.size)
        self.assertEqual("RGB", result.mode)

    def test_predict_uses_generate_content_for_red_marker_flow(self):
        class FakeModels:
            def __init__(self):
                self.edit_called = False
                self.generate_called = False
                self.generate_kwargs = None

            def edit_image(self, **kwargs):
                self.edit_called = True
                raise AssertionError("edit_image should not be called")

            def generate_content(self, **kwargs):
                self.generate_called = True
                self.generate_kwargs = kwargs
                image = Image.new("RGB", (2, 2), (128, 128, 128))
                part = SimpleNamespace(text=None, as_image=lambda: image)
                return SimpleNamespace(parts=[part])

        client = SimpleNamespace(vertexai=False, models=FakeModels())
        fp32_image = np.zeros((2, 2, 3), dtype=np.float32)
        mask = np.ones((2, 2), dtype=np.float32)

        result = nano_banana_helper.predict(client, fp32_image, mask)

        self.assertFalse(client.models.edit_called)
        self.assertTrue(client.models.generate_called)
        self.assertEqual(2, len(client.models.generate_kwargs["contents"]))
        self.assertEqual((2, 2, 3), result.shape)

    def test_extract_edit_image_supports_generated_images(self):
        image = Image.new("RGB", (2, 1), (0, 255, 0))
        genai_image = SimpleNamespace(_pil_image=image)
        response = SimpleNamespace(generated_images=[SimpleNamespace(image=genai_image)])

        result = nano_banana_helper._extract_edit_image(response)

        self.assertEqual((2, 1), result.size)
        self.assertEqual("RGB", result.mode)

    def test_build_edit_prompt_can_include_user_prompt(self):
        prompt = nano_banana_helper._build_edit_prompt("continue the same wallpaper")

        self.assertIn("inside the mask", prompt)
        self.assertIn("continue the same wallpaper", prompt)

    def test_soft_edit_mask_keeps_far_pixels_zero(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[16, 16] = 1.0

        edit_mask = nano_banana_helper._soft_edit_mask(mask, dilate_px=4, blur_px=2)

        self.assertEqual((32, 32, 1), edit_mask.shape)
        self.assertGreater(float(edit_mask[16, 16, 0]), 0.8)
        self.assertEqual(0.0, float(edit_mask[0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
