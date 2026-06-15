import base64
import os
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import qwen_image_helper


class QwenImageHelperTest(unittest.TestCase):
    def test_numpy_to_base64_png_uses_8bit_rgb(self):
        image = np.array([[[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]]], dtype=np.float32)

        encoded = qwen_image_helper.numpy_to_base64_png(image)
        png = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        decoded = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)

        self.assertEqual(np.uint8, decoded.dtype)
        self.assertEqual((1, 2, 3), decoded.shape)
        np.testing.assert_array_equal(decoded[0, 0], [0, 0, 255])

    def test_extract_image_urls_from_all_choices(self):
        output = {
            "results": [
                {"url": "https://example.test/result.png"},
            ],
            "choices": [
                {"message": {"content": [{"image": "https://example.test/a.png"}]}},
                {"message": {"content": [{"text": "note"}, {"image": "https://example.test/b.png"}]}},
            ]
        }

        self.assertEqual(
            [
                "https://example.test/result.png",
                "https://example.test/a.png",
                "https://example.test/b.png",
            ],
            qwen_image_helper._extract_image_urls(output),
        )

    def test_extract_image_urls_from_image_synthesis_output(self):
        output = SimpleNamespace(results=[SimpleNamespace(url="https://example.test/masked.png")])

        self.assertEqual(
            ["https://example.test/masked.png"],
            qwen_image_helper._extract_image_urls(output),
        )

    def test_extract_image_urls_handles_dashscope_dict_mixin_missing_results(self):
        class DashScopeLike(dict):
            def __getattr__(self, attr):
                return self[attr]

        output = DashScopeLike({
            "choices": [
                DashScopeLike({
                    "message": DashScopeLike({
                        "content": [
                            DashScopeLike({"image": "https://example.test/fallback.png"})
                        ]
                    })
                })
            ]
        })

        self.assertEqual(
            ["https://example.test/fallback.png"],
            qwen_image_helper._extract_image_urls(output),
        )

    def test_mask_to_base64_png_uses_8bit_grayscale(self):
        mask = np.array([[0.0, 1.0]], dtype=np.float32)

        encoded = qwen_image_helper.mask_to_base64_png(mask)
        png = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        decoded = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)

        self.assertEqual(np.uint8, decoded.dtype)
        self.assertEqual((1, 2), decoded.shape)
        np.testing.assert_array_equal(decoded[0], [0, 255])

    def test_make_mask_marker_image_uses_solid_red_by_default(self):
        image = np.zeros((1, 2, 3), dtype=np.float32)
        image[0, 0] = [0.2, 0.3, 0.4]
        image[0, 1] = [0.5, 0.6, 0.7]
        mask = np.array([[0.0, 1.0]], dtype=np.float32)

        marked = qwen_image_helper._make_mask_marker_image(image, mask)

        np.testing.assert_allclose(marked[0, 0], [0.2, 0.3, 0.4])
        np.testing.assert_allclose(marked[0, 1], [1.0, 0.0, 0.0])

    def test_soft_edit_mask_expands_mask_but_keeps_far_pixels_zero(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32, 32] = 1.0

        edit_mask = qwen_image_helper._soft_edit_mask(mask, dilate_px=8, blur_px=3)

        self.assertEqual((64, 64, 1), edit_mask.shape)
        self.assertGreater(float(edit_mask[32, 32, 0]), 0.9)
        self.assertEqual(0.0, float(edit_mask[0, 0, 0]))

    def test_predict_helper_keeps_original_pixels_when_prediction_fails(self):
        image = np.zeros((1024, 1024, 3), dtype=np.float32)
        image[..., 0] = 0.25
        image[..., 1] = 0.5
        image[..., 2] = 0.75
        mask = np.zeros((1024, 1024), dtype=np.float32)
        mask[448:576, 448:576] = 1.0

        result = qwen_image_helper.predict_helper(
            image.copy(),
            mask,
            (448, 448, 128, 128),
            lambda _image, _mask: None,
        )

        np.testing.assert_allclose(result, image)


if __name__ == "__main__":
    unittest.main()
