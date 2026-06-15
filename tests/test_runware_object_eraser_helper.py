import base64
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import runware_object_eraser_helper as runware


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class RunwareObjectEraserHelperTest(unittest.TestCase):
    def test_mask_data_uri_is_binary_png(self):
        mask = np.array([[0.0, 0.2, 1.0]], dtype=np.float32)

        data_uri = runware._mask_data_uri(mask)
        png = base64.b64decode(data_uri.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        np.testing.assert_array_equal(decoded[0], [0, 255, 255])

    def test_extract_result_image_decodes_base64_data(self):
        image = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        encoded = runware._image_data_uri(image).split(",", 1)[1]

        result = runware._extract_result_image({"data": [{"imageBase64Data": encoded}]})

        self.assertEqual((1, 1, 3), result.shape)
        np.testing.assert_allclose(result[0, 0], [1.0, 0.0, 0.0], atol=1 / 255)

    def test_predict_posts_object_eraser_payload(self):
        image = np.zeros((2, 2, 3), dtype=np.float32)
        mask = np.zeros((2, 2), dtype=np.float32)
        mask[0, 0] = 1.0
        encoded = runware._image_data_uri(image).split(",", 1)[1]
        response = FakeResponse({"data": [{"imageBase64Data": encoded, "cost": 0.01}]})

        with patch.object(runware.requests, "post", return_value=response) as post:
            result = runware.predict("test-key", image, mask, prompt="remove it")

        self.assertEqual((2, 2, 3), result.shape)
        args, kwargs = post.call_args
        self.assertEqual(runware.API_URL, args[0])
        self.assertEqual("Bearer test-key", kwargs["headers"]["Authorization"])
        task = kwargs["json"][0]
        self.assertEqual("imageInference", task["taskType"])
        self.assertEqual("sync", task["deliveryMethod"])
        self.assertEqual("base64Data", task["outputType"])
        self.assertEqual("runware:300@1", task["model"])
        self.assertIn("image", task["inputs"])
        self.assertIn("mask", task["inputs"])

    def test_default_prompt_is_short_object_removal_instruction(self):
        self.assertEqual(
            "Remove the masked unwanted object and naturally continue the surrounding background.",
            runware.DEFAULT_PROMPT,
        )

    def test_context_match_result_aligns_generated_context_to_original(self):
        original = np.full((64, 64, 3), 0.4, dtype=np.float32)
        result = np.full((64, 64, 3), 0.6, dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[24:40, 24:40] = 1.0

        matched = runware._context_match_result(result, original, mask, ring_px=16)

        self.assertLess(abs(float(matched[20, 20, 0]) - 0.4), abs(float(result[20, 20, 0]) - 0.4))

    def test_predict_without_key_returns_none(self):
        image = np.zeros((1, 1, 3), dtype=np.float32)
        mask = np.ones((1, 1), dtype=np.float32)

        self.assertIsNone(runware.predict(None, image, mask))


if __name__ == "__main__":
    unittest.main()
