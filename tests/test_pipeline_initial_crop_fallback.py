import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import params


class PipelineInitialCropFallbackTest(unittest.TestCase):
    def test_initial_crop_rect_is_created_only_when_missing(self):
        param = {"original_img_size": (120, 80), "img_size": (120, 80)}

        self.assertTrue(params.ensure_initial_crop_rect(param))
        self.assertEqual(params.get_crop_rect(param), (0, 20, 120, 100))

        param["crop_rect"] = (0.1, 0.2, 0.8, 0.9)
        self.assertFalse(params.ensure_initial_crop_rect(param))
        self.assertEqual(params.get_crop_rect(param), (12, 24, 96, 108))

    def test_initial_crop_rect_is_not_created_without_original_size(self):
        param = {}

        self.assertFalse(params.ensure_initial_crop_rect(param))
        self.assertIsNone(params.get_crop_rect(param))


if __name__ == "__main__":
    unittest.main()
