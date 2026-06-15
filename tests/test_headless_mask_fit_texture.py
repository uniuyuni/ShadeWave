import inspect
import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import params
from cores.mask2 import headless_masks


class DummyContext:
    def __init__(self, texture_size, original_img_size, disp_info):
        self.texture_size = texture_size
        self.tcg_info = {"original_img_size": original_img_size}
        params.set_disp_info(self.tcg_info, disp_info)


class HeadlessMaskFitTextureTest(unittest.TestCase):
    def test_fit_image_mask_to_texture_handles_out_of_bounds_crop(self):
        ctx = DummyContext(
            texture_size=(80, 60),
            original_img_size=(100, 50),
            disp_info=(-20, -10, 140, 90, 1.0),
        )
        image = np.ones((50, 100), dtype=np.float32)

        fitted = headless_masks._fit_image_mask_to_texture(ctx, image)

        self.assertEqual((60, 80), fitted.shape)
        self.assertEqual(np.float32, fitted.dtype)
        self.assertTrue(np.isfinite(fitted).all())
        self.assertGreater(float(fitted.max()), 0.0)

    def test_depth_map_headless_uses_safe_texture_fit(self):
        source = inspect.getsource(headless_masks.HeadlessDepthMapMask.get_mask_image)

        self.assertIn("_fit_image_mask_to_texture(self.ctx, depth_map_mask)", source)
        self.assertNotIn("np.pad", source)
        self.assertNotIn("crop_image_with_disp_info", source)

    def test_headless_inference_masks_no_longer_use_manual_pad(self):
        for cls in (
            headless_masks.HeadlessSegmentMask,
            headless_masks.HeadlessFaceMask,
            headless_masks.HeadlessTargetTextMask,
        ):
            with self.subTest(mask=cls.__name__):
                source = inspect.getsource(cls.get_mask_image)

                self.assertIn("_fit_image_mask_to_texture(self.ctx,", source)
                self.assertNotIn("np.pad", source)


if __name__ == "__main__":
    unittest.main()
