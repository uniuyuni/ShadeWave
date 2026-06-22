import os
import sys
import unittest

import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cores.core as core
import params


class LiquifyCropOrderCoordinateFlowTest(unittest.TestCase):
    def _tcg_info_for_cropped_preview(self):
        original_size = (100, 60)
        crop_rect = (20, 10, 80, 50)
        texture_size = (100, 100)
        scale = min(
            texture_size[0] / (crop_rect[2] - crop_rect[0]),
            texture_size[1] / (crop_rect[3] - crop_rect[1]),
        )
        disp_info = core.convert_rect_to_info(crop_rect, scale)
        param = {
            "original_img_size": original_size,
            "img_size": original_size,
            "rotation": 0.0,
            "rotation2": 0.0,
            "flip_mode": 0,
            "matrix": np.eye(3),
        }
        params.set_crop_rect(param, crop_rect)
        params.set_disp_info(param, disp_info)
        return params.param_to_tcg_info(param), crop_rect, texture_size

    def test_direct_pre_crop_liquify_replay_needs_different_ref_mapping(self):
        tcg_info, crop_rect, texture_size = self._tcg_info_for_cropped_preview()
        disp_info = params.get_disp_info(tcg_info)
        draw_w, draw_h, offset_x, offset_y = core.crop_size_and_offset_from_texture(*texture_size, disp_info)
        self.assertEqual((draw_w, draw_h, offset_x, offset_y), (100, 66, 0, 17))

        crop_canvas = np.zeros((texture_size[1], texture_size[0], 3), np.float32)
        full_image = np.zeros((60, 100, 3), np.float32)

        def original_to_crop_canvas(x, y):
            dx, dy, _dw, _dh, scale = disp_info
            return offset_x + (x - dx) * scale, offset_y + (y - dy) * scale

        checks = [
            ("left", (crop_rect[0], 30.0)),
            ("right", (crop_rect[2], 30.0)),
            ("top", (50.0, crop_rect[1] + 1.0)),
            ("bottom", (50.0, crop_rect[3] - 1.0)),
        ]
        for _label, original_point in checks:
            crop_canvas_point = original_to_crop_canvas(*original_point)
            recorded_tcg = params.ref_image_to_tcg(
                *crop_canvas_point,
                crop_canvas,
                tcg_info,
                apply_disp_info=True,
            )
            crop_ref = params.tcg_to_ref_image(
                *recorded_tcg,
                crop_canvas,
                tcg_info,
                apply_disp_info=True,
                apply_ref_img_divide=True,
            )
            pre_crop_same_mapping = params.tcg_to_ref_image(
                *recorded_tcg,
                full_image,
                tcg_info,
                apply_disp_info=True,
                apply_ref_img_divide=True,
            )
            pre_crop_original_mapping = params.tcg_to_ref_image(
                *recorded_tcg,
                full_image,
                tcg_info,
                apply_disp_info=False,
                apply_ref_img_divide=False,
            )

            self.assertAlmostEqual(crop_ref[0], crop_canvas_point[0], places=4)
            self.assertAlmostEqual(crop_ref[1], crop_canvas_point[1], places=4)
            self.assertGreater(np.linalg.norm(np.subtract(pre_crop_same_mapping, original_point)), 10.0)
            self.assertAlmostEqual(pre_crop_original_mapping[0], original_point[0], delta=0.25)
            self.assertAlmostEqual(pre_crop_original_mapping[1], original_point[1], delta=0.25)


if __name__ == "__main__":
    unittest.main()
