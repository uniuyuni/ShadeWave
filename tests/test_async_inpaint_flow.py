import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from async_worker import AsyncWorker
from effects import InpaintDiff, InpaintEffect


class AsyncInpaintFlowTest(unittest.TestCase):
    def test_rebuilds_full_mask_from_saved_inpaint_regions(self):
        effect = InpaintEffect()
        effect.inpaint_mask_list = [
            InpaintDiff(
                type="mask",
                disp_info=(1, 2, 2, 2),
                image=np.array([[0, 255], [128, 0]], dtype=np.uint8),
            )
        ]

        mask = effect._build_mask_from_inpaint_list((5, 5, 3))

        self.assertEqual((5, 5), mask.shape)
        self.assertEqual(1.0, float(mask[2, 2]))
        self.assertAlmostEqual(128.0 / 255.0, float(mask[3, 1]))
        self.assertEqual(0.0, float(mask[0, 0]))

    def test_sets_diff_list_from_async_result_crops(self):
        effect = InpaintEffect()
        effect.inpaint_mask_list = [
            InpaintDiff(type="mask", disp_info=(1, 1, 2, 2), image=np.ones((2, 2), dtype=np.float32))
        ]
        result = np.zeros((4, 4, 3), dtype=np.float32)
        result[1:3, 1:3] = [0.2, 0.4, 0.6]

        effect._set_diff_list_from_result(result)

        self.assertEqual(1, len(effect.inpaint_diff_list))
        self.assertEqual((1, 1, 2, 2), effect.inpaint_diff_list[0].disp_info)
        np.testing.assert_allclose(effect.inpaint_diff_list[0].image, result[1:3, 1:3])

    def test_async_worker_reports_pending_effect_by_name(self):
        worker = AsyncWorker.__new__(AsyncWorker)
        worker.active_effects = {3: "InpaintEffect", 4: "AINoiseReductonEffect"}

        self.assertTrue(worker.has_pending_effect("InpaintEffect"))
        self.assertFalse(worker.has_pending_effect("PatchmatchInpaintEffect"))


if __name__ == "__main__":
    unittest.main()
