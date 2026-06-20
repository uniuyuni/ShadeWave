import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import effects
from async_worker import AsyncWorker
from enums import EffectMode, PipelineStatus
from effects import InpaintDiff, InpaintEffect, _ai_noise_blend_raw


ROOT = Path(__file__).resolve().parents[1]


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

    def test_inpaint_effect_uses_runware_helper(self):
        source = (ROOT / "effects.py").read_text(encoding="utf-8")

        self.assertIn("helpers.runware_object_eraser_helper", source)
        self.assertNotIn("import helpers.qwen_image_helper as qih", source)

    def test_inpaint_diff_reuses_cached_image_key(self):
        diff = InpaintDiff(type="mask", disp_info=(0, 0, 2, 2), image=np.ones((2, 2), dtype=np.float32))

        with patch("effects.np.ascontiguousarray", wraps=effects.np.ascontiguousarray) as wrapped:
            first = diff.image_key()
            second = diff.image_key()

        self.assertEqual(first, second)
        self.assertEqual(1, wrapped.call_count)

    def test_ai_noise_blend_skips_full_array_conversion_at_extreme_intensities(self):
        base = np.zeros((2, 2, 3), dtype=np.float32)
        raw = np.ones((2, 2, 3), dtype=np.float32)

        self.assertIs(base, _ai_noise_blend_raw(raw, base, 0))
        self.assertIs(raw, _ai_noise_blend_raw(raw, base, 100))

    def test_ai_noise_async_result_is_discarded_after_cache_hit(self):
        effect = effects.AINoiseReductonEffect()
        raw = np.ones((2, 2, 3), dtype=np.float32)
        discarded = []

        class Processor:
            def get_result(self, effect_name, param_hash):
                return {"status": "COMPLETE", "result": raw}

            def discard_result(self, effect_name, param_hash):
                discarded.append((effect_name, param_hash))

        efconfig = SimpleNamespace(
            processor=Processor(),
            mode=EffectMode.PREVIEW,
            upstream_hash="upstream",
            upstream_status=PipelineStatus.COMPLETE,
            layer_status=PipelineStatus.COMPLETE,
        )

        handled, result = effect.try_async_execution(raw, {}, efconfig, "param")

        self.assertTrue(handled)
        self.assertIs(result, raw)
        self.assertFalse(effect.keep_async_result)
        self.assertEqual(1, len(discarded))
        self.assertEqual("AINoiseReductonEffect", discarded[0][0])


if __name__ == "__main__":
    unittest.main()
