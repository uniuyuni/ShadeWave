import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import effects
from async_worker import _worker_result_image
from async_worker import AsyncWorker
from async_worker import _task_params_for_worker
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

    def test_worker_uses_ai_noise_raw_only_for_ai_noise_effect(self):
        input_image = np.zeros((2, 2, 3), dtype=np.float32)
        ai_noise_raw = np.ones_like(input_image) * np.float32(0.25)
        exposure_result = np.ones_like(input_image) * np.float32(0.75)

        class TargetEffect:
            def apply_diff(self, image):
                return exposure_result

        params = {"ai_noise_reduction_result": ai_noise_raw}

        result = _worker_result_image(
            "ExposureFusionDebevecEffect",
            params,
            TargetEffect(),
            input_image,
            exposure_result,
        )

        self.assertIs(result, exposure_result)

        ai_result = _worker_result_image(
            "AINoiseReductonEffect",
            params,
            TargetEffect(),
            input_image,
            exposure_result,
        )

        np.testing.assert_array_equal(ai_result, ai_noise_raw)

    def test_worker_strips_ai_noise_payload_for_non_ai_effect_tasks(self):
        raw = np.ones((2, 2, 3), dtype=np.float32)
        params = {
            "ai_noise_reduction_result": raw,
            "ai_noise_reduction_content_key": "content",
            "_ai_noise_reduction_result_deferred": True,
            "cross_filter_strength": 0.5,
        }

        worker_params = _task_params_for_worker("CrossFilterEffect", params)

        self.assertNotIn("ai_noise_reduction_result", worker_params)
        self.assertNotIn("ai_noise_reduction_content_key", worker_params)
        self.assertNotIn("_ai_noise_reduction_result_deferred", worker_params)
        self.assertEqual(0.5, worker_params["cross_filter_strength"])
        self.assertIs(raw, params["ai_noise_reduction_result"])

    def test_worker_keeps_ai_noise_payload_for_ai_noise_tasks(self):
        params = {"ai_noise_reduction_result": np.ones((1, 1, 3), dtype=np.float32)}

        self.assertIs(params, _task_params_for_worker("AINoiseReductonEffect", params))

    def test_effect_config_does_not_carry_ai_job_manager(self):
        self.assertFalse(hasattr(effects.EffectConfig(), "ai_job_manager"))

    def test_bind_ai_job_manager_only_updates_ai_noise_effect(self):
        manager = object()
        effect_sets = effects.create_effects()

        effects.bind_ai_job_manager(effect_sets, manager)

        self.assertIs(manager, effect_sets[0]["ai_noise_reduction"].ai_job_manager)
        self.assertFalse(hasattr(effect_sets[0]["cross_filter"], "ai_job_manager"))

    def test_async_submit_failure_falls_back_to_sync_path(self):
        class Processor:
            def get_result(self, effect_name, param_hash):
                return None

            def submit_task(self, effect_name, img, param, efconfig, combined_hash):
                return None

        effect = effects.CrossFilterEffect()
        efconfig = SimpleNamespace(
            processor=Processor(),
            mode=EffectMode.PREVIEW,
            upstream_hash="upstream",
            upstream_status=PipelineStatus.COMPLETE,
            layer_status=PipelineStatus.COMPLETE,
        )

        handled, result = effect.try_async_execution(
            np.zeros((1, 1, 3), dtype=np.float32),
            {},
            efconfig,
            "param",
        )

        self.assertFalse(handled)
        self.assertIsNone(result)
        self.assertEqual("async_submit_failed", effect._last_cache_event)

    def test_has_pending_tasks_ignores_unreliable_queue_empty(self):
        class QueueThatLooksBusy:
            def empty(self):
                return False

        worker = AsyncWorker.__new__(AsyncWorker)
        worker.input_queue = QueueThatLooksBusy()
        worker.active_shms = set()
        worker.active_effects = {}

        self.assertFalse(worker.has_pending_tasks())

        worker.active_effects = {9: "CrossFilterEffect"}
        self.assertTrue(worker.has_pending_tasks())

    def test_has_pending_tasks_reaps_dead_worker_with_pending_task(self):
        class DeadProcess:
            exitcode = 9

            def is_alive(self):
                return False

        class FakeShm:
            def __init__(self):
                self.closed = False
                self.unlinked = False

            def close(self):
                self.closed = True

            def unlink(self):
                self.unlinked = True

        shm = FakeShm()
        worker = AsyncWorker.__new__(AsyncWorker)
        worker.thread_mode = False
        worker.process = DeadProcess()
        worker.input_queue = object()
        worker.active_shms = {(12, shm)}
        worker.active_effects = {12: "CrossFilterEffect"}
        worker.active_started_at = {12: 1.0}

        self.assertFalse(worker.has_pending_tasks())
        self.assertTrue(shm.closed)
        self.assertTrue(shm.unlinked)
        self.assertEqual({}, worker.active_effects)
        self.assertEqual(set(), worker.active_shms)
        self.assertIsNone(worker.process)

    def test_poll_results_does_not_trust_queue_empty(self):
        class QueueThatLooksEmpty:
            def __init__(self):
                self.items = [{"task_id": 7, "status": "error", "message": "done"}]

            def empty(self):
                return True

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                from queue import Empty
                raise Empty

        worker = AsyncWorker.__new__(AsyncWorker)
        worker.result_queue = QueueThatLooksEmpty()
        worker.active_effects = {7: "ExposureFusionDebevecEffect"}
        worker.active_started_at = {7: 1.0}
        worker.active_shms = set()

        self.assertEqual([(7, None, "done")], worker.poll_results())
        self.assertEqual({}, worker.active_effects)

    def test_poll_messages_does_not_trust_queue_empty(self):
        class QueueThatLooksEmpty:
            def __init__(self):
                self.items = [{"type": "waitinfo", "tag": "x", "text": "ready"}]

            def empty(self):
                return True

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                from queue import Empty
                raise Empty

        worker = AsyncWorker.__new__(AsyncWorker)
        worker.msg_queue = QueueThatLooksEmpty()

        self.assertEqual(
            [{"type": "waitinfo", "tag": "x", "text": "ready"}],
            list(worker.poll_messages()),
        )

    def test_cross_filter_is_configured_for_async_execution(self):
        effect = effects.CrossFilterEffect()

        self.assertEqual(effects.ExecutionMode.ASYNC, effect.execution_mode)

    def test_async_manager_clears_running_cache_when_worker_restart_cancels_tasks(self):
        import pipeline

        class Worker:
            def __init__(self):
                self.restart_count = 0
                self.submitted = []

            def restart(self):
                self.restart_count += 1

            def submit_task(self, effect_name, img, params, efconfig):
                task_id = len(self.submitted) + 10
                self.submitted.append((task_id, effect_name))
                return task_id

        worker = Worker()
        manager = pipeline.AsyncPipelineManager(worker)
        manager.cache = {
            ("ExposureFusionDebevecEffect", "old"): {
                "status": "RUNNING",
                "task_id": 1,
                "result": None,
            },
            ("SubpixelShiftEffect", "other"): {
                "status": "RUNNING",
                "task_id": 2,
                "result": None,
            },
        }

        manager.submit_task(
            "ExposureFusionDebevecEffect",
            np.zeros((1, 1, 3), dtype=np.float32),
            {},
            SimpleNamespace(),
            "new",
        )

        self.assertEqual(1, worker.restart_count)
        self.assertEqual(
            {
                ("ExposureFusionDebevecEffect", "new"): {
                    "status": "RUNNING",
                    "task_id": 10,
                    "result": None,
                }
            },
            manager.cache,
        )

    def test_async_manager_clears_all_running_cache_when_cancel_restarts_worker(self):
        import pipeline

        class Worker:
            def __init__(self):
                self.cancelled = []

            def cancel_effect(self, effect_name):
                self.cancelled.append(effect_name)

        worker = Worker()
        manager = pipeline.AsyncPipelineManager(worker)
        manager.cache = {
            ("ExposureFusionDebevecEffect", "exposure"): {
                "status": "RUNNING",
                "task_id": 1,
                "result": None,
            },
            ("SubpixelShiftEffect", "subpixel"): {
                "status": "RUNNING",
                "task_id": 2,
                "result": None,
            },
            ("CrossFilterEffect", "complete"): {
                "status": "COMPLETE",
                "task_id": 3,
                "result": object(),
            },
        }

        manager.cancel_effect("SubpixelShiftEffect")

        self.assertEqual(["SubpixelShiftEffect"], worker.cancelled)
        self.assertEqual(
            [("CrossFilterEffect", "complete")],
            list(manager.cache.keys()),
        )


if __name__ == "__main__":
    unittest.main()
