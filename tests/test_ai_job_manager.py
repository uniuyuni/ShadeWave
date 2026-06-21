import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgpack
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import effects
from cores.ai_job_manager import (
    AI_NOISE_KIND,
    AIJob,
    AIJobManager,
    AIJobStatus,
    ai_noise_content_key,
    ai_noise_source_signature,
    ai_noise_valid_content_keys,
    current_param_accepts_ai_noise_result,
    merge_ai_noise_result_into_param,
    merge_ai_noise_result_into_pmck,
)
from enums import EffectMode, ImageFidelity, PipelineStatus


class AIJobManagerTest(unittest.TestCase):
    def test_ai_noise_content_key_is_stable_for_same_file_signature(self):
        image = np.zeros((4, 5, 3), dtype=np.float32)
        param = {"ai_noise_reduction": True, "image_fidelity": ImageFidelity.FULL.value}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.jpg")
            pathlib.Path(path).write_bytes(b"image")
            sig1 = ai_noise_source_signature(path, image, param)
            key1 = ai_noise_content_key(path, image, param, source_signature=sig1)
            sig2 = ai_noise_source_signature(path, image.copy(), param.copy())
            key2 = ai_noise_content_key(path, image.copy(), param.copy(), source_signature=sig2)

        self.assertEqual(sig1, sig2)
        self.assertEqual(key1, key2)

    def test_ai_noise_content_key_ignores_fidelity_when_pixels_match(self):
        image = np.zeros((4, 5, 3), dtype=np.float32)
        param = {"ai_noise_reduction": True, "image_fidelity": ImageFidelity.PREVIEW.value}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.jpg")
            pathlib.Path(path).write_bytes(b"image")
            preview_sig = ai_noise_source_signature(path, image, param)
            preview_key = ai_noise_content_key(path, image, param, source_signature=preview_sig)
            param["image_fidelity"] = ImageFidelity.FULL.value
            full_sig = ai_noise_source_signature(path, image, param)
            full_key = ai_noise_content_key(path, image, param, source_signature=full_sig)

        self.assertEqual(preview_sig, full_sig)
        self.assertEqual(preview_key, full_key)

    def test_ai_noise_content_key_changes_when_input_pixels_shift(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[1:3, 1:3] = 1.0
        shifted = np.zeros_like(image)
        shifted[2:4, 2:4] = 1.0
        param = {"ai_noise_reduction": True, "image_fidelity": ImageFidelity.FULL.value}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.jpg")
            pathlib.Path(path).write_bytes(b"image")
            sig1 = ai_noise_source_signature(path, image, param)
            key1 = ai_noise_content_key(path, image, param, source_signature=sig1)
            sig2 = ai_noise_source_signature(path, shifted, param)
            key2 = ai_noise_content_key(path, shifted, param, source_signature=sig2)

        self.assertNotEqual(sig1, sig2)
        self.assertNotEqual(key1, key2)

    def test_current_param_accepts_only_matching_result(self):
        image = np.zeros((4, 5, 3), dtype=np.float32)
        param = {
            "switch_ai_noise_reduction": True,
            "ai_noise_reduction": True,
            "image_fidelity": ImageFidelity.FULL.value,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.jpg")
            pathlib.Path(path).write_bytes(b"image")
            sig = ai_noise_source_signature(path, image, param)
            key = ai_noise_content_key(path, image, param, source_signature=sig)

            self.assertTrue(
                current_param_accepts_ai_noise_result(
                    param,
                    file_path=path,
                    image=image,
                    content_key=key,
                    source_signature=sig,
                )
            )

            param["ai_noise_reduction"] = False
            self.assertFalse(
                current_param_accepts_ai_noise_result(
                    param,
                    file_path=path,
                    image=image,
                    content_key=key,
                    source_signature=sig,
                )
            )

    def test_merge_current_param_keeps_raw_and_keys(self):
        raw = np.ones((2, 2, 3), dtype=np.float32)
        param = {}

        merge_ai_noise_result_into_param(param, raw, "content", "source")

        np.testing.assert_array_equal(param["ai_noise_reduction_result"], raw)
        self.assertEqual("content", param["ai_noise_reduction_content_key"])
        self.assertEqual("source", param["ai_noise_reduction_source_signature"])

    def test_pmck_merge_rejects_stale_or_disabled_targets(self):
        raw = np.ones((2, 2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.jpg")
            pmck_path = image_path + ".pmck"
            data = {
                "make": "Platypus",
                "primary_param": {
                    "switch_ai_noise_reduction": True,
                    "ai_noise_reduction": True,
                    "ai_noise_reduction_content_key": "old",
                },
            }
            pathlib.Path(pmck_path).write_bytes(msgpack.packb(data, use_bin_type=True))

            sig = ai_noise_source_signature(image_path, raw, data["primary_param"])
            key = ai_noise_content_key(image_path, raw, data["primary_param"], source_signature=sig)
            self.assertIn(key, ai_noise_valid_content_keys(image_path, raw, data["primary_param"]))
            pathlib.Path(pmck_path).write_bytes(msgpack.packb(data, use_bin_type=True))
            self.assertTrue(
                merge_ai_noise_result_into_pmck(
                    image_path,
                    raw,
                    content_key=key,
                    source_signature=sig,
                )
            )
            loaded = msgpack.unpackb(pathlib.Path(pmck_path).read_bytes(), raw=False)
            pp = loaded["primary_param"]
            self.assertEqual(key, pp["ai_noise_reduction_content_key"])
            self.assertEqual(sig, pp["ai_noise_reduction_source_signature"])
            self.assertIn("ai_noise_reduction_result", pp)

            pp["ai_noise_reduction"] = False
            pathlib.Path(pmck_path).write_bytes(msgpack.packb(loaded, use_bin_type=True))
            self.assertFalse(
                merge_ai_noise_result_into_pmck(
                    image_path,
                    raw,
                    content_key=key,
                    source_signature=sig,
                )
            )

    def test_manager_replaces_queued_job_for_same_file(self):
        states = []
        mgr = AIJobManager(viewer_state_callback=lambda path, state: states.append((path, state)))
        mgr.start = lambda: None
        image = np.zeros((2, 3, 3), dtype=np.float32)

        class FakeSharedMemory:
            counter = 0

            def __init__(self, create=False, size=0, name=None):
                type(self).counter += 1
                self.name = name or f"fake-{type(self).counter}"
                self.buf = bytearray(size)

            def close(self):
                pass

            def unlink(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.jpg")
            pathlib.Path(path).write_bytes(b"image")
            with patch("cores.ai_job_manager.manager.shared_memory.SharedMemory", FakeSharedMemory):
                job1 = mgr.enqueue_image_job(
                    kind=AI_NOISE_KIND,
                    file_path=path,
                    image=image,
                    param_snapshot={"ai_noise_reduction": True},
                    content_key="one",
                    source_signature="one-source",
                )
                job2 = mgr.enqueue_image_job(
                    kind=AI_NOISE_KIND,
                    file_path=path,
                    image=image,
                    param_snapshot={"ai_noise_reduction": True},
                    content_key="two",
                    source_signature="two-source",
                )

        self.assertNotEqual(job1.job_id, job2.job_id)
        self.assertEqual(job2.job_id, mgr.job_by_target[(AI_NOISE_KIND, path)])
        self.assertIn((path, "queued"), states)
        mgr.stop()

    def test_foreground_job_runs_before_waiting_background_jobs_after_current_finishes(self):
        class ListQueue:
            def __init__(self, items=None):
                self.items = list(items or [])

            def put(self, item):
                self.items.append(item)

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                from queue import Empty

                raise Empty

        class FakeSharedMemory:
            counter = 0

            def __init__(self, create=False, size=0, name=None):
                type(self).counter += 1
                self.name = name or f"fake-{type(self).counter}"
                self.buf = bytearray(size)

            def close(self):
                pass

            def unlink(self):
                pass

        mgr = AIJobManager()
        mgr.start = lambda: None
        mgr.input_queue = ListQueue()
        bg1 = mgr.enqueue_ai_noise_file("/tmp/bg1.jpg", {"ai_noise_reduction": True})
        bg2 = mgr.enqueue_ai_noise_file("/tmp/bg2.jpg", {"ai_noise_reduction": True})
        bg3 = mgr.enqueue_ai_noise_file("/tmp/bg3.jpg", {"ai_noise_reduction": True})
        self.assertEqual([bg1.job_id], [p["job_id"] for p in mgr.input_queue.items])

        mgr.input_queue.items.clear()
        mgr.result_queue = ListQueue([{"job_id": bg1.job_id, "status": AIJobStatus.RUNNING.value}])
        mgr.poll_results()
        self.assertEqual(AIJobStatus.RUNNING, mgr.status_by_job[bg1.job_id])

        image = np.zeros((2, 3, 3), dtype=np.float32)
        with patch("cores.ai_job_manager.manager.shared_memory.SharedMemory", FakeSharedMemory):
            fg = mgr.enqueue_image_job(
                kind=AI_NOISE_KIND,
                file_path="/tmp/current.jpg",
                image=image,
                param_snapshot={"ai_noise_reduction": True},
                content_key="foreground",
                source_signature="foreground",
            )

        self.assertEqual([], mgr.input_queue.items)
        self.assertIn(bg2.job_id, mgr.pending_payloads)
        self.assertIn(bg3.job_id, mgr.pending_payloads)
        self.assertIn(fg.job_id, mgr.pending_payloads)

        mgr.result_queue = ListQueue([{"job_id": bg1.job_id, "status": AIJobStatus.ERROR.value, "error": "done"}])
        mgr.poll_results()

        self.assertEqual([fg.job_id], [p["job_id"] for p in mgr.input_queue.items])

    def test_manager_polls_results_without_trusting_queue_empty(self):
        class EmptyLiesQueue:
            def __init__(self):
                self.items = [{"job_id": 1, "status": AIJobStatus.RUNNING.value}]

            def empty(self):
                return True

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                from queue import Empty

                raise Empty

        mgr = AIJobManager()
        job = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/a.jpg",
            content_key="content",
            source_signature="source",
        )
        mgr.jobs[job.job_id] = job
        mgr.status_by_job[job.job_id] = AIJobStatus.QUEUED
        mgr.pending_payloads[job.job_id] = {"job_id": job.job_id}
        mgr.result_queue = EmptyLiesQueue()

        results = mgr.poll_results()

        self.assertEqual([], results)
        self.assertEqual(AIJobStatus.RUNNING, mgr.status_by_job[job.job_id])
        self.assertNotIn(job.job_id, mgr.pending_payloads)

    def test_dead_worker_does_not_leave_pending_jobs_forever(self):
        states = []

        class DeadProcess:
            def is_alive(self):
                return False

        mgr = AIJobManager(viewer_state_callback=lambda path, state: states.append((path, state)))
        job = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/a.jpg",
            content_key="content",
            source_signature="source",
        )
        mgr.process = DeadProcess()
        mgr.jobs[job.job_id] = job
        mgr.status_by_job[job.job_id] = AIJobStatus.RUNNING
        mgr.job_by_target[(AI_NOISE_KIND, job.file_path)] = job.job_id
        mgr.pending_payloads[job.job_id] = {"job_id": job.job_id}

        self.assertFalse(mgr.has_pending_jobs())
        self.assertEqual(AIJobStatus.ERROR, mgr.status_by_job[job.job_id])
        self.assertNotIn((AI_NOISE_KIND, job.file_path), mgr.job_by_target)
        self.assertIn((job.file_path, "error"), states)

    def test_manager_reports_viewer_states_and_pending_count(self):
        mgr = AIJobManager()
        queued_job = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/queued.jpg",
            content_key="queued",
            source_signature="queued",
        )
        complete_job = AIJob(
            job_id=2,
            kind=AI_NOISE_KIND,
            file_path="/tmp/complete.jpg",
            content_key="complete",
            source_signature="complete",
        )
        mgr.jobs[queued_job.job_id] = queued_job
        mgr.jobs[complete_job.job_id] = complete_job
        mgr.status_by_job[queued_job.job_id] = AIJobStatus.QUEUED
        mgr.status_by_job[complete_job.job_id] = AIJobStatus.COMPLETE
        mgr.job_by_target[(AI_NOISE_KIND, queued_job.file_path)] = queued_job.job_id
        mgr.job_by_target[(AI_NOISE_KIND, complete_job.file_path)] = complete_job.job_id

        self.assertEqual({queued_job.file_path: "queued"}, mgr.viewer_states())
        self.assertEqual(1, mgr.pending_count())
        self.assertTrue(mgr.has_pending_job_for_path(queued_job.file_path))
        self.assertFalse(mgr.has_pending_job_for_path(complete_job.file_path))
        self.assertFalse(mgr.has_pending_job_for_path("/tmp/other.jpg"))

    def test_existing_file_job_reemits_viewer_state_when_requeued(self):
        states = []
        mgr = AIJobManager(viewer_state_callback=lambda path, state: states.append((path, state)))
        mgr.start = lambda: None
        job = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/a.jpg",
            content_key="pending:1",
            source_signature="pending:1",
        )
        mgr.jobs[job.job_id] = job
        mgr.status_by_job[job.job_id] = AIJobStatus.RUNNING
        mgr.job_by_target[(AI_NOISE_KIND, job.file_path)] = job.job_id

        returned = mgr.enqueue_ai_noise_file(job.file_path, {"ai_noise_reduction": True})

        self.assertEqual(job, returned)
        self.assertIn((job.file_path, "running"), states)

    def test_ai_noise_effect_uses_ai_job_manager_before_legacy_processor(self):
        image = np.full((2, 2, 3), 0.5, dtype=np.float32)
        raw = np.full_like(image, 0.25)
        calls = []

        class Manager:
            def request_ai_noise(self, file_path, img, param):
                calls.append((file_path, img.shape))
                return "complete", raw, "content", "source"

        class Processor:
            def submit_task(self, *args, **kwargs):
                raise AssertionError("legacy async processor should not be used")

            def get_result(self, *args, **kwargs):
                return None

        effect = effects.AINoiseReductonEffect()
        param = {
            "switch_ai_noise_reduction": True,
            "ai_noise_reduction": True,
            "ai_noise_reduction_intensity": 100,
        }
        efconfig = SimpleNamespace(
            ai_job_manager=Manager(),
            file_path="/tmp/a.jpg",
            processor=Processor(),
            mode=EffectMode.PREVIEW,
            upstream_hash="u",
            upstream_status=PipelineStatus.COMPLETE,
            layer_status=PipelineStatus.COMPLETE,
        )

        out = effect.make_diff(image, param, efconfig)

        self.assertEqual([("/tmp/a.jpg", image.shape)], calls)
        np.testing.assert_array_equal(out, raw)
        np.testing.assert_array_equal(param["ai_noise_reduction_result"], raw)

    def test_ai_noise_effect_reuses_stored_ai_job_result_without_resubmitting(self):
        image = np.full((2, 2, 3), 0.5, dtype=np.float32)
        raw = np.full_like(image, 0.25)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.jpg")
            pathlib.Path(path).write_bytes(b"image")
            param = {
                "switch_ai_noise_reduction": True,
                "ai_noise_reduction": True,
                "ai_noise_reduction_intensity": 100,
                "image_fidelity": ImageFidelity.FULL.value,
            }
            sig = ai_noise_source_signature(path, image, param)
            key = ai_noise_content_key(path, image, param, source_signature=sig)
            param["ai_noise_reduction_result"] = raw
            param["ai_noise_reduction_content_key"] = key
            param["ai_noise_reduction_source_signature"] = sig

            class Manager:
                def request_ai_noise(self, *args, **kwargs):
                    raise AssertionError("stored raw should be reused")

            effect = effects.AINoiseReductonEffect()
            efconfig = SimpleNamespace(
                ai_job_manager=Manager(),
                file_path=path,
                processor=None,
                mode=EffectMode.PREVIEW,
                upstream_hash="u",
                upstream_status=PipelineStatus.COMPLETE,
                layer_status=PipelineStatus.COMPLETE,
            )

            out = effect.make_diff(image, param, efconfig)

        np.testing.assert_array_equal(out, raw)


if __name__ == "__main__":
    unittest.main()
