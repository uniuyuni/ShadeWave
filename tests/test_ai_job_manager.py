import os
import pathlib
import sys
import tempfile
import threading
import time
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
    AISidecarMergeQueue,
    ai_noise_content_key,
    ai_noise_source_signature,
    ai_noise_valid_content_keys,
    current_param_accepts_ai_noise_result,
    merge_ai_noise_result_into_param,
    merge_ai_noise_result_into_pmck,
)
from cores.ai_job_manager import worker as ai_job_worker_mod
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

    def test_ai_noise_content_key_does_not_depend_on_absolute_path(self):
        image = np.zeros((4, 5, 3), dtype=np.float32)
        param = {"ai_noise_reduction": True, "image_fidelity": ImageFidelity.FULL.value}

        with tempfile.TemporaryDirectory() as tmp:
            path1 = os.path.join(tmp, "a.jpg")
            path2 = os.path.join(tmp, "nested", "a.jpg")
            pathlib.Path(path1).write_bytes(b"image")
            pathlib.Path(path2).parent.mkdir()
            pathlib.Path(path2).write_bytes(b"image")
            sig1 = ai_noise_source_signature(path1, image, param)
            key1 = ai_noise_content_key(path1, image, param, source_signature=sig1)
            sig2 = ai_noise_source_signature(path2, image, param)
            key2 = ai_noise_content_key(path2, image, param, source_signature=sig2)

        self.assertEqual(sig1, sig2)
        self.assertEqual(key1, key2)

    def test_ai_noise_content_key_distinguishes_same_size_different_files(self):
        image = np.zeros((4, 5, 3), dtype=np.float32)
        param = {"ai_noise_reduction": True, "image_fidelity": ImageFidelity.FULL.value}

        with tempfile.TemporaryDirectory() as tmp:
            path1 = os.path.join(tmp, "a.jpg")
            path2 = os.path.join(tmp, "b.jpg")
            pathlib.Path(path1).write_bytes(b"image-a")
            pathlib.Path(path2).write_bytes(b"image-b")
            sig1 = ai_noise_source_signature(path1, image, param)
            key1 = ai_noise_content_key(path1, image, param, source_signature=sig1)
            sig2 = ai_noise_source_signature(path2, image, param)
            key2 = ai_noise_content_key(path2, image, param, source_signature=sig2)

        self.assertNotEqual(sig1, sig2)
        self.assertNotEqual(key1, key2)

    def test_file_ai_noise_content_key_changes_when_decode_sample_differs(self):
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

    def test_memory_ai_noise_content_key_changes_when_input_pixels_shift(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[1:3, 1:3] = 1.0
        shifted = np.zeros_like(image)
        shifted[2:4, 2:4] = 1.0
        param = {"ai_noise_reduction": True, "image_fidelity": ImageFidelity.FULL.value}

        sig1 = ai_noise_source_signature(None, image, param)
        key1 = ai_noise_content_key(None, image, param, source_signature=sig1)
        sig2 = ai_noise_source_signature(None, shifted, param)
        key2 = ai_noise_content_key(None, shifted, param, source_signature=sig2)

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

    def test_sidecar_merge_queue_does_not_block_on_slow_pmck_io(self):
        started = threading.Event()
        release = threading.Event()

        def slow_merge(_file_path, _raw, *, content_key, source_signature):
            started.set()
            release.wait(timeout=1.0)
            return bool(content_key and source_signature)

        queue = AISidecarMergeQueue(merge_ai_noise_fn=slow_merge)
        try:
            raw = np.ones((2, 2, 3), dtype=np.float32)
            job = AIJob(
                job_id=1,
                kind=AI_NOISE_KIND,
                file_path="/tmp/a.jpg",
                content_key="content",
                source_signature="source",
            )

            future = queue.submit_ai_noise_result(job, raw)

            self.assertTrue(started.wait(timeout=0.5))
            self.assertFalse(future.done())
            self.assertEqual([], queue.poll_results())

            release.set()
            for _ in range(100):
                results = queue.poll_results()
                if results:
                    break
                time.sleep(0.01)
            else:
                self.fail("sidecar merge did not finish")

            self.assertEqual(1, len(results))
            self.assertIs(results[0].job, job)
            self.assertTrue(results[0].merged)
            self.assertIsNone(results[0].error)
            np.testing.assert_array_equal(results[0].result, raw)
        finally:
            release.set()
            queue.shutdown()

    def test_sidecar_merge_queue_applies_backpressure(self):
        release = threading.Event()

        def slow_merge(_file_path, _raw, *, content_key, source_signature):
            release.wait(timeout=1.0)
            return bool(content_key and source_signature)

        queue = AISidecarMergeQueue(merge_ai_noise_fn=slow_merge, max_pending=1)
        try:
            raw = np.ones((2, 2, 3), dtype=np.float32)
            job1 = AIJob(1, AI_NOISE_KIND, "/tmp/a.jpg", "content1", "source1")
            job2 = AIJob(2, AI_NOISE_KIND, "/tmp/b.jpg", "content2", "source2")

            self.assertIsNotNone(queue.submit_ai_noise_result(job1, raw))
            self.assertIsNone(queue.submit_ai_noise_result(job2, raw))
            self.assertEqual(1, queue.pending_count())
            self.assertTrue(queue.has_pending_job(job1))
            self.assertFalse(queue.has_pending_job(job2))
        finally:
            release.set()
            queue.shutdown()

    def test_ai_job_worker_nice_env_is_clamped_and_optional(self):
        with patch.dict(os.environ, {"PLATYPUS_AI_JOB_NICE": "7"}):
            self.assertEqual(7, ai_job_worker_mod.ai_job_nice_increment())
        with patch.dict(os.environ, {"PLATYPUS_AI_JOB_NICE": "999"}):
            self.assertEqual(20, ai_job_worker_mod.ai_job_nice_increment())
        with patch.dict(os.environ, {"PLATYPUS_AI_JOB_NICE": "-5"}):
            self.assertEqual(0, ai_job_worker_mod.ai_job_nice_increment())
        with patch.dict(os.environ, {"PLATYPUS_AI_JOB_NICE": "bad"}):
            self.assertEqual(10, ai_job_worker_mod.ai_job_nice_increment())

    def test_thread_worker_does_not_apply_process_nice(self):
        captured = {}

        class FakeThread:
            def __init__(self, *, target, name, args, daemon):
                captured["target"] = target
                captured["name"] = name
                captured["args"] = args
                captured["daemon"] = daemon

            def start(self):
                captured["started"] = True

        with patch("cores.ai_job_manager.worker.threading.Thread", FakeThread):
            worker = ai_job_worker_mod.start_thread_worker("in", "out", "stop", {"x": 1})

        self.assertIsInstance(worker, FakeThread)
        self.assertEqual("AIJobWorkerThread", captured["name"])
        self.assertEqual(("in", "out", "stop", {"x": 1}, False), captured["args"])
        self.assertTrue(captured["started"])

    def test_worker_returns_stale_when_file_job_ai_noise_is_disabled_before_inference(self):
        class InputQueue:
            def __init__(self):
                self.items = [
                    {
                        "job_id": 1,
                        "kind": AI_NOISE_KIND,
                        "file_path": "/tmp/a.jpg",
                        "param_snapshot": {"ai_noise_reduction": True},
                    }
                ]
                self.stopped = False

            def get(self, timeout=0.1):
                if self.items:
                    return self.items.pop(0)
                self.stopped = True
                raise ai_job_worker_mod.Empty()

        class ResultQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        class StopEvent:
            def __init__(self, input_queue):
                self.input_queue = input_queue

            def is_set(self):
                return self.input_queue.stopped

        input_queue = InputQueue()
        result_queue = ResultQueue()
        image = np.zeros((2, 2, 3), dtype=np.float32)

        with patch("cores.ai_job_manager.worker._load_file_image", return_value=(image, {"ai_noise_reduction": False})):
            with patch("cores.ai_job_manager.worker.run_ai_noise", side_effect=AssertionError("SCUNet should not run")):
                ai_job_worker_mod.ai_job_worker(input_queue, result_queue, StopEvent(input_queue), {}, apply_process_nice=False)

        self.assertEqual(AIJobStatus.RUNNING.value, result_queue.items[0]["status"])
        self.assertEqual(AIJobStatus.STALE.value, result_queue.items[1]["status"])

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
        mgr.thread_mode = True
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

        self.assertEqual(AIJobStatus.CANCELLED, mgr.status_by_job[bg1.job_id])
        self.assertEqual([fg.job_id], [p["job_id"] for p in mgr.input_queue.items])
        self.assertIn(bg2.job_id, mgr.pending_payloads)
        self.assertIn(bg3.job_id, mgr.pending_payloads)
        self.assertIn(fg.job_id, mgr.pending_payloads)

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

    def test_background_completed_result_is_cached_until_sidecar_consumes_it(self):
        class ListQueue:
            def __init__(self, items):
                self.items = list(items)

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                from queue import Empty

                raise Empty

        class FakeSharedMemory:
            def __init__(self, create=False, size=0, name=None):
                self.name = name or "fake-result"
                self.buf = bytearray(np.ones((2, 2, 3), dtype=np.float32).tobytes())
                self.closed = False
                self.unlinked = False

            def close(self):
                self.closed = True

            def unlink(self):
                self.unlinked = True

        mgr = AIJobManager()
        job = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/bg.jpg",
            content_key="pending:1",
            source_signature="pending:1",
        )
        mgr.jobs[job.job_id] = job
        mgr.status_by_job[job.job_id] = AIJobStatus.RUNNING
        mgr.job_by_target[(AI_NOISE_KIND, job.file_path)] = job.job_id
        mgr.result_queue = ListQueue([
            {
                "job_id": job.job_id,
                "status": AIJobStatus.COMPLETE.value,
                "shm_name": "fake-result",
                "shape": (2, 2, 3),
                "dtype": "float32",
                "content_key": "real-content",
                "source_signature": "real-source",
            }
        ])

        with patch("cores.ai_job_manager.manager.shared_memory.SharedMemory", FakeSharedMemory):
            results = mgr.poll_results()

        self.assertEqual(1, len(results))
        self.assertGreater(mgr.completed_results_bytes(), 0)
        self.assertIsNotNone(mgr.get_completed_result(AI_NOISE_KIND, "/tmp/bg.jpg", "real-content"))
        mgr.discard_completed_result(results[0].job)
        self.assertEqual(0, mgr.completed_results_bytes())

    def test_foreground_completed_result_can_be_discarded_after_handling(self):
        mgr = AIJobManager()
        job = AIJob(1, AI_NOISE_KIND, "/tmp/current.jpg", "content", "source")
        raw = np.ones((2, 2, 3), dtype=np.float32)
        mgr.completed_results[(job.kind, job.file_path, job.content_key)] = raw

        self.assertGreater(mgr.completed_results_bytes(), 0)
        mgr.discard_completed_result(job)

        self.assertEqual(0, mgr.completed_results_bytes())

    def test_completed_result_cache_evicts_oldest_when_over_budget(self):
        mgr = AIJobManager()
        raw = np.ones((2, 2, 3), dtype=np.float32)
        job1 = AIJob(1, AI_NOISE_KIND, "/tmp/one.jpg", "content-1", "source-1")
        job2 = AIJob(2, AI_NOISE_KIND, "/tmp/two.jpg", "content-2", "source-2")

        with patch.dict(os.environ, {"PLATYPUS_AI_COMPLETED_CACHE_MAX_MB": "0.00008"}):
            self.assertTrue(mgr._store_completed_result(job1, raw))
            self.assertTrue(mgr._store_completed_result(job2, raw))

        self.assertIsNone(mgr.get_completed_result(AI_NOISE_KIND, job1.file_path, job1.content_key))
        self.assertIsNotNone(mgr.get_completed_result(AI_NOISE_KIND, job2.file_path, job2.content_key))
        self.assertLessEqual(mgr.completed_results_bytes(), 83)

    def test_completed_result_larger_than_cache_budget_is_not_stored(self):
        mgr = AIJobManager()
        job = AIJob(1, AI_NOISE_KIND, "/tmp/huge.jpg", "content", "source")
        raw = np.ones((2, 2, 3), dtype=np.float32)

        with patch.dict(os.environ, {"PLATYPUS_AI_COMPLETED_CACHE_MAX_MB": "0.000001"}):
            self.assertFalse(mgr._store_completed_result(job, raw))

        self.assertEqual(0, mgr.completed_results_bytes())
        self.assertIsNone(mgr.get_completed_result(AI_NOISE_KIND, job.file_path, job.content_key))

    def test_progress_event_updates_viewer_state_with_tile_count(self):
        class ListQueue:
            def __init__(self, items):
                self.items = list(items)

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                from queue import Empty

                raise Empty

        states = []
        mgr = AIJobManager(viewer_state_callback=lambda path, state, progress: states.append((path, state, progress)))
        job = AIJob(1, AI_NOISE_KIND, "/tmp/a.jpg", "content", "source")
        mgr.jobs[job.job_id] = job
        mgr.status_by_job[job.job_id] = AIJobStatus.RUNNING
        mgr.result_queue = ListQueue([
            {
                "job_id": job.job_id,
                "status": "progress",
                "done": 12,
                "total": 84,
            },
        ])

        self.assertEqual([], mgr.poll_results())

        self.assertIn((job.file_path, "running", "12/84"), states)

    def test_cancelled_or_unknown_complete_unlinks_result_shm(self):
        class ListQueue:
            def __init__(self, items):
                self.items = list(items)

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                from queue import Empty

                raise Empty

        discarded = []

        class FakeSharedMemory:
            def __init__(self, create=False, size=0, name=None):
                self.name = name

            def close(self):
                discarded.append((self.name, "close"))

            def unlink(self):
                discarded.append((self.name, "unlink"))

        mgr = AIJobManager()
        cancelled = AIJob(1, AI_NOISE_KIND, "/tmp/cancelled.jpg", "content", "source")
        mgr.jobs[cancelled.job_id] = cancelled
        mgr.status_by_job[cancelled.job_id] = AIJobStatus.CANCELLED
        mgr.result_queue = ListQueue([
            {
                "job_id": cancelled.job_id,
                "status": AIJobStatus.COMPLETE.value,
                "shm_name": "cancelled-result",
                "shape": (2, 2, 3),
                "dtype": "float32",
            },
            {
                "job_id": 999,
                "status": AIJobStatus.COMPLETE.value,
                "shm_name": "unknown-result",
                "shape": (2, 2, 3),
                "dtype": "float32",
            },
        ])

        with patch("cores.ai_job_manager.manager.shared_memory.SharedMemory", FakeSharedMemory):
            self.assertEqual([], mgr.poll_results())

        self.assertIn(("cancelled-result", "unlink"), discarded)
        self.assertIn(("unknown-result", "unlink"), discarded)

    def test_dead_worker_does_not_leave_pending_jobs_forever(self):
        states = []
        from cores.ai_job_manager import manager as manager_mod

        class DeadProcess:
            def is_alive(self):
                return False

        mgr = AIJobManager(viewer_state_callback=lambda path, state: states.append((path, state)))
        mgr.worker_restart_count = manager_mod.MAX_WORKER_RESTARTS
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

    def test_dead_worker_requeues_recoverable_background_file_job(self):
        states = []
        mgr = AIJobManager(viewer_state_callback=lambda path, state: states.append((path, state)))
        job = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/bg.jpg",
            content_key="pending:1",
            source_signature="pending:1",
            param_snapshot={"ai_noise_reduction": True},
        )
        mgr.jobs[job.job_id] = job
        mgr.status_by_job[job.job_id] = AIJobStatus.RUNNING
        mgr.job_by_target[(AI_NOISE_KIND, job.file_path)] = job.job_id
        mgr.dispatched_job_id = job.job_id

        mgr._prepare_recoverable_dead_worker_jobs()

        self.assertEqual(AIJobStatus.QUEUED, mgr.status_by_job[job.job_id])
        self.assertIn(job.job_id, mgr.pending_payloads)
        self.assertEqual("/tmp/bg.jpg", mgr.pending_payloads[job.job_id]["file_path"])
        self.assertIn((job.file_path, "queued"), states)

    def test_thread_mode_cancel_running_does_not_restart_worker(self):
        dispatched = []
        mgr = AIJobManager()
        mgr.thread_mode = True
        mgr._restart_worker = lambda: (_ for _ in ()).throw(AssertionError("thread mode must not restart worker"))
        job = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/bg.jpg",
            content_key="pending:1",
            source_signature="pending:1",
        )
        mgr.jobs[job.job_id] = job
        mgr.status_by_job[job.job_id] = AIJobStatus.RUNNING
        mgr.job_by_target[(AI_NOISE_KIND, job.file_path)] = job.job_id
        mgr.dispatched_job_id = job.job_id
        mgr.input_queue = SimpleNamespace(put=lambda payload: dispatched.append(payload), get_nowait=lambda: (_ for _ in ()).throw(Exception("unused")))

        mgr.cancel_path(job.file_path)

        self.assertEqual(AIJobStatus.CANCELLED, mgr.status_by_job[job.job_id])
        self.assertIsNone(mgr.dispatched_job_id)
        self.assertEqual([], dispatched)

    def test_foreground_request_preempts_running_background_in_thread_mode(self):
        dispatched = []

        class FakeSharedMemory:
            def __init__(self, create=False, size=0, name=None):
                self.name = name or "fake-input"
                self.buf = bytearray(size)

            def close(self):
                pass

            def unlink(self):
                pass

        mgr = AIJobManager()
        mgr.thread_mode = True
        mgr.start = lambda: None
        mgr.input_queue = SimpleNamespace(
            put=lambda payload: dispatched.append(payload),
            get_nowait=lambda: (_ for _ in ()).throw(Exception("unused")),
        )
        background = AIJob(
            job_id=1,
            kind=AI_NOISE_KIND,
            file_path="/tmp/bg.jpg",
            content_key="pending:1",
            source_signature="pending:1",
        )
        cancel_event = threading.Event()
        mgr.jobs[background.job_id] = background
        mgr._job_counter = background.job_id
        mgr.status_by_job[background.job_id] = AIJobStatus.RUNNING
        mgr.job_by_target[(AI_NOISE_KIND, background.file_path)] = background.job_id
        mgr.priority_by_job[background.job_id] = 100
        mgr.cancel_events[background.job_id] = cancel_event
        mgr.dispatched_job_id = background.job_id

        with patch("cores.ai_job_manager.manager.shared_memory.SharedMemory", FakeSharedMemory):
            foreground = mgr.enqueue_image_job(
                kind=AI_NOISE_KIND,
                file_path="/tmp/current.jpg",
                image=np.zeros((2, 2, 3), dtype=np.float32),
                param_snapshot={"ai_noise_reduction": True},
                content_key="content",
                source_signature="source",
                priority=0,
            )

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(AIJobStatus.CANCELLED, mgr.status_by_job[background.job_id])
        self.assertEqual(foreground.job_id, mgr.dispatched_job_id)
        self.assertEqual([foreground.job_id], [payload["job_id"] for payload in dispatched])
        self.assertIsNotNone(dispatched[0].get("cancel_event"))

    def test_run_ai_noise_maps_helper_cancel_to_worker_cancel(self):
        class HelperCancelled(RuntimeError):
            pass

        state = {"progress": object(), "cancel": object()}

        def set_progress_callback(callback):
            state["progress"] = callback

        def set_cancel_callback(callback):
            state["cancel"] = callback

        def predict_helper(_engine, _image):
            if state["cancel"] and state["cancel"]():
                raise HelperCancelled("cancelled")
            return _image

        fake_helper = SimpleNamespace(
            setup=lambda: object(),
            set_progress_callback=set_progress_callback,
            set_cancel_callback=set_cancel_callback,
            predict_helper=predict_helper,
            is_cancelled_error=lambda exc: isinstance(exc, HelperCancelled),
        )
        if hasattr(ai_job_worker_mod.run_ai_noise, "_engine"):
            delattr(ai_job_worker_mod.run_ai_noise, "_engine")

        with patch.dict(sys.modules, {"helpers.scunet_coreml_helper": fake_helper}):
            with self.assertRaises(ai_job_worker_mod.AIJobCancelled):
                ai_job_worker_mod.run_ai_noise(
                    np.zeros((1, 1, 3), dtype=np.float32),
                    cancel_callback=lambda: True,
                )

        self.assertIsNone(state["progress"])
        self.assertIsNone(state["cancel"])

    def test_scunet_coreml_helper_wraps_tile_predict_with_cancel_check(self):
        import helpers.scunet_coreml_helper as helper

        calls = {"predict": 0}

        class FakeModel:
            def predict(self, payload):
                calls["predict"] += 1
                return payload

        engine = SimpleNamespace(model=FakeModel())
        original_predict = engine.model.predict
        helper.set_cancel_callback(lambda: True)
        try:
            with helper._cancelable_engine_predict(engine):
                with self.assertRaises(helper.SCUNetCancelledError):
                    engine.model.predict({"input": "tile"})
            self.assertEqual(original_predict, engine.model.predict)
            self.assertEqual(0, calls["predict"])
        finally:
            helper.set_cancel_callback(None)

    def test_finished_jobs_are_pruned_to_retention_limit(self):
        mgr = AIJobManager()
        from cores.ai_job_manager import manager as manager_mod

        keep = manager_mod.FINISHED_JOB_RETENTION
        for i in range(keep + 3):
            job = AIJob(i + 1, AI_NOISE_KIND, f"/tmp/{i}.jpg", f"content-{i}", f"source-{i}")
            mgr.jobs[job.job_id] = job
            mgr.status_by_job[job.job_id] = AIJobStatus.COMPLETE
            mgr.job_by_target[(AI_NOISE_KIND, job.file_path)] = job.job_id
            mgr.completed_results[(job.kind, job.file_path, job.content_key)] = np.zeros((1, 1, 3), dtype=np.float32)

        mgr._prune_finished_jobs()

        self.assertLessEqual(len(mgr.jobs), keep)
        self.assertLessEqual(len(mgr.status_by_job), keep)
        self.assertLessEqual(len(mgr.completed_results), keep)

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
            "image_fidelity": ImageFidelity.FULL.value,
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

    def test_ai_noise_effect_waits_for_full_decode_before_file_job(self):
        image = np.full((2, 2, 3), 0.5, dtype=np.float32)

        class Manager:
            def request_ai_noise(self, *args, **kwargs):
                raise AssertionError("AI-NR must not run from RAW preview data")

        effect = effects.AINoiseReductonEffect()
        param = {
            "switch_ai_noise_reduction": True,
            "ai_noise_reduction": True,
            "ai_noise_reduction_intensity": 100,
            "image_fidelity": ImageFidelity.PREVIEW.value,
        }
        efconfig = SimpleNamespace(
            ai_job_manager=Manager(),
            file_path="/tmp/a.raw",
            processor=None,
            mode=EffectMode.PREVIEW,
            upstream_hash="u",
            upstream_status=PipelineStatus.COMPLETE,
            layer_status=PipelineStatus.COMPLETE,
        )

        out = effect.make_diff(image, param, efconfig)

        self.assertIsNone(out)
        self.assertEqual(PipelineStatus.PREVIEW, efconfig.layer_status)

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
