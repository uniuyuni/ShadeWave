import concurrent.futures
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import file_cache_system
import numpy as np
import cores.color as color
from enums import LoadStage
from imageset import ImageSet


class _Task:
    def __init__(self, worker):
        self.worker = worker


class _FastFullDecodeImageSet:
    def preload(self, file_path, exif_data, param):
        return [_Task("_load_raw_preview"), _Task("_load_raw_full")]

    def _load_raw_preview(self, _raw, file_path, exif_data, param):
        time.sleep(0.05)
        return (file_path, self, exif_data, param, LoadStage.FIRST_PAINTABLE)

    def _load_raw_full(self, _raw, file_path, exif_data, param):
        return (file_path, self, exif_data, param, LoadStage.FULL_DECODE)


class _RecordingExecutor:
    def __init__(self, events):
        self.events = events

    def submit(self, fn, *args, **kwargs):
        self.events.append(("submit", args[1]))
        future = concurrent.futures.Future()
        future.set_result(fn(*args, **kwargs))
        return future


class RawLoadOrderFlowTest(unittest.TestCase):
    def test_full_decode_callback_waits_until_preview_callback_is_sent(self):
        path = "/tmp/order-test.raf"
        calls = []
        old_task_callback = file_cache_system._task_callback

        def record_callback(_file_callbacks, _shared_resources, result):
            value = result.result() if isinstance(result, concurrent.futures.Future) else result
            calls.append(value[-1])

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        shared_resources = {
            "cache": {},
            "preload_registry": {path: ({}, {}, None, None)},
            "active_processes": {path: time.time()},
            "executor": executor,
            "process_queue_flag": False,
        }

        try:
            file_cache_system._task_callback = record_callback
            file_cache_system._load_file_thread(
                shared_resources,
                path,
                {},
                {},
                _FastFullDecodeImageSet(),
                {},
            )
        finally:
            file_cache_system._task_callback = old_task_callback
            executor.shutdown(wait=True)

        self.assertEqual(calls, [LoadStage.FIRST_PAINTABLE, LoadStage.FULL_DECODE])

    def test_full_decode_submit_waits_until_preview_callback_is_sent(self):
        path = "/tmp/preview-first-test.raf"
        events = []
        old_task_callback = file_cache_system._task_callback

        def record_callback(_file_callbacks, _shared_resources, result):
            value = result.result() if isinstance(result, concurrent.futures.Future) else result
            events.append(("callback", value[-1]))

        shared_resources = {
            "cache": {},
            "preload_registry": {path: ({}, {}, None, None)},
            "active_processes": {path: time.time()},
            "executor": _RecordingExecutor(events),
            "process_queue_flag": False,
        }

        try:
            file_cache_system._task_callback = record_callback
            file_cache_system._load_file_thread(
                shared_resources,
                path,
                {},
                {},
                _FastFullDecodeImageSet(),
                {},
            )
        finally:
            file_cache_system._task_callback = old_task_callback

        self.assertLess(
            events.index(("callback", LoadStage.FIRST_PAINTABLE)),
            events.index(("submit", "_load_raw_full")),
        )

    def test_non_fuji_preview_geometry_probe_is_skipped_when_exif_has_size(self):
        imgset = ImageSet()

        self.assertFalse(imgset._needs_raw_preview_geometry_probe({"RawImageCroppedSize": "6240x4160"}))
        self.assertFalse(imgset._needs_raw_preview_geometry_probe({"ImageSize": "6240x4160"}))
        self.assertTrue(imgset._needs_raw_preview_geometry_probe({}))

    def test_fuji_preview_geometry_probe_uses_fuji_layout_when_exif_has_size(self):
        imgset = ImageSet()

        self.assertTrue(imgset._needs_raw_preview_geometry_probe({
            "Make": "FUJIFILM",
            "FujiLayout": "1 2 1 0",
            "RawImageCroppedSize": "6240x4160",
        }))
        self.assertFalse(imgset._needs_raw_preview_geometry_probe({
            "Make": "FUJIFILM",
            "FujiLayout": "12 12 12 12",
            "ImageSize": "6240x4160",
        }))

    def test_fuji_layout_parser_matches_libraw_fuji_width_bit(self):
        imgset = ImageSet()

        self.assertTrue(imgset._fuji_layout_requests_geometry_probe({"FujiLayout": "1 2 1 0"}))
        self.assertFalse(imgset._fuji_layout_requests_geometry_probe({"FujiLayout": "12 12 12 12"}))
        self.assertIsNone(imgset._fuji_layout_requests_geometry_probe({}))

    def test_threaded_full_decode_uses_isolated_state_from_preview(self):
        source = (Path(__file__).resolve().parents[1] / "file_cache_system.py").read_text()

        self.assertIn("task_imgset = type(imgset)()", source)
        self.assertIn("exif_data.copy() if isinstance(exif_data, dict) else exif_data", source)
        self.assertIn("param.copy() if isinstance(param, dict) else param", source)

    def test_frozen_load_executor_defaults_to_process_pool(self):
        source = (Path(__file__).resolve().parents[1] / "file_cache_system.py").read_text()

        self.assertIn("freeze_support()", (Path(__file__).resolve().parents[1] / "main.py").read_text())
        self.assertIn("PLATYPUS_FORCE_THREAD_LOAD_POOL", source)
        self.assertIn("self.ppe = ProcessPoolExecutor(max_workers=2, **ppe_kwargs)", source)
        self.assertIn('start_method = "fork"', source)
        self.assertIn("PLATYPUS_LOAD_POOL_START_METHOD", source)
        self.assertNotIn('if getattr(sys, "frozen", False):\n            self.ppe = ThreadPoolExecutor', source)

    def test_raw_preview_keeps_full_raw_geometry_for_preview_display(self):
        source = (Path(__file__).resolve().parents[1] / "imageset.py").read_text()

        self.assertIn("RAW画像のサイズに合わせてリサイズ", source)
        self.assertIn("cv2.resize(img_array, (width, height), interpolation=cv2.INTER_AREA)", source)

    def test_raw_preview_uses_uint8_srgb_decode_lut(self):
        rgb8 = np.array([[[0, 1, 12], [128, 200, 255]]], dtype=np.uint8)

        actual = color.sRGB_to_linear_LUT(rgb8)
        expected = color.rgb_gamma_decode(rgb8.astype(np.float32) / 255.0, 'sRGB')

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
