import os
import random
import sys
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import file_cache_system
import memory_manager


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = ROOT.parent / "test_photos"
SUPPORTED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".jxl",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".raf", ".rw2", ".sr2",
    ".pef", ".raw", ".3fr", ".fff", ".exr",
}


def _photo_paths():
    paths = [
        str(path)
        for path in sorted(TEST_PHOTOS.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return paths


def random_selection_interval(rng, max_interval=None):
    if max_interval is None:
        try:
            max_interval = float(os.getenv("PLATYPUS_SELECTION_STRESS_MAX_INTERVAL", "10"))
        except ValueError:
            max_interval = 10.0
    return rng.uniform(0.0, max(0.0, float(max_interval)))


def maybe_wait_selection_interval(interval):
    if os.getenv("PLATYPUS_SELECTION_STRESS_REAL_SLEEP", "1").strip().lower() in {"1", "true", "yes", "on"}:
        time.sleep(interval)


class FakeEffect:
    def __init__(self):
        self.diff = np.ones((8, 8, 3), dtype=np.float32)
        self.hash = 123
        self.reeffect_count = 0
        self._cached_predict = np.ones((4, 4, 3), dtype=np.float32)
        self._cached_predict_key = "predict"

    def reeffect(self):
        self.diff = None
        self.hash = None
        self.reeffect_count += 1


class FakeProcessor:
    def __init__(self):
        self.cache = {
            ("done", 1): {"status": "COMPLETE", "result": np.ones((8, 8, 3), dtype=np.float32)},
            ("running", 2): {"status": "RUNNING", "result": None},
        }

    def clear_completed_cache(self):
        keys = [key for key, value in self.cache.items() if value.get("status") == "COMPLETE"]
        for key in keys:
            del self.cache[key]
        return len(keys)


class FakeOwner:
    def __init__(self):
        self._last_pmck_dict = None
        self.imgset = None
        self.primary_effects = [{f"effect_{i}": FakeEffect() for i in range(3)}]
        self.processor = FakeProcessor()
        self._expected_file_path = None
        self.accepted_callbacks = []
        self.ignored_callbacks = []

    def select_path(self, cache_system, file_path):
        previous = self.imgset.file_path if self.imgset is not None else None
        self._last_pmck_dict = (previous, {"heavy": np.ones((16, 16), dtype=np.float32)})
        cache_system.on_image_selection_changed(
            owner=self,
            previous_file_path=previous,
            current_file_path=file_path,
        )
        self._expected_file_path = file_path
        self.imgset = SimpleNamespace(file_path=file_path)

    def on_fcs_get_file(self, file_path, imgset):
        if file_path != self._expected_file_path:
            self.ignored_callbacks.append(file_path)
            return
        self.accepted_callbacks.append(file_path)
        self.imgset = imgset


def make_cache_system(max_final_display_cache=5):
    cache_system = file_cache_system.FileCacheSystem.__new__(file_cache_system.FileCacheSystem)
    cache_system.cache = {}
    cache_system.preload_registry = {}
    cache_system.active_processes = {}
    cache_system.file_callbacks = {}
    cache_system.final_display_cache = OrderedDict()
    cache_system.max_final_display_cache = max_final_display_cache
    cache_system.max_cache_size = 100
    cache_system.max_concurrent_loads = 20
    return cache_system


class FileSelectionMemoryStressTest(unittest.TestCase):
    def test_rapid_selection_changes_keep_loading_and_memory_state_consistent(self):
        cache_system = make_cache_system(max_final_display_cache=7)
        owner = FakeOwner()
        source_paths = _photo_paths()
        if len(source_paths) < 2:
            self.skipTest(f"test photos not available: {TEST_PHOTOS}")
        paths = [source_paths[i % len(source_paths)] for i in range(50)]
        rng = random.Random(20260616)
        virtual_elapsed = 0.0

        old_memory_pressure = memory_manager.memory_pressure
        try:
            memory_manager.memory_pressure = lambda: (False, "ok")
            for i, path in enumerate(paths):
                interval = random_selection_interval(rng)
                self.assertGreaterEqual(interval, 0.0)
                self.assertLessEqual(interval, 10.0)
                virtual_elapsed += interval
                maybe_wait_selection_interval(interval)

                if i % 4 == 0:
                    cache_system.preload_registry[path] = ({}, {}, SimpleNamespace(img=None), None)
                elif i % 4 == 1:
                    cache_system.preload_registry[path] = ({}, {}, SimpleNamespace(img=None), None)
                    cache_system.active_processes[path] = float(i)
                elif i % 4 == 2:
                    cache_system.cache[path] = (
                        SimpleNamespace(img=np.ones((2, 2, 3), dtype=np.float32), file_path=path),
                        {},
                        {"param": np.ones((2,), dtype=np.float32)},
                        None,
                    )

                owner.select_path(cache_system, path)
                self.assertIsNone(owner._last_pmck_dict)
                self.assertEqual(owner._expected_file_path, path)

                display = np.full((4, 5, 3), i / 100.0, dtype=np.float32)
                self.assertTrue(
                    cache_system.remember_final_display_image(
                        path,
                        display,
                        stage="FULL_DECODE",
                        frame_version=virtual_elapsed,
                    )
                )
                cached = cache_system.get_final_display_image(path)
                self.assertIsNot(cached, display)
                np.testing.assert_array_equal(cached, display)
                self.assertLessEqual(len(cache_system.final_display_cache), cache_system.max_final_display_cache)

                if i > 0:
                    owner.on_fcs_get_file(paths[i - 1], SimpleNamespace(file_path=paths[i - 1]))
                    self.assertEqual(owner.imgset.file_path, path)
                owner.on_fcs_get_file(path, SimpleNamespace(file_path=path))
                self.assertEqual(owner.imgset.file_path, path)

            self.assertEqual(owner.accepted_callbacks[-1], paths[-1])
            self.assertGreater(len(owner.ignored_callbacks), 0)
            self.assertGreater(virtual_elapsed, 0.0)
            self.assertLessEqual(virtual_elapsed, 500.0)
            self.assertGreater(cache_system.cache_memory_bytes(), 0)
            self.assertLessEqual(len(cache_system.final_display_cache), cache_system.max_final_display_cache)

            current_path = paths[-1]
            cache_system.delete_file(current_path)
            self.assertNotIn(current_path, cache_system.final_display_cache)
            self.assertNotIn(current_path, cache_system.cache)
            self.assertNotIn(current_path, cache_system.preload_registry)
        finally:
            memory_manager.memory_pressure = old_memory_pressure

    def test_memory_pressure_drops_effect_caches_before_final_display_cache(self):
        cache_system = make_cache_system(max_final_display_cache=4)
        owner = FakeOwner()
        owner.imgset = SimpleNamespace(file_path="/tmp/current.jpg")
        for i in range(4):
            cache_system.remember_final_display_image(
                f"/tmp/final_{i}.jpg",
                np.full((8, 8, 3), i, dtype=np.float32),
                stage="FULL_DECODE",
                frame_version=i,
            )

        old_memory_pressure = memory_manager.memory_pressure
        try:
            memory_manager.memory_pressure = lambda: (True, "forced")
            result = cache_system.enforce_memory_policy(owner=owner, reason="stress")
        finally:
            memory_manager.memory_pressure = old_memory_pressure

        self.assertTrue(result.get("cleared"))
        for layer in owner.primary_effects:
            for effect in layer.values():
                self.assertIsNone(effect.diff)
                self.assertIsNone(effect.hash)
                self.assertIsNone(effect._cached_predict)
                self.assertIsNone(effect._cached_predict_key)
                self.assertGreater(effect.reeffect_count, 0)
        self.assertNotIn(("done", 1), owner.processor.cache)
        self.assertIn(("running", 2), owner.processor.cache)
        self.assertGreater(result.get("final_display_evicted", 0), 0)
        self.assertEqual(len(cache_system.final_display_cache), 0)


if __name__ == "__main__":
    unittest.main()
