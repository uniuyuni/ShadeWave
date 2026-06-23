import logging
import threading

import numpy as np

import utils.utils as utils


AI_IMAGE_CACHE_VERSION = 1


def _freeze_key(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_freeze_key(v) for v in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_key(v)) for k, v in value.items()))
    return value


def _thaw_key(value):
    if isinstance(value, tuple):
        return [_thaw_key(v) for v in value]
    if isinstance(value, list):
        return [_thaw_key(v) for v in value]
    return value


class AIImageCache:
    def __init__(self, serialized=None):
        self._lock = threading.RLock()
        self._pending = {}
        self._generation = 0
        self._depth_cache_key = None
        self._depth_map = None
        self._derived_depth_maps = {}
        self.deserialize(serialized)

    def clear(self):
        with self._lock:
            self._generation += 1
            removed = (1 if self._depth_map is not None else 0) + len(self._derived_depth_maps)
            removed_bytes = int(getattr(self._depth_map, "nbytes", 0) or 0)
            removed_bytes += sum(int(getattr(image, "nbytes", 0) or 0) for image in self._derived_depth_maps.values())
            self._depth_cache_key = None
            self._depth_map = None
            self._derived_depth_maps.clear()
            return {"ai_image_cache_entries": removed, "ai_image_cache_bytes": removed_bytes}

    def get_depth_map(self, cache_key, compute_func):
        frozen_key = _freeze_key(cache_key)
        while True:
            with self._lock:
                if self._depth_map is not None and self._depth_cache_key == frozen_key:
                    return self._depth_map
                event = self._pending.get(frozen_key)
                if event is None:
                    event = threading.Event()
                    self._pending[frozen_key] = event
                    generation = self._generation
                    break

            event.wait()

        try:
            result = compute_func()
        except BaseException:
            with self._lock:
                if self._pending.get(frozen_key) is event:
                    self._pending.pop(frozen_key, None)
                    event.set()
            raise

        with self._lock:
            if self._generation == generation:
                self._depth_cache_key = frozen_key
                self._depth_map = result
            if self._pending.get(frozen_key) is event:
                self._pending.pop(frozen_key, None)
                event.set()
        return result

    def get_derived_depth_map(self, cache_key, compute_func):
        frozen_key = ("derived-depth", _freeze_key(cache_key))
        while True:
            with self._lock:
                cached = self._derived_depth_maps.get(frozen_key)
                if cached is not None:
                    return cached
                event = self._pending.get(frozen_key)
                if event is None:
                    event = threading.Event()
                    self._pending[frozen_key] = event
                    generation = self._generation
                    break

            event.wait()

        try:
            result = compute_func()
        except BaseException:
            with self._lock:
                if self._pending.get(frozen_key) is event:
                    self._pending.pop(frozen_key, None)
                    event.set()
            raise

        with self._lock:
            if self._generation == generation:
                self._derived_depth_maps[frozen_key] = result
            if self._pending.get(frozen_key) is event:
                self._pending.pop(frozen_key, None)
                event.set()
        return result

    def serialize(self):
        with self._lock:
            if self._depth_map is None or self._depth_cache_key is None:
                return None
            return {
                "version": AI_IMAGE_CACHE_VERSION,
                "depth": {
                    "cache_key": _thaw_key(self._depth_cache_key),
                    "image": utils.convert_image_to_list(self._depth_map),
                },
            }

    def deserialize(self, data):
        with self._lock:
            self._generation += 1
            self._depth_cache_key = None
            self._depth_map = None
            self._derived_depth_maps.clear()
            if not isinstance(data, dict):
                return
            if int(data.get("version", 0) or 0) != AI_IMAGE_CACHE_VERSION:
                return
            depth = data.get("depth")
            if not isinstance(depth, dict):
                return
            try:
                image = depth.get("image")
                cache_key = depth.get("cache_key")
                if image is None or cache_key is None:
                    return
                self._depth_map = utils.convert_image_from_list(image)
                self._depth_cache_key = _freeze_key(cache_key)
            except Exception:
                logging.exception("failed to restore ai_image_cache depth map")
                self._depth_cache_key = None
                self._depth_map = None
