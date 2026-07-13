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
        # AI マスク(Segment/Face/TargetText)ビットマップの共有ストア。
        # キーは cache_keys の msgpack セーフなキーを _freeze_key した frozen tuple。
        self._mask_bitmaps = {}
        # get_serialized_mask_bitmap の遅延生成結果(圧縮dict)。put で同キー上書き時に破棄する。
        self._mask_bitmaps_serialized = {}
        self.deserialize(serialized)

    def _clear_depth_locked(self):
        """深度マップ+派生キャッシュのみを消去(呼び出し元でロック済み前提)。"""
        removed = (1 if self._depth_map is not None else 0) + len(self._derived_depth_maps)
        removed_bytes = int(getattr(self._depth_map, "nbytes", 0) or 0)
        removed_bytes += sum(int(getattr(image, "nbytes", 0) or 0) for image in self._derived_depth_maps.values())
        self._depth_cache_key = None
        self._depth_map = None
        self._derived_depth_maps.clear()
        return removed, removed_bytes

    def clear(self):
        """深度+派生+マスクビットマップを全消去する(画像ロード時用)。"""
        with self._lock:
            self._generation += 1
            removed, removed_bytes = self._clear_depth_locked()
            mask_removed = len(self._mask_bitmaps)
            mask_removed_bytes = sum(int(getattr(image, "nbytes", 0) or 0) for image in self._mask_bitmaps.values())
            self._mask_bitmaps.clear()
            self._mask_bitmaps_serialized.clear()
            return {
                "ai_image_cache_entries": removed + mask_removed,
                "ai_image_cache_bytes": removed_bytes + mask_removed_bytes,
            }

    def clear_transient(self):
        """深度+派生のみ消去し、マスクビットマップ共有ストアは残す(メモリ逼迫クリア用。従来の clear() 相当)。"""
        with self._lock:
            self._generation += 1
            removed, removed_bytes = self._clear_depth_locked()
            return {"ai_image_cache_entries": removed, "ai_image_cache_bytes": removed_bytes}

    def get_mask_bitmap(self, cache_key):
        """共有ストアから AI マスクビットマップを取得する(未格納なら None)。"""
        frozen_key = _freeze_key(cache_key)
        with self._lock:
            return self._mask_bitmaps.get(frozen_key)

    def put_mask_bitmap(self, cache_key, image):
        """AI マスクビットマップを共有ストアへ格納する。同キー上書き時は圧縮キャッシュも破棄する。"""
        frozen_key = _freeze_key(cache_key)
        with self._lock:
            self._mask_bitmaps[frozen_key] = image
            self._mask_bitmaps_serialized.pop(frozen_key, None)

    def sweep_mask_bitmaps(self, live_keys):
        """live_keys(生キーの集合/イテラブル)に含まれないエントリと、対応する圧縮キャッシュを削除する。"""
        with self._lock:
            live_frozen = {_freeze_key(k) for k in live_keys}
            stale = [k for k in self._mask_bitmaps if k not in live_frozen]
            removed_bytes = 0
            for k in stale:
                image = self._mask_bitmaps.pop(k, None)
                removed_bytes += int(getattr(image, "nbytes", 0) or 0)
                self._mask_bitmaps_serialized.pop(k, None)
            return {"mask_bitmap_entries": len(stale), "mask_bitmap_bytes": removed_bytes}

    def get_serialized_mask_bitmap(self, cache_key):
        """圧縮済み(convert_image_to_list)のマスクビットマップを返す(ストアに無ければ None)。
        履歴スナップショットが編集のたびに走るため、同一キーなら2回目以降は再圧縮しない。"""
        frozen_key = _freeze_key(cache_key)
        with self._lock:
            cached = self._mask_bitmaps_serialized.get(frozen_key)
            if cached is not None:
                return cached
            image = self._mask_bitmaps.get(frozen_key)
            if image is None:
                return None
            serialized = utils.convert_image_to_list(image)
            self._mask_bitmaps_serialized[frozen_key] = serialized
            return serialized

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

    def peek_depth_map(self, cache_key):
        """既に格納済みの深度マップのみ返す(未計算なら None)。重い推論はしない。"""
        frozen_key = _freeze_key(cache_key)
        with self._lock:
            if self._depth_map is not None and self._depth_cache_key == frozen_key:
                return self._depth_map
        return None

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
