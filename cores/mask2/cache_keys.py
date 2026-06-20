"""
Stable, msgpack-safe cache keys for persisted Mask2 AI inference results.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


_KEY_PREFIX = "mask2-ai-cache"
_KEY_VERSION = 1


def _msgpack_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_msgpack_safe(v) for v in value]
    if isinstance(value, list):
        return [_msgpack_safe(v) for v in value]
    if isinstance(value, dict):
        return {
            str(_msgpack_safe(k)): _msgpack_safe(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_cache_key(kind: str, *parts: Any):
    return [_KEY_PREFIX, _KEY_VERSION, kind, *[_msgpack_safe(part) for part in parts]]


def segment_cache_key(original_image_size, center, corner, invert):
    return make_cache_key("segment", original_image_size, center, corner)


def depth_cache_key(original_image_size, algorithm_version):
    return make_cache_key("depth", original_image_size, int(algorithm_version))


def face_cache_key(original_image_size, exclude_names):
    return make_cache_key("face", original_image_size, list(exclude_names))


def target_text_cache_key(original_image_size, text, invert):
    return make_cache_key("target_text", original_image_size, str(text))
