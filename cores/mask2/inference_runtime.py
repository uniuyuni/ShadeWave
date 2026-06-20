"""
Mask2 の推論モデルキャッシュ。UI（mask_editor2）とヘッドレス export で共有する。
"""
from __future__ import annotations

from threading import RLock

import logging
import os
import time

import config
import numpy as np

import cores.mask2.cutout_guided as cutout_guided
import utils.aiutils as aiutils

_sam3_processor = None
_sam3_lock = RLock()
_depth_model = None
_faces = None

DEPTH_MAP_ALGORITHM_VERSION = 2
_LOGGER = logging.getLogger(__name__)


def _sam3_bbox_clip_enabled():
    value = os.environ.get("PLATYPUS_SAM3_BBOX_CLIP")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _clip_bbox_xywh(bbox, shape):
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return [0, 0, 0, 0]
    x, y, bw, bh = [float(v) for v in bbox]
    x0 = int(np.floor(min(x, x + bw)))
    y0 = int(np.floor(min(y, y + bh)))
    x1 = int(np.ceil(max(x, x + bw)))
    y1 = int(np.ceil(max(y, y + bh)))
    x0 = min(max(x0, 0), w)
    y0 = min(max(y0, 0), h)
    x1 = min(max(x1, 0), w)
    y1 = min(max(y1, 0), h)
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def _mask_bbox_stats(mask, bbox, threshold=0.001):
    x, y, w, h = [int(v) for v in bbox]
    mask_arr = np.asarray(mask, dtype=np.float32)
    total = int(np.count_nonzero(mask_arr > threshold))
    inside = int(np.count_nonzero(mask_arr[y:y + h, x:x + w] > threshold)) if w > 0 and h > 0 else 0
    outside = max(0, total - inside)
    max_value = float(np.nanmax(mask_arr)) if mask_arr.size else 0.0
    return total, inside, outside, max_value


def _zero_outside_bbox(mask, bbox):
    x, y, w, h = [int(v) for v in bbox]
    out = np.array(mask, dtype=np.float32, copy=True)
    if w <= 0 or h <= 0:
        out[...] = 0.0
        return out
    if y > 0:
        out[:y, :] = 0.0
    y1 = y + h
    if y1 < out.shape[0]:
        out[y1:, :] = 0.0
    if x > 0:
        out[:, :x] = 0.0
    x1 = x + w
    if x1 < out.shape[1]:
        out[:, x1:] = 0.0
    return out


def _expanded_bbox_xyxy(bbox, shape, pad):
    h, w = int(shape[0]), int(shape[1])
    x, y, bw, bh = [int(v) for v in bbox]
    pad = max(0, int(pad))
    x0 = min(max(x - pad, 0), w)
    y0 = min(max(y - pad, 0), h)
    x1 = min(max(x + bw + pad, 0), w)
    y1 = min(max(y + bh + pad, 0), h)
    return x0, y0, x1, y1


def _sam3_roi_pad():
    value = os.environ.get("PLATYPUS_SAM3_ROI_PAD")
    if value is None:
        return 96
    try:
        return max(0, int(round(float(value))))
    except Exception:
        return 96


def _sam3_roi_enabled():
    value = os.environ.get("PLATYPUS_SAM3_ROI_INPUT")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _elapsed_ms(start):
    return (time.perf_counter() - start) * 1000.0


def _array_cache_key(image):
    arr = np.asarray(image)
    return (
        int(arr.__array_interface__["data"][0]),
        tuple(int(v) for v in arr.shape),
        tuple(int(v) for v in arr.strides),
        str(arr.dtype),
    )


def _predict_sam3_bbox_roi(sam3_helper, processor, image, bbox, source_image_key):
    if not _sam3_roi_enabled():
        start = time.perf_counter()
        image_key = ("full", source_image_key, tuple(int(v) for v in image.shape))
        mask = sam3_helper.predict_sam3_for_bbox(processor, image, bbox, image_key=image_key)
        _LOGGER.info(
            "SAM3 bbox predict full_image elapsed=%.1fms image_shape=%s bbox=%s",
            _elapsed_ms(start),
            image.shape[:2],
            bbox,
        )
        return mask
    x0, y0, x1, y1 = _expanded_bbox_xyxy(bbox, image.shape, _sam3_roi_pad())
    if x1 <= x0 or y1 <= y0:
        return np.zeros(image.shape[:2], dtype=np.float32)
    roi = image[y0:y1, x0:x1]
    bx, by, bw, bh = [int(v) for v in bbox]
    roi_bbox = [bx - x0, by - y0, bw, bh]
    image_key = (
        "roi",
        source_image_key,
        (x0, y0, x1, y1),
        tuple(int(v) for v in roi.shape),
    )
    start = time.perf_counter()
    roi_mask = sam3_helper.predict_sam3_for_bbox(processor, roi, roi_bbox, image_key=image_key)
    _LOGGER.info(
        "SAM3 bbox predict roi elapsed=%.1fms roi=(%d,%d)-(%d,%d) roi_shape=%s roi_bbox=%s source_bbox=%s",
        _elapsed_ms(start),
        x0,
        y0,
        x1,
        y1,
        roi.shape[:2],
        roi_bbox,
        bbox,
    )
    full_mask = np.zeros(image.shape[:2], dtype=np.float32)
    full_mask[y0:y1, x0:x1] = np.asarray(roi_mask, dtype=np.float32)
    return full_mask


def _guided_filter_bbox_mask(image, mask, bbox, radius=60, eps=0.0001):
    """Run guided filter only around the SAM3 bbox.

    cv2.ximgproc.guidedFilter on a full-resolution photo can stall the UI spinner
    for large images even though the bbox prompt is local. The bbox is still
    clipped after filtering, so padding only gives the filter local context.
    """
    x0, y0, x1, y1 = _expanded_bbox_xyxy(bbox, image.shape, radius)
    out = np.zeros(np.asarray(mask).shape[:2], dtype=np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    start = time.perf_counter()
    refined = cutout_guided.create_cutout_mask_guided(
        image[y0:y1, x0:x1],
        np.asarray(mask, dtype=np.float32)[y0:y1, x0:x1],
        radius=radius,
        eps=eps,
    )
    _LOGGER.info(
        "SAM3 guided filter elapsed=%.1fms roi=(%d,%d)-(%d,%d) roi_shape=%s radius=%s eps=%s",
        _elapsed_ms(start),
        x0,
        y0,
        x1,
        y1,
        image[y0:y1, x0:x1].shape[:2],
        radius,
        eps,
    )
    out[y0:y1, x0:x1] = refined
    return out


def _apply_sam3_bbox_limit(mask, bbox, *, stage):
    if not _sam3_bbox_clip_enabled():
        return mask
    mask_arr = np.asarray(mask, dtype=np.float32)
    total, inside, outside, max_value = _mask_bbox_stats(mask_arr, bbox)
    _LOGGER.info(
        "SAM3 %s mask stats total=%d inside_bbox=%d outside_bbox=%d max=%.4f bbox=%s",
        stage,
        total,
        inside,
        outside,
        max_value,
        bbox,
    )
    if outside:
        _LOGGER.info("SAM3 %s mask had %d px outside bbox=%s; clipping", stage, outside, bbox)
    return _zero_outside_bbox(mask_arr, bbox)

def delete_faces():
    global _faces
    _faces = None


def predict_sam3_bbox(img: np.ndarray, bbox, invert: bool) -> np.ndarray:
    global _sam3_processor
    from helpers import sam3_helper

    total_start = time.perf_counter()
    source_image_key = _array_cache_key(img)
    phase_start = time.perf_counter()
    ai_img = aiutils.to_ai_display_rgb(img)
    _LOGGER.info("SAM3 ai display input elapsed=%.1fms image_shape=%s", _elapsed_ms(phase_start), ai_img.shape[:2])
    phase_start = time.perf_counter()
    bbox = _clip_bbox_xywh(bbox, ai_img.shape)
    if bbox[2] <= 0 or bbox[3] <= 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    _LOGGER.info("SAM3 bbox clipped elapsed=%.1fms bbox=%s", _elapsed_ms(phase_start), bbox)

    with _sam3_lock:
        phase_start = time.perf_counter()
        if _sam3_processor is None:
            _sam3_processor = sam3_helper.setup_sam3(config.get_config("gpu_device"))
            _LOGGER.info("SAM3 setup elapsed=%.1fms", _elapsed_ms(phase_start))
        phase_start = time.perf_counter()
        mask_original = _predict_sam3_bbox_roi(sam3_helper, _sam3_processor, ai_img, bbox, source_image_key)
        _LOGGER.info("SAM3 predict locked section elapsed=%.1fms", _elapsed_ms(phase_start))
    phase_start = time.perf_counter()
    mask_original = _apply_sam3_bbox_limit(mask_original, bbox, stage="raw")
    _LOGGER.info("SAM3 raw bbox limit elapsed=%.1fms", _elapsed_ms(phase_start))

    phase_start = time.perf_counter()
    mask_original = _guided_filter_bbox_mask(ai_img, mask_original, bbox, radius=60, eps=0.0001)
    _LOGGER.info("SAM3 guided phase elapsed=%.1fms", _elapsed_ms(phase_start))
    phase_start = time.perf_counter()
    mask_original = _apply_sam3_bbox_limit(mask_original, bbox, stage="guided")
    _LOGGER.info("SAM3 guided bbox limit elapsed=%.1fms", _elapsed_ms(phase_start))
    if invert:
        phase_start = time.perf_counter()
        mask_original = 1 - mask_original
        _LOGGER.info("SAM3 invert elapsed=%.1fms", _elapsed_ms(phase_start))
    _LOGGER.info("SAM3 bbox total elapsed=%.1fms invert=%s final_shape=%s", _elapsed_ms(total_start), invert, mask_original.shape)
    return mask_original


def predict_sam3_text(img: np.ndarray, text: str, invert: bool) -> np.ndarray:
    global _sam3_processor
    from helpers import sam3_helper

    ai_img = aiutils.to_ai_display_rgb(img)
    with _sam3_lock:
        if _sam3_processor is None:
            _sam3_processor = sam3_helper.setup_sam3(config.get_config("gpu_device"))
        mask_original = sam3_helper.predict_sam3_for_text(_sam3_processor, ai_img, text)

    mask_original = cutout_guided.create_cutout_mask_guided(
        ai_img, mask_original, radius=60, eps=0.0001
    )
    if invert:
        mask_original = 1 - mask_original
    return mask_original


def predict_depth_map(img: np.ndarray) -> np.ndarray:
    global _depth_model
    from helpers import depth_pro_helper

    ai_img = aiutils.to_ai_display_rgb(img)
    if _depth_model is None:
        _depth_model = depth_pro_helper.setup_model(device=config.get_config("gpu_device"))
    return depth_pro_helper.predict_model(_depth_model, ai_img)


def predict_face_mask(img: np.ndarray, exclude_names: list) -> np.ndarray:
    global _faces
    from helpers import facer_helper

    ai_img = aiutils.to_ai_display_rgb(img)
    if _faces is None:
        _faces = facer_helper.create_faces(ai_img, device="cpu")

    if _faces == 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    result = facer_helper.draw_face_mask(_faces, exclude_names)
    return cutout_guided.create_cutout_mask_guided(ai_img, result, radius=60, eps=0.0001)
