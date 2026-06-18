"""
Mask2 の推論モデルキャッシュ。UI（mask_editor2）とヘッドレス export で共有する。
"""
from __future__ import annotations

from threading import RLock

import config
import numpy as np

import cores.mask2.cutout_guided as cutout_guided

_sam3_processor = None
_sam3_lock = RLock()
_depth_model = None
_faces = None

DEPTH_MAP_ALGORITHM_VERSION = 2

def delete_faces():
    global _faces
    _faces = None


def predict_sam3_bbox(img: np.ndarray, bbox, invert: bool) -> np.ndarray:
    global _sam3_processor
    from helpers import sam3_helper

    bbox = [int(x) for x in bbox]
    if bbox[0] == bbox[0] + bbox[2] or bbox[1] == bbox[1] + bbox[3]:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    with _sam3_lock:
        if _sam3_processor is None:
            _sam3_processor = sam3_helper.setup_sam3(config.get_config("gpu_device"))
        mask_original = sam3_helper.predict_sam3_for_bbox(_sam3_processor, img, bbox)

    mask_original = cutout_guided.create_cutout_mask_guided(
        img, mask_original, radius=60, eps=0.0001
    )
    if invert:
        mask_original = 1 - mask_original
    return mask_original


def predict_sam3_text(img: np.ndarray, text: str, invert: bool) -> np.ndarray:
    global _sam3_processor
    from helpers import sam3_helper

    with _sam3_lock:
        if _sam3_processor is None:
            _sam3_processor = sam3_helper.setup_sam3(config.get_config("gpu_device"))
        mask_original = sam3_helper.predict_sam3_for_text(_sam3_processor, img, text)

    mask_original = cutout_guided.create_cutout_mask_guided(
        img, mask_original, radius=60, eps=0.0001
    )
    if invert:
        mask_original = 1 - mask_original
    return mask_original


def predict_depth_map(img: np.ndarray) -> np.ndarray:
    global _depth_model
    from helpers import depth_pro_helper

    if _depth_model is None:
        _depth_model = depth_pro_helper.setup_model(device=config.get_config("gpu_device"))
    return depth_pro_helper.predict_model(_depth_model, img)


def predict_face_mask(img: np.ndarray, exclude_names: list) -> np.ndarray:
    global _faces
    from helpers import facer_helper

    if _faces is None:
        _faces = facer_helper.create_faces(img, device="cpu")

    if _faces == 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    result = facer_helper.draw_face_mask(_faces, exclude_names)
    return cutout_guided.create_cutout_mask_guided(img, result, radius=60, eps=0.0001)
