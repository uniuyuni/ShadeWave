import logging
import time
from threading import RLock

import torch
import numpy as np
import cv2
from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from helpers import sam3_coreml_backbone_helper

RESIZE_FACTOR = 1.0

__model = None
__model_lock = RLock()

_logger = logging.getLogger(__name__)


def _elapsed_ms(start):
    return (time.perf_counter() - start) * 1000.0


def _device_equal(a: torch.device, b: torch.device) -> bool:
    """torch.device の同値比較。`mps` と `mps:0` のように index が None vs 0 の
    違いを吸収する (PyTorch の素の `==` は False を返してしまうため)。"""
    if a.type != b.type:
        return False
    if a.type == 'cpu':
        return True
    a_idx = a.index if a.index is not None else 0
    b_idx = b.index if b.index is not None else 0
    return a_idx == b_idx


def _normalize_bbox(bbox_xywh, img_w, img_h):
    normalized_bbox = bbox_xywh.clone()
    normalized_bbox[..., 0] /= img_w
    normalized_bbox[..., 1] /= img_h
    normalized_bbox[..., 2] /= img_w
    normalized_bbox[..., 3] /= img_h
    return normalized_bbox


def _clip_bbox_xywh_int(bbox, img_w, img_h):
    x, y, w, h = [float(v) for v in bbox]
    x0 = int(np.floor(min(x, x + w)))
    y0 = int(np.floor(min(y, y + h)))
    x1 = int(np.ceil(max(x, x + w)))
    y1 = int(np.ceil(max(y, y + h)))
    x0 = min(max(x0, 0), img_w)
    y0 = min(max(y0, 0), img_h)
    x1 = min(max(x1, 0), img_w)
    y1 = min(max(y1, 0), img_h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _materialize_mask_tensor(mask_tensor, org_w, org_h):
    mask = mask_tensor.squeeze(0).cpu().numpy()
    mask = np.array(mask, dtype=np.float32)
    return cv2.resize(mask, (org_w, org_h))


def _select_bbox_mask_candidate(mask_tensors, bbox, org_w, org_h):
    x, y, w, h = _clip_bbox_xywh_int(bbox, org_w, org_h)
    best = None
    best_score = None
    scores = []
    for index, mask_tensor in enumerate(mask_tensors):
        start = time.perf_counter()
        mask = _materialize_mask_tensor(mask_tensor, org_w, org_h)
        materialize_elapsed = _elapsed_ms(start)
        total = int(np.count_nonzero(mask > 0.001))
        inside = int(np.count_nonzero(mask[y:y + h, x:x + w] > 0.001)) if w > 0 and h > 0 else 0
        outside = max(0, total - inside)
        max_value = float(np.nanmax(mask)) if mask.size else 0.0
        score = (inside, -outside, max_value)
        scores.append((index, inside, outside, max_value, materialize_elapsed))
        if best_score is None or score > best_score:
            best_score = score
            best = mask
    _logger.info(
        "SAM3 bbox mask candidates bbox=%s selected=%s scores=%s",
        [x, y, w, h],
        None if best_score is None else best_score,
        scores,
    )
    return best if best is not None else np.zeros((org_h, org_w), dtype=np.float32)


def setup_sam3(device="cpu"):
    global __model
    if isinstance(device, str):
        device = device.strip().lower()
    torch_device = torch.device(device)
    with __model_lock:
        if __model is None:
            # bpe_path 省略時は公式パッケージ同梱の
            # sam3/assets/bpe_simple_vocab_16e6.txt.gz が pkg_resources で解決される
            __model = build_sam3_image_model(
                checkpoint_path="checkpoints/sam3.1_multiplex.pt",
                device=device,
            )
        elif not _device_equal(next(__model.parameters()).device, torch_device):
            # config の gpu_device が変わったときにモデルと入力のデバイスを揃える
            __model = __model.to(torch_device)
    # device を省略すると get_default_device() が MPS を選び、モデルが CPU のときに不一致になる
    processor = Sam3Processor(__model, device=torch_device)
    sam3_coreml_backbone_helper.install(processor, __model)
    param_dev = next(__model.parameters()).device
    if not _device_equal(param_dev, torch_device) or not _device_equal(processor.device, torch_device):
        _logger.error(
            "SAM3 device mismatch: requested=%s model=%s processor=%s",
            torch_device,
            param_dev,
            processor.device,
        )
    else:
        _logger.info(
            "SAM3 ready: requested=%s model.parameters=%s Sam3Processor.device=%s",
            torch_device,
            param_dev,
            processor.device,
        )
    sam3_dict = {
        "processor": processor,
        "image": None,
        "image_key": None,
        "inference_state": None,
        "_device_logged": False,
    }
    return sam3_dict


def _log_backbone_device_once(sam3_dict):
    """set_image 後に backbone 出力のデバイスを 1 回だけログ（MPS が効いているか確認用）。"""
    if sam3_dict.get("_device_logged"):
        return
    st = sam3_dict.get("inference_state") or {}
    bo = st.get("backbone_out") or {}
    vf = bo.get("vision_features")
    if vf is not None:
        exp = sam3_dict["processor"].device
        _logger.info(
            "SAM3 backbone vision_features device=%s (expect %s) mps_available=%s",
            vf.device,
            exp,
            torch.backends.mps.is_available(),
        )
        if not _device_equal(vf.device, exp):
            _logger.warning(
                "SAM3: backbone output device %s != processor %s",
                vf.device,
                exp,
            )
    sam3_dict["_device_logged"] = True

def predict_sam3_for_bbox(sam3_dict, image, bbox, image_key=None):
    processor = sam3_dict["processor"]
    org_h, org_w = image.shape[0:2]
    cache_key = image_key if image_key is not None else ("object", id(image))

    with torch.inference_mode():
        if sam3_dict.get("image_key") != cache_key:
            start = time.perf_counter()
            sam3_dict["image"] = image
            sam3_dict["image_key"] = cache_key
            image = cv2.resize(image, (int(org_w * RESIZE_FACTOR), int(org_h * RESIZE_FACTOR)))
            resize_elapsed = _elapsed_ms(start)
            start = time.perf_counter()
            sam3_dict["inference_state"] = processor.set_image(image)
            _logger.info(
                "SAM3 set_image elapsed=%.1fms resize_elapsed=%.1fms image_shape=%s",
                _elapsed_ms(start),
                resize_elapsed,
                image.shape[:2],
            )
            _log_backbone_device_once(sam3_dict)
        else:
            _logger.info("SAM3 set_image skipped cache_key=%s image_shape=%s", cache_key, image.shape[:2])
        inference_state = sam3_dict["inference_state"]

        start = time.perf_counter()
        box_input_xywh = torch.tensor(bbox).view(-1, 4)
        box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
        norm_box_cxcywh = _normalize_bbox(box_input_cxcywh, org_w, org_h).flatten().tolist()
        _logger.info("SAM3 bbox normalize elapsed=%.1fms norm_box=%s", _elapsed_ms(start), norm_box_cxcywh)

        start = time.perf_counter()
        processor.reset_all_prompts(inference_state)
        reset_elapsed = _elapsed_ms(start)
        start = time.perf_counter()
        results = processor.add_geometric_prompt(state=inference_state, box=norm_box_cxcywh, label=True)
        _logger.info(
            "SAM3 add_geometric_prompt elapsed=%.1fms reset_elapsed=%.1fms masks=%d",
            _elapsed_ms(start),
            reset_elapsed,
            len(results.get("masks", [])),
        )
    if len(results["masks"]) == 0:
        return np.zeros((org_h, org_w), dtype=np.float32)

    start = time.perf_counter()
    mask = _select_bbox_mask_candidate(results["masks"], bbox, org_w, org_h)
    _logger.info(
        "SAM3 mask materialize elapsed=%.1fms output_shape=%s",
        _elapsed_ms(start),
        mask.shape,
    )

    return mask

def predict_sam3_for_text(sam3_dict, image, text):
    processor = sam3_dict["processor"]
    org_h, org_w = image.shape[0:2]

    with torch.inference_mode():
        if sam3_dict["image"] is not image:
            sam3_dict["image"] = image
            image = cv2.resize(image, (int(org_w * RESIZE_FACTOR), int(org_h * RESIZE_FACTOR)))
            sam3_dict["inference_state"] = processor.set_image(image)
            _log_backbone_device_once(sam3_dict)
        inference_state = sam3_dict["inference_state"]

        processor.reset_all_prompts(inference_state)
        results = processor.set_text_prompt(state=inference_state, prompt=text)
    if len(results["masks"]) == 0:
        return np.zeros((org_h, org_w), dtype=np.float32)

    mask = results["masks"][0].squeeze(0).cpu().numpy()
    mask = np.array(mask, dtype=np.float32)
    mask = cv2.resize(mask, (org_w, org_h))

    return mask
