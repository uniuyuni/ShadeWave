import logging
from pathlib import Path

import torch
import numpy as np
import cv2
from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

# setup.sh で clone した SAM3 リポジトリ直下の assets（pip install -e ではパッケージ外）
_PLATYPUS_ROOT = Path(__file__).resolve().parents[1]
_BPE_VOCAB = _PLATYPUS_ROOT / "SAM3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

RESIZE_FACTOR = 1.0

__model = None

_logger = logging.getLogger(__name__)


def setup_sam3(device="cpu"):
    global __model
    if isinstance(device, str):
        device = device.strip().lower()
    torch_device = torch.device(device)
    if __model is None:
        __model = build_sam3_image_model(
            bpe_path=str(_BPE_VOCAB),
            checkpoint_path="checkpoints/sam3.pt",
            device=device,
        )
    elif next(__model.parameters()).device != torch_device:
        # config の gpu_device が変わったときにモデルと入力のデバイスを揃える
        __model = __model.to(torch_device)
    # device を省略すると get_default_device() が MPS を選び、モデルが CPU のときに不一致になる
    processor = Sam3Processor(__model, device=torch_device)
    param_dev = next(__model.parameters()).device
    if param_dev != torch_device or processor.device != torch_device:
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
        if vf.device != exp:
            _logger.warning(
                "SAM3: backbone output device %s != processor %s",
                vf.device,
                exp,
            )
    sam3_dict["_device_logged"] = True

def predict_sam3_for_bbox(sam3_dict, image, bbox):
    processor = sam3_dict["processor"]
    org_h, org_w = image.shape[0:2]

    if sam3_dict["image"] is not image:
        sam3_dict["image"] = image
        image = cv2.resize(image, (int(org_w * RESIZE_FACTOR), int(org_h * RESIZE_FACTOR)))
        sam3_dict["inference_state"] = processor.set_image(image)
        _log_backbone_device_once(sam3_dict)
    inference_state = sam3_dict["inference_state"]

    box_input_xywh = torch.tensor(bbox).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, org_w, org_h).flatten().tolist()

    processor.reset_all_prompts(inference_state)
    results = processor.add_geometric_prompt(state=inference_state, box=norm_box_cxcywh, label=True)
    if len(results["masks"]) == 0:
        return np.zeros((org_h, org_w), dtype=np.float32)

    mask = results["masks"][0].squeeze(0).cpu().numpy()
    mask = np.array(mask, dtype=np.float32)
    mask = cv2.resize(mask, (org_w, org_h))

    return mask

def predict_sam3_for_text(sam3_dict, image, text):
    processor = sam3_dict["processor"]
    org_h, org_w = image.shape[0:2]

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