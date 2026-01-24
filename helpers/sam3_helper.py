
import torch
import numpy as np
import cv2
from SAM3.sam3.model_builder import build_sam3_image_model
from SAM3.sam3.model.box_ops import box_xywh_to_cxcywh
from SAM3.sam3.model.sam3_image_processor import Sam3Processor
from SAM3.sam3.visualization_utils import normalize_bbox

RESIZE_FACTOR = 1.0

__model = None

def setup_sam3(device='cpu'):
    global __model
    if __model is None:
        __model = build_sam3_image_model(bpe_path="SAM3/assets/bpe_simple_vocab_16e6.txt.gz", checkpoint_path="checkpoints/sam3.pt", device=device)
    processor = Sam3Processor(__model)
    sam3_dict = {"processor": processor, "image": None, "inference_state": None}
    return sam3_dict

def predict_sam3_for_bbox(sam3_dict, image, bbox):
    processor = sam3_dict["processor"]
    org_h, org_w = image.shape[0:2]

    if sam3_dict["image"] is not image:
        sam3_dict["image"] = image
        image = cv2.resize(image, (int(org_w * RESIZE_FACTOR), int(org_h * RESIZE_FACTOR)))
        sam3_dict["inference_state"] = processor.set_image(image)
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
    inference_state = sam3_dict["inference_state"]

    processor.reset_all_prompts(inference_state)
    results = processor.set_text_prompt(state=inference_state, prompt=text)
    if len(results["masks"]) == 0:
        return np.zeros((org_h, org_w), dtype=np.float32)

    mask = results["masks"][0].squeeze(0).cpu().numpy()
    mask = np.array(mask, dtype=np.float32)
    mask = cv2.resize(mask, (org_w, org_h))

    return mask