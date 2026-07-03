"""
Runware Object Eraser helper.

Configuration:
    RUNWARE_API_KEY: Required API key.
    RUNWARE_API_URL: Optional API endpoint. Defaults to https://api.runware.ai/v1.
    RUNWARE_OBJECT_ERASER_MODEL: Optional model id. Defaults to runware:300@1.
    RUNWARE_REQUEST_TIMEOUT: Optional request timeout in seconds. Defaults to 120.
    RUNWARE_OBJECT_ERASER_STEPS: Optional denoising steps. Defaults to 4.
    RUNWARE_OBJECT_ERASER_CFG: Optional CFG scale. Defaults to 1.
    RUNWARE_OBJECT_ERASER_BLEND_DILATE: Optional edit-mask expansion in pixels. Defaults to 8.
    RUNWARE_OBJECT_ERASER_BLEND_BLUR: Optional edit-mask blur radius in pixels. Defaults to 5.

Request shape:
    The "Object Eraser" model (runware:300@1) is documented at
    https://runware.ai/docs/models/object-eraser with this exact shape (NOT the
    generic seedImage/maskImage imageInference-inpainting shape used by SD-style
    models): a nested "inputs" object plus deliveryMethod=sync.
    - inputs.image: original RGB image as PNG data URI.
    - inputs.mask: binary mask as PNG data URI, white pixels are erased.

Response handling:
    Uses outputType=base64Data to avoid depending on result URL downloads.
    includeCost=True asks Runware to include the task cost in the response.
"""

import base64
import logging
import os
import uuid

import cv2
import numpy as np
import requests

import cores.splitimage as splitimage
import utils.aiutils as aiutils


API_URL = os.environ.get("RUNWARE_API_URL", "https://api.runware.ai/v1")
MODEL = os.environ.get("RUNWARE_OBJECT_ERASER_MODEL", "runware:300@1")
DEFAULT_PROMPT = "Remove the masked unwanted object and naturally continue the surrounding background."


def setup():
    return os.environ.get("RUNWARE_API_KEY")


def _request_timeout():
    try:
        return float(os.environ.get("RUNWARE_REQUEST_TIMEOUT", "120"))
    except (TypeError, ValueError):
        return 120.0


def _image_data_uri(image):
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    image_u8 = (image * 255.0).round().astype(np.uint8)
    image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise ValueError("Image encoding failed.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")


def _mask_data_uri(mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_u8 = ((mask > 0.0).astype(np.uint8) * 255)
    ok, encoded = cv2.imencode(".png", mask_u8, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise ValueError("Mask encoding failed.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")


def _decode_image_data(data):
    if not data:
        return None
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image_bgr = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _extract_result_image(response_json):
    items = response_json
    if isinstance(response_json, dict):
        if response_json.get("errors"):
            # Runware can report task-level failures with an HTTP 200 (the errors
            # array names the actual problem), so this must raise too, not just
            # the HTTP-status branch in predict().
            raise RuntimeError(f"Runware error: {response_json['errors']}")
        items = response_json.get("data", response_json.get("result", response_json))
    if isinstance(items, dict):
        items = [items]
    for item in items or []:
        if not isinstance(item, dict):
            continue
        image = _decode_image_data(item.get("imageBase64Data") or item.get("imageDataURI"))
        if image is not None:
            if item.get("cost") is not None:
                logging.info("Runware cost: $%s", item['cost'])
            return image
        if item.get("imageURL"):
            return _download_image(item["imageURL"])
    return None


def _ensure_result_size(result, target_shape):
    if result is None:
        return None
    if result.shape[:2] == target_shape[:2]:
        return result
    return cv2.resize(result, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def _soft_edit_mask(mask, dilate_px=4, blur_px=3):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    kernel = np.ones((max(1, int(dilate_px)), max(1, int(dilate_px))), dtype=np.uint8)
    edit_mask = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    if blur_px > 0:
        ksize = int(blur_px) * 2 + 1
        edit_mask = cv2.GaussianBlur(edit_mask, (ksize, ksize), 0)
    return np.clip(edit_mask, 0.0, 1.0)[..., np.newaxis]


def _context_match_result(result, original, mask, ring_px=32):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return result

    kernel = np.ones((max(1, int(ring_px)), max(1, int(ring_px))), dtype=np.uint8)
    outer = cv2.dilate(mask_u8, kernel, iterations=1)
    ring = (outer > 0) & (mask_u8 == 0)
    if int(np.count_nonzero(ring)) < 16:
        return result

    matched = np.asarray(result, dtype=np.float32).copy()
    original = np.asarray(original, dtype=np.float32)
    for c in range(3):
        src = matched[..., c][ring]
        ref = original[..., c][ring]
        src_std = float(np.std(src))
        src_mean = float(np.mean(src))
        ref_mean = float(np.mean(ref))
        if src_std < 1e-4:
            matched[..., c] = matched[..., c] + (ref_mean - src_mean)
            continue
        matched[..., c] = (matched[..., c] - src_mean) * (float(np.std(ref)) / src_std) + ref_mean
    return np.clip(matched, 0.0, 1.0)


def predict(api_key, image, mask, prompt=None):
    if not api_key:
        raise RuntimeError("RUNWARE_API_KEY is not set.")

    payload = [{
        "taskType": "imageInference",
        "taskUUID": str(uuid.uuid4()),
        "deliveryMethod": "sync",
        "outputType": "base64Data",
        "outputFormat": "PNG",
        "includeCost": True,
        "numberResults": 1,
        "inputs": {
            "image": _image_data_uri(image),
            "mask": _mask_data_uri(mask),
        },
        "model": MODEL,
        "positivePrompt": prompt or DEFAULT_PROMPT,
        "steps": int(os.environ.get("RUNWARE_OBJECT_ERASER_STEPS", "4")),
        "CFGScale": float(os.environ.get("RUNWARE_OBJECT_ERASER_CFG", "1")),
    }]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = None
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=_request_timeout())
        response.raise_for_status()
    except Exception as e:
        # Surface Runware's actual error body (it names the offending parameter),
        # otherwise a bare "400 Bad Request" tells us nothing. This is shown to
        # the user verbatim in a failure dialog (see main.py update_async_results),
        # so raise rather than silently swallowing and returning None.
        body = None
        if response is not None:
            try:
                body = response.text
            except Exception:
                body = None
        message = f"Runware request failed (status={getattr(response, 'status_code', None)}): {body or e}"
        logging.exception(message)
        raise RuntimeError(message) from e

    result = _extract_result_image(response.json())
    if result is None:
        message = f"Runware returned no image result: {response.text[:500]}"
        logging.error(message)
        raise RuntimeError(message)
    return result


def predict_helper(api_key, image, mask, bbox, prompt=None):
    target_width = max(1024, (bbox[2] * 2 + 7) // 8 * 8)
    target_height = max(1024, (bbox[3] * 2 + 7) // 8 * 8)
    x, y, w, h = aiutils.calculate_expanded_crop(
        image.shape[1], image.shape[0], bbox[0], bbox[1], bbox[2], bbox[3], target_width, target_height
    )

    crop_image = image[y:y + h, x:x + w, :]
    crop_mask = mask[y:y + h, x:x + w, np.newaxis].astype(np.float32)
    blocks, split_info = splitimage.split_image_with_overlap(crop_image, 1024, 1024, 192)
    mask_blocks, _ = splitimage.split_image_with_overlap(np.repeat(crop_mask, 3, axis=2), 1024, 1024, 192)

    predict_blocks = []
    for i, block in enumerate(blocks):
        block_mask = mask_blocks[i][..., 0]
        if np.any(block_mask > 0):
            logging.info("Runware object erase %s/%s %s.", i + 1, len(blocks), block.shape)
            # predict() now raises on failure (see below) instead of silently
            # returning None, so a failed request stops the whole erase here and
            # propagates up to the caller rather than being masked as a no-op.
            result = _ensure_result_size(predict(api_key, block, block_mask, prompt), block.shape)
            result = _context_match_result(result, block, block_mask)
            edit_mask = _soft_edit_mask(
                block_mask,
                dilate_px=int(os.environ.get("RUNWARE_OBJECT_ERASER_BLEND_DILATE", "8")),
                blur_px=int(os.environ.get("RUNWARE_OBJECT_ERASER_BLEND_BLUR", "5")),
            )
            predict_blocks.append(result * edit_mask + block * (1.0 - edit_mask))
        else:
            predict_blocks.append(block)

    combined = splitimage.combine_image_with_overlap(predict_blocks, split_info)
    blend_mask = np.ones_like(combined[..., 0])
    for i in range(192):
        alpha = (i + 1) / 192
        blend_mask[i, :] *= alpha
        blend_mask[-(i + 1), :] *= alpha
        blend_mask[:, i] *= alpha
        blend_mask[:, -(i + 1)] *= alpha
    image[y:y + h, x:x + w, :] = combined * blend_mask[..., np.newaxis] + crop_image * (1.0 - blend_mask[..., np.newaxis])
    return image
