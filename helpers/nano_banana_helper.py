import os
import logging
from io import BytesIO

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image

import cores.splitimage as splitimage
import utils.aiutils as aiutils
import cores.core as core


DEFAULT_MODEL = os.environ.get("PLATYPUS_NANO_BANANA_MODEL", "gemini-3.1-flash-image")
DEFAULT_EDIT_MODEL = os.environ.get("PLATYPUS_NANO_BANANA_EDIT_MODEL", "imagen-3.0-capability-001")
FALLBACK_MODELS = tuple(
    model.strip()
    for model in os.environ.get("PLATYPUS_NANO_BANANA_FALLBACK_MODELS", "gemini-2.5-flash-image").split(",")
    if model.strip()
)


def setup():
    return genai.Client()


def _image_to_pil(image):
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray((image * 255.0).round().astype(np.uint8), mode="RGB")


def _mask_to_pil(mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")


def _make_red_marker_image(image, mask):
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    marker = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return image * (1.0 - mask) + marker * mask


def _pil_to_genai_image(image, mime_type="image/png"):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return types.Image(imageBytes=buffer.getvalue(), mimeType=mime_type)


def _ensure_result_size(result_array, target_shape):
    if result_array is None:
        return None
    if result_array.shape[:2] == target_shape[:2]:
        return result_array
    return cv2.resize(result_array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def _soft_edit_mask(mask, dilate_px=4, blur_px=3):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if not np.any(mask > 0):
        return mask[..., np.newaxis]

    kernel_size = max(1, int(dilate_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    edit_mask = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(np.float32)

    blur_px = max(0, int(blur_px))
    if blur_px > 0:
        ksize = blur_px * 2 + 1
        edit_mask = cv2.GaussianBlur(edit_mask, (ksize, ksize), 0)
        edit_mask = np.clip(edit_mask, 0.0, 1.0)

    return edit_mask[..., np.newaxis]


def _iter_response_parts(response):
    parts = getattr(response, "parts", None)
    if parts is not None:
        yield from parts
        return

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            yield part


def _part_to_image(part):
    if getattr(part, "text", None) is not None:
        logging.debug("%s", part.text)
        return None

    as_image = getattr(part, "as_image", None)
    if callable(as_image):
        try:
            image = as_image()
            pil_image = _genai_image_to_pil(image)
            if pil_image is not None:
                return pil_image
        except Exception:
            pass

    inline_data = getattr(part, "inline_data", None)
    if inline_data is None:
        inline_data = getattr(part, "inlineData", None)
    if inline_data is None:
        return None

    data = getattr(inline_data, "data", None)
    if data is None:
        return None
    return Image.open(BytesIO(data))


def _genai_image_to_pil(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    pil_image = getattr(image, "_pil_image", None)
    if pil_image is not None:
        return pil_image.convert("RGB")

    image_bytes = getattr(image, "image_bytes", None)
    if image_bytes is None:
        image_bytes = getattr(image, "imageBytes", None)
    if image_bytes is not None:
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    return None


def _extract_image(response):
    for part in _iter_response_parts(response):
        image = _part_to_image(part)
        if image is not None:
            return image.convert("RGB")
    return None


def _extract_edit_image(response):
    generated_images = getattr(response, "generated_images", None)
    if generated_images is None:
        generated_images = getattr(response, "generatedImages", None)
    for generated_image in generated_images or []:
        image = getattr(generated_image, "image", None)
        pil_image = _genai_image_to_pil(image)
        if pil_image is not None:
            return pil_image
    return None


def _build_prompt(user_prompt=None):
    base_prompt = """
Only edit the pure red masked area.
The pure red color (#FF0000) is an editing mask, not an object to keep.
Remove the red masked content and fill it naturally so it blends with the surrounding area.
Do not change anything outside the red area.
Remove all red pixels.
"""
    if user_prompt and str(user_prompt).strip():
        return base_prompt + "\nAdditional user instruction:\n" + str(user_prompt).strip()
    return base_prompt


def _build_edit_prompt(user_prompt=None):
    prompt = """
Remove the unwanted content inside the mask and fill it by continuing the surrounding background.
Match nearby color, gradient, texture, grain, lighting, perspective, and sharpness.
If the mask covers a wall, continue the same wall surface. Do not create sky or outdoor scenery.
If the mask covers sky, continue the nearby sky gradient smoothly and avoid adding new cloud shapes.
Do not add new objects, text, outlines, seams, patches, or style changes.
"""
    if user_prompt and str(user_prompt).strip():
        return prompt + "\nAdditional user instruction:\n" + str(user_prompt).strip()
    return prompt


def predict_helper(client, image, mask, bbox, prompt=None):
    target_width = max(1024, (bbox[2] * 2 + 7) // 8 * 8)
    target_height = max(1024, (bbox[3] * 2 + 7) // 8 * 8)

    x, y, w, h = aiutils.calculate_expanded_crop(
        img_width=image.shape[1],
        img_height=image.shape[0],
        x=bbox[0],
        y=bbox[1],
        w=bbox[2],
        h=bbox[3],
        width=target_width,
        height=target_height,
    )

    crop_image = image[y:y + h, x:x + w, :]
    crop_mask = mask[y:y + h, x:x + w, np.newaxis].astype(np.float32)
    marker_image = _make_red_marker_image(crop_image, crop_mask)

    blocks, split_info = splitimage.split_image_with_overlap(marker_image, 1024, 1024, 192)
    mask_blocks, _ = splitimage.split_image_with_overlap(
        np.repeat(crop_mask, 3, axis=2),
        1024,
        1024,
        192,
    )

    predict_blocks = []
    for i, block in enumerate(blocks):
        block_mask = mask_blocks[i][..., 0]
        if np.any(block_mask > 0):
            logging.info("Nano banana inpainting predict %s/%s %s.", i + 1, len(blocks), block.shape)
            pre_image = predict(client, block, block_mask, prompt=prompt)
            pre_image = _ensure_result_size(pre_image, block.shape)
            if pre_image is None:
                predict_blocks.append(block)
                continue
            edit_mask = _soft_edit_mask(block_mask)
            predict_blocks.append(pre_image * edit_mask + block * (1.0 - edit_mask))
        else:
            predict_blocks.append(block)

    combine = splitimage.combine_image_with_overlap(predict_blocks, split_info)

    blend_width = min(192, max(1, combine.shape[0] // 2), max(1, combine.shape[1] // 2))
    blend_mask = np.ones_like(combine[..., 0])
    for i in range(blend_width):
        alpha = (i + 1) / blend_width
        blend_mask[i, :] *= alpha
        blend_mask[-(i + 1), :] *= alpha
        blend_mask[:, i] *= alpha
        blend_mask[:, -(i + 1)] *= alpha
    blend_mask = blend_mask[..., np.newaxis]

    image_crop = image[y:y + h, x:x + w, :]
    image[y:y + h, x:x + w, :] = combine * blend_mask + image_crop * (1.0 - blend_mask)
    return image


def predict_edit_image(client, fp32_image, mask, prompt=None, model=None):
    pil_image = _image_to_pil(fp32_image)
    pil_mask = _mask_to_pil(mask)
    raw_ref_image = types.RawReferenceImage(
        referenceId=1,
        referenceImage=_pil_to_genai_image(pil_image),
    )
    mask_ref_image = types.MaskReferenceImage(
        referenceId=2,
        referenceImage=_pil_to_genai_image(pil_mask),
        config=types.MaskReferenceConfig(
            maskMode=types.MaskReferenceMode.MASK_MODE_USER_PROVIDED,
            maskDilation=0.0,
        ),
    )

    model_name = model or DEFAULT_EDIT_MODEL
    response = client.models.edit_image(
        model=model_name,
        prompt=_build_edit_prompt(prompt),
        reference_images=[raw_ref_image, mask_ref_image],
        config=types.EditImageConfig(
            editMode=types.EditMode.EDIT_MODE_INPAINT_REMOVAL,
            numberOfImages=1,
            includeRaiReason=True,
            outputMimeType="image/png",
        ),
    )
    result_image = _extract_edit_image(response)
    if result_image is None:
        return None
    result = np.asarray(result_image).astype(np.float32) / 255.0
    logging.info("Nano banana edit_image done with %s. Output size: %s", model_name, result.shape)
    return result


def predict_generate_content(client, fp32_image, mask=None, prompt=None, model=None):
    pil_image = _image_to_pil(fp32_image)
    contents = [_build_prompt(prompt), pil_image]
    models = [model or DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != (model or DEFAULT_MODEL)]

    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    responseModalities=["TEXT", "IMAGE"],
                ),
            )
            result_image = _extract_image(response)
            if result_image is None:
                logging.warning("Nano banana returned no image for model %s.", model_name)
                continue
            result = np.asarray(result_image).astype(np.float32) / 255.0
            logging.info("Nano banana done with %s. Output size: %s", model_name, result.shape)
            return result
        except Exception:
            logging.exception("Nano banana failed with %s", model_name)

    return fp32_image


def predict(client, fp32_image, mask, prompt=None, model=None):
    return predict_generate_content(client, fp32_image, mask, prompt=prompt, model=model)


if __name__ == "__main__":
    image = Image.open("../test/X-T5 Room image.jpg").convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0

    mask = Image.open("../test/X-T5 Room mask.png").convert("L")
    mask = np.array(mask).astype(np.float32) / 255.0

    bboxs = core.get_multiple_mask_bbox(mask)
    if len(bboxs) > 0:
        client = setup()
        predict_image = predict_helper(client, image, mask, bboxs[0])
        Image.fromarray((predict_image * 255).astype(np.uint8)).save("../test/X-T5 Room complete.jpg")
