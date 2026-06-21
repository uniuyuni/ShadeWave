from __future__ import annotations

import copy
import hashlib
import os
from typing import Any

import numpy as np

from cores import pmck_store
from enums import ImageFidelity
import params
import utils.utils as utils


AI_NOISE_KIND = "ai_noise_reduction"
_AI_NOISE_EXIF_KEYS = {
    "RawImageCropTopLeft",
    "RawImageCroppedSize",
    "FullImageSize",
    "RawImageSize",
    "ImageSize",
    "ExifImageWidth",
    "ExifImageHeight",
    "Orientation",
    "WB_GRBLevels",
    "Make",
    "Model",
    "FujiLayout",
}


def ai_noise_enabled(param: dict[str, Any] | None) -> bool:
    if not isinstance(param, dict):
        return False
    return bool(param.get("switch_ai_noise_reduction", True)) and bool(param.get("ai_noise_reduction", False))


def _file_fingerprint(file_path: str | None) -> tuple[Any, ...]:
    if not file_path:
        return ("file", None)
    try:
        st = os.stat(file_path)
        size = int(st.st_size)
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            if size <= 192 * 1024:
                digest.update(f.read())
            else:
                for offset in (0, max(0, size // 2 - 32 * 1024), max(0, size - 64 * 1024)):
                    f.seek(offset)
                    digest.update(f.read(64 * 1024))
        return ("file", size, digest.hexdigest())
    except OSError:
        return ("file", None)


def _image_sample_digest(image: np.ndarray | None) -> str:
    if image is None:
        return ""
    arr = np.asarray(image)
    if arr.size == 0:
        return "empty"
    try:
        h, w = arr.shape[:2]
        ys = np.linspace(0, max(0, h - 1), num=min(32, max(1, h)), dtype=np.int64)
        xs = np.linspace(0, max(0, w - 1), num=min(32, max(1, w)), dtype=np.int64)
        sample = arr[np.ix_(ys, xs)]
        sample = np.ascontiguousarray(np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0), dtype=np.float32)
        return hashlib.sha256(sample.tobytes()).hexdigest()
    except Exception:
        return hashlib.sha256(repr((arr.shape, str(arr.dtype))).encode("utf-8")).hexdigest()


def ai_noise_source_signature(file_path: str | None, image: np.ndarray | None, param: dict[str, Any] | None = None) -> str:
    shape = tuple(getattr(image, "shape", ()) or ())
    dtype = str(getattr(image, "dtype", ""))
    sample_digest = _image_sample_digest(image)
    payload = repr(("scunet_source_v5", _file_fingerprint(file_path), shape, dtype, sample_digest)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ai_noise_valid_source_signatures(file_path: str | None, image: np.ndarray | None, param: dict[str, Any] | None = None) -> set[str]:
    return {ai_noise_source_signature(file_path, image, param)}


def ai_noise_content_key(
    file_path: str | None,
    image: np.ndarray | None,
    param: dict[str, Any] | None,
    *,
    source_signature: str | None = None,
) -> str:
    nr = bool((param or {}).get("ai_noise_reduction", False))
    source_signature = source_signature or ai_noise_source_signature(file_path, image, param)
    payload = repr(("scunet_v3", nr, source_signature)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ai_noise_valid_content_keys(file_path: str | None, image: np.ndarray | None, param: dict[str, Any] | None = None) -> set[str]:
    return {
        ai_noise_content_key(file_path, image, param, source_signature=sig)
        for sig in ai_noise_valid_source_signatures(file_path, image, param)
    }


def ai_noise_source_debug_info(file_path: str | None, image: np.ndarray | None) -> dict[str, Any]:
    return {
        "file_fingerprint": _file_fingerprint(file_path),
        "shape": tuple(getattr(image, "shape", ()) or ()),
        "dtype": str(getattr(image, "dtype", "")),
        "sample_digest": _image_sample_digest(image),
    }


def _clone_ai_noise_exif(exif_data: Any) -> dict[str, Any]:
    if not isinstance(exif_data, dict):
        return {}
    cloned: dict[str, Any] = {}
    for key, value in exif_data.items():
        if key in _AI_NOISE_EXIF_KEYS or (
            isinstance(key, str) and key.rsplit(":", 1)[-1] in _AI_NOISE_EXIF_KEYS
        ):
            cloned[key] = copy.deepcopy(value)
    return cloned


def clone_ai_noise_param_snapshot(param: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "switch_ai_noise_reduction": bool(param.get("switch_ai_noise_reduction", True)),
        "ai_noise_reduction": bool(param.get("ai_noise_reduction", False)),
        "ai_noise_reduction_intensity": float(param.get("ai_noise_reduction_intensity", 70.0)),
        "image_fidelity": param.get("image_fidelity"),
    }
    exif_data = _clone_ai_noise_exif(param.get("exif_data"))
    if exif_data:
        snapshot["exif_data"] = exif_data
    return snapshot


def current_param_accepts_ai_noise_result(
    param: dict[str, Any] | None,
    *,
    file_path: str | None,
    image: np.ndarray | None,
    content_key: str,
    source_signature: str,
) -> bool:
    if not ai_noise_enabled(param):
        return False
    valid_sources = ai_noise_valid_source_signatures(file_path, image, param)
    if source_signature not in valid_sources:
        return False
    return content_key in ai_noise_valid_content_keys(file_path, image, param)


def merge_ai_noise_result_into_param(
    param: dict[str, Any],
    raw_result: np.ndarray,
    content_key: str,
    source_signature: str | None = None,
) -> None:
    param["ai_noise_reduction_result"] = np.ascontiguousarray(raw_result, dtype=np.float32)
    param["ai_noise_reduction_content_key"] = content_key
    if source_signature is not None:
        param["ai_noise_reduction_source_signature"] = source_signature


def _read_pmck_with_token(image_path: str) -> tuple[dict[str, Any], tuple[int, int, int] | None]:
    data, token = pmck_store.read_image_with_token(image_path, default_empty=True)
    return pmck_store.ensure_primary_param(data), token


def _pmck_param_accepts_result(pmck_primary: dict[str, Any]) -> bool:
    if not ai_noise_enabled(pmck_primary):
        return False
    return True


def merge_ai_noise_result_into_pmck(
    image_path: str,
    raw_result: np.ndarray,
    *,
    content_key: str,
    source_signature: str,
) -> bool:
    data, token = _read_pmck_with_token(image_path)
    primary = data.setdefault("primary_param", {})
    raw = np.ascontiguousarray(raw_result, dtype=np.float32)
    if not _pmck_param_accepts_result(primary):
        return False

    primary = copy.deepcopy(primary)
    primary["ai_noise_reduction_result"] = utils.convert_image_to_list(raw)
    primary["ai_noise_reduction_content_key"] = content_key
    primary["ai_noise_reduction_source_signature"] = source_signature
    primary["heavy_saved_at_fidelity"] = ImageFidelity.FULL.value
    data["primary_param"] = params._msgpack_safe_value(primary)
    return pmck_store.write_image(image_path, data, expected_token=token)
