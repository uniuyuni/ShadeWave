"""Python-facing HLS per-color adjust backend adapter.

cores/core.py の adjust_hls_colors がリファレンス実装であり、このアダプタは
その前処理(width/fade_width の正規化、hue のラップ、カーネルサイズ計算、
cv2.getGaussianKernel によるガウシアン重み生成)を core.py と同一に行った上で
Metal バックエンドへ委譲する。Metal が使えない場合や失敗した場合は
cores.core.adjust_hls_colors にフォールバックする(strict 時は例外を再送出)。
"""

from __future__ import annotations

import numpy as np
import cv2

from .backend_utils import BackendSelector, BackendStatus, optional_backend


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_hls_adjust_metal")

_SELECTOR = BackendSelector(
    "hls_adjust",
    globals(),
    env="PLATYPUS_HLS_ADJUST_BACKEND",
    metal_strict_env="PLATYPUS_HLS_ADJUST_METAL_STRICT",
    metal_name="effect_backends._hls_adjust_metal",
    reference_name="cores.core",
    metal_disabled_values={"reference", "python", "cpu", "off", "0", "false", "no"},
    metal_forced_values={"metal"},
    reference_requested_detail=True,
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def _metal_strict() -> bool:
    return _SELECTOR.metal_strict()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


def _normalize_settings(color_settings, resolution_scale):
    """cores/core.py::adjust_hls_colors のNumba設定変換ループと同一の正規化を行う。

    呼び出し元の dict は変更しない(コピーしてから読む)。
    settings 配列 (N, 12) float32, kernels(連結 float32), offsets(int32),
    radii(int32) を返す。
    """
    n = len(color_settings)
    settings = np.zeros((n, 12), dtype=np.float32)
    kernel_chunks = []
    offsets = np.zeros(n, dtype=np.int32)
    radii = np.zeros(n, dtype=np.int32)

    cursor = 0
    for i, raw in enumerate(color_settings):
        s = dict(raw)

        center = np.float32(s["center"])

        w_val = s["width"]
        if np.isscalar(w_val):
            width = np.array([w_val, w_val], dtype=np.float32)
        else:
            width = np.array(w_val, dtype=np.float32)

        f_val = s["fade_width"]
        if np.isscalar(f_val):
            fade_width = np.array([f_val, f_val], dtype=np.float32)
        else:
            fade_width = np.array(f_val, dtype=np.float32)

        adjust = np.array(s["adjust"], dtype=np.float32).copy()
        if adjust[0] >= 180.0:
            adjust[0] -= 360.0
        elif adjust[0] < -180.0:
            adjust[0] += 360.0

        l_range = np.array(s["l_range"], dtype=np.float32)
        s_range = np.array(s["s_range"], dtype=np.float32)

        settings[i, 0] = center
        settings[i, 1] = width[0]
        settings[i, 2] = width[1]
        settings[i, 3] = fade_width[0]
        settings[i, 4] = fade_width[1]
        settings[i, 5] = l_range[0]
        settings[i, 6] = l_range[1]
        settings[i, 7] = s_range[0]
        settings[i, 8] = s_range[1]
        settings[i, 9] = adjust[0]
        settings[i, 10] = adjust[1]
        settings[i, 11] = adjust[2]

        ksize = max(3, int(int(s["kernel_size"]) * resolution_scale))
        if ksize % 2 == 0:
            ksize += 1

        # cv2 の sigma=0 は bit-exact なダイアディック値を返すため、C++ で
        # 再現せず cv2 から取得することが parity 上重要。
        kernel_1d = cv2.getGaussianKernel(ksize, 0).astype(np.float32).reshape(-1)
        kernel_chunks.append(kernel_1d)
        offsets[i] = cursor
        radii[i] = ksize // 2
        cursor += kernel_1d.shape[0]

    if kernel_chunks:
        kernels = np.concatenate(kernel_chunks).astype(np.float32)
    else:
        kernels = np.zeros((0,), dtype=np.float32)

    return settings, kernels, offsets, radii


def adjust_hls_colors(hls_img, color_settings, resolution_scale=1.0):
    if not color_settings:
        from cores import core as _core

        return _core.adjust_hls_colors(hls_img, color_settings, resolution_scale)

    if (
        isinstance(hls_img, np.ndarray)
        and hls_img.ndim == 3
        and hls_img.shape[-1] >= 3
        and _metal_backend is not None
        and _metal_backend_enabled()
        and _metal_device_available()
    ):
        try:
            settings, kernels, offsets, radii = _normalize_settings(color_settings, resolution_scale)
            image32 = np.ascontiguousarray(hls_img, dtype=np.float32)
            return _metal_backend.apply_hls_adjust(image32, settings, kernels, offsets, radii)
        except Exception:
            if _metal_strict():
                raise

    from cores import core as _core

    return _core.adjust_hls_colors(hls_img, color_settings, resolution_scale)


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "adjust_hls_colors",
]
