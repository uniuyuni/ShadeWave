"""Python-facing Lens Simulator sub-effect backend adapter."""

from __future__ import annotations

import math
import numpy as np
import cv2

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import lens_effect_reference

try:
    import scipy.fft as _scipy_fft
except ImportError:
    _scipy_fft = None


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_lens_effect_metal")

_SELECTOR = BackendSelector(
    "lens_effect",
    globals(),
    env="PLATYPUS_LENS_EFFECT_BACKEND",
    metal_strict_env="PLATYPUS_LENS_EFFECT_METAL_STRICT",
    metal_name="effect_backends._lens_effect_metal",
    reference_name="effect_backends.lens_effect_reference",
    metal_disabled_values={"reference", "python", "opencv", "off", "0", "false", "no"},
    metal_forced_values={"metal"},
)

# 形状ボケの畳み込みコスト: Metal 直畳み込みは O(カーネル面積)/px、FFT は解像度のみ
# 依存。この半径以上では FFT が直畳み込みを上回る(6MP 実測で決定)。
_FFT_MIN_RADIUS = 14


def _fft_shaped_convolve(source: np.ndarray, kernel: np.ndarray, colored_kernel: bool) -> np.ndarray | None:
    """shaped_* Metal カーネルの畳み込み項を FFT で等価計算する。

    Metal 側の reflect_edge(端画素を含む対称反射)は np.pad の 'symmetric' と一致。
    パディング幅が画像より大きい場合は None(呼び出し元で直畳み込みへ)。
    """
    if _scipy_fft is None:
        return None
    kh, kw = int(kernel.shape[0]), int(kernel.shape[1])
    kry, krx = kh // 2, kw // 2
    h, w = source.shape[:2]
    pad_t, pad_b = kry, kh - 1 - kry
    pad_l, pad_r = krx, kw - 1 - krx
    if max(pad_t, pad_b) >= h or max(pad_l, pad_r) >= w:
        return None

    padded = np.pad(source, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="symmetric")
    fh = padded.shape[0] + kh - 1
    fw = padded.shape[1] + kw - 1
    fh2 = _scipy_fft.next_fast_len(fh)
    fw2 = _scipy_fft.next_fast_len(fw)

    acc = np.empty_like(source)
    kernel_flipped = kernel[::-1, ::-1]
    kernel_freq = None
    for c in range(3):
        if colored_kernel:
            kernel_freq = _scipy_fft.rfft2(
                np.ascontiguousarray(kernel_flipped[:, :, c], dtype=np.float32),
                s=(fh2, fw2), workers=-1)
        elif kernel_freq is None:
            kernel_freq = _scipy_fft.rfft2(
                np.ascontiguousarray(kernel_flipped, dtype=np.float32),
                s=(fh2, fw2), workers=-1)
        source_freq = _scipy_fft.rfft2(padded[:, :, c], s=(fh2, fw2), workers=-1)
        full = _scipy_fft.irfft2(source_freq * kernel_freq, s=(fh2, fw2), workers=-1)
        acc[:, :, c] = full[kh - 1:kh - 1 + h, kw - 1:kw - 1 + w]
    return acc


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_strict() -> bool:
    return _SELECTOR.metal_strict()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


def apply_bokeh_color_fringe(
    image: np.ndarray,
    depth_map: np.ndarray | None,
    focus_depth: float,
    strength: float,
    resolution_scale: float,
) -> np.ndarray:
    if depth_map is None or strength <= 0.0:
        return image

    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            return _metal_backend.apply_bokeh_color_fringe(
                np.ascontiguousarray(image, dtype=np.float32),
                np.ascontiguousarray(depth_map, dtype=np.float32),
                float(focus_depth),
                float(strength),
                float(resolution_scale),
            )
        except Exception:
            if _metal_strict():
                raise

    return lens_effect_reference.apply_bokeh_color_fringe(
        image,
        depth_map,
        focus_depth,
        strength,
        resolution_scale,
    )


def aperture_mask(shape, radius):
    return lens_effect_reference.aperture_mask(shape, radius)


def rainbow_rgb(phase, sat=1.8):
    return lens_effect_reference.rainbow_rgb(phase, sat=sat)


def angle_warp(theta):
    return lens_effect_reference.angle_warp(theta)


def aperture_kernel(shape, radius):
    return lens_effect_reference.aperture_kernel(shape, radius)


def aperture_kernel_colored(shape, radius, amount):
    return lens_effect_reference.aperture_kernel_colored(shape, radius, amount)


def apply_shaped_bokeh(image, depth_map, focus_depth, strength, radius, shape, rim=0.0):
    if strength <= 0.0 or radius < 2:
        return image

    # scipy FFT 経路は reference(cv2.filter2D + BORDER_REFLECT)と数学的に同一の
    # 相関計算で、解像度のみ依存(半径非依存)。metal 指定時は小半径だけ直畳み込み、
    # それ以外(デフォルト含む)は FFT を優先する。
    preference = _backend_preference()
    use_metal = _metal_backend is not None and preference == "metal" and _metal_device_available()
    use_fft = _scipy_fft is not None and preference not in {"reference", "python", "opencv", "off", "0", "false", "no"}

    if use_metal or use_fft:
        try:
            img = np.ascontiguousarray(image, dtype=np.float32)
            s = float(strength) / 100.0
            lum = np.mean(img, axis=2, dtype=np.float32)
            rim_n = float(np.clip(rim / 100.0, 0.0, 1.0))
            colored_kernel = rim_n > 1e-4
            if colored_kernel:
                kernel = np.ascontiguousarray(lens_effect_reference.aperture_kernel_colored(shape, radius, rim_n), dtype=np.float32)
            else:
                kernel = np.ascontiguousarray(lens_effect_reference.aperture_kernel(shape, radius), dtype=np.float32)
            kernel_radius = max(kernel.shape[0], kernel.shape[1]) // 2
            prefer_fft = use_fft and (not use_metal or kernel_radius >= _FFT_MIN_RADIUS)

            if depth_map is None:
                local_sigma = max(2.0, float(radius) * 0.35)
                local_base = cv2.GaussianBlur(lum, (0, 0), local_sigma, borderType=cv2.BORDER_REFLECT)
                peak_floor = np.maximum(np.float32(0.8), local_base + np.float32(0.2))
                peak = np.clip(lum - peak_floor, 0.0, None).astype(np.float32)
                if float(np.max(peak)) <= 1e-6:
                    return image
                peak_ratio = peak / np.maximum(lum, np.float32(1e-6))
                energy_boost = np.float32(1.0) + np.log1p(peak) * np.float32(1.0 + 2.0 * s)
                source = np.ascontiguousarray(
                    img * peak_ratio[..., np.newaxis] * energy_boost[..., np.newaxis], dtype=np.float32)
                gain = np.float32(0.45 + 1.25 * s)
                if prefer_fft:
                    acc = _fft_shaped_convolve(source, kernel, colored_kernel)
                    if acc is not None:
                        return img + acc * gain
                if use_metal:
                    return _metal_backend.apply_shaped_bokeh_no_depth(
                        img,
                        source,
                        kernel,
                        bool(colored_kernel),
                        float(gain),
                    )
            else:
                hl_excess = np.clip(lum - np.float32(0.8), 0.0, None)
                source = np.ascontiguousarray(
                    img * (np.float32(1.0) + (hl_excess * np.float32(2.0 + 6.0 * s))[..., np.newaxis]),
                    dtype=np.float32)
                if prefer_fft:
                    acc = _fft_shaped_convolve(source, kernel, colored_kernel)
                    if acc is not None:
                        w = np.clip(
                            np.abs(np.asarray(depth_map, dtype=np.float32) - np.float32(focus_depth)) * np.float32(2.5),
                            0.0, 1.0)
                        w = (w * np.float32(np.clip(0.4 + 0.6 * s, 0.0, 1.0)))[..., np.newaxis]
                        return img * (np.float32(1.0) - w) + acc * w
                if use_metal:
                    return _metal_backend.apply_shaped_bokeh_depth(
                        img,
                        source,
                        np.ascontiguousarray(depth_map, dtype=np.float32),
                        kernel,
                        bool(colored_kernel),
                        float(focus_depth),
                        float(strength),
                    )
        except Exception:
            if _metal_strict():
                raise

    return lens_effect_reference.apply_shaped_bokeh(
        image,
        depth_map,
        focus_depth,
        strength,
        radius,
        shape,
        rim=rim,
    )


def optical_geometry(img_shape, disp_info=None, original_img_size=None, crop_size_offset=None):
    return lens_effect_reference.optical_geometry(
        img_shape,
        disp_info=disp_info,
        original_img_size=original_img_size,
        crop_size_offset=crop_size_offset,
    )


def apply_swirl_bokeh(image, depth_map, focus_depth, strength, resolution_scale, center_xy, radial_norm):
    if strength <= 0.0:
        return image

    if _metal_backend is not None and _backend_preference() == "metal" and _metal_device_available():
        try:
            img = np.ascontiguousarray(image, dtype=np.float32)
            if depth_map is None:
                depth = np.zeros(img.shape[:2], dtype=np.float32)
                use_depth = False
            else:
                depth = np.ascontiguousarray(depth_map, dtype=np.float32)
                use_depth = True
            return _metal_backend.apply_swirl_bokeh_direct(
                img,
                depth,
                np.ascontiguousarray(radial_norm, dtype=np.float32),
                bool(use_depth),
                float(center_xy[0]),
                float(center_xy[1]),
                float(focus_depth),
                float(strength),
            )
        except Exception:
            if _metal_strict():
                raise

    return lens_effect_reference.apply_swirl_bokeh(
        image,
        depth_map,
        focus_depth,
        strength,
        resolution_scale,
        center_xy,
        radial_norm,
    )


def spike_count_from_blades(blades):
    return lens_effect_reference.spike_count_from_blades(blades)


def apply_sunstar(image, strength, length, threshold, blades, aperture, mag, orig_size):
    if strength <= 0.0:
        return image

    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            img = np.ascontiguousarray(image, dtype=np.float32)
            h, w = img.shape[:2]
            s = float(strength) / 100.0
            mag = max(1e-3, float(mag))
            try:
                ow, oh = float(orig_size[0]), float(orig_size[1])
            except Exception:
                ow, oh = float(w), float(h)
            scene_min = max(1.0, min(ow, oh))

            f_open, f_max = 1.4, 16.0
            ap_raw = float(np.clip((float(aperture) - f_open) / (f_max - f_open), 0.0, 1.0))
            lum = np.max(img, axis=2)
            thr = 0.55 + 0.44 * (float(threshold) / 100.0)
            mask = (lum > thr).astype(np.uint8)
            if int(mask.sum()) == 0:
                return image

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num <= 1:
                return image

            raw_sources = []
            for i in range(1, num):
                area = int(stats[i, cv2.CC_STAT_AREA])
                cx, cy = float(centroids[i][0]), float(centroids[i][1])
                ys = slice(int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]))
                xs = slice(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]))
                peak = float(lum[ys, xs].max())
                blob = labels[ys, xs] == i
                col = img[ys, xs][blob].reshape(-1, 3).mean(axis=0)
                cmax = max(float(col.max()), 1e-4)
                src_tint = np.clip(col / cmax, 0.25, 1.0).astype(np.float32)
                raw_sources.append((peak, area, cx, cy, src_tint))
            raw_sources.sort(key=lambda t: t[0], reverse=True)
            raw_sources = raw_sources[:16]
            if not raw_sources:
                return image

            scl = min(1.0, float(lens_effect_reference.SUNSTAR_RENDER_MAXDIM) / float(max(h, w)))
            W = max(1, int(round(w * scl)))
            H = max(1, int(round(h * scl)))
            spike_count = lens_effect_reference.spike_count_from_blades(blades)
            spacing = 2.0 * math.pi / spike_count
            ap_len = 0.4 + 0.6 * ap_raw
            base_len = (0.02 + 1.6 * (float(length) / 100.0)) * scene_min * ap_len * mag * scl
            width_ap = 1.2 - 0.6 * ap_raw
            base_rot = float(np.random.default_rng(0x5A17).uniform(0.0, math.pi))

            sources = np.zeros((len(raw_sources), 9), dtype=np.float32)
            jitter = np.zeros((len(raw_sources), spike_count, 4), dtype=np.float32)
            for idx, (peak, area, cx, cy, src_tint) in enumerate(raw_sources):
                inten = float(np.clip((peak - thr) / max(1e-3, 4.0 - thr), 0.05, 1.0)) ** 0.5
                radius_src = max(0.6, math.sqrt(area / math.pi) * scl)
                L = float(np.clip(base_len * (0.6 + 0.6 * inten), 3.0, 1.4 * max(H, W)))
                spike_w0 = max(0.6, (0.003 * L + 0.38 * radius_src) * width_ap)
                core_sigma = max(1.0, radius_src * 1.2)
                sources[idx] = (
                    np.float32(inten),
                    np.float32(cx * scl),
                    np.float32(cy * scl),
                    np.float32(L),
                    np.float32(spike_w0),
                    np.float32(core_sigma),
                    np.float32(src_tint[0]),
                    np.float32(src_tint[1]),
                    np.float32(src_tint[2]),
                )

                rng = np.random.default_rng(0x9E37 + idx)
                jitter[idx, :, 0] = rng.uniform(-0.04, 0.04, size=spike_count).astype(np.float32)
                jitter[idx, :, 1] = rng.uniform(0.45, 1.25, size=spike_count).astype(np.float32)
                jitter[idx, :, 2] = rng.uniform(0.6, 1.6, size=spike_count).astype(np.float32)
                jitter[idx, :, 3] = rng.uniform(0.4, 1.0, size=spike_count).astype(np.float32)

            overlay = _metal_backend.render_sunstar_overlay(
                np.ascontiguousarray(sources, dtype=np.float32),
                np.ascontiguousarray(jitter, dtype=np.float32),
                W,
                H,
                int(len(raw_sources)),
                int(spike_count),
                float(base_rot),
                float(spacing),
            )
            overlay = np.asarray(overlay, dtype=np.float32)
            if scl < 1.0:
                overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

            gain = np.float32(s * (0.7 + 0.3 * ap_raw))
            return (img + overlay * gain).astype(np.float32, copy=False)
        except Exception:
            if _metal_strict():
                raise

    return lens_effect_reference.apply_sunstar(
        image,
        strength,
        length,
        threshold,
        blades,
        aperture,
        mag,
        orig_size,
    )


__all__ = [
    "angle_warp",
    "aperture_kernel",
    "aperture_kernel_colored",
    "aperture_mask",
    "apply_bokeh_color_fringe",
    "apply_shaped_bokeh",
    "apply_sunstar",
    "apply_swirl_bokeh",
    "backend_status",
    "native_available",
    "optical_geometry",
    "rainbow_rgb",
    "spike_count_from_blades",
]
