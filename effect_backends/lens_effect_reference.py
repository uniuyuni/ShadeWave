"""Reference implementations for Lens Simulator sub-effects."""

from __future__ import annotations

import cv2
import math
import numpy as np


SUNSTAR_RENDER_MAXDIM = 512


def apply_bokeh_color_fringe(
    image: np.ndarray,
    depth_map: np.ndarray | None,
    focus_depth: float,
    strength: float,
    resolution_scale: float,
) -> np.ndarray:
    """ボケの色縁（spherochromatism / LoCA）。前ボケ=マゼンタ / 後ボケ=グリーン。

    平面的に色を足すのではなく、波長ごとのピント差をチャンネルの差分ボケとして再現する。
    前ボケ領域では R/B を、後ボケ領域では G を相対的にぼかし、高コントラスト境界や
    ハイライトの縁に色をにじませる。前後判定に depth が要るため depth がある時のみ動く。
    """
    if depth_map is None or strength <= 0.0:
        return image

    img = np.asarray(image, dtype=np.float32)
    rs = max(1.0, float(resolution_scale))
    signed = (np.asarray(depth_map, dtype=np.float32) - np.float32(focus_depth))
    defocus = cv2.GaussianBlur(np.abs(signed), (0, 0), max(0.5, 2.0 * rs))
    s = float(strength) / 100.0

    amount = np.clip(defocus * np.float32(0.6 + 1.4 * s), 0.0, 1.0).astype(np.float32)
    front_w = np.where(signed < 0.0, amount, np.float32(0.0)).astype(np.float32)
    back_w = np.where(signed > 0.0, amount, np.float32(0.0)).astype(np.float32)

    sigma = (1.0 + 4.0 * s) * rs
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    blur_r = cv2.GaussianBlur(r, (0, 0), sigma)
    blur_g = cv2.GaussianBlur(g, (0, 0), sigma)
    blur_b = cv2.GaussianBlur(b, (0, 0), sigma)

    out = img.copy()
    out[..., 0] = r * (np.float32(1.0) - front_w) + blur_r * front_w
    out[..., 2] = b * (np.float32(1.0) - front_w) + blur_b * front_w
    out[..., 1] = g * (np.float32(1.0) - back_w) + blur_g * back_w
    return out.astype(np.float32, copy=False)


def aperture_mask(shape, radius):
    """絞り形状（多角形/星/ハート/円）の塗りつぶしマスク（softened, 未正規化, 未反転）。"""
    radius = max(2, int(radius))
    size = 2 * radius + 1
    canvas = np.zeros((size, size), dtype=np.float32)
    c = float(radius)
    name = str(shape or "hexagon").strip().lower()

    if name == "circle":
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        canvas = (np.sqrt((xx * xx + yy * yy).astype(np.float32)) <= radius).astype(np.float32)
    else:
        pts = []
        if name == "star":
            n = 5
            for i in range(2 * n):
                rr = radius if (i % 2 == 0) else radius * 0.45
                a = -math.pi / 2 + i * math.pi / n
                pts.append((c + rr * math.cos(a), c + rr * math.sin(a)))
        elif name == "heart":
            for i in range(72):
                t = 2.0 * math.pi * i / 72.0
                hx = 16.0 * (math.sin(t) ** 3)
                hy = 13.0 * math.cos(t) - 5.0 * math.cos(2 * t) - 2.0 * math.cos(3 * t) - math.cos(4 * t)
                pts.append((c + hx / 17.0 * radius, c - hy / 17.0 * radius))
        else:
            n = 5 if name == "pentagon" else 6
            for i in range(n):
                a = -math.pi / 2 + 2.0 * math.pi * i / n
                pts.append((c + radius * math.cos(a), c + radius * math.sin(a)))
        poly = np.array([pts], dtype=np.int32)
        cv2.fillPoly(canvas, poly, 1.0)

    return cv2.GaussianBlur(canvas, (0, 0), max(0.3, radius * 0.022))


def rainbow_rgb(phase, sat=1.8):
    """位相(rad)を滑らかなコサイン虹の RGB(...,3) へ変換する。"""
    phase = np.asarray(phase, dtype=np.float32)
    r = 0.5 + 0.5 * np.cos(phase)
    g = 0.5 + 0.5 * np.cos(phase - np.float32(2.0 * math.pi / 3.0))
    b = 0.5 + 0.5 * np.cos(phase - np.float32(4.0 * math.pi / 3.0))
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    m = rgb.mean(axis=-1, keepdims=True)
    return np.clip(m + (rgb - m) * np.float32(sat), 0.0, 1.0).astype(np.float32)


def angle_warp(theta):
    """角度θに決定論的な乱数ワープを加え、虹の並びをちらつかず不規則にする。"""
    rng = np.random.default_rng(20240517)
    warp = np.zeros_like(theta, dtype=np.float32)
    for _ in range(7):
        n = int(rng.integers(2, 9))
        amp = float(rng.uniform(0.3, 1.1))
        ph = float(rng.uniform(0.0, 2.0 * math.pi))
        warp += np.float32(amp) * np.sin(n * theta + ph).astype(np.float32)
    return warp


def aperture_kernel(shape, radius):
    """絞り形状の正規化カーネル（sum=1, 単色）。filter2D は相関なので 180° 反転して返す。"""
    canvas = aperture_mask(shape, radius)
    grad = cv2.morphologyEx(canvas, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    canvas = canvas + np.float32(0.35) * grad
    total = float(canvas.sum())
    if total < 1e-6:
        r = canvas.shape[0] // 2
        canvas[r, r] = 1.0
        total = 1.0
    canvas /= total
    return cv2.flip(canvas, -1).astype(np.float32)


def aperture_kernel_colored(shape, radius, amount):
    """シャボン玉の玉虫色の縁を持つ3chカーネル（H,W,3, 各ch sum=1）。

    色はボケの外周に閉じ込め、色相は縁を一周する角度θ方向に変える。
    amount を上げると彩度・明度・リム幅が増え、リムの帯が内側へ太る。
    """
    radius = max(2, int(radius))
    a = float(np.clip(amount, 0.0, 1.0))
    mask = aperture_mask(shape, radius)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.sqrt((xx * xx + yy * yy).astype(np.float32))
    theta = np.arctan2(yy.astype(np.float32), xx.astype(np.float32))
    rn = np.clip(dist / float(radius), 0.0, 1.0)

    phase = theta * np.float32(3.0) + angle_warp(theta)
    sat = 1.0 + 4.0 * a
    rim_rgb = rainbow_rgb(phase, sat)

    rim_w = np.float32(0.18 + 0.42 * a)
    rim = np.clip((rn - (np.float32(1.0) - rim_w)) / rim_w, 0.0, 1.0).astype(np.float32)
    cover = np.clip(rim * np.float32(0.6 + 0.4 * a), 0.0, 1.0)[..., np.newaxis]
    gain = np.float32(1.0 + 3.0 * a)
    white = np.repeat(mask[..., np.newaxis], 3, axis=2).astype(np.float32)
    colored = rim_rgb * mask[..., np.newaxis] * gain
    k = white * (np.float32(1.0) - cover) + colored * cover
    k = np.clip(k, 0.0, None)
    for c in range(3):
        s = float(k[..., c].sum())
        if s < 1e-6:
            k[radius, radius, c] = 1.0
            s = 1.0
        k[..., c] /= s
    return cv2.flip(k, -1).astype(np.float32)


def apply_shaped_bokeh(image, depth_map, focus_depth, strength, radius, shape, rim=0.0):
    """形状ボケ（被写界ボケ）。

    画像を絞り形状カーネルで畳み込み、非合焦領域にブレンドする。点を貼るのではなく
    実レンズと同じ積分なので、ハイライトが自然に星/ハート等の形になり、平坦部は滑らかに
    ボケる。depth があれば背景に限定し、無ければ HDR ハイライト領域だけにブレンドして
    被写体のシャープさを保つ。radius は呼び出し側で表示倍率に追従させたテクスチャ px。
    """
    if strength <= 0.0 or radius < 2:
        return image

    img = np.asarray(image, dtype=np.float32)
    s = float(strength) / 100.0
    lum = np.mean(img, axis=2, dtype=np.float32)

    rim_n = float(np.clip(rim / 100.0, 0.0, 1.0))
    if rim_n <= 1e-4:
        kernel = aperture_kernel(shape, radius)
        kc = None
    else:
        kernel = None
        kc = aperture_kernel_colored(shape, radius, rim_n)

    if depth_map is None:
        # 均一な白面や肌ハイライトを極力巻き込まず、局所背景から浮いたピークだけを source にする。
        local_sigma = max(2.0, float(radius) * 0.35)
        local_base = cv2.GaussianBlur(lum, (0, 0), local_sigma, borderType=cv2.BORDER_REFLECT)
        peak_floor = np.maximum(np.float32(0.8), local_base + np.float32(0.2))
        peak = np.clip(lum - peak_floor, 0.0, None).astype(np.float32)
        if float(np.max(peak)) <= 1e-6:
            return image

        peak_ratio = peak / np.maximum(lum, np.float32(1e-6))
        energy_boost = np.float32(1.0) + np.log1p(peak) * np.float32(1.0 + 2.0 * s)
        highlight = img * peak_ratio[..., np.newaxis] * energy_boost[..., np.newaxis]
        if kc is None:
            halo = cv2.filter2D(highlight, -1, kernel, borderType=cv2.BORDER_REFLECT)
        else:
            halo = np.empty_like(highlight)
            halo[..., 0] = cv2.filter2D(highlight[..., 0], -1, kc[..., 0], borderType=cv2.BORDER_REFLECT)
            halo[..., 1] = cv2.filter2D(highlight[..., 1], -1, kc[..., 1], borderType=cv2.BORDER_REFLECT)
            halo[..., 2] = cv2.filter2D(highlight[..., 2], -1, kc[..., 2], borderType=cv2.BORDER_REFLECT)
        gain = np.float32(0.45 + 1.25 * s)
        return (img + halo * gain).astype(np.float32, copy=False)

    # ハイライトの寄与を少し強め、畳み込み内で形状を見えやすくする。
    hl_excess = np.clip(lum - np.float32(0.8), 0.0, None)
    src = img * (np.float32(1.0) + (hl_excess * np.float32(2.0 + 6.0 * s))[..., np.newaxis])
    if kc is None:
        blurred = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REFLECT)
    else:
        blurred = np.empty_like(src)
        blurred[..., 0] = cv2.filter2D(src[..., 0], -1, kc[..., 0], borderType=cv2.BORDER_REFLECT)
        blurred[..., 1] = cv2.filter2D(src[..., 1], -1, kc[..., 1], borderType=cv2.BORDER_REFLECT)
        blurred[..., 2] = cv2.filter2D(src[..., 2], -1, kc[..., 2], borderType=cv2.BORDER_REFLECT)

    dm = np.asarray(depth_map, dtype=np.float32)
    w = np.clip(np.abs(dm - np.float32(focus_depth)) * np.float32(2.5), 0.0, 1.0)
    w = (w * np.float32(np.clip(0.4 + 0.6 * s, 0.0, 1.0))).astype(np.float32)[..., np.newaxis]
    return img * (np.float32(1.0) - w) + blurred * w


def optical_geometry(img_shape, disp_info=None, original_img_size=None, crop_size_offset=None):
    """渦の光学中心と、元座標基準の半径マップを返す。

    拡大/クロップしても渦の中心が元画像中心に固定されるよう、disp_info と original_img_size
    から元中心を逆算する。失敗時は従来どおり画像中心へフォールバックする。
    """
    th, tw = int(img_shape[0]), int(img_shape[1])
    try:
        if disp_info is None or original_img_size is None or crop_size_offset is None:
            raise ValueError("missing geometry")
        ow, oh = original_img_size
        dx, dy, dw, dh, _scale = disp_info
        if dw <= 0 or dh <= 0:
            raise ValueError("invalid disp_info")
        new_w, new_h, off_x, off_y = crop_size_offset
        sx = new_w / float(dw)
        sy = new_h / float(dh)
        maxsize = float(max(ow, oh))
        ocx = ocy = maxsize * 0.5
        half_diag = float(np.sqrt((ow * 0.5) ** 2 + (oh * 0.5) ** 2)) or 1.0
        cx = off_x + (ocx - dx) * sx
        cy = off_y + (ocy - dy) * sy
        yy, xx = np.mgrid[0:th, 0:tw]
        ox = dx + (xx.astype(np.float32) - off_x) / sx
        oy = dy + (yy.astype(np.float32) - off_y) / sy
        radial = (np.sqrt((ox - ocx) ** 2 + (oy - ocy) ** 2) / half_diag).astype(np.float32)
        return float(cx), float(cy), np.clip(radial, 0.0, 1.0)
    except Exception:
        cx, cy = (tw - 1) * 0.5, (th - 1) * 0.5
        yy, xx = np.mgrid[0:th, 0:tw]
        r = np.sqrt(((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32))
        rmax = float(np.sqrt(cx * cx + cy * cy)) or 1.0
        return cx, cy, np.clip(r / rmax, 0.0, 1.0).astype(np.float32)


def apply_swirl_bokeh(image, depth_map, focus_depth, strength, resolution_scale, center_xy, radial_norm):
    """非合焦領域へ接線方向の回転ブラーを掛けて渦ボケを作る。

    center_xy と radial_norm は元画像基準。渦化する範囲は depth があれば非合焦領域、
    無ければ半径のみで近似する。
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    cx, cy = center_xy

    # 中心が完全に静止したディスクにならないよう、半径ウェイトに下駄を履かせる。
    center_floor = np.float32(0.35)
    radial = center_floor + (np.float32(1.0) - center_floor) * np.clip(radial_norm, 0.0, 1.0).astype(np.float32)

    if depth_map is not None:
        defocus = np.clip(np.abs(np.asarray(depth_map, dtype=np.float32) - np.float32(focus_depth)) * np.float32(2.5), 0.0, 1.0)
    else:
        defocus = np.float32(1.0)
    wmap = np.clip(radial * defocus, 0.0, 1.0).astype(np.float32)

    max_angle = 12.0 * (strength / 100.0)
    if max_angle < 1e-3:
        return image

    # 極座標で角度軸方向にブラーし、離散回転コピー由来のゴーストを避ける。
    corners = ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h))
    r_max = float(max(np.hypot(px - cx, py - cy) for px, py in corners))
    if r_max < 1.0:
        return image
    radius_bins = int(min(2048, max(64, round(r_max))))
    angle_bins = int(min(2880, max(256, round(2.0 * np.pi * r_max))))
    flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR
    polar = np.zeros((angle_bins, radius_bins, img.shape[2]), dtype=img.dtype)
    polar = cv2.warpPolar(img, (radius_bins, angle_bins), (cx, cy), r_max, flags, polar)

    k = int(round((2.0 * np.radians(max_angle) / (2.0 * np.pi)) * angle_bins))
    k = max(1, k | 1)
    if k > 1:
        pad = k // 2
        polar = np.concatenate([polar[-pad:], polar, polar[:pad]], axis=0)
        polar = cv2.GaussianBlur(polar, (1, k), 0)
        polar = polar[pad:pad + angle_bins]

    swirl = np.zeros_like(img)
    swirl = cv2.warpPolar(polar, (w, h), (cx, cy), r_max, flags + cv2.WARP_INVERSE_MAP, swirl)

    sigma = (strength / 100.0) * 2.0 * max(1.0, resolution_scale)
    if sigma > 0.3:
        kk = int(sigma * 3) | 1
        swirl = cv2.GaussianBlur(swirl, (kk, kk), sigma)

    wm = wmap[..., np.newaxis]
    return img * (1.0 - wm) + swirl * wm


def spike_count_from_blades(blades):
    """絞り羽根枚数 → サンスターの本数。偶数=N本 / 奇数=2N本（回折の物理整合）。"""
    try:
        b = int(str(blades).strip())
    except (TypeError, ValueError):
        b = 9
    b = max(3, b)
    return b if (b % 2 == 0) else 2 * b


def apply_sunstar(image, strength, length, threshold, blades, aperture, mag, orig_size, render_maxdim=SUNSTAR_RENDER_MAXDIM):
    """点光源に回折スパイク（光条/サンスター）を描く。

    クリップ気味のハイライト塊だけを検出し、絞り羽根枚数から本数を物理整合で決める。
    各スパイクは中心から先端まで太さ一定の細い光条で、長さ・太さ・濃さには決定論的な
    個体差を入れる。linear/HDR 空間で加算合成し、clip は下流へ委ねる。
    """
    img = np.asarray(image, dtype=np.float32)
    s = float(strength) / 100.0
    if s <= 0.0:
        return image
    h, w = img.shape[:2]
    mag = max(1e-3, float(mag))
    try:
        ow, oh = float(orig_size[0]), float(orig_size[1])
    except Exception:
        ow, oh = float(w), float(h)
    scene_min = max(1.0, min(ow, oh))

    f_open, f_max = 1.4, 16.0
    ap_raw = float(np.clip((float(aperture) - f_open) / (f_max - f_open), 0.0, 1.0))

    # 検出は全解像度で行う。縮小してから検出するとズーム時に光源が平均化され閾値割れしやすい。
    lum = np.max(img, axis=2)
    thr = 0.55 + 0.44 * (float(threshold) / 100.0)
    mask = (lum > thr).astype(np.uint8)
    if int(mask.sum()) == 0:
        return image

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return image

    sources = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        ys = slice(int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]))
        xs = slice(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]))
        peak = float(lum[ys, xs].max())
        blob = labels[ys, xs] == i
        # 光源色を正規化して tint にし、光条のベース色にする。
        col = img[ys, xs][blob].reshape(-1, 3).mean(axis=0)
        cmax = max(float(col.max()), 1e-4)
        src_tint = np.clip(col / cmax, 0.25, 1.0).astype(np.float32)
        sources.append((peak, area, cx, cy, src_tint))
    sources.sort(key=lambda t: t[0], reverse=True)
    sources = sources[:16]

    # 描画は縮小キャンバスで行い最後に拡大合成する。光条は滑らかなので劣化は小さい。
    scl = min(1.0, float(render_maxdim) / float(max(h, w)))
    W = max(1, int(round(w * scl)))
    H = max(1, int(round(h * scl)))

    M = spike_count_from_blades(blades)
    spacing = 2.0 * math.pi / M
    # 開放でも十分伸ばし、絞り込むほど更に長く細くする。
    ap_len = 0.4 + 0.6 * ap_raw
    base_len = (0.02 + 1.6 * (float(length) / 100.0)) * scene_min * ap_len * mag * scl
    width_ap = 1.2 - 0.6 * ap_raw

    overlay = np.zeros((H, W, 3), dtype=np.float32)
    cool_bias = np.array([0.92, 0.97, 1.10], dtype=np.float32)
    base_rot = float(np.random.default_rng(0x5A17).uniform(0.0, math.pi))

    for idx, (peak, area, cx, cy, src_tint) in enumerate(sources):
        inten = float(np.clip((peak - thr) / max(1e-3, 4.0 - thr), 0.05, 1.0)) ** 0.5
        radius_src = max(0.6, math.sqrt(area / math.pi) * scl)
        L = float(np.clip(base_len * (0.6 + 0.6 * inten), 3.0, 1.4 * max(H, W)))
        spike_w0 = max(0.6, (0.003 * L + 0.38 * radius_src) * width_ap)
        core_sigma = max(1.0, radius_src * 1.2)
        Lpx = int(math.ceil(L * 1.3))

        cxs = cx * scl
        cys = cy * scl
        x0 = max(0, int(math.floor(cxs - Lpx)))
        x1 = min(W, int(math.ceil(cxs + Lpx + 1)))
        y0 = max(0, int(math.floor(cys - Lpx)))
        y1 = min(H, int(math.ceil(cys + Lpx + 1)))
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx.astype(np.float32) - cxs
        dy = yy.astype(np.float32) - cys
        r = np.sqrt(dx * dx + dy * dy) + np.float32(1e-3)
        theta = np.arctan2(dy, dx)

        # 羽根の個体差（決定論シード）：長さ・太さ・濃さ・微小角度をスパイクごとに振る。
        rng = np.random.default_rng(0x9E37 + idx)
        ang_jit = rng.uniform(-0.04, 0.04, size=M).astype(np.float32)
        len_jit = rng.uniform(0.45, 1.25, size=M).astype(np.float32)
        wid_jit = rng.uniform(0.6, 1.6, size=M).astype(np.float32)
        amp_jit = rng.uniform(0.4, 1.0, size=M).astype(np.float32)

        # 各画素を最も近いスパイク1本だけに割り当て、M回ループを1パスに畳む。
        k = np.mod(np.round((theta - base_rot) / spacing).astype(np.int32), M)
        a_k = base_rot + k.astype(np.float32) * spacing + ang_jit[k]
        d_ang = theta - a_k
        d_ang = np.arctan2(np.sin(d_ang), np.cos(d_ang))
        perp = r * np.sin(d_ang)
        along = r * np.cos(d_ang)
        Lm = np.maximum(1.5, L * len_jit[k])
        wm = np.maximum(0.5, spike_w0 * wid_jit[k])
        cross = np.exp(-(perp * perp) / (2.0 * wm * wm))
        t = np.clip(along / Lm, 0.0, 1.0)
        prof = (1.0 - t) * (0.30 + 0.70 * np.exp(-along / (0.45 * Lm)))
        ray = np.where((along > 0.0) & (along < Lm), cross * prof, np.float32(0.0)) * amp_jit[k]

        core = np.exp(-(r * r) / (2.0 * core_sigma * core_sigma))
        scalar = (ray + 0.9 * core) * inten

        # 光条の色は光源色。先端へ向けて僅かに青く分散させる。
        tip_tint = np.clip(src_tint * cool_bias, 0.0, 1.2).astype(np.float32)
        rn = np.clip(r / max(L, 1.0), 0.0, 1.0)[..., np.newaxis]
        tint = src_tint * (1.0 - rn) + tip_tint * rn
        overlay[y0:y1, x0:x1] += scalar[..., np.newaxis] * tint

    if scl < 1.0:
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

    gain = np.float32(s * (0.7 + 0.3 * ap_raw))
    return (img + overlay * gain).astype(np.float32, copy=False)


__all__ = [
    "SUNSTAR_RENDER_MAXDIM",
    "angle_warp",
    "aperture_kernel",
    "aperture_kernel_colored",
    "aperture_mask",
    "apply_bokeh_color_fringe",
    "apply_shaped_bokeh",
    "apply_sunstar",
    "apply_swirl_bokeh",
    "optical_geometry",
    "rainbow_rgb",
    "spike_count_from_blades",
]
