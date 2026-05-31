"""
失敗ケース (orig=4896x3264, 中央 CP を 1 個動かす) をそのまま再現して、
各 warp 手法が「CP付近が動く / 端が動かない」を満たすかを数値で比較する。

理想:
  center shift_x ≈ -131 px (= 動かした CP の移動量がそのまま反映される = 補間性)
  4 端 shift  ≈ 0 px        (= 動かしてない端は動かない = 局所性 / 遠方 identity)

実行: python3 docs/mesh-method-experiment.py
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import Rbf, RBFInterpolator

# ---- 失敗ケースの設定 (実機ログと同一) ----
orig_w, orig_h = 4896.0, 3264.0
imax = max(orig_w, orig_h) / 2.0           # 2448
image_dim = max(orig_w, orig_h)            # 4896 (正方化された image)
rows = cols = 4
moved_cp = (2, 2)
off_x_norm, off_y_norm = -0.0268, 0.0      # 中央 CP を左へ -131px


def tcg_to_imgpx(nx, ny):
    return (nx * orig_w + imax, ny * orig_h + imax)


def outer_ring_pins_tcg():
    # warp_correction.outer_ring_pins_tcg() 相当 (±0.65 の 8 点)
    e = 0.65
    return [(-e, -e), (0.0, -e), (e, -e),
            (-e, 0.0),            (e, 0.0),
            (-e, e),  (0.0, e),  (e, e)]


def build_points():
    src, dst = [], []
    for r in range(rows + 1):
        for c in range(cols + 1):
            tx = -0.5 + c / cols
            ty = -0.5 + r / rows
            sx, sy = tcg_to_imgpx(tx, ty)
            ox, oy = (off_x_norm, off_y_norm) if (r, c) == moved_cp else (0.0, 0.0)
            dx, dy = tcg_to_imgpx(tx + ox, ty + oy)
            src.append((sx, sy)); dst.append((dx, dy))
    for tx, ty in outer_ring_pins_tcg():
        px, py = tcg_to_imgpx(tx, ty)
        src.append((px, py)); dst.append((px, py))
    return np.array(src, float), np.array(dst, float)


# 評価点: 中央 + 画像エリア 4 端 (image-px)
CENTER = (imax, imax)                                  # (2448, 2448)
TOP    = (imax, imax - orig_h / 2)                     # y=816  (画像エリア上端)
BOTTOM = (imax, imax + orig_h / 2)                     # y=4080
LEFT   = (imax - orig_w / 2, imax)                     # x=0
RIGHT  = (imax + orig_w / 2, imax)                     # x=4896
EVAL = {'center': CENTER, 'top': TOP, 'bottom': BOTTOM, 'left': LEFT, 'right': RIGHT}
IDEAL_CENTER_SX = off_x_norm * orig_w                  # -131.2 px


def report(name, fn_map):
    """fn_map: callable(query_xy_array (N,2)) -> src_xy (N,2)。
    各評価点で shift = map(p) - p を出す。"""
    pts = np.array(list(EVAL.values()), float)
    src = fn_map(pts)
    print(f"\n=== {name} ===")
    for (label, p), s in zip(EVAL.items(), src):
        sx, sy = s[0] - p[0], s[1] - p[1]
        mark = ''
        if label == 'center':
            mark = f'   (理想 sx≈{IDEAL_CENTER_SX:+.0f})'
        print(f"  {label:7s} shift=({sx:+8.2f}, {sy:+8.2f}){mark}")


def main():
    src, dst = build_points()
    print(f"CP数={len(src)}  動かした CP {moved_cp} off=({off_x_norm},{off_y_norm}) "
          f"= {off_x_norm*orig_w:+.0f}px")

    # --- 1. 現状: Gaussian IDW (Nadaraya-Watson) sigma=CP間隔 ---
    sigma = image_dim / max(rows, cols)   # 1224
    inv = 1.0 / (2.0 * sigma * sigma)
    def idw(pts):
        out = np.zeros_like(pts)
        for k, p in enumerate(pts):
            d2 = (dst[:, 0] - p[0])**2 + (dst[:, 1] - p[1])**2
            w = np.exp(-d2 * inv)
            out[k, 0] = (w * src[:, 0]).sum() / w.sum()
            out[k, 1] = (w * src[:, 1]).sum() / w.sum()
        return out
    report(f"1. Gaussian IDW (現状, sigma={sigma:.0f})  ※平滑化=非補間", idw)

    # --- 2. TPS (scipy legacy Rbf thin_plate) ---
    rx = Rbf(dst[:, 0], dst[:, 1], src[:, 0], function='thin_plate', smooth=0)
    ry = Rbf(dst[:, 0], dst[:, 1], src[:, 1], function='thin_plate', smooth=0)
    report("2. TPS thin_plate (補間だが affine 遠方項)",
           lambda pts: np.stack([rx(pts[:,0], pts[:,1]), ry(pts[:,0], pts[:,1])], 1))

    # --- 3. RBFInterpolator (近代 API, 正規化込) 各 kernel ---
    # 座標を [0,1] に正規化してから解く (条件数対策)。query も同じ scale。
    scale = image_dim
    dst_n = dst / scale
    src_n = src / scale
    for kernel, kw in [
        ('thin_plate_spline', {}),
        ('linear', {}),
        ('cubic', {}),
        ('multiquadric', {'epsilon': 1.0}),     # 正規化後の長さscale ~1
        ('gaussian', {'epsilon': 4.0}),          # 正規化後で CP間隔 ~0.25 → epsilon~4
        ('gaussian', {'epsilon': 8.0}),
    ]:
        try:
            interp = RBFInterpolator(dst_n, src_n, kernel=kernel, smoothing=0.0, **kw)
            def make(interp):
                return lambda pts: interp(pts / scale) * scale
            report(f"3. RBFInterpolator kernel={kernel} {kw}", make(interp))
        except Exception as e:
            print(f"\n=== 3. RBFInterpolator kernel={kernel} {kw} -> FAILED: {e} ===")

    # --- 4. MLS (affine, Schaefer 2006) ---
    # 各 query 点で、CP を距離で重み付けした affine 変換を解く。
    # 局所性 + 補間性 (CP上で重み無限大 → 厳密通過) を両立。
    def mls(pts, alpha=2.0):
        out = np.zeros_like(pts)
        P = dst  # 制御点 (変形後 = 元画像の格子点を動かした位置)
        Q = src  # 対応 (元位置)
        for k, v in enumerate(pts):
            d2 = ((P - v)**2).sum(1)
            d2 = np.maximum(d2, 1e-8)
            w = 1.0 / d2**alpha
            wsum = w.sum()
            pstar = (w[:, None] * P).sum(0) / wsum
            qstar = (w[:, None] * Q).sum(0) / wsum
            Phat = P - pstar
            Qhat = Q - qstar
            # affine M: minimize sum w |Phat M - Qhat|^2  → M = (Phat^T W Phat)^-1 Phat^T W Qhat
            A = (w[:, None] * Phat).T @ Phat
            B = (w[:, None] * Phat).T @ Qhat
            try:
                M = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                M = np.linalg.lstsq(A, B, rcond=None)[0]
            out[k] = (v - pstar) @ M + qstar
        return out
    report("4. MLS affine (Schaefer2006, 局所+補間)", mls)


if __name__ == '__main__':
    main()
