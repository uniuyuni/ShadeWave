"""
MLS の端残留シフトの切り分け。実機 mask warp パイプライン (coarse grid + INTER_CUBIC
upscale) を忠実に再現し、画像エリア端での shift を測る。

実機ログ (mls): CP(2,2) off=(-0.0679, 0.0027), 端 top/bottom y=±10px, left/right x=±3px。

検証する仮説:
  H1: coarse grid + INTER_CUBIC の overshoot が端残留の主因
      → 解析 MLS(端) ≈ 0 なのに pipeline(端) が ±10 なら H1
  H2: MLS affine の shear/scale 場が境界まで漏れる (構造的)
      → 解析 MLS(端) 自体が ±10 なら H2
  対策A: 画像エリア境界に identity pin を追加 → 境界が補間で固定されるか
実行: python3 docs/mesh-edge-diagnostic.py
"""
from __future__ import annotations
import numpy as np
import cv2

orig_w, orig_h = 4896.0, 3264.0
imax = max(orig_w, orig_h) / 2.0
image_dim = max(orig_w, orig_h)
rows = cols = 4
moved_cp = (2, 2)
off_x_norm, off_y_norm = -0.0679, 0.0027   # 実機ログ値


def tcg_to_imgpx(nx, ny):
    return (nx * orig_w + imax, ny * orig_h + imax)


def outer_ring_pins_tcg(margin=0.15):
    e = 0.5 + margin
    return [(-e, -e), (0.0, -e), (e, -e),
            (-e, 0.0),            (e, 0.0),
            (-e, e),  (0.0, e),  (e, e)]


def build_points(extra_edge_pins=False, n_edge=5):
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
    if extra_edge_pins:
        # 画像エリア境界 (TCG ±0.5 のちょうど縁) に identity pin を密に追加
        ts = np.linspace(-0.5, 0.5, n_edge)
        edge = []
        for t in ts:
            edge += [(t, -0.5), (t, 0.5), (-0.5, t), (0.5, t)]
        for tx, ty in edge:
            px, py = tcg_to_imgpx(tx, ty)
            src.append((px, py)); dst.append((px, py))
    return np.array(src, float), np.array(dst, float)


def mls_affine(P, Q, G, alpha=2.0):
    diff = G[:, None, :] - P[None, :, :]
    d2 = np.maximum((diff * diff).sum(2), 1e-8)
    w = 1.0 / np.power(d2, alpha)
    wsum = w.sum(1)
    pstar = (w[:, :, None] * P[None]).sum(1) / wsum[:, None]
    qstar = (w[:, :, None] * Q[None]).sum(1) / wsum[:, None]
    Phat = P[None] - pstar[:, None]
    Qhat = Q[None] - qstar[:, None]
    A = np.einsum('mn,mni,mnj->mij', w, Phat, Phat)
    B = np.einsum('mn,mni,mnj->mij', w, Phat, Qhat)
    diagA = A[:, 0, 0] + A[:, 1, 1]
    ridge = (1e-8 * np.maximum(diagA, 1e-12))[:, None, None] * np.eye(2)[None]
    M = np.linalg.solve(A + ridge, B)
    return np.einsum('mi,mij->mj', (G - pstar), M) + qstar


# 画像エリア境界点 (image-px): 上下端 = y 816/4080, 左右端 = x 0/4896
EDGES = {
    'center': (imax, imax),
    'top':    (imax, 816.0),
    'bottom': (imax, 4080.0),
    'left':   (0.0, imax),
    'right':  (4896.0, imax),
}


def analytic(label_pts, P, Q, alpha):
    pts = np.array(list(label_pts.values()), float)
    out = mls_affine(P, Q, pts, alpha)
    return {k: (out[i, 0] - p[0], out[i, 1] - p[1])
            for i, (k, p) in enumerate(label_pts.items())}


def pipeline(P, Q, alpha, w=657, h=657, interp=cv2.INTER_CUBIC, clamp_bbox=False, grid_step=32):
    """実機 mask パイプライン: coarse grid を image-px [0,image_dim] で sample → MLS →
    composit scale に変換 → cubic upscale → 各端で shift を測る。"""
    grid_h = max(2, (h + grid_step - 1) // grid_step)
    grid_w = max(2, (w + grid_step - 1) // grid_step)
    cx = np.linspace(0, image_dim - 1, grid_w)
    cy = np.linspace(0, image_dim - 1, grid_h)
    cnx, cny = np.meshgrid(cx, cy)
    G = np.stack([cnx.ravel(), cny.ravel()], 1)
    mapped = mls_affine(P, Q, G, alpha)
    if clamp_bbox:
        # CP/src の bounding box (= 絵の領域) の外では identity に固定 (外挿させない)
        x0, y0 = P[:, 0].min(), P[:, 1].min()
        x1, y1 = P[:, 0].max(), P[:, 1].max()
        # ↑ pin 込みだと bbox が広がるので、CP grid (絵の領域) で算出すべき。
        #   ここでは絵の領域 = image-px x[0,4896] y[816,4080] を直接使う。
        x0, x1 = 0.0, image_dim
        y0, y1 = 816.0, 4080.0
        outside = (G[:, 0] < x0) | (G[:, 0] > x1) | (G[:, 1] < y0) | (G[:, 1] > y1)
        mapped[outside] = G[outside]
    map_x_c = mapped[:, 0].reshape(grid_h, grid_w) * ((w - 1) / (image_dim - 1))
    map_y_c = mapped[:, 1].reshape(grid_h, grid_w) * ((h - 1) / (image_dim - 1))
    map_x = cv2.resize(map_x_c.astype(np.float32), (w, h), interpolation=interp)
    map_y = cv2.resize(map_y_c.astype(np.float32), (w, h), interpolation=interp)
    # composit 内の画像エリア端 (img_area=657x438, off=(0,109))
    img_area_h = int(round(h * orig_h / image_dim))
    off_y = (h - img_area_h) // 2
    pts = {
        'center': (h // 2, w // 2),
        'top':    (off_y, w // 2),
        'bottom': (off_y + img_area_h - 1, w // 2),
        'left':   (h // 2, 0),
        'right':  (h // 2, w - 1),
    }
    return {k: (float(map_x[py, px]) - px, float(map_y[py, px]) - py)
            for k, (py, px) in pts.items()}


def show(title, d):
    print(f"\n=== {title} ===")
    for k, (sx, sy) in d.items():
        print(f"  {k:7s} shift=({sx:+8.2f}, {sy:+8.2f})")


def main():
    alpha = 2.0
    P, Q = build_points(extra_edge_pins=False)
    print(f"CP off=({off_x_norm},{off_y_norm}) = ({off_x_norm*orig_w:+.0f}, {off_y_norm*orig_h:+.0f})px  alpha={alpha}")

    show("解析 MLS (端 = 画像エリア境界, image-px)", analytic(EDGES, P, Q, alpha))
    show("pipeline MLS (coarse+CUBIC, composit-px)  ※実機相当", pipeline(P, Q, alpha))
    show("pipeline MLS (coarse+LINEAR)", pipeline(P, Q, alpha, interp=cv2.INTER_LINEAR))

    Pe, Qe = build_points(extra_edge_pins=True, n_edge=5)
    print(f"\n--- 画像エリア境界 identity pin 追加 (各辺5点) → CP数 {len(Pe)} ---")
    show("解析 MLS + edge pin", analytic(EDGES, Pe, Qe, alpha))
    show("pipeline MLS + edge pin (CUBIC)", pipeline(Pe, Qe, alpha))

    # 対策: パディング帯 (絵の領域外) を identity 固定して外挿のにじみを止める
    show("pipeline MLS + clamp_bbox (CUBIC) ★対策案", pipeline(P, Q, alpha, clamp_bbox=True))
    show("pipeline MLS + clamp_bbox (LINEAR)", pipeline(P, Q, alpha, interp=cv2.INTER_LINEAR, clamp_bbox=True))
    # coarse 解像度を細かく (grid_step=4) → coarse-grid 起因なら消えるはず
    show("pipeline MLS grid_step=4 (CUBIC)", pipeline(P, Q, alpha, grid_step=4))
    show("pipeline MLS grid_step=4 (CUBIC)+clamp", pipeline(P, Q, alpha, grid_step=4, clamp_bbox=True))
    show("pipeline MLS grid_step=8 (CUBIC)", pipeline(P, Q, alpha, grid_step=8))

    # ★本命: coarse grid を image-px 密度 (image_dim/32 ノード) で取る
    def pipeline_imgdensity(P, Q, alpha, w=657, h=657, interp=cv2.INTER_CUBIC):
        grid_step = 32
        gn = max(2, int((image_dim + grid_step - 1) // grid_step))  # 153 ノード
        cx = np.linspace(0, image_dim - 1, gn)
        cnx, cny = np.meshgrid(cx, cx)
        G = np.stack([cnx.ravel(), cny.ravel()], 1)
        mapped = mls_affine(P, Q, G, alpha)
        mxc = mapped[:, 0].reshape(gn, gn) * ((w - 1) / (image_dim - 1))
        myc = mapped[:, 1].reshape(gn, gn) * ((h - 1) / (image_dim - 1))
        map_x = cv2.resize(mxc.astype(np.float32), (w, h), interpolation=interp)
        map_y = cv2.resize(myc.astype(np.float32), (w, h), interpolation=interp)
        img_area_h = int(round(h * orig_h / image_dim)); off_y = (h - img_area_h) // 2
        pts = {'center': (h//2, w//2), 'top': (off_y, w//2),
               'bottom': (off_y+img_area_h-1, w//2), 'left': (h//2, 0), 'right': (h//2, w-1)}
        return {k: (float(map_x[py,px])-px, float(map_y[py,px])-py) for k,(py,px) in pts.items()}
    show("★ pipeline MLS image-px密度 grid (153ノード, CUBIC)", pipeline_imgdensity(P, Q, alpha))


if __name__ == '__main__':
    main()
