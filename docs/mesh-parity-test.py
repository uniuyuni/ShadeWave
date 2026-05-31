"""
画像 mesh と マスク mesh の同一性テスト。

同じ test 画像 + 同じ tcg_info + 同じ CP データを与えて、両者の warp 結果が
pixel-wise に一致するかを確認する。連動コピー機能 (画像 mesh CP → マスク mesh CP)
の前提が成り立つかを検証する。

実行: python3 docs/mesh-parity-test.py
"""
from __future__ import annotations

import sys
import os
import numpy as np

# project root を path に追加
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import importlib.util

# 重い依存 (numba 等) を踏まずに、ファイル指定で直接モジュールを load する。
# effects などの heavy import が連鎖しないよう、必要分だけ stub する。

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# stub heavy modules to avoid loading the whole app
def _stub_effects_module():
    """cores.mask2.mask_mesh は effects.Mask2Effect.get_param を使うので最小実装で stub。"""
    if 'effects' in sys.modules:
        return
    import types
    fake = types.ModuleType('effects')

    class FakeMask2Effect:
        @staticmethod
        def get_param(param, key, default=None):
            if default is not None:
                return param.get(key, default)
            defaults = {'mask_mesh_size': [4, 4], 'mask_mesh_control_points': {}}
            return param.get(key, defaults.get(key))

    fake.Mask2Effect = FakeMask2Effect
    sys.modules['effects'] = fake


# まず params の依存を避けるため、warp_correction.py だけ load
def _stub_params_module():
    if 'params' in sys.modules:
        return
    import types
    fake = types.ModuleType('params')

    def tcg_to_ref_image(cx, cy, ref_img, tcg_info, apply_disp_info=False, apply_ref_img_divide=False):
        # 簡易再実装 (apply_disp_info=False のみ):
        # [-0.5, +0.5] norm → [0, max(orig)] image-px
        orig = tcg_info['original_img_size']
        imax = max(orig) / 2.0
        # denorm (per-dim)
        px = cx * orig[0]
        py = cy * orig[1]
        # center_rotate (rotation=0 / matrix=I のみ想定)
        # ...そのまま
        px += imax
        py += imax
        return (px, py)

    fake.tcg_to_ref_image = tcg_to_ref_image
    sys.modules['params'] = fake


_stub_effects_module()
_stub_params_module()

# cores.distortion_correction の __init__.py が numba を引くので、別名で warp_correction
# を直接 load し、cores.distortion_correction.warp_correction としても登録する。
warp_correction = _load_module(
    'warp_correction',
    os.path.join(_ROOT, 'cores', 'distortion_correction', 'warp_correction.py'),
)
# cores.mask2.mask_mesh から import される名前にも登録
sys.modules['cores.distortion_correction.warp_correction'] = warp_correction
# 同様に親パッケージも stub
import types as _types
_pkg_cd = _types.ModuleType('cores.distortion_correction')
_pkg_cd.warp_correction = warp_correction
sys.modules['cores.distortion_correction'] = _pkg_cd
_pkg_c = _types.ModuleType('cores')
_pkg_c.distortion_correction = _pkg_cd
# cores.mask2 も同様
_pkg_cm = _types.ModuleType('cores.mask2')
_pkg_c.mask2 = _pkg_cm
sys.modules.setdefault('cores', _pkg_c)
sys.modules.setdefault('cores.mask2', _pkg_cm)

warp_mesh = warp_correction.warp_mesh
outer_ring_pins_tcg = warp_correction.outer_ring_pins_tcg

mask_mesh = _load_module(
    'cores.mask2.mask_mesh',
    os.path.join(_ROOT, 'cores', 'mask2', 'mask_mesh.py'),
)
warp_mask_tps = mask_mesh.warp_mask_tps


def build_test_image(N: int) -> np.ndarray:
    """N×N の格子パターン (見た目で warp が分かりやすい test 画像)。"""
    img = np.zeros((N, N), dtype=np.float32)
    for r in range(0, N, 50):
        img[r:r + 10, :] = 1.0
    for c in range(0, N, 50):
        img[:, c:c + 10] = 1.0
    # 中央に大きい○も入れる (warp 効果の中心観察用)
    cy, cx = N // 2, N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    img += np.clip(1.0 - rad / (N * 0.25), 0, 1) * 0.5
    return img.astype(np.float32)


def diff_stats(name: str, a: np.ndarray, b: np.ndarray):
    print(f"=== {name} ===")
    print(f"  画像 mesh shape: {a.shape}, dtype: {a.dtype}, range: [{a.min():.3f}, {a.max():.3f}]")
    print(f"  マスク mesh shape: {b.shape}, dtype: {b.dtype}, range: [{b.min():.3f}, {b.max():.3f}]")
    if a.shape != b.shape:
        print(f"  shape mismatch — 比較不能")
        return
    diff = np.abs(a - b)
    print(f"  pixel-wise diff: max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")
    print(f"  np.allclose (atol=1e-3): {np.allclose(a, b, atol=1e-3)}")
    print(f"  np.allclose (atol=1e-2): {np.allclose(a, b, atol=1e-2)}")
    print(f"  np.allclose (atol=1e-1): {np.allclose(a, b, atol=1e-1)}")


def _measure_shifts(warp_fn, size, get_src_x_norm, get_src_y_norm):
    """warp_fn: callable(src_image) -> warped_image. size: 画像サイズ N (NxN square)。
    中央 pixel での shift_x, shift_y を測る。"""
    # X 方向の shift を測る用画像 (x 値を value に符号化)
    src_x_grid = np.zeros((size, size), dtype=np.float32)
    xx = np.arange(size, dtype=np.float32)[None, :]
    src_x_grid[:] = xx / size
    warped_x = warp_fn(src_x_grid.copy())
    cy, cx = size // 2, size // 2
    src_x_norm = float(warped_x[cy, cx])
    shift_x_px = src_x_norm * size - cx

    # Y 方向の shift
    src_y_grid = np.zeros((size, size), dtype=np.float32)
    yy = np.arange(size, dtype=np.float32)[:, None]
    src_y_grid[:] = yy / size
    warped_y = warp_fn(src_y_grid.copy())
    src_y_norm = float(warped_y[cy, cx])
    shift_y_px = src_y_norm * size - cy

    return shift_x_px, shift_y_px


def measure_center_shift_image(image_dim, orig_size, cps):
    tcg_info = {
        'original_img_size': orig_size,
        'disp_info': (0.0, 0.0, 1.0, 1.0, 1.0),
        'rotation': 0.0, 'rotation2': 0.0, 'flip_mode': 0,
        'matrix': np.eye(3),
    }
    def _warp(src):
        return warp_mesh(src, (4, 4), cps, tcg_info=tcg_info,
                         interpolation='bilinear', border_value=0)
    return _measure_shifts(_warp, image_dim, None, None)


def measure_center_shift_mask(composit_size, orig_size, cps):
    def _warp(src):
        return warp_mask_tps(src, (4, 4), cps, orig_size)
    return _measure_shifts(_warp, composit_size, None, None)


def run_one_case(case_name: str, cps: dict, orig_size, composit_size: int):
    print(f"\n----- {case_name} -----")
    image_dim = max(orig_size)
    print(f"  orig_img_size = {orig_size}")
    print(f"  画像 mesh image_dim = {image_dim} (core.rotation 後)")
    print(f"  マスク mesh composit = {composit_size}")
    print(f"  cps = {cps}")

    img_sx, img_sy = measure_center_shift_image(image_dim, orig_size, cps)
    msk_sx, msk_sy = measure_center_shift_mask(composit_size, orig_size, cps)

    rel_img_x = img_sx / image_dim
    rel_img_y = img_sy / image_dim
    rel_msk_x = msk_sx / composit_size
    rel_msk_y = msk_sy / composit_size

    def _ratio(a, b):
        return b / a if abs(a) > 1e-9 else float('inf')

    print(f"  画像 mesh center shift = ({img_sx:+.2f}, {img_sy:+.2f}) px "
          f"= ({rel_img_x*100:+.4f}%, {rel_img_y*100:+.4f}%)")
    print(f"  マスク mesh center shift = ({msk_sx:+.2f}, {msk_sy:+.2f}) px "
          f"= ({rel_msk_x*100:+.4f}%, {rel_msk_y*100:+.4f}%)")
    rx = _ratio(rel_img_x, rel_msk_x)
    ry = _ratio(rel_img_y, rel_msk_y)
    # 0.1px 程度の補間ノイズは、画像側/マスク側で相対値の分母が違うと ratio が
    # 大きく見える。実質ゼロ変形の軸は absolute px で判定する。
    ok_x = (abs(img_sx) < 0.5 and abs(msk_sx) < 0.5) or abs(rx - 1.0) < 0.05
    ok_y = (abs(img_sy) < 0.5 and abs(msk_sy) < 0.5) or abs(ry - 1.0) < 0.05
    ok = ok_x and ok_y
    print(f"  ratio (mask / image) = (x={rx:.3f}, y={ry:.3f})    "
          f"{'✓ OK (両者の相対 shift がほぼ一致)' if ok else '✗ MISMATCH'}")


def _set_rbf_function(name):
    """テスト中に config.get_config('mesh_rbf_function') を一時的に書き換える簡易 stub。"""
    import config as _cfg
    if _cfg._config is None:
        # init_config が呼ばれていない (stand-alone test) → minimum dict を直接 set
        _cfg._config = {}
    _cfg._config['mesh_rbf_function'] = name


def main():
    print("画像 mesh vs マスク mesh: 同一性テスト")
    print(f"  warp_mesh        = cores.distortion_correction.warp_correction.warp_mesh")
    print(f"  warp_mask_tps    = cores.mask2.mask_mesh.warp_mask_tps")
    print()

    # config stub: テストでは config モジュール経由で RBF を切替
    import types
    if 'config' not in sys.modules:
        fake_cfg = types.ModuleType('config')
        fake_cfg._config = {'mesh_rbf_function': 'thin_plate'}
        fake_cfg.get_config = lambda k: fake_cfg._config.get(k)
        sys.modules['config'] = fake_cfg

    for rbf_name in ('mls', 'thin_plate'):
        _set_rbf_function(rbf_name)
        print(f"\n========== RBF: {rbf_name} ==========")

        run_one_case(
            "Case 1: square original (1024x1024), composit 512",
            cps={(2, 2): (0.05, 0.0)},
            orig_size=(1024, 1024),
            composit_size=512,
        )
        run_one_case(
            "Case 2: landscape (4896x3264), composit 657 (= 実機相当)",
            cps={(2, 2): (0.05, 0.0)},
            orig_size=(4896, 3264),
            composit_size=657,
        )
        run_one_case(
            "Case 3: portrait (3264x4896), composit 657",
            cps={(2, 2): (0.05, 0.0)},
            orig_size=(3264, 4896),
            composit_size=657,
        )
        run_one_case(
            "Case 4: corner CP (4896x3264), composit 657",
            cps={(1, 1): (0.08, -0.03)},
            orig_size=(4896, 3264),
            composit_size=657,
        )


if __name__ == '__main__':
    main()
