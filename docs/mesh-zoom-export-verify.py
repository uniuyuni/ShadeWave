"""
warp_mask_tps の disp_info 追従検証 (export / 拡大表示バグの回帰防止)。

coordinate_context.tcg_to_texture を再現した tcg_to_texture_fn を渡し、
preview(正方657) / export(実アスペクト full) / zoom(拡大) の 3 条件で
マスク mesh warp が composit と整合するか (中心に正しく変形が置かれ、ズームに
追従するか) を数値で確認する。

実行: python3 docs/mesh-zoom-export-verify.py
"""
from __future__ import annotations
import sys, os, types
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import importlib.util

# heavy import 回避の stub (mesh-parity-test.py と同じ流儀)
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

if 'effects' not in sys.modules:
    fake = types.ModuleType('effects')
    class FM:
        @staticmethod
        def get_param(p, k, default=None):
            d = {'mask_mesh_size': [4, 4], 'mask_mesh_control_points': {}}
            return p.get(k, default if default is not None else d.get(k))
    fake.Mask2Effect = FM; sys.modules['effects'] = fake
if 'params' not in sys.modules:
    fake = types.ModuleType('params'); sys.modules['params'] = fake
if 'config' not in sys.modules:
    fc = types.ModuleType('config'); fc._config = {'mesh_rbf_function': 'mls'}
    fc.get_config = lambda k: fc._config.get(k); sys.modules['config'] = fc

wc = _load('warp_correction', os.path.join(_ROOT, 'cores', 'distortion_correction', 'warp_correction.py'))
sys.modules['cores.distortion_correction.warp_correction'] = wc
pkg_cd = types.ModuleType('cores.distortion_correction'); pkg_cd.warp_correction = wc
sys.modules['cores.distortion_correction'] = pkg_cd
pkg_c = types.ModuleType('cores'); pkg_c.distortion_correction = pkg_cd
pkg_cm = types.ModuleType('cores.mask2'); pkg_c.mask2 = pkg_cm
sys.modules.setdefault('cores', pkg_c); sys.modules.setdefault('cores.mask2', pkg_cm)
mm = _load('cores.mask2.mask_mesh', os.path.join(_ROOT, 'cores', 'mask2', 'mask_mesh.py'))
warp_mask_tps = mm.warp_mask_tps


def crop_size_and_offset(tw, th, disp):
    ca = disp[2] / disp[3]; ta = tw / th
    if ca > ta:
        nw = tw; nh = int(tw / ca)
    else:
        nw = int(th * ca); nh = th
    return nw, nh, (tw - nw) // 2, (th - nh) // 2


def make_t2t(orig, texture_size, disp):
    """coordinate_context.tcg_to_texture の再現 (rotation なし)。
    入力は画像中心 px (norm*orig)。"""
    imax = max(orig) / 2.0
    tw, th = texture_size
    _, _, ox, oy = crop_size_and_offset(tw, th, disp)
    def t2t(cx, cy):
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - disp[0], cy - disp[1]
        cx, cy = cx * disp[4], cy * disp[4]
        return (cx + ox, cy + oy)
    return t2t


def make_t2f(orig):
    """coordinate_context.tcg_to_full_image の再現 (rotation/matrix なし)。
    入力は画像中心 px (norm*orig)。"""
    imax = max(orig) / 2.0
    def t2f(cx, cy):
        return (cx + imax, cy + imax)
    return t2f


def center_shift(orig, texture_size, disp, cps):
    """texture の中心 pixel で warp の shift (composit px) を測る。
    x 値を符号化した composit を warp して読む。"""
    w, h = texture_size
    grid = np.zeros((h, w), np.float32)
    grid[:] = np.arange(w, dtype=np.float32)[None, :] / w
    t2t = make_t2t(orig, texture_size, disp)
    t2f = make_t2f(orig)
    out = warp_mask_tps(grid.copy(), (4, 4), cps, orig, t2t, t2f)
    cy, cx = h // 2, w // 2
    return float(out[cy, cx]) * w - cx


def main():
    orig = (4896, 3264)            # landscape
    cps = {(2, 2): (0.05, 0.0)}    # 中央 CP を +x へ
    print(f"orig={orig} cps={cps}\n")

    # 1. preview: 正方 657, 全体表示, scale = 657/4896
    s_prev = 657 / 4896
    disp_prev = (0, 816, 4896, 3264, s_prev)   # crop原点 y=imax-orig_h/2=816
    sx = center_shift(orig, (657, 657), disp_prev, cps)
    # remap は「出力 pixel が入力のどこを読むか」なので、+x へ押した CP は
    # 中央 pixel では負の source shift として観測される。
    print(f"preview  texture=657²        scale={s_prev:.4f}  center_shift_x={sx:+.2f} px"
          f"  (期待: -0.05*4896*scale ≈ {-0.05*4896*s_prev:+.1f})")

    # 2. export: 実アスペクト 4896x3264, 全体表示, scale=1
    disp_exp = (0, 816, 4896, 3264, 1.0)
    sx2 = center_shift(orig, (4896, 3264), disp_exp, cps)
    print(f"export   texture=4896x3264   scale=1.0     center_shift_x={sx2:+.2f} px"
          f"  (期待: -0.05*4896 近傍 ≈ {-0.05*4896:+.1f})")

    # 3. zoom: 正方 657 に、画像中央 1/2 領域を 2倍ズーム表示
    #    crop rect = 中央 half: 原点(orig_w/4, 816+orig_h/4), size(orig_w/2, orig_h/2)
    s_zoom = 657 / (4896 / 2)      # half crop を 657 に → scale 2倍相当
    disp_zoom = (4896/4, 816 + 3264/4, 4896/2, 3264/2, s_zoom)
    sx3 = center_shift(orig, (657, 657), disp_zoom, cps)
    print(f"zoom2x   texture=657² (half) scale={s_zoom:.4f}  center_shift_x={sx3:+.2f} px"
          f"  (期待: preview の約2倍 ≈ {sx*2:+.1f})")

    print("\n判定:")
    print(f"  export が preview と同じ相対変形か: "
          f"{abs((sx2/4896) - (sx/657)) < 1e-3}  "
          f"(export rel={sx2/4896:+.5f}, preview rel={sx/657:+.5f})")
    print(f"  zoom が preview の約2倍か: ratio={sx3/sx:.2f} (期待≈2.0)")


if __name__ == '__main__':
    main()
