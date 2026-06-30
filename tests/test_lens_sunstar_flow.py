"""LensSimulatorEffect のサンスター（光条）サブ効果の headless 検証。

リアルさの肝を回帰として固定する:
  - クリップ気味のハイライト塊だけに出る（gating）。点光源が無い平坦画像では無変化。
  - 絞り羽根枚数から本数が物理整合で決まる（偶数=N / 奇数=2N）。
  - 絞り込む（F値を上げる）ほど光条が長くなる。
"""
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import effects
from effect_backends import lens_effect_adapter


def _point_source_image(size=257, value=8.0, bg=0.02):
    """暗背景の中央に飽和した白点光源を 1 つ置いた linear/HDR 画像。"""
    img = np.full((size, size, 3), bg, dtype=np.float32)
    c = size // 2
    img[c - 1:c + 2, c - 1:c + 2, :] = value  # 3x3 の飽和点
    return img


def _reach_radius(diff_lum, center, thresh=1e-3):
    """中心から、加算量が thresh を超える最遠半径（光条の到達距離）。"""
    ys, xs = np.where(diff_lum > thresh)
    if len(xs) == 0:
        return 0.0
    dr = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
    return float(dr.max())


def _count_spikes(diff_lum, center, radius, frac=0.2):
    """中心まわり半径 radius のリング上で角度サンプルし、スパイク本数を数える。

    スパイク間の谷はほぼ 0 なので、リング最大値の frac を超える区間（回転連結成分）の
    数を本数とする（平均基準だと振幅ジッターで弱いスパイクを取りこぼす）。
    """
    n = 2880
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xs = (center + radius * np.cos(ang)).astype(int)
    ys = (center + radius * np.sin(ang)).astype(int)
    h, w = diff_lum.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    prof = np.zeros(n, dtype=np.float64)
    prof[valid] = diff_lum[ys[valid], xs[valid]]
    if prof.max() <= 0:
        return 0
    above = prof > (prof.max() * frac)
    return sum(1 for i in range(n) if above[i] and not above[i - 1])


class SunstarTest(unittest.TestCase):
    def setUp(self):
        self.eff = effects.LensSimulatorEffect()

    def test_blade_count_physics(self):
        # 偶数=N本 / 奇数=2N本。
        self.assertEqual(lens_effect_adapter.spike_count_from_blades('6'), 6)
        self.assertEqual(lens_effect_adapter.spike_count_from_blades('8'), 8)
        self.assertEqual(lens_effect_adapter.spike_count_from_blades('9'), 18)
        self.assertEqual(lens_effect_adapter.spike_count_from_blades('7'), 14)
        self.assertEqual(lens_effect_adapter.spike_count_from_blades('bad'), 18)  # 既定9→18

    def test_flat_image_no_change(self):
        # 点光源が無い平坦画像では何も足さない（gating）。
        flat = np.full((129, 129, 3), 0.4, dtype=np.float32)
        out = lens_effect_adapter.apply_sunstar(flat, strength=100, length=50, threshold=60,
                                                blades='9', aperture=11.0, mag=1.0, orig_size=None)
        self.assertTrue(np.array_equal(out, flat))

    def test_emits_radial_spikes(self):
        img = _point_source_image()
        out = lens_effect_adapter.apply_sunstar(img, strength=100, length=60, threshold=50,
                                                blades='9', aperture=11.0, mag=1.0, orig_size=None)
        self.assertEqual(out.shape, img.shape)
        diff = (out - img).mean(axis=2)
        c = img.shape[0] // 2
        reach = _reach_radius(diff, c)
        # 光条が中心からそれなりに伸びている。
        self.assertGreater(reach, img.shape[0] * 0.10)

    def test_even_blades_give_n_spikes(self):
        # 偶数羽根は本数=羽根数（決定論的に厳密）。中心近く(全スパイクが存在する半径)で数える。
        img = _point_source_image(size=401)
        c = img.shape[0] // 2
        for blades, expected in (('6', 6), ('8', 8), ('14', 14)):
            out = lens_effect_adapter.apply_sunstar(img, strength=100, length=80, threshold=50,
                                                    blades=blades, aperture=14.0, mag=1.0, orig_size=None)
            diff = (out - img).mean(axis=2)
            ring = _reach_radius(diff, c) * 0.15
            count = _count_spikes(diff, c, ring)
            self.assertEqual(count, expected,
                             f"blades={blades}: expected {expected} spikes, got {count}")

    def test_odd_blades_double_spikes(self):
        # 奇数羽根は本数=2N（回折の物理）。9枚→18本、かつ 8枚(=8本)より明確に多い。
        img = _point_source_image(size=401)
        c = img.shape[0] // 2

        def spikes(blades):
            out = lens_effect_adapter.apply_sunstar(img, strength=100, length=80, threshold=50,
                                                    blades=blades, aperture=14.0, mag=1.0, orig_size=None)
            diff = (out - img).mean(axis=2)
            return _count_spikes(diff, c, _reach_radius(diff, c) * 0.15)

        count9 = spikes('9')
        count8 = spikes('8')
        self.assertEqual(count9, 18)          # 9枚 → 2N = 18本
        self.assertGreater(count9, count8 + 6)  # 8本に対しほぼ倍

    def test_aperture_lengthens_spikes(self):
        # 絞り込む（F大）ほど光条が長い。長い光条がフレームで切れないよう大きめ画像＋短めlength。
        img = _point_source_image(size=513)
        c = img.shape[0] // 2
        out_open = lens_effect_adapter.apply_sunstar(img, strength=100, length=30, threshold=50,
                                                     blades='9', aperture=2.0, mag=1.0, orig_size=None)
        out_stop = lens_effect_adapter.apply_sunstar(img, strength=100, length=30, threshold=50,
                                                     blades='9', aperture=14.0, mag=1.0, orig_size=None)
        reach_open = _reach_radius((out_open - img).mean(axis=2), c)
        reach_stop = _reach_radius((out_stop - img).mean(axis=2), c)
        self.assertGreater(reach_stop, reach_open * 1.3)

    def test_length_zero_is_minimal(self):
        # length=0 では光条が最小（ほぼコアのみ）。length を上げると明確に伸びる。
        img = _point_source_image(size=257)
        c = img.shape[0] // 2
        r0 = _reach_radius((lens_effect_adapter.apply_sunstar(img, 100, 0, 50, '9', 11.0, mag=1.0, orig_size=None) - img).mean(axis=2), c)
        r50 = _reach_radius((lens_effect_adapter.apply_sunstar(img, 100, 50, 50, '9', 11.0, mag=1.0, orig_size=None) - img).mean(axis=2), c)
        self.assertLess(r0, r50 * 0.3)        # length=0 は十分短い
        self.assertLess(r0, img.shape[0] * 0.08)  # 最小（コア近傍）

    def test_spike_width_constant_along_length(self):
        # 各光条は中心から先端まで太さ一定（角度ガウスではなく直交距離ガウスで描くため）。
        img = _point_source_image(size=401)
        c = img.shape[0] // 2
        out = lens_effect_adapter.apply_sunstar(img, strength=100, length=90, threshold=50,
                                                blades='8', aperture=14.0, mag=1.0, orig_size=None)
        diff = (out - img).mean(axis=2)
        R = _reach_radius(diff, c)

        def fwhm_px(radius):
            n = 3600
            ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            xs = (c + radius * np.cos(ang)).astype(int)
            ys = (c + radius * np.sin(ang)).astype(int)
            prof = diff[ys, xs]
            k = int(prof.argmax()); pk = prof[k]
            half = 0
            for j in range(1, n // 2):
                if prof[(k + j) % n] < 0.5 * pk:
                    half = j
                    break
            return 2 * half * (2.0 * math.pi / n) * radius  # FWHM[px]

        near = fwhm_px(R * 0.25)
        far = fwhm_px(R * 0.55)
        # 中心から広がらない＝遠方の幅が近傍の幅より極端に太くならない（1.6倍以内）。
        self.assertLess(far, near * 1.6)

    def _spike_fwhm(self, diff, c, radius):
        n = 3600
        ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        xs = (c + radius * np.cos(ang)).astype(int)
        ys = (c + radius * np.sin(ang)).astype(int)
        prof = diff[ys, xs]
        k = int(prof.argmax()); pk = prof[k]
        half = 0
        for j in range(1, n // 2):
            if prof[(k + j) % n] < 0.5 * pk:
                half = j
                break
        return 2 * half * (2.0 * math.pi / n) * radius

    def test_larger_source_thicker_spikes(self):
        # 光源が大きいほど光条も太い。ただし連動は弱め（強すぎると不自然＆ズーム差が出るため）。
        c = 200
        small = _point_source_image(size=401)               # 3x3 の点光源
        large = np.full((401, 401, 3), 0.02, dtype=np.float32)
        large[194:207, 194:207, :] = 10.0                   # 13x13 の大きな光源
        ds = (lens_effect_adapter.apply_sunstar(small, 100, 60, 50, '8', 11.0, mag=1.0, orig_size=None) - small).mean(axis=2)
        dl = (lens_effect_adapter.apply_sunstar(large, 100, 60, 50, '8', 11.0, mag=1.0, orig_size=None) - large).mean(axis=2)
        w_small = self._spike_fwhm(ds, c, _reach_radius(ds, c) * 0.3)
        w_large = self._spike_fwhm(dl, c, _reach_radius(dl, c) * 0.3)
        ratio = w_large / max(w_small, 1e-6)
        self.assertGreater(ratio, 1.2)   # 太くはなる
        self.assertLess(ratio, 3.5)      # が、強すぎない（mild）

    def test_spike_color_matches_source(self):
        # 光条の色は光源の色に追従する（アンバー→R>G>B、青→B>R）。
        c = 200

        def spike_rgb(src):
            d = lens_effect_adapter.apply_sunstar(src, 100, 70, 50, '8', 12.0, mag=1.0, orig_size=None) - src
            R = _reach_radius(d.mean(axis=2), c)
            n = 3600
            ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            xs = (c + R * 0.3 * np.cos(ang)).astype(int)
            ys = (c + R * 0.3 * np.sin(ang)).astype(int)
            lp = d.mean(axis=2)[ys, xs]
            k = int(lp.argmax())
            return d[ys[k], xs[k]]

        amber = np.full((401, 401, 3), 0.02, dtype=np.float32)
        amber[199:202, 199:202] = (10.0, 5.0, 1.0)
        px = spike_rgb(amber)
        self.assertTrue(px[0] > px[1] > px[2])  # 暖色を保持

        blue = np.full((401, 401, 3), 0.02, dtype=np.float32)
        blue[199:202, 199:202] = (1.0, 4.0, 10.0)
        pxb = spike_rgb(blue)
        self.assertGreater(pxb[2], pxb[0])      # 青を保持

    def test_make_diff_integration(self):
        # make_diff 経由でも動く（depth 不要・swirl/shaped 無しなので disp_info も不要）。
        img = _point_source_image(size=193)
        efconfig = SimpleNamespace(
            resolution_scale=1.0,
            upstream_hash=0,
            get_ai_depth_map=lambda **kw: None,
        )
        param = {
            'switch_lens_simulator': True,
            'sunstar_strength': 80,
            'sunstar_length': 50,
            'sunstar_threshold': 50,
            'aperture_blades': '6',
            'lens_aperture': 11.0,
        }
        out = self.eff.make_diff(img, param, efconfig)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(float((out - img).max()), 0.0)

    def test_shaped_bokeh_without_depth_spreads_hdr_point_highlight(self):
        # depth が無い場合でも、HDR点光源の周囲に絞り形状の広がりが見えること。
        img = _point_source_image(size=161, value=8.0, bg=0.02)
        out = lens_effect_adapter.apply_shaped_bokeh(
            img,
            depth_map=None,
            focus_depth=0.5,
            strength=100,
            radius=18,
            shape='Hexagon',
            rim=0.0,
        )
        diff = (out - img).mean(axis=2)
        c = img.shape[0] // 2
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        outside_source = (rr > 5) & (rr < 18)

        self.assertEqual(out.shape, img.shape)
        self.assertGreater(float(diff[outside_source].max()), 0.05)
        self.assertGreater(float(out.max()), float(img.max()))

    def test_shaped_bokeh_without_depth_ignores_uniform_bright_area(self):
        # depth 無しフォールバックは点状ハイライト専用。均一に明るい面は勝手に増光しない。
        img = np.full((81, 81, 3), 1.2, dtype=np.float32)
        out = lens_effect_adapter.apply_shaped_bokeh(
            img,
            depth_map=None,
            focus_depth=0.5,
            strength=100,
            radius=14,
            shape='Hexagon',
            rim=0.0,
        )
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_make_diff_off_when_zero(self):
        img = _point_source_image(size=129)
        efconfig = SimpleNamespace(
            resolution_scale=1.0, upstream_hash=0, get_ai_depth_map=lambda **kw: None,
        )
        param = {'switch_lens_simulator': True, 'sunstar_strength': 0}
        out = self.eff.make_diff(img, param, efconfig)
        self.assertIsNone(out)  # 全効果オフ → diff なし


if __name__ == "__main__":
    unittest.main()
