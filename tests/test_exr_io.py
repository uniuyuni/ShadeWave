"""
EXR 読み込み（cores.exr_io）と EXIF 取得の単体テスト。

- exiftool(PyExifTool) が EXR から ImageWidth/Height を返せること（ビューア経路の前提）
- read_exr がリニア RGB float32 を返し HDR(>1) を保持すること
- chromaticities → 色空間判定（Rec.709 既定 / ACES 等）が正しいこと
- ProPhoto への変換が任意原色を厳密に扱うこと
"""

import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import OpenEXR

import cores.exr_io as exr_io
import cores.colour_functions as cf


def _write_exr(path, rgb, chromaticities=None):
    header = {'compression': OpenEXR.ZIP_COMPRESSION, 'type': OpenEXR.scanlineimage}
    if chromaticities is not None:
        header['chromaticities'] = tuple(float(v) for v in chromaticities)
    channels = {'RGB': rgb.astype(np.float16)}
    OpenEXR.File(header, channels).write(path)


class TestExrIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='exr_test_')
        cls.h, cls.w = 8, 12
        cls.rgb = np.zeros((cls.h, cls.w, 3), dtype=np.float32)
        cls.rgb[..., 0] = 0.5
        cls.rgb[..., 1] = 1.0
        cls.rgb[..., 2] = 4.0   # HDR > 1.0
        cls.plain = os.path.join(cls.tmp, 'plain.exr')
        cls.acescg = os.path.join(cls.tmp, 'acescg.exr')
        _write_exr(cls.plain, cls.rgb, chromaticities=None)
        # ACEScg(AP1) 原色 + ACES white
        _write_exr(cls.acescg, cls.rgb,
                   chromaticities=(0.7130, 0.2930, 0.1650, 0.8300, 0.1280, 0.0440, 0.32168, 0.33767))

    def test_read_shape_dtype_and_hdr(self):
        img, chroma = exr_io.read_exr(self.plain)
        self.assertEqual(img.shape, (self.h, self.w, 3))
        self.assertEqual(img.dtype, np.float32)
        self.assertIsNone(chroma)
        # リニア値がそのまま（HALF 丸めの範囲で）読めている。HDR を保持。
        self.assertAlmostEqual(float(img[..., 2].max()), 4.0, places=2)
        self.assertGreater(float(img.max()), 1.0)

    def test_chromaticities_default_is_linear_srgb(self):
        self.assertEqual(exr_io.resolve_input_colourspace(None), 'Linear sRGB')

    def test_chromaticities_acescg_detected(self):
        _, chroma = exr_io.read_exr(self.acescg)
        self.assertIsNotNone(chroma)
        name = exr_io.resolve_input_colourspace(chroma)
        self.assertEqual(name, 'ACEScg')

    def test_custom_chromaticities_registered(self):
        # 既知に無い任意原色 -> 動的登録され、ProPhoto への変換が成立する
        chroma = (0.6295, 0.3409, 0.2867, 0.6108, 0.1489, 0.0712, 0.3154, 0.3316)
        name = exr_io.resolve_input_colourspace(chroma)
        self.assertIn(name, cf.RGB_COLOURSPACES)
        out = cf.RGB_to_RGB(self.rgb, name, 'ProPhoto RGB', 'Bradford',
                            apply_cctf_decoding=False, apply_gamut_mapping=False)
        self.assertEqual(out.shape, self.rgb.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_read_exr_linear_to_prophoto_preserves_hdr(self):
        out, src = exr_io.read_exr_linear(self.acescg, output_colourspace='ProPhoto RGB', cat='Bradford')
        self.assertEqual(src, 'ACEScg')
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 1.0)  # ガマットマップ無しで HDR 保持

    def test_thumbnail_is_display_range(self):
        thumb = exr_io.read_exr_thumbnail(self.acescg)
        self.assertEqual(thumb.dtype, np.float32)
        self.assertGreaterEqual(float(thumb.min()), 0.0)
        self.assertLessEqual(float(thumb.max()), 1.0)


class TestExrExif(unittest.TestCase):
    """exiftool(PyExifTool) が EXR から寸法メタを取得できること。"""

    def test_safe_get_metadata_returns_size(self):
        from utils.exiftool_safe import safe_get_metadata

        tmp = tempfile.mkdtemp(prefix='exr_exif_')
        path = os.path.join(tmp, 'sized.exr')
        rgb = np.zeros((16, 24, 3), dtype=np.float32)
        _write_exr(path, rgb)

        md = safe_get_metadata([path], common_args=['-b', '-s', '-a', '-G1'])
        self.assertEqual(len(md), 1)
        m = md[0]
        self.assertEqual(int(m.get('ImageWidth')), 24)
        self.assertEqual(int(m.get('ImageHeight')), 16)
        # get_exif_image_size が読む ImageSize タグ（"WxH"）も存在する
        self.assertEqual(m.get('ImageSize'), '24x16')


if __name__ == '__main__':
    unittest.main(verbosity=2)
