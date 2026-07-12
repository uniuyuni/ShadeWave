import copy
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores import core
from effect_backends import hls_adjust_adapter

try:
    from effect_backends import _hls_adjust_metal
    if not _hls_adjust_metal.metal_available():
        _hls_adjust_metal = None
except Exception:  # pragma: no cover - depends on local build state.
    _hls_adjust_metal = None


def _make_hls_image(rng, h, w, channels, gain_max=1.0):
    hue = (rng.random((h, w), dtype=np.float32) * 360.0).astype(np.float32)
    lightness = rng.random((h, w), dtype=np.float32).astype(np.float32)
    saturation = rng.random((h, w), dtype=np.float32).astype(np.float32)
    planes = [hue, lightness, saturation]
    if channels > 3:
        gain = (rng.random((h, w), dtype=np.float32) * gain_max).astype(np.float32)
        planes.append(gain)
    for _ in range(4, channels):
        planes.append(rng.random((h, w), dtype=np.float32).astype(np.float32))
    return np.stack(planes, axis=-1).astype(np.float32)


def _setting(color_name, adjust, kernel_size=None):
    setting = dict(core.HLS_COLOR_SETTING[color_name])
    setting["adjust"] = list(adjust)
    if kernel_size is not None:
        setting["kernel_size"] = kernel_size
    return setting


class HlsAdjustBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = hls_adjust_adapter.backend_status()

        self.assertEqual(status.effect, "hls_adjust")
        self.assertIn(
            status.backend,
            {
                "effect_backends._hls_adjust_metal",
                "cores.core",
            },
        )

    def _assert_parity(self, hls_img, color_settings, resolution_scale):
        if _hls_adjust_metal is None:
            self.skipTest("hls_adjust metal backend is not built/available")

        core_settings = copy.deepcopy(color_settings)
        metal_settings = copy.deepcopy(color_settings)

        expected = core.adjust_hls_colors(np.array(hls_img), core_settings, resolution_scale)
        actual = hls_adjust_adapter.adjust_hls_colors(np.array(hls_img), metal_settings, resolution_scale)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1.0e-4, atol=1.0e-5)
        return actual

    def test_parity_4ch_sdr(self):
        rng = np.random.default_rng(101)
        image = _make_hls_image(rng, 48, 64, 4, gain_max=1.0)
        settings = [_setting("red", [0.1, 0.05, 0.1])]

        self._assert_parity(image, settings, 1.0)

    def test_parity_4ch_hdr_gain(self):
        rng = np.random.default_rng(102)
        image = _make_hls_image(rng, 48, 64, 4, gain_max=3.0)
        settings = [_setting("sky", [5.0, 0.2, 0.1])]

        self._assert_parity(image, settings, 1.0)

    def test_parity_3ch(self):
        rng = np.random.default_rng(103)
        image = _make_hls_image(rng, 48, 64, 3)
        settings = [_setting("green", [-0.05, 0.0, 0.1])]

        self._assert_parity(image, settings, 1.0)

    def test_parity_5ch_extra_channel_copied(self):
        rng = np.random.default_rng(104)
        image = _make_hls_image(rng, 48, 64, 5, gain_max=2.0)
        settings = [_setting("skin", [-2.0, 0.1, -0.05])]

        actual = self._assert_parity(image, settings, 1.0)

        # 5ch目(ch index 4)は完全にコピーされているはず。
        np.testing.assert_allclose(np.asarray(actual)[..., 4], image[..., 4], rtol=1.0e-6, atol=1.0e-6)

    def test_parity_multiple_settings_mixed_kernel_size(self):
        rng = np.random.default_rng(105)
        image = _make_hls_image(rng, 48, 64, 4, gain_max=2.0)
        settings = [
            _setting("red", [0.1, 0.05, 0.1]),
            _setting("skin", [-2.0, 0.1, -0.05]),
        ]

        self._assert_parity(image, settings, 1.0)

    def test_parity_small_resolution_scale(self):
        rng = np.random.default_rng(106)
        image = _make_hls_image(rng, 48, 64, 4, gain_max=1.5)
        settings = [
            _setting("red", [0.1, 0.05, 0.1]),
            _setting("skin", [-2.0, 0.1, -0.05]),
        ]

        self._assert_parity(image, settings, 0.05)

    def test_parity_negative_adjust(self):
        rng = np.random.default_rng(107)
        image = _make_hls_image(rng, 48, 64, 4, gain_max=1.0)
        settings = [_setting("blue", [-30.0, -0.5, -0.8])]

        self._assert_parity(image, settings, 1.0)

    def test_parity_hue_normalization_positive_branch(self):
        rng = np.random.default_rng(108)
        image = _make_hls_image(rng, 48, 64, 4, gain_max=1.0)
        settings = [_setting("purple", [185.0, 0.1, 0.1])]

        self._assert_parity(image, settings, 1.0)

    def test_input_array_is_not_modified(self):
        if _hls_adjust_metal is None:
            self.skipTest("hls_adjust metal backend is not built/available")

        rng = np.random.default_rng(109)
        image = _make_hls_image(rng, 32, 40, 4, gain_max=1.5)
        original = image.copy()
        settings = [_setting("red", [0.1, 0.05, 0.1])]

        hls_adjust_adapter.adjust_hls_colors(image, settings, 1.0)

        np.testing.assert_array_equal(image, original)

    def test_empty_settings_falls_back_to_core(self):
        rng = np.random.default_rng(110)
        image = _make_hls_image(rng, 16, 20, 4, gain_max=1.2)

        expected = core.adjust_hls_colors(np.array(image), [], 1.0)
        actual = hls_adjust_adapter.adjust_hls_colors(np.array(image), [], 1.0)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
