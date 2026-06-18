import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import effects
import params


class HLSPipelineChannelGuardTest(unittest.TestCase):
    def test_hls2rgb_converts_four_channel_hls_even_when_params_now_inactive(self):
        rgb = np.random.default_rng(2).random((8, 8, 3), dtype=np.float32)
        param = {}
        params.set_image_param(param, rgb)
        param.update({
            "switch_color_mixer": True,
            "switch_hls_skin": True,
            "hls_skin_hue": 12.0,
            "hls_skin_lum": 0.0,
            "hls_skin_sat": 0.0,
        })
        efconfig = effects.EffectConfig()
        efconfig.disp_info = (0, 0, 8, 8, 1.0)
        efconfig.resolution_scale = 1.0

        hls = effects.RGB2HLSEffect().make_diff(rgb, param, efconfig)
        self.assertEqual((8, 8, 4), hls.shape)

        param["hls_skin_hue"] = 0.0
        hls2rgb = effects.HLS2RGBEffect()
        converted = hls2rgb.make_diff(hls, param, efconfig)

        self.assertIsNotNone(converted)
        self.assertEqual((8, 8, 3), converted.shape)
        self.assertEqual((8, 8, 3), hls2rgb.apply_diff(hls).shape)

    def test_hls2rgb_noops_rgb_input_when_pipeline_inactive(self):
        rgb = np.random.default_rng(3).random((8, 8, 3), dtype=np.float32)
        param = {}
        params.set_image_param(param, rgb)
        efconfig = effects.EffectConfig()

        hls2rgb = effects.HLS2RGBEffect()
        self.assertIsNone(hls2rgb.make_diff(rgb, param, efconfig))
        self.assertIs(hls2rgb.apply_diff(rgb), rgb)


if __name__ == "__main__":
    unittest.main()
