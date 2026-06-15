import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import effects
from widgets import switch_reset_map


class EffectPipelineOrderTest(unittest.TestCase):
    def test_lut_is_split_between_input_and_look_stages(self):
        effect_layers = effects.create_effects()
        lv2_keys = list(effect_layers[2].keys())

        self.assertIn("input_lut", lv2_keys)
        self.assertIn("look_lut", lv2_keys)
        self.assertLess(lv2_keys.index("input_lut"), lv2_keys.index("exposure"))
        self.assertGreater(lv2_keys.index("look_lut"), lv2_keys.index("hls2rgb2"))
        self.assertLess(lv2_keys.index("look_lut"), lv2_keys.index("film_emulation"))

    def test_orton_is_a_finishing_effect_next_to_glow(self):
        effect_layers = effects.create_effects()

        self.assertNotIn("orton", effect_layers[1])
        lv2_keys = list(effect_layers[2].keys())
        self.assertLess(lv2_keys.index("solid_color"), lv2_keys.index("orton"))
        self.assertLess(lv2_keys.index("orton"), lv2_keys.index("glow"))
        self.assertLess(lv2_keys.index("glow"), lv2_keys.index("unsharp_mask"))

    def test_switch_targets_follow_pipeline_levels(self):
        targets = switch_reset_map.build_switch_reset_targets()

        self.assertEqual(targets["switch_lut"], (2, ["input_lut", "look_lut"], None))
        self.assertEqual(targets["switch_orton_effect"], (2, "orton", None))


if __name__ == "__main__":
    unittest.main()
