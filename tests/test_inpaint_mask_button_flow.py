import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]


class InpaintMaskButtonFlowTest(unittest.TestCase):
    def test_mask_buttons_use_plain_toggle_button_rule(self):
        kv = (ROOT / "main.kv").read_text(encoding="utf-8")

        self.assertIn("<ParamToggleButton@ToggleButton>:", kv)
        self.assertIn("ParamToggleButton:\n                                        id: switch_inpaint", kv)
        self.assertIn("ParamToggleButton:\n                                        id: switch_patchmatch_inpaint", kv)

    def test_inpaint_effect_reads_toggle_state(self):
        source = (ROOT / "effects.py").read_text(encoding="utf-8")

        self.assertIn('StateBinding(\'inpaint\', False, "switch_inpaint")', source)
        self.assertIn('StateBinding(\'patchmatch_inpaint\', False, "switch_patchmatch_inpaint")', source)
        self.assertIn('StateBinding(\'inpaint_predict\', False, "button_inpaint_predict")', source)
        self.assertIn('StateBinding(\'patchmatch_inpaint_predict\', False, "button_patchmatch_inpaint_predict")', source)
        self.assertIn("def get_state_widget(self, widget, param, state_config):", source)
        self.assertIn("return widget.ids[widget_id].state == true_state", source)


if __name__ == "__main__":
    unittest.main()
