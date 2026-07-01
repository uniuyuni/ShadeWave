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

        # The persistent "Make mask" toggles stay state-bound (their state is durable).
        self.assertIn('StateBinding(\'inpaint\', False, "switch_inpaint")', source)
        self.assertIn('StateBinding(\'patchmatch_inpaint\', False, "switch_patchmatch_inpaint")', source)
        self.assertIn("def get_state_widget(self, widget, param, state_config):", source)
        self.assertIn("return widget.ids[widget_id].state == true_state", source)

    def test_erase_predict_is_durable_flag_not_button_state_binding(self):
        # Regression guard: the Erase buttons are LongPressScaledButtons whose state is
        # already "normal" when on_press fires (and the AI path is async, needing the flag
        # to persist across passes). Binding predict to the button state made Erase a
        # no-op. It must instead be a durable one-shot param flag.
        effects = (ROOT / "effects.py").read_text(encoding="utf-8")
        main_py = (ROOT / "main.py").read_text(encoding="utf-8")
        kv = (ROOT / "main.kv").read_text(encoding="utf-8")

        # No StateBinding may read the momentary Erase button state.
        self.assertNotIn('"button_inpaint_predict"', effects)
        self.assertNotIn('"button_patchmatch_inpaint_predict"', effects)

        # Defaults for the one-shot flags must exist so _get_param never KeyErrors.
        self.assertIn("'inpaint_predict': False", effects)
        self.assertIn("'patchmatch_inpaint_predict': False", effects)

        # The Erase buttons drive the flag through the trigger methods.
        self.assertIn("def _trigger_inpaint_predict(self):", main_py)
        self.assertIn("def _trigger_patchmatch_inpaint_predict(self):", main_py)
        self.assertIn("self.primary_param['inpaint_predict'] = True", main_py)
        self.assertIn("self.primary_param['patchmatch_inpaint_predict'] = True", main_py)
        self.assertIn("root._trigger_inpaint_predict()", kv)
        self.assertIn("root._trigger_patchmatch_inpaint_predict()", kv)


if __name__ == "__main__":
    unittest.main()
