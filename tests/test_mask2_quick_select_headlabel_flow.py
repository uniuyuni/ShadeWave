from pathlib import Path
import inspect
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAIN_KV_PATH = ROOT / "main.kv"
MASK_EDITOR2_PATH = ROOT / "widgets" / "mask_editor2.py"
EXTENDED_PARAMS_PATH = ROOT / "cores" / "mask2" / "extended_params.py"
MAIN_PATH = ROOT / "main.py"

import effects
from widgets.switch_reset_map import build_switch_reset_targets


class Mask2QuickSelectHeadLabelFlowTest(unittest.TestCase):
    def test_quick_select_has_own_headlabel_and_history_subname(self):
        kv = MAIN_KV_PATH.read_text(encoding="utf-8")

        self.assertIn("id: switch_mask2_quick_select", kv)
        self.assertIn('text: " Quick Select"', kv)
        self.assertIn("subname='mask2_quick_select'", kv)
        self.assertLess(
            kv.index("id: switch_mask2_quick_select"),
            kv.index("id: spinner_mask2_edge_refine_mode"),
        )

    def test_quick_select_subparam_is_resettable(self):
        defaults = effects.Mask2Effect.get_param_dict({})
        options = effects.Mask2Effect.get_param_dict({}, "mask2_options")
        quick_select = effects.Mask2Effect.get_param_dict({}, "mask2_quick_select")
        targets = build_switch_reset_targets()

        self.assertTrue(defaults["switch_mask2_quick_select"])
        self.assertNotIn("mask2_edge_refine_mode", options)
        self.assertEqual(
            set(quick_select),
            {
                "switch_mask2_quick_select",
                "mask2_edge_refine_mode",
                "mask2_edge_refine_radius",
                "mask2_edge_refine_strength",
                "mask2_edge_refine_bias",
            },
        )
        self.assertEqual(
            targets["switch_mask2_quick_select"],
            (3, "mask2", "mask2_quick_select"),
        )

    def test_effect_binding_transfers_quick_select_switch(self):
        source_set = inspect.getsource(effects.Mask2Effect.set2widget)
        source_get = inspect.getsource(effects.Mask2Effect.set2param)

        self.assertIn("switch_mask2_quick_select", source_set)
        self.assertIn("switch_mask2_quick_select", source_get)

    def test_quick_select_switch_gates_all_edge_refine_paths(self):
        mask_editor_source = MASK_EDITOR2_PATH.read_text(encoding="utf-8")
        extended_source = EXTENDED_PARAMS_PATH.read_text(encoding="utf-8")

        self.assertIn("def _quick_select_switch_enabled", mask_editor_source)
        # GUI 側はゲート判定を headless の共有実装へ委譲する(スイッチのリテラルは
        # extended_params に集約)。委譲が残っていればゲートは効いている。
        self.assertIn("extended_params._quick_select_switch_enabled", mask_editor_source)
        self.assertIn("def _quick_select_switch_enabled", extended_source)
        self.assertIn('"switch_mask2_quick_select"', extended_source)

    def test_quick_select_param_changes_keep_mask_overlay_visible(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")

        self.assertIn('("mask2", "mask2_quick_select", "mask_geometry")', main_source)


if __name__ == "__main__":
    unittest.main()
