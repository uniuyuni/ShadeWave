import os
import sys
import unittest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import effects
import pipeline


class PipelinePreviewEffectConfigTest(unittest.TestCase):
    def test_edit_tab_is_cropped_preview_state(self):
        efconfig = effects.EffectConfig()

        mask2_geometry_full_preview = pipeline._configure_preview_effect_config(
            efconfig,
            "Ed",
            mask2_active=False,
        )

        self.assertFalse(mask2_geometry_full_preview)
        self.assertEqual(efconfig.current_tab, "Ed")
        self.assertFalse(efconfig.full_preview)
        self.assertFalse(efconfig.crop_editing)

    def test_geometry_tab_edits_crop_when_mask2_is_inactive(self):
        efconfig = effects.EffectConfig()

        mask2_geometry_full_preview = pipeline._configure_preview_effect_config(
            efconfig,
            "Ge",
            mask2_active=False,
        )

        self.assertFalse(mask2_geometry_full_preview)
        self.assertEqual(efconfig.current_tab, "Ge")
        self.assertTrue(efconfig.full_preview)
        self.assertTrue(efconfig.crop_editing)

    def test_geometry_tab_with_mask2_is_full_preview_but_not_crop_editing(self):
        efconfig = effects.EffectConfig()

        mask2_geometry_full_preview = pipeline._configure_preview_effect_config(
            efconfig,
            "Ge",
            mask2_active=True,
        )

        self.assertTrue(mask2_geometry_full_preview)
        self.assertEqual(efconfig.current_tab, "Ge")
        self.assertTrue(efconfig.full_preview)
        self.assertFalse(efconfig.crop_editing)

    def test_debug_param_summary_uses_current_grain_and_vignette_keys(self):
        old_debug = pipeline._DEBUG_PIPELINE_STATS
        pipeline._DEBUG_PIPELINE_STATS = True
        try:
            grain = pipeline._debug_pipeline_param_summary(
                "grain",
                {},
                effect=effects.GrainEffect(),
            )
            vignette = pipeline._debug_pipeline_param_summary(
                "vignette",
                {},
                effect=effects.VignetteEffect(),
            )
        finally:
            pipeline._DEBUG_PIPELINE_STATS = old_debug

        self.assertIn("switch_grain=True(default)", grain)
        self.assertIn("grain_intensity=0(default)", grain)
        self.assertIn("grain_color_noise_ratio=0(default)", grain)
        self.assertNotIn("grain_radius", grain)
        self.assertIn("switch_vignette=True(default)", vignette)
        self.assertIn("vignette_radius_percent=80(default)", vignette)
        self.assertNotIn("vignette_radius=", vignette)


if __name__ == "__main__":
    unittest.main()
