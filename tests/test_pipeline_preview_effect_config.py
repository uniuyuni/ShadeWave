import os
import sys
import unittest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import effects
import pipeline


class PipelinePreviewEffectConfigTest(unittest.TestCase):
    def test_geometry_preview_render_env_controls_full_render(self):
        old_value = os.environ.get("PLATYPUS_GE_PREVIEW_RENDER")
        try:
            os.environ.pop("PLATYPUS_GE_PREVIEW_RENDER", None)
            self.assertTrue(pipeline.preview_full_render_enabled("Ge"))

            os.environ["PLATYPUS_GE_PREVIEW_RENDER"] = "full"
            self.assertTrue(pipeline.preview_full_render_enabled("Ge"))

            os.environ["PLATYPUS_GE_PREVIEW_RENDER"] = "fast"
            self.assertFalse(pipeline.preview_full_render_enabled("Ge"))
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_GE_PREVIEW_RENDER", None)
            else:
                os.environ["PLATYPUS_GE_PREVIEW_RENDER"] = old_value

    def test_geometry_preview_drain_all_env_is_enabled_by_default_but_can_be_disabled(self):
        old_value = os.environ.get("PLATYPUS_GE_PREVIEW_DRAIN_ALL")
        try:
            os.environ.pop("PLATYPUS_GE_PREVIEW_DRAIN_ALL", None)
            self.assertTrue(pipeline.preview_drain_all_enabled("Ge"))

            os.environ["PLATYPUS_GE_PREVIEW_DRAIN_ALL"] = "0"
            self.assertFalse(pipeline.preview_drain_all_enabled("Ge"))
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_GE_PREVIEW_DRAIN_ALL", None)
            else:
                os.environ["PLATYPUS_GE_PREVIEW_DRAIN_ALL"] = old_value

    def test_geometry_preview_stale_frames_are_allowed_by_default_but_can_be_disabled(self):
        old_value = os.environ.get("PLATYPUS_GE_PREVIEW_ALLOW_STALE")
        try:
            os.environ.pop("PLATYPUS_GE_PREVIEW_ALLOW_STALE", None)
            self.assertTrue(pipeline.preview_allow_stale_enabled("Ge"))

            os.environ["PLATYPUS_GE_PREVIEW_ALLOW_STALE"] = "0"
            self.assertFalse(pipeline.preview_allow_stale_enabled("Ge"))
        finally:
            if old_value is None:
                os.environ.pop("PLATYPUS_GE_PREVIEW_ALLOW_STALE", None)
            else:
                os.environ["PLATYPUS_GE_PREVIEW_ALLOW_STALE"] = old_value

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
        self.assertIn("grain_amount=0(default)", grain)
        self.assertIn("grain_color=10(default)", grain)
        self.assertIn("grain_seed=0(default)", grain)
        self.assertNotIn("grain_radius", grain)
        self.assertNotIn("grain_intensity", grain)
        self.assertIn("switch_vignette=True(default)", vignette)
        self.assertIn("vignette_radius_percent=80(default)", vignette)
        self.assertNotIn("vignette_radius=", vignette)


if __name__ == "__main__":
    unittest.main()
