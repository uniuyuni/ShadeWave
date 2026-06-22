import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import params
from effects import LensModifierEffect


class LensfunDefaultPolicyTest(unittest.TestCase):
    def test_missing_lensfun_user_uses_libraw_tca_friendly_default(self):
        self.assertEqual(params.get_lensfun_user_tuple({}), (True, False, True))

    def test_default_lensfun_user_is_not_persisted(self):
        param = {params.LENSFUN_USER_KEY: (True, False, True)}

        self.assertFalse(params.should_persist_lensfun_in_pmck(param))

        params.collapse_default_lensfun_user(param)
        self.assertNotIn(params.LENSFUN_USER_KEY, param)

    def test_subpixel_enabled_is_persisted(self):
        param = {params.LENSFUN_USER_KEY: (True, True, True)}

        self.assertTrue(params.should_persist_lensfun_in_pmck(param))

        param2 = {params.LENSFUN_USER_KEY: (True, True, True)}
        params._strip_default_lensfun_from_pmck_primary_param(param2)
        self.assertEqual(param2[params.LENSFUN_USER_KEY], (True, True, True))

    def test_lens_modifier_effect_default_matches_param_default(self):
        effect = LensModifierEffect()

        self.assertEqual(
            effect.get_param_dict({})[params.LENSFUN_USER_KEY],
            params.DEFAULT_LENSFUN_USER,
        )

    def test_lens_modifier_switch_uses_head_label_active_state(self):
        effect = LensModifierEffect()
        widget = SimpleNamespace(ids={
            "switch_lens_modifier": SimpleNamespace(active=False, enabled=True),
            "checkbox_color_modification": SimpleNamespace(active=True),
            "checkbox_subpixel_distortion": SimpleNamespace(active=False),
            "checkbox_geometry_distortion": SimpleNamespace(active=True),
        })
        param = {}

        effect.set2param(param, widget)

        self.assertFalse(param["switch_lens_modifier"])

        param["switch_lens_modifier"] = True
        effect.set2widget(widget, param)

        self.assertTrue(widget.ids["switch_lens_modifier"].active)

    def test_lens_modifier_stays_sync_and_does_not_use_async_worker(self):
        effect = LensModifierEffect()
        image = np.ones((2, 2, 3), dtype=np.float32)
        expected = np.zeros_like(image)
        param = {
            "switch_lens_modifier": True,
            "lens_modifier": True,
            "original_img_size": (2, 2),
            "exif_data": {},
        }
        efconfig = SimpleNamespace(
            loading_flag=0,
            processor=object(),
            upstream_hash="u",
        )

        with (
            patch.object(effect, "try_async_execution", side_effect=AssertionError("LensModifier must stay sync")),
            patch("effects.core.setup_lensfun", return_value=object()),
            patch("effects.core.get_lensfun_capability", return_value=(True, False, True)),
            patch("effects.core.modify_lensfun", return_value=(expected, True, False, True)),
        ):
            actual = effect.make_diff(image, param, efconfig)

        self.assertIs(actual, expected)

    def test_lens_modifier_off_clears_capability_and_notifies_once(self):
        calls = []
        effect = LensModifierEffect(lens_modifier_callback=lambda: calls.append("callback"))
        param = {
            "switch_lens_modifier": False,
            "lens_modifier": True,
        }
        params.set_lensfun_capability(param, (True, False, True))
        efconfig = SimpleNamespace(loading_flag=0, processor=None)

        self.assertIsNone(effect.make_diff(np.ones((1, 1, 3), dtype=np.float32), param, efconfig))
        self.assertEqual(["callback"], calls)


if __name__ == "__main__":
    unittest.main()
