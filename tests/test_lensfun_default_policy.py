import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


if __name__ == "__main__":
    unittest.main()
