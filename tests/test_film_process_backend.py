import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import film_process_adapter, film_process_reference
from effects import FilmSimulationEffect


_COMMON = dict(
    latitude=65.0,
    contrast=55.0,
    color_bias=30.0,
    color_drift=40.0,
    dye_purity=70.0,
    layer_crosstalk=35.0,
    aging=20.0,
)


class FilmProcessBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = film_process_adapter.backend_status()

        self.assertEqual(status.effect, "film_process")
        self.assertIn(
            status.backend,
            {"effect_backends._film_process_cpu", "effect_backends.film_process_reference"},
        )

    @unittest.skipUnless(film_process_adapter.native_available(), "film_process native backend is not built")
    def test_native_matches_reference_all_modes_hdr(self):
        rng = np.random.default_rng(20240517)
        # HDR ramp: values well above 1.0 to exercise highlight headroom paths.
        image = (rng.random((40, 56, 3), dtype=np.float32) * np.float32(10.0)).astype(np.float32)

        for mode in ("Negative", "Slide", "B&W"):
            for halation in (0.0, 60.0):
                with self.subTest(mode=mode, halation=halation):
                    expected = film_process_reference.apply_film_process(
                        image, mode=mode, halation=halation, **_COMMON
                    )
                    actual = film_process_adapter.apply_film_process(
                        image, mode=mode, halation=halation, **_COMMON
                    )
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertEqual(actual.shape, image.shape)
                    self.assertTrue(np.all(np.isfinite(actual)))
                    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)

    @unittest.skipUnless(film_process_adapter.native_available(), "film_process native backend is not built")
    def test_native_preserves_hdr_highlights(self):
        # A bright HDR highlight must survive (>1) rather than be tone-mapped to ~1.
        image = np.full((8, 8, 3), 8.0, dtype=np.float32)
        out = film_process_adapter.apply_film_process(image, mode="Slide", **_COMMON)
        self.assertGreater(float(out.max()), 1.0)

    def test_film_effect_dispatches_to_adapter(self):
        image = np.full((6, 8, 3), 0.5, dtype=np.float32)
        effect = FilmSimulationEffect()
        param = {
            "switch_film_simulation": True,
            "film_mode": "Negative",
            "film_latitude": 55,
            "film_contrast": 50,
            "film_color_bias": 0,
            "film_color_drift": 0,
            "film_dye_purity": 75,
            "film_layer_crosstalk": 30,
            "film_halation": 0,
            "film_aging": 0,
            "film_intensity": 100,
        }
        efconfig = SimpleNamespace()

        sentinel = np.full_like(image, 0.25)
        with mock.patch.object(
            film_process_adapter, "apply_film_process", return_value=sentinel
        ) as patched:
            effect.hash = None
            actual = effect.make_diff(image, param, efconfig)

        patched.assert_called_once()
        # intensity == 100 -> blend is fully the adapter output.
        np.testing.assert_allclose(actual, sentinel, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
