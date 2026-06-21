import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores import core
from effect_backends import film_grain_adapter

try:
    from effect_backends import _film_grain_cpu
except Exception:  # pragma: no cover - depends on local build state.
    _film_grain_cpu = None


class FilmGrainV2Test(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = film_grain_adapter.backend_status()

        self.assertEqual(status.effect, "film_grain")
        self.assertIn(
            status.backend,
            {
                "effect_backends._film_grain_cpu",
                "effect_backends.film_grain_reference",
            },
        )

    def test_amount_zero_is_noop(self):
        image = np.full((24, 32, 3), 0.5, dtype=np.float32)

        result = film_grain_adapter.apply_film_grain(image, amount=0)

        np.testing.assert_array_equal(result, image)

    def test_same_seed_is_deterministic(self):
        image = np.linspace(0.1, 0.9, 40 * 48 * 3, dtype=np.float32).reshape(40, 48, 3)

        first = film_grain_adapter.apply_film_grain(
            image,
            amount=50,
            grain_size=2.2,
            roughness=45,
            shadow=60,
            highlight=30,
            color=10,
            seed=123,
        )
        second = film_grain_adapter.apply_film_grain(
            image,
            amount=50,
            grain_size=2.2,
            roughness=45,
            shadow=60,
            highlight=30,
            color=10,
            seed=123,
        )

        np.testing.assert_array_equal(first, second)

    def test_different_seed_changes_grain_pattern(self):
        image = np.full((48, 48, 3), 0.5, dtype=np.float32)

        first = film_grain_adapter.apply_film_grain(image, amount=60, grain_size=1.8, color=0, seed=1)
        second = film_grain_adapter.apply_film_grain(image, amount=60, grain_size=1.8, color=0, seed=2)

        self.assertGreater(float(np.mean(np.abs(first - second))), 1e-4)

    def test_color_zero_adds_monochrome_grain(self):
        image = np.full((32, 36, 3), 0.5, dtype=np.float32)

        result = film_grain_adapter.apply_film_grain(image, amount=70, grain_size=1.6, color=0, seed=42)
        delta = result - image

        np.testing.assert_allclose(delta[..., 0], delta[..., 1], rtol=0, atol=1e-7)
        np.testing.assert_allclose(delta[..., 1], delta[..., 2], rtol=0, atol=1e-7)

    def test_shadow_and_highlight_controls_affect_luma_regions(self):
        row = np.linspace(0.05, 0.95, 96, dtype=np.float32)
        image = np.repeat(row[np.newaxis, :, np.newaxis], 64, axis=0)
        image = np.repeat(image, 3, axis=2)

        result = film_grain_adapter.apply_film_grain(
            image,
            amount=80,
            grain_size=1.4,
            roughness=50,
            shadow=100,
            highlight=0,
            color=0,
            seed=9,
        )
        delta = result[..., 0] - image[..., 0]
        shadow_std = float(np.std(delta[:, :24]))
        highlight_std = float(np.std(delta[:, -24:]))

        self.assertGreater(shadow_std, highlight_std * 1.4)

    def test_core_no_longer_exposes_film_grain_wrapper(self):
        self.assertFalse(hasattr(core, "apply_film_grain"))

    def test_native_backend_preserves_extra_channels_when_available(self):
        if _film_grain_cpu is None:
            self.skipTest("native film grain backend is not built")

        image = np.full((20, 24, 4), 0.5, dtype=np.float32)
        image[..., 3] = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)

        result = _film_grain_cpu.apply_film_grain(
            image,
            amount=65,
            grain_size=2.0,
            roughness=50,
            shadow=60,
            highlight=30,
            color=20,
            seed=11,
        )

        np.testing.assert_array_equal(result[..., 3], image[..., 3])


if __name__ == "__main__":
    unittest.main()
