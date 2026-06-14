import unittest
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import colour_functions_reference as ref
from effect_backends import colour_functions_adapter


def _encode_srgb(linear):
    out = np.asarray(linear, dtype=np.float32)
    encoded = np.empty_like(out, dtype=np.float32)
    low = out <= 0.0031308
    encoded[low] = out[low] * 12.92
    encoded[~low] = 1.055 * np.power(out[~low], 1.0 / 2.4) - 0.055
    return encoded


class DisplayColorTransformMathTest(unittest.TestCase):
    def test_formal_linear_transform_matches_cached_basis_path(self):
        rng = np.random.default_rng(17)
        img = rng.normal(0.2, 0.45, (64, 32, 3)).astype(np.float32)
        img[:8] *= 4.0
        img[8:12] -= 0.4

        src_space = "ProPhoto RGB"
        dst_space = "sRGB"
        cat = "Bradford"

        formal = ref.RGB_to_RGB(
            img,
            src_space,
            dst_space,
            cat,
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        ).astype(np.float32)
        formal = ref.compress_negative_display_gamut(formal)
        formal = _encode_srgb(formal)

        basis = ref.RGB_to_RGB(
            np.eye(3, dtype=np.float32),
            src_space,
            dst_space,
            cat,
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        ).astype(np.float32)
        fast = (img.reshape(-1, 3) @ basis).reshape(img.shape)
        fast = ref.compress_negative_display_gamut(fast)
        fast = _encode_srgb(fast)

        np.testing.assert_allclose(formal, fast, rtol=1e-5, atol=1e-5)

    def test_canonical_display_transform_matches_explicit_steps(self):
        rng = np.random.default_rng(23)
        img = rng.normal(0.2, 0.45, (40, 24, 3)).astype(np.float32)
        img[:4] *= 3.0
        img[4:8] -= 0.35

        explicit = ref.RGB_to_RGB(
            img,
            "ProPhoto RGB",
            "sRGB",
            "CAT16",
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        )
        explicit = ref.compress_negative_display_gamut(explicit).astype(np.float32)
        explicit = ref.encode_display_output(explicit, "sRGB")

        canonical = colour_functions_adapter.display_color_transform(
            img,
            "ProPhoto RGB",
            "sRGB",
            "CAT16",
        )

        np.testing.assert_allclose(canonical, explicit, rtol=1e-5, atol=1e-5)

    def test_display_transform_basis_apply_matches_canonical_transform(self):
        rng = np.random.default_rng(29)
        img = rng.random((32, 16, 3), dtype=np.float32) * 2.0 - 0.2

        basis = colour_functions_adapter.display_color_transform_basis("ProPhoto RGB", "sRGB", "CAT16")
        applied = colour_functions_adapter.apply_display_color_transform(img, basis, "sRGB")
        canonical = colour_functions_adapter.display_color_transform(img, "ProPhoto RGB", "sRGB", "CAT16")

        np.testing.assert_allclose(applied, canonical, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
