import unittest

import numpy as np

from cores import colour_functions


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

        formal = colour_functions.RGB_to_RGB(
            img,
            src_space,
            dst_space,
            cat,
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        ).astype(np.float32)
        formal = colour_functions.compress_negative_display_gamut(formal)
        formal = _encode_srgb(formal)

        basis = colour_functions.RGB_to_RGB(
            np.eye(3, dtype=np.float32),
            src_space,
            dst_space,
            cat,
            apply_cctf_decoding=False,
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        ).astype(np.float32)
        fast = (img.reshape(-1, 3) @ basis).reshape(img.shape)
        fast = colour_functions.compress_negative_display_gamut(fast)
        fast = _encode_srgb(fast)

        np.testing.assert_allclose(formal, fast, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
