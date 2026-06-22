from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class LicenseNoticeFlowTest(unittest.TestCase):
    def test_project_license_and_notices_exist(self):
        license_text = read_text("LICENSE")
        notices = read_text("THIRD_PARTY_NOTICES.md")
        app_distribution_notices = read_text("APP_NOTICES.md")

        self.assertIn("GNU GENERAL PUBLIC LICENSE", license_text)
        self.assertIn("Version 3", license_text)
        for required in ("RawTherapee", "Adobe DNG SDK", "Colour Science", "LibRaw", "Lensfun", "lensfunpy"):
            self.assertIn(required, notices)
        self.assertIn("GPL-3.0-or-later", app_distribution_notices)
        self.assertIn("THIRD_PARTY_NOTICES.md", app_distribution_notices)
        self.assertIn("licenses/DNG_SDK_LICENSE.txt", app_distribution_notices)
        self.assertIn("lensfunpy", app_distribution_notices)

    def test_lensfun_notice_declares_wrapper_and_native_library_terms(self):
        notices = read_text("THIRD_PARTY_NOTICES.md")
        readme = read_text("README.md")
        readme_ja = read_text("README_JA.md")

        self.assertIn("lensfunpy", notices)
        self.assertIn("MIT License", notices)
        self.assertIn("Lensfun", notices)
        self.assertIn("LGPL-3.0-or-later", notices)
        self.assertIn("https://github.com/letmaik/lensfunpy", notices)
        self.assertIn("https://github.com/lensfun/lensfun", notices)
        self.assertIn("Lensfun/lensfunpy", readme)
        self.assertIn("Lensfun/lensfunpy", readme_ja)

    def test_libraw_enhanced_declares_gpl_3_or_later(self):
        pyproject = read_text("external/libraw_enhanced/pyproject.toml")
        readme = read_text("external/libraw_enhanced/README.md")

        self.assertIn('license = {text = "GPL-3.0-or-later"}', pyproject)
        self.assertIn("GPL-3.0-or-later", readme)

    def test_ported_sources_keep_license_attribution(self):
        ported_sources = [
            "external/libraw_enhanced/core/cpu_accelerator.cpp",
            "external/libraw_enhanced/core/metal/demosaic_bayer_amaze.metal",
            "external/libraw_enhanced/core/metal/demosaic_xtrans_1pass.metal",
            "external/libraw_enhanced/core/metal/demosaic_xtrans_3pass.metal",
        ]

        for source in ported_sources:
            text = read_text(source)
            self.assertIn("GPL-3.0", text)
            self.assertIn("RawTherapee", text)

    def test_dng_and_colour_derived_sources_keep_attribution(self):
        dng_license = read_text("licenses/DNG_SDK_LICENSE.txt")

        self.assertIn("Adobe DNG SDK", read_text("cores/dng_temperature.py"))
        self.assertIn("DNG SDK License Agreement", dng_license)
        self.assertIn("BSD-3-Clause", read_text("effect_backends/colour_functions_reference.py"))
        self.assertIn("BSD-3-Clause", read_text("effect_backends/colour_functions_cpu.c"))
        self.assertIn("BSD-3-Clause", read_text("effect_backends/colour_functions_capi.h"))
        self.assertIn("BSD-3-Clause", read_text("effect_backends/colour_functions_pybind.cpp"))
