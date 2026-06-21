import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import exiftool_safe


class ExiftoolSafeAliasesTest(unittest.TestCase):
    def test_embedded_other_image_alias_prefers_grouped_preview_value(self):
        row = {
            "SourceFile": "/tmp/sample.raf",
            "All:OtherImage": "base64:other-preview",
            "IFD1:OtherImage": "base64:other-thumbnail",
        }

        exiftool_safe._add_ungrouped_aliases(row)

        self.assertEqual(row["OtherImage"], "base64:other-preview")

    def test_thumbnail_tiff_alias_accepts_jfxx_group(self):
        row = {
            "SourceFile": "/tmp/sample.jpg",
            "JFXX:ThumbnailTIFF": "base64:jfxx-thumbnail",
        }

        exiftool_safe._add_ungrouped_aliases(row)

        self.assertEqual(row["ThumbnailTIFF"], "base64:jfxx-thumbnail")


if __name__ == "__main__":
    unittest.main()
