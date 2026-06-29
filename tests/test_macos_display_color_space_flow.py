import ast
import pathlib
import types
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MACOS_PATH = PROJECT_ROOT / "macos.py"


def _load_functions(*names):
    source = MACOS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node
    missing = [name for name in names if name not in found]
    if missing:
        raise AssertionError(f"missing functions: {missing}")
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, str(MACOS_PATH), "exec"), ns)
    return ns


class FakeNSString:
    def __init__(self, text):
        self._text = text

    def UTF8String(self):
        return self._text.encode("utf-8")


class FakeColorSpace:
    def __init__(self, localized_name, icc_data=None):
        self._localized_name = localized_name
        self._icc_data = icc_data

    def localizedName(self):
        return self._localized_name

    def ICCProfileData(self):
        return self._icc_data


class FakeScreen:
    def __init__(self, localized_name=None, device_name=None, icc_data=None, can_p3=False, can_srgb=True):
        self._localized_name = localized_name
        self._device_name = device_name
        self._icc_data = icc_data
        self._can_p3 = can_p3
        self._can_srgb = can_srgb

    def colorSpace(self):
        if self._localized_name is None and self._icc_data is None:
            return None
        return FakeColorSpace(self._localized_name, self._icc_data)

    def deviceDescription(self):
        if self._device_name is None:
            return {}
        return {"NSDeviceColorSpaceName": self._device_name}

    def canRepresentDisplayGamut_(self, gamut):
        if gamut == 2:
            return self._can_p3
        if gamut == 1:
            return self._can_srgb
        return False


class FakeWindow:
    def __init__(self, screen):
        self._screen = screen

    def screen(self):
        return self._screen


class FakeNSScreen:
    main_screen = None

    @classmethod
    def mainScreen(cls):
        return cls.main_screen


def _fixed(value):
    return int(round(value * 65536)).to_bytes(4, "big", signed=True)


def _xyz_tag(xy):
    x, y = xy
    y_val = 1.0
    x_val = x * y_val / y
    z_val = (1.0 - x - y) * y_val / y
    return b"XYZ " + b"\0\0\0\0" + _fixed(x_val) + _fixed(y_val) + _fixed(z_val)


def _icc_with_primaries(red, green, blue):
    tags = [
        (b"rXYZ", _xyz_tag(red)),
        (b"gXYZ", _xyz_tag(green)),
        (b"bXYZ", _xyz_tag(blue)),
    ]
    header = bytearray(128)
    table_size = 4 + len(tags) * 12
    offset = 128 + table_size
    table = bytearray(len(tags).to_bytes(4, "big"))
    data = bytearray()
    for sig, payload in tags:
        table.extend(sig)
        table.extend(offset.to_bytes(4, "big"))
        table.extend(len(payload).to_bytes(4, "big"))
        data.extend(payload)
        offset += len(payload)
    return bytes(header + table + data)


DISPLAY_P3_ICC = _icc_with_primaries(
    (0.6820, 0.3196),
    (0.2845, 0.6746),
    (0.1559, 0.0660),
)

ADOBE_RGB_ICC = _icc_with_primaries(
    (0.6484, 0.3309),
    (0.2310, 0.7040),
    (0.1559, 0.0660),
)


class MacOSDisplayColorSpaceFlowTest(unittest.TestCase):
    def _namespace(self):
        ns = _load_functions(
            "_objc_text",
            "_normalize_color_space_name",
            "_nsdata_bytes",
            "_icc_tag_data",
            "_s15_fixed16_to_float",
            "_icc_xyz_tag_xy",
            "_icc_rgb_primary_xy",
            "_primary_distance",
            "_classify_rgb_primary_gamut",
            "_screen_icc_profile_data",
            "_screen_display_gamut_from_capability",
            "_screen_display_color_gamut_name",
            "_known_display_color_gamut_name",
            "get_screen_color_space_name",
            "_get_app_window_screen",
            "get_app_window_screen_color_space_name",
        )
        ns["NSScreen"] = FakeNSScreen
        ns["AppKit"] = types.SimpleNamespace(NSDisplayGamutSRGB=1, NSDisplayGamutP3=2)
        ns["define"] = types.SimpleNamespace(APPNAME="Shade Wave")
        return ns

    def test_known_display_profile_names_are_normalized_for_display_transform(self):
        ns = self._namespace()
        normalize = ns["_normalize_color_space_name"]

        self.assertEqual(normalize("sRGB IEC61966-2.1"), "sRGB")
        self.assertEqual(normalize("AdobeRGB"), "Adobe RGB (1998)")
        self.assertEqual(normalize("Display P3"), "Display P3")
        self.assertEqual(normalize("ITU-R BT.2020"), "Rec.2020")
        self.assertEqual(normalize("Custom Calibrated Display"), "Custom Calibrated Display")

    def test_app_window_screen_color_space_uses_current_window_screen(self):
        ns = self._namespace()
        app_screen = FakeScreen(FakeNSString("Display P3"), icc_data=DISPLAY_P3_ICC)
        FakeNSScreen.main_screen = FakeScreen("sRGB IEC61966-2.1")
        ns["get_window"] = lambda app_name: FakeWindow(app_screen)

        self.assertEqual(ns["get_app_window_screen_color_space_name"](), "Display P3")

    def test_profile_display_name_is_classified_from_icc_primaries(self):
        ns = self._namespace()
        screen = FakeScreen(localized_name="Built-in Retina Display", icc_data=DISPLAY_P3_ICC)

        self.assertEqual(ns["get_screen_color_space_name"](screen), "Display P3")

    def test_adobe_rgb_profile_is_classified_from_icc_primaries(self):
        ns = self._namespace()
        screen = FakeScreen(localized_name="Calibrated Photo Display", icc_data=ADOBE_RGB_ICC)

        self.assertEqual(ns["get_screen_color_space_name"](screen), "Adobe RGB (1998)")

    def test_screen_color_space_falls_back_to_capability_then_device_description(self):
        ns = self._namespace()
        p3_screen = FakeScreen(localized_name="Studio Display", can_p3=True)
        device_screen = FakeScreen(localized_name=None, device_name="AdobeRGB")

        self.assertEqual(ns["get_screen_color_space_name"](p3_screen), "Display P3")
        self.assertEqual(ns["get_screen_color_space_name"](device_screen), "Adobe RGB (1998)")

    def test_app_window_screen_color_space_falls_back_to_main_screen(self):
        ns = self._namespace()
        FakeNSScreen.main_screen = FakeScreen("sRGB IEC61966-2.1")
        ns["get_window"] = lambda app_name: None

        self.assertEqual(ns["get_app_window_screen_color_space_name"](), "sRGB")


if __name__ == "__main__":
    unittest.main()
