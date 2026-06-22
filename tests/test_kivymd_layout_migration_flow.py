import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SEARCH_FILES = [
    PROJECT_ROOT / "main.py",
    PROJECT_ROOT / "main.kv",
    PROJECT_ROOT / "history.py",
    PROJECT_ROOT / "async_worker.py",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "scripts" / "build_macos_app_pyinstaller.py",
    PROJECT_ROOT / "utils" / "kvutils.py",
    *sorted((PROJECT_ROOT / "widgets").glob("*.py")),
    *sorted((PROJECT_ROOT / "widgets").glob("*.kv")),
]


class KivyMDLayoutMigrationFlowTest(unittest.TestCase):
    def test_box_grid_label_components_do_not_use_kivymd(self):
        forbidden = (
            "MDBoxLayout",
            "MDGridLayout",
            "MDLabel",
            "MDScrollView",
            "MDScreen",
            "MDCard",
            "MDExpansionPanel",
            "MDOneLineListItem",
            "MDSlider",
            "MDApp",
            "theme_cls",
            "md_icons",
            "from kivymd",
            "import kivymd",
            "kivymd.",
            'font_name: "Icons"',
            "from kivymd.uix.boxlayout",
            "from kivymd.uix.gridlayout",
            "from kivymd.uix.label",
            "from kivymd.uix.scrollview",
            "from kivymd.uix.screen",
            "from kivymd.uix.card",
            "from kivymd.uix.expansionpanel",
            "from kivymd.uix.list",
            "from kivymd.uix.slider",
        )

        hits = []
        for path in SEARCH_FILES:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(PROJECT_ROOT)}: {token}")

        self.assertEqual([], hits)

    def test_tabs_use_stable_panel_and_ref_height_for_resize_layout(self):
        source = (PROJECT_ROOT / "main.kv").read_text(encoding="utf-8")
        stable_source = (PROJECT_ROOT / "widgets" / "stable_tabbed_panel.py").read_text(encoding="utf-8")
        icon_tab_rule = source.split("<IconTabbedPanelItem>:", 1)[1].split("<TextTabbedPanelItem>:", 1)[0]
        text_tab_rule = source.split("<TextTabbedPanelItem>:", 1)[1].split("<CurveTabbedPanel@StableTabbedPanel>:", 1)[0]
        main_tabs_block = source.split("StableTabbedPanel:\n                    id: effects", 1)[1].split("IconTabbedPanelItem:", 1)[0]

        self.assertIn("#:import kvutils utils.kvutils", source)
        self.assertIn("#:import StableTabbedPanel widgets.stable_tabbed_panel.StableTabbedPanel", source)
        self.assertIn("strip.row_force_default = True", stable_source)
        self.assertIn("tab.size_hint_y = None", stable_source)
        self.assertIn("tab.height = tab_height", stable_source)
        self.assertIn("icon_scale = NumericProperty(0.7)", stable_source)
        self.assertIn("font_scale = NumericProperty(0.62)", stable_source)
        self.assertIn("icon_scale: 0.82 if self.text else 0.7", icon_tab_rule)
        self.assertIn("size: self.height * self.icon_scale, self.height * self.icon_scale", icon_tab_rule)
        self.assertIn("source: iconutils.variant_source(root.icon_source, self.height * self.icon_scale)", icon_tab_rule)
        self.assertNotIn("max(kvutils.dpi_scale_height(16), self.height * 0.7)", icon_tab_rule)
        self.assertNotIn("text_size: self.size", text_tab_rule)
        self.assertIn("font_size: max(1, self.height * self.font_scale)", text_tab_rule)
        self.assertIn("valign: 'middle'", text_tab_rule)
        self.assertIn("ref_tab_height: 28", main_tabs_block)
        self.assertIn("tab_height: kvutils.dpi_scale_height(self.ref_tab_height)", main_tabs_block)
        self.assertNotIn("self.height * 0.05", main_tabs_block)

    def test_mdcard_replaced_with_plain_card(self):
        viewer_source = (PROJECT_ROOT / "widgets" / "viewer.py").read_text(encoding="utf-8")
        color_picker_source = (PROJECT_ROOT / "widgets" / "color_picker.py").read_text(encoding="utf-8")
        plain_card_source = (PROJECT_ROOT / "widgets" / "plain_card.py").read_text(encoding="utf-8")

        self.assertIn("class PlainCard(BoxLayout):", plain_card_source)
        self.assertIn("bg_color = ListProperty", plain_card_source)
        self.assertIn("shadow_color = ListProperty", plain_card_source)
        self.assertIn("self._shadow_rect = RoundedRectangle", plain_card_source)
        self.assertIn("RoundedRectangle", plain_card_source)
        self.assertIn("class ThumbnailCard(RecycleDataViewBehavior, PlainCard):", viewer_source)
        self.assertIn("self.bg_color = [0.1, 0.1, 0.1, 1]", viewer_source)
        self.assertIn("self.shadow_color = [0, 0, 0, 0.5]", viewer_source)
        self.assertIn("self.bg_color = [0.32, 0.32, 0.32, 1] if value else [0.1, 0.1, 0.1, 1]", viewer_source)
        self.assertIn("class CWColorPreview(PlainCard):", color_picker_source)
        self.assertIn("class CWColorPicker(PlainCard):", color_picker_source)
        self.assertNotIn("md_bg_color", viewer_source)
        self.assertNotIn("elevation", viewer_source)

    def test_expansion_and_list_items_do_not_use_kivymd(self):
        main_source = (PROJECT_ROOT / "main.kv").read_text(encoding="utf-8")
        mask2_source = (PROJECT_ROOT / "widgets" / "mask2_content.py").read_text(encoding="utf-8")
        history_source = (PROJECT_ROOT / "widgets" / "history_content.py").read_text(encoding="utf-8")
        mask2_kv_source = (PROJECT_ROOT / "widgets" / "mask2_content.kv").read_text(encoding="utf-8")

        self.assertIn("class Mask2CustomHeader(KVBoxLayout):", mask2_source)
        self.assertIn("<SelectableListItem@Button>:", main_source)
        self.assertNotIn("OneLineListItem", main_source)
        self.assertNotIn("MDExpansionPanel", mask2_source)
        self.assertNotIn("MDExpansionPanelOneLine", mask2_source)
        self.assertNotIn("MDOneLineListItem", mask2_source)
        self.assertNotIn("MDOneLineListItem", history_source)
        self.assertNotIn("MDOneLineListItem", mask2_kv_source)

    def test_exposure_toggles_use_local_image_icons(self):
        main_source = (PROJECT_ROOT / "main.kv").read_text(encoding="utf-8")
        expected_assets = (
            PROJECT_ROOT / "assets" / "ExposureUnderArrow_16.png",
            PROJECT_ROOT / "assets" / "ExposureUnderArrow_32.png",
            PROJECT_ROOT / "assets" / "ExposureUnderArrow_64.png",
            PROJECT_ROOT / "assets" / "ExposureOverArrow_16.png",
            PROJECT_ROOT / "assets" / "ExposureOverArrow_32.png",
            PROJECT_ROOT / "assets" / "ExposureOverArrow_64.png",
        )

        self.assertIn("icon_source: 'assets/ExposureUnderArrow.png'", main_source)
        self.assertIn("icon_source: 'assets/ExposureOverArrow.png'", main_source)
        self.assertIn("size: min(self.width, self.height) * 0.9, min(self.width, self.height) * 0.9", main_source)
        self.assertIn("source: iconutils.variant_source(root.icon_source, min(self.width, self.height))", main_source)
        self.assertNotIn("kivymd.icon_definitions", main_source)
        self.assertTrue(all(path.exists() for path in expected_assets))


if __name__ == "__main__":
    unittest.main()
