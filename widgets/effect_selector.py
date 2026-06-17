import os

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder as KVBuilder
from kivy.properties import (
    BooleanProperty as KVBooleanProperty,
    NumericProperty as KVNumericProperty,
    StringProperty as KVStringProperty,
)
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup as KVPopup
from kivy.uix.widget import Widget as KVWidget
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel

from utils import dialogutils
from utils import kvutils
from widgets.modern_checkbox import ModernCheckBox
from widgets.switch_reset_map import (
    BASE_SWITCH_TARGETS,
    HLS_COLORS,
    flatten_targets_to_pipeline_ids,
)


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
KVBuilder.load_file(os.path.join(CUR_DIR, "effect_selector.kv"))


_SWITCH_LABELS = {
    "switch_white_balance": "White balance",
    "switch_exposure_contrast": "Exposure & Contrast",
    "switch_tone": "Tone",
    "switch_level": "Level",
    "switch_precence": "Presence",
    "switch_saturation": "Saturation",
    "switch_color_mixer": "Color Mixer",
    "switch_unsharp_mask": "Sharpening",
    "switch_vignette": "Vignette",
    "switch_lens_modifier": "Lens Modifier",
    "switch_tone_curves": "Tone Curves",
    "switch_color_gradings": "Color Gradings",
    "switch_hue_vs_hue": "Hue vs Hue",
    "switch_hue_vs_lum": "Hue vs Lum",
    "switch_hue_vs_sat": "Hue vs Sat",
    "switch_lum_vs_lum": "Lum vs Lum",
    "switch_lum_vs_sat": "Lum vs Sat",
    "switch_sat_vs_lum": "Sat vs Lum",
    "switch_sat_vs_sat": "Sat vs Sat",
    "switch_ai_noise_reduction": "AI noise reduction",
    "switch_light_noise_reduction": "Light NR",
    "switch_details": "Details",
    "switch_lut": "LUT",
    "switch_solid_color": "Solid color",
    "switch_global": "Global Color",
    "switch_fringe_removal": "Fringe Removal",
    "switch_film_simulation": "Film Simulation",
    "switch_lens_simulator": "Lens Simulator",
    "switch_filters": "Filters",
    "switch_orton_effect": "Orton",
    "switch_glow_effect": "Glow",
    "switch_grain": "Grain",
    "switch_cross_filter": "Cross filter",
}

_SECTION_BG = (0.05, 0.056, 0.082, 1)
_HEADER_BG = (0.15, 0.162, 0.2, 1)
_TEXT_HEADER = (0.97, 0.975, 0.99, 1)
_ACCENT = (0.45, 0.72, 0.98, 1)

_REF_POPUP_W = 860
_REF_POPUP_H = 560
_REF_ROW_H = 20
_REF_CHECKBOX = 22
_REF_SECTION_CHECKBOX = 24
_REF_SECTION_HEADER_H = 34
_REF_SECTION_PAD = 8
_REF_SECTION_PAD_LEFT = 20
_REF_SECTION_SPACING = 8
_REF_GRID_ROW_SPACING = 2
_REF_FOOTER_H = 42
_REF_FOOTER_BTN_H = 24
_REF_TOGGLE_BTN_W = 86
_REF_CANCEL_BTN_W = 68
_REF_OK_BTN_W = 56
_REF_TITLE_FONT = 13
_REF_ROW_FONT = 12
_REF_BTN_FONT = 11

_VS_PARENT_SWITCH = "switch_color_curves"
_VS_DETAIL_SWITCH_KEYS = (
    "switch_hue_vs_hue",
    "switch_hue_vs_lum",
    "switch_hue_vs_sat",
    "switch_lum_vs_lum",
    "switch_lum_vs_sat",
    "switch_sat_vs_lum",
    "switch_sat_vs_sat",
)
_VS_CHILD_SET = frozenset(_VS_DETAIL_SWITCH_KEYS)

_SWITCH_SELECTION_SECTIONS = (
    (
        "Basic",
        (
            "switch_white_balance",
            "switch_exposure_contrast",
            "switch_tone",
            "switch_level",
            "switch_precence",
            "switch_saturation",
            "switch_color_mixer",
            "switch_unsharp_mask",
            "switch_vignette",
            "switch_lens_modifier",
        ),
    ),
    (
        "Curves",
        (
            "switch_tone_curves",
            "switch_color_gradings",
        ),
    ),
    ("Color curves", _VS_DETAIL_SWITCH_KEYS),
    (
        "Effects",
        (
            "switch_ai_noise_reduction",
            "switch_light_noise_reduction",
            "switch_details",
            "switch_lut",
            "switch_solid_color",
            "switch_global",
            "switch_fringe_removal",
        ),
    ),
    (
        "Looks",
        (
            "switch_film_simulation",
            "switch_lens_simulator",
            "switch_filters",
            "switch_orton_effect",
            "switch_glow_effect",
            "switch_grain",
            "switch_cross_filter",
        ),
    ),
)

_ALL_IDS_IN_MAIN = frozenset(BASE_SWITCH_TARGETS) | {
    f"switch_hls_{color}" for color in HLS_COLORS
}
_ALL_SECTION_KEYS = tuple(
    key for _title, keys in _SWITCH_SELECTION_SECTIONS for key in keys
)
assert set(_ALL_SECTION_KEYS) <= _ALL_IDS_IN_MAIN
assert len(_ALL_SECTION_KEYS) == len(set(_ALL_SECTION_KEYS))


def _sw(ref):
    return kvutils.dpi_scale_width(ref)


def _sh(ref):
    return kvutils.dpi_scale_height(ref)


class EffectSelectionItem(MDBoxLayout):
    text = KVStringProperty("")
    key = KVStringProperty("")
    is_selected = KVBooleanProperty(False)
    row_height = KVNumericProperty(_REF_ROW_H)

    def on_kv_post(self, *_args):
        self._apply_layout()

    def on_row_height(self, *_args):
        self._apply_layout()

    def on_width(self, *_args):
        Clock.schedule_once(lambda _dt: self._sync_label_text_size(), 0)

    def _apply_layout(self):
        self.ref_height = self.row_height
        self.height = _sh(self.row_height)
        self.padding = _sw([7, 0, 7, 0])
        self.ref_layout_spacing = 8
        self.spacing = _sw(self.ref_layout_spacing)
        if "row_chk" in self.ids:
            self.ids.row_chk.ref_width = _REF_CHECKBOX
            self.ids.row_chk.ref_height = _REF_CHECKBOX
            self.ids.row_chk.size = (_sw(_REF_CHECKBOX), _sh(_REF_CHECKBOX))
        if "effect_lbl" in self.ids:
            self.ids.effect_lbl.font_size = _sh(_REF_ROW_FONT)
        self._sync_label_text_size()

    def _sync_label_text_size(self):
        if "effect_lbl" not in self.ids:
            return
        label = self.ids.effect_lbl
        label.text_size = (max(0, label.width), _sh(self.row_height))


class EffectSelector(KVPopup):
    def __init__(self, selected_switch_keys=None, **kwargs):
        self.register_event_type("on_cancel")
        self.register_event_type("on_decide")
        self._effect_items = []
        self._section_rows = []
        self._section_checkboxes = []
        self.last_selected_switch_keys = []
        self._initial_selected_switch_keys = set(selected_switch_keys or [])
        self._syncing_section_state = False
        super().__init__(**kwargs)
        self.opacity = 0
        dialogutils.install_ref_scaling(self, center=True, on_rescale=self._on_dialog_rescale)
        Clock.schedule_once(lambda _dt: self._build_sections(), 0)
        self.bind(on_open=lambda *_args: Clock.schedule_once(self._show_after_layout, 0))

    def _on_dialog_rescale(self):
        Clock.schedule_once(lambda _dt: self._rebuild_preserving_state(), 0)

    def _rebuild_preserving_state(self):
        if self._effect_items:
            selected = set(self.get_selected_switch_keys())
            self._initial_selected_switch_keys = selected
        self._build_sections()

    def _build_sections(self):
        if "sections_box" not in self.ids:
            return

        box = self.ids.sections_box
        box.clear_widgets()
        box.ref_layout_spacing = _REF_SECTION_SPACING
        box.spacing = _sw(_REF_SECTION_SPACING)
        self._effect_items = []
        self._section_rows = []
        self._section_checkboxes = []

        for title, keys in _SWITCH_SELECTION_SECTIONS:
            rows = []
            section = self._create_section(title, keys, rows)
            box.add_widget(section)
            self._section_rows.append(rows)

        self._apply_initial_selection()
        self._refresh_section_states()
        self._refresh_toggle_all_text()
        self._scale_layout()
        Clock.schedule_once(lambda _dt: self._sync_sections_height(), 0)

    def _create_section(self, title, keys, rows):
        section = MDBoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=0,
            md_bg_color=_SECTION_BG,
        )

        header = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=_sh(_REF_SECTION_HEADER_H),
            padding=_sw([10, 0, 12, 0]),
            spacing=_sw(8),
            md_bg_color=_HEADER_BG,
        )
        header.ref_height = _REF_SECTION_HEADER_H
        header.ref_layout_spacing = 8
        section_checkbox = self._create_checkbox()
        header.add_widget(section_checkbox)
        header.add_widget(
            MDLabel(
                text=title,
                bold=True,
                font_size=_sh(_REF_TITLE_FONT),
                halign="left",
                valign="middle",
                theme_text_color="Custom",
                text_color=_TEXT_HEADER,
            )
        )
        section.add_widget(header)

        grid = GridLayout(
            cols=self._preferred_columns(keys),
            size_hint_y=None,
            padding=_sw(
                [
                    _REF_SECTION_PAD_LEFT,
                    _REF_SECTION_PAD,
                    _REF_SECTION_PAD,
                    _REF_SECTION_PAD,
                ]
            ),
            spacing=[_sw(_REF_SECTION_SPACING), _sw(_REF_GRID_ROW_SPACING)],
            row_default_height=_sh(_REF_ROW_H),
            row_force_default=True,
        )
        grid._effect_selector_ref_layout_spacing = (
            _REF_SECTION_SPACING,
            _REF_GRID_ROW_SPACING,
        )
        for key in keys:
            label = _SWITCH_LABELS.get(
                key, key.replace("switch_", "").replace("_", " ").title()
            )
            row = EffectSelectionItem(
                key=key,
                text=label,
                size_hint_x=1,
            )
            row.bind(is_selected=lambda *_args: self._on_item_selected())
            self._effect_items.append(row)
            rows.append(row)
            grid.add_widget(row)

        # Fill the last row so columns keep the same width and labels align.
        missing = (-len(keys)) % max(1, grid.cols)
        for _index in range(missing):
            grid.add_widget(KVWidget(opacity=0, disabled=True))

        row_count = max(1, (len(keys) + grid.cols - 1) // grid.cols)
        grid.ref_height = (
            2 * _REF_SECTION_PAD
            + row_count * _REF_ROW_H
            + max(0, row_count - 1) * _REF_GRID_ROW_SPACING
        )
        grid.height = _sh(grid.ref_height)
        section.add_widget(grid)

        section.ref_height = _REF_SECTION_HEADER_H + grid.ref_height
        section.height = _sh(section.ref_height)
        section_checkbox.bind(
            active=lambda _chk, active, rs=rows: self._on_section_toggled(
                rs, active
            )
        )
        self._section_checkboxes.append((section_checkbox, rows))
        return section

    def _create_checkbox(self):
        checkbox = ModernCheckBox(
            size_hint=(None, None),
            size=(_sw(_REF_SECTION_CHECKBOX), _sh(_REF_SECTION_CHECKBOX)),
            pos_hint={"center_y": 0.5},
            box_color=(1, 1, 1, 1),
            border_color_inactive=(0.9, 0.91, 0.94, 1),
            border_color_active=_ACCENT,
            check_color=_ACCENT,
        )
        checkbox.ref_width = _REF_SECTION_CHECKBOX
        checkbox.ref_height = _REF_SECTION_CHECKBOX
        return checkbox

    def _preferred_columns(self, keys):
        width = float(Window.width or 900)
        key_count = len(keys)
        if width >= 1080:
            return min(4, key_count)
        if width >= 860:
            return min(3, key_count)
        if width >= 640:
            return min(2, key_count)
        return 1

    def _apply_initial_selection(self):
        selected = self._initial_selected_switch_keys
        self._syncing_section_state = True
        try:
            for row in self._effect_items:
                row.is_selected = row.key in selected
        finally:
            self._syncing_section_state = False

    def _sync_sections_height(self):
        box = self.ids.sections_box
        box.height = sum(child.height for child in box.children) + max(
            0, len(box.children) - 1
        ) * box.spacing

    def _scale_layout(self, *_args):
        self.size_hint = (None, None)
        kvutils.traverse_widget(self)
        if "main_column" in self.ids:
            self.ids.main_column.padding = _sw([10, 8, 10, 8])
        self.width = min(
            _sw(_REF_POPUP_W),
            float(Window.width or _sw(_REF_POPUP_W)) * 0.92,
        )
        self.height = min(
            _sh(_REF_POPUP_H),
            float(Window.height or _sh(_REF_POPUP_H)) * 0.86,
        )
        if "sections_scroll" in self.ids:
            self.ids.sections_scroll.bar_width = _sw(6)
        if "footer_bar" in self.ids:
            self.ids.footer_bar.height = _sh(_REF_FOOTER_H)
            self.ids.footer_bar.padding = _sw([0, 0, 0, 0])
            self._scale_footer_button("btn_toggle_all", _REF_TOGGLE_BTN_W)
            self._scale_footer_button("btn_footer_cancel", _REF_CANCEL_BTN_W)
            self._scale_footer_button("btn_footer_ok", _REF_OK_BTN_W)
        for row in self._effect_items:
            row._apply_layout()
        for section in self.ids.sections_box.children:
            for child in getattr(section, "children", []):
                spacing = getattr(child, "_effect_selector_ref_layout_spacing", None)
                if spacing is not None:
                    child.spacing = [_sw(spacing[0]), _sw(spacing[1])]
        self._sync_sections_height()
        self.center = Window.center

    def _show_after_layout(self, *_args):
        self._scale_layout()
        self.center = Window.center
        Clock.schedule_once(lambda _dt: setattr(self, "opacity", 1), 0)

    def _scale_footer_button(self, widget_id, ref_width):
        button = self.ids.get(widget_id)
        if button is None:
            return
        button.size_hint = (None, None)
        button.width = _sw(ref_width)
        button.height = _sh(_REF_FOOTER_BTN_H)
        button.font_size = _sh(_REF_BTN_FONT)

    def _on_section_toggled(self, rows, active):
        if self._syncing_section_state:
            return
        for row in rows:
            if row.is_selected != active:
                row.is_selected = active
        self._refresh_toggle_all_text()

    def _on_item_selected(self):
        if self._syncing_section_state:
            return
        self._refresh_section_states()
        self._refresh_toggle_all_text()

    def _refresh_section_states(self):
        self._syncing_section_state = True
        try:
            for checkbox, rows in self._section_checkboxes:
                checkbox.active = bool(rows) and all(row.is_selected for row in rows)
        finally:
            self._syncing_section_state = False

    def _refresh_toggle_all_text(self):
        selected = bool(self._effect_items) and all(
            row.is_selected for row in self._effect_items
        )
        if "btn_toggle_all" in self.ids:
            self.ids.btn_toggle_all.text = "Deselect all" if selected else "Select all"

    def on_toggle_all(self):
        new_state = not (
            bool(self._effect_items)
            and all(row.is_selected for row in self._effect_items)
        )
        for row in self._effect_items:
            row.is_selected = new_state
        self._refresh_section_states()
        self._refresh_toggle_all_text()

    def _selected_switch_keys(self):
        selected = [row.key for row in self._effect_items if row.is_selected]
        if _VS_CHILD_SET.intersection(selected) and _VS_PARENT_SWITCH not in selected:
            selected.append(_VS_PARENT_SWITCH)
        return list(dict.fromkeys(selected))

    def get_selection(self):
        return flatten_targets_to_pipeline_ids(self._selected_switch_keys())

    def get_selected_switch_keys(self):
        return list(self._selected_switch_keys())

    def do_decide(self):
        keys = self._selected_switch_keys()
        self.last_selected_switch_keys = list(keys)
        self.dismiss()
        self.dispatch("on_decide", flatten_targets_to_pipeline_ids(keys))

    def do_cancel(self):
        self.dismiss()
        self.dispatch("on_cancel")

    def on_cancel(self):
        pass

    def on_decide(self, selection):
        pass


if __name__ == "__main__":
    from kivy.uix.anchorlayout import AnchorLayout
    from kivymd.app import MDApp
    from widgets.scaled_button import ScaledButton

    class EffectSelectorDebugApp(MDApp):
        def build(self):
            root = AnchorLayout()
            btn = ScaledButton(text="Open effect selector", size_hint=(None, None), size=(180, 32))
            btn.bind(on_release=self._open_selector)
            root.add_widget(btn)
            return root

        def on_start(self):
            Clock.schedule_once(lambda _dt: self._open_selector(), 0)

        def _open_selector(self, *_args):
            popup = EffectSelector()
            popup.bind(on_cancel=lambda *_a: print("[effect_selector debug] cancel"))

            def _on_decide(inst, flattened):
                print("[effect_selector debug] pipeline ids:", flattened)
                print(
                    "[effect_selector debug] switch keys:",
                    inst.last_selected_switch_keys,
                )

            popup.bind(on_decide=_on_decide)
            popup.open()

    EffectSelectorDebugApp().run()
