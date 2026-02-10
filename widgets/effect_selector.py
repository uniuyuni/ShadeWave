
from kivy.lang import Builder as KVBuilder
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivy.uix.popup import Popup as KVPopup
import os

# Identify the directory of this file
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
KVBuilder.load_file(os.path.join(CUR_DIR, 'effect_selector.kv'))

# Explicit mapping based on main.kv and effects.py analysis (Ordered by appearance in main.kv)
EFFECT_KEY_MAPPING = {
    # Basic Tab
    'switch_white_balance': ['color_temperature'],
    'switch_exposure': ['exposure'],
    'switch_contrast': ['contrast'],
    'switch_tone': ['tone'],
    'switch_level': ['level'],
    'switch_clarity': ['clarity'],
    'switch_texture': ['texture'],
    'switch_microcontrast': ['microcontrast'],
    'switch_dehaze': ['dehaze'],
    'switch_clahe': ['clahe'],
    'switch_saturation': ['vs_and_saturation'],
    'switch_color_mixer': ['hls'],
    'switch_unsharp_mask': ['unsharp_mask'],
    'switch_vignette': ['vignette'],
    'switch_lens_modifier': ['lens_modifier'],

    # Curves Tab
    'switch_tone_curves': ['curves'],
    'switch_color_gradings': ['curves'],
    'switch_color_curves': ['vs_and_saturation'],

    # Extra Tab
    'switch_ai_noise_reduction': ['ai_noise_reduction'],
    'switch_light_noise_reduction': ['light_noise_reduction'],
    'switch_subpixel_shift': ['subpixel_shift'],
    'switch_lut': ['lut'],
    'switch_solid_color': ['solid_color'],
    'switch_highlight_compress': ['highlight_compress'],
    'switch_fringe_removal': ['remove_chromatic_aberration'],

    # Effect Tab
    'switch_film_simulation': ['film_emulation'],
    'switch_lens_simulator': ['lens_simulator'],
    'switch_lensblur': ['lensblur_filter'],
    'switch_scratch': ['scratch'],
    'switch_frosted_glass': ['frosted_glass'],
    'switch_mosaic': ['mosaic'],
    'switch_orton_effect': ['orton'],
    'switch_glow_effect': ['glow'],
    'switch_grain': ['grain'],
    'switch_cross_filter': ['cross_filter'],

    # Looks Tab
    'switch_face': ['face'],
}


class EffectSelectionItem(MDBoxLayout):
    text = KVStringProperty()
    key = KVStringProperty()
    is_selected = KVBooleanProperty(False)

class EffectSelector(KVPopup):

    def __init__(self, **kwargs):
        self.register_event_type('on_cancel')
        self.register_event_type('on_decide')
        super().__init__(**kwargs)
        self.populate_effects()

    def populate_effects(self):
        container = self.ids.container
        container.clear_widgets()
        
        # Use simple iteration to respect the definition order (Python 3.7+ guarantees insertion order)
        for key in EFFECT_KEY_MAPPING:
            # Generate a nice label
            label_text = key.replace('switch_', '').replace('_', ' ').title()
            
            # Special cases for better readability
            if label_text.startswith("Rca"):
                label_text = "RCA"
            if label_text.startswith("Lut"):
                label_text = "LUT"
            
            item = EffectSelectionItem(key=key, text=label_text)
            container.add_widget(item)

    def on_toggle_all(self):
        container = self.ids.container
        # Check if all are selected
        all_selected = True
        for child in container.children:
            if isinstance(child, EffectSelectionItem):
                if not child.is_selected:
                    all_selected = False
                    break
        
        # If all are selected, we want to deselect all.
        # If not all are selected (some or none), we want to select all.
        new_state = not all_selected
        
        for child in container.children:
             if isinstance(child, EffectSelectionItem):
                 child.is_selected = new_state
                 
        self.ids.btn_toggle_all.text = "Deselect All" if new_state else "Select All"

    def get_selection(self):
        selection = []
        container = self.ids.container
        for child in container.children:
            if isinstance(child, EffectSelectionItem):
                if child.is_selected:
                    for effect_name in EFFECT_KEY_MAPPING[child.key]:
                        selection.append(effect_name)
        return selection

    def do_decide(self):
        selection = self.get_selection()
        self.dismiss()
        self.dispatch('on_decide', selection)

    def do_cancel(self):
        self.dismiss()
        self.dispatch('on_cancel')

    def on_cancel(self):
        pass

    def on_decide(self, selection):
        pass

