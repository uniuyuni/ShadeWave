
import numpy as np
import cv2
import copy
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty

from lens_ghost import create_ghost
from param_slider import ParamSlider

class GhostEditor(BoxLayout):
    image_widget = ObjectProperty(None)
    param_container = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.light_source = [(300, 200)]
        self.base_image = self.create_base_image()
        self.processed_image = None
        Clock.schedule_once(self.init_ui, 0.1)
        self.update_event = None

    def create_base_image(self, size=(700, 500)):
        img = np.zeros((size[1], size[0], 3), dtype=np.float32)
        for x, y in self.light_source:
            cv2.circle(img, (x, y), 15, (1.0, 1.0, 1.0), -1)
        return img

    def init_ui(self, dt):
        params = [
            # (name, min, max, default, step, is_float)
            ('global_intensity', 0.0, 1.0, 0.5, 0.01, True),
            ('base_radius', 1, 500, 150, 1, False),
            ('num_components', 1, 20, 1, 1, False),
            ('component_spread_factor', 0.0, 1.0, 0.3, 0.01, True),
            ('blur_sigma', 0.0, 50.0, 5.0, 0.1, True),
            ('chromatic_aberration_strength', 0.0, 5.0, 1.0, 0.1, True),
            ('ghost_ring_thickness', 0.01, 1.0, 0.4, 0.01, True),
            ('radial_deformation_strength', 0.0, 1.0, 0.6, 0.01, True),
            ('max_eccentricity', 0.0, 1.0, 0.7, 0.01, True),
            ('max_offset_ratio_x', -5.0, 5.0, 0, 0.1, True),
            ('max_offset_ratio_y', -5.0, 5.0, 0, 0.1, True),
            ('ghost_decay_rate', 0.1, 5.0, 3.0, 0.1, True),
            ('perspective_distortion', 0.0, 1.0, 0.4, 0.01, True),
            ('curvature_strength', 0.0, 1.0, 0.2, 0.01, True),
            ('spherical_aberration_strength', 0.0, 1.0, 0.1, 0.01, True),
            ('post_blur_irregularity_strength', 0.0, 5.0, 0.8, 0.1, True),
            ('post_irregularity_noise_scale', 0.001, 0.1, 0.02, 0.001, True),
            ('post_irregularity_micro_displacement', 0.0, 5.0, 0.5, 0.1, True),
            ('post_irregularity_blur_sigma', 0.0, 10.0, 0.0, 0.1, True),

        ]

        for name, min_val, max_val, default, step, is_float in params:
            slider = ParamSlider(
                text=name.replace('_', ' ').title(),
                min=min_val,
                max=max_val,
                value=default,
                step=step,
                for_float=is_float,
                label_width=300,
            )
            slider.bind(slider=self.on_param_change)
            self.param_container.add_widget(slider)

        self.update_image()

    def on_param_change(self, instance, value):
        if self.update_event:
            self.update_event.cancel()
        self.update_event = Clock.schedule_once(lambda dt: self.update_image(), 0.1)

    def update_image(self):
        params = {}
        for child in self.param_container.children:
            if isinstance(child, ParamSlider):
                params[child.text.replace(' ', '_').lower()] = child.value

        processed = create_ghost(
            self.base_image.copy(),
            light_source_coords=self.light_source,
            **params,
            random_seed=45
        )
        
        buf = (processed * 255).astype(np.uint8)
        buf = cv2.flip(buf, 0)
        texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image_widget.texture = texture

class GhostEditorApp(MDApp):
    def build(self):
        return GhostEditor()

if __name__ == '__main__':
    GhostEditorApp().run()
