import os

from kivy.animation import Animation
from kivy.lang import Builder as KVBuilder
from kivy.metrics import dp
from kivy.properties import (
    BooleanProperty,
    ColorProperty,
    ListProperty,
    NumericProperty,
)
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.widget import Widget


CUR_DIR = os.path.dirname(__file__)
KVBuilder.load_file(os.path.join(CUR_DIR, "modern_checkbox.kv"))


class ModernCheckBox(ButtonBehavior, Widget):
    active = BooleanProperty(False)
    disabled = BooleanProperty(False)

    box_color = ColorProperty([1, 1, 1, 1])
    border_color_inactive = ColorProperty([0.796, 0.827, 0.882, 1])
    border_color_active = ColorProperty([0.122, 0.247, 0.686, 1])
    focus_color = ColorProperty([0.863, 0.902, 1, 1])
    check_color = ColorProperty([0.122, 0.247, 0.686, 1])

    _border_color = ColorProperty([0.796, 0.827, 0.882, 1])
    _focus_alpha = NumericProperty(0)
    _check_progress = NumericProperty(0)
    _content_scale = NumericProperty(1)
    _box_side = NumericProperty(dp(14))
    _box_pos = ListProperty([0, 0])
    _focus_side = NumericProperty(dp(20))
    _focus_pos = ListProperty([0, 0])
    _check_points = ListProperty([0, 0, 0, 0, 0, 0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            active=self._update_visual_state,
            disabled=self._update_visual_state,
            pos=self._update_geometry,
            size=self._update_geometry,
            _content_scale=self._update_geometry,
        )
        self._update_geometry()
        self._update_visual_state(animate=False)

    def on_press(self):
        if self.disabled:
            return
        Animation.cancel_all(self, "_focus_alpha", "_content_scale")
        Animation(_focus_alpha=1, _content_scale=0.96, duration=0.08, t="out_quad").start(self)

    def on_release(self):
        if self.disabled:
            return
        Animation.cancel_all(self, "_focus_alpha", "_content_scale")
        Animation(_focus_alpha=0, _content_scale=1, duration=0.14, t="out_quad").start(self)

    def on_touch_down(self, touch):
        if self.disabled:
            return False
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.disabled:
            return False
        if touch.grab_current is self:
            should_toggle = self.collide_point(*touch.pos)
            if should_toggle:
                self.active = not self.active
        return super().on_touch_up(touch)

    def _update_visual_state(self, *args, animate=True):
        target_border = self.border_color_active if self.active else self.border_color_inactive
        target_check = 1 if self.active else 0

        Animation.cancel_all(self, "_border_color", "_check_progress")
        if animate:
            Animation(_border_color=target_border, duration=0.14, t="out_quad").start(self)
            Animation(_check_progress=target_check, duration=0.16, t="out_quad").start(self)
        else:
            self._border_color = target_border
            self._check_progress = target_check

    def _box_size(self):
        side = min(self.width, self.height)
        base_size = min(max(side * 0.6, dp(11)), dp(15))
        return base_size * self._content_scale

    def _update_geometry(self, *args):
        box_side = self._box_size()
        box_x = self.x + (self.width - box_side) / 2
        box_y = self.y + (self.height - box_side) / 2
        focus_side = min(min(self.width, self.height), box_side + dp(2))
        focus_x = self.x + (self.width - focus_side) / 2
        focus_y = self.y + (self.height - focus_side) / 2

        self._box_side = box_side
        self._box_pos = [box_x, box_y]
        self._focus_side = focus_side
        self._focus_pos = [focus_x, focus_y]
        self._check_points = [
            box_x + box_side * 0.26,
            box_y + box_side * 0.50,
            box_x + box_side * 0.43,
            box_y + box_side * 0.36,
            box_x + box_side * 0.72,
            box_y + box_side * 0.64,
        ]

    def _box_radius(self):
        return [self._box_side * 0.28]

    def _focus_size(self):
        return (self._focus_side, self._focus_side)

    def _focus_radius(self):
        return [self._focus_side * 0.32]

    def _border_line_width(self):
        s = float(self._box_side)
        # Pixel座標の Line.width：dp で下限を上げ過ぎると低 DPI で太く見える
        return max(1.0, min(s * 0.044, s * 0.075))

    def _check_line_width(self):
        s = float(self._box_side)
        return max(1.0, min(s * 0.058, s * 0.10))
