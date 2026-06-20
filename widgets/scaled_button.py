from kivy.clock import Clock
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton

from utils import kvutils


class ScaledButtonMixin:
    bg_color_normal = ListProperty([0.38, 0.5, 0.54, 0.92])
    bg_color_down = ListProperty([0.13, 0.23, 0.74, 0.95])
    bg_color_disabled = ListProperty([0.18, 0.18, 0.18, 0.45])
    bg_color_reset_normal = ListProperty([0.12, 0.13, 0.16, 0.92])
    ref_layout_padding = ListProperty([0, 0])
    icon = StringProperty("")
    icon_size = StringProperty("")
    type = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_disabled_normal = ""
        self.background_disabled_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = (1, 1, 1, 1)
        self.bind(state=self._update_bg, disabled=self._update_bg, text=self._update_bg)
        self._update_bg()

    def _update_bg(self, *_args):
        if self.disabled:
            self.background_color = self.bg_color_disabled
        elif self.state == "down":
            self.background_color = self.bg_color_down
        elif getattr(self, "text", "") == "Reset":
            self.background_color = self.bg_color_reset_normal
        else:
            self.background_color = self.bg_color_normal

    def set_ref_metrics(
            self,
            width_ref=50,
            height_ref=22,
            font_ref=11,
            padding_x_ref=8,
            padding_y_ref=2):
        self.size_hint = (None, None)
        self.ref_width = width_ref
        self.ref_height = height_ref
        self.ref_font_size = font_ref
        self.ref_layout_padding = [padding_x_ref, padding_y_ref]
        kvutils.traverse_widget(self)
        return self


class ScaledButton(ScaledButtonMixin, Button):
    """Fixed-size Kivy button that does not impose KivyMD minimum sizes."""


class ScaledToggleButton(ScaledButtonMixin, ToggleButton):
    """Fixed-size Kivy toggle button with the same styling as ScaledButton."""


class LongPressScaledButton(ScaledButton):
    """ScaledButton that dispatches press/release only after a hold."""

    long_press_time = NumericProperty(0.45)
    long_press_ready = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._long_press_event = None
        self._long_press_touch = None

    def on_touch_down(self, touch):
        if self.disabled or not self.collide_point(*touch.pos):
            return False
        touch.grab(self)
        self._long_press_touch = touch
        self.long_press_ready = False
        self.state = "down"
        self._long_press_event = Clock.schedule_once(
            self._mark_long_press_ready,
            self.long_press_time,
        )
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        touch.ungrab(self)
        if self._long_press_event is not None:
            self._long_press_event.cancel()
            self._long_press_event = None
        ready = self.long_press_ready and self.collide_point(*touch.pos)
        self.long_press_ready = False
        self._long_press_touch = None
        self.state = "normal"
        if ready:
            self.dispatch("on_press")
            self.dispatch("on_release")
        return True

    def _mark_long_press_ready(self, *_args):
        self._long_press_event = None
        self.long_press_ready = True
