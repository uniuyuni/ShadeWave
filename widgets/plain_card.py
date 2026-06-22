from kivy.graphics import Color, RoundedRectangle
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout


class PlainCard(BoxLayout):
    bg_color = ListProperty([0, 0, 0, 0])
    radius = ListProperty([0])
    shadow_color = ListProperty([0, 0, 0, 0])
    shadow_offset = ListProperty([0, -2])
    shadow_spread = ListProperty([0, 0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            self._shadow_color_instruction = Color(*self.shadow_color)
            self._shadow_rect = RoundedRectangle(
                pos=self._shadow_pos(),
                size=self._shadow_size(),
                radius=self.radius,
            )
            self._bg_color_instruction = Color(*self.bg_color)
            self._bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=self.radius)
        self.bind(
            pos=self._update_card_canvas,
            size=self._update_card_canvas,
            bg_color=self._update_card_canvas,
            radius=self._update_card_canvas,
            shadow_color=self._update_card_canvas,
            shadow_offset=self._update_card_canvas,
            shadow_spread=self._update_card_canvas,
        )

    def _shadow_pos(self):
        spread_x, spread_y = self.shadow_spread
        offset_x, offset_y = self.shadow_offset
        return self.x + offset_x - spread_x, self.y + offset_y - spread_y

    def _shadow_size(self):
        spread_x, spread_y = self.shadow_spread
        return self.width + spread_x * 2, self.height + spread_y * 2

    def _update_card_canvas(self, *_args):
        self._shadow_color_instruction.rgba = self.shadow_color
        self._shadow_rect.pos = self._shadow_pos()
        self._shadow_rect.size = self._shadow_size()
        self._shadow_rect.radius = self.radius
        self._bg_color_instruction.rgba = self.bg_color
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._bg_rect.radius = self.radius
