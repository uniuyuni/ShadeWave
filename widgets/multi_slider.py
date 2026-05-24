
from kivy.uix.widget import Widget
from kivy.properties import ListProperty, NumericProperty, ColorProperty, BooleanProperty, ObjectProperty, StringProperty
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, RoundedRectangle, Line, Rectangle

import utils.kvutils as kvutils

class MultiSlider(Widget):
    # --- プロパティ定義 ---
    value = NumericProperty(0)
    min = NumericProperty(0)
    max = NumericProperty(100)
    step = NumericProperty(0)
    values = ListProperty([])
    
    allow_overlap = BooleanProperty(False)
    draw_from_anchor = BooleanProperty(False)
    anchor_value = NumericProperty(0)
    
    track_color_active = ColorProperty([0.85, 0.85, 0.85, 1])
    track_color_inactive = ColorProperty([0.2, 0.2, 0.2, 1])
    track_texture = ObjectProperty(None, allownone=True)
    track_source = StringProperty("")
    track_opacity = NumericProperty(1.0)
    track_show_active_overlay = BooleanProperty(True)
    track_show_anchor_marker = BooleanProperty(False)
    thumb_color = ColorProperty([1, 1, 1, 1])
    thumb_colors = ListProperty([])
    disabled_color = ColorProperty([0.6, 0.6, 0.6, 1])
    
    debug_mode = BooleanProperty(False)
    
    _active_idx = None
    _selection_locked = False

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        
        if not self.values:
            self.values = [self.value]

        self.bind(values=self._on_values_change)
        self.bind(value=self._on_value_change)
        
        # 描画更新: 独自のスケール計算が入るため、レイアウト確定後に行うのが安全
        update_trigger = lambda *dt: Clock.schedule_once(self._refresh_view, 0)
        
        self.bind(pos=update_trigger, size=update_trigger, 
                  thumb_colors=update_trigger, disabled=update_trigger, debug_mode=update_trigger,
                  track_texture=update_trigger, track_source=update_trigger,
                  track_opacity=update_trigger, track_show_active_overlay=update_trigger,
                  track_show_anchor_marker=update_trigger)
        
        update_trigger()

    def _on_value_change(self, instance, new_val):
        if len(self.values) == 1 and self.values[0] == new_val:
            return
        self.values = [new_val]

    def _on_values_change(self, instance, new_list):
        if len(new_list) == 1:
            if self.value != new_list[0]:
                self.value = new_list[0]
        Clock.schedule_once(self._refresh_view, 0)

    # --- レイアウト計算 (kvutil対応) ---
    def _get_track_layout(self):
        # パディング: dp(8)相当 -> 16
        padding = kvutils.dpi_scale_width(16)
        
        # Y軸オフセット: トラックの高さ(4)の半分 -> 2
        offset_y = kvutils.dpi_scale_height(2)
        
        track_x = self.x + padding
        track_y = self.center_y - offset_y
        
        usable_width = self.width - (padding * 2)
        if usable_width < 0: usable_width = 0
        
        return track_x, track_y, usable_width

    def _get_x_from_value(self, value):
        track_x, _, usable_width = self._get_track_layout()
        if self.max == self.min or usable_width <= 0:
            return track_x
        ratio = (value - self.min) / (self.max - self.min)
        return track_x + (ratio * usable_width)

    def _get_value_from_x(self, touch_x):
        track_x, _, usable_width = self._get_track_layout()
        if usable_width <= 0: return self.min
        rel_x = touch_x - track_x
        ratio = rel_x / usable_width
        value = ratio * (self.max - self.min) + self.min
        return max(self.min, min(self.max, value))

    # --- タッチイベント ---
    def on_touch_down(self, touch):
        if self.disabled: return False
        
        # Kivy標準のcollide_pointを使う (pos/sizeは親レイアウトによって制御されている前提)
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)

        closest_idx = -1
        min_dist = float('inf')

        for i, val in enumerate(self.values):
            thumb_x = self._get_x_from_value(val)
            dist = abs(touch.x - thumb_x)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if closest_idx != -1:
            self._active_idx = closest_idx
            self._selection_locked = False
            self._update_value_from_touch_x(touch.x)
            return True
            
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.disabled: return False
        if self._active_idx is None: return super().on_touch_move(touch)
        self._update_value_from_touch_x(touch.x)
        return True

    def on_touch_up(self, touch):
        if self.disabled: return False
        if self._active_idx is not None:
            self._active_idx = None
            self._selection_locked = False
            return True
        return super().on_touch_up(touch)

    def _update_value_from_touch_x(self, touch_x):
        raw_val = self._get_value_from_x(touch_x)
        if self.step > 0:
            new_val = round(raw_val / self.step) * self.step
        else:
            new_val = raw_val

        current_active_val = self.values[self._active_idx]
        if not self._selection_locked:
            diff = new_val - current_active_val
            if abs(diff) > 0:
                overlaps = [i for i, v in enumerate(self.values) 
                           if abs(v - current_active_val) < 1e-5]
                if len(overlaps) > 1:
                    if diff > 0:
                        self._active_idx = max(overlaps)
                    else:
                        self._active_idx = min(overlaps)

        if not self.allow_overlap:
            if self._active_idx > 0:
                limit_min = self.values[self._active_idx - 1]
                if new_val < limit_min: new_val = limit_min
            if self._active_idx < len(self.values) - 1:
                limit_max = self.values[self._active_idx + 1]
                if new_val > limit_max: new_val = limit_max

        current_vals = list(self.values)
        if current_vals[self._active_idx] != new_val:
            current_vals[self._active_idx] = new_val
            self.values = current_vals
            self._selection_locked = True

    # --- 描画処理 ---
    def _refresh_view(self, *dt):
        self.canvas.clear()
        self.canvas.after.clear()
        
        # 共通レイアウト計算
        track_x, track_y, usable_width = self._get_track_layout()
        
        # トラックの高さ:
        track_h = kvutils.dpi_scale_height(4)
        
        # 角丸の半径:
        radius_val = kvutils.dpi_scale_height(2)
        
        active_color = self.disabled_color if self.disabled else self.track_color_active
        inactive_color = [c * 0.5 for c in self.track_color_inactive[:3]] + [self.track_color_inactive[3]] \
                         if self.disabled else self.track_color_inactive

        with self.canvas:
            if self.debug_mode:
                Color(1, 0, 0, 0.3)
                Rectangle(pos=self.pos, size=self.size)
                Color(0, 1, 0, 0.8)
                Line(points=[self.x, self.center_y, self.right, self.center_y], width=1)

            has_track_image = self.track_texture is not None or bool(self.track_source)
            if has_track_image:
                Color(1, 1, 1, self.track_opacity)
                if self.track_texture is not None:
                    RoundedRectangle(texture=self.track_texture, pos=(track_x, track_y), size=(usable_width, track_h), radius=[radius_val])
                else:
                    RoundedRectangle(source=self.track_source, pos=(track_x, track_y), size=(usable_width, track_h), radius=[radius_val])
                if self.disabled:
                    Color(0, 0, 0, 0.35)
                    RoundedRectangle(pos=(track_x, track_y), size=(usable_width, track_h), radius=[radius_val])
            else:
                # Inactive Track
                Color(rgba=inactive_color)
                RoundedRectangle(pos=(track_x, track_y), size=(usable_width, track_h), radius=[radius_val])

            # Active Track
            if self.track_show_active_overlay:
                Color(rgba=active_color)
                if len(self.values) == 1:
                    val = self.values[0]
                    curr_x = self._get_x_from_value(val)
                    if self.draw_from_anchor:
                        anchor_x = self._get_x_from_value(self.anchor_value)
                        start_x = min(anchor_x, curr_x)
                        w = abs(curr_x - anchor_x)
                    else:
                        start_x = track_x
                        w = curr_x - track_x
                    RoundedRectangle(pos=(start_x, track_y), size=(w, track_h), radius=[radius_val])

                elif len(self.values) == 2:
                    x1 = self._get_x_from_value(self.values[0])
                    x2 = self._get_x_from_value(self.values[1])
                    v1, v2 = self.values[0], self.values[1]
                    if self.allow_overlap and v1 > v2:
                        RoundedRectangle(pos=(track_x, track_y), size=(x2 - track_x, track_h), radius=[radius_val])
                        max_x = track_x + usable_width
                        RoundedRectangle(pos=(x1, track_y), size=(max_x - x1, track_h), radius=[radius_val])
                    else:
                        RoundedRectangle(pos=(min(x1, x2), track_y), size=(abs(x2 - x1), track_h), radius=[radius_val])

            if self.track_show_anchor_marker and self.draw_from_anchor:
                anchor_x = self._get_x_from_value(self.anchor_value)
                marker_h = max(track_h + kvutils.dpi_scale_height(6), kvutils.dpi_scale_height(10))
                marker_top = track_y + (track_h + marker_h) / 2
                marker_bottom = track_y + (track_h - marker_h) / 2
                Color(0, 0, 0, 0.75)
                Line(points=[anchor_x, marker_bottom, anchor_x, marker_top], width=2)
                Color(1, 1, 1, 0.9)
                Line(points=[anchor_x, marker_bottom, anchor_x, marker_top], width=1)

        with self.canvas.after:
            # サムサイズ:
            thumb_size = kvutils.dpi_scale_height(16) 
            
            # 中心からオフセット
            thumb_y = self.center_y - thumb_size / 2
            
            # 影のオフセット: dp(1)相当 -> 2
            shadow_offset = kvutils.dpi_scale_height(2)
            
            # 枠線の太さ: 1.2 (ピクセル固定で良いか、スケールするかはお好みで。一旦そのまま)
            line_width = 1.2

            for i, val in enumerate(self.values):
                t_x = self._get_x_from_value(val) - thumb_size / 2
                
                if self.disabled:
                    c_thumb = self.disabled_color
                elif i < len(self.thumb_colors):
                    c_thumb = self.thumb_colors[i]
                else:
                    c_thumb = self.thumb_color

                if not self.disabled:
                    Color(0, 0, 0, 0.2)
                    Ellipse(pos=(t_x, thumb_y - shadow_offset), size=(thumb_size, thumb_size))
                
                Color(rgba=c_thumb)
                Ellipse(pos=(t_x, thumb_y), size=(thumb_size, thumb_size))
                
                if not self.disabled:
                    Color(rgba=active_color)
                    Line(circle=(t_x + thumb_size/2, thumb_y + thumb_size/2, thumb_size/2), width=line_width)
