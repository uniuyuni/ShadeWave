
from kivy.app import App as KVApp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.uix.widget import Widget as KVWidget
from kivy.properties import NumericProperty as KVNumericProperty, StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, ObjectProperty as KVObjectProperty, ListProperty as KVListProperty
from kivy.graphics.texture import Texture as KVTexture
from kivy.metrics import dp
import logging
import math
import numpy as np
import os

import widgets.float_input
import widgets.multi_slider
import widgets.tiny_button


_BAR_TEXTURE_CACHE = {}
_DEBUG_MASK_GEOMETRY = os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"}
_MASK_GEOM_SLIDER_TEXTS = {"Rotation", "Translate X", "Translate Y", "Scale X", "Scale Y"}


def _linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


def _context_value(context, key, default):
    if context is None:
        return default
    return context.get(key, default)


def _cache_float(value):
    return round(float(value), 6)


def _boost_saturation(rgb, saturation):
    if saturation == 1.0:
        return rgb

    luma = (
        rgb[..., 0:1] * 0.2126
        + rgb[..., 1:2] * 0.7152
        + rgb[..., 2:3] * 0.0722
    )
    return np.clip(luma + (rgb - luma) * saturation, 0.0, 1.0)


def _effect_rgb_to_display(rgb, saturation=1.0):
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, None)
    rgb_max = np.max(rgb, axis=1, keepdims=True)
    rgb = rgb / np.maximum(rgb_max, 1e-8)
    rgb = _linear_to_srgb(rgb)
    return _boost_saturation(rgb, saturation)


def _make_bar_texture_from_rgb(rgb_line, width, height, cache_key, saturation=1.0):
    texture = _BAR_TEXTURE_CACHE.get(cache_key)
    if texture is not None:
        return texture

    img = np.round(_effect_rgb_to_display(rgb_line, saturation) * 255.0).astype(np.uint8)
    img = img[np.newaxis, :, :]
    if height > 1:
        img = np.repeat(img, height, axis=0)

    texture = KVTexture.create(size=(width, height), colorfmt='rgb', bufferfmt='ubyte')
    texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
    texture.mag_filter = 'linear'
    texture.min_filter = 'linear'
    _BAR_TEXTURE_CACHE[cache_key] = texture
    return texture


def _convert_temp_tint_line(temps, tints, y):
    import cores.core as core

    return np.array(
        [
            core.convert_TempTint2RGB(float(temp), float(tint), float(y))
            for temp, tint in zip(temps, tints)
        ],
        dtype=np.float32,
    )


def _make_color_temperature_bar_texture(slider, width=512, height=1):
    context = slider.bar_context or {}
    reset_temp = _context_value(context, "reset_temp", getattr(slider, "reset_value", 5000.0))
    reset_tint = _context_value(context, "reset_tint", 0.0)
    fixed_tint = _context_value(context, "fixed_tint", reset_tint)
    y = _context_value(context, "Y", 1.0)
    cache_key = (
        "color_temperature",
        width,
        height,
        _cache_float(slider.min),
        _cache_float(slider.max),
        _cache_float(reset_temp),
        _cache_float(reset_tint),
        _cache_float(fixed_tint),
        _cache_float(y),
        _cache_float(slider.bar_saturation),
    )
    cached = _BAR_TEXTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    temps = np.linspace(float(slider.min), float(slider.max), width, dtype=np.float32)
    tints = np.full(width, float(fixed_tint), dtype=np.float32)
    reset_rgb = _convert_temp_tint_line([reset_temp], [reset_tint], y)[0]
    sample_rgb = _convert_temp_tint_line(temps, tints, y)
    rgb_line = reset_rgb[np.newaxis, :] / np.maximum(sample_rgb, 1e-6)
    return _make_bar_texture_from_rgb(rgb_line, width, height, cache_key, slider.bar_saturation)


def _make_color_tint_bar_texture(slider, width=512, height=1):
    context = slider.bar_context or {}
    reset_temp = _context_value(context, "reset_temp", 5000.0)
    reset_tint = _context_value(context, "reset_tint", getattr(slider, "reset_value", 0.0))
    fixed_temp = _context_value(context, "fixed_temp", reset_temp)
    y = _context_value(context, "Y", 1.0)
    cache_key = (
        "color_tint",
        width,
        height,
        _cache_float(slider.min),
        _cache_float(slider.max),
        _cache_float(reset_temp),
        _cache_float(reset_tint),
        _cache_float(fixed_temp),
        _cache_float(y),
        _cache_float(slider.bar_saturation),
    )
    cached = _BAR_TEXTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    temps = np.full(width, float(fixed_temp), dtype=np.float32)
    tints = np.linspace(float(slider.min), float(slider.max), width, dtype=np.float32)
    reset_rgb = _convert_temp_tint_line([reset_temp], [reset_tint], y)[0]
    sample_rgb = _convert_temp_tint_line(temps, tints, y)
    rgb_line = reset_rgb[np.newaxis, :] / np.maximum(sample_rgb, 1e-6)
    return _make_bar_texture_from_rgb(rgb_line, width, height, cache_key, slider.bar_saturation)


def _context_range_midpoint(context, key, default):
    values = _context_value(context, key, default)
    try:
        if values is None or len(values) < 2:
            values = default
    except TypeError:
        values = default
    return (float(values[0]) + float(values[1])) * 0.5


def _make_hls_hue_shift_bar_texture(slider, width=512, height=1):
    context = slider.bar_context or {}
    center = float(_context_value(context, "center", 0.0))
    sample_l = _context_range_midpoint(context, "l_range", (0.1, 0.9))
    sample_c = _context_range_midpoint(context, "s_range", (0.25, 0.75))
    gain = float(_context_value(context, "gain", 1.0))
    cache_key = (
        "hls_hue_shift",
        width,
        height,
        _cache_float(slider.min),
        _cache_float(slider.max),
        _cache_float(center),
        _cache_float(sample_l),
        _cache_float(sample_c),
        _cache_float(gain),
        _cache_float(slider.bar_saturation),
    )
    cached = _BAR_TEXTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    hue_delta = np.linspace(float(slider.min), float(slider.max), width, dtype=np.float32)
    hue_rad = np.deg2rad((center + hue_delta) % 360.0)
    chroma = sample_c * 1.5
    cb = chroma * np.cos(hue_rad)
    cr = chroma * np.sin(hue_rad)

    kr, kg, kb = 0.2126, 0.7152, 0.0722
    r = (sample_l + cr) * gain
    g = (sample_l - (kr / kg) * cr - (kb / kg) * cb) * gain
    b = (sample_l + cb) * gain
    rgb_line = np.stack((r, g, b), axis=1).astype(np.float32)
    return _make_bar_texture_from_rgb(rgb_line, width, height, cache_key, slider.bar_saturation)


_BAR_RENDERERS = {
    "color_temperature": _make_color_temperature_bar_texture,
    "color_tint": _make_color_tint_bar_texture,
    "hls_hue_shift": _make_hls_hue_shift_bar_texture,
}


class ParamFloatInput(widgets.float_input.FloatInput):
    """数値枠内で水平ドラッグするとスライダーの step 単位で値を増減する。"""

    _SCRUB_THRESHOLD_PX = dp(2)
    _PIXELS_PER_STEP = dp(2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scrub_touch_uid = None
        self._scrub_active = False
        self._scrub_start = (0.0, 0.0)
        self._scrub_last_x = 0.0

    def _scrub_owner(self):
        parent = self.parent
        while parent is not None:
            if hasattr(parent, 'on_input_scrub_pixels'):
                return parent
            parent = getattr(parent, 'parent', None)
        return None

    def on_touch_down(self, touch):
        self._scrub_touch_uid = None
        self._scrub_start = (touch.x, touch.y)
        self._scrub_last_x = touch.x
        self._scrub_active = False
        if self.collide_point(*touch.pos):
            self._scrub_touch_uid = touch.uid
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._scrub_touch_uid != touch.uid:
            return super().on_touch_move(touch)
        owner = self._scrub_owner()
        sx, sy = self._scrub_start
        dx_total = touch.x - sx
        dy_total = touch.y - sy
        if not self._scrub_active:
            if abs(dx_total) < self._SCRUB_THRESHOLD_PX and abs(dy_total) < self._SCRUB_THRESHOLD_PX:
                return super().on_touch_move(touch)
            if abs(dx_total) < abs(dy_total):
                return super().on_touch_move(touch)
            self._scrub_active = True
            self.focus = False
            touch.grab(self)
            if owner is not None and hasattr(owner, 'on_input_scrub_begin'):
                owner.on_input_scrub_begin()
            self._scrub_last_x = touch.x
        dx = touch.x - self._scrub_last_x
        self._scrub_last_x = touch.x
        if owner is not None:
            owner.on_input_scrub_pixels(dx)
        return True

    def on_touch_up(self, touch):
        if self._scrub_touch_uid == touch.uid:
            if self._scrub_active:
                owner = self._scrub_owner()
                if owner is not None and hasattr(owner, 'on_input_scrub_end'):
                    owner.on_input_scrub_end()
                if touch.grab_current is self:
                    touch.ungrab(self)
                self._scrub_active = False
            self._scrub_touch_uid = None
        return super().on_touch_up(touch)


class HeadToggleButton(ButtonBehavior, KVWidget):
    active = KVBooleanProperty(True)

    def on_release(self):
        if not self.disabled:
            self.active = not self.active


class HeadLabel(KVBoxLayout):
    press = KVBooleanProperty(True)
    active = KVBooleanProperty(True)
    enabled = KVBooleanProperty(True)
    release = KVBooleanProperty(True)

    def on_active(self, _instance, value):
        if self.enabled != value:
            self.enabled = value

    def on_enabled(self, _instance, value):
        if self.active != value:
            self.active = value

    def on_label_touch_down(self, touch):
        if not touch.is_double_tap:
            return False
        app = KVApp.get_running_app()
        main_widget = getattr(app, "main_widget", None) if app else None
        if main_widget and hasattr(main_widget, "reset_switch_defaults_for_label"):
            return bool(main_widget.reset_switch_defaults_for_label(self))
        return False

class ParamSlider(KVBoxLayout):
    text = KVStringProperty()
    min = KVNumericProperty(-100)
    max = KVNumericProperty(100)
    value = KVNumericProperty(0)
    step = KVNumericProperty(1)
    for_float = KVBooleanProperty(False)
    slider = KVNumericProperty(float('inf')) #　最初の変更は必ずコールバックが呼ばれるようにする
    label_width = KVNumericProperty(100)
    before_edit = KVNumericProperty(0)
    after_edit = KVNumericProperty(0)
    bar_renderer = KVStringProperty("")
    bar_source = KVStringProperty("")
    bar_texture = KVObjectProperty(None, allownone=True)
    bar_opacity = KVNumericProperty(1.0)
    bar_saturation = KVNumericProperty(1.0)
    bar_show_active_overlay = KVBooleanProperty(True)
    bar_show_anchor_marker = KVBooleanProperty(False)
    bar_context = KVObjectProperty(None, allownone=True)
    multi_value_edit_mode = KVStringProperty("active")
    slider_values = KVListProperty([])
    allow_overlap = KVBooleanProperty(False)
    multi_point_count = KVNumericProperty(1)
    show_multi_value_boxes = KVBooleanProperty(False)
    show_right_value_controls = KVBooleanProperty(True)

    def __init__(self, **kwargs):
        super(ParamSlider, self).__init__(**kwargs)
        self._editing = False
        self.reset_values = None
    
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.disabled = True
        self.set_slider_reset(self.value)
        self.ids['label'].text = self.text
        #self.ids['label'].width = self.label_width
        self.ids['slider'].max = self.max
        self.ids['slider'].min = self.min
        self.ids['slider'].value = self.value
        self.ids['slider'].step = self.step
        self.ids['slider'].allow_overlap = self.allow_overlap
        self.ids['input'].set_value(self.value)
        if self.slider_values:
            self.set_slider_reset(self.slider_values)
            self.set_slider_value(self.slider_values)
        self._sync_bar_to_slider()
        self._sync_multi_value_ui()
        self.bind(
            min=self._sync_bar_to_slider,
            max=self._sync_bar_to_slider,
            bar_renderer=self._sync_bar_to_slider,
            bar_source=self._sync_bar_to_slider,
            bar_texture=self._sync_bar_to_slider,
            bar_opacity=self._sync_bar_to_slider,
            bar_saturation=self._sync_bar_to_slider,
            bar_show_active_overlay=self._sync_bar_to_slider,
            bar_show_anchor_marker=self._sync_bar_to_slider,
            bar_context=self._sync_bar_to_slider,
            multi_value_edit_mode=self._sync_multi_value_ui,
            slider_values=self._on_slider_values_property,
            allow_overlap=self._sync_allow_overlap_to_slider,
        )
        self.disabled = False

    def on_slider_values(self, *args):
        self._on_slider_values_property(*args)

    def _on_slider_values_property(self, *args):
        if not hasattr(self, "ids") or "slider" not in self.ids:
            return
        if self.slider_values:
            self.set_slider_value(self.slider_values)

    def _sync_allow_overlap_to_slider(self, *args):
        if not hasattr(self, "ids") or "slider" not in self.ids:
            return
        self.ids['slider'].allow_overlap = self.allow_overlap
        self._sync_multi_value_ui()
    
    def on_label_text(self):
        self.ids['label'].text = self.text

    def on_slider_value(self):
        if len(getattr(self.ids['slider'], "values", []) or []) > 1:
            self._sync_multi_value_ui()
            return
        self.value = self.ids['slider'].value
        self.ids['input'].set_value(self.value)
        if self.disabled == False:
            self._emit_slider_value(self.value)

    def on_multi_slider_values(self):
        self._sync_multi_value_ui()
        if self.disabled == False and self._is_multi_value_slider():
            self._emit_slider_value(self._active_slider_value())

    def on_slider_active_index(self):
        self._sync_multi_value_ui()

    def _emit_slider_value(self, value):
        v = value
        s = self.slider
        # set_slider_value 中に self.disabled が True だと self.slider を入れ替えておらず、
        # 直前の s と今回の v が同じ数になると self.slider = v が等値扱いで
        # on_slider(→ apply_effects) が1回も飛ばない。Ge の Lens など、ボタンで set_slider
        # 同期が重なると出やすい。一度 inf に戻してから v を入れて on_slider を確実に出す。
        if s != float("inf") and (
            s == v
            or (
                self.for_float
                and math.isclose(s, v, rel_tol=0.0, abs_tol=1e-5)
            )
        ):
            self.slider = float("inf")
        self.slider = v

    def _is_multi_value_slider(self):
        return len(getattr(self.ids['slider'], "values", []) or []) > 1

    def _active_slider_index(self):
        slider = self.ids['slider']
        values = list(getattr(slider, "values", []) or [slider.value])
        if not values:
            return 0
        return max(0, min(int(getattr(slider, "active_index", 0)), len(values) - 1))

    def _active_slider_value(self):
        slider = self.ids['slider']
        values = list(getattr(slider, "values", []) or [slider.value])
        if not values:
            return slider.value
        return values[self._active_slider_index()]

    def _format_input_value(self, value):
        return round(value, 2) if self.for_float else int(value)

    def _value_from_input(self, input_widget):
        if self.for_float:
            return round(input_widget.get_value(), 2)
        return int(input_widget.get_value())

    def _clamp_value_for_index(self, index, value, values):
        value = min(self.max, max(self.min, value))
        if not getattr(self.ids['slider'], "allow_overlap", False) and len(values) > 1:
            if index > 0:
                value = max(values[index - 1], value)
            if index < len(values) - 1:
                value = min(values[index + 1], value)
        return value

    def _set_slider_value_at(self, index, value):
        slider = self.ids['slider']
        values = list(getattr(slider, "values", []) or [slider.value])
        if not values:
            values = [slider.value]
        index = max(0, min(int(index), len(values) - 1))
        value = self._clamp_value_for_index(index, value, values)
        if len(values) == 1:
            slider.value = value
        else:
            values[index] = value
            slider.active_index = index
            slider.values = values
            self._sync_multi_value_ui()

    def _visual_value_indices(self, count=None):
        slider = self.ids['slider']
        values = list(getattr(slider, "values", []) or [slider.value])
        if count is None:
            count = len(values)
        visual = sorted(
            range(count),
            key=lambda i: (slider._get_x_from_value(values[i]), i),
        )
        if count <= 3:
            return visual
        return [visual[0], visual[count // 2], visual[-1]]

    def _multi_slot_value_index(self, slot, count=None):
        if count is None:
            count = len(getattr(self.ids['slider'], "values", []) or [self.ids['slider'].value])
        visual = self._visual_value_indices(count)
        if slot == 0 and count >= 1:
            return visual[0]
        if slot == 1 and count >= 3:
            return visual[1]
        if slot == 2 and count >= 2:
            return visual[-1]
        return None

    def _sync_multi_value_ui(self, *args):
        if not hasattr(self, "ids") or "slider" not in self.ids:
            return
        slider = self.ids['slider']
        values = list(getattr(slider, "values", []) or [slider.value])
        count = len(values)
        self.multi_point_count = max(1, min(count, 3))
        multi = count > 1
        split = multi and self.multi_value_edit_mode == "split"
        self.show_multi_value_boxes = split
        self.show_right_value_controls = not split
        active_value = self._active_slider_value()
        self.value = active_value
        self.ids['input'].set_value(self._format_input_value(active_value))
        for slot, input_id in enumerate(("input_multi_0", "input_multi_1", "input_multi_2")):
            input_widget = self.ids.get(input_id)
            value_index = self._multi_slot_value_index(slot, count)
            if input_widget is not None and value_index is not None:
                input_widget.set_value(self._format_input_value(values[value_index]))

    def on_slider(self, *args):
        if _DEBUG_MASK_GEOMETRY and self.text in _MASK_GEOM_SLIDER_TEXTS:
            logging.warning(
                "[MASK_GEOM] ParamSlider.on_slider text=%s slider=%s value=%s disabled=%s",
                self.text,
                self.slider,
                self.value,
                self.disabled,
            )

    def on_input_text_validate(self):
        try:
            val = self._value_from_input(self.ids['input'])
        except ValueError:
            val = self.reset_value
        self._notify_before_edit()
        val = min(self.max, max(self.min, val))
        self.ids['input'].set_value(val)
        self._set_slider_value_at(self._active_slider_index(), val)
        self._notify_after_edit()

    def on_button_press(self, step):
        self._notify_before_edit()
        self._set_slider_value_at(self._active_slider_index(), self._active_slider_value() + step)
        self._notify_after_edit()

    def _set_pipeline_drag_state(self, active):
        # スライダーの連続ドラッグ中だけ half-res プレビュー(main 側)を有効化する。
        # ボタン押し/テキスト入力の瞬間編集はドラッグ扱いにしない。
        app = KVApp.get_running_app()
        main_widget = getattr(app, "main_widget", None) if app else None
        notify = getattr(main_widget, "set_param_slider_drag", None)
        if callable(notify):
            notify(bool(active))

    def on_input_scrub_begin(self):
        self._notify_before_edit()
        self._set_pipeline_drag_state(True)
        self._input_scrub_accum = 0.0

    def on_input_scrub_pixels(self, dx):
        self._input_scrub_accum += dx
        pps = ParamFloatInput._PIXELS_PER_STEP
        st = self.ids['slider'].step
        while self._input_scrub_accum >= pps:
            self._apply_input_scrub_step(st)
            self._input_scrub_accum -= pps
        while self._input_scrub_accum <= -pps:
            self._apply_input_scrub_step(-st)
            self._input_scrub_accum += pps

    def _apply_input_scrub_step(self, delta):
        if delta == 0:
            return
        self._set_slider_value_at(self._active_slider_index(), self._active_slider_value() + delta)

    def on_multi_input_text_validate(self, index):
        input_widget = self.ids.get(f"input_multi_{index}")
        if input_widget is None:
            return
        value_index = self._multi_slot_value_index(index)
        if value_index is None:
            return
        try:
            val = self._value_from_input(input_widget)
        except ValueError:
            val = self.reset_value
        self._notify_before_edit()
        self._set_slider_value_at(value_index, val)
        self._notify_after_edit()

    def on_multi_button_press(self, index, step):
        values = list(getattr(self.ids['slider'], "values", []) or [self.ids['slider'].value])
        value_index = self._multi_slot_value_index(index, len(values))
        if value_index is None:
            return
        self._notify_before_edit()
        self._set_slider_value_at(value_index, values[value_index] + step)
        self._notify_after_edit()

    def _reset_slider_to_default(self):
        self._notify_before_edit()
        if self.reset_values and len(self.reset_values) > 1:
            slider = self.ids['slider']
            slider.active_index = min(slider.active_index, len(self.reset_values) - 1)
            slider.values = list(self.reset_values)
            self._sync_multi_value_ui()
        else:
            self._set_slider_value_at(self._active_slider_index(), self.reset_value)
        self._notify_after_edit()

    def on_label_touch_down(self, touch):
        if not touch.is_double_tap:
            return False
        self._reset_slider_to_default()
        return True

    def on_input_scrub_end(self):
        self._set_pipeline_drag_state(False)
        self._notify_after_edit()
        self._input_scrub_accum = 0.0

    def on_slider_interaction_start(self):
        self._notify_before_edit()

    def on_slider_interaction_end(self):
        self._notify_after_edit()

    def on_slider_touch_down(self, touch):
        if touch.is_double_tap:
            if self.ids['label'].collide_point(*touch.pos):
                self._reset_slider_to_default()
                return True

        # 以前ここで reset_value へ戻しており、MultiSlider の on_touch 後に
        # これが走るとスライダーが常に初期値(例: レンズ歪み 0)のままになる。
        if self.ids['slider'].collide_point(*touch.pos):
            self._notify_before_edit()
            self._set_pipeline_drag_state(True)
            return True

        return False

    def on_slider_touch_up(self, touch):
        self._set_pipeline_drag_state(False)
        self._notify_after_edit()
        return False

    def _notify_before_edit(self):
        if self._editing:
            return
        self._editing = True
        if _DEBUG_MASK_GEOMETRY and self.text in _MASK_GEOM_SLIDER_TEXTS:
            logging.warning("[MASK_GEOM] ParamSlider.before_edit text=%s value=%s", self.text, self.value)
        self.before_edit += 1

    def _notify_after_edit(self):
        if not self._editing:
            return
        self._editing = False
        if _DEBUG_MASK_GEOMETRY and self.text in _MASK_GEOM_SLIDER_TEXTS:
            logging.warning("[MASK_GEOM] ParamSlider.after_edit text=%s value=%s", self.text, self.value)
        self.after_edit += 1

    def _make_bar_texture(self):
        if not self.bar_renderer:
            return None

        renderer = _BAR_RENDERERS.get(self.bar_renderer)
        if renderer is None:
            logging.warning(f"Unknown slider bar renderer: {self.bar_renderer}")
            return None

        return renderer(self)

    def _sync_bar_to_slider(self, *args):
        if not hasattr(self, "ids") or "slider" not in self.ids:
            return

        slider = self.ids["slider"]
        slider.track_opacity = self.bar_opacity
        slider.track_show_active_overlay = self.bar_show_active_overlay
        slider.track_show_anchor_marker = self.bar_show_anchor_marker

        if self.bar_texture is not None:
            slider.track_texture = self.bar_texture
            slider.track_source = ""
        elif self.bar_source:
            slider.track_texture = None
            slider.track_source = self.bar_source
        else:
            slider.track_texture = self._make_bar_texture()
            slider.track_source = ""

    def set_bar_context(self, context):
        self.bar_context = dict(context or {})

    def set_slider_value(self, value):
        self.disabled = True
        if isinstance(value, (list, tuple)):
            values = list(value)
            if values:
                self.ids['slider'].values = values
                if len(values) == 1:
                    self.ids['slider'].value = values[0]
                self.ids['slider'].active_index = min(self.ids['slider'].active_index, len(values) - 1)
        else:
            self.ids['slider'].value = value
        self.disabled = False
        self._sync_multi_value_ui()

    def set_slider_range(self, min_value, max_value, step=None):
        self.disabled = True
        self.min = min_value
        self.max = max_value
        if step is not None:
            self.step = step
            self.ids['slider'].step = step
        self.ids['slider'].min = min_value
        self.ids['slider'].max = max_value
        values = list(getattr(self.ids['slider'], "values", []) or [self.ids['slider'].value])
        if len(values) > 1:
            self.ids['slider'].values = [min(max(v, min_value), max_value) for v in values]
        else:
            self.ids['slider'].value = min(max(self.ids['slider'].value, min_value), max_value)
        self.value = self._active_slider_value()
        self.ids['input'].set_value(self.value)
        self._sync_bar_to_slider()
        self.disabled = False
        self._sync_multi_value_ui()

    def set_slider_reset(self, value):
        if isinstance(value, (list, tuple)):
            self.reset_values = list(value)
            self.reset_value = self.reset_values[0] if self.reset_values else self.value
        else:
            self.reset_values = None
            self.reset_value = value
        self.ids['slider'].anchor_value = self.reset_value
        self._sync_bar_to_slider()
    
class Param_SliderApp(KVApp):
    def __init__(self, **kwargs):
        super(Param_SliderApp, self).__init__(**kwargs)

    def build(self): 
        widget = ParamSlider()

        return widget

if __name__ == '__main__':
    Param_SliderApp().run()
