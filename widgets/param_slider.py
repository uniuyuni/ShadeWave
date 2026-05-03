
from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.behaviors.toggle_behavior import MDToggleButton
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import NumericProperty as KVNumericProperty, StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty
from kivy.metrics import dp
import math

import widgets.float_input
import widgets.multi_slider
import widgets.tiny_button


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
        parent = self.parent
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
            if hasattr(parent, 'on_input_scrub_begin'):
                parent.on_input_scrub_begin()
            self._scrub_last_x = touch.x
        dx = touch.x - self._scrub_last_x
        self._scrub_last_x = touch.x
        if hasattr(parent, 'on_input_scrub_pixels'):
            parent.on_input_scrub_pixels(dx)
        return True

    def on_touch_up(self, touch):
        if self._scrub_touch_uid == touch.uid:
            if self._scrub_active:
                parent = self.parent
                if hasattr(parent, 'on_input_scrub_end'):
                    parent.on_input_scrub_end()
                if touch.grab_current is self:
                    touch.ungrab(self)
                self._scrub_active = False
            self._scrub_touch_uid = None
        return super().on_touch_up(touch)


class HeadLabel(KVBoxLayout):
    press = KVBooleanProperty(True)
    active = KVBooleanProperty(True)
    release = KVBooleanProperty(True)

    def on_label_touch_down(self, touch):
        if not touch.is_double_tap:
            return False
        app = MDApp.get_running_app()
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
    before_edit = KVNumericProperty(float('inf'))
    after_edit = KVNumericProperty(float('inf'))

    #def __init__(self, **kwargs):
    #    super(ParamSlider, self).__init__(**kwargs)
    
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.disabled = True
        self.reset_value = self.value
        self.ids['label'].text = self.text
        #self.ids['label'].width = self.label_width
        self.ids['slider'].max = self.max
        self.ids['slider'].min = self.min
        self.ids['slider'].value = self.value
        self.ids['slider'].step = self.step
        self.ids['input'].set_value(self.value)
        self.disabled = False
    
    def on_label_text(self):
        self.ids['label'].text = self.text

    def on_slider_value(self):
        self.value = self.ids['slider'].value
        self.ids['input'].set_value(self.value)
        if self.disabled == False:
            v = self.value
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

    def on_input_text_validate(self):
        try:
            if self.for_float:
                val = round(self.ids['input'].get_value(), 2)
            else:
                val = int(self.ids['input'].get_value())
        except ValueError:
            val = self.reset_value
        self.before_edit = self.value
        val = min(self.max, max(self.min, val))
        self.ids['input'].set_value(val)
        self.value = val
        self.ids['slider'].value = self.value
        self.after_edit = self.value

    def on_button_press(self, step):
        self.before_edit = self.value
        self.value = min(self.max, max(self.min, self.ids['slider'].value + step))
        self.ids['slider'].value = self.value
        self.after_edit = self.value

    def on_input_scrub_begin(self):
        self.before_edit = self.value
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
        new_val = min(self.max, max(self.min, self.ids['slider'].value + delta))
        if new_val != self.ids['slider'].value:
            self.ids['slider'].value = new_val

    def on_input_scrub_end(self):
        self.after_edit = self.value
        self._input_scrub_accum = 0.0

    def on_slider_touch_down(self, touch):
        if touch.is_double_tap:
            if self.ids['label'].collide_point(*touch.pos):
                self.before_edit = self.value
                self.ids['slider'].value = self.reset_value
                return True

        # 以前ここで reset_value へ戻しており、MultiSlider の on_touch 後に
        # これが走るとスライダーが常に初期値(例: レンズ歪み 0)のままになる。
        if self.ids['slider'].collide_point(*touch.pos):
            self.before_edit = self.value
            return True

        return False

    def on_slider_touch_up(self, touch):
        self.after_edit = self.value
        return False

    def set_slider_value(self, value):
        self.disabled = True
        self.ids['slider'].value = value
        self.disabled = False

    def set_slider_range(self, min_value, max_value, step=None):
        self.disabled = True
        self.min = min_value
        self.max = max_value
        if step is not None:
            self.step = step
            self.ids['slider'].step = step
        self.ids['slider'].min = min_value
        self.ids['slider'].max = max_value
        self.ids['slider'].value = min(max(self.ids['slider'].value, min_value), max_value)
        self.value = self.ids['slider'].value
        self.ids['input'].set_value(self.ids['slider'].value)
        self.disabled = False

    def set_slider_reset(self, value):
        self.reset_value = value
    
class Param_SliderApp(MDApp):
    def __init__(self, **kwargs):
        super(Param_SliderApp, self).__init__(**kwargs)
        
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

    def build(self): 
        widget = ParamSlider()

        return widget

if __name__ == '__main__':
    Param_SliderApp().run()

