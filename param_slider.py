
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import NumericProperty as KVNumericProperty, StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty
from kivy.metrics import dp

import switchable_float_input

class ParamSlider(KVBoxLayout):
    text = KVStringProperty()
    min = KVNumericProperty(-100)
    max = KVNumericProperty(100)
    value = KVNumericProperty(0)
    step = KVNumericProperty(1)
    for_float = KVBooleanProperty(False)
    slider = KVNumericProperty(float('inf')) #　最初の変更は必ずコールバックが呼ばれるようにする
    label_width = KVNumericProperty(100)

    #def __init__(self, **kwargs):
    #    super(ParamSlider, self).__init__(**kwargs)
    
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.disabled = True
        self.reset_value = self.value
        self.ids['label'].text = self.text
        #self.ids['label'].width = self.label_width
        self.ids['slider'].value = self.value
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
            self.slider = self.value

    def on_input_text_validate(self):
        try:
            if self.for_float:
                val = round(self.ids['input'].get_value(), 2)
            else:
                val = int(self.ids['input'].get_value())
        except ValueError:
            val = self.reset_value
        val = min(self.max, max(self.min, val))
        self.ids['input'].set_value(val)
        self.value = val
        self.ids['slider'].value = self.value
    
    def on_button_press(self, step):
        self.value = min(self.max, max(self.min, self.ids['slider'].value + step))
        self.ids['slider'].value = self.value
    
    def on_slider_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_double_tap:
                self.ids['slider'].value = self.reset_value

    def set_slider_value(self, value):
        from kivy.event import EventDispatcher
        self.disabled = True
        self.ids['slider'].value = value
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

