
from kivymd.app import MDApp
from kivymd.uix.widget import MDWidget
from kivy.core.window import Window

import histogram_widget
import curve
import param_slider
import viewer_widget
import spacer
import metainfo
import mask_editor2

class MainWidget(MDWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_effects_lv(self, lv, effect):
        return True
   
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        
        self.title = 'escargot'
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

    def build(self): 
        widget = MainWidget()

        Window.size = (1024, 800)

        return widget


if __name__ == '__main__':
    MainApp().run()


