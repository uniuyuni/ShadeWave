
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import StringProperty as KVStringProperty

class MetaInfo(KVBoxLayout):
    key = KVStringProperty()
    value = KVStringProperty()


class MetaInfoApp(MDApp):
    def __init__(self, **kwargs):
        super(MetaInfoApp, self).__init__(**kwargs)
        
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

    def build(self): 
        widget = MetaInfo()

        return widget

if __name__ == '__main__':
    MetaInfoApp().run()

