
from kivy.app import App as KVApp
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import StringProperty as KVStringProperty

class MetaInfo(KVBoxLayout):
    key = KVStringProperty()
    value = KVStringProperty()


class MetaInfoApp(KVApp):
    def __init__(self, **kwargs):
        super(MetaInfoApp, self).__init__(**kwargs)

    def build(self): 
        widget = MetaInfo()

        return widget

if __name__ == '__main__':
    MetaInfoApp().run()
