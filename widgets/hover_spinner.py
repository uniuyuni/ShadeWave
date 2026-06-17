
from kivy.app import App as KVApp
from kivy.uix.spinner import Spinner as KVSpinner
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.core.window import Window as KVWindow
from kivy.properties import ObjectProperty as KVObjectProperty, StringProperty as KVStringProperty

class HoverSpinner(KVSpinner):
    hovered_item = KVObjectProperty(None, allownone=True)
    value = KVStringProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        KVWindow.bind(mouse_pos=self.on_mouse_pos)

    def set_text(self, text):
        self.disabled = True
        self.text = text
        self.disabled = False

    def on_text(self, instance, value):
        if self.disabled == False:
            if self.value == value:
                self.property('value').dispatch(self)
            else:
                self.value = value
        
    def on_mouse_pos(self, window, pos):
        # ドロップダウンが開いている場合のアイテムホバー検出
        if hasattr(self, '_dropdown') and self._dropdown and self.is_open:
            for item in self._dropdown.container.children:
                if item.collide_point(*self._dropdown.to_widget(*pos)):
                    if self.hovered_item != item:
                        self.hovered_item = item
                        print(f"Cursor entered Spinner: {item.text}")
                        return
                    else:
                        return
                    
        if self.hovered_item is not None:
            self.hovered_item = None
            print(f"Cursor left Spinner")

class Hover_SpinnerApp(KVApp):
    def build(self):
        layout = KVBoxLayout()
        spinner = HoverSpinner(
            values=("Option 1", "Option 2", "Option 3"),
            size_hint=(None, None),
            size=(200, 44),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        layout.add_widget(spinner)
        return layout

if __name__ == '__main__':
    Hover_SpinnerApp().run()
