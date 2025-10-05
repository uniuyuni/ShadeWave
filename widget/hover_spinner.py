
from kivy.app import App
from kivy.uix.spinner import Spinner
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.properties import ObjectProperty

class HoverSpinner(Spinner):
    hovered_item = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouse_pos)

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

class Hover_SpinnerApp(App):
    def build(self):
        layout = BoxLayout()
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