
import logging

from kivy.app import App as KVApp
from kivy.uix.spinner import Spinner as KVSpinner, SpinnerOption as KVSpinnerOption
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.core.window import Window as KVWindow
from kivy.factory import Factory as KVFactory
from kivy.properties import (
    ObjectProperty as KVObjectProperty,
    StringProperty as KVStringProperty,
    BooleanProperty as KVBooleanProperty,
)


class HoverSpinnerOption(KVSpinnerOption):
    # ホバー中の項目だけ白枠を描くためのフラグ（kv の canvas.after で参照）。
    hovered = KVBooleanProperty(False)


# kv の option_cls: "HoverSpinnerOption" 文字列を解決できるよう Factory に登録する。
KVFactory.register('HoverSpinnerOption', cls=HoverSpinnerOption)


class HoverSpinner(KVSpinner):
    hovered_item = KVObjectProperty(None, allownone=True)
    value = KVStringProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        KVWindow.bind(mouse_pos=self.on_mouse_pos)

    def _set_hovered(self, item):
        # 旧項目の白枠を消し、新項目に白枠を付けてから hovered_item を更新する
        # （hovered_item の dispatch は on_hovered_item: の効果プレビューに使われる）。
        prev = self.hovered_item
        if prev is not None and hasattr(prev, 'hovered'):
            prev.hovered = False
        if item is not None and hasattr(item, 'hovered'):
            item.hovered = True
        self.hovered_item = item
        if item is not None:
            logging.debug("Cursor entered Spinner: %s", item.text)
        else:
            logging.debug("Cursor left Spinner")

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
            wpos = self._dropdown.to_widget(*pos)
            for item in self._dropdown.container.children:
                if item.collide_point(*wpos):
                    if self.hovered_item is not item:
                        self._set_hovered(item)
                    return

        if self.hovered_item is not None:
            self._set_hovered(None)

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
