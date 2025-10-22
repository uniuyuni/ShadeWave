
from kivy.uix.recycleview.views import RecycleDataViewBehavior as KVRecycleDataViewBehavior
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelOneLine
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivymd.uix.list import OneLineListItem as MDOneLineListItem
from kivymd.uix.scrollview import MDScrollView

import re

class HistoryCustomHeader(MDExpansionPanelOneLine):
    pass

class HistoryItem(MDOneLineListItem, KVRecycleDataViewBehavior):
    active = KVBooleanProperty(True)
    is_selected = KVBooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = None
    
    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            return self.parent.select_with_touch(self.index, touch)
        return False
    
    def apply_selection(self, rv, index, is_selected):
        self.is_selected = is_selected
        if is_selected:
            # コールバック処理をここに記述
            if self.parent is not None:
                self.parent.parent.parent.callback(index)

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    pass

class HistoryContentPanel(MDScrollView):    
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback

    def on_kv_post(self, *args, **kwargs):
        super(HistoryContentPanel, self).on_kv_post(*args, **kwargs)
        self.ids['history_rv'].data=[]

    def set_op_list(self, op_list):
        self.ids['history_rv'].data.clear()
        for op in op_list:
            self._append_op(op)
        self.ids['history_rv'].refresh_from_data()
        
    def _append_op(self, op):
        # バックアップとアップデートで共通のキーかつ値が違うものを抽出
        result = [
            [key, op.backup[key], op.update[key]]
            for key in op.backup.keys() & op.update.keys()
            if op.backup[key] != op.update[key]
        ][0]
        # 項目の追加
        list_item = f"{self._format_string(result[0], 15)}{self._format_number(result[1])}, {self._format_number(result[2])}"
        self.ids['history_rv'].data.append({'text': list_item})

    def set_active_index(self, index):
        pass

    def _format_number(self, num):
        integer_part = int(num)
        decimal_part = num - integer_part
        
        if decimal_part == 0:
            # 小数部がない場合
            return f"{integer_part: 7d}"
        else:
            # 小数部がある場合
            d = f"{decimal_part:.2f}".split('.')[1]
            return f"{integer_part: 4d}.{d}"

    def _format_string(self, text, n):
        """
        正規表現を使用した簡潔なバージョン
        """
        if not text:
            return " " * n
        
        # アンダースコアをスペースに変換し、次の文字を大文字にする
        # 正規表現でアンダースコア+次の文字を検出して処理
        result = re.sub(r'_([a-z])?', lambda m: ' ' + m.group(1).upper() if m.group(1) else ' ', text)
        
        # 先頭文字を大文字にする
        result = result[0].upper() + result[1:] if result else ""
        
        # n文字に成形
        return result.ljust(n)[:n]

def create_history_content_panel(callback):
    return HistoryContentPanel(callback)

    content = HistoryContentPanel()
    header = HistoryCustomHeader(text="History")

    # ExpansionPanelを作成して追加
    panel = MDExpansionPanel(
        content=content,
        panel_cls=header
    )

    return panel

