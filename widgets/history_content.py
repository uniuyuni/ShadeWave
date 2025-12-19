
from kivy.uix.recycleview.views import RecycleDataViewBehavior as KVRecycleDataViewBehavior
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelOneLine
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivymd.uix.list import OneLineListItem as MDOneLineListItem
from kivymd.uix.scrollview import MDScrollView
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout

import re

class HistoryCustomHeader(MDExpansionPanelOneLine):
    pass

class HistoryItem(MDOneLineListItem, KVRecycleDataViewBehavior):
    active = KVBooleanProperty(True)
    selected = KVBooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = None
    
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

    def refresh_view_attrs(self, rv, index, data):
        ret = super().refresh_view_attrs(rv, index, data)
        self.active = data.get('active', True)
        self.index = index
        self.ids['_lbl_primary'].color = (.8, .8, .8, 1) if self.active == True else (.4, .4, .4, 1)
        return ret

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        
        if self.collide_point(*touch.pos):
            return self.parent.select_with_touch(self.index, touch)
        return False
    
    def apply_selection(self, rv, index, is_selected):
        if index >= len(rv.data):
            return
        
        self.selected = is_selected
        if is_selected:
            # コールバック処理をここに記述
            if self.parent is not None:
                self.parent.parent.parent.callback(index-1)

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    pass

class HistoryContentPanel(KVBoxLayout):    
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback

    def on_kv_post(self, *args, **kwargs):
        super(HistoryContentPanel, self).on_kv_post(*args, **kwargs)
        self.ids['history_rv'].data = []

    def _scroll_to_index(self, rv, index):
        layout = rv.children[0]
        # 一例: visible_widgetsは現在表示されているwidgetリスト
        visible_widgets = [w for w in layout.children if w.parent == layout]
        if not visible_widgets:
#            rv.scroll_y = 1 # 初期状態
            return
        first_index = min([w.index for w in visible_widgets])
        last_index = max([w.index for w in visible_widgets])
        if first_index <= index <= last_index+1:
            return  # 表示範囲内なので何もしない
        # 範囲外ならスクロール実施
        items_count = len(rv.data)
        pos_ratio = 1 - (index / float(items_count - 1))
        rv.scroll_y = max(0, min(1, pos_ratio))

    def set_history(self, history):
        op_dict_list = [{'text': 'Load', 'active': True}]
        active_index = 0
        for i, op in enumerate(history.operations):
            is_active = i <= history.current_index
            op_dict_list.append(self._make_op_dict(op, is_active))
            if is_active and active_index <= i:
                active_index += 1
        self.ids['history_rv'].data = op_dict_list

        layout = self.ids['history_rv'].children[0]
        layout.clear_selection()
        if active_index not in layout.selected_nodes:
            layout.selected_nodes.append(active_index)
            # 表示アイテム自体の 'selected' 属性も更新
            if active_index < len(self.ids['history_rv'].view_adapter.views):
                widget = self.ids['history_rv'].view_adapter.views[active_index]
                widget.selected = True
            self._scroll_to_index(self.ids['history_rv'], active_index)

    def _make_op_dict(self, op, is_active):
        def _format_value(value):
            if isinstance(value, bool):
                return str(value)
            elif isinstance(value, (int, float)):
                return self._format_number(value)
            elif isinstance(value, list):
                return f"{len(value)} pts"
            elif value is None:
                return "2 pts"
            return str(value)

        # バックアップとアップデートで共通のキーかつ値が違うものを抽出
        result = op.diff[0]
        bps = _format_value(result[1])
        ups = _format_value(result[2])
        list_item = f"{self._format_string(result[0], 15)} {bps}, {ups}"

        return {'text': list_item, 'active': is_active}

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
    """
    content = HistoryContentPanel()
    header = HistoryCustomHeader(text="History")

    # ExpansionPanelを作成して追加
    panel = MDExpansionPanel(
        content=content,
        panel_cls=header
    )

    return panel
    """