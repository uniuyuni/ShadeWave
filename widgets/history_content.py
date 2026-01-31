
from kivy.uix.recycleview.views import RecycleDataViewBehavior as KVRecycleDataViewBehavior
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty
from kivy.uix.recycleboxlayout import RecycleBoxLayout as KVRecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior as KVFocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior as KVLayoutSelectionBehavior
from kivymd.uix.list import OneLineListItem as MDOneLineListItem
from kivymd.uix.scrollview import MDScrollView
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from utils import kvutils

import re


class HistoryItem(KVBoxLayout, KVRecycleDataViewBehavior):
    active = KVBooleanProperty(True)
    selected = KVBooleanProperty(False)
    col1_text = KVStringProperty("")
    col2_text = KVStringProperty("")
    col3_text = KVStringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = None
    
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

    def refresh_view_attrs(self, rv, index, data):
        ret = super().refresh_view_attrs(rv, index, data)
        self.active = data.get('active', True)
        self.index = index
        self.col1_text = data.get('col1_text', "")
        self.col2_text = data.get('col2_text', "")
        self.col3_text = data.get('col3_text', "")
        
        # Force height update on refresh
        if self.ref_height:
             self.height = kvutils.dpi_scale_height(self.ref_height)
        
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

class SelectableRecycleBoxLayout(KVFocusBehavior, KVLayoutSelectionBehavior, KVRecycleBoxLayout):
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
        items_count = 2 if len(rv.data) < 2 else len(rv.data)
        pos_ratio = 1 - (index / float(items_count - 1))
        rv.scroll_y = max(0, min(1, pos_ratio))

    def set_history(self, history):
        op_dict_list = [{'col1_text': 'Load', 'col2_text': '', 'col3_text': '', 'active': True}]
        active_index = 0
        for i, op in enumerate(history.operations):
            is_active = i <= history.current_index
            op_dict_list.append(self._make_op_dict(op, is_active))
            if is_active and active_index <= i:
                active_index += 1
        self.ids['history_rv'].data = []
        self.ids['history_rv'].data = op_dict_list

        layout = self.ids['history_rv'].children[0]
        layout.clear_selection()
        if active_index not in layout.selected_nodes:
            layout.selected_nodes.append(active_index)
            # 表示アイテム自体の 'selected' 属性も更新
            views = self.ids['history_rv'].view_adapter.views
            if len(views) > 0 and active_index >= min(views.keys()) and active_index <= max(views.keys()):
                widget = views[active_index]
                widget.selected = True
            self._scroll_to_index(self.ids['history_rv'], active_index)

    def _make_op_dict(self, op, is_active):
        def _format_value(value):
            if isinstance(value, bool):
                return "ON" if value else "OFF"
            elif isinstance(value, (int, float)):
                return self._format_number(value)
            elif isinstance(value, list):
                return f"{len(value)} pts"
            elif value is None:
                return "2 pts"
            return str(value)

        # バックアップとアップデートで共通のキーかつ値が違うものを抽出
        if not op.diff:
             return {'col1_text': op.name, 'col2_text': '', 'col3_text': '', 'active': is_active}

        result = op.diff[0]
        bps = _format_value(result[1])
        ups = _format_value(result[2])
        col1 = self._format_string(result[0])
        
        return {'col1_text': col1, 'col2_text': bps, 'col3_text': ups, 'active': is_active}

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

    def _format_string(self, text):
        """
        正規表現を使用した簡潔なバージョン
        """
        if not text:
            return ""
        
        # アンダースコアをスペースに変換し、次の文字を大文字にする
        # 正規表現でアンダースコア+次の文字を検出して処理
        result = re.sub(r'_([a-z])?', lambda m: ' ' + m.group(1).upper() if m.group(1) else ' ', text)
        
        # 先頭文字を大文字にする
        result = result[0].upper() + result[1:] if result else ""
        
        return result

def create_history_content_panel(callback):
    return HistoryContentPanel(callback)
