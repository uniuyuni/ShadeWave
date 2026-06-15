
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
            elif isinstance(value, tuple):
                return f"{value[0]}x{value[1]}"
            elif value is None:
                return "2 pts"
            return str(value)

        # バックアップとアップデートで共通のキーかつ値が違うものを抽出
        if not op.diff:
             return {'col1_text': op.name, 'col2_text': '', 'col3_text': '', 'active': is_active}

        # matrix操作の場合はバックアップを優先して表示
        if len(op.diff) <= 1 or op.diff[0][0] is not "matrix":
            result = op.diff[0]
        else:
            result = op.diff[1]
        
        if result[2] is "Reset": # Reset操作の場合
            title = "Reset"
            bps = ""
            ups = ""
        elif result[0] is "crop_rect": # Crop操作の場合
            title = "Crop"
            maxsize = max(op.effects_param['original_img_size'])
            bp_w = int((result[1][2] - result[1][0]) * maxsize)
            bp_h = int((result[1][3] - result[1][1]) * maxsize)
            up_w = int((result[2][2] - result[2][0]) * maxsize)
            up_h = int((result[2][3] - result[2][1]) * maxsize)
            bps = _format_value((bp_w, bp_h))
            ups = _format_value((up_w, up_h))
        elif result[0] is "flip_mode": # Flip操作の場合

            def _format_flip(value):
                if value == 0:
                    return "Normal"
                elif value == 1:
                    return "Horizontal"
                elif value == 2:
                    return "Vertical"
                elif value == 3:
                    return "Both"
                return str(value)

            title = "Flip"
            bps = _format_flip(result[1])
            ups = _format_flip(result[2])
        else:
            REPLACE_TEXT = {
                "color_temperature": "temperature",
                "color_tint": "tint",
                "hls_red_hue": "red_hue",
                "hls_red_lum": "red_luminance",
                "hls_red_sat": "red_saturation",
                "hls_red_hue_full_range": "red_hue_full_range",
                "hls_skin_hue": "skin_hue",
                "hls_skin_lum": "skin_luminance",
                "hls_skin_sat": "skin_saturation",
                "hls_skin_hue_full_range": "skin_hue_full_range",
                "hls_orange_hue": "orange_hue",
                "hls_orange_lum": "orange_luminance",
                "hls_orange_sat": "orange_saturation",
                "hls_orange_hue_full_range": "orange_hue_full_range",
                "hls_yellow_hue": "yellow_hue",
                "hls_yellow_lum": "yellow_luminance",
                "hls_yellow_sat": "yellow_saturation",
                "hls_yellow_hue_full_range": "yellow_hue_full_range",
                "hls_green_hue": "green_hue",
                "hls_green_lum": "green_luminance",
                "hls_green_sat": "green_saturation",
                "hls_green_hue_full_range": "green_hue_full_range",
                "hls_cyan_hue": "cyan_hue",
                "hls_cyan_lum": "cyan_luminance",
                "hls_cyan_sat": "cyan_saturation",
                "hls_cyan_hue_full_range": "cyan_hue_full_range",
                "hls_blue_hue": "blue_hue",
                "hls_blue_lum": "blue_luminance",
                "hls_blue_sat": "blue_saturation",
                "hls_blue_hue_full_range": "blue_hue_full_range",
                "hls_purple_hue": "purple_hue",
                "hls_purple_lum": "purple_luminance",
                "hls_purple_sat": "purple_saturation",
                "hls_purple_hue_full_range": "purple_hue_full_range",
                "hls_magenta_hue": "magenta_hue",
                "hls_magenta_lum": "magenta_luminance",
                "hls_magenta_sat": "magenta_saturation",
                "hls_magenta_hue_full_range": "magenta_hue_full_range",
                "vignette_radius_percent": "vignette_radius",
                "tonecurve": "tone_curve",
                "tonecurve_red": "tone_curve_red",
                "tonecurve_green": "tone_curve_green",
                "tonecurve_blue": "tone_curve_blue",
                "grading1": "grading1_curve",
                "grading1_hue": "grading1_hue",
                "grading1_lum": "grading1_luminance",
                "grading1_sat": "grading1_saturation",
                "grading2": "grading2_curve",
                "grading2_hue": "grading2_hue",
                "grading2_lum": "grading2_luminance",
                "grading2_sat": "grading2_saturation",
                "HuevsHue": "Hue vs Hue",
                "HuevsLum": "Hue vs Lum",
                "HuevsSat": "Hue vs Sat",
                "LumvsLum": "Lum vs Lum",
                "LumvsSat": "Lum vs Sat",
                "SatvsLum": "Sat vs Lum",
                "SatvsSat": "Sat vs Sat",
                "light_noise_reduction": "light_luminance_NR",
                "light_color_noise_reduction": "light_color_NR",
                "deblur_filter": "deblur",
                "switch_lut": "switch_LUT",
                "lut_name": "LUT_file",
                "lut_intensity": "LUT_intensity",
                "lut_to_log": "LUT_to_log",
                "lensblur_filter": "lensblur",
                "grain_color_noise_ratio": "grain_color_noise",
                "cross_filter_num_points": "cross_points",
                "cross_filter_length": "cross_length",
                "cross_filter_angle": "cross_angle",
                "cross_filter_threshold": "cross_threshold",
                "cross_filter_intensity": "cross_intensity",
                "cross_filter_spectral": "cross_spectral",
                "cross_filter_thickness": "cross_thickness",
                "cross_filter_distance": "cross_distance",
                "cross_filter_random": "cross_random",
                "rotation2": "Rotation_90",
                "highlight_compress": "tonemap",
                "detail_tonemap": "detail_tonemap",
            }
            title = REPLACE_TEXT[result[0]] if result[0] in REPLACE_TEXT else result[0]
            title = self._format_string(title)
            bps = _format_value(result[1])
            ups = _format_value(result[2])
        
        return {'col1_text': title, 'col2_text': bps, 'col3_text': ups, 'active': is_active}

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
