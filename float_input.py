
from kivy.uix.textinput import TextInput as KVTextInput
import re

class FloatInput(KVTextInput):
    """
    浮動小数点の値を内部的に保持しながら、
    for_floatフラグによって表示を整数のみにするか小数点も含めるか切り替え可能なTextInputクラス
    """
    pat = re.compile('[^0-9]')
    
    def __init__(self, for_float=False, **kwargs):
        super().__init__(**kwargs)
        self.for_float = for_float  # 小数点表示を許可するかどうかのフラグ
        self.bind(text=self._on_text_change)
        self._internal_value = ""  # 内部で保持する浮動小数点の文字列
        
    def insert_text(self, substring, from_undo=False):
        pat = self.pat
        if '.' in self._internal_value:
            s = re.sub(pat, '', substring)
        else:
            s = '.'.join(
                re.sub(pat, '', s)
                for s in substring.split('.', 1)
            )
        
        # 内部値を更新
        old_cursor_pos = self.cursor[0]
        cursor_pos = old_cursor_pos
        
        if cursor_pos > len(self._internal_value):
            cursor_pos = len(self._internal_value)
            
        self._internal_value = self._internal_value[:cursor_pos] + s + self._internal_value[cursor_pos:]
        
        # 表示値を更新（for_floatフラグに基づく）
        self.text = self._get_display_text()
        
        return True
    
    def _on_text_change(self, instance, value):
        """
        ユーザー入力による直接的なtext変更を監視し、
        必要に応じて整数/浮動小数点表示に修正する
        """
        # プログラム的に変更されたときは何もしない
        if self.text != self._get_display_text():
            # 内部値を更新
            if '.' in self.text:
                parts = self.text.split('.')
                integer_part = parts[0]
                
                if '.' in self._internal_value:
                    # 既存の小数部分があり、かつ新しい入力に小数部分がある場合
                    if len(parts) > 1:
                        decimal_part = parts[1]
                    else:
                        decimal_part = self._internal_value.split('.')[1]
                    self._internal_value = integer_part + '.' + decimal_part
                else:
                    # 既存の小数部分がない場合
                    if len(parts) > 1:
                        self._internal_value = integer_part + '.' + parts[1]
                    else:
                        self._internal_value = integer_part
            else:
                # 新しい入力に小数点がない場合
                if '.' in self._internal_value:
                    decimal_part = self._internal_value.split('.')[1]
                    self._internal_value = self.text + '.' + decimal_part
                else:
                    self._internal_value = self.text
            
            # フラグに基づいて表示を更新
            self.text = self._get_display_text()
    
    def _get_display_text(self):
        """
        for_floatフラグに基づいて表示用テキストを取得
        True: 浮動小数点表示
        False: 整数部分のみ表示
        """
        if not self.for_float:
            # 整数表示モード
            if '.' in self._internal_value:
                return self._internal_value.split('.')[0]
            return self._internal_value
        else:
            # 浮動小数点表示モード（小数点第二位まで）
            if '.' in self._internal_value:
                parts = self._internal_value.split('.')
                integer_part = parts[0]
                decimal_part = parts[1]
                # 小数点以下を最大2桁に制限
                decimal_part = decimal_part[:2]
                return integer_part + '.' + decimal_part
            return self._internal_value
            
    def get_value(self):
        """浮動小数点の実際の値を取得"""
        try:
            return float(self._internal_value)
        except ValueError:
            return 0.0
        
    def set_value(self, value):
        """浮動小数点の値を設定"""
        self._internal_value = str(value)
        self.text = self._get_display_text()
        
    def set_float_mode(self, for_float):
        """表示モードを切り替える"""
        self.for_float = for_float
        self.text = self._get_display_text()  # 表示を更新
