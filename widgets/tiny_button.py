
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.properties import ColorProperty

class TinyButton(Button):
    """
    Kivy標準のButtonと完全互換ですが、最小1pxまで縮小可能です。
    画像を使わず、CanvasのRectangleで描画します。
    """
    
    # 色設定 (RGBA) - 好みに合わせて変更可能
    bg_color_normal = ColorProperty([0.35, 0.35, 0.35, 1])   # 通常時 (グレー)
    bg_color_down = ColorProperty([0.2, 0.6, 1, 1])         # 押下時 (青)
    bg_color_disabled = ColorProperty([0.5, 0.5, 0.5, 0.5]) # 無効時 (薄いグレー)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
