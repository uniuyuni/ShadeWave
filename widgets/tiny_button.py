
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.widget import Widget
from kivy.properties import ColorProperty, ListProperty, StringProperty

class TinyButton(ButtonBehavior, Widget):
    """
    Kivy標準のButton背景スタイルを避けた、最小1pxまで縮小可能なボタンです。
    画像を使わず、CanvasのRectangleで描画します。
    """
    
    # 色設定 (RGBA) - 好みに合わせて変更可能
    bg_color_normal = ColorProperty([0.18, 0.18, 0.18, 1])   # 通常時 (入力欄になじませる)
    bg_color_down = ColorProperty([0.13, 0.23, 0.74, 1])    # 押下時 (紺)
    bg_color_disabled = ColorProperty([0.18, 0.18, 0.18, 0.45]) # 無効時
    direction = StringProperty("")
    triangle_points = ListProperty([0, 0, 0, 0, 0, 0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self._update_metrics,
                  size=self._update_metrics,
                  direction=self._update_metrics)
        self._update_metrics()

    def _update_metrics(self, *_args):
        # 低解像度ではボタンが ~10px と小さく、頂点がサブピクセルに来ると
        # 三角形のラスタライズが崩れる（「⊤」状になる）。頂点を整数ピクセルへ
        # スナップし、左右対称・最小サイズを保証してくっきり描く。
        base = min(self.width, self.height)
        # 低解像度では潰れて頂点が消えるので、底辺幅より高さを優先して最小値を確保。
        half_w = max(2, round(base * 0.24))
        half_h = max(3, round(base * 0.20))
        # 頂点 X は中心列のピクセル中心(.5)に置き、左右の底辺を整数境界に揃える
        cx = round(self.center_x - 0.5) + 0.5
        cy = round(self.center_y - 0.5) + 0.5
        if self.direction == "up":
            self.triangle_points = [
                cx, cy + half_h,
                cx - half_w, cy - half_h,
                cx + half_w, cy - half_h,
            ]
        elif self.direction == "down":
            self.triangle_points = [
                cx, cy - half_h,
                cx - half_w, cy + half_h,
                cx + half_w, cy + half_h,
            ]
        else:
            self.triangle_points = [0, 0, 0, 0, 0, 0]
