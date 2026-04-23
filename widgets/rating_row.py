"""
5 段レーティング。slot 1～5 をタップ。
EXIF 左パネルでは ref_width/ref_height を kv で与え、traverse で寸法を確保。
サムネ側は ref_*=0 のまま親レイアウトに幅・高さを任せる。
表示は ASCII（* / 空スロットは -）。MDLabel で KivyMD テーマ上も文字色を明示する。
"""
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel

from kivy.properties import NumericProperty, ObjectProperty, BooleanProperty


class RatingRow(MDBoxLayout):
    """左詰めの 5 スロット。Viewer または左ペインの main に委譲。"""

    rating = NumericProperty(0)
    card_index = NumericProperty(-1)
    ctx = ObjectProperty(None, allownone=True)
    exif_pane = BooleanProperty(False)
    # kvutils.traverse_widget が数値を見るため Property 必須（0 のときはスキップ）
    ref_width = NumericProperty(0)
    ref_height = NumericProperty(0)

    def __init__(self, **kwargs):
        # super 内の on_kv_post が _apply_display を呼ぶため、先に用意する
        self._labels = []
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.spacing = 0
        self.padding = (0, 0, 0, 0)
        self.md_bg_color = [0, 0, 0, 0]
        for i in range(5):
            lb = MDLabel(
                text="-",
                font_style="Body2",
                font_size="12sp",
                theme_text_color="Custom",
                text_color=(0.92, 0.92, 0.92, 1.0),
                size_hint_x=0.2,
                halign="center",
                valign="middle",
            )
            lb.bind(size=lb.setter("text_size"))
            slot = i + 1

            def _on_touch(w, touch, s=slot):
                if w.collide_point(*touch.pos) and touch.button == "left" and not touch.is_mouse_scrolling:
                    self._dispatch_slot(s)
                    return True
                return False

            lb.bind(on_touch_down=_on_touch)
            self._labels.append(lb)
            self.add_widget(lb)
        self.bind(rating=self._apply_display)
        self._apply_display()

    def on_kv_post(self, base_widget):
        self._apply_display()

    def _dispatch_slot(self, slot: int):
        if not self.ctx:
            return
        if self.exif_pane:
            self.ctx.apply_exif_pane_rating_slot(slot)
        else:
            self.ctx.on_rating_slot(self.card_index, slot)

    def _apply_display(self, *_a):
        labels = self._labels
        if not labels:
            return
        r = int(self.rating or 0)
        for i, lb in enumerate(labels):
            lb.text = "*" if i < r else "-"
