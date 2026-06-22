from kivy.lang import Builder
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ColorProperty, ObjectProperty, NumericProperty
from kivy.animation import Animation
from kivy.metrics import dp

class CompactSwitch(ButtonBehavior, Widget):
    active = BooleanProperty(False)
    
    # 色設定 (RGBA)
    track_color_active = ColorProperty([0.13, 0.23, 0.74, 1])   # ONの時の背景色 (黒)
    track_color_inactive = ColorProperty([0.2, 0.2, 0.2, 1]) # OFFの時の背景色 (グレー)
    thumb_color = ColorProperty([1, 1, 1, 1])                # 丸い部分の色 (白)

    # コールバック用プロパティ
    pre_active = BooleanProperty(active)
    post_active = BooleanProperty(active)

    # 内部制御用プロパティ
    _thumb_x = NumericProperty(0)
    _track_color = ColorProperty([0, 0, 0, 0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(active=self._update_state)
        # 初期状態の設定（少し遅延させてサイズ確定後に実行するとより安全ですが、簡易的にここで設定）
        self._track_color = self.track_color_inactive
        self._thumb_x = dp(2)

    def on_size(self, *args):
        """サイズ変更時にサムの位置を再計算"""
        self._update_thumb_position(animate=False)

    def on_press(self):
        """タッチされた時の処理"""
        # 1. PRE-CALLBACK (変更前)
        self.pre_active = not self.pre_active

        # 2. STATE CHANGE
        self.active = not self.active

        # 3. POST-CALLBACK (変更後)
        self.post_active = not self.post_active

    def _update_state(self, instance, value):
        """activeプロパティが変更されたら色と位置を更新"""
        target_color = self.track_color_active if value else self.track_color_inactive
        
        # 色のアニメーション
        anim = Animation(_track_color=target_color, duration=0.2)
        anim.start(self)
        
        # 位置のアニメーション
        self._update_thumb_position(animate=True)

    def _update_thumb_position(self, animate=True):
        """サムのX座標を計算して移動"""
        if self.active:
            target_x = self.width - self.height + 4
        else:
            target_x = 4

        if animate:
            anim = Animation(_thumb_x=target_x, duration=0.15, t='out_quad')
            anim.start(self)
        else:
            self._thumb_x = target_x
