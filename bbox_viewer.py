
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, PopMatrix, PushMatrix, PushMatrix, PopMatrix
from kivy.core.window import Window


class BoundingBoxViewer(Widget):
    def __init__(self, size=(800, 600), initial_view=(0, 0, 1000, 800, 1.0), on_delete=None, **kwargs):
        super().__init__(**kwargs)

        self.pos_hint = {'x': 0, 'top': 1}
        
        # 最大表示サイズ（ビュー表示用）
        self.max_display_width, self.max_display_height = size
        
        # 表示範囲とスケール
        self.view_x, self.view_y, self.view_w, self.view_h, self.scale = initial_view
        
        # バウンディングボックス
        self.boxes = []  # [(x, y, w, h), ...]
        self.selected_index = None
        self.overlapping_indices = []  # 重複しているボックスのインデックス
        self.overlap_cycle_index = 0  # 重複時の選択サイクル
        
        # コールバック
        self.on_delete_callback = on_delete
        
        # 描画スタイル設定（カスタマイズ可能）
        self.normal_color = (1, 0, 0, 1)  # 赤
        self.selected_color = (0, 1, 0, 1)  # 緑
        self.line_width = 2
        self.selected_line_width = 3
        
        # イベントバインド
        self.bind(size=self.on_size_change)
        self.bind(pos=self.on_pos_change)

        # キーボードイベントをバインド
        Window.bind(on_key_down=self.on_key_down)

    def __del__(self):
        Window.unbind(on_key_down=self.on_key_down)

        
    def set_boxes(self, boxes):
        """バウンディングボックスのリストを設定"""
        self.boxes = list(boxes)
        self.selected_index = None
        self.overlapping_indices = []
        self.overlap_cycle_index = 0
        self.redraw()
        
    def set_view(self, x, y, w, h, scale):
        """表示範囲とスケールを設定"""
        self.view_x, self.view_y, self.view_w, self.view_h, self.scale = x, y, w, h, scale
        self._check_selection_visibility()
        self.redraw()
        
    def set_style(self, normal_color=None, selected_color=None, line_width=None, selected_line_width=None):
        """描画スタイルを設定"""
        if normal_color:
            self.normal_color = normal_color
        if selected_color:
            self.selected_color = selected_color
        if line_width:
            self.line_width = line_width
        if selected_line_width:
            self.selected_line_width = selected_line_width
        self.redraw()
        
    def _check_selection_visibility(self):
        """選択されたボックスが表示範囲内にあるかチェック"""
        if self.selected_index is None or self.selected_index >= len(self.boxes):
            return
            
        box = self.boxes[self.selected_index]
        box_x, box_y, box_w, box_h = box
        
        # ボックスが完全に表示範囲外かチェック
        if (box_x + box_w < self.view_x or 
            box_x > self.view_x + self.view_w or
            box_y + box_h < self.view_y or 
            box_y > self.view_y + self.view_h):
            self.selected_index = None
            self.overlapping_indices = []
            self.overlap_cycle_index = 0
            
    def _calculate_display_area(self):
        """実際の表示エリアとオフセットを計算（パディングを考慮）"""
        # ビューをスケールした時のサイズ
        scaled_view_w = self.view_w * self.scale
        scaled_view_h = self.view_h * self.scale
        
        # 最大表示サイズと比較してパディングを計算
        if scaled_view_w <= self.max_display_width:
            display_width = scaled_view_w
            offset_x = (self.width - display_width) / 2
        else:
            display_width = self.max_display_width
            offset_x = (self.width - display_width) / 2
            
        if scaled_view_h <= self.max_display_height:
            display_height = scaled_view_h
            offset_y = (self.height - display_height) / 2
        else:
            display_height = self.max_display_height
            offset_y = (self.height - display_height) / 2
            
        return display_width, display_height, offset_x, offset_y
        
    def _world_to_display(self, world_x, world_y):
        """ワールド座標を表示座標に変換（パディング考慮）"""
        display_width, display_height, offset_x, offset_y = self._calculate_display_area()
        
        # ワールド座標をビュー相対座標に変換
        rel_x = (world_x - self.view_x) * self.scale
        rel_y = (world_y - self.view_y) * self.scale
        
        # 表示座標に変換（パディング追加）
        display_x = rel_x + offset_x
        display_y = rel_y + offset_y
        
        return display_x, display_y
        
    def _display_to_world(self, display_x, display_y):
        """表示座標をワールド座標に変換（パディング考慮）"""
        display_width, display_height, offset_x, offset_y = self._calculate_display_area()
        
        # パディングを除去してビュー相対座標に変換
        rel_x = display_x - offset_x
        rel_y = display_y - offset_y
        
        # ワールド座標に変換
        world_x = rel_x / self.scale + self.view_x
        world_y = rel_y / self.scale + self.view_y
        
        return world_x, world_y
        
    def _point_in_box(self, point_x, point_y, box):
        """点がボックス内にあるかチェック"""
        box_x, box_y, box_w, box_h = box
        return (box_x <= point_x <= box_x + box_w and 
                box_y <= point_y <= box_y + box_h)
                
    def _find_overlapping_boxes(self, world_x, world_y):
        """指定した点に重なるボックスのインデックスを取得（後方から前方の順）"""
        overlapping = []
        for i in range(len(self.boxes) - 1, -1, -1):  # 後方から検索（上位優先）
            if self._point_in_box(world_x, world_y, self.boxes[i]):
                overlapping.append(i)
        return overlapping
        
    def redraw(self):
        """画面を再描画"""
        self.canvas.clear()
        
        with self.canvas:
            PushMatrix()
            for i, box in enumerate(self.boxes):
                box_x, box_y, box_w, box_h = box
                
                # 表示範囲内のボックスのみ描画
                if (box_x + box_w >= self.view_x and box_x <= self.view_x + self.view_w and
                    box_y + box_h >= self.view_y and box_y <= self.view_y + self.view_h):
                    
                    # ボックスの四隅を表示座標に変換
                    display_x1, display_y1 = self._world_to_display(box_x, box_y)
                    display_x2, display_y2 = self._world_to_display(box_x + box_w, box_y + box_h)
                    
                    # 実際の描画位置を計算（ウィジェット位置 + Y軸反転）
                    draw_x = self.x + display_x1
                    draw_y = self.y + self.height - display_y2  # Y軸反転
                    draw_w = display_x2 - display_x1
                    draw_h = display_y2 - display_y1
                    
                    # 選択状態に応じて色と線幅を設定
                    if i == self.selected_index:
                        Color(*self.selected_color)
                        width = self.selected_line_width
                    else:
                        Color(*self.normal_color)
                        width = self.line_width
                    
                    # 矩形の枠を描画
                    Line(rectangle=(draw_x, draw_y, draw_w, draw_h), width=width)
            PopMatrix()

    def on_touch_down(self, touch):
        """マウスクリック処理"""
        if self.collide_point(*touch.pos):
            # ウィジェット内の相対座標を計算
            relative_x = touch.x - self.x
            relative_y = touch.y - self.y
            
            # Kivyの座標系（左下原点）を表示座標系（左上原点）に変換
            display_x = relative_x
            display_y = self.height - relative_y  # Y軸反転
            
            # 表示座標をワールド座標に変換
            world_x, world_y = self._display_to_world(display_x, display_y)
            
            # デバッグ出力
            print(f"Touch: ({touch.x}, {touch.y})")
            print(f"Relative: ({relative_x}, {relative_y})")
            print(f"Display: ({display_x}, {display_y})")
            print(f"World: ({world_x}, {world_y})")
            print(f"Widget pos/size: ({self.x}, {self.y}) / ({self.width}, {self.height})")
            
            # 重なるボックスを検索
            overlapping = self._find_overlapping_boxes(world_x, world_y)
            
            if overlapping:
                if overlapping != self.overlapping_indices:
                    # 新しい場所をクリック
                    self.overlapping_indices = overlapping
                    self.overlap_cycle_index = 0
                    self.selected_index = overlapping[0]
                else:
                    # 同じ場所を再クリック - 次のボックスに切り替え
                    self.overlap_cycle_index = (self.overlap_cycle_index + 1) % len(overlapping)
                    self.selected_index = overlapping[self.overlap_cycle_index]
            else:
                # 何もない場所をクリック - 選択解除
                self.selected_index = None
                self.overlapping_indices = []
                self.overlap_cycle_index = 0
                
            self.redraw()
            return True
        return super().on_touch_down(touch)
        
    def on_key_down(self, window, key, scancode, codepoint, modifier):
        """キー入力処理"""
        if key == 8:  # Backspace
            if self.selected_index is not None and self.selected_index < len(self.boxes):
                deleted_box = self.boxes[self.selected_index]
                deleted_index = self.selected_index
                
                # ボックスを削除
                del self.boxes[self.selected_index]
                
                # 選択状態をリセット
                self.selected_index = None
                self.overlapping_indices = []
                self.overlap_cycle_index = 0
                
                # コールバック呼び出し
                if self.on_delete_callback:
                    self.on_delete_callback(deleted_index, deleted_box)
                    
                self.redraw()
                
    def on_size_change(self, instance, value):
        """ウィンドウサイズ変更時の処理"""
        self.redraw()
        
    def on_pos_change(self, instance, value):
        """位置変更時の処理"""
        self.redraw()


class BoundingBoxApp(App):
    def build(self):
        from kivy.uix.floatlayout import FloatLayout
        from kivy.uix.button import Button
        
        def on_delete(index, box):
            print(f"Deleted box {index}: {box}")
            
        # テスト用のバウンディングボックス
        test_boxes = [
            (50, 50, 100, 80),
            (120, 70, 90, 60),
            (80, 80, 70, 50),  # 重複するボックス
            (200, 150, 120, 90),
            (180, 200, 80, 70),
        ]
        
        # FloatLayoutを作成してテスト
        layout = FloatLayout()
        
        viewer = BoundingBoxViewer(
            size=(400, 300),  # ビューの最大表示サイズ
            initial_view=(0, 0, 400, 300, 1.0),
            on_delete=on_delete
        )
        # pos_hintとsize_hintを使用して位置とサイズを指定
        viewer.pos_hint = {'x': 0.1, 'y': 0.2}  # 画面の10%,20%の位置
        viewer.size_hint = (0.8, 0.7)  # 画面の80%,70%のサイズ
        
        viewer.set_boxes(test_boxes)
        
        # テスト用ボタンを追加
        btn = Button(
            text='Change View',
            size_hint=(None, None),
            size=(100, 50),
            pos=(10, 10)
        )
        
        def change_view_button(instance):
            viewer.set_view(100, 100, 300, 200, 1.5)
            print("View changed to: (100, 100, 300, 200, 1.5)")
            print(f"Viewer pos: {viewer.pos}, size: {viewer.size}")
            
        btn.bind(on_press=change_view_button)
        
        layout.add_widget(viewer)
        layout.add_widget(btn)
        
        
        return layout


if __name__ == '__main__':
    BoundingBoxApp().run()