
from kivy.uix.recycleview.views import RecycleDataViewBehavior as KVRecycleDataViewBehavior
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelOneLine
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivymd.uix.list import OneLineListItem as MDOneLineListItem
from kivymd.uix.scrollview import MDScrollView
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout

import re

class Mask2CustomHeader(MDExpansionPanelOneLine):
    pass

class Mask2Item(KVBoxLayout, KVRecycleDataViewBehavior):
    active = KVBooleanProperty(True)
    text = KVStringProperty('')
    is_composit = KVBooleanProperty(False)
    indent = KVNumericProperty(0)
    mask_op = KVStringProperty('')
    mask_ref = KVObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos):
            self.set_active()
            return True
        return False

    def set_active(self):
        if self.mask_ref and self.mask_ref.editor:
            self.mask_ref.editor.set_active_mask(self.mask_ref)
            # リスト更新はイベント経由で行われるが、即時反映のため更新呼ぶ？
            # イベントが飛ぶはずなので不要だが、active状態だけの変更だと構造変更イベントは飛ばないかも？
            # active切り替えは構造変更ではない。
            # MaskEditor.set_active_mask でイベント飛ばしてない。
            # Mask2ContentPanel側で active_mask の変更を監視する必要があるか、
            # あるいはリフレッシュを呼ぶか。
            # ここではリフレッシュを呼ぶのが確実。
            self.parent.parent.parent.refresh_list() # 構造依存...

    def add_child(self, maskop):
        if self.mask_ref:
            # MaskEditor2で実装していた show_add_mask_popup ロジックをここに持ってくるか、委譲する
            # Mask2ContentPanelに委譲するのが良い
            # panel = self.parent.parent.parent
            # panel.show_add_popup(self.mask_ref, maskop)
            self.show_add_mask_popup(maskop)

    def delete_item(self):
        if self.mask_ref:
            self.mask_ref.editor.del_mask(self.mask_ref)
            self.mask_ref.editor.dispatch('on_structure_change')

    def show_add_mask_popup(self, maskop):
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from functools import partial
        import widgets.mask_editor2 as me2

        content = BoxLayout(orientation='vertical')
        popup = Popup(title=f'Select Mask Type ({maskop})', content=content, size_hint=(0.5, 0.8))
        
        types = [
            ('Circle', me2.MaskType.CIRCULAR),
            ('Line', me2.MaskType.GRADIENT),
            ('Full', me2.MaskType.FULL),
            ('Draw', me2.MaskType.FREEDRAW),
            ('Segment', me2.MaskType.SEGMENT),
            ('Depth', me2.MaskType.DEPTHMAP),
            ('Face', me2.MaskType.FACE),
            ('Scene', me2.MaskType.SCENE)
        ]

        for name, type_key in types:
            btn = Button(text=name)
            btn.bind(on_release=partial(self._add_child_mask, type_key, maskop, popup))
            content.add_widget(btn)
        
        popup.open()

    def _add_child_mask(self, type_key, maskop, popup, instance):
        popup.dismiss()
        self.mask_ref.editor.set_active_mask(None) # 一旦アクティブなし
        new_mask = self.mask_ref.editor.add_mask(type_key, maskop,self.mask_ref.editor.get_mask_list().index(self.mask_ref)+1)
        self.mask_ref.add_mask(new_mask, maskop) # CompositMask.add_mask -> dispatch event

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    pass

class Mask2ContentPanel(KVBoxLayout):    
    def __init__(self, mask2_editor, **kwargs):
        self.editor = mask2_editor
        super().__init__(**kwargs)
        self.editor.bind(on_structure_change=self.refresh_list)
        self.editor.bind(active_mask=self.refresh_list)

    def on_kv_post(self, *args, **kwargs):
        super(Mask2ContentPanel, self).on_kv_post(*args, **kwargs)
        self.ids['mask2_rv'].data = []
        self.refresh_list()

    def set_active_index(self, index):
        pass
    
    def refresh_list(self, *args):
        # レイヤーリストを再構築
        data = []
        # mask_listは下から順に描画されるが、リスト表示は上が手前（リストの上）に来るのが一般的か、あるいは逆か？
        # Photoshop等は上が手前（後から描画されるもの）。なので reversed(mask_list) の順でリストに追加する。
        # mask_editor2.pyでは reversed(self.mask_list) でループして処理している（下から順に描画処理等）。
        # ここではリストの上を「一番上のレイヤー」とするなら、reversed する。

        for mask in self.editor.get_mask_list():
            # CompositMask (Root)
            is_composit = mask.is_composit()
            if is_composit:
                maskop = 'Composit'
            else:
                composit_mask = self.editor.find_composit_mask(mask)
                if composit_mask:
                    maskop = composit_mask.find_mask_op(mask)
                else:
                    continue # まだ子供に追加されてない

            data.append({
                'text': mask.name,
                'is_composit': is_composit,
                'indent': 0 if is_composit else 1,
                'mask_ref': mask,
                'mask_op': maskop,
                'active': mask == self.editor.active_mask
            })
            """
            # Children
            if hasattr(mask, 'mask_list'):
                # mask_listも描画順に入っている？
                # CompositMask.mask_listは append で追加されている。
                # 描画順（合成順）はリスト順序通りなら、リストの後ろが「上」に来る。
                # なのでUIリストとしては reversed して表示する。
                for child, maskop in reversed(mask.mask_list):
                    data.append({
                        'text': child.name,
                        'is_composit': False,
                        'indent': 1,
                        'mask_ref': child,
                        'mask_op': maskop,
                        'active': child == self.editor.active_mask
                    })
            """
        self.ids['mask2_rv'].data = data

    def add_layer(self):
        self.editor.add_composit_mask(None)

def create_mask2_content_panel(mask2_editor):
    return Mask2ContentPanel(mask2_editor)

