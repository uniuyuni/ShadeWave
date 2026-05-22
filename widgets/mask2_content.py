
from kivy.uix.recycleview.views import RecycleDataViewBehavior as KVRecycleDataViewBehavior
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelOneLine
from kivy.uix.recycleboxlayout import RecycleBoxLayout as KVRecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivymd.uix.list import OneLineListItem as MDOneLineListItem
from kivymd.uix.scrollview import MDScrollView
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from utils import dialogutils, kvutils

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
        ret = super().refresh_view_attrs(rv, index, data)
        
        # Force height update on refresh
        if self.ref_height:
             self.height = kvutils.dpi_scale_height(self.ref_height)

        return ret

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
        from kivy.uix.popup import Popup as KVPopup
        from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
        from kivy.uix.button import Button as KVButton
        from functools import partial
        import widgets.mask_editor2 as me2

        content = KVBoxLayout(orientation='vertical')
        content.ref_padding = 5
        content.ref_spacing = 5
        popup = KVPopup(title=f'Select Mask Type ({maskop})', content=content, size_hint=(None, None))
        popup.ref_width = 300
        popup.ref_height = 420
        dialogutils.install_ref_scaling(popup)
        
        types = [
            ('Circle', me2.MaskType.CIRCULAR),
            ('Line', me2.MaskType.GRADIENT),
            ('Full', me2.MaskType.FULL),
            ('Draw', me2.MaskType.FREEDRAW),
            ('Polyline', me2.MaskType.POLYLINE),
            ('Segment', me2.MaskType.SEGMENT),
            ('Depth', me2.MaskType.DEPTHMAP),
            ('Face', me2.MaskType.FACE),
            ('Target Text', me2.MaskType.TARGET_TEXT)
        ]

        for name, type_key in types:
            btn = KVButton(text=name, size_hint_y=None)
            btn.ref_height = 36
            btn.bind(on_release=partial(self._add_child_mask, type_key, maskop, popup))
            content.add_widget(btn)
        
        popup.open()

    def _add_child_mask(self, type_key, maskop, popup, instance):
        popup.dismiss()
        self.mask_ref.editor.set_active_mask(None) # 一旦アクティブなし
        new_mask = self.mask_ref.editor.add_mask(type_key, maskop,self.mask_ref.editor.get_mask_list().index(self.mask_ref)+1)
        self.mask_ref.add_mask(new_mask, maskop) # CompositMask.add_mask -> dispatch event

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, KVRecycleBoxLayout):
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
        self.disabled = True

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
                'text': mask.name if mask.name != "Composit" else "",
                'is_composit': is_composit,
                'indent': 0 if is_composit else 1,
                'mask_ref': mask,
                'mask_op': maskop,
                'active': mask == self.editor.active_mask
            })
        self.ids['mask2_rv'].data = data

    def add_layer(self):
        self.editor.add_composit_mask(None)

def create_mask2_content_panel(mask2_editor):
    panel = Mask2ContentPanel(mask2_editor)
    panel.id = 'mask2_content_panel'
    return panel
