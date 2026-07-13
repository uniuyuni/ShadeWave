
from kivy.uix.recycleview.views import RecycleDataViewBehavior as KVRecycleDataViewBehavior
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty
from kivy.uix.recycleboxlayout import RecycleBoxLayout as KVRecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.core.window import Window as KVWindow
from kivy.clock import Clock
from utils import dialogutils, kvutils
from history import Operation, get_history_ctrl

import logging
import re


# マスク作成メニューの 'Copy' はここだけの UI 専用アクション(MaskType には追加しない)
_COPY_ACTION = 'copy'

# マスクを選択/作成した直後に表示する操作ヒントの表示秒数(この後 自動的に消す)。
_MASK_HINT_SECONDS = 5.0

# マスククラス -> 1行の操作ヒント。widgets.mask_editor2 との循環 import を避けるため
# 初回参照時に遅延構築する(Composit / 未知タイプはヒントなし)。
_MASK_HINTS = None


def _mask_hint_for(mask):
    global _MASK_HINTS
    if _MASK_HINTS is None:
        import widgets.mask_editor2 as me2
        # 作成直後に最初に行う操作だけを短く示す。点設定系(Full/Depth/Face/TargetText)は
        # まず画像をタップして点を置く必要があるため、その手順を明示する。
        _MASK_HINTS = {
            me2.CircularGradientMask: "Drag to draw an ellipse",
            me2.GradientMask: "Drag to draw a gradient line",
            me2.FullMask: "Tap the image to place a point",
            me2.FreeDrawMask: "Paint to draw the mask",
            me2.PolylineMask: "Click to add points",
            me2.SegmentMask: "Drag a box around the subject",
            me2.DepthMapMask: "Tap to place a point, then set the depth range",
            me2.FaceMask: "Tap to place a point, then pick face parts",
            me2.TargetTextMask: "Tap to place a point, then type the target",
        }
    if mask is None:
        return ""
    return _MASK_HINTS.get(type(mask), "")

_ADD_MASK_POPUP_WIDTH_REF = 300
_ADD_MASK_POPUP_BUTTON_HEIGHT_REF = 36
_ADD_MASK_POPUP_PADDING_REF = 5
_ADD_MASK_POPUP_SPACING_REF = 5
_ADD_MASK_POPUP_CHROME_HEIGHT_REF = 76
_ADD_MASK_POPUP_MIN_HEIGHT_REF = 420
_ADD_MASK_POPUP_WINDOW_MARGIN_REF = 48


def _add_mask_popup_ref_height(item_count):
    content_height = (
        _ADD_MASK_POPUP_PADDING_REF * 2
        + _ADD_MASK_POPUP_BUTTON_HEIGHT_REF * item_count
        + _ADD_MASK_POPUP_SPACING_REF * max(0, item_count - 1)
    )
    desired_height = max(
        _ADD_MASK_POPUP_MIN_HEIGHT_REF,
        content_height + _ADD_MASK_POPUP_CHROME_HEIGHT_REF,
    )
    try:
        scale = max(1e-6, kvutils.dpi_scale_height(1))
        max_height = max(
            _ADD_MASK_POPUP_MIN_HEIGHT_REF,
            (float(KVWindow.height or 0) / scale) - _ADD_MASK_POPUP_WINDOW_MARGIN_REF,
        )
        return min(desired_height, max_height)
    except Exception:
        return desired_height

class Mask2CustomHeader(KVBoxLayout):
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
            panel = self.parent.parent.parent
            if getattr(panel, 'copy_mode', False):
                # コピー元選択モード中は行タップを既存の active 切り替えより先に横取りする
                panel.handle_copy_selection(self.mask_ref)
                return True
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

    def show_add_mask_popup(self, maskop):
        from kivy.uix.popup import Popup as KVPopup
        from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
        from kivy.uix.button import Button as KVButton
        from functools import partial
        import widgets.mask_editor2 as me2

        # 別の +/- 押下で以前のコピー選択モードは解除する
        self.parent.parent.parent.cancel_copy_mode()

        types = [
            ('Circle', me2.MaskType.CIRCULAR),
            ('Line', me2.MaskType.GRADIENT),
            ('Full', me2.MaskType.FULL),
            ('Draw', me2.MaskType.FREEDRAW),
            ('Polyline', me2.MaskType.POLYLINE),
            ('Segment', me2.MaskType.SEGMENT),
            ('Depth', me2.MaskType.DEPTHMAP),
            ('Face', me2.MaskType.FACE),
            ('Target Text', me2.MaskType.TARGET_TEXT),
            ('Copy', _COPY_ACTION),
        ]

        content = KVBoxLayout(orientation='vertical')
        content.ref_layout_padding = _ADD_MASK_POPUP_PADDING_REF
        content.ref_layout_spacing = _ADD_MASK_POPUP_SPACING_REF
        popup = KVPopup(title=f'Select Mask Type ({maskop})', content=content, size_hint=(None, None))
        popup.ref_width = _ADD_MASK_POPUP_WIDTH_REF

        def _fit_popup_height():
            popup.ref_height = _add_mask_popup_ref_height(len(types))

        _fit_popup_height()
        dialogutils.install_ref_scaling(popup, on_rescale=_fit_popup_height)

        for name, type_key in types:
            btn = KVButton(text=name, size_hint_y=None)
            btn.ref_height = _ADD_MASK_POPUP_BUTTON_HEIGHT_REF
            btn.bind(on_release=partial(self._add_child_mask, type_key, maskop, popup))
            content.add_widget(btn)
        
        popup.open()

    def _add_child_mask(self, type_key, maskop, popup, instance):
        popup.dismiss()
        if type_key == _COPY_ACTION:
            # コピー元選択モードへ移行(対象コンポジットは押した +/- のあった行自身)
            self.parent.parent.parent.start_copy_mode(self.mask_ref, maskop)
            return
        self.mask_ref.editor.set_active_mask(None) # 一旦アクティブなし
        new_mask = self.mask_ref.editor.add_mask(type_key, maskop,self.mask_ref.editor.get_mask_list().index(self.mask_ref)+1)
        self.mask_ref.add_mask(new_mask, maskop) # CompositMask.add_mask -> dispatch event

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, KVRecycleBoxLayout):
    pass

class Mask2ContentPanel(KVBoxLayout):
    # コピー元選択モード中かどうか(kv からヒント表示の bool として参照)
    copy_mode = KVBooleanProperty(False)
    # アクティブなマスクの操作ヒント(kv から表示。空なら非表示。copy_mode が優先)
    active_hint = KVStringProperty("")

    def __init__(self, mask2_editor, **kwargs):
        self.editor = mask2_editor
        self._copy_target_composit = None
        self._copy_target_maskop = None
        self._hint_clock = None
        super().__init__(**kwargs)
        self.editor.bind(on_structure_change=self.refresh_list)
        self.editor.bind(active_mask=self.refresh_list)
        self.editor.bind(active_mask=self._update_active_hint)
        KVWindow.bind(on_key_down=self._on_key_down)

    def _update_active_hint(self, *args):
        # マスクを選択/作成したときにそのタイプの操作ヒントを表示し、数秒後に自動的に消す。
        hint = _mask_hint_for(self.editor.active_mask)
        self.active_hint = hint
        if self._hint_clock is not None:
            self._hint_clock.cancel()
            self._hint_clock = None
        if hint:
            self._hint_clock = Clock.schedule_once(self._clear_active_hint, _MASK_HINT_SECONDS)

    def _clear_active_hint(self, *args):
        self.active_hint = ""
        self._hint_clock = None

    def on_kv_post(self, *args, **kwargs):
        super(Mask2ContentPanel, self).on_kv_post(*args, **kwargs)
        self.ids['mask2_rv'].data = []
        self.refresh_list()
        self.disabled = True

    def set_active_index(self, index):
        pass

    def _on_key_down(self, window, key, scancode, codepoint, modifier):
        if self.copy_mode and key == 27:  # ESC
            self.cancel_copy_mode()
            return True
        return False

    def on_touch_down(self, touch):
        if self.copy_mode and self.collide_point(*touch.pos):
            # 行タップは Mask2Item.on_touch_down 側で処理されるので、ここに来るのは
            # パネル内の行以外(余白等)のタップ。コピー選択モードを解除する。
            handled = super().on_touch_down(touch)
            if not handled:
                self.cancel_copy_mode()
            return handled
        return super().on_touch_down(touch)

    def start_copy_mode(self, target_composit, maskop):
        self._copy_target_composit = target_composit
        self._copy_target_maskop = maskop
        self.copy_mode = True

    def cancel_copy_mode(self):
        self._copy_target_composit = None
        self._copy_target_maskop = None
        self.copy_mode = False

    def handle_copy_selection(self, src_mask):
        target_composit = self._copy_target_composit
        maskop = self._copy_target_maskop
        self.cancel_copy_mode()

        if src_mask is None or target_composit is None:
            return
        if target_composit not in self.editor.get_mask_list():
            return

        # マスク追加と同じ経路(set_backup_all/set_update_all)で 1 操作 = 1 undo として記録する。
        # コンポジット展開で複数マスクが増えても、この方式なら全体が 1 操作で戻る。
        main_widget = get_history_ctrl()
        op = Operation(type="All")
        op.set_backup_all(main_widget.primary_param, self.editor)

        try:
            if src_mask.is_composit():
                created = self.editor.copy_composit_children_into(src_mask, target_composit)
            else:
                new_mask = self.editor.copy_mask_into(src_mask, target_composit, maskop)
                created = [new_mask] if new_mask is not None else []
        except Exception:
            logging.exception("mask copy failed")
            created = []

        if created:
            self.editor.set_active_mask(created[-1])

        if op.set_update_all(main_widget.primary_param, self.editor, "Copy Mask") is not None:
            main_widget.history.append(op)
            main_widget.history_panel.set_history(main_widget.history)

        self.refresh_list()

    def refresh_list(self, *args):
        # レイヤーリストを再構築
        data = []
        # mask_listは下から順に描画されるが、リスト表示は上が手前（リストの上）に来るのが一般的か、あるいは逆か？
        # Photoshop等は上が手前（後から描画されるもの）。なので reversed(mask_list) の順でリストに追加する。
        # mask_editor2.pyでは reversed(self.mask_list) でループして処理している（下から順に描画処理等）。
        # ここではリストの上を「一番上のレイヤー」とするなら、reversed する。

        active_mask = self.editor.active_mask
        created_mask = self.editor.get_created_mask()
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
                'active': mask == active_mask or mask == created_mask
            })
        self.ids['mask2_rv'].data = data

    def add_layer(self):
        self.editor.add_composit_mask(None)

def create_mask2_content_panel(mask2_editor):
    panel = Mask2ContentPanel(mask2_editor)
    panel.id = 'mask2_content_panel'
    return panel
