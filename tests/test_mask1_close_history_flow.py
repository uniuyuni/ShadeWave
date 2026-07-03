"""mask1 (inpaint/patchmatch_inpaint) を閉じる操作の履歴登録に関する回帰テスト。

MainWidget._cancel_mask1_mode はタブ切替(on_current_tab)や画像切替(on_select)から
無条件に呼ばれ、mask1 が開いていれば primary_param を直接書き換えて閉じる。以前は
begin/end_history_effect_ctrl で囲んでいなかったため、この「閉じる」変化が履歴に
一切残らず、実際の param 状態と履歴スタックが食い違って以降の undo/redo が壊れて
いた(Make mask ボタンの状態もおかしくなる)。

MainWidget は kivy App に依存しヘッドレスで import できないため、
_cancel_mask1_mode 本体は AST でソース抽出して exec で検証する。
"""

import ast
import pathlib
import sys
import textwrap
import unittest
from types import SimpleNamespace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_method_source(path, class_name, function_name):
    source_text = path.read_text()
    tree = ast.parse(source_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source_text, child)
    return None


def _make_cancel_mask1_mode():
    src = _load_method_source(MAIN_PATH, "MainWidget", "_cancel_mask1_mode")
    assert src is not None
    ns = {}
    exec(textwrap.dedent(src), ns)
    return ns["_cancel_mask1_mode"]


class _Switch:
    def __init__(self, state="normal"):
        self.state = state


class _StubWidget:
    def __init__(self, *, inpaint_open=False, patchmatch_open=False, pre_existing_op=None):
        self.primary_param = {
            'inpaint': inpaint_open,
            'inpaint_predict': inpaint_open,
            'inpaint_mask_list': (["mask"] if inpaint_open else []),
            'patchmatch_inpaint': patchmatch_open,
            'patchmatch_inpaint_predict': patchmatch_open,
            'patchmatch_inpaint_mask_list': (["mask"] if patchmatch_open else []),
        }
        self.ids = {
            'switch_inpaint': _Switch("down" if inpaint_open else "normal"),
            'button_inpaint_predict': _Switch(),
            'switch_patchmatch_inpaint': _Switch("down" if patchmatch_open else "normal"),
            'button_patchmatch_inpaint_predict': _Switch(),
        }
        self.current_op = pre_existing_op
        self.begin_calls = []
        self.end_calls = []
        self.removed_editors = []
        self.exited_sources = []

    def begin_history_effect_ctrl(self, lv, effect, subname=None):
        self.begin_calls.append((lv, effect, subname))
        self.current_op = object()
        return True

    def end_history_effect_ctrl(self, lv, effect, subname=None):
        self.end_calls.append((lv, effect, subname))
        self.current_op = None

    def _remove_mask1_editor_for_effect(self, effect_name):
        self.removed_editors.append(effect_name)

    def exit_mask1_full_preview_mode(self, source):
        self.exited_sources.append(source)

    def _image_interaction_ready(self):
        return False

    def start_draw_image_and_crop(self, imgset):
        pass


class CancelMask1ModeHistoryTest(unittest.TestCase):
    def test_closing_open_inpaint_mask_is_wrapped_in_history(self):
        cancel = _make_cancel_mask1_mode()
        widget = _StubWidget(inpaint_open=True)

        cancel(widget, sources=("inpaint",))

        self.assertEqual(widget.begin_calls, [(0, 'inpaint', None)])
        self.assertEqual(widget.end_calls, [(0, 'inpaint', None)])
        self.assertFalse(widget.primary_param['inpaint'])
        self.assertEqual(widget.primary_param['inpaint_mask_list'], [])

    def test_closing_open_patchmatch_inpaint_mask_is_wrapped_in_history(self):
        cancel = _make_cancel_mask1_mode()
        widget = _StubWidget(patchmatch_open=True)

        cancel(widget, sources=("patchmatch_inpaint",))

        self.assertEqual(widget.begin_calls, [(0, 'patchmatch_inpaint', None)])
        self.assertEqual(widget.end_calls, [(0, 'patchmatch_inpaint', None)])

    def test_noop_when_nothing_was_open(self):
        # タブ切替のたびに無条件で呼ばれるので、何も開いていないときに空の履歴
        # エントリを作ってはいけない。
        cancel = _make_cancel_mask1_mode()
        widget = _StubWidget(inpaint_open=False, patchmatch_open=False)

        cancel(widget)

        self.assertEqual(widget.begin_calls, [])
        self.assertEqual(widget.end_calls, [])

    def test_does_not_clobber_an_already_active_history_op(self):
        # begin_history_effect_ctrl は self.current_op を無条件に上書きするので、
        # 既に別の begin/end ペアの内側から呼ばれた場合はネストして開始してはいけない
        # (呼び出し元の pending op を握りつぶしてしまうため)。
        outer_op = object()
        cancel = _make_cancel_mask1_mode()
        widget = _StubWidget(inpaint_open=True, pre_existing_op=outer_op)

        cancel(widget, sources=("inpaint",))

        self.assertEqual(widget.begin_calls, [])
        self.assertEqual(widget.end_calls, [])
        # State はそれでも正しく閉じられる。
        self.assertFalse(widget.primary_param['inpaint'])
        self.assertIs(widget.current_op, outer_op)


if __name__ == "__main__":
    unittest.main()
