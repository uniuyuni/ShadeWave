"""Liquify Reset 回帰テスト。

- DistortionEffect の distortion_recorded デフォルト([])が _get_param のキャッシュで
  共有ミュータブルリストにならないこと(ペインタが in-place append するため、共有されると
  ALL Reset(param.clear())後もデフォルト経路からストロークが復活する)。
"""

import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import effects


class DistortionRecordedDefaultIsolationTest(unittest.TestCase):
    def test_default_recorded_list_is_not_shared(self):
        effect = effects.DistortionEffect()
        param = {}

        # ペインタ開時と同じ経路: キー未設定の param からデフォルトを参照で受け取る
        recorded = effect._get_param(param, 'distortion_recorded')
        self.assertEqual(recorded, [])
        recorded.append({'stroke': 1})  # ペインタの in-place append を模擬

        # ALL Reset 相当: param にキーが無い状態で再取得してもストロークが復活しない
        self.assertEqual(effect._get_param({}, 'distortion_recorded'), [])
        self.assertEqual(effect._get_param(param, 'distortion_recorded'), [])

    def test_default_recorded_not_shared_across_calls(self):
        effect = effects.DistortionEffect()
        a = effect._get_param({}, 'distortion_recorded')
        b = effect._get_param({}, 'distortion_recorded')
        self.assertIsNot(a, b)

    def test_param_value_still_returned_by_reference(self):
        # param にキーがあるときは従来通り参照が返る(ペインタ append の永続化に必要)
        effect = effects.DistortionEffect()
        strokes = [{'stroke': 1}]
        param = {'distortion_recorded': strokes}
        self.assertIs(effect._get_param(param, 'distortion_recorded'), strokes)


if __name__ == '__main__':
    unittest.main()
