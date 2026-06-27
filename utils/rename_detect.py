"""watchfiles の変更バッチ(delete+add)から rename を best-effort で相関する純関数。

watchfiles は rename を「deleted(old) + added(new)」としてしか通知しない。
ここでは Kivy 等の重い依存を持たない純関数として相関ロジックを切り出し、単体テスト可能にする。
"""

from __future__ import annotations

import os


def detect_rename_pair(added, deleted, *, exists=os.path.exists):
    """rename とみなせる (old, new) を返す。該当しなければ None。

    Parameters
    ----------
    added, deleted : list[str]
        「可視画像にフィルタ済み」のパスリスト想定（呼び出し側で is_visible_image 済み）。
    exists : callable
        パス存在判定（テストで差し替え可能）。

    判定（best-effort・完全性は保証しない）:
      - added/deleted がそれぞれちょうど1件
      - 同一ディレクトリ
      - old が消えており(new でない)、new が存在する
    複数同時 rename やバッチをまたぐ rename は相関できず None（呼び出し側はフォールバック）。
    """
    if len(added) == 1 and len(deleted) == 1:
        new = added[0]
        old = deleted[0]
        if (os.path.dirname(old) == os.path.dirname(new)
                and not exists(old) and exists(new)):
            return (old, new)
    return None
