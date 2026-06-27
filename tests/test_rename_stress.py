"""リネーム安全性のストレステスト。

対象:
1) rename 相関ヒューリスティック（utils.rename_detect.detect_rename_pair）の組み合わせ網羅。
   依存が軽いのでどの環境でも走る。
2) .pmck ストア（cores.pmck_store）を「各処理が同時進行している最中にリネーム/削除/再作成が
   嵐のように起きる」状況でストレスし、データ破損・部分書き込み・クラッシュ・一時ファイル残留が
   起きないことを検証する。msgpack 等が無い環境では skip。

注: AI-NR ジョブの cancel、imgset.file_path の remap、エクスポートの skip-continue は GUI/別プロセス
統合が必要なため本ファイルでは扱わない（手動/別統合テスト）。ここはデータ層(.pmck)の安全性を担保する。
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rename_detect import detect_rename_pair

try:
    from cores import pmck_store
    _PMCK_AVAILABLE = True
    _PMCK_IMPORT_ERROR = ""
except Exception as exc:  # msgpack 等が無い環境
    pmck_store = None
    _PMCK_AVAILABLE = False
    _PMCK_IMPORT_ERROR = repr(exc)


def _chk(n: int) -> int:
    """val から一意に決まるチェック値。読み取った dict の自己整合性検査に使う。"""
    return (int(n) * 2654435761) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# 1) rename 相関ヒューリスティック
# ---------------------------------------------------------------------------
class RenameDetectTests(unittest.TestCase):
    def _exists(self, present):
        present = set(present)
        return lambda p: p in present

    def test_simple_rename_detected(self):
        added = ["/d/new.cr3"]
        deleted = ["/d/old.cr3"]
        # rename 後: new 存在, old 消失
        pair = detect_rename_pair(added, deleted, exists=self._exists({"/d/new.cr3"}))
        self.assertEqual(pair, ("/d/old.cr3", "/d/new.cr3"))

    def test_delete_only(self):
        pair = detect_rename_pair([], ["/d/old.cr3"], exists=self._exists(set()))
        self.assertIsNone(pair)

    def test_add_only(self):
        pair = detect_rename_pair(["/d/new.cr3"], [], exists=self._exists({"/d/new.cr3"}))
        self.assertIsNone(pair)

    def test_multi_rename_not_correlated(self):
        added = ["/d/a2.cr3", "/d/b2.cr3"]
        deleted = ["/d/a1.cr3", "/d/b1.cr3"]
        pair = detect_rename_pair(added, deleted, exists=self._exists({"/d/a2.cr3", "/d/b2.cr3"}))
        self.assertIsNone(pair)

    def test_cross_directory_not_rename(self):
        pair = detect_rename_pair(
            ["/dst/new.cr3"], ["/src/old.cr3"],
            exists=self._exists({"/dst/new.cr3"}),
        )
        self.assertIsNone(pair)

    def test_old_still_exists_is_not_rename(self):
        # 削除イベントが古い/誤りで old がまだ存在 → rename とみなさない
        pair = detect_rename_pair(
            ["/d/new.cr3"], ["/d/old.cr3"],
            exists=self._exists({"/d/new.cr3", "/d/old.cr3"}),
        )
        self.assertIsNone(pair)

    def test_new_missing_is_not_rename(self):
        pair = detect_rename_pair(["/d/new.cr3"], ["/d/old.cr3"], exists=self._exists(set()))
        self.assertIsNone(pair)

    def test_real_fs_rename(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            old = os.path.join(d, "old.cr3")
            new = os.path.join(d, "new.cr3")
            Path(new).write_bytes(b"x")  # rename 後の状態（new だけ存在）
            pair = detect_rename_pair([new], [old])  # 実 os.path.exists を使用
            self.assertEqual(pair, (old, new))


# ---------------------------------------------------------------------------
# 2) .pmck ストアの並行ストレス（処理中リネーム/削除/再作成）
# ---------------------------------------------------------------------------
@unittest.skipUnless(_PMCK_AVAILABLE, f"pmck_store unavailable: {_PMCK_IMPORT_ERROR}")
class PmckRenameStressTests(unittest.TestCase):
    N_SLOTS = 8
    DURATION_SEC = 3.0  # 各ワーカーの実行時間

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.slots = [os.path.join(self.dir, f"img_{i}.cr3") for i in range(self.N_SLOTS)]
        for i, img in enumerate(self.slots):
            pmck_store.write_path(self._pmck(img), self._data(i))
        self._errors = []
        self._err_lock = threading.Lock()
        self._stop = threading.Event()

    def tearDown(self):
        self._tmp.cleanup()

    # --- helpers ---
    def _pmck(self, img_path):
        return pmck_store.image_pmck_path(img_path)

    def _data(self, n):
        return {"val": int(n), "chk": _chk(n), "primary_param": {}}

    def _assert_consistent(self, d):
        if isinstance(d, dict) and "val" in d:
            self.assertEqual(d.get("chk"), _chk(d["val"]),
                             msg=f"torn/inconsistent pmck dict: {d!r}")

    def _record_error(self, where, exc):
        with self._err_lock:
            self._errors.append((where, repr(exc)))

    def _run_for_duration(self, fn, where):
        deadline = time.time() + self.DURATION_SEC
        try:
            while not self._stop.is_set() and time.time() < deadline:
                fn()
        except Exception as exc:  # ワーカーは例外を握って記録（テスト本体で集計）
            self._record_error(where, exc)
            self._stop.set()

    # --- workers ---
    def _writer(self):
        c = [1000]
        def step():
            img = random.choice(self.slots)
            c[0] += 1
            pmck_store.write_path(self._pmck(img), self._data(c[0]))
        self._run_for_duration(step, "writer")

    def _updater(self):
        c = [5000]
        def step():
            img = random.choice(self.slots)
            c[0] += 1
            v = c[0]
            pmck_store.update_path(
                self._pmck(img),
                lambda d: {**(d or {}), "val": v, "chk": _chk(v), "primary_param": {}},
                default_empty=True,
            )
        self._run_for_duration(step, "updater")

    def _reader(self):
        def step():
            img = random.choice(self.slots)
            d = pmck_store.read_path(self._pmck(img), default_empty=False)
            self._assert_consistent(d)  # 部分書き込み/破損なら AssertionError
        self._run_for_duration(step, "reader")

    def _mover(self):
        # rename 追従(.pmck 移動)の往復。読み書きと衝突しても破損/クラッシュしないこと。
        def step():
            img = random.choice(self.slots)
            src = self._pmck(img)
            bak = src + ".renamebak"
            if pmck_store.move_path_to_path(src, bak):
                pmck_store.move_path_to_path(bak, src)
        self._run_for_duration(step, "mover")

    def _deleter(self):
        # 削除 → 直後に再作成（処理中に消えるケース）。
        def step():
            img = random.choice(self.slots)
            pmck_store.delete_path(self._pmck(img))
            pmck_store.write_path(self._pmck(img), self._data(7))
        self._run_for_duration(step, "deleter")

    def test_concurrent_processing_under_rename_delete_storm(self):
        workers = (
            [self._writer] * 4
            + [self._reader] * 4
            + [self._updater] * 2
            + [self._mover] * 3
            + [self._deleter] * 2
        )
        threads_ = [threading.Thread(target=w, daemon=True) for w in workers]
        for t in threads_:
            t.start()
        for t in threads_:
            t.join(timeout=self.DURATION_SEC + 30)
            self.assertFalse(t.is_alive(), "worker thread hung (possible deadlock)")

        # 1) どのワーカーも例外/破損検出を出していない
        self.assertEqual(self._errors, [], f"stress errors: {self._errors}")

        # 2) mkstemp の一時ファイル(.pmck.*.tmp)が残っていない＝原子的書き込みが完結している
        leftovers = [f for f in os.listdir(self.dir) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [], f"leftover temp files: {leftovers}")

        # 3) 残っている全 .pmck が読めて自己整合
        for img in self.slots:
            d = pmck_store.read_path(self._pmck(img), default_empty=False)
            self._assert_consistent(d)

    def test_move_follow_preserves_content(self):
        img_old = os.path.join(self.dir, "follow_old.cr3")
        img_new = os.path.join(self.dir, "follow_new.cr3")
        pmck_store.write_path(self._pmck(img_old), self._data(42))

        # 読み手が走っている最中に rename 追従（.pmck 移動）しても内容が壊れない
        reader_hits = []
        def reader():
            end = time.time() + 0.5
            while time.time() < end:
                for p in (self._pmck(img_old), self._pmck(img_new)):
                    d = pmck_store.read_path(p, default_empty=False)
                    if isinstance(d, dict) and "val" in d:
                        self._assert_consistent(d)
                        reader_hits.append(d["val"])
        rt = threading.Thread(target=reader, daemon=True)
        rt.start()
        time.sleep(0.05)
        ok = pmck_store.move_path_to_path(self._pmck(img_old), self._pmck(img_new))
        rt.join(timeout=5)

        self.assertTrue(ok)
        self.assertFalse(os.path.exists(self._pmck(img_old)))
        d = pmck_store.read_path(self._pmck(img_new), default_empty=False)
        self.assertIsNotNone(d)
        self._assert_consistent(d)
        self.assertEqual(d["val"], 42)

    def test_swap_paths_preserves_value_set(self):
        a = os.path.join(self.dir, "swap_a.cr3")
        b = os.path.join(self.dir, "swap_b.cr3")
        pmck_store.write_path(self._pmck(a), self._data(111))
        pmck_store.write_path(self._pmck(b), self._data(222))

        def swapper(idx):
            def step():
                tmp = os.path.join(self.dir, f".swaptmp_{idx}_{random.randint(0, 1<<30)}")
                pmck_store.swap_paths(self._pmck(a), self._pmck(b), tmp)
            self._run_for_duration(step, "swapper")

        # writer なし → 値集合 {111,222} は保存されるはず
        ths = [threading.Thread(target=swapper, args=(i,), daemon=True) for i in range(4)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=self.DURATION_SEC + 30)
            self.assertFalse(t.is_alive(), "swapper hung")

        self.assertEqual(self._errors, [], f"swap errors: {self._errors}")
        da = pmck_store.read_path(self._pmck(a), default_empty=False)
        db = pmck_store.read_path(self._pmck(b), default_empty=False)
        self._assert_consistent(da)
        self._assert_consistent(db)
        self.assertEqual({da["val"], db["val"]}, {111, 222},
                         msg="swap_paths lost/duplicated content")
        leftovers = [f for f in os.listdir(self.dir) if f.startswith(".swaptmp_")]
        self.assertEqual(leftovers, [], f"leftover swap temp files: {leftovers}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
