
import logging
import os
import sys

import numpy as np

# nagi_inpaint は姉妹プロジェクト（インストール不要でパス追加して使う）
_RI_PROJECT = os.environ.get(
    "RI_PROJECT", os.path.join(os.path.dirname(__file__), "..", "..", "nagi_inpaint"))
if _RI_PROJECT not in sys.path:
    sys.path.insert(0, _RI_PROJECT)

from nagi_inpaint import api as _api

# 重みは ShadeWave の慣習に合わせて ./checkpoints/ 配下（環境変数で上書き可）
_CKPT = os.environ.get("RI_CKPT", "./checkpoints/nagi_inpaint/latest.pt")
_REFINER_CKPT = os.environ.get("RI_REFINER_CKPT", "./checkpoints/nagi_inpaint/refiner.pt")


def setup(device="mps"):
    logging.info(f"nagi_inpaint ({_CKPT}) をロード中...")
    refiner = _REFINER_CKPT if os.path.exists(_REFINER_CKPT) else None
    pipeline, device = _api.setup(device=device, ckpt=_CKPT, refiner_ckpt=refiner)
    return (pipeline, device)


def predict(pipe, image, mask):
    """
    画像全体に対する単発インペイント。
    image: (H, W, 3) float32 RGB。1.0 超（リニア HDR）可 — 内部でトーンマップ往復する。
    mask:  (H, W) or (H, W, 1) float32、1 が除去対象。
    """
    logging.info("nagi_inpaint でインペイント処理を実行中...")
    return _api.predict(pipe[0], image, mask)


def predict_helper(pipe, image, mask, bbox):
    """
    sdxl_helper.predict_helper と同一契約。
    bbox (x, y, w, h) 周辺の文脈を読み、マスク領域をインペイントして
    image を in-place 更新し、同じ配列を返す。

    大域文脈はマスク周辺クロップの縮小版（長辺 <=1280）で一度に読むため、
    40MP 級でもタイル分割は細部リファインメントのみに使われる。
    """
    logging.info("nagi_inpaint でインペイント処理を実行中...")
    image = np.ascontiguousarray(image, dtype=np.float32)
    return _api.predict_helper(pipe[0], image, mask, bbox)
