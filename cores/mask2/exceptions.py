"""Mask2 ヘッドレス処理で未対応の場合に送出する例外。"""


class HeadlessMaskNotSupported(Exception):
    """pmck に含まれるマスク型がヘッドレス実装にまだない。"""
