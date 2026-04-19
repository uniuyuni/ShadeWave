"""Mask2 処理コア（Kivy 非依存のヘッドレス経路）。"""

from cores.mask2.exceptions import HeadlessMaskNotSupported
from cores.mask2.headless_pipeline import Mask2HeadlessPipeline
from cores.mask2.pmck_support import pmck_mask2_is_headless_only

__all__ = [
    "HeadlessMaskNotSupported",
    "Mask2HeadlessPipeline",
    "pmck_mask2_is_headless_only",
]
