"""
ガイデッドフィルターによるマスク調整（mask_editor2 と共有）。
"""
from __future__ import annotations

import cv2
import numpy as np


def create_cutout_mask_guided(image, rough_mask, radius=8, eps=0.001):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    refined = cv2.ximgproc.guidedFilter(
        guide=gray,
        src=rough_mask,
        radius=radius,
        eps=eps,
    )
    return refined
