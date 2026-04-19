"""
pmck 内の mask2 がヘッドレス実装のみで復元可能か判定する。
"""
from __future__ import annotations

from cores.mask2.mask_types import MaskTypeStr

_HEADLESS_TYPES = frozenset(
    {
        MaskTypeStr.COMPOSIT,
        MaskTypeStr.CIRCULAR,
        MaskTypeStr.GRADIENT,
        MaskTypeStr.FULL,
        MaskTypeStr.FREEDRAW,
        MaskTypeStr.SEGMENT,
        MaskTypeStr.DEPTHMAP,
        MaskTypeStr.FACE,
        MaskTypeStr.TARGET_TEXT,
    }
)


def _normalize_mask_type(t) -> str:
    if isinstance(t, str):
        return t
    return getattr(t, "value", str(t))


def _walk_mask_nodes(node: dict):
    t = _normalize_mask_type(node.get("type"))
    yield t
    if t == MaskTypeStr.COMPOSIT:
        for entry in node.get("mask_list", []):
            yield from _walk_mask_nodes(entry[0])


def pmck_mask2_is_headless_only(pmck: dict) -> bool:
    """mask2 に含まれるすべての型がヘッドレス対応なら True。"""
    ml = pmck.get("mask2")
    if not ml:
        return True
    for root in ml:
        for t in _walk_mask_nodes(root):
            if t not in _HEADLESS_TYPES:
                return False
    return True
