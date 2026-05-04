import os


ICON_VARIANT_SIZES = (16, 32, 64)


def variant_source(source, desired_px, sizes=ICON_VARIANT_SIZES):
    if not source:
        return ""

    try:
        desired = float(desired_px)
    except (TypeError, ValueError):
        desired = sizes[-1]

    chosen = sizes[-1]
    for size in sizes:
        if desired <= size:
            chosen = size
            break

    root, ext = os.path.splitext(source)
    candidate = f"{root}_{chosen}{ext}"
    return candidate if os.path.exists(candidate) else source
