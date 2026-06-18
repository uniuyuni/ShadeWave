from dataclasses import replace

import numpy as np

from utils.external_paths import add_external_path, external_path

add_external_path("depth_pro", "src")


def setup_model(device="cpu"):
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    checkpoint_path = external_path("depth_pro", "checkpoints", "depth_pro.pt")
    config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(checkpoint_path),
    )
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    return (model, transform)


def normalize_depth_for_mask(depth, lower_percentile=1.0, upper_percentile=99.0):
    depth = np.asarray(depth, dtype=np.float32)
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim > 2:
        depth = np.squeeze(depth)
    finite = np.isfinite(depth) & (depth > np.float32(1e-6))
    if not np.any(finite):
        return np.zeros(depth.shape, dtype=np.float32)

    inverse_depth = np.zeros(depth.shape, dtype=np.float32)
    inverse_depth[finite] = np.float32(1.0) / depth[finite]
    valid = inverse_depth[finite]

    lo, hi = np.percentile(valid, [lower_percentile, upper_percentile])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.float32)

    result = (inverse_depth - np.float32(lo)) / np.float32(hi - lo)
    result = np.clip(result, np.float32(0.0), np.float32(1.0))
    result[~finite] = np.float32(0.0)
    return result.astype(np.float32, copy=False)


def predict_model(mt, image):
    import torch

    model, transform = mt
    image = transform(image)

    with torch.no_grad():
        prediction = model.infer(image, f_px=None)

    depth = prediction["depth"]
    return normalize_depth_for_mask(depth.cpu().numpy())
