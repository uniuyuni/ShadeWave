# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Depth Pro package."""

import torch
import numpy as np

from .depth_pro import create_model_and_transforms  # noqa
from .utils import load_rgb  # noqa

def setup_model(device='cpu'):

    model, transform = create_model_and_transforms(device=device)
    model.eval()

    return (model, transform)

def predict_model(mt, image):
    model, transform = mt

    image = transform(image)

    # Run inference.
    with torch.no_grad():
        prediction = model.infer(image, f_px=None)

    depth = prediction["depth"]  # Depth in [m].
    depth_cpu = depth.cpu().numpy()
    min = np.min(depth_cpu)
    max = np.max(depth_cpu)
    return (depth_cpu - min) / (max - min)
