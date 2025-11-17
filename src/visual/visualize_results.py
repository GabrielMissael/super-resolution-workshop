from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2


def overlay_points(img, pts, color=(1.0, 0, 0), alpha=0.1, radius=3):
    """
    Draws colored points with transparency (alpha) on a grayscale or RGB image.

    Args:
        img: Input image, shape (H, W) or (H, W, 3)
        pts: Nx2 array of coordinates (x, y)
        color: RGB tuple, values in 0-1
        alpha: Opacity of the drawn dots, between 0 (transparent) and 1 (opaque)
        radius: Circle radius in pixels
    Returns:
        Numpy array in uint8 with points drawn and alpha blended.
    """
    # Make sure image is 3-channel
    if img.ndim == 2:
        base = np.stack([img] * 3, axis=-1).astype(np.float32)
    else:
        base = img.astype(np.float32).copy()

    overlay = np.zeros_like(base)

    dot_color = tuple(int(255 * c) for c in color)
    for (x, y) in pts:
        cv2.circle(overlay, (int(x), int(y)), radius=radius, color=dot_color, thickness=-1)

    # Create a mask where circles are drawn
    mask = (overlay > 0).any(axis=2, keepdims=True).astype(np.float32)
    # Alpha blending only where mask is 1
    result = base * (1 - alpha * mask) + overlay * (alpha * mask)

    return np.clip(result, 0, 255).astype(np.uint8)