from astropy.stats import sigma_clip
import numpy as np


def normalize_flat(master_flat, method='median'):
    master_flat = master_flat.astype(float)
    if method == 'median':
        norm = np.median(master_flat)
    else:
        norm = np.mean(master_flat)
    if norm == 0:
        raise ValueError("Master flat has zero normalization")
    return master_flat / norm


def calibrate_frame(raw, master_dark, master_flat_norm, master_bias=None):
    raw = raw.astype(float)
    master_dark = master_dark.astype(float)
    if master_bias is not None:
        raw = raw - master_bias
        master_dark = master_dark - master_bias
    calibrated = (raw - master_dark) / master_flat_norm
    # avoid divisions by zero or NaNs from bad flat pixels
    calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0)
    return calibrated


def make_master_frame(calibrated_stack, method='sigma_clip_mean', sigma=3.0, iters=3):
    arr = np.stack(calibrated_stack, axis=0)  # shape (N, H, W, C)
    if method == 'median':
        return np.median(arr, axis=0)
    elif method == 'sigma_clip_mean':
        clipped = sigma_clip(arr, sigma=sigma, maxiters=iters, axis=0)
        # clipped is a MaskedArray; compute mean skipping masked values
        return np.ma.mean(clipped, axis=0).filled(0.0)
    else:
        raise ValueError("Unknown method")


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def percentile_normalize(data, pmin=1, pmax=99):
    vmin, vmax = np.percentile(data, (pmin, pmax))
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)


def clip(data):
    return np.clip(data, 0, 1)


def normalize_clip(data):
    return clip(normalize(data))


def crop(data, center=None, size=400):
    h, w = data.shape[:2]
    cy, cx = center if center is not None else (h // 2, w // 2)
    half_size = size // 2
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size)
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size)
    return data[y1:y2, x1:x2]
