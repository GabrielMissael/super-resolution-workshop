"""Simple smoke tests for src/stacking.py

Run: python scripts/test_stacking.py
"""
import sys
from pathlib import Path
# ensure project root is on sys.path so 'import src' works when running this script
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

import numpy as np
from src import stacking


def make_synthetic_frames(N=20, H=64, W=64, channels=1, seed=0):
    rng = np.random.RandomState(seed)
    # base image: circular gradient (planet-like)
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    base = 255 * (1 - (r / r.max()))
    if channels == 1:
        base = base.astype(np.float64)
        frames = np.stack([base + rng.normal(scale=2.0, size=base.shape) for _ in range(N)], axis=0)
    else:
        base3 = np.stack([base, base * 0.9, base * 0.8], axis=-1)
        frames = np.stack([base3 + rng.normal(scale=2.0, size=base3.shape) for _ in range(N)], axis=0)
    # add few salt-and-pepper outliers
    for i in range(3):
        idx = rng.randint(0, N)
        yi = rng.randint(0, H)
        xi = rng.randint(0, W)
        if channels == 1:
            frames[idx, yi, xi] = 255 * rng.choice([0, 1])
        else:
            frames[idx, yi, xi, :] = 255 * rng.choice([0, 1])
    # clip to uint8 range
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def run_all():
    frames = make_synthetic_frames(N=20, H=64, W=64, channels=1)
    print('frames shape', frames.shape, 'dtype', frames.dtype)
    methods = []
    out = stacking.stack_average(frames)
    print('average ok', out.shape, out.dtype, np.nanmax(out))
    out = stacking.stack_median(frames)
    print('median ok', out.shape, np.nanmax(out))
    out, mask = stacking.stack_sigma_clip(frames, sigma=2.5, max_iters=4, return_mask=True)
    print('sigma ok', out.shape, 'masked pixels:', mask.sum())
    out = stacking.stack_trimmed_mean(frames, proportiontocut=0.1)
    print('trimmed ok', out.shape)
    out = stacking.stack_winsorized_mean(frames, limits=0.1)
    print('winsorized ok', out.shape)
    weights = np.linspace(1.0, 2.0, frames.shape[0])
    out = stacking.stack_weighted_average(frames, weights)
    print('weighted ok', out.shape)
    # quality-based
    scores = np.linspace(0, 1, frames.shape[0])
    chosen = stacking.choose_top_k_by_quality(frames, scores, frac=0.2)
    print('choose_top_k_by_quality', chosen.shape)
    out = stacking.stack_by_quality(frames, scores, method='trimmed', top_frac=0.2)
    print('stack_by_quality trimmed', out.shape)
    out = stacking.stack_by_quality(frames, scores, method='weighted')
    print('stack_by_quality weighted', out.shape)
    print('All stacking smoke tests passed')

if __name__ == '__main__':
    run_all()
