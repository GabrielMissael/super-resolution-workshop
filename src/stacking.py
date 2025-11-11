"""
Image stacking utilities for aligned, debayered and filtered planetary frames.

Provides several stacking schemes commonly used in astrophotography:
- simple average
- median
- sigma-clipped mean (iterative)
- trimmed mean
- winsorized mean
- weighted mean (using quality scores)
- choose_top_k_by_quality

API contract (per function):
- Input: frames as numpy array with shape (N, H, W) or (N, H, W, C). dtype can be uint8/uint16/float.
- Optional masks: boolean array with shape (N, H, W) or (N, H, W, 1) marking invalid pixels (True=masked).
- Optional scores: per-frame 1D array of length N used by weighting or selection.
- Output: stacked image as float64 numpy array with same spatial and channel dims (H, W[, C]).

Functions are defensive about small N and will fall back to median when trimming removes all values.

No external dependencies other than numpy.
"""

from typing import Optional, Tuple
import numpy as np


def _as_float(frames: np.ndarray) -> np.ndarray:
    """Convert frames to float64 for processing without modifying input."""
    return frames.astype(np.float64, copy=False)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if s == 0:
        # fallback to uniform
        return np.ones_like(w) / len(w)
    return w / s


def _ensure_mask(frames: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Return boolean mask of shape (N, ...) where True means masked/invalid."""
    N = frames.shape[0]
    if mask is None:
        return np.zeros((N,) + frames.shape[1:], dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != N:
        raise ValueError("mask must have same first dimension as frames (per-frame)")
    # if mask has fewer dims, broadcast
    if m.ndim == 1:
        # per-frame scalar masks
        return m.reshape((N,) + (1,) * (frames.ndim - 1))
    # else try to broadcast
    return m


def stack_average(frames: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Simple arithmetic mean across frames, ignoring masked pixels.

    frames: (N, H, W[, C])
    mask: boolean shape (N, H, W[, C]) or (N,) etc. True means invalid.
    """
    f = _as_float(frames)
    m = _ensure_mask(frames, mask)
    # use nan to ignore masked values
    f_masked = np.where(m, np.nan, f)
    out = np.nanmean(f_masked, axis=0)
    # where all values were masked, nanmean returns nan; replace with zeros
    out = np.nan_to_num(out, nan=0.0)
    return out


def stack_median(frames: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixel-wise median across frames, ignoring masked pixels."""
    f = _as_float(frames)
    m = _ensure_mask(frames, mask)
    f_masked = np.where(m, np.nan, f)
    out = np.nanmedian(f_masked, axis=0)
    out = np.nan_to_num(out, nan=0.0)
    return out


def stack_weighted_average(frames: np.ndarray, weights: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted mean across frames using per-frame weights.

    weights: 1D array of length N (per-frame). Frames may have channels; weights are applied per-frame.
    """
    f = _as_float(frames)
    m = _ensure_mask(frames, mask)
    w = _normalize_weights(weights)
    # expand weights to broadcast across image dims
    shape = (f.shape[0],) + (1,) * (f.ndim - 1)
    w_exp = w.reshape(shape)
    f_masked = np.where(m, np.nan, f)
    # multiply, sum ignoring nans
    num = np.nansum(f_masked * w_exp, axis=0)
    denom = np.nansum(~np.isnan(f_masked) * w_exp, axis=0)
    # denom can be zero where all masked -> set to 1 to avoid div0 then zero result
    denom_safe = np.where(denom == 0, 1.0, denom)
    out = num / denom_safe
    out = np.nan_to_num(out, nan=0.0)
    return out


def stack_sigma_clip(frames: np.ndarray, sigma: float = 3.0, max_iters: int = 5,
                     mask: Optional[np.ndarray] = None, return_mask: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Iterative sigma-clipped mean across frames.

    Iteratively computes per-pixel mean and std, masks values that are farther than
    sigma * std from the mean, and repeats.

    Returns (stacked_image, final_mask) if return_mask True; final_mask is boolean
    array shaped like frames indicating pixels excluded by clipping.
    """
    f = _as_float(frames)
    N = f.shape[0]
    m = _ensure_mask(frames, mask)
    # current mask: True where invalid/outlier
    current_mask = m.copy()

    for it in range(max_iters):
        # compute mean and std ignoring masked
        f_masked = np.where(current_mask, np.nan, f)
        mean = np.nanmean(f_masked, axis=0)
        std = np.nanstd(f_masked, axis=0)
        # avoid zero stds
        std_safe = np.where(std == 0, 1e-12, std)
        # compute deviations
        # broadcast mean/std to shape (N, ...)
        mean_exp = np.expand_dims(mean, 0)
        std_exp = np.expand_dims(std_safe, 0)
        dev = np.abs(f - mean_exp)
        newly_masked = dev > (sigma * std_exp)
        # if no new masks, break
        combined = current_mask | newly_masked
        if np.array_equal(combined, current_mask):
            break
        current_mask = combined
    # final mean
    f_masked = np.where(current_mask, np.nan, f)
    out = np.nanmean(f_masked, axis=0)
    out = np.nan_to_num(out, nan=0.0)
    if return_mask:
        return out, current_mask
    return out, None


def stack_trimmed_mean(frames: np.ndarray, proportiontocut: float = 0.1, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Trimmed mean: drop lowest and highest fraction and average the remainder.

    proportiontocut: fraction [0, 0.5). If trims remove all values, falls back to median.
    """
    if not (0.0 <= proportiontocut < 0.5):
        raise ValueError("proportiontocut must be in [0, 0.5)")
    f = _as_float(frames)
    m = _ensure_mask(frames, mask)
    # mask values as nan and sort along axis 0
    f_masked = np.where(m, np.nan, f)
    # move axis 0 to front, compute sorted along axis 0 even with NaNs
    # strategy: sort nan to end by using nan_to_num with large sentinel
    N = f_masked.shape[0]
    if N == 0:
        raise ValueError("frames must have length > 0")
    k = int(np.floor(N * proportiontocut))
    if 2 * k >= N:
        # trimmed everything -> use median
        return stack_median(frames, mask=mask)
    # Replace nan with +/-inf so they sort to ends
    # But np.nan sorts to end; use numpy sort which leaves nans at end
    sorted_vals = np.sort(f_masked, axis=0)
    # take middle chunk [k: N-k]
    middle = sorted_vals[k:N - k]
    out = np.nanmean(middle, axis=0)
    out = np.nan_to_num(out, nan=0.0)
    return out


def stack_winsorized_mean(frames: np.ndarray, limits: float = 0.1, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Winsorized mean: replace extremes with boundary values then average.

    limits: fraction in [0, 0.5)
    """
    if not (0.0 <= limits < 0.5):
        raise ValueError("limits must be in [0, 0.5)")
    f = _as_float(frames)
    m = _ensure_mask(frames, mask)
    N = f.shape[0]
    k = int(np.floor(N * limits))
    if 2 * k >= N:
        return stack_median(frames, mask=mask)
    f_masked = np.where(m, np.nan, f)
    # sort values
    sorted_vals = np.sort(f_masked, axis=0)
    lower_bound = sorted_vals[k]
    upper_bound = sorted_vals[N - k - 1]
    # clip values to bounds
    clipped = np.clip(f_masked, lower_bound, upper_bound)
    out = np.nanmean(clipped, axis=0)
    out = np.nan_to_num(out, nan=0.0)
    return out


def choose_top_k_by_quality(frames: np.ndarray, scores: np.ndarray, k: Optional[int] = None, frac: Optional[float] = None) -> np.ndarray:
    """Return subset of frames chosen by top quality scores.

    Provide either k (integer) or frac (0<frac<=1) to select top-K frames.
    Returns frames[top_indices].
    """
    if k is None and frac is None:
        raise ValueError("Either k or frac must be provided")
    N = frames.shape[0]
    scores = np.asarray(scores)
    if scores.shape[0] != N:
        raise ValueError("scores length must match number of frames")
    if frac is not None:
        if not (0 < frac <= 1):
            raise ValueError("frac must be in (0, 1]")
        k = max(1, int(np.round(N * frac)))
    k = min(k, N)
    idx = np.argsort(scores)[-k:][::-1]
    return frames[idx]


def stack_by_quality(frames: np.ndarray, scores: np.ndarray, method: str = "trimmed", top_frac: float = 0.2, **kwargs) -> np.ndarray:
    """Convenience wrapper: choose top frames by quality then apply stacking method.

    method: one of 'average', 'median', 'sigma', 'trimmed', 'winsorized', 'weighted'
    top_frac: fraction of frames to keep based on scores (descending). For 'weighted', top_frac is ignored.
    kwargs: forwarded to the specific stacking function.
    """
    if method == 'weighted':
        # weights are the scores
        return stack_weighted_average(frames, weights=scores, mask=kwargs.get('mask', None))
    if top_frac is None:
        top_frac = 0.2
    chosen = choose_top_k_by_quality(frames, scores, frac=top_frac)
    if method == 'average':
        return stack_average(chosen, mask=None)
    if method == 'median':
        return stack_median(chosen, mask=None)
    if method == 'sigma':
        out, _ = stack_sigma_clip(chosen, sigma=kwargs.get('sigma', 3.0), max_iters=kwargs.get('max_iters', 5), mask=None)
        return out
    if method == 'trimmed':
        return stack_trimmed_mean(chosen, proportiontocut=kwargs.get('proportiontocut', 0.1), mask=None)
    if method == 'winsorized':
        return stack_winsorized_mean(chosen, limits=kwargs.get('limits', 0.1), mask=None)
    raise ValueError(f"unknown method {method}")


# end of file

