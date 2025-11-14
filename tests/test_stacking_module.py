import numpy as np
import pytest
from src import stacking as st


def make_frames(N=5, H=8, W=8):
    rng = np.random.RandomState(0)
    base = np.linspace(0, 255, H*W).reshape(H, W).astype(np.float64)
    frames = np.stack([base + rng.normal(scale=2.0, size=base.shape) for _ in range(N)], axis=0).astype(np.uint8)
    return frames


def test_average_and_median_basic():
    frames = make_frames(6, 8, 8)
    avg = st.stack_average(frames)
    med = st.stack_median(frames)
    assert avg.shape == frames.shape[1:]
    assert med.shape == frames.shape[1:]
    assert avg.dtype == np.float64


def test_weighted_and_normalization():
    frames = make_frames(4, 8, 8)
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    out = st.stack_weighted_average(frames, weights)
    assert out.shape == frames.shape[1:]


def test_sigma_clip_and_return_mask():
    frames = make_frames(6, 8, 8)
    # inject outliers in frame 0
    frames[0, 0, 0] = 255
    out, mask = st.stack_sigma_clip(frames, sigma=1.0, max_iters=3, return_mask=True)
    assert out.shape == frames.shape[1:]
    assert mask.shape == frames.shape


def test_trimmed_and_winsorized_and_errors():
    frames = make_frames(5, 8, 8)
    tm = st.stack_trimmed_mean(frames, proportiontocut=0.1)
    wm = st.stack_winsorized_mean(frames, limits=0.1)
    assert tm.shape == frames.shape[1:]
    assert wm.shape == frames.shape[1:]
    with pytest.raises(ValueError):
        st.stack_trimmed_mean(frames, proportiontocut=0.6)
    with pytest.raises(ValueError):
        st.stack_winsorized_mean(frames, limits=0.6)


def test_choose_top_k_and_stack_by_quality():
    frames = make_frames(10, 8, 8)
    scores = np.linspace(0, 1, frames.shape[0])
    chosen = st.choose_top_k_by_quality(frames, scores, frac=0.2)
    assert chosen.shape[0] >= 1
    out = st.stack_by_quality(frames, scores, method='trimmed', top_frac=0.3)
    assert out.shape == frames.shape[1:]
    outw = st.stack_by_quality(frames, scores, method='weighted')
    assert outw.shape == frames.shape[1:]
    with pytest.raises(ValueError):
        st.choose_top_k_by_quality(frames, scores)
    with pytest.raises(ValueError):
        st.stack_by_quality(frames, scores, method='unknown')

