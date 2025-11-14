import numpy as np
from src import debayer_methods as dm


def make_synthetic_rgb(H=32, W=48):
    # R horizontal ramp, G vertical ramp, B constant
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    rgb[..., 2] = 128
    return rgb


def bayer_from_rgb(rgb: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    H, W, _ = rgb.shape
    Rmask, Gmask, Bmask = dm._pattern_masks(H, W, pattern)
    # masks are torch tensors
    Rm = Rmask.cpu().numpy().astype(bool)
    Gm = Gmask.cpu().numpy().astype(bool)
    Bm = Bmask.cpu().numpy().astype(bool)
    raw = np.zeros((H, W), dtype=np.uint8)
    raw[Rm] = rgb[..., 0][Rm]
    raw[Gm] = rgb[..., 1][Gm]
    raw[Bm] = rgb[..., 2][Bm]
    return raw


def test_vng_and_edgeaware_preserve_samples():
    rgb = make_synthetic_rgb(32, 48)
    raw = bayer_from_rgb(rgb, 'RGGB')

    out_vng = dm.debayer_vng_torch(raw, pattern='RGGB')
    out_ea = dm.debayer_edgeaware_torch(raw, pattern='RGGB')

    assert out_vng.shape == (32, 48, 3)
    assert out_ea.shape == (32, 48, 3)
    assert out_vng.dtype == np.uint8
    assert out_ea.dtype == np.uint8

    # check that known sampled pixels are preserved in output
    H, W = raw.shape
    Rmask, Gmask, Bmask = dm._pattern_masks(H, W, 'RGGB')
    Rm = Rmask.cpu().numpy().astype(bool)
    Gm = Gmask.cpu().numpy().astype(bool)
    Bm = Bmask.cpu().numpy().astype(bool)

    # outputs are BGR
    # R channel is index 2, G index 1, B index 0
    assert np.array_equal(out_vng[..., 2][Rm], rgb[..., 0][Rm])
    assert np.array_equal(out_vng[..., 1][Gm], rgb[..., 1][Gm])
    assert np.array_equal(out_vng[..., 0][Bm], rgb[..., 2][Bm])

    assert np.array_equal(out_ea[..., 2][Rm], rgb[..., 0][Rm])
    assert np.array_equal(out_ea[..., 1][Gm], rgb[..., 1][Gm])
    assert np.array_equal(out_ea[..., 0][Bm], rgb[..., 2][Bm])


def test_uint16_input_and_stack_helpers():
    rgb = make_synthetic_rgb(16, 20)
    raw8 = bayer_from_rgb(rgb, 'RGGB')
    # make uint16 raw by scaling
    raw16 = (raw8.astype(np.uint16) * 257)

    out_vng_16 = dm.debayer_vng_torch(raw16, pattern='RGGB')
    out_ea_16 = dm.debayer_edgeaware_torch(raw16, pattern='RGGB')

    assert out_vng_16.shape == (16, 20, 3)
    assert out_ea_16.shape == (16, 20, 3)

    # Check stack helpers
    frames = np.stack([raw8, raw8], axis=0)
    stacked_vng = dm.debayer_stack_vng(frames, pattern='RGGB')
    stacked_ea = dm.debayer_stack_edgeaware(frames, pattern='RGGB')

    assert stacked_vng.shape == (2, 16, 20, 3)
    assert stacked_ea.shape == (2, 16, 20, 3)

    # Compare first frame result to single-frame function
    single_vng = dm.debayer_vng_torch(raw8, pattern='RGGB')
    assert np.array_equal(stacked_vng[0], single_vng)

    single_ea = dm.debayer_edgeaware_torch(raw8, pattern='RGGB')
    assert np.array_equal(stacked_ea[0], single_ea)

