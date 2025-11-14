import numpy as np
from src import debayer_methods as dm


def make_synthetic_rgb(H=16, W=20):
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    rgb[..., 2] = 128
    return rgb


def bayer_from_rgb(rgb: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    H, W, _ = rgb.shape
    Rmask, Gmask, Bmask = dm._pattern_masks(H, W, pattern)
    Rm = Rmask.cpu().numpy().astype(bool)
    Gm = Gmask.cpu().numpy().astype(bool)
    Bm = Bmask.cpu().numpy().astype(bool)
    raw = np.zeros((H, W), dtype=np.uint8)
    raw[Rm] = rgb[..., 0][Rm]
    raw[Gm] = rgb[..., 1][Gm]
    raw[Bm] = rgb[..., 2][Bm]
    return raw


def main():
    rgb = make_synthetic_rgb(16, 20)
    raw = bayer_from_rgb(rgb, 'RGGB')
    print('raw shape', raw.shape, 'dtype', raw.dtype)
    vng = dm.debayer_vng_torch(raw, pattern='RGGB')
    ea = dm.debayer_edgeaware_torch(raw, pattern='RGGB')
    print('vng shape', vng.shape, 'dtype', vng.dtype)
    print('ea shape', ea.shape, 'dtype', ea.dtype)

    H, W = raw.shape
    Rm, Gm, Bm = dm._pattern_masks(H, W, 'RGGB')
    Rm = Rm.cpu().numpy().astype(bool)
    Gm = Gm.cpu().numpy().astype(bool)
    Bm = Bm.cpu().numpy().astype(bool)

    # outputs are BGR
    r_vng = vng[..., 2]
    g_vng = vng[..., 1]
    b_vng = vng[..., 0]

    r_ea = ea[..., 2]
    g_ea = ea[..., 1]
    b_ea = ea[..., 0]

    print('VNG mismatches: R_known:', np.count_nonzero(r_vng[Rm] != rgb[...,0][Rm]),
          'G_known:', np.count_nonzero(g_vng[Gm] != rgb[...,1][Gm]),
          'B_known:', np.count_nonzero(b_vng[Bm] != rgb[...,2][Bm]))

    print('EA mismatches: R_known:', np.count_nonzero(r_ea[Rm] != rgb[...,0][Rm]),
          'G_known:', np.count_nonzero(g_ea[Gm] != rgb[...,1][Gm]),
          'B_known:', np.count_nonzero(b_ea[Bm] != rgb[...,2][Bm]))

    # test stack helpers
    frames = np.stack([raw, raw], axis=0)
    sv = dm.debayer_stack_vng(frames, pattern='RGGB')
    se = dm.debayer_stack_edgeaware(frames, pattern='RGGB')
    print('stack shapes', sv.shape, se.shape)

    # Compare equality with single-frame
    sv0 = sv[0]
    se0 = se[0]
    print('stack equals single for VNG:', np.array_equal(sv0, vng))
    print('stack equals single for EA:', np.array_equal(se0, ea))


if __name__ == '__main__':
    main()

