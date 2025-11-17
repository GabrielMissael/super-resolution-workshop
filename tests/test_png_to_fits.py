import numpy as np
from pathlib import Path
from astropy.io import fits
from src.io.png_to_fits import png_to_fits


def test_png_to_fits_roundtrip(tmp_path):
    # create a small RGB PNG
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    img[..., 0] = np.arange(12, dtype=np.uint8)
    img[..., 1] = np.arange(10, dtype=np.uint8)[:, None]
    img[..., 2] = 128
    infile = tmp_path / 'test.png'
    outfile = tmp_path / 'test.fits'
    try:
        import imageio
        imageio.v3.imwrite(str(infile), img)
    except Exception:
        from PIL import Image
        Image.fromarray(img).save(infile)

    png_to_fits(str(infile), str(outfile), header={'TESTKEY': 'VAL'}, overwrite=True)
    hdul = fits.open(outfile)
    data = hdul[0].data
    # Expect channels-first shape (3,H,W)
    assert data.shape == (3, 10, 12)
    assert hdul[0].header['TESTKEY'] == 'VAL'
    assert 'Converted from' in hdul[0].header['HISTORY'][0]

