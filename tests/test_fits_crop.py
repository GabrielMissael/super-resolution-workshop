import numpy as np
from astropy.io import fits
import tempfile
from src import fits_crop as fc


def make_test_hdul(h=10, w=12):
    data = np.arange(h*w, dtype=np.int32).reshape((h, w))
    hdr = fits.Header()
    hdr['TESTKEY'] = 'VALUE'
    hdr['CRPIX1'] = 5.0
    hdr['CRPIX2'] = 4.0
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdul = fits.HDUList([hdu])
    return hdul


def test_crop_hdul_and_header_adjustments():
    hdul = make_test_hdul(10, 12)
    x, y, w, h = 2, 3, 5, 4
    new_hdul = fc.crop_fits_hdul(hdul, x, y, w, h, hdu_index=0)
    assert isinstance(new_hdul, fits.HDUList)
    prim = new_hdul[0]
    assert prim.data.shape == (h, w)
    # check a few pixel values match original
    orig = hdul[0].data
    np.testing.assert_array_equal(prim.data, orig[y:y+h, x:x+w])
    # header keys adjusted
    assert prim.header['NAXIS1'] == w
    assert prim.header['NAXIS2'] == h
    # CRPIX adjusted by subtracting origin
    assert np.isclose(prim.header['CRPIX1'], 5.0 - x)
    assert np.isclose(prim.header['CRPIX2'], 4.0 - y)
    # history present
    assert 'Cropped to' in repr(prim.header)


def test_crop_fits_file_roundtrip(tmp_path):
    hdul = make_test_hdul(8, 8)
    inpath = tmp_path / 'in.fits'
    outpath = tmp_path / 'out.fits'
    hdul.writeto(inpath)
    fc.crop_fits_file(str(inpath), str(outpath), x=1, y=1, w=4, h=4, overwrite=False)
    # read back
    new = fits.open(outpath)
    assert new[0].data.shape == (4, 4)
    assert new[0].header['TESTKEY'] == 'VALUE'


def test_preserve_all_hdus(tmp_path):
    # create HDUList with multiple image HDUs
    hdu1 = fits.PrimaryHDU(data=np.ones((10, 10), dtype=np.uint8), header=fits.Header())
    hdu2 = fits.ImageHDU(data=(np.arange(100).reshape(10,10)).astype(np.int16), header=fits.Header())
    hdul = fits.HDUList([hdu1, hdu2])
    inpath = tmp_path / 'multi.fits'
    outpath = tmp_path / 'multi_out.fits'
    hdul.writeto(inpath)
    fc.crop_fits_file(str(inpath), str(outpath), x=2, y=2, w=4, h=4, preserve_all_hdus=True)
    new = fits.open(outpath)
    # primary should be an image with cropped size
    assert new[0].data.shape == (4, 4)
    # second HDU should also be cropped
    assert new[1].data.shape == (4, 4)

