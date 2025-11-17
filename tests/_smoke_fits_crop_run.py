import numpy as np
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, r'C:\Users\npayo\Qsync\Maitrise\Projects\super-resolution-workshop')
from astropy.io import fits
from src import fits_center_crop as fcc

TMP = Path('tests') / 'tmp_fits_smoke'
TMP.mkdir(parents=True, exist_ok=True)

inpath = TMP / 'in_large.fits'
outpath = TMP / 'out_crop.fits'

H = 4176
W = 6248
print('Creating synthetic FITS', inpath, 'shape', H, W)
data = (np.arange(H*W, dtype=np.uint32) % 65535).reshape((H, W)).astype(np.uint16)
# create header with CRPIX values
hdr = fits.Header()
hdr['CRPIX1'] = 3124.0
hdr['CRPIX2'] = 2088.0
hdu = fits.PrimaryHDU(data=data, header=hdr)
hdu.writeto(inpath, overwrite=True)

# desired crop
Wout = 3000
Hout = 2000
print('Running center crop to', Wout, Hout)
fcc.crop_file(inpath, outpath, Wout, Hout, overwrite=True)

# read back
with fits.open(outpath) as hdul:
    new = hdul[0]
    print('Output shape:', new.data.shape)
    print('Header NAXIS1/NAXIS2:', new.header.get('NAXIS1'), new.header.get('NAXIS2'))
    print('Header CRPIX1/CRPIX2:', new.header.get('CRPIX1'), new.header.get('CRPIX2'))

# verify
x = (W - Wout) // 2
y = (H - Hout) // 2
expected_crpix1 = 3124.0 - x
expected_crpix2 = 2088.0 - y
print('Expected x,y:', x, y)
print('Expected CRPIX1/2:', expected_crpix1, expected_crpix2)

assert new.data.shape == (Hout, Wout), 'Wrong output shape'
assert new.header['NAXIS1'] == Wout
assert new.header['NAXIS2'] == Hout
assert abs(new.header['CRPIX1'] - expected_crpix1) < 1e-6
assert abs(new.header['CRPIX2'] - expected_crpix2) < 1e-6
print('Smoke test passed')

