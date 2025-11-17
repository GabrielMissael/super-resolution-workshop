"""Utilities to convert PNG (or other common image formats) to FITS files.

Functions:
- png_to_fits(infile, outfile, header=None, overwrite=False)

Behavior:
- Reads image using imageio (falls back to PIL if needed).
- If image is color (H,W,3), writes FITS with shape (3, H, W) (channels first) and sets NAXIS=3.
- If grayscale, writes 2D image (H,W).
- Accepts header as dict to add header keywords to FITS header.
- Adds a HISTORY card describing conversion.

Requires: astropy, imageio or pillow
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np

try:
    from astropy.io import fits
except Exception as e:  # pragma: no cover
    raise ImportError("astropy is required. Install with: pip install astropy") from e

# Prefer imageio for broad format support
try:
    import imageio
    _USE_IMAGEIO = True
except Exception:  # pragma: no cover
    _USE_IMAGEIO = False
    try:
        from PIL import Image
    except Exception:
        raise ImportError("Either imageio or Pillow is required to read PNG files. Install with: pip install imageio or pillow")


def _read_image_to_array(infile: str) -> np.ndarray:
    """Read an image file to a numpy array (uint8 or uint16).

    Returns array with shape (H,W) or (H,W,3).
    """
    if _USE_IMAGEIO:
        img = imageio.v3.imread(infile)
        arr = np.asarray(img)
    else:
        from PIL import Image
        img = Image.open(infile)
        arr = np.asarray(img)
    # if image has alpha, drop it
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


def png_to_fits(infile: str, outfile: str, header: Optional[Dict[str, object]] = None, overwrite: bool = False) -> None:
    """Convert an image (png/jpg/...) to a FITS file, preserving pixel values.

    infile: path to image file
    outfile: path to write FITS
    header: optional dict of header keywords to add
    overwrite: whether to overwrite existing outfile
    """
    infile = str(infile)
    outfile = str(outfile)
    if os.path.abspath(infile) == os.path.abspath(outfile):
        raise ValueError('infile and outfile must be different')
    arr = _read_image_to_array(infile)
    # Convert boolean or float to numeric types
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    # For color images, convert to shape (C,H,W) as typical FITS multi-dimensional array
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Convert to channels-first and ensure a writeable numeric dtype
        data = np.transpose(arr, (2, 0, 1)).astype(arr.dtype)
    elif arr.ndim == 2:
        data = arr.astype(arr.dtype)
    else:
        # Unsupported number of channels
        raise ValueError(f'Unsupported image shape for conversion: {arr.shape}')

    hdr = fits.Header()
    if header:
        for k, v in header.items():
            # FITS header keys must be <=8 chars; skip invalid keys
            try:
                hdr[k] = v
            except Exception:
                # ignore invalid header entries
                pass
    hdr.add_history(f'Converted from {os.path.basename(infile)}')

    # Create PrimaryHDU and write
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(outfile, overwrite=overwrite)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Convert PNG/JPG to FITS')
    p.add_argument('--infile', required=True, help='input image file (png/jpg...)')
    p.add_argument('--outfile', required=True, help='output FITS file')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    png_to_fits(args.infile, args.outfile, overwrite=args.overwrite)


if __name__ == '__main__':  # pragma: no cover
    main()

