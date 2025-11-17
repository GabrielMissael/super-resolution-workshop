"""Utilities to crop FITS images while preserving headers.

Functions:
- crop_fits_file(infile, outfile, x, y, w, h, hdu_index=0, preserve_all_hdus=False)
- crop_fits_hdul(hdul, x, y, w, h, hdu_index=0, preserve_all_hdus=False)

Behavior notes:
- Coordinates: x (column) and y (row) are zero-based pixel indices into the image.
- Cropping keeps header keywords intact, updates NAXIS1/NAXIS2, and adjusts CRPIX1/CRPIX2
  when present (they are reduced by the crop origin).
- A HISTORY card is appended describing the crop.
- If preserve_all_hdus=True, all image HDUs in the input HDUList are cropped and written
  into the output; otherwise only the specified HDU index is cropped and stored as the
  PrimaryHDU of the output file.

Requires: astropy
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np

try:
    from astropy.io import fits
except Exception as e:  # pragma: no cover
    raise ImportError("astropy is required for fits_crop utilities. Install with: pip install astropy") from e


def _adjust_header_for_crop(header: fits.Header, x: int, y: int, w: int, h: int) -> fits.Header:
    hdr = header.copy()
    # Update NAXIS1/2 if present
    if 'NAXIS1' in hdr:
        hdr['NAXIS1'] = w
    if 'NAXIS2' in hdr:
        hdr['NAXIS2'] = h

    # Adjust CRPIX1/CRPIX2 if present: subtract crop origin (x,y) and keep zero if outside
    for key, coord in (('CRPIX1', x), ('CRPIX2', y)):
        if key in hdr:
            try:
                val = float(hdr[key])
                newv = val - coord
                hdr[key] = newv
            except Exception:
                # leave unchanged if cannot convert
                pass

    # Update or add history
    hdr.add_history(f"Cropped to x={x}, y={y}, w={w}, h={h}")
    return hdr


def _crop_data_array(data: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if data is None:
        return None
    # Support 2D or >2D where last two axes are (Y,X) or first two? Common FITS image is 2D.
    if data.ndim == 2:
        H, W = data.shape
        if x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(f"Crop rectangle out of bounds: data shape={data.shape}, x={x}, y={y}, w={w}, h={h}")
        return data[y:y+h, x:x+w].copy()
    elif data.ndim >= 3:
        # assume the image axes are the last two axes (e.g., (Z, Y, X) or (C, Y, X))
        H, W = data.shape[-2], data.shape[-1]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(f"Crop rectangle out of bounds for array with shape {data.shape}")
        slicer = [slice(None)] * (data.ndim - 2) + [slice(y, y+h), slice(x, x+w)]
        return data[tuple(slicer)].copy()
    else:
        raise ValueError("Unsupported data dimensionality for cropping")


def crop_fits_hdul(hdul: fits.HDUList, x: int, y: int, w: int, h: int, hdu_index: int = 0, preserve_all_hdus: bool = False) -> fits.HDUList:
    """Return a new HDUList with cropped image(s).

    hdul: input HDUList
    x,y,w,h: crop rectangle (zero-based)
    hdu_index: which image HDU to treat as primary when preserve_all_hdus=False
    preserve_all_hdus: if True, crop all image HDUs and preserve their order/types.
    """
    out_hdus = []

    if preserve_all_hdus:
        for hdu in hdul:
            if hasattr(hdu, 'data') and hdu.data is not None:
                newdata = _crop_data_array(hdu.data, x, y, w, h)
                newhdr = _adjust_header_for_crop(hdu.header, x, y, w, h)
                newhdu = fits.ImageHDU(data=newdata, header=newhdr)
                out_hdus.append(newhdu)
            else:
                # non-image HDU: keep as-is (copy)
                out_hdus.append(hdu.copy())
        # ensure primary is PrimaryHDU
        if not isinstance(out_hdus[0], fits.PrimaryHDU):
            # promote first image HDU to Primary
            first = out_hdus[0]
            primary = fits.PrimaryHDU(data=first.data, header=first.header)
            out_hdus = [primary] + out_hdus[1:]
        return fits.HDUList(out_hdus)

    # otherwise only crop the specified hdu_index and write it as PrimaryHDU
    if hdu_index < 0 or hdu_index >= len(hdul):
        raise IndexError('hdu_index out of range')
    target = hdul[hdu_index]
    if target.data is None:
        raise ValueError('Target HDU has no data to crop')
    newdata = _crop_data_array(target.data, x, y, w, h)
    newhdr = _adjust_header_for_crop(target.header, x, y, w, h)
    primary = fits.PrimaryHDU(data=newdata, header=newhdr)
    # Optionally copy over other HDUs as ImageHDU without modification
    out_hdul = fits.HDUList([primary])
    # append other non-primary HDUs as copies (unchanged)
    for i, hdu in enumerate(hdul):
        if i == hdu_index:
            continue
        out_hdul.append(hdu.copy())
    return out_hdul


def crop_fits_file(infile: str, outfile: str, x: int, y: int, w: int, h: int, hdu_index: int = 0, preserve_all_hdus: bool = False, overwrite: bool = False) -> None:
    """Crop FITS file on disk and write result to outfile, preserving header info.

    Coordinates are zero-based (x=column, y=row).
    """
    if os.path.abspath(infile) == os.path.abspath(outfile) and not overwrite:
        raise ValueError('infile and outfile are the same; pass overwrite=True to allow in-place')

    with fits.open(infile, mode='readonly') as hdul:
        new_hdul = crop_fits_hdul(hdul, x, y, w, h, hdu_index=hdu_index, preserve_all_hdus=preserve_all_hdus)
        new_hdul.writeto(outfile, overwrite=overwrite)


# Simple CLI
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Crop FITS image(s) while preserving header metadata')
    p.add_argument('infile')
    p.add_argument('outfile')
    p.add_argument('--x', type=int, required=True, help='x origin (column) zero-based')
    p.add_argument('--y', type=int, required=True, help='y origin (row) zero-based')
    p.add_argument('--w', type=int, required=True, help='crop width')
    p.add_argument('--h', type=int, required=True, help='crop height')
    p.add_argument('--hdu-index', type=int, default=0, help='index of HDU to crop (default 0)')
    p.add_argument('--preserve-all-hdus', action='store_true', help='crop all image HDUs and keep them in output')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    crop_fits_file(args.infile, args.outfile, args.x, args.y, args.w, args.h, hdu_index=args.hdu_index, preserve_all_hdus=args.preserve_all_hdus, overwrite=args.overwrite)


if __name__ == '__main__':  # pragma: no cover
    main()

