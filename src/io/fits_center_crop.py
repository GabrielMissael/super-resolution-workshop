"""Center-crop FITS files preserving header metadata.

Usage examples (Windows cmd):
  python -m src.fits_center_crop --infile "C:\\...\\frame_00001.fits" --outfile "C:\\...\\frame_00001_crop.fits" --width 3000 --height 2000

  # Batch mode: process all .fits in input directory (non-recursive)
  python -m src.fits_center_crop --indir "C:\\path\\to\\rawframes" --outdir "C:\\path\\to\\cropped" --width 3000 --height 2000

The script will call `src.fits_crop.crop_fits_file` to preserve headers and adjust CRPIX values.
"""
from __future__ import annotations
import os
from pathlib import Path
import argparse
from typing import Optional

try:
    from astropy.io import fits
except Exception as e:  # pragma: no cover
    raise ImportError("astropy is required. Install with: pip install astropy") from e

from src import fits_crop


def compute_center_crop_coords(w_in: int, h_in: int, w_out: int, h_out: int) -> tuple[int, int, int, int]:
    """Return (x, y, w_out, h_out) zero-based crop rectangle centered in input."""
    if w_out > w_in or h_out > h_in:
        raise ValueError("Requested crop is larger than input image")
    x = (w_in - w_out) // 2
    y = (h_in - h_out) // 2
    return x, y, w_out, h_out


def crop_file(infile: Path, outfile: Path, width: int, height: int, overwrite: bool = False) -> None:
    # read header to get shape (without loading full data if possible)
    with fits.open(str(infile), mode='readonly', memmap=False) as hdul:
        # assume primary HDU contains image data
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No image data found in primary HDU of {infile}")
        # determine shape: data can be (H,W) or (...,H,W) where H/W are last two dims
        if data.ndim == 2:
            h_in, w_in = data.shape
        else:
            h_in, w_in = data.shape[-2], data.shape[-1]
    x, y, w, h = compute_center_crop_coords(w_in, h_in, width, height)
    print(f"Cropping {infile} -> {outfile}: x={x}, y={y}, w={w}, h={h} (in {w_in}x{h_in})")
    fits_crop.crop_fits_file(str(infile), str(outfile), x=x, y=y, w=w, h=h, hdu_index=0, preserve_all_hdus=False, overwrite=overwrite)


def batch_crop(indir: Path, outdir: Path, width: int, height: int, overwrite: bool = False, recursive: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pattern = "**/*.fits" if recursive else "*.fits"
    files = list(indir.glob(pattern))
    if not files:
        print(f"No .fits files found in {indir} (recursive={recursive})")
        return
    for f in files:
        rel = f.name
        outp = outdir / rel
        # avoid overwriting unless allowed
        if outp.exists() and not overwrite:
            print(f"Skipping existing {outp} (use --overwrite to replace)")
            continue
        try:
            crop_file(f, outp, width, height, overwrite=overwrite)
        except Exception as e:
            print(f"Failed to crop {f}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description='Center-crop FITS files while preserving headers')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--infile', help='single FITS file to crop')
    g.add_argument('--indir', help='directory with .fits files to crop (non-recursive by default)')
    p.add_argument('--outfile', help='output file for single input (required with --infile)')
    p.add_argument('--outdir', help='output directory for batch mode (required with --indir)')
    p.add_argument('--width', type=int, required=True, help='output crop width (pixels)')
    p.add_argument('--height', type=int, required=True, help='output crop height (pixels)')
    p.add_argument('--overwrite', action='store_true', help='overwrite existing outputs')
    p.add_argument('--recursive', action='store_true', help='search input directory recursively')
    return p.parse_args()


def main():
    args = parse_args()
    if args.infile:
        if not args.outfile:
            raise SystemExit('When using --infile, --outfile is required')
        infile = Path(args.infile)
        outfile = Path(args.outfile)
        crop_file(infile, outfile, args.width, args.height, overwrite=args.overwrite)
    else:
        indir = Path(args.indir)
        outdir = Path(args.outdir) if args.outdir else (indir / 'cropped')
        batch_crop(indir, outdir, args.width, args.height, overwrite=args.overwrite, recursive=args.recursive)


if __name__ == '__main__':  # pragma: no cover
    main()

