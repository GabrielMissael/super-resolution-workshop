"""
Debayer (demosaic) utilities for RGGB Bayer frames.

Usage (from project root):
    python src\debayer.py --input data\jupiter_frames_uint8.npz --out-dir data\debayer_test --max-frames 10 --method BILINEAR

The script supports methods: BILINEAR (default), VNG (if OpenCV supports it), EA (edge-aware if supported).
Assumes input frames are single-channel uint8 Bayer mosaics in RGGB ordering (top-left is R).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import cv2
import os


def _method_flag(pattern: str, method: str) -> int:
    # pattern: one of 'RGGB', 'BGGR', 'GRBG', 'GBRG'
    pat = pattern.upper()
    method = method.upper()
    base_name = f'COLOR_BAYER_{pat}'  # e.g. COLOR_BAYER_RGGB not existing; mapping needed
    # OpenCV uses names like COLOR_BAYER_RG2BGR (pattern code 2) where RG means top-left R
    # Map patterns to short code used by OpenCV functions
    short = {
        'RGGB': 'RG',
        'BGGR': 'BG',
        'GRBG': 'GR',
        'GBRG': 'GB',
    }.get(pat, 'RG')

    # method suffixes
    if method == 'BILINEAR' or method == 'DEFAULT':
        suffix = ''
    elif method == 'VNG':
        suffix = '_VNG'
    elif method == 'EA' or method == 'EDGEAWARE' or method == 'EDGE_AWARE':
        suffix = '_EA'
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build attribute name like COLOR_BAYER_RG2BGR_VNG
    attr_name = f'COLOR_BAYER_{short}2BGR{suffix}'
    if hasattr(cv2, attr_name):
        return getattr(cv2, attr_name)
    # try BGRA variant (in case BGR->BGRA flag exists)
    attr_name2 = f'COLOR_BAYER_{short}2BGRA{suffix}'
    if hasattr(cv2, attr_name2):
        return getattr(cv2, attr_name2)
    # fallback: try generic COLOR_BAYER_RG2BGR
    fallback = f'COLOR_BAYER_{short}2BGR'
    if hasattr(cv2, fallback):
        return getattr(cv2, fallback)
    # final fallback: use integer 46 which often corresponds to RG2BGR; but avoid hardcoding
    raise RuntimeError(f"Demosaic flag not found in OpenCV for pattern={pattern}, method={method}")


def debayer_image(raw: np.ndarray, pattern: str = 'RGGB', method: str = 'BILINEAR') -> np.ndarray:
    """Convert single-channel Bayer raw image to BGR color image (uint8).

    raw: HxW uint8
    returns: HxWx3 uint8 (BGR)
    """
    if raw.ndim != 2:
        raise ValueError('raw must be a 2D single-channel image')
    flag = _method_flag(pattern, method)
    # cvtColor expects the raw image as uint8 or uint16 depending on depth
    out = cv2.cvtColor(raw, flag)
    return out


def debayer_stack(frames: np.ndarray, pattern: str = 'RGGB', method: str = 'BILINEAR') -> np.ndarray:
    """Debayer a stack of single-channel frames.

    frames: (T, H, W) uint8
    returns: (T, H, W, 3) uint8
    """
    if frames.ndim != 3:
        raise ValueError('frames must be (T, H, W)')
    T = frames.shape[0]
    out_list = []
    flag = _method_flag(pattern, method)
    for i in range(T):
        raw = frames[i]
        out = cv2.cvtColor(raw, flag)
        out_list.append(out)
    return np.stack(out_list, axis=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out-dir', default='data/debayer', help='output directory')
    p.add_argument('--max-frames', type=int, default=100)
    p.add_argument('--pattern', default='RGGB', help='Bayer pattern: RGGB, BGGR, GRBG, GBRG')
    p.add_argument('--method', default='BILINEAR', help='demosaic method: BILINEAR, VNG, EA')
    p.add_argument('--preview', action='store_true', help='save preview PNG for first frame')
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(inp))
    frames = data['frames']
    fps = float(data.get('fps', 30.0))
    if frames.ndim == 4 and frames.shape[-1] == 3:
        # if frames are 3-channel, attempt to interpret as already debayered; just copy
        print('Input frames already 3-channel; skipping debayer and saving copy')
        deb = frames.astype(np.uint8)
    else:
        # assume single-channel
        if args.max_frames is not None:
            frames2 = frames[:args.max_frames]
        else:
            frames2 = frames
        print(f'Debayering {len(frames2)} frames with pattern={args.pattern} method={args.method}')
        deb = debayer_stack(frames2, pattern=args.pattern, method=args.method)
    # save result
    np.savez_compressed(out_dir / 'debayered_frames_uint8.npz', frames=deb, fps=fps)
    print(f'Saved debayered frames to: {out_dir / "debayered_frames_uint8.npz"}')

    if args.preview:
        try:
            import imageio
            preview = deb[0]
            # convert BGR->RGB for saving
            preview_rgb = preview[..., ::-1]
            imageio.imwrite(str(out_dir / 'preview_frame0.png'), preview_rgb)
            print(f'Saved preview to: {out_dir / "preview_frame0.png"}')
        except Exception as e:
            print('Preview save failed:', e)


if __name__ == '__main__':
    main()

