"""
Visualization utilities for Jupiter frames (works inside Jupyter notebooks too).

Provides:
- compute_quality_scores(frames)
- save_quality_plot(scores, out_path)
- save_before_after_pair(orig_frames, aligned_frames, idx, out_path, ref_pts=None)
- a CLI to generate preview images (quality plot and a before/after composite)

Usage (from project root):
    python src\visualize_results.py --input data\jupiter_frames_uint8.npz --aligned data\aligned_test2\aligned_frames_uint8.npz --out-dir data\vis_test --max-frames 60

The saved images can be displayed in a notebook using IPython.display.Image or plt.imshow.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_npz(path: Path, max_frames: int | None = None):
    data = np.load(str(path))
    frames = data["frames"]
    fps = float(data.get("fps", 30.0))
    if max_frames is not None:
        frames = frames[:max_frames]
    if frames.ndim == 4 and frames.shape[-1] == 3:
        frames = frames[..., 0]
    frames = frames.astype(np.uint8)
    return frames, fps


# Duplicate quality metrics used in quality_and_align for visualization convenience

def laplacian_variance(img: np.ndarray) -> float:
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def tenengrad(img: np.ndarray) -> float:
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx * gx + gy * gy
    return float(fm.mean())


def compute_quality_scores(frames: np.ndarray):
    n = len(frames)
    lap = np.zeros(n, dtype=float)
    ten = np.zeros(n, dtype=float)
    for i in range(n):
        lap[i] = laplacian_variance(frames[i])
        ten[i] = tenengrad(frames[i])
    def norm(x):
        if np.all(np.isclose(x, x[0])):
            return np.zeros_like(x)
        x = np.array(x, dtype=float)
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    lap_n = norm(lap)
    ten_n = norm(ten)
    score = 0.45 * lap_n + 0.55 * ten_n
    return score, lap, ten


def save_quality_plot(scores: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(scores, '-k')
    ax.set_xlabel('frame')
    ax.set_ylabel('quality (normalized)')
    ax.set_title('Per-frame quality scores')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved quality plot to: {out_path}")


def overlay_points(img, pts, color=(1.0, 0, 0), alpha=0.1, radius=3):
    """
    Draws colored points with transparency (alpha) on a grayscale or RGB image.

    Args:
        img: Input image, shape (H, W) or (H, W, 3)
        pts: Nx2 array of coordinates (x, y)
        color: RGB tuple, values in 0-1
        alpha: Opacity of the drawn dots, between 0 (transparent) and 1 (opaque)
        radius: Circle radius in pixels
    Returns:
        Numpy array in uint8 with points drawn and alpha blended.
    """
    # Make sure image is 3-channel
    if img.ndim == 2:
        base = np.stack([img] * 3, axis=-1).astype(np.float32)
    else:
        base = img.astype(np.float32).copy()

    overlay = np.zeros_like(base)

    dot_color = tuple(int(255 * c) for c in color)
    for (x, y) in pts:
        cv2.circle(overlay, (int(x), int(y)), radius=radius, color=dot_color, thickness=-1)

    # Create a mask where circles are drawn
    mask = (overlay > 0).any(axis=2, keepdims=True).astype(np.float32)
    # Alpha blending only where mask is 1
    result = base * (1 - alpha * mask) + overlay * (alpha * mask)

    return np.clip(result, 0, 255).astype(np.uint8)


def save_before_after_pair(orig_frames: np.ndarray, aligned_frames: np.ndarray, idx: int, out_path: Path, ref_pts: np.ndarray | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(orig_frames)
    idx = int(max(0, min(n-1, idx)))
    orig = orig_frames[idx]
    if aligned_frames is None:
        aligned = orig
    else:
        aligned = aligned_frames[idx]
    if ref_pts is not None and len(ref_pts) > 0:
        orig_rgb = overlay_points(orig, ref_pts, color=(1.0, 0.0, 0.0))
        aligned_rgb = overlay_points(aligned, ref_pts, color=(0.0, 1.0, 0.0))
    else:
        orig_rgb = np.stack([orig, orig, orig], axis=-1)
        aligned_rgb = np.stack([aligned, aligned, aligned], axis=-1)
    # create side-by-side
    h, w = orig.shape
    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
    canvas[:, :w] = orig_rgb
    canvas[:, w:] = aligned_rgb
    # annotate
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(canvas)
    ax.axis('off')
    ax.set_title(f'Frame {idx}  (left: original, right: aligned)')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved before/after image to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--aligned", required=False, help="aligned npz path (optional)")
    p.add_argument("--out-dir", default="data/vis", help="output dir for preview images")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--frame-idx", type=int, default=None, help="index for before/after preview, default=best")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    aligned_path = Path(args.aligned) if args.aligned else None
    out_dir = Path(args.out_dir)

    frames, fps = load_npz(inp, max_frames=args.max_frames)
    print(f"Loaded {len(frames)} frames from {inp}")
    scores, lap, ten = compute_quality_scores(frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_quality_plot(scores, out_dir / "quality_plot.png")

    # choose frame index for preview
    if args.frame_idx is not None:
        idx = int(args.frame_idx)
    else:
        idx = int(np.argmax(scores))
    aligned_frames = None
    if aligned_path is not None and aligned_path.exists():
        try:
            aligned_frames, _ = load_npz(aligned_path, max_frames=args.max_frames)
            print(f"Loaded aligned frames from {aligned_path}")
        except Exception:
            aligned_frames = None
    # try to detect reference points on the chosen frame
    try:
        ref_pts = cv2.goodFeaturesToTrack(frames[idx], maxCorners=400, qualityLevel=0.01, minDistance=8)
        if ref_pts is not None:
            ref_pts = ref_pts.reshape(-1, 2)
        else:
            ref_pts = np.zeros((0,2), dtype=np.float32)
    except Exception:
        ref_pts = np.zeros((0,2), dtype=np.float32)

    # If aligned frames exist but are shorter than original frames, adjust index used for the aligned array
    if aligned_frames is not None and len(aligned_frames) > 0:
        if idx >= len(aligned_frames):
            print(f"Warning: chosen frame index {idx} >= number of aligned frames ({len(aligned_frames)}). Clamping to {len(aligned_frames)-1} for the aligned preview.")
            aligned_idx = len(aligned_frames) - 1
        else:
            aligned_idx = idx
    else:
        aligned_idx = idx

    save_before_after_pair(frames, aligned_frames, aligned_idx, out_dir / "before_after.png", ref_pts=ref_pts)


if __name__ == '__main__':
    main()
