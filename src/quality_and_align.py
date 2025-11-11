"""
Quality estimation and alignment for Jupiter frames.

Usage (from project root):
    python src\quality_and_align.py --input data\jupiter_frames_uint8.npz --out-dir data\aligned --max-frames 400

What it does:
- Loads frames and fps from the provided NPZ file.
- Computes per-frame quality scores (Laplacian variance and Tenengrad energy).
- Picks a reference frame (highest quality by default).
- Detects alignment points (goodFeaturesToTrack) on the reference frame.
- Tracks points in each frame using calcOpticalFlowPyrLK.
- Estimates an affine transform per-frame using RANSAC and warps frames to the reference.
- Saves aligned frames (npz), a CSV with quality scores and transform metadata, and reference alignment points.

The script requires OpenCV (cv2) and numpy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import numpy as np
from typing import Tuple

try:
    import cv2
except Exception as e:
    print("OpenCV (cv2) is required but not installed. Install with: pip install opencv-python")
    raise


def load_npz(path: Path, max_frames: int | None = None):
    data = np.load(str(path))
    frames = data["frames"]
    fps = float(data.get("fps", 30.0))
    if max_frames is not None:
        frames = frames[:max_frames]
    # ensure frames are uint8 and 2D (grayscale)
    if frames.ndim == 4 and frames.shape[-1] == 3:
        frames = frames[..., 0]
    frames = frames.astype(np.uint8)
    return frames, fps


# Quality metrics

def laplacian_variance(img: np.ndarray) -> float:
    # img: uint8 grayscale
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def tenengrad(img: np.ndarray) -> float:
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx * gx + gy * gy
    return float(fm.mean())


def compute_quality_scores(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frames)
    lap = np.zeros(n, dtype=float)
    ten = np.zeros(n, dtype=float)
    for i in range(n):
        img = frames[i]
        lap[i] = laplacian_variance(img)
        ten[i] = tenengrad(img)
    # normalize each metric to [0,1]
    def norm(x):
        if np.all(np.isclose(x, x[0])):
            return np.zeros_like(x)
        x = np.array(x, dtype=float)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x
    lap_n = norm(lap)
    ten_n = norm(ten)
    # weighted sum (favor Tenengrad slightly)
    score = 0.45 * lap_n + 0.55 * ten_n
    return score, lap, ten


# Alignment

def choose_reference_frame(scores: np.ndarray, prefer_range: int = -1) -> int:
    # prefer the best within the first prefer_range frames (if available)
    n = len(scores)
    r = min(prefer_range, n)
    idx = int(np.argmax(scores[:r]))
    return idx


def detect_ref_points(ref_img: np.ndarray, max_corners: int = 300, quality_level: float = 0.01, min_distance: int = 8):
    if ref_img.ndim == 3 and ref_img.shape[2] == 3:
        ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    else:
        ref_img_gray = ref_img.astype(np.float32) / 255

    pts = cv2.goodFeaturesToTrack(ref_img_gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def align_frames(frames: np.ndarray,
                 ref_idx: int,
                 ref_pts: np.ndarray,
                 win_size=(21, 21),
                 max_level=3,
                 min_inliers=6):
    n = len(frames)
    h, w, c = frames[0].shape
    aligned = np.zeros_like(frames)
    transforms = []  # store affine matrices (2x3) or None
    ref_img = frames[ref_idx]
    aligned[ref_idx] = ref_img.copy()

    lk_params = dict(winSize=win_size, maxLevel=max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Use forward-backward check by tracking to frame and back
    for i in range(n):
        if i == ref_idx:
            transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=float))
            continue
        img = frames[i]
        if ref_pts.shape[0] < 3:
            transforms.append(None)
            aligned[i] = img.copy()
            continue
        # track ref_pts -> points in img
        pts1 = ref_pts.reshape(-1, 1, 2)
        pts2, st, err = cv2.calcOpticalFlowPyrLK(ref_img, img, pts1, None, **lk_params)
        # track back to ref for fb check
        pts1_rb, st2, err2 = cv2.calcOpticalFlowPyrLK(img, ref_img, pts2, None, **lk_params)
        st = st.reshape(-1)
        st2 = st2.reshape(-1)
        pts2 = pts2.reshape(-1, 2)
        pts1_rb = pts1_rb.reshape(-1, 2)
        pts1 = pts1.reshape(-1, 2)
        # keep points with forward and backward success and small displacement
        good = (st == 1) & (st2 == 1)
        # fb error
        fb_err = np.linalg.norm(pts1 - pts1_rb, axis=1)
        good &= (fb_err < 1.5)
        if np.count_nonzero(good) < min_inliers:
            # fallback: try feature matching (ORB)
            M = _align_with_orb(ref_img, img)
            if M is None:
                transforms.append(None)
                aligned[i] = img.copy()
            else:
                aligned[i] = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                transforms.append(M)
            continue
        src_pts = pts1[good]
        dst_pts = pts2[good]
        # estimate affine with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            transforms.append(None)
            aligned[i] = img.copy()
            continue
        aligned_i = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned[i] = aligned_i
        transforms.append(M)
    return aligned, transforms


def _align_with_orb(ref_img: np.ndarray, img: np.ndarray):
    # fallback using ORB feature detection + BFMatcher + estimateAffinePartial2D
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    kp2, des2 = orb.detectAndCompute(img, None)
    if des1 is None or des2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:500]
    if len(matches) < 6:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M


def save_results(out_dir: Path, aligned: np.ndarray, fps: float, scores: np.ndarray, lap: np.ndarray, ten: np.ndarray, transforms: list, ref_idx: int, ref_pts: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = out_dir / "aligned_frames_uint8.npz"
    np.savez_compressed(aligned_path, frames=aligned, fps=fps)
    csv_path = out_dir / "quality_and_transforms.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score", "laplacian", "tenengrad", "has_transform", "m00", "m01", "m02", "m10", "m11", "m12"])
        for i, (s, l, t, M) in enumerate(zip(scores, lap, ten, transforms)):
            if M is None:
                writer.writerow([i, float(s), float(l), float(t), 0, "", "", "", "", "", ""])
            else:
                writer.writerow([i, float(s), float(l), float(t), 1, float(M[0,0]), float(M[0,1]), float(M[0,2]), float(M[1,0]), float(M[1,1]), float(M[1,2])])
    np.save(out_dir / "ref_points.npy", ref_pts)
    print(f"Saved aligned frames to: {aligned_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved ref points to: {out_dir / 'ref_points.npy'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input npz with frames and optional fps")
    p.add_argument("--out-dir", default="data/aligned", help="output directory")
    p.add_argument("--max-frames", type=int, default=600, help="max frames to process (for speed)" )
    p.add_argument("--n-ref-points", type=int, default=400, help="number of alignment points to detect in reference")
    p.add_argument("--ref-index", type=int, default=None, help="manual reference frame index")
    p.add_argument("--prefer-range", type=int, default=120, help="search best frame within this range")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    frames, fps = load_npz(inp, max_frames=args.max_frames)
    n = len(frames)
    print(f"Loaded {n} frames @ {fps:.2f} fps from {inp}")

    print("Computing quality scores...")
    scores, lap, ten = compute_quality_scores(frames)

    if args.ref_index is not None:
        ref_idx = int(args.ref_index)
    else:
        ref_idx = choose_reference_frame(scores, prefer_range=args.prefer_range)
    print(f"Reference frame chosen: {ref_idx}")

    print("Detecting reference alignment points...")
    ref_pts = detect_ref_points(frames[ref_idx], max_corners=args.n_ref_points)
    print(f"Detected {len(ref_pts)} reference points")

    print("Aligning frames (this may take a while)...")
    aligned, transforms = align_frames(frames, ref_idx, ref_pts)

    print("Saving results...")
    save_results(out_dir, aligned, fps, scores, lap, ten, transforms, ref_idx, ref_pts)


if __name__ == '__main__':
    main()
