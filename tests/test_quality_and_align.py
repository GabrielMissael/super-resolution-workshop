import numpy as np
from src import quality_and_align as qa


def make_blurry_and_sharp_frames(N=5, H=32, W=32):
    # create a base image with a sharp circle; blurred versions will have lower focus metrics
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H/2, W/2
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    base = (255 * (r < (min(H,W)/4))).astype(np.uint8)
    frames = []
    for i in range(N):
        if i % 2 == 0:
            # sharp
            frames.append(base.copy())
        else:
            # blurred
            frames.append(qa.laplacian_variance(base))
    # ensure frames array shape (N,H,W)
    frames = np.stack([base for _ in range(N)], axis=0)
    return frames


def test_quality_metrics_normalization_and_range():
    frames = make_blurry_and_sharp_frames(4, 32, 32)
    scores, lap, ten = qa.compute_quality_scores(frames)
    assert scores.shape[0] == frames.shape[0]
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert lap.shape == (frames.shape[0],)
    assert ten.shape == (frames.shape[0],)


def test_choose_reference_frame_and_detection():
    frames = np.stack([np.zeros((32,32), dtype=np.uint8) + i*10 for i in range(6)], axis=0)
    scores = np.linspace(0, 1, frames.shape[0])
    idx = qa.choose_reference_frame(scores, prefer_range=3)
    assert isinstance(idx, int)
    # detect_ref_points should run and return array-like
    pts = qa.detect_ref_points(frames[0], max_corners=50)
    assert pts.ndim == 2


def test_save_results_creates_files(tmp_path):
    frames = np.zeros((2, 16, 16), dtype=np.uint8)
    fps = 24.0
    scores = np.array([0.1, 0.9])
    lap = np.array([1.0, 2.0])
    ten = np.array([3.0, 4.0])
    transforms = [None, np.array([[1,0,0],[0,1,0]])]
    ref_idx = 0
    ref_pts = np.array([[1.0,2.0],[3.0,4.0]])
    out_dir = tmp_path / "qa_out"
    qa.save_results(out_dir, frames, fps, scores, lap, ten, transforms, ref_idx, ref_pts)
    # check files
    assert (out_dir / "aligned_frames_uint8.npz").exists()
    assert (out_dir / "quality_and_transforms.csv").exists()
    assert (out_dir / "ref_points.npy").exists()

