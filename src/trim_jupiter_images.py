from pathlib import Path
import cv2
import numpy as np

src = r"C:\Users\npayo\Desktop\SharpCap Captures\2025-02-04\Jupiter\21_04_55.avi"
dst = Path("../data/jupiter_frames_uint8.npz")
max_frames = 9000

cap = cv2.VideoCapture(src)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {src}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or not np.isfinite(fps) or fps < 1 or fps > 300:
    fps = 30.0

frames = []
while len(frames) < max_frames:
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame[..., 0])

cap.release()

arr = np.stack(frames, axis=0).astype(np.uint8)  # shape: (T, H, W, 3)
np.savez_compressed(dst, frames=arr, fps=fps)
print(f"Saved {arr.shape} @ {fps:.2f} fps -> {dst}")
