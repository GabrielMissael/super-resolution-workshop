from pathlib import Path
import cv2
import numpy as np


def output_npz():
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


def output_avi():
    src = r"C:\Users\npayo\Desktop\SharpCap Captures\2025-02-04\Jupiter\21_04_55.avi"
    dst = Path("../data/jupiter_frames.avi")
    max_frames = 8000

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

    frames = np.stack(frames, axis=0).astype(np.uint8)
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(dst, fourcc, fps, (w, h), isColor=False)

    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Saved grayscale AVI to {dst}")


def get_frames(avi_path):
    cap = cv2.VideoCapture(avi_path)
    frames_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_list.append(frame)

    cap.release()
    frames_array = np.stack(frames_list)
    return frames_array


if __name__ == '__main__':
    output_avi()
