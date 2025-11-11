import cv2
import os
import numpy

input_path = r"C:\Users\npayo\Desktop\SharpCap Captures\2025-02-04\Jupiter\21_04_55.avi"
out_path = r"../data/jupiter.avi"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Cannot find `{input_path}`")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open `{input_path}`")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if w == 0 or h == 0:
    raise RuntimeError("Captured frame size is zero")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError("Failed to open VideoWriter. Try a different fourcc/extension")

max_frames = 30 * 300
count = 0
while count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    writer.write(frame)
    count += 1

cap.release()
writer.release()

print(f"Wrote {count} frames to `{out_path}` (fps={fps}, size={w}x{h})")