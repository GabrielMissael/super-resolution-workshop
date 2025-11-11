"""
Image enhancement utilities: unsharp mask, Laplacian-pyramid sharpening (multi-scale), CLAHE contrast, and denoising.
They avoid external dependencies (use only numpy and OpenCV) and include a small CLI to test/apply to a debayered NPZ.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import argparse


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img2 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img2


def unsharp_mask_color(img: np.ndarray, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """Apply unsharp mask to a color image (operates on luminance channel).

    img: HxWx3 uint8 (BGR)
    sigma: Gaussian blur sigma (controls radius)
    amount: sharpening strength (1.0 = add 100% of mask)
    threshold: minimal brightness difference to sharpen
    """
    ycrcb = None
    if img.ndim == 2:
        gray = img
        color = False
    else:
        # convert to YCrCb for luminance processing
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        gray = ycrcb[..., 0]
        color = True

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = cv2.subtract(gray.astype(np.int16), blurred.astype(np.int16)).astype(np.float32)
    if threshold > 0:
        low_contrast_mask = np.abs(mask) < threshold
        mask[low_contrast_mask] = 0.0
    sharpened_gray = (gray.astype(np.float32) + amount * mask)
    sharpened_gray = np.clip(sharpened_gray, 0, 255).astype(np.uint8)

    if color:
        # ycrcb guaranteed to be set when color is True
        ycrcb[..., 0] = sharpened_gray
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return out
    else:
        return sharpened_gray


def laplacian_pyramid_sharpen(img: np.ndarray, levels: int = 3, weight: float = 1.0) -> np.ndarray:
    """Multi-scale sharpening via Laplacian pyramid.

    img: HxWx3 uint8 (BGR) or HxW uint8
    weight: how much to amplify detail layers (1.0 leaves as-is, >1 amplifies)
    """
    is_color = (img.ndim == 3 and img.shape[2] == 3)
    if is_color:
        channels = cv2.split(img)
    else:
        channels = [img]

    out_ch = []
    for ch in channels:
        chf = ch.astype(np.float32)
        gp = [chf]
        for i in range(levels):
            gp.append(cv2.pyrDown(gp[-1]))
        # build laplacian
        lp = []
        for i in range(levels):
            size = (gp[i].shape[1], gp[i].shape[0])
            up = cv2.pyrUp(gp[i+1], dstsize=size)
            lap = gp[i] - up
            lp.append(lap)
        base = gp[-1]
        # amplify laplacian layers
        for i in range(len(lp)):
            lp[i] = lp[i] * (1.0 + weight)
        # reconstruct
        res = base
        for i in range(len(lp)-1, -1, -1):
            size = (lp[i].shape[1], lp[i].shape[0])
            res = cv2.pyrUp(res, dstsize=size) + lp[i]
        res = np.clip(res, 0, 255).astype(np.uint8)
        out_ch.append(res)

    if is_color:
        out = cv2.merge(out_ch)
    else:
        out = out_ch[0]
    return out


def apply_clahe(img: np.ndarray, clipLimit: float = 2.0, tileGridSize: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE on the luminance channel to improve local contrast.

    img: HxWx3 BGR or HxW grayscale
    """
    if img.ndim == 2:
        gray = img
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(gray)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out


def denoise_color(img: np.ndarray, h: float = 10, hColor: float = 10, templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
    """Denoise color image using OpenCV fastNlMeansDenoisingColored.

    h/hColor: strength for luminance/color components.
    """
    out = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
    return out


def adjust_saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust saturation by a multiplicative factor in HSV space.

    img: HxWx3 BGR uint8
    factor: >0, 1.0 means no change, >1 increases saturation, <1 decreases.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_hue(img: np.ndarray, shift_deg: float = 0.0) -> np.ndarray:
    """Shift hue by shift_deg degrees (approx) in HSV space.

    OpenCV H range is [0,179] mapping to 0-360 degrees (factor ~0.5).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    # convert degrees to OpenCV H units
    delta = int(np.round(shift_deg / 2.0))
    hsv[..., 0] = (hsv[..., 0] + delta) % 180
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_vibrance(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust vibrance: increase saturation more for less-saturated pixels.

    factor: 1.0 = no change; >1 increases vibrance; <1 reduces.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    # amount to add is proportional to (1 - s/255) so low-sat pixels get a bigger boost
    add = (255.0 - s) * (factor - 1.0) * 0.5
    s2 = np.clip(s + add, 0, 255)
    hsv[..., 1] = s2
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast_brightness(img: np.ndarray, contrast: float = 1.0, brightness: float = 0.0) -> np.ndarray:
    """Adjust contrast (alpha) and brightness (beta) using linear transform.

    new = img * contrast + brightness
    contrast: 1.0 = no change; >1 increases contrast
    brightness: added value in 0-255 range (can be negative)
    """
    imgf = img.astype(np.float32)
    out = imgf * float(contrast) + float(brightness)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def enhance_color(img: np.ndarray,
                  saturation: float = 1.0,
                  vibrance: float = 1.0,
                  hue_shift: float = 0.0,
                  contrast: float = 1.0,
                  brightness: float = 0.0) -> np.ndarray:
    """Convenience function to adjust color properties in a chain.

    Order: hue -> saturation/vibrance -> contrast/brightness
    """
    out = img.copy()
    if hue_shift != 0.0:
        out = adjust_hue(out, hue_shift)
    if saturation != 1.0:
        out = adjust_saturation(out, saturation)
    if vibrance != 1.0:
        out = adjust_vibrance(out, vibrance)
    if contrast != 1.0 or brightness != 0.0:
        out = adjust_contrast_brightness(out, contrast=contrast, brightness=brightness)
    return out


def enhance_pipeline(img: np.ndarray,
                     clahe_clip: float = 2.0,
                     denoise_h: float = 8.0,
                     pyramid_levels: int = 3,
                     pyramid_weight: float = 0.8,
                     usm_sigma: float = 1.0,
                     usm_amount: float = 0.4,
                     # color tweaks
                     saturation: float = 1.0,
                     vibrance: float = 1.0,
                     hue_shift: float = 0.0,
                     contrast: float = 1.0,
                     brightness: float = 0.0) -> np.ndarray:
    """Run a recommended enhancement pipeline on a BGR uint8 image and return enhanced image.

    Steps:
      1. CLAHE on luminance
      2. Denoise (non-local means)
      3. Laplacian pyramid sharpening
      4. Small unsharp mask pass
      5. Color tweaks (saturation/vibrance/contrast)
    """
    # 1: CLAHE
    img1 = apply_clahe(img, clipLimit=clahe_clip)
    # 2: denoise
    img2 = denoise_color(img1, h=denoise_h, hColor=denoise_h)
    # 3: pyramid sharpen
    img3 = laplacian_pyramid_sharpen(img2, levels=pyramid_levels, weight=pyramid_weight)
    # 4: unsharp mask (light)
    out = unsharp_mask_color(img3, sigma=usm_sigma, amount=usm_amount, threshold=0)
    # 5: color tweaks
    out = enhance_color(out, saturation=saturation, vibrance=vibrance, hue_shift=hue_shift,
                        contrast=contrast, brightness=brightness)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='input npz of debayered frames (frames key)')
    p.add_argument('--out-dir', default='data/enhance_test')
    p.add_argument('--max-frames', type=int, default=10)
    p.add_argument('--preview', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(inp))
    frames = data['frames']
    if frames.ndim == 3:
        # (T,H,W) grayscale -> treat as single-channel
        raise RuntimeError('Please provide debayered (color) frames (T,H,W,3)')
    T = min(args.max_frames, frames.shape[0])
    enhanced = []
    for i in range(T):
        img = frames[i]
        # if grayscale, convert to BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = enhance_pipeline(img)
        enhanced.append(out)
        if args.preview and i == 0:
            cv2.imwrite(str(out_dir / 'enhanced_preview0.png'), out)
            print('Saved preview:', out_dir / 'enhanced_preview0.png')
    enhanced = np.stack(enhanced, axis=0)
    np.savez_compressed(out_dir / 'enhanced_frames_uint8.npz', frames=enhanced)
    print('Saved enhanced frames to:', out_dir / 'enhanced_frames_uint8.npz')


if __name__ == '__main__':
    main()
