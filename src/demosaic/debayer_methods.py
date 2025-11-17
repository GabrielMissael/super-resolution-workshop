"""PyTorch implementations of simple VNG (Variable Number of Gradients) and
Edge-Aware demosaicing for Bayer raw images.

These implementations are educational, simplified approximations of the
respective algorithms. They trade accuracy for clarity and vectorized
operations in PyTorch. For production-quality demosaicing, prefer
specialized libraries or OpenCV's built-in methods.

API (all functions accept uint8 or uint16 numpy arrays, 2D Bayer image):
    debayer_vng_torch(raw: np.ndarray, pattern: str = 'RGGB', device='cpu') -> np.ndarray
    debayer_edgeaware_torch(raw: np.ndarray, pattern: str = 'RGGB', device='cpu') -> np.ndarray
    debayer_stack_vng(frames: np.ndarray, pattern='RGGB', device='cpu') -> np.ndarray
    debayer_stack_edgeaware(frames: np.ndarray, pattern='RGGB', device='cpu') -> np.ndarray

Patterns supported: 'RGGB', 'BGGR', 'GRBG', 'GBRG'.
Returned image dtype is uint8 (values clipped / scaled if input is higher bit depth).

Edge cases & notes:
- Borders are handled with simple replication padding.
- If torch is not installed, an ImportError is raised.
- For uint16 input (>255), values are linearly scaled down to 0-255 for processing.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required for debayer_methods. Please install torch.") from e

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pattern_masks(h: int, w: int, pattern: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return boolean masks (torch.bool) for R, G, B given Bayer pattern.

    pattern codes:
        RGGB: R G / G B
        BGGR: B G / G R
        GRBG: G R / B G
        GBRG: G B / R G
    """
    pat = pattern.upper()
    yy = torch.arange(h).view(-1, 1)
    xx = torch.arange(w).view(1, -1)
    even_y = (yy % 2 == 0)
    even_x = (xx % 2 == 0)
    odd_y = ~even_y
    odd_x = ~even_x

    if pat == 'RGGB':
        R = even_y & even_x
        B = odd_y & odd_x
        G = (even_y & odd_x) | (odd_y & even_x)
    elif pat == 'BGGR':
        B = even_y & even_x
        R = odd_y & odd_x
        G = (even_y & odd_x) | (odd_y & even_x)
    elif pat == 'GRBG':
        G = even_y & even_x | odd_y & odd_x  # diag greens
        R = even_y & odd_x
        B = odd_y & even_x
    elif pat == 'GBRG':
        G = even_y & even_x | odd_y & odd_x
        B = even_y & odd_x
        R = odd_y & even_x
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    return R.to(torch.bool), G.to(torch.bool), B.to(torch.bool)


def _prepare_raw(raw: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    if raw.ndim != 2:
        raise ValueError('raw must be 2D')
    # Accept uint8 or uint16; scale if needed
    if raw.dtype == np.uint16:
        arr = (raw.astype(np.float32) / 257.0)  # approx scale 0-65535 -> 0-255
    else:
        arr = raw.astype(np.float32)
    return torch.from_numpy(arr).to(device)


def _to_uint8_image(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    img = torch.stack([r, g, b], dim=0)  # (3, H, W)
    img = torch.clamp(img, 0.0, 255.0).round().to(torch.uint8)
    # Convert to numpy HxWx3 BGR (to stay consistent with OpenCV style used in repo)
    np_img = img.permute(1, 2, 0).cpu().numpy()
    # Current order is [R,G,B]; convert to BGR by reversing channels
    bgr = np_img[:, :, ::-1].copy()
    return bgr


def _replicate_pad(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)

# ---------------------------------------------------------------------------
# VNG Demosaicing (Simplified)
# ---------------------------------------------------------------------------

def debayer_vng_torch(raw: np.ndarray, pattern: str = 'RGGB', device: str = 'cpu') -> np.ndarray:
    """Simplified VNG demosaicing.

    Steps (approximation):
    1. Build masks for R,G,B.
    2. For each missing channel value at pixel, evaluate gradients to 8 neighbors.
    3. Determine threshold = 1.2 * median(gradients); include neighbors with gradient <= threshold.
    4. Average available neighbor samples of that channel within selected directions.
    5. Assemble channels and clip.

    Returns BGR uint8 image.
    """
    t = _prepare_raw(raw, device)
    h, w = t.shape
    Rmask, Gmask, Bmask = _pattern_masks(h, w, pattern)

    # Known channel values
    R = torch.where(Rmask, t, torch.zeros_like(t))
    G = torch.where(Gmask, t, torch.zeros_like(t))
    B = torch.where(Bmask, t, torch.zeros_like(t))

    # Count of known positions per channel for normalization later
    zero = torch.zeros_like(t)

    # Pad raw for neighbor access
    tp = _replicate_pad(t)

    # Neighbor coordinates (offsets)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),  # N S W E
               (-1, -1), (-1, 1), (1, -1), (1, 1)]  # NW NE SW SE

    grads = []
    samples = []
    for dy, dx in offsets:
        neigh = tp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        grad = (t - neigh).abs()
        grads.append(grad)
        samples.append(neigh)

    # Stack gradients: (D, H, W)
    grad_stack = torch.stack(grads, dim=0)
    # Median gradient per pixel
    med = grad_stack.median(dim=0).values
    thresh = med * 1.2 + 1e-6

    # For each channel, fill missing pixels
    def _interp_channel(ch_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ch_tensor currently has known values at its mask positions; zeros elsewhere.
        out = ch_tensor.clone()
        missing = ~mask
        if missing.any():
            acc = torch.zeros_like(t)
            cnt = torch.zeros_like(t)
            for grad, samp in zip(grads, samples):
                use_dir = grad <= thresh  # (H,W) bool-like
                # Only consider neighbor samples where target channel exists at neighbor pixel
                # We need neighbor's mask for that channel:
                # For simplicity, assume neighbor pixel has same mask set as current (approximation). Instead, use mask itself shifted.
                # Build shifted channel mask
                # Pad mask then shift
                mp = _replicate_pad(mask.float())
                chmask_shift = mp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] > 0.5
                valid = use_dir & chmask_shift & missing
                acc = torch.where(valid, acc + samp, acc)
                cnt = torch.where(valid, cnt + 1.0, cnt)
            # Average
            filled = torch.where((missing & (cnt > 0)), acc / torch.clamp(cnt, min=1.0), zero)
            out = torch.where(missing, filled, out)
        return out

    R_full = _interp_channel(R, Rmask)
    G_full = _interp_channel(G, Gmask)
    B_full = _interp_channel(B, Bmask)

    return _to_uint8_image(R_full, G_full, B_full)

# ---------------------------------------------------------------------------
# Edge-Aware Demosaicing (Simplified)
# ---------------------------------------------------------------------------

def debayer_edgeaware_torch(raw: np.ndarray, pattern: str = 'RGGB', device: str = 'cpu') -> np.ndarray:
    """Edge-aware demosaicing (simplified).

    Approach:
    1. Interpolate Green at R/B locations using directional gradients (horizontal vs vertical).
    2. Interpolate Red and Blue at Green locations using lower-gradient direction.
    3. Interpolate Red at Blue locations and Blue at Red locations using diagonal neighbors.
    4. Assemble and clip.

    Returns BGR uint8 image.
    """
    t = _prepare_raw(raw, device)
    h, w = t.shape
    Rmask, Gmask, Bmask = _pattern_masks(h, w, pattern)

    R = torch.where(Rmask, t, torch.zeros_like(t))
    G = torch.where(Gmask, t, torch.zeros_like(t))
    B = torch.where(Bmask, t, torch.zeros_like(t))

    tp = _replicate_pad(t)

    # Helper to slice padded version
    def shifted(dy: int, dx: int) -> torch.Tensor:
        return tp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    # Directional gradients for green interpolation at R/B sites
    left = shifted(0, -1)
    right = shifted(0, 1)
    up = shifted(-1, 0)
    down = shifted(1, 0)

    grad_h = (left - right).abs()
    grad_v = (up - down).abs()

    # Known green neighbor averages
    # Build green mask padded for shifting lookups
    Gp = _replicate_pad(Gmask.float())
    G_left = Gp[1:1 + h, 0:w] > 0.5
    G_right = Gp[1:1 + h, 2:2 + w] > 0.5
    G_up = Gp[0:h, 1:1 + w] > 0.5
    G_down = Gp[2:2 + h, 1:1 + w] > 0.5

    green_left = shifted(0, -1)
    green_right = shifted(0, 1)
    green_up = shifted(-1, 0)
    green_down = shifted(1, 0)

    # Average horizontally / vertically if green neighbors exist
    avg_h = (green_left * G_left + green_right * G_right)
    cnt_h = G_left.float() + G_right.float()
    avg_h = torch.where(cnt_h > 0, avg_h / torch.clamp(cnt_h, min=1.0), (left + right) * 0.5)

    avg_v = (green_up * G_up + green_down * G_down)
    cnt_v = G_up.float() + G_down.float()
    avg_v = torch.where(cnt_v > 0, avg_v / torch.clamp(cnt_v, min=1.0), (up + down) * 0.5)

    need_green = ~Gmask
    use_h = (grad_h < grad_v) & need_green
    use_v = (grad_v < grad_h) & need_green
    tie = (grad_h == grad_v) & need_green

    G_interp = torch.zeros_like(t)
    G_interp = torch.where(use_h, avg_h, G_interp)
    G_interp = torch.where(use_v, avg_v, G_interp)
    G_interp = torch.where(tie, 0.5 * (avg_h + avg_v), G_interp)

    G_full = torch.where(Gmask, G, G_interp)

    # Interpolate R & B at green positions using directional gradients
    Rp = _replicate_pad(Rmask.float())
    Bp = _replicate_pad(Bmask.float())

    R_left = Rp[1:1 + h, 0:w] > 0.5
    R_right = Rp[1:1 + h, 2:2 + w] > 0.5
    R_up = Rp[0:h, 1:1 + w] > 0.5
    R_down = Rp[2:2 + h, 1:1 + w] > 0.5

    B_left = Bp[1:1 + h, 0:w] > 0.5
    B_right = Bp[1:1 + h, 2:2 + w] > 0.5
    B_up = Bp[0:h, 1:1 + w] > 0.5
    B_down = Bp[2:2 + h, 1:1 + w] > 0.5

    red_left = shifted(0, -1)
    red_right = shifted(0, 1)
    red_up = shifted(-1, 0)
    red_down = shifted(1, 0)

    blue_left = shifted(0, -1)
    blue_right = shifted(0, 1)
    blue_up = shifted(-1, 0)
    blue_down = shifted(1, 0)

    # Directional differences using green channel to detect edges
    grad_r_h = (red_left - red_right).abs() + (green_left - green_right).abs()
    grad_r_v = (red_up - red_down).abs() + (green_up - green_down).abs()

    grad_b_h = (blue_left - blue_right).abs() + (green_left - green_right).abs()
    grad_b_v = (blue_up - blue_down).abs() + (green_up - green_down).abs()

    avg_r_h = (red_left * R_left + red_right * R_right)
    cnt_r_h = R_left.float() + R_right.float()
    avg_r_h = torch.where(cnt_r_h > 0, avg_r_h / torch.clamp(cnt_r_h, min=1.0), (red_left + red_right) * 0.5)
    avg_r_v = (red_up * R_up + red_down * R_down)
    cnt_r_v = R_up.float() + R_down.float()
    avg_r_v = torch.where(cnt_r_v > 0, avg_r_v / torch.clamp(cnt_r_v, min=1.0), (red_up + red_down) * 0.5)

    avg_b_h = (blue_left * B_left + blue_right * B_right)
    cnt_b_h = B_left.float() + B_right.float()
    avg_b_h = torch.where(cnt_b_h > 0, avg_b_h / torch.clamp(cnt_b_h, min=1.0), (blue_left + blue_right) * 0.5)
    avg_b_v = (blue_up * B_up + blue_down * B_down)
    cnt_b_v = B_up.float() + B_down.float()
    avg_b_v = torch.where(cnt_b_v > 0, avg_b_v / torch.clamp(cnt_b_v, min=1.0), (blue_up + blue_down) * 0.5)

    need_r_at_g = Gmask & ~Rmask
    need_b_at_g = Gmask & ~Bmask

    r_use_h = (grad_r_h < grad_r_v) & need_r_at_g
    r_use_v = (grad_r_v < grad_r_h) & need_r_at_g
    r_tie = (grad_r_h == grad_r_v) & need_r_at_g

    b_use_h = (grad_b_h < grad_b_v) & need_b_at_g
    b_use_v = (grad_b_v < grad_b_h) & need_b_at_g
    b_tie = (grad_b_h == grad_b_v) & need_b_at_g

    R_from_g = torch.zeros_like(t)
    R_from_g = torch.where(r_use_h, avg_r_h, R_from_g)
    R_from_g = torch.where(r_use_v, avg_r_v, R_from_g)
    R_from_g = torch.where(r_tie, 0.5 * (avg_r_h + avg_r_v), R_from_g)

    B_from_g = torch.zeros_like(t)
    B_from_g = torch.where(b_use_h, avg_b_h, B_from_g)
    B_from_g = torch.where(b_use_v, avg_b_v, B_from_g)
    B_from_g = torch.where(b_tie, 0.5 * (avg_b_h + avg_b_v), B_from_g)

    # Diagonal interpolation for missing at R/B opposite sites
    # For red at blue sites
    diag1 = shifted(-1, -1)
    diag2 = shifted(-1, 1)
    diag3 = shifted(1, -1)
    diag4 = shifted(1, 1)

    Rp_diag = _replicate_pad(Rmask.float())
    R_d1 = Rp_diag[0:h, 0:w] > 0.5
    R_d2 = Rp_diag[0:h, 2:2 + w] > 0.5
    R_d3 = Rp_diag[2:2 + h, 0:w] > 0.5
    R_d4 = Rp_diag[2:2 + h, 2:2 + w] > 0.5

    Bp_diag = _replicate_pad(Bmask.float())
    B_d1 = Bp_diag[0:h, 0:w] > 0.5
    B_d2 = Bp_diag[0:h, 2:2 + w] > 0.5
    B_d3 = Bp_diag[2:2 + h, 0:w] > 0.5
    B_d4 = Bp_diag[2:2 + h, 2:2 + w] > 0.5

    need_r_at_b = Bmask & ~Rmask
    need_b_at_r = Rmask & ~Bmask

    # Average available diagonals
    def _avg_diagonals(diag_vals, diag_masks):
        acc = torch.zeros_like(t)
        cnt = torch.zeros_like(t)
        for v, m in zip(diag_vals, diag_masks):
            acc = torch.where(m, acc + v * m.float(), acc)
            cnt = torch.where(m, cnt + m.float(), cnt)
        return torch.where(cnt > 0, acc / torch.clamp(cnt, min=1.0), torch.zeros_like(t))

    red_at_b = _avg_diagonals([diag1, diag2, diag3, diag4], [R_d1, R_d2, R_d3, R_d4])
    blue_at_r = _avg_diagonals([diag1, diag2, diag3, diag4], [B_d1, B_d2, B_d3, B_d4])

    R_full = torch.where(Rmask, R, torch.where(need_r_at_g, R_from_g, torch.where(need_r_at_b, red_at_b, R)))
    B_full = torch.where(Bmask, B, torch.where(need_b_at_g, B_from_g, torch.where(need_b_at_r, blue_at_r, B)))

    return _to_uint8_image(R_full, G_full, B_full)

# ---------------------------------------------------------------------------
# Stack helpers
# ---------------------------------------------------------------------------

def debayer_stack_vng(frames: np.ndarray, pattern: str = 'RGGB', device: str = 'cpu') -> np.ndarray:
    if frames.ndim != 3:
        raise ValueError('frames must be (T,H,W)')
    out = []
    for i in range(frames.shape[0]):
        out.append(debayer_vng_torch(frames[i], pattern=pattern, device=device))
    return np.stack(out, axis=0)


def debayer_stack_edgeaware(frames: np.ndarray, pattern: str = 'RGGB', device: str = 'cpu') -> np.ndarray:
    if frames.ndim != 3:
        raise ValueError('frames must be (T,H,W)')
    out = []
    for i in range(frames.shape[0]):
        out.append(debayer_edgeaware_torch(frames[i], pattern=pattern, device=device))
    return np.stack(out, axis=0)

# ---------------------------------------------------------------------------
# Self-test (optional usage)
# ---------------------------------------------------------------------------
if __name__ == '__main__':  # pragma: no cover
    import cv2
    H, W = 32, 48
    # Create synthetic RGGB raw with gradients
    pattern = 'RGGB'
    # Synthetic color image
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]  # R horizontal ramp
    rgb[..., 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]  # G vertical ramp
    rgb[..., 2] = 128  # B constant
    # Convert to Bayer raw using masks
    Rm, Gm, Bm = _pattern_masks(H, W, pattern)
    raw = np.zeros((H, W), dtype=np.uint8)
    raw[Rm.numpy()] = rgb[..., 0][Rm.numpy()]
    raw[Gm.numpy()] = rgb[..., 1][Gm.numpy()]
    raw[Bm.numpy()] = rgb[..., 2][Bm.numpy()]

    vng_img = debayer_vng_torch(raw, pattern=pattern)
    ea_img = debayer_edgeaware_torch(raw, pattern=pattern)

    print('VNG image shape:', vng_img.shape, 'dtype:', vng_img.dtype)
    print('EA image shape:', ea_img.shape, 'dtype:', ea_img.dtype)
    # Show using OpenCV if available (will open windows)
    try:
        cv2.imshow('raw (scaled)', raw)
        cv2.imshow('vng', vng_img)
        cv2.imshow('edgeaware', ea_img)
        cv2.waitKey(0)
    except Exception:
        pass

