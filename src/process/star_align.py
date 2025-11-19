"""Fast star detection, pairwise matching and relative astrometric transforms.

This module provides lightweight, dependency-minimal routines to detect star-like
peaks in monochrome images, match them between images, estimate affine transforms
(with RANSAC via OpenCV), and visualize matches and image-relative positions.

APIs:
- detect_stars(image, threshold_sigma=5.0, min_distance=8, max_stars=500) -> (N,2) (x,y)
- match_points_and_estimate_affine(src_pts, dst_pts, max_match_dist=10.0) -> (M, inliers_mask, matched_src, matched_dst)
- compute_transforms_relative_to_reference(image_paths_or_arrays, ref_idx=0, detect_kwargs=None, match_kwargs=None) -> dict
- visualize_matches(ax, img_ref, img_other, ref_pts, other_pts, M=None, matches=None, inliers=None)
- visualize_image_positions(transforms, image_shapes, labels=None)
- warp_images_to_reference(images, transforms, ref_idx=0, output_shape=None, interpolation='linear', border_value=0.0)

Notes:
- Coordinates are in (x, y) format (column, row) to be consistent with OpenCV and earlier code.
- Uses only numpy and OpenCV (cv2). For large numbers of points, matching uses a simple
  vectorized nearest-neighbor; if you have scipy, a KD-tree would be faster.
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict


def detect_stars(image: np.ndarray,
                 threshold_sigma: float = 5.0,
                 min_distance: int = 8,
                 max_stars: int = 500) -> np.ndarray:
    """Detect bright compact peaks in a 2D image.

    Returns Nx2 float array of (x, y) coordinates (columns, rows).
    """
    if image.ndim != 2:
        raise ValueError('image must be 2D grayscale')
    img = image.astype(np.float32)
    # Quick background estimate
    med = np.median(img)
    std = np.std(img)
    thr = med + float(threshold_sigma) * (std if std > 0 else 1.0)

    # Smooth to reduce noise (kernel size proportional to min_distance)
    k = max(3, (min_distance // 2) * 2 + 1)
    img_blur = cv2.GaussianBlur(img, (k, k), sigmaX=0)

    # Local maxima via dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(img_blur, kernel)
    local_max = (img_blur >= dil)

    # Threshold
    mask = local_max & (img_blur >= thr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=float)

    intensities = img_blur[ys, xs]
    # sort by intensity desc
    order = np.argsort(intensities)[::-1]
    xs = xs[order]
    ys = ys[order]
    intens = intensities[order]

    # Non-maximum suppression (min_distance)
    picked_x = []
    picked_y = []
    min_d2 = (min_distance ** 2)
    for x, y in zip(xs, ys):
        if len(picked_x) == 0:
            picked_x.append(x); picked_y.append(y)
        else:
            dx = np.array(picked_x, dtype=float) - float(x)
            dy = np.array(picked_y, dtype=float) - float(y)
            d2 = dx * dx + dy * dy
            if np.all(d2 > min_d2):
                picked_x.append(x); picked_y.append(y)
        if len(picked_x) >= max_stars:
            break

    coords = np.stack((picked_x, picked_y), axis=1).astype(float)
    return coords


def _nearest_matches(src: np.ndarray, dst: np.ndarray, max_dist: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Naive nearest-neighbor matching from src->dst. Returns matched indices arrays (i_src, i_dst)."""
    if len(src) == 0 or len(dst) == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    # compute squared distances between points (vectorized)
    dX = src[:, None, 0] - dst[None, :, 0]
    dY = src[:, None, 1] - dst[None, :, 1]
    d2 = dX * dX + dY * dY
    idx = np.argmin(d2, axis=1)
    dist2 = d2[np.arange(d2.shape[0]), idx]
    mask = dist2 <= (max_dist * max_dist)
    i_src = np.where(mask)[0]
    i_dst = idx[mask]
    return i_src.astype(int), i_dst.astype(int)


def match_points_and_estimate_affine(src_pts: np.ndarray,
                                     dst_pts: np.ndarray,
                                     max_match_dist: float = 10.0,
                                     ransac_thresh: float = 3.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Match src_pts->dst_pts and estimate affine transform mapping src->dst.

    Returns (M, inliers_mask, matched_src_points, matched_dst_points)
    where M is 2x3 affine matrix (or None on failure), inliers_mask is boolean mask of length K or None.
    """
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return None, None, None, None
    i_src, i_dst = _nearest_matches(src_pts, dst_pts, max_dist=max_match_dist)
    if len(i_src) < 3:
        return None, None, None, None
    src_mat = src_pts[i_src].astype(np.float32).reshape(-1, 1, 2)
    dst_mat = dst_pts[i_dst].astype(np.float32).reshape(-1, 1, 2)
    # OpenCV expects (N,2) or (N,1,2) shapes; estimateAffinePartial2D uses src->dst mapping
    try:
        M, inliers = cv2.estimateAffinePartial2D(src_mat.reshape(-1,2), dst_mat.reshape(-1,2), method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000)
    except Exception:
        return None, None, None, None
    if M is None:
        return None, None, None, None
    inliers_mask = (inliers.reshape(-1) == 1) if inliers is not None else None
    matched_src = src_pts[i_src]
    matched_dst = dst_pts[i_dst]
    return M, inliers_mask, matched_src, matched_dst


def compute_transforms_relative_to_reference(images: List[np.ndarray],
                                             ref_idx: int = 0,
                                             detect_kwargs: Optional[Dict] = None,
                                             match_kwargs: Optional[Dict] = None) -> Dict:
    """Compute affine transforms mapping each image into the reference image frame.

    images: list of 2D numpy arrays (grayscale)
    Returns dict with keys:
      'transforms': list of (2x3) or None mapping image->reference
      'star_coords': list of detected star arrays for each image
      'matches': list of match dicts for each image (src_idx, dst_idx, inliers)
    """
    if detect_kwargs is None:
        detect_kwargs = dict(threshold_sigma=5.0, min_distance=8, max_stars=800)
    if match_kwargs is None:
        match_kwargs = dict(max_match_dist=10.0, ransac_thresh=3.0)

    n = len(images)
    stars = [None] * n
    for i in range(n):
        stars[i] = detect_stars(images[i], **detect_kwargs)

    ref_stars = stars[ref_idx]
    transforms = [None] * n
    matches = [None] * n
    transforms[ref_idx] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    matches[ref_idx] = dict(matched_src_idx=np.array([], dtype=int), matched_dst_idx=np.array([], dtype=int), inliers=np.array([], dtype=bool))

    for i in range(n):
        if i == ref_idx:
            continue
        M, inliers_mask, matched_src, matched_dst = match_points_and_estimate_affine(stars[i], ref_stars, **match_kwargs)
        transforms[i] = M
        matches[i] = dict(matched_src=matched_src, matched_dst=matched_dst, inliers=inliers_mask)

    return dict(transforms=transforms, star_coords=stars, matches=matches)


def visualize_matches(ax,
                      img_ref: np.ndarray,
                      img_other: np.ndarray,
                      ref_pts: np.ndarray,
                      other_pts: np.ndarray,
                      M: Optional[np.ndarray] = None,
                      matched_src: Optional[np.ndarray] = None,
                      matched_dst: Optional[np.ndarray] = None,
                      inliers: Optional[np.ndarray] = None,
                      point_color='yellow'):
    """Plot reference and other images side-by-side on the given matplotlib Axes.

    If M is provided (2x3), the other_pts will be transformed into the reference frame
    for visualization. If matched_src/matched_dst are provided, lines between matches
    are drawn; inliers (boolean mask) highlighted.
    """
    import matplotlib.pyplot as plt

    # transform other points into ref frame if M supplied
    if M is not None and other_pts is not None and len(other_pts) > 0:
        # convert to homogeneous (x,y,1)
        pts_h = np.concatenate([other_pts, np.ones((other_pts.shape[0], 1), dtype=float)], axis=1)
        T = np.vstack([M, [0.0, 0.0, 1.0]])
        transformed = (T @ pts_h.T).T[:, :2]
    else:
        transformed = other_pts

    ax.imshow(img_ref, cmap='gray')
    # plot ref pts
    if ref_pts is not None and len(ref_pts) > 0:
        ax.scatter(ref_pts[:, 0], ref_pts[:, 1], s=30, edgecolor='black', facecolor=point_color, alpha=0.8, label="ref stars")
    # plot transformed other pts
    if transformed is not None and len(transformed) > 0:
        ax.scatter(transformed[:, 0], transformed[:, 1], s=20, marker='+', color='cyan', alpha=0.8, label="other stars")

    # if matches provided, draw lines for inliers
    if matched_src is not None and matched_dst is not None:
        # matched_src are points from other image in other frame coords; transform them
        if M is not None:
            src_pts_tr = transformed
        else:
            src_pts_tr = matched_src
        dst_pts = matched_dst  # these are reference points
        K = min(len(dst_pts), len(src_pts_tr))
        for k in range(K):
            color = 'lime' if (inliers is not None and inliers[k]) else 'red'
            ax.plot([dst_pts[k, 0], src_pts_tr[k, 0]], [dst_pts[k, 1], src_pts_tr[k, 1]], color=color, linewidth=0.7, alpha=0.6)

    ax.set_xlim(0, img_ref.shape[1])
    ax.set_ylim(img_ref.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])


def visualize_image_positions(transforms: List[Optional[np.ndarray]],
                              image_shapes: List[Tuple[int, int]],
                              labels: Optional[List[str]] = None,
                              translate_scale: float = 1.0,
                              show_original: bool = True,
                              auto_zoom: bool = True,
                              margin: float = 0.12):
    """Plot centers and transformed corners of images relative to reference (index 0).

    New parameters:
      translate_scale: multiply the translation component of each transform by this factor
                       (useful to exaggerate small shifts for visualization).
      show_original: plot the unscaled outlines with faint gray so scaled effect is visible.
      auto_zoom: if True, zoom the axes to the union of transformed corners with a margin.
      margin: fraction of width/height to add around the computed bounding box when zooming.

    The function preserves the original transform list; it computes a scaled copy for plotting.
    """
    import matplotlib.pyplot as plt

    # compute corners in homogeneous coords for each image
    centers = []
    corners = []
    for (h, w) in image_shapes:
        cx = w / 2.0
        cy = h / 2.0
        centers.append((cx, cy))
        c = np.array([[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]]).T
        corners.append(c)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 8))

    all_x = []
    all_y = []

    for i, M in enumerate(transforms):
        if M is None:
            continue
        # make a scaled copy of M (scale only the translation component)
        M = np.array(M, dtype=float)
        Ms = M.copy()
        Ms[0, 2] = M[0, 2] * float(translate_scale)
        Ms[1, 2] = M[1, 2] * float(translate_scale)

        T_orig = np.vstack([M, [0.0, 0.0, 1.0]])
        T_scaled = np.vstack([Ms, [0.0, 0.0, 1.0]])

        c = corners[i]
        tc_orig = (T_orig @ c).T[:, :2]
        tc = (T_scaled @ c).T[:, :2]

        # Optionally plot original faint outlines
        if show_original:
            ax.plot(np.append(tc_orig[:, 0], tc_orig[0, 0]), np.append(tc_orig[:, 1], tc_orig[0, 1]), '-', color='0.7', linewidth=0.8, alpha=0.7)

        # Plot scaled outline and center
        ax.plot(np.append(tc[:, 0], tc[0, 0]), np.append(tc[:, 1], tc[0, 1]), '-', label=(labels[i] if labels else f'img{i}'))
        cen = (T_scaled @ np.array([centers[i][0], centers[i][1], 1.0]))[:2]
        ax.scatter(cen[0], cen[1], s=40)
        if labels:
            ax.text(cen[0], cen[1], labels[i])

        all_x.extend(tc[:, 0].tolist())
        all_y.extend(tc[:, 1].tolist())

    if len(all_x) == 0:
        ax.set_title('No transforms to display')
        plt.show()
        return

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title('Image positions in reference frame (scaled)')

    if auto_zoom:
        # compute bounding box and add margin
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        dx = maxx - minx
        dy = maxy - miny
        if dx == 0:
            dx = 1.0
        if dy == 0:
            dy = 1.0
        padx = dx * float(margin)
        pady = dy * float(margin)
        ax.set_xlim(minx - padx, maxx + padx)
        # invert y axis already; set_ylim expects (top,bottom) in data coords but we inverted
        ax.set_ylim(maxy + pady, miny - pady)
    else:
        # set to full ref image extents if reference is provided
        # try to set to max image size if available
        try:
            h0, w0 = image_shapes[0]
            ax.set_xlim(0, w0)
            ax.set_ylim(h0, 0)
        except Exception:
            pass

    plt.show()


def warp_images_to_reference(images: List[np.ndarray],
                             transforms: List[Optional[np.ndarray]],
                             ref_idx: int = 0,
                             output_shape: Optional[Tuple[int, int]] = None,
                             interpolation: str = 'linear',
                             border_value: float = 0.0) -> List[np.ndarray]:
    """Warp a list of images into the reference image frame using provided 2x3 transforms.

    Parameters
    - images: list of ndarray, each either 2D (H,W) or 3D (H,W,C).
    - transforms: list of 2x3 numpy arrays mapping image -> reference. transforms[i] maps images[i]
      into the coordinate system of the reference image (index ref_idx). If transforms[i] is None,
      identity is used.
    - ref_idx: index of the reference image (default 0). When output_shape is None, the reference
      image shape is used as the warp target.
    - output_shape: (H_out, W_out) tuple for the output canvas. If None, uses images[ref_idx].shape.
    - interpolation: 'nearest', 'linear', or 'cubic'.
    - border_value: fill value for areas outside the source after warping.

    Returns a list of warped images (same dtype as inputs) where each image is mapped into
    the reference coordinate frame. The returned list length equals len(images).

    Example:
        warped = warp_images_to_reference(imgs, transforms, ref_idx=0)
    """
    import cv2

    if len(images) != len(transforms):
        raise ValueError('images and transforms must have the same length')

    # Determine output canvas shape
    if output_shape is None:
        # pick reference image shape
        ref_img = images[ref_idx]
        if ref_img.ndim == 2:
            H_out, W_out = ref_img.shape
        else:
            H_out, W_out = ref_img.shape[0], ref_img.shape[1]
    else:
        H_out, W_out = int(output_shape[0]), int(output_shape[1])

    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }
    if interpolation not in interp_map:
        raise ValueError(f"Unknown interpolation: {interpolation}")
    interp_flag = interp_map[interpolation]

    warped_list = []
    for i, (img, M) in enumerate(zip(images, transforms)):
        if img is None:
            warped_list.append(None)
            continue
        # ensure numpy array
        arr = np.asarray(img)
        # choose dtype to preserve
        out_dtype = arr.dtype
        # if transform missing, use identity
        if M is None:
            M_use = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        else:
            M_use = np.array(M, dtype=float)
        # OpenCV warp expects src as uint8/uint16/float32; convert to float32 for safe handling
        need_cast_back = False
        save_dtype = arr.dtype
        if arr.dtype != np.float32:
            arr_in = arr.astype(np.float32)
            need_cast_back = True
        else:
            arr_in = arr
        # warpAffine supports multi-channel images
        try:
            warped = cv2.warpAffine(arr_in, M_use, (W_out, H_out), flags=interp_flag, borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_value))
        except Exception:
            # Fallback: if warpAffine fails (e.g., for unusual dtypes), try channel-wise
            if arr_in.ndim == 3:
                channels = []
                for c in range(arr_in.shape[2]):
                    ch = cv2.warpAffine(arr_in[..., c], M_use, (W_out, H_out), flags=interp_flag, borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_value))
                    channels.append(ch)
                warped = np.stack(channels, axis=-1)
            else:
                raise
        # cast back to original dtype when possible
        if need_cast_back:
            # clip to valid range for integer types
            if np.issubdtype(save_dtype, np.integer):
                info = np.iinfo(save_dtype)
                warped = np.clip(np.round(warped), info.min, info.max).astype(save_dtype)
            else:
                warped = warped.astype(save_dtype)
        warped_list.append(warped)
    return warped_list

