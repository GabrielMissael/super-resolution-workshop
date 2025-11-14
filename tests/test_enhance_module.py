import numpy as np
import tempfile
from src import enhance as en


def make_color_img(H=16, W=16):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # horizontal gradient R, vertical gradient G, constant B
    img[..., 2] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    img[..., 0] = 128
    return img


def test_unsharp_and_pyramid_shapes_and_types():
    img = make_color_img(16, 16)
    out_usm = en.unsharp_mask_color(img, sigma=1.0, amount=1.0)
    assert out_usm.shape == img.shape
    assert out_usm.dtype == np.uint8

    out_lp = en.laplacian_pyramid_sharpen(img, levels=2, weight=0.8)
    assert out_lp.shape == img.shape
    assert out_lp.dtype == np.uint8


def test_clahe_grayscale_and_color():
    img = make_color_img(16, 16)
    gray = img[..., 0]
    out_gray = en.apply_clahe(gray)
    assert out_gray.shape == gray.shape
    assert out_gray.dtype == np.uint8

    out_color = en.apply_clahe(img)
    assert out_color.shape == img.shape
    assert out_color.dtype == np.uint8


def test_color_adjustments_affect_pixels():
    img = make_color_img(16, 16)
    # saturation increase should change at least one pixel
    sat = en.adjust_saturation(img, factor=1.5)
    assert sat.shape == img.shape
    assert not np.array_equal(sat, img)

    hue = en.adjust_hue(img, shift_deg=60.0)
    assert hue.shape == img.shape
    assert not np.array_equal(hue, img)

    vib = en.adjust_vibrance(img, factor=1.5)
    assert vib.shape == img.shape
    assert not np.array_equal(vib, img)

    cb = en.adjust_contrast_brightness(img, contrast=1.2, brightness=5.0)
    assert cb.shape == img.shape
    assert cb.dtype == np.uint8


def test_enhance_pipeline_runs_and_outputs_uint8():
    img = make_color_img(32, 32)
    out = en.enhance_pipeline(img, clahe_clip=2.0, denoise_h=3.0, pyramid_levels=2, pyramid_weight=0.5,
                              usm_sigma=1.0, usm_amount=0.3,
                              saturation=1.1, vibrance=1.05, hue_shift=5.0, contrast=1.05, brightness=2.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_functions_handle_grayscale_inputs():
    img = make_color_img(16, 16)[..., 0]
    # functions should return grayscale unmodified type-wise
    assert en.unsharp_mask_color(img).ndim == 2
    assert en.laplacian_pyramid_sharpen(img).ndim == 2
    assert en.apply_clahe(img).ndim == 2
    assert en.adjust_saturation(img).shape == img.shape

