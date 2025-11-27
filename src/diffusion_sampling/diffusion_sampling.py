"""
Helper utilities for the "Improving Telescope Images with AI" notebook.

This module contains:
- Visualization helpers for galaxy and telescope images.
- PSF and downsampling operators.
- Construction of the linear operator A combining PSF + downsampling.
- A linear-Gaussian posterior sampler that uses a diffusion-model prior.

Feel free to inspect this cell if you are curious.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

# --------------------------------------------------------------------------- #
#  Visualization helpers                                                      #
# --------------------------------------------------------------------------- #

def img_to_show(
    img: torch.Tensor,
    log_scale: bool = False,
    min_val: float = 0.01,
) -> "np.ndarray":
    """
    Convert a tensor image into a NumPy array ready for Matplotlib.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor. Can have arbitrary shape; only the last two dimensions
        are interpreted as (H, W).
    log_scale : bool, optional
        If True, apply a log stretch before normalizing to [0, 1].
    min_val : float, optional
        Small positive constant added before the log to avoid log(0).

    Returns
    -------
    np.ndarray
        2D array suitable for plt.imshow.
    """
    if log_scale:
        img = torch.log(img + min_val)
        # normalize ignoring NaN / inf
        valid = torch.isfinite(img)
        min_non_nan = img[valid].min()
        max_non_nan = img[valid].max()
        img = (img - min_non_nan) / (max_non_nan - min_non_nan)
        img[~valid] = 0.0
    return img.squeeze().detach().cpu().numpy()


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Add Gaussian noise to a batch of images.

    Parameters
    ----------
    images : torch.Tensor
        Input images of shape (B, H, W) or (B, 1, H, W) or similar.
    sigma : float
        Standard deviation of the additive Gaussian noise.

    Returns
    -------
    torch.Tensor
        Noisy images with the same shape as `images`.
    """
    return images + sigma * torch.randn_like(images)


def show_grid(
    images: torch.Tensor,
    log_scale: bool = False,
    title: Optional[str] = None,
    n_show: int = 5,
) -> None:
    """
    Show up to `n_show` images in a grid with a linear and log row.

    Top row: linear scale.
    Bottom row: log scale.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images of shape (B, H, W).
    log_scale : bool, optional
        This argument is kept for backwards compatibility but is ignored;
        both rows are always shown.
    title : str, optional
        Figure title.
    n_show : int, optional
        Number of images to display (from the start of the batch).
    """
    n_show = min(n_show, images.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))

    for i in range(n_show):
        img_lin = img_to_show(images[i], log_scale=False)
        axes[0, i].imshow(img_lin, cmap="magma")
        axes[0, i].axis("off")

    for i in range(n_show):
        img_log = img_to_show(images[i], log_scale=True)
        axes[1, i].imshow(img_log, cmap="magma")
        axes[1, i].axis("off")

    if title is not None:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_grid_final(
    images: torch.Tensor,
    selected_idx: int = 0,
    n_show: int = 5,
) -> None:
    """
    Show a single row of images, highlighting one of them with a green frame.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images of shape (B, H, W).
    selected_idx : int, optional
        Index of the image to highlight with a green border.
    n_show : int, optional
        Number of images from the batch to show.
    """
    n_show = min(n_show, images.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 3))

    for i in range(n_show):
        img = img_to_show(images[i], log_scale=False)
        ax = axes[i]
        ax.imshow(img, cmap="magma")
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw a frame around each subplot, highlighting the selected one.
        for spine in ax.spines.values():
            spine.set_visible(True)
            if i == selected_idx:
                spine.set_edgecolor("green")
                spine.set_linewidth(4) # Reduced linewidth slightly for smaller plot
            else:
                spine.set_edgecolor("black")
                spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  PSF and downsampling operators                                             #
# --------------------------------------------------------------------------- #

def psf_on_image(
    images: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Apply a Gaussian PSF to a batch of images.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images with shape (B, H, W).
    sigma : float
        Standard deviation of the Gaussian PSF in pixel units.

    Returns
    -------
    torch.Tensor
        Blurred images with the same shape as the input.
    """
    device = images.device
    dtype = images.dtype
    img_np = images.detach().cpu().numpy()  # (B, H, W)
    img_blur = gaussian_filter(img_np, sigma=(0, sigma, sigma))
    img_out = torch.from_numpy(img_blur).to(device=device, dtype=dtype)
    return img_out


def downsample_img(
    images: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """
    Downsample a batch of images to a given spatial resolution, preserving flux.

    The downsampling is done with bilinear interpolation and then rescaled so
    that the total flux per image is unchanged.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images with shape (B, H, W).
    size : int
        Target spatial size (size x size).

    Returns
    -------
    torch.Tensor
        Downsampled images with shape (B, size, size).
    """
    device = images.device
    dtype = images.dtype

    B, H, W = images.shape
    x = images.unsqueeze(1)  # (B,1,H,W)

    flux_before = x.sum(dim=(2, 3), keepdim=True)  # (B,1,1,1)

    x_ds = F.interpolate(
        x,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )  # (B,1,size,size)

    flux_after = x_ds.sum(dim=(2, 3), keepdim=True)  # (B,1,1,1)

    eps = 1e-12
    x_ds = x_ds * (flux_before / (flux_after + eps))

    image_ds = x_ds.squeeze(1).to(device=device, dtype=dtype)  # (B,size,size)
    return image_ds


def psf_downsample_build_A(
    images: torch.Tensor,
    sigma_psf: float,
    S: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the linear operator A corresponding to PSF + downsampling,
    and apply it to a batch of images.

    This function is conceptually:

        A: R^{H*W} -> R^{S*S}
        y = A x, where x is the flattened high-res galaxy image.

    It is constructed by applying the PSF + downsample pipeline to each
    basis vector (delta pixel) in turn.

    Parameters
    ----------
    images : torch.Tensor
        Reference images with shape (B, H, W). Only the shape and device
        are used to define H, W and dtype; the actual values do not affect A.
    sigma_psf : float
        Standard deviation of the Gaussian PSF used in the forward model.
    S : int
        Target spatial size for the downsampled images (S x S).

    Returns
    -------
    y_lin : torch.Tensor
        Result of applying A to `images`, with shape (B, S, S).
    A : torch.Tensor
        Linear operator matrix of shape (S*S, H*W).
    """
    device = images.device
    dtype = images.dtype
    B, H, W = images.shape

    def linear_psf_downsample(img_batch: torch.Tensor) -> torch.Tensor:
        """
        Helper that applies PSF + flux-preserving downsampling to a batch.
        """
        img_np = img_batch.detach().cpu().numpy()  # (B,H,W)
        flux_before = img_np.sum(axis=(1, 2), keepdims=True)  # (B,1,1)

        img_blur = gaussian_filter(img_np, sigma=(0, sigma_psf, sigma_psf))
        flux_after = img_blur.sum(axis=(1, 2), keepdims=True)  # (B,1,1)

        eps = 1e-12
        scale = flux_before / (flux_after + eps)
        img_blur *= scale

        img_blur_t = torch.from_numpy(img_blur).to(device=device, dtype=dtype)  # (B,H,W)

        x = img_blur_t.unsqueeze(1)  # (B,1,H,W)
        flux_before_ds = x.sum(dim=(2, 3), keepdim=True)  # (B,1,1,1)

        x_ds = F.interpolate(
            x,
            size=(S, S),
            mode="bilinear",
            align_corners=True,  # kept as in the original notebook
        )

        flux_after_ds = x_ds.sum(dim=(2, 3), keepdim=True)  # (B,1,1,1)

        x_ds = x_ds * (flux_before_ds / (flux_after_ds + eps))
        img_ds = x_ds.squeeze(1)  # (B,S,S)
        return img_ds

    # Build A by applying the linear operator to each basis vector e_j.
    N_in = H * W
    N_out = S * S
    A = torch.zeros(N_out, N_in, device=device, dtype=dtype)

    for j in range(N_in):
        basis = torch.zeros(1, H, W, device=device, dtype=dtype)
        y_idx = j // W
        x_idx = j % W
        basis[0, y_idx, x_idx] = 1.0

        out = linear_psf_downsample(basis)   # (1,S,S)
        A[:, j] = out.view(-1)               # (S*S,)

    # Apply A to the provided images for convenience
    x_flat = images.view(B, -1)              # (B,N_in)
    y_flat = x_flat @ A.t()                  # (B,N_out)
    y_lin = y_flat.view(B, S, S)             # (B,S,S)

    return y_lin, A

# --------------------------------------------------------------------------- #
#  Linear-Gaussian posterior sampler                                          #
# --------------------------------------------------------------------------- #

class LinearGaussianPosteriorSampler:
    """
    Score-based posterior sampler for a simple linear model:

        y_b = A x + n_b,   n_b ~ N(0, sigma_n^2 I),  b = 1..B,

    with a diffusion-model prior over x.

    This is a simplified, teaching-oriented engine:
    - One latent object x, but possibly multiple independent observations y_b.
    - Diagonal covariance approximation for the likelihood.
    - Fixed linear forward model A @ x.
    """

    def __init__(
        self,
        observation: torch.Tensor,
        A: torch.Tensor,
        model,
        sigma_n: float,
        C: float = 1.0,
        M: float = 0.0,
        anneal_factor: float = 7.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Parameters
        ----------
        observation : torch.Tensor
            Observations in image space. Acceptable shapes are:
            (H,W), (B,H,W) or (B,1,H,W).
        A : torch.Tensor
            Linear operator with shape (Mobs, Msrc) mapping flattened sources
            to flattened observations.
        model : object
            Score-based diffusion model with attributes `sde`, `score`,
            and `tweedie`, loaded beforehand.
        sigma_n : float
            Standard deviation of the Gaussian noise in the observation space.
        C : float, optional
            Global scaling factor from score space to physical space.
        M : float, optional
            Global offset from score space to physical space.
        anneal_factor : float, optional
            Extra factor used in the variance term (kept from the original
            research code, can be set to 0.0 for a simpler version).
        device : torch.device, optional
            Device on which to run inference; defaults to A.device.
        """
        if device is None:
            device = A.device
        self.device = device

        # Move model to the correct device if needed
        if hasattr(model, "to"):
            model = model.to(self.device)
        self.model = model

        # Standardize observation shape to (B,1,H_obs,W_obs)
        obs = observation
        if obs.dim() == 2:          # (H,W)
            obs = obs.unsqueeze(0).unsqueeze(0)           # (1,1,H,W)
        elif obs.dim() == 3:        # (B,H,W)
            obs = obs.unsqueeze(1)                        # (B,1,H,W)
        elif obs.dim() == 4:        # (B,C,H,W)
            if obs.shape[1] != 1:
                raise ValueError("Observation must have a single channel (C=1).")
        else:
            raise ValueError("Unsupported observation shape.")

        self.obs = obs.to(self.device)                    # (B,1,H,W)
        B, C_obs, H_obs, W_obs = self.obs.shape
        if C_obs != 1:
            raise ValueError("Observation must have a single channel (C=1).")
        self.B = B

        # Flattened observations: (B,Mobs)
        self.Mobs = H_obs * W_obs
        self.obs_flat = self.obs.view(B, -1)              # (B,Mobs)

        # Check A shape: (Mobs,Msrc)
        if A.dim() != 2 or A.shape[0] != self.Mobs:
            raise ValueError(
                f"A must have shape (Mobs, Msrc) with Mobs={self.Mobs}, "
                f"got {tuple(A.shape)}"
            )
        self.A = A.to(self.device)                        # (Mobs,Msrc)
        self.Msrc = self.A.shape[1]

        # Assume square source images: Msrc = Hs * Hs
        Hs = int(math.sqrt(self.Msrc))
        if Hs * Hs != self.Msrc:
            raise ValueError(
                f"Source dimension Msrc={self.Msrc} is not a perfect square (Hs*Hs)."
            )
        self.Hs = Hs

        # diag(A A^T): row-wise squared norm of A
        self.diag_AAT = (self.A ** 2).sum(dim=1)          # (Mobs,)

        # Noise level and scaling
        self.sigma_n = float(sigma_n)
        self.C = float(C)
        self.M = float(M)
        self.anneal_factor = float(anneal_factor)

    # -------------------------- likelihood score --------------------------- #

    def _likelihood_score_diag(
        self,
        t: torch.Tensor,   # (n,)
        x: torch.Tensor,   # (n,1,Hs,Hs)
        n: int,
    ) -> torch.Tensor:
        """
        Diagonal-covariance approximation to ∇_x log p({y_b} | x).

        Parameters
        ----------
        t : torch.Tensor
            Diffusion times with shape (n,).
        x : torch.Tensor
            Latent source samples with shape (n,1,Hs,Hs).
        n : int
            Number of posterior samples.

        Returns
        -------
        torch.Tensor
            Gradient of the log-likelihood with respect to x, flattened
            to shape (n, Msrc).
        """
        B = self.B
        Bn = x.shape[0]   # == n
        if Bn != n:
            raise ValueError(
                f"`n` inconsistent with sample batch size: expected {n}, got {Bn}"
            )

        Mobs = self.Mobs
        Msrc = self.Msrc

        # Enable gradient tracking on x
        x_score = x.detach().clone().requires_grad_(True)      # (n,1,Hs,Hs)

        # Map from score space to physical space
        x_phys = x_score * self.C + self.M                     # (n,1,Hs,Hs)

        # Forward model: mean = A x_phys
        x_flat = x_phys.view(Bn, Msrc)                         # (n,Msrc)
        mean_flat = x_flat @ self.A.t()                        # (n,Mobs)

        # Expand mean and observation across exposures
        mean_expanded = mean_flat.unsqueeze(1).expand(-1, B, -1)   # (n,B,Mobs)
        y_expanded = self.obs_flat.unsqueeze(0).expand(Bn, -1, -1) # (n,B,Mobs)

        # Time-dependent variance term (diagonal covariance)
        sigma_t = self.model.sde.sigma(t).view(Bn, 1, 1)       # (n,1,1)
        rt_2 = sigma_t**2 * ((self.C + self.anneal_factor)**2 *
                             t.view(Bn, 1, 1)**4 + 1.0)

        diag = self.diag_AAT.view(1, 1, Mobs)                  # (1,1,Mobs)
        var = (self.sigma_n**2) + rt_2 * diag                  # (n,1,Mobs) -> (n,B,Mobs)

        # Diagonal Gaussian log-likelihood for all exposures:
        # log p({y_b} | x) = sum_b log p(y_b | x)
        diff = y_expanded - mean_expanded                      # (n,B,Mobs)
        z = diff / var                                         # (n,B,Mobs)
        ll_sum = -0.5 * (diff * z).sum()                       # scalar

        # Backpropagate to get ∇_x log p({y_b} | x)
        ll_sum.backward()
        grad = x_score.grad.view(Bn, Msrc)                     # (n,Msrc)
        return grad

    # ---------------------- predictor–corrector sampler -------------------- #

    def run(
        self,
        n_samples: int = 8,
        steps: int = 1_000,
        progress: bool = True,
        true: Optional[torch.Tensor] = None,
        plot_trajectory: bool = False,
        trajectory_stride: int = 2,
    ) -> torch.Tensor:
        """
        Run the predictor–corrector sampler and return posterior samples.

        Parameters
        ----------
        n_samples : int, optional
            Number of posterior samples to draw.
        steps : int, optional
            Number of predictor (Heun) steps in diffusion time.
        progress : bool, optional
            Whether to display a tqdm progress bar.
        true : torch.Tensor, optional
            Optional true high-resolution image (for debug plots only).
        plot_trajectory : bool, optional
            If True, show intermediate samples and mock observations during
            the diffusion process.
        trajectory_stride : int, optional
            Plot every `trajectory_stride` steps if `plot_trajectory` is True.

        Returns
        -------
        torch.Tensor
            Posterior samples with shape (n_samples, 1, Hs, Hs) in physical
            units (after the Tweedie denoising and affine transform).
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x: x
            if progress:
                print("tqdm is not installed; progress bar will be disabled.")
                progress = False

        n = n_samples
        Hs = self.Hs
        Msrc = self.Msrc

        # Initial latent x sampled from the SDE prior
        x = (
            self.model.sde.prior((1, Hs, Hs))
            .sample([n])
            .to(self.device)
        )  # (n,1,Hs,Hs)

        # Time grid
        T = self.model.sde.T
        eps = self.model.sde.epsilon
        t = torch.ones(n, device=self.device) * T            # (n,)
        dt = -(T - eps) / steps

        iterator = tqdm(range(steps)) if progress else range(steps)

        for step in iterator:
            t_old, t_new = t, t + dt

            # Heun predictor step
            with torch.no_grad():
                g1 = self.model.sde.diffusion(t_old, x)
                f1 = self.model.sde.drift(t_old, x)
                s1 = self.model.score(t_old, x)
            lk1 = (
                self._likelihood_score_diag(t_old, x, n)
                .view(n, 1, Hs, Hs)
            )
            drift1 = f1 - g1**2 * (s1 + lk1)

            dw = torch.randn_like(x) * (-dt) ** 0.5
            x_e = x + drift1 * dt + g1 * dw

            with torch.no_grad():
                g2 = self.model.sde.diffusion(t_new, x_e)
                f2 = self.model.sde.drift(t_new, x_e)
                s2 = self.model.score(t_new, x_e)
            lk2 = (
                self._likelihood_score_diag(t_new, x_e, n)
                .view(n, 1, Hs, Hs)
            )
            drift2 = f2 - g2**2 * (s2 + lk2)

            x = x + 0.5 * (drift1 + drift2) * dt + g1 * dw
            x_mean = x - g1 * dw   # "noise-free" part
            t = t_new

            # Optional visualisation of the trajectory
            if plot_trajectory and (step % max(1, trajectory_stride) == 0):
                self._plot_step(x_mean, step, true)

        # Final Tweedie denoising and reshape
        with torch.no_grad():
            t0 = self.model.sde.t_min * torch.ones(x_mean.shape[0], device=self.device)
            x0 = self.model.tweedie(t0, x_mean)

        samples = x0.view(n, 1, Hs, Hs) * self.C + self.M
        return samples

    # ------------------------------- plotting ------------------------------ #

    def _plot_step(
        self,
        x_mean: torch.Tensor,
        step: int,
        true: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Internal helper to visualise the current samples and mock observations.
        """
        n_plot = min(4, x_mean.shape[0])
        Hs = self.Hs
        H_obs, W_obs = self.obs.shape[2], self.obs.shape[3]

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))

        # Clear previous output in the notebook
        clear_output(wait=True)

        for i in range(n_plot):
            img = img_to_show(x_mean[i], log_scale=False)
            axes[0, i + 1].imshow(img, cmap="magma")
            axes[0, i + 1].set_title(f"Sample {i+1}", fontsize=8)
            axes[0, i + 1].axis("off")

            mock = x_mean[i].view(1, Msrc := Hs * Hs) @ self.A.t()  # (1,Mobs)
            mock_img = mock.view(1, 1, H_obs, W_obs)
            mock_img_to_show = img_to_show(mock_img[0, 0], log_scale=False)
            axes[1, i + 1].imshow(mock_img_to_show, cmap="magma")
            axes[1, i + 1].set_title(f"Mock Obs {i+1}", fontsize=8)
            axes[1, i + 1].axis("off")

        if true is not None:
            if true.dim() == 3:
                true_img = img_to_show(true[0], log_scale=False)
            else:
                true_img = img_to_show(true, log_scale=False)
            axes[0, 0].imshow(true_img, cmap="magma")
            axes[0, 0].set_title("True", fontsize=8)
        else:

            axes[0, 0].text(0.5, 0.5, "?", fontsize=40, ha="center", va="center")
            axes[0, 0].set_title("True (unknown)", fontsize=8)
        axes[0, 0].axis("off")

        obs_img = img_to_show(self.obs[0, 0], log_scale=False)
        axes[1, 0].imshow(obs_img, cmap="magma")
        axes[1, 0].set_title("Observed (1st)", fontsize=8)
        axes[1, 0].axis("off")

        fig.suptitle(f"Step {step}", fontsize=10)
        plt.tight_layout()
        plt.show()