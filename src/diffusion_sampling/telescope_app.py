import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
import matplotlib.cm as cm


def launch_telescope_app(
    model,
    LinearGaussianPosteriorSampler,
    psf_downsample_build_A,
    *,
    share: bool = True,
    debug: bool = True,
):
    """
    Build and launch the Gradio 'Telescope Image Posterior Sampler' app.

    Parameters
    ----------
    model : diffusion model instance
        Must have attributes .sde, .score, .tweedie like in the workshop.
    LinearGaussianPosteriorSampler : class
        The sampler class from diffusion_sampling.py.
    psf_downsample_build_A : function
        Function(images, sigma_psf, S) -> (y_lin, A).
    share : bool, optional
        Passed to demo.launch(share=...).
    debug : bool, optional
        Passed to demo.launch(debug=...).

    Returns
    -------
    gradio.Blocks
        The launched Gradio app object.
    """

    # ------------------------------------------------------------------
    # Constants & global-ish state for this app
    # ------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SOURCE_RES = 64      # high-res latent size (Hs x Hs)
    DEFAULT_S = 64       # default downsample size S
    DEFAULT_SIGMA_PSF = 0.2
    DEFAULT_SIGMA_NOISE = 0.1
    DISPLAY_SIZE = 256   # pixel size for all displayed images

    # cache for A so we don't rebuild it on every click
    A_CACHE = {}
    MAGMA = cm.get_cmap("magma")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def center_crop_to_square(t: torch.Tensor) -> torch.Tensor:
        """Center-crop last two dims of a tensor to a square."""
        h, w = t.shape[-2:]
        m = min(h, w)
        top = (h - m) // 2
        left = (w - m) // 2
        return t[..., top:top + m, left:left + m]

    def rgb_to_gray_numpy(img: np.ndarray) -> np.ndarray:
        """
        Convert (H,W,3) uint8 RGB to (H,W) float32 grayscale in [0,255].
        Normalization to [0,1] is done later in torch.
        """
        if img is None:
            return None
        if img.ndim == 3 and img.shape[-1] == 3:
            r = img[..., 0].astype(np.float32)
            g = img[..., 1].astype(np.float32)
            b = img[..., 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray
        return img.astype(np.float32)

    def normalize_to_01(img_t: torch.Tensor) -> np.ndarray:
        """
        Normalize an image tensor to [0,1] for display, using per-image min/max.
        Works with arbitrary real-valued inputs (can be negative or >1).
        """
        img = img_t.detach().cpu()
        vmin = img.min()
        vmax = img.max()
        if float(vmax) == float(vmin):
            return torch.zeros_like(img).numpy()
        img = (img - vmin) / (vmax - vmin)
        return img.numpy()

    def upscale_to_size(img: np.ndarray, size: int = DISPLAY_SIZE) -> np.ndarray:
        """
        Upscale a small (H, W) grayscale image to (size, size) using nearest
        neighbour (keeps the blocky pixel look, but big enough for students).
        """
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t_big = F.interpolate(t, size=(size, size), mode="nearest")
        return t_big.squeeze(0).squeeze(0).cpu().numpy()             # (size,size)

    def to_magma_rgb(img_01: np.ndarray) -> np.ndarray:
        """
        Map a grayscale [0,1] image (H,W) to magma RGB (H,W,3) in [0,1].
        """
        img_clipped = np.clip(img_01, 0.0, 1.0)
        rgba = MAGMA(img_clipped)       # (H,W,4)
        rgb = rgba[..., :3].astype(np.float32)
        return rgb

    def apply_center_and_vignette(
        img_batch: torch.Tensor,
        center_sigma: float = 0.07,
        vignette_sigma: float = 0.7,
    ) -> torch.Tensor:
        """
        Add a bright Gaussian center (max 1) and darken borders with a
        Gaussian-like vignette (1 at center, ~0 at borders).

        img_batch: (B, H, W), values assumed in [0,1].
        Returns: (B, H, W), clipped to [0,1].
        """
        B, H, W = img_batch.shape
        device = img_batch.device

        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r2 = xx**2 + yy**2

        # Vignette: 1 at center, 0 at edges
        vignette = torch.exp(-r2 / (2 * vignette_sigma**2))
        vignette = (vignette - vignette.min()) / (vignette.max() - vignette.min())
        vignette = vignette.unsqueeze(0)  # (1,H,W), broadcast over batch

        # Bright central Gaussian blob, narrower
        center = torch.exp(-r2 / (2 * center_sigma**2))
        center = center / center.max()
        center = center.unsqueeze(0)      # (1,H,W)

        img_vignetted = img_batch * vignette
        img_with_center = img_vignetted + center
        img_with_center = img_with_center.clamp(0.0, 1.0)

        return img_with_center

    def get_y_lin_and_A(images: torch.Tensor, sigma_psf: float, S: int):
        """
        Wrapper around psf_downsample_build_A with caching.
        Rebuilds A only when (sigma_psf, S, H, W, device) change.
        """
        B, H, W = images.shape
        key = (float(sigma_psf), int(S), H, W, images.device)

        if key in A_CACHE:
            A = A_CACHE[key]
            x_flat = images.view(B, -1)           # (B, H*W)
            y_flat = x_flat @ A.t()               # (B, S*S)
            y_lin = y_flat.view(B, S, S)          # (B, S, S)
            return y_lin, A
        else:
            y_lin, A = psf_downsample_build_A(
                images,
                sigma_psf=float(sigma_psf),
                S=int(S),
            )
            A_CACHE[key] = A
            return y_lin, A

    # ------------------------------------------------------------------
    # Gradio callbacks
    # ------------------------------------------------------------------

    def apply_noise(
        src_choice: str,
        upload_img: np.ndarray,
        camera_img: np.ndarray,
        sigma_psf: float,
        sigma_noise: float,
        S: int,
    ):
        """
        From selected image + sliders:
          - convert to grayscale and normalize to [0,1] (before any preprocessing),
          - build high-res source (1, SOURCE_RES, SOURCE_RES),
          - apply bright center + vignette,
          - build A and noiseless observation y_lin,
          - add noise to get y_obs,
          - return magma-colored, upscaled y_obs and store tensors/params in State.
        """
        # choose image
        # Note: Translated string check
        img = upload_img if src_choice == "Importer" else camera_img
        if img is None:
            raise gr.Error("Veuillez d'abord prendre une photo ou importer une image.")

        # 1. grayscale + normalization to [0,1]
        img_gray = rgb_to_gray_numpy(img)              # (H, W) in [0,255]
        img_tensor = torch.from_numpy(img_gray) / 255.0  # (H,W) in [0,1]
        img_tensor = img_tensor.to(DEVICE)

        # 2. crop + resize to SOURCE_RES (still in [0,1])
        img_tensor = img_tensor.unsqueeze(0)           # (1, H, W)
        cropped = center_crop_to_square(img_tensor)    # (1, m, m)
        high_res = F.interpolate(
            cropped.unsqueeze(0),                      # (1,1,m,m)
            size=(SOURCE_RES, SOURCE_RES),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                   # (1, Hs, Hs)

        # 2b. apply bright center + vignette
        high_res = apply_center_and_vignette(high_res)  # (1,Hs,Hs), in [0,1]

        # 3. linear operator A and noiseless observation
        y_lin, A = get_y_lin_and_A(
            high_res,
            sigma_psf=float(sigma_psf),
            S=int(S),
        )  # y_lin: (1,S,S), A: (S*S, Hs*Hs)

        y_clean = y_lin[0]                             # (S,S)
        y_obs = y_clean + float(sigma_noise) * torch.randn_like(y_clean)

        # 4. for display: min–max normalize, upscale, colormap
        obs_np_small01 = normalize_to_01(y_obs)        # (S,S) in [0,1]
        obs_up = upscale_to_size(obs_np_small01)       # (DISPLAY_SIZE,DISPLAY_SIZE)
        obs_rgb = to_magma_rgb(obs_up)                 # (DISPLAY_SIZE,DISPLAY_SIZE,3)

        # return: observation image + all internal state
        return (
            obs_rgb,            # colored noisy observation image
            high_res,           # state: high-res tensor (1,Hs,Hs)
            A,                  # state: linear operator
            y_obs,              # state: noisy observation (S,S)
            float(sigma_psf),   # state: psf used
            float(sigma_noise), # state: noise used
            int(S),             # state: S used
        )

    def stream_inference(
        high_res,
        A,
        y_obs,
        sigma_psf_val,
        sigma_noise_val,
        S_val,
    ):
        """
        Generator used by Gradio:
          - runs the diffusion posterior sampler with n_samples=1,
          - after every few steps, yields an upscaled magma-colored current x_mean[0],
          - at the end, yields the final posterior sample.
        Outputs per yield: (current_frame_img_RGB, final_sample_img_RGB_or_None)
        """
        if y_obs is None or A is None or high_res is None:
            raise gr.Error("Veuillez cliquer sur 'Appliquer le bruit' d'abord.")

        # ensure tensors on correct device
        high_res_t = high_res.to(DEVICE)
        A_t = A.to(DEVICE)
        y_obs_t = y_obs.to(DEVICE)

        # Build sampler
        sampler = LinearGaussianPosteriorSampler(
            observation=y_obs_t,                      # (S,S)
            A=A_t,
            model=model,
            sigma_n=float(sigma_noise_val),
            C=1.0,
            M=0.0,
        )

        n = 1
        Hs = sampler.Hs
        # Initial latent x from SDE prior
        x = (
            sampler.model.sde.prior((1, Hs, Hs))
            .sample([n])
            .to(sampler.device)
        )  # (n,1,Hs,Hs)

        # Time grid
        steps = 250
        trajectory_stride = 5
        T = sampler.model.sde.T
        eps = sampler.model.sde.epsilon
        t = torch.ones(n, device=sampler.device) * T
        dt = -(T - eps) / steps

        # Main loop (copy of .run, but with yields)
        for step in range(steps):
            t_old, t_new = t, t + dt

            # Heun predictor step
            with torch.no_grad():
                g1 = sampler.model.sde.diffusion(t_old, x)
                f1 = sampler.model.sde.drift(t_old, x)
                s1 = sampler.model.score(t_old, x)
            lk1 = sampler._likelihood_score_diag(t_old, x, n).view(n, 1, Hs, Hs)
            drift1 = f1 - g1**2 * (s1 + lk1)

            dw = torch.randn_like(x) * (-dt) ** 0.5
            x_e = x + drift1 * dt + g1 * dw

            with torch.no_grad():
                g2 = sampler.model.sde.diffusion(t_new, x_e)
                f2 = sampler.model.sde.drift(t_new, x_e)
                s2 = sampler.model.score(t_new, x_e)
            lk2 = sampler._likelihood_score_diag(t_new, x_e, n).view(n, 1, Hs, Hs)
            drift2 = f2 - g2**2 * (s2 + lk2)

            x = x + 0.5 * (drift1 + drift2) * dt + g1 * dw
            x_mean = x - g1 * dw   # "noise-free" part
            t = t_new

            # Every few steps, stream the current x_mean as upscaled magma RGB
            if step % max(1, trajectory_stride) == 0:
                img_small01 = normalize_to_01(x_mean[0, 0])  # (Hs,Hs) in [0,1]
                img_up = upscale_to_size(img_small01)        # (DISPLAY_SIZE,DISPLAY_SIZE)
                img_rgb = to_magma_rgb(img_up)               # (DISPLAY_SIZE,DISPLAY_SIZE,3)
                yield img_rgb, None                          # current frame only

        # Final Tweedie denoising and reshape (same as .run)
        with torch.no_grad():
            t0 = sampler.model.sde.t_min * torch.ones(x_mean.shape[0], device=sampler.device)
            x0 = sampler.model.tweedie(t0, x_mean)

        samples = x0.view(n, 1, Hs, Hs) * sampler.C + sampler.M
        sample0 = samples[0, 0]
        final_small01 = normalize_to_01(sample0)             # (Hs,Hs) in [0,1]
        final_up = upscale_to_size(final_small01)
        final_rgb = to_magma_rgb(final_up)

        # Final yield: freeze on final image and also show it in the “final” slot
        yield final_rgb, final_rgb

    # ------------------------------------------------------------------
    # Gradio UI
    # ------------------------------------------------------------------

    with gr.Blocks(title="Reconstruction d'Image de Télescope") as demo:
        gr.Markdown(
            "# Reconstruction d'Image de Télescope 🔭\n"
            "1. Choisissez **Caméra** ou **Importer** et sélectionnez une image.\n"
            "2. Ajustez la PSF (flou), le bruit et la résolution, puis cliquez sur **Appliquer le bruit** "
            "pour voir l'observation bruitée.\n"
            "3. Cliquez sur **▶ Lancer l'inférence** pour voir l'IA reconstruire la galaxie."
        )

        # 1. source selection
        with gr.Row():
            src_choice = gr.Radio(
                ["Caméra", "Importer"],
                value="Caméra",
                label="Source de l'image",
            )

        # 2. image input: camera vs upload (snapshot only)
        with gr.Row():
            camera_img = gr.Image(
                label="Photo de la caméra",
                sources=["webcam"],
                type="numpy",
                image_mode="RGB",
                height=320,
                streaming=False,
                visible=True,
            )
            upload_img = gr.Image(
                label="Importer une image",
                sources=["upload"],
                type="numpy",
                image_mode="RGB",
                height=320,
                visible=False,
            )

        def toggle_source(src):
            return (
                gr.update(visible=(src == "Caméra")),
                gr.update(visible=(src == "Importer")),
            )

        src_choice.change(
            fn=toggle_source,
            inputs=src_choice,
            outputs=[camera_img, upload_img],
        )

        # 3. sliders for PSF, noise, downsampling
        with gr.Row():
            sigma_psf_slider = gr.Slider(
                minimum=0.2,
                maximum=2.0,
                value=DEFAULT_SIGMA_PSF,
                step=0.01,
                label="PSF (flou)",
            )
            sigma_noise_slider = gr.Slider(
                minimum=0.05,
                maximum=0.8,
                value=DEFAULT_SIGMA_NOISE,
                step=0.01,
                label="Bruit σ",
            )
            downsample_slider = gr.Slider(
                minimum=24,
                maximum=64,
                value=DEFAULT_S,
                step=1,
                label="Résolution (pixels)",
            )

        # 4. buttons
        with gr.Row():
            apply_btn = gr.Button("Appliquer le bruit", variant="secondary")
            run_btn = gr.Button("▶ Lancer l'inférence", variant="primary")

        # 5. outputs
        with gr.Row():
            obs_out = gr.Image(
                label="Observation bruitée",
                image_mode="RGB",
                height=320,
            )

        with gr.Row():
            traj_out = gr.Image(
                label="Image actuelle",
                image_mode="RGB",
                height=320,
            )
            recon_out = gr.Image(
                label="Résultat final",
                image_mode="RGB",
                height=320,
            )

        gr.Markdown(
            """
            ---
            # 📝 Sauvegarder et Soumettre
            1. Une fois satisfait de votre 'galaxie', faites un **clic droit** sur l'image du **Résultat final** et choisissez **Enregistrer l'image sous...**
            2. Rendez-vous sur le lien suivant: [**Padlet de soumission**](https://padlet.com/missaelgabo/submission-request/qPBkXlaal12dveOl?section=338293155)
            3. Cliquez sur le bouton **+**, écrivez votre **nom** comme titre et **téléchargez** votre image.
            4. **N'oubliez pas** de noter le mot de passe qui sera affiché sur l'écran en classe pour pouvoir voter!
            """
        )

        # 7. hidden state objects
        high_res_state = gr.State()
        A_state = gr.State()
        y_obs_state = gr.State()
        sigma_psf_state = gr.State()
        sigma_noise_state = gr.State()
        S_state = gr.State()

        # wire up buttons
        apply_btn.click(
            fn=apply_noise,
            inputs=[
                src_choice,
                upload_img,
                camera_img,
                sigma_psf_slider,
                sigma_noise_slider,
                downsample_slider,
            ],
            outputs=[
                obs_out,
                high_res_state,
                A_state,
                y_obs_state,
                sigma_psf_state,
                sigma_noise_state,
                S_state,
            ],
        )

        run_btn.click(
            fn=stream_inference,
            inputs=[
                high_res_state,
                A_state,
                y_obs_state,
                sigma_psf_state,
                sigma_noise_state,
                S_state,
            ],
            outputs=[
                traj_out,   # streamed frames (magma RGB)
                recon_out,  # only filled on the final yield
            ],
        )

    # actually launch the app
    demo.launch(share=share, debug=debug)
    return demo
