import torch
from tqdm import tqdm

from .Rin import Rin
from .utils import diffusion_utils
from einops import rearrange
from .utils.mask import downsample_mask
from .utils.pos_embedding import create_2d_sin_cos_pos_emb
from torch.nn.functional import pad

def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    # N, C, H, W -> N, T, D
    n, c, h, w = x.shape
    nh, nw = h // p, w // p
    x = x.view(n, c, nh, p, nw, p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.view(n, nh * nw, p * p * c)
    return x

def unpatchify(x: torch.Tensor, nh: int, nw: int, p: int, c: int) -> torch.Tensor:
    # N, T, D -> N, C, H, W
    n, _, _ = x.shape
    x = x.view(n, nh, nw, p, p, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(n, c, nh * p, nw * p)
    return x

class RinDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        rin: Rin,
        train_schedule: str,
        inference_schedule: str,
        pred_type: str,
        self_cond: str = "none",
        num_classes: int = 1000,
        conditional: str = "class",
        self_cond_rate: float = 0.9,
        loss_type: str = "x",
    ):
        super().__init__()
        self._inference_schedule = inference_schedule
        self._pred_type = pred_type
        self._self_cond = self_cond
        self._num_classes = num_classes
        self._conditional = conditional
        self._self_cond_rate = self_cond_rate
        self._loss_type = loss_type

        self.scheduler = diffusion_utils.Scheduler(train_schedule)

        self.denoiser = rin

    def denoise(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        cond: torch.Tensor | None,
        masks: torch.Tensor | None = None,
        pos_embs: torch.Tensor | None = None,
        nmh: torch.Tensor | None = None,
        nmw: torch.Tensor | None = None,
        block_masks: torch.Tensor | None = None,
        num_images: int = 1,
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure gamma is a 1-D tensor of length batch_size
        gamma = gamma.squeeze()
        if gamma.ndim == 0:
            gamma = gamma.unsqueeze(0)  # shape (1,)

        # Debug prints to help trace shape issues later on
        if gamma.ndim != 1:
            print(f"[DEBUG] Unexpected gamma shape after squeeze: {gamma.shape}")

        # Expand gamma if necessary to match the batch dimension
        if gamma.size(0) != x.size(0):
            gamma = gamma.expand(x.size(0))

        # print(f'pos_embs_denoise: {pos_embs.shape}')
        output, latent, tape = self.denoiser(x, gamma, masks, cond, pos_embs, nmh, nmw, block_masks, num_images, latent_prev, tape_prev)
        return output, latent, tape

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 64,
        tape_dim: int = 256,
        iterations: int = 100,
        method: str = "ddim",
        seed: int | None = None,
        class_override: int | None = None,
        mask: torch.Tensor | None = None,
        image_height: int = 32,
        image_width: int = 32,
    ):
        """
        Generate samples using the given diffusion model.

        Args:
            num_samples: Number of samples to generate.
            iterations: Number of sampling steps (diffusion timesteps).
            method: Sampler name (e.g. "ddim", "ddpm", etc.).
            seed: Optional random seed for reproducible sampling.
            class_override: If set, use this single class for all samples.
            mask: Optionally supply a float/binary mask of shape:
                  (num_samples, image_height, image_width) or (num_samples, 1, image_height, image_width).
                  If None, a default mask of all 1s is used.

        Returns:
            (samples) Generated images in [0, 1].
        """
        device = self.denoiser.device
        image_shape = [self.denoiser._image_channels, image_height, image_width]  # [C, H, W]
        patch_size = self.denoiser._patch_size
        nmh = image_shape[1] // patch_size
        nmw = image_shape[2] // patch_size

        # Prepare class-conditional input if needed
        if self._conditional == "class":
            if class_override is not None:
                cond_classes = torch.full(
                    [num_samples],
                    class_override,
                    device=device,
                    dtype=torch.long
                )
            else:
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device).manual_seed(seed)
                cond_classes = torch.randint(
                    self._num_classes,
                    [num_samples],
                    device=device,
                    generator=generator,
                )
            cond = torch.nn.functional.one_hot(cond_classes, self._num_classes).float()
        else:
            cond = None

        # Create initial noise per sample (standard batch mode)
        samples_shape = [num_samples, *image_shape]  # (B , C , H , W)
        samples = self.scheduler.sample_noise(samples_shape, device=device, seed=seed)
        samples = patchify(samples, patch_size)  # (B , T , D)

        # If no mask is provided, default to an all-ones mask in the *image* resolution.
        if mask is None:
            mask = torch.ones(
                (num_samples, image_shape[1], image_shape[2]),
                dtype=torch.float,
                device=device,
            )
        # If mask has shape (B, 1, H, W), reduce it to (B, H, W) for convenience
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        # Downsample the mask per sample to patch resolution
        mask_out = downsample_mask(mask, patch_size)  # (B , H/p , W/p)
        mask_out = rearrange(mask_out, "b h w -> b (h w)").bool()  # (B , T)

        height = mask.shape[1] // patch_size
        width = mask.shape[2] // patch_size
        # Positional embedding for one image is reused; broadcasting over batch works automatically
        pos_embs = create_2d_sin_cos_pos_emb(height, width, tape_dim)  # (T , D)

        # Prepare schedule transforms
        if self._inference_schedule is None:
            time_transform = self.scheduler.time_transform
        else:
            time_transform = self.scheduler.get_time_transform(
                self._inference_schedule
            )

        # Helper to get the (1 - t/iterations) step
        def get_step(t_step):
            return torch.full([num_samples, 1, 1], 1.0 - t_step / iterations, device=device)

        data_pred = torch.zeros_like(samples, device=device)
        latent_prev = None
        tape_prev = None

        for t in tqdm(
            torch.arange(iterations, dtype=torch.float32, device=device),
            desc="sampling",
            leave=False,
        ):
            time_step = get_step(t)
            time_step_p = torch.max(get_step(t + 1), torch.tensor(0.0, device=device))
            gamma, gamma_prev = time_transform(time_step), time_transform(time_step_p)
            # Drop last dimension of gamma and gamma_prev
            # gamma = gamma.squeeze(-1)
            # gamma_prev = gamma_prev.squeeze(-1)

            # Denoise with current samples.  `num_images` equals the physical batch size
            # (one logical image per batch element in the sampling path).
            pred_out, latent_prev, tape_prev = self.denoise(
                samples,
                gamma,
                cond,
                mask_out,
                pos_embs,
                nmh,
                nmw,
                block_masks=None,
                num_images=1,
                latent_prev=latent_prev,
                tape_prev=tape_prev,
            )

            # Convert model output to x0 and eps
            x0_eps = diffusion_utils.get_x0_eps(
                samples,
                gamma,
                pred_out,
                self._pred_type,
                truncate_noise=True,
                clip_x0=True,
            )
            noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]

            # Take one sampling step
            samples = self.scheduler.transition_step(
                samples=samples,
                data_pred=data_pred,
                noise_pred=noise_pred,
                gamma_now=gamma,
                gamma_prev=gamma_prev,
                sampler_name=method,
            )

            # print(f'samples_t: {samples.shape}')

        # Map final samples from [-1,1] to [0,1]
        samples = data_pred * 0.5 + 0.5
        samples.clamp_(0.0, 1.0)

        # Unpack tokens → (N , T , D)
        samples = samples.reshape(num_samples, nmh * nmw, -1)
        samples = unpatchify(samples, nmh, nmw, patch_size, image_shape[0])
        return samples

    def noise_denoise(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
        pos_embs: torch.Tensor,
        nmh: torch.Tensor,
        nmw: torch.Tensor,
        block_masks: torch.Tensor,
        num_images: int = 1,
        t: torch.Tensor | None = None,
    ):
        # Guarantee a batch dimension. If images come as (T, D), treat it as batch size 1.
        if images.ndim == 2:
            images = images.unsqueeze(0)
            masks = masks.unsqueeze(0) if masks is not None else masks

        images = images * 2.0 - 1.0
        images_noised, noise, _, gamma = self.scheduler.add_noise(images, t=t)
        # print(f'pos_embs_noise_denoise: {pos_embs.shape}')
        # print(f'images_noised: {images_noised.shape}')

        total_slots = num_images * self.denoiser.latent_shape[0]
        latent_prev = torch.zeros((1, total_slots, self.denoiser.latent_shape[1]), device=images.device)
        tape_prev = torch.zeros((1, *self.denoiser.tape_shape), device=images.device)
        # TODO: add self-cond with correct masking inside of one batch ie filter by index and not directly by mask
        # if self._self_cond != "none" and self._self_cond_rate > 0.0:
        #     mask = torch.rand(bsz) < self._self_cond_rate

        #     if torch.any(mask):
        #         # print(f'mask: {mask}')
        #         # print(f'mask shape: {mask.shape}')
        #         # print(f'tape_prev shape: {tape_prev.shape}')
        #         with torch.no_grad():
        #             _, latent_prev_out, tape_prev_out = self.denoise(
        #                 x=images_noised[mask],
        #                 gamma=gamma[mask],
        #                 cond=labels[mask],
        #                 masks=masks[mask],
        #                 pos_embs=pos_embs[mask],
        #                 nmh=nmh,
        #                 nmw=nmw,
        #                 block_masks=block_masks,
        #             )

        #         # print(f'latent_prev_out: {latent_prev_out.shape}')
        #         # print(f'tape_prev_out: {tape_prev_out.shape}')

        #         latent_prev[mask] = latent_prev_out.detach()
        #         tape_prev[mask] = tape_prev_out.detach()

        # Debug prints for tracing shapes
        # print(f"[DEBUG] images_noised: {images_noised.shape}, masks: {masks.shape if masks is not None else None}, pos_embs: {pos_embs.shape}")
        # print(f"[DEBUG] gamma (before denoise): {gamma.shape}")

        # pass masks to denoise
        denoise_out, _, _ = self.denoise(images_noised, gamma, labels, masks, pos_embs, nmh, nmw, block_masks, num_images, latent_prev, tape_prev)
        # print(f"[DEBUG] denoise_out: {denoise_out.shape}")

        # Flatten the output to 2D for direct comparison
        denoise_out = denoise_out.reshape(-1, denoise_out.shape[-1])  # (N_pred, D)

        # Flatten xt to same 2-D shape
        xt_flat = images_noised.reshape(-1, images_noised.shape[-1])  # (N_tokens, D)

        # Take the prefix that matches the number of predictions. This matches how Rin.readout_tape
        # keeps only the first (nmh*nmw) tokens.
        if denoise_out.size(0) > xt_flat.size(0):
            raise RuntimeError(
                f"Predicted {denoise_out.size(0)} tokens but only have {xt_flat.size(0)} input tokens")
        xt_aligned = xt_flat[: denoise_out.size(0)]

        # Reduce gamma to scalar for broadcasting
        gamma_scalar = gamma.reshape(-1)[0]

        pred_dict = diffusion_utils.get_x0_eps(
            xt_aligned, gamma_scalar, denoise_out, self._pred_type, truncate_noise=False, clip_x0=True
        )
        return images, noise, images_noised, pred_dict

    def compute_loss(
        self,
        images: torch.Tensor,
        noise: torch.Tensor,
        pred_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._loss_type == "x":
            loss = torch.nn.functional.mse_loss(images, pred_dict["data_pred"])
        elif self._loss_type == "eps":
            loss = torch.nn.functional.mse_loss(noise, pred_dict["noise_pred"])
        else:
            raise ValueError(f"Unknown loss_type `{self._pred_type}`")
        return loss

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        image_mask: torch.Tensor,
        labels: torch.Tensor,
        pos_embs: torch.Tensor,
        nmh: torch.Tensor,
        nmw: torch.Tensor,
        block_masks: torch.Tensor,
        num_images: int = 1,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # print(f'pos_embs_fwd_diff: {pos_embs.shape}')
        # print(f'nmh: {nmh}')
        # print(f'nmw: {nmw}')
        # print(f'images_init: {images.shape}')
        # print(f'masks: {masks.shape}')
        # print(f'image_masks: {image_mask.shape}')
        images, noise, _, pred_dict = self.noise_denoise(images, masks, labels, pos_embs, nmh, nmw, block_masks, num_images=num_images, t=t)
        # print(f'images: {images.shape}')
        # print(f'noise: {noise.shape}')
        # print(f'pred_noise: {pred_dict["noise_pred"].shape}')

        # image_mask = image_mask.unsqueeze(-1)  # Add channel dim
        # image_mask = image_mask.expand(-1, 3, -1, -1)  # Expand across all 3 channels
        # print(f'image_mask: {image_mask.shape}')
        # print(f'masks: {masks.shape}')
        # print(f'images: {images.shape}')
        # Align ground-truth and noise tensors to the number of tokens predicted
        pred_len = pred_dict["data_pred"].shape[0]
        images = images.reshape(-1, images.shape[-1])[:pred_len]
        noise = noise.reshape(-1, noise.shape[-1])[:pred_len]
        loss = self.compute_loss(images, noise, pred_dict)
        return loss