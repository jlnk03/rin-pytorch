import torch
from tqdm import tqdm

from .Rin import Rin
from .utils import diffusion_utils
from .utils.data_utils import patchify, unpatchify
from .utils.logging_utils import log_first_document
from .utils.pos_embedding import create_2d_sin_cos_pos_emb
from .utils.ragged_tensor import get_document_ids, ragged_list_to_tensor


class RinDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        rin: Rin,
        train_schedule: str,
        inference_schedule: str,
        pred_type: str,
        self_cond: str = "none",
        num_classes: int = 10,
        conditional: str = "class",
        self_cond_rate: float = 0.9,
        loss_type: str = "x",
        cond_dropout: float = 0.0,
        guidance: float = 0.0,
    ):
        super().__init__()
        self._inference_schedule = inference_schedule
        self._pred_type = pred_type
        self._self_cond = self_cond
        self._num_classes = num_classes
        self._conditional = conditional
        self._self_cond_rate = self_cond_rate
        self._loss_type = loss_type
        self._cond_dropout = cond_dropout
        self._guidance = guidance

        self.scheduler = diffusion_utils.Scheduler(train_schedule)
        self.denoiser = rin

    def denoise(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        cond: torch.Tensor | None,
        pos_embs: torch.Tensor,
        offsets: torch.Tensor,
        offsets_pos_embs: torch.Tensor,
        document_ids: torch.Tensor,
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = gamma.squeeze()
        output, latent, tape = self.denoiser(
            x,
            gamma,
            cond,
            pos_embs,
            offsets,
            offsets_pos_embs,
            document_ids,
            latent_prev,
            tape_prev,
        )
        return output, latent, tape

    @torch.no_grad()
    def sample(
        self,
        num_samples=64,
        tape_dim: int | None = None,
        iterations=100,
        method="ddim",
        seed=None,
        class_override=None,
        image_height: int | None = None,
        image_width: int | None = None,
    ):
        channels, default_height, default_width = self.denoiser.image_shape
        patch_size = self.denoiser.patch_size
        patch_dim = self.denoiser.patch_dim
        target_height = image_height if image_height is not None else default_height
        target_width = image_width if image_width is not None else default_width
        if target_height % patch_size != 0 or target_width % patch_size != 0:
            raise ValueError("image_height and image_width must be divisible by patch size.")

        device = self.denoiser.device
        nh = target_height // patch_size
        nw = target_width // patch_size
        seq_len = nh * nw

        if self._conditional == "class":
            if class_override is not None:
                cond_classes = torch.full([num_samples], class_override, device=device, dtype=torch.long)
            else:
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device).manual_seed(seed)
                cond_classes = torch.randint(self._num_classes, [num_samples], device=device, generator=generator)
            cond = torch.nn.functional.one_hot(cond_classes, self._num_classes).float()
        else:
            cond = None

        noise_images = self.scheduler.sample_noise(
            [num_samples, channels, target_height, target_width], device=device, seed=seed
        )
        samples_list = [patchify(noise_images[i], patch_size) for i in range(num_samples)]
        samples_flat, offsets = ragged_list_to_tensor(samples_list)
        offsets = offsets.to(device)
        offsets_pos_embs = offsets.clone()
        document_ids = get_document_ids(offsets)

        tape_dim_value = tape_dim if tape_dim is not None else self.denoiser.tape_dim
        pos_emb = create_2d_sin_cos_pos_emb(nh, nw, tape_dim_value).view(1, seq_len, tape_dim_value)
        pos_list = [pos_emb.squeeze(0) for _ in range(num_samples)]
        pos_flat, pos_offsets = ragged_list_to_tensor(pos_list)
        pos_flat = pos_flat.to(device)
        pos_offsets = pos_offsets.to(device)

        def _get_step(t):
            return torch.full([num_samples], 1.0 - t / iterations, device=device)

        if self._inference_schedule is None:
            time_transform = self.scheduler.time_transform
        else:
            time_transform = self.scheduler.get_time_transform(self._inference_schedule)

        latent_prev = None
        tape_prev = None

        guidance_scale = max(self._guidance, 0.0) if cond is not None else 0.0
        cond_null = torch.zeros_like(cond) if (cond is not None and guidance_scale > 0.0) else None

        for t in tqdm(torch.arange(iterations, dtype=torch.float32, device=device), desc="sampling", leave=False):
            time_step = _get_step(t)
            time_step_p = torch.max(_get_step(t + 1), torch.tensor(0.0, device=device))
            gamma, gamma_prev = time_transform(time_step), time_transform(time_step_p)

            patches_per_sample = samples_flat.shape[0] // num_samples
            gamma_tokens = torch.repeat_interleave(gamma, patches_per_sample, dim=0).unsqueeze(-1)
            gamma_prev_tokens = torch.repeat_interleave(gamma_prev, patches_per_sample, dim=0).unsqueeze(-1)

            pred_cond, latent_prev, tape_prev = self.denoise(
                samples_flat,
                gamma_tokens,
                cond,
                pos_flat,
                offsets,
                pos_offsets,
                document_ids,
                latent_prev,
                tape_prev,
            )
            final_pred = pred_cond
            if guidance_scale > 0.0 and cond_null is not None:
                pred_uncond, _, _ = self.denoise(
                    samples_flat,
                    gamma_tokens,
                    cond_null,
                    pos_flat,
                    offsets,
                    pos_offsets,
                    document_ids,
                    latent_prev,
                    tape_prev,
                )
                final_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            x0_eps = diffusion_utils.get_x0_eps(samples_flat, gamma_tokens, final_pred, self._pred_type, truncate_noise=True, clip_x0=True)
            noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]
            samples_flat = self.scheduler.transition_step(
                samples=samples_flat,
                data_pred=data_pred,
                noise_pred=noise_pred,
                gamma_now=gamma_tokens,
                gamma_prev=gamma_prev_tokens,
                sampler_name=method,
            )

        data_pred = data_pred.view(num_samples, seq_len, patch_dim)
        tokens = data_pred * 0.5 + 0.5  # convert -1,1 -> 0,1
        tokens.clamp_(0.0, 1.0)
        images = unpatchify(tokens, patch_size, channels, target_height, target_width)
        # images = unpatchify(data_pred * 0.5 + 0.5, patch_size, channels, target_height, target_width)
        return images

    def noise_denoise(
        self,
        images: torch.Tensor,
        pos_embs: torch.Tensor,
        labels: torch.Tensor,
        offsets: torch.Tensor,
        offsets_pos_embs: torch.Tensor,
        document_ids: torch.Tensor,
        t: torch.Tensor | None = None,
    ):
        log_first_document("diff.images_in", images, document_ids)
        images = images * 2.0 - 1.0
        log_first_document("diff.images_scaled", images, document_ids)

        if t is None:
            num_docs = labels.shape[0]
            t_per_doc = torch.rand(num_docs, device=images.device)
            t_tokens = t_per_doc[document_ids]
        elif isinstance(t, torch.Tensor) and t.ndim == 1 and t.shape[0] == labels.shape[0]:
            t_tokens = t.to(device=images.device)[document_ids]
        else:
            t_tokens = t

        images_noised, noise, _, gamma = self.scheduler.add_noise(images, t=t_tokens)
        log_first_document("diff.noise", noise, document_ids)
        log_first_document("diff.images_noised", images_noised, document_ids)

        latent_prev = None
        tape_prev = None

        labels = self._apply_cond_dropout(labels)

        use_self_cond = (
            self._self_cond != "none"
            and self._self_cond_rate > 0.0
            and torch.rand(1, device=images.device).item() < self._self_cond_rate
        )
        if use_self_cond:
            with torch.no_grad():
                _, latent_prev_out, tape_prev_out = self.denoise(
                    images_noised,
                    gamma,
                    labels,
                    pos_embs,
                    offsets,
                    offsets_pos_embs,
                    document_ids,
                    latent_prev=None,
                    tape_prev=None,
                )
            latent_prev = latent_prev_out.detach()
            tape_prev = tape_prev_out.detach()

        denoise_out, _, _ = self.denoise(
            images_noised,
            gamma,
            labels,
            pos_embs,
            offsets,
            offsets_pos_embs,
            document_ids,
            latent_prev,
            tape_prev,
        )

        pred_dict = diffusion_utils.get_x0_eps(
            images_noised,
            gamma,
            denoise_out,
            self._pred_type,
            truncate_noise=False,
            clip_x0=True,
        )
        return images, noise, images_noised, pred_dict

    def compute_loss(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        pred_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._loss_type == "x":
            target = data
            pred = pred_dict["data_pred"]
        elif self._loss_type == "eps":
            target = noise
            pred = pred_dict["noise_pred"]
        else:
            raise ValueError(f"Unknown loss_type `{self._pred_type}`")
        return torch.nn.functional.mse_loss(pred, target)

    def forward(
        self,
        images: torch.Tensor,
        pos_embs: torch.Tensor,
        labels: torch.Tensor,
        offsets: torch.Tensor,
        offsets_pos_embs: torch.Tensor,
        document_ids: torch.Tensor,
    ) -> torch.Tensor:
        data, noise, _, pred_dict = self.noise_denoise(
            images,
            pos_embs,
            labels,
            offsets,
            offsets_pos_embs,
            document_ids,
        )
        return self.compute_loss(data, noise, pred_dict)

    def _apply_cond_dropout(self, labels: torch.Tensor | None, force_drop: bool = False):
        if labels is None:
            return None
        if force_drop:
            return torch.zeros_like(labels)
        if self._cond_dropout <= 0.0:
            return labels
        drop_mask = (torch.rand(labels.size(0), device=labels.device) > self._cond_dropout).float().unsqueeze(-1)
        return labels * drop_mask
