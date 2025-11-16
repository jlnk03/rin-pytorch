import torch
from tqdm import tqdm

from .Rin import Rin
from .utils import diffusion_utils
from .utils.data_utils import unpatchify
from .utils.pos_embedding import create_2d_sin_cos_pos_emb


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
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
        tape_padding_mask: torch.Tensor | None = None,
        tape_pos_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = gamma.squeeze()
        assert gamma.ndim == 1
        output, latent, tape = self.denoiser(
            x,
            gamma,
            cond,
            latent_prev,
            tape_prev,
            tape_padding_mask=tape_padding_mask,
            tape_pos_emb=tape_pos_emb,
        )
        return output, latent, tape

    @torch.no_grad()
    def sample(
        self,
        num_samples=64,
        iterations=100,
        method="ddim",
        seed=None,
        class_override=None,
        image_height: int | None = None,
        image_width: int | None = None,
        tape_dim: int | None = None,
        tape_pos_emb: torch.Tensor | None = None,
    ):
        channels, default_height, default_width = self.denoiser.image_shape
        patch_size = self.denoiser.patch_size
        patch_dim = self.denoiser.patch_dim
        target_height = image_height if image_height is not None else default_height
        target_width = image_width if image_width is not None else default_width
        if target_height % patch_size != 0 or target_width % patch_size != 0:
            raise ValueError("image_height and image_width must be divisible by patch size.")
        nh = target_height // patch_size
        nw = target_width // patch_size
        seq_len = nh * nw
        samples_shape = [num_samples, seq_len, patch_dim]
        device = self.denoiser.device
        if self._conditional == "class":
            # generate random classes
            if class_override is not None:
                cond = torch.full([num_samples], class_override, device=device, dtype=torch.long)
            else:
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device).manual_seed(seed)
                cond = torch.randint(self._num_classes, [num_samples], device=device, generator=generator)
            cond = torch.nn.functional.one_hot(cond, self._num_classes).float()
        else:
            cond = None

        get_step = lambda t: torch.full([num_samples, 1, 1], 1.0 - t / iterations, device=device)
        if self._inference_schedule is None:
            time_transform = self.scheduler.time_transform
        else:
            time_transform = self.scheduler.get_time_transform(self._inference_schedule)

        samples = self.scheduler.sample_noise(samples_shape, device=device, seed=seed)
        data_pred = torch.zeros_like(samples, device=device)

        latent_prev = None
        tape_prev = None

        def _prepare_pos_emb(pos_emb: torch.Tensor) -> torch.Tensor:
            if pos_emb.ndim == 2:
                pos_emb = pos_emb.unsqueeze(0)
            if pos_emb.size(0) == 1 and num_samples > 1:
                pos_emb = pos_emb.repeat(num_samples, 1, 1)
            if pos_emb.size(0) != num_samples:
                raise ValueError("tape_pos_emb batch dimension must equal num_samples or be 1.")
            if pos_emb.size(1) != seq_len:
                raise ValueError("tape_pos_emb sequence length must match target sequence length.")
            return pos_emb

        auto_pos_emb = None
        if tape_pos_emb is not None:
            auto_pos_emb = _prepare_pos_emb(tape_pos_emb)
        else:
            tape_dim_value = tape_dim if tape_dim is not None else self.denoiser.tape_dim
            base_pos = create_2d_sin_cos_pos_emb(nh, nw, tape_dim_value).view(1, seq_len, tape_dim_value)
            auto_pos_emb = base_pos.repeat(num_samples, 1, 1)

        if auto_pos_emb is not None:
            auto_pos_emb = auto_pos_emb.to(device)

        for t in tqdm(
            torch.arange(iterations, dtype=torch.float32, device=device), desc="sampling", leave=False, position=1
        ):
            time_step = get_step(t)
            time_step_p = torch.max(get_step(t + 1), torch.tensor(0.0, device=device))
            gamma, gamma_prev = time_transform(time_step), time_transform(time_step_p)

            pred_out, latent_prev, tape_prev = self.denoise(
                samples,
                gamma,
                cond,
                latent_prev,
                tape_prev,
                tape_pos_emb=auto_pos_emb,
            )
            x0_eps = diffusion_utils.get_x0_eps(
                samples, gamma, pred_out, self._pred_type, truncate_noise=True, clip_x0=True
            )
            noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]
            samples = self.scheduler.transition_step(
                samples=samples,
                data_pred=data_pred,
                noise_pred=noise_pred,
                gamma_now=gamma,
                gamma_prev=gamma_prev,
                sampler_name=method,
            )

        tokens = data_pred * 0.5 + 0.5  # convert -1,1 -> 0,1
        tokens.clamp_(0.0, 1.0)
        images = unpatchify(tokens, patch_size, channels, target_height, target_width)
        return images

    def noise_denoise(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        tape_pos_emb: torch.Tensor | None = None,
    ):
        tokens = tokens * 2.0 - 1.0
        tokens_noised, noise, _, gamma = self.scheduler.add_noise(tokens, t=t)

        bsz, seq_len, _ = tokens.size()
        latent_prev = torch.zeros((bsz, *self.denoiser.latent_shape), device=tokens.device)
        tape_prev = torch.zeros((bsz, seq_len, self.denoiser.tape_dim), device=tokens.device)

        if attn_mask is not None:
            attn_mask = attn_mask.to(tokens.device).bool()
        if tape_pos_emb is not None:
            tape_pos_emb = tape_pos_emb.to(tokens.device)

        if self._self_cond != "none" and self._self_cond_rate > 0.0:
            mask = torch.rand(bsz, device=tokens.device) < self._self_cond_rate

            if torch.any(mask):
                with torch.no_grad():
                    _, latent_prev_out, tape_prev_out = self.denoise(
                        x=tokens_noised[mask],
                        gamma=gamma[mask],
                        cond=labels[mask],
                        tape_padding_mask=attn_mask[mask] if attn_mask is not None else None,
                        tape_pos_emb=tape_pos_emb[mask] if tape_pos_emb is not None else None,
                    )

                latent_prev[mask] = latent_prev_out.detach()
                tape_prev[mask] = tape_prev_out.detach()

        denoise_out, _, _ = self.denoise(
            tokens_noised,
            gamma,
            labels,
            latent_prev,
            tape_prev,
            tape_padding_mask=attn_mask,
            tape_pos_emb=tape_pos_emb,
        )

        pred_dict = diffusion_utils.get_x0_eps(
            tokens_noised, gamma, denoise_out, self._pred_type, truncate_noise=False, clip_x0=True
        )
        return tokens, noise, tokens_noised, pred_dict

    def compute_loss(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        pred_dict: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._loss_type == "x":
            target = data
            pred = pred_dict["data_pred"]
        elif self._loss_type == "eps":
            target = noise
            pred = pred_dict["noise_pred"]
        else:
            raise ValueError(f"Unknown loss_type `{self._pred_type}`")

        diff = torch.nn.functional.mse_loss(pred, target, reduction="none")
        if mask is not None:
            weight = (~mask).float().unsqueeze(-1)
            diff = diff * weight
            denom = (weight.sum() * diff.size(-1)).clamp(min=1.0)
            loss = diff.sum() / denom
        else:
            loss = diff.mean()
        return loss

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        tape_pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = attn_mask
        if mask is not None:
            mask = mask.to(tokens.device).bool()
        data, noise, _, pred_dict = self.noise_denoise(
            tokens,
            labels,
            t=t,
            attn_mask=mask,
            tape_pos_emb=tape_pos_emb,
        )
        loss = self.compute_loss(data, noise, pred_dict, mask=mask)
        return loss
