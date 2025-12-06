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
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
        # tape_padding_mask: torch.Tensor | None = None,
        tape_pos_emb: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = gamma.squeeze()
        assert gamma.ndim == 1
        output, latent, tape = self.denoiser(
            x,
            gamma,
            cond,
            latent_prev,
            tape_prev,
            # tape_padding_mask=tape_padding_mask,
            tape_pos_emb=tape_pos_emb,
            offsets=offsets,
            doc_ids=doc_ids,
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

        # Create doc_ids and offsets for packed format (all samples have same seq_len)
        doc_ids = torch.arange(num_samples, device=device).repeat_interleave(seq_len)
        offsets = torch.arange(num_samples + 1, device=device) * seq_len

        # Pack positional embeddings: [num_samples, seq_len, dim] -> [num_samples * seq_len, dim]
        auto_pos_emb_packed = auto_pos_emb.reshape(-1, auto_pos_emb.shape[-1])

        get_step = lambda t: torch.full([num_samples], 1.0 - t / iterations, device=device)
        if self._inference_schedule is None:
            time_transform = self.scheduler.time_transform
        else:
            time_transform = self.scheduler.get_time_transform(self._inference_schedule)

        # Sample noise in packed format: [num_samples * seq_len, patch_dim]
        samples = self.scheduler.sample_noise([num_samples * seq_len, patch_dim], device=device, seed=seed)
        data_pred = torch.zeros_like(samples, device=device)

        latent_prev = None
        tape_prev = None

        guidance_scale = max(self._guidance, 0.0) if cond is not None else 0.0
        cond_null = torch.zeros_like(cond) if (cond is not None and guidance_scale > 0.0) else None

        for t in tqdm(
            torch.arange(iterations, dtype=torch.float32, device=device), desc="sampling", leave=False, position=1
        ):
            time_step = get_step(t)
            time_step_p = torch.clamp(get_step(t + 1), min=0.0)
            gamma, gamma_prev = time_transform(time_step), time_transform(time_step_p)
            
            # Expand gamma to per-token for diffusion math: [num_samples] -> [num_samples * seq_len, 1]
            gamma_expanded = gamma[doc_ids].unsqueeze(-1)
            gamma_prev_expanded = gamma_prev[doc_ids].unsqueeze(-1)

            pred_cond, latent_prev, tape_prev = self.denoise(
                samples,
                gamma,
                cond,
                latent_prev,
                tape_prev,
                tape_pos_emb=auto_pos_emb_packed,
                offsets=offsets,
                doc_ids=doc_ids,
            )
            final_pred = pred_cond
            if guidance_scale > 0.0 and cond_null is not None:
                pred_uncond, _, _ = self.denoise(
                    samples,
                    gamma,
                    cond_null,
                    latent_prev,
                    tape_prev,
                    tape_pos_emb=auto_pos_emb_packed,
                    offsets=offsets,
                    doc_ids=doc_ids,
                )
                final_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            x0_eps = diffusion_utils.get_x0_eps(
                samples, gamma_expanded, final_pred, self._pred_type, truncate_noise=True, clip_x0=True
            )
            noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]
            samples = self.scheduler.transition_step(
                samples=samples,
                data_pred=data_pred,
                noise_pred=noise_pred,
                gamma_now=gamma_expanded,
                gamma_prev=gamma_prev_expanded,
                sampler_name=method,
            )

        # Convert from packed [num_samples * seq_len, patch_dim] to batched [num_samples, seq_len, patch_dim]
        tokens = data_pred.view(num_samples, seq_len, patch_dim)
        tokens = tokens * 0.5 + 0.5  # convert -1,1 -> 0,1
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
        doc_ids: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ):
        tokens = tokens * 2.0 - 1.0
        tokens_noised, noise, gamma_per_doc, gamma = self.scheduler.add_noise(tokens, doc_ids, t=t)

        # bsz, seq_len, _ = tokens.size()
        num_docs = doc_ids.max().item() + 1
        # latent_prev = torch.zeros((bsz, *self.denoiser.latent_shape), device=tokens.device)
        # tape_prev = torch.zeros((bsz, seq_len, self.denoiser.tape_dim), device=tokens.device)

        latent_prev = None
        tape_prev = None

        # if attn_mask is not None:
        #     attn_mask = attn_mask.to(tokens.device).bool()
        if tape_pos_emb is not None:
            tape_pos_emb = tape_pos_emb.to(tokens.device)

        labels = self._apply_cond_dropout(labels)

        latent_prev = None
        tape_prev = None

        if self._self_cond != "none" and self._self_cond_rate > 0.0:
            # Determine which docs get self-conditioning (per-doc mask)
            sc_mask = torch.rand(num_docs, device=tokens.device) < self._self_cond_rate

            if torch.any(sc_mask):
                with torch.no_grad():
                    _, latent_prev_out, tape_prev_out = self.denoise(
                        x=tokens_noised,
                        gamma=gamma_per_doc,
                        cond=labels if labels is not None else None,
                        tape_pos_emb=tape_pos_emb if tape_pos_emb is not None else None,
                        offsets=offsets,
                        doc_ids=doc_ids,
                    )
                    # Zero out latent_prev for docs that shouldn't get self-cond
                    # latent_prev is [num_docs * latent_slots, latent_dim]
                    latent_slots = self.denoiser._latent_slots
                    latent_dim = self.denoiser._latent_dim
                    latent_prev = latent_prev_out.detach().view(num_docs, latent_slots, latent_dim)
                    latent_prev = latent_prev * sc_mask.view(-1, 1, 1)  # Zero out non-masked docs
                    latent_prev = latent_prev.view(-1, latent_dim)
                    
                    # tape_prev is [total_tokens, tape_dim] - mask per-token based on doc membership
                    # doc_ids tells us which doc each token belongs to
                    tape_prev = tape_prev_out.detach()
                    # Create per-token mask from per-doc mask: sc_mask[doc_ids] gives mask for each token
                    token_mask = sc_mask[doc_ids].unsqueeze(-1)  # [total_tokens, 1]
                    tape_prev = tape_prev * token_mask  # Zero out tokens from non-masked docs

        denoise_out, _, _ = self.denoise(
            tokens_noised,
            gamma_per_doc,
            labels,
            latent_prev,
            tape_prev,
            tape_pos_emb=tape_pos_emb,
            offsets=offsets,
            doc_ids=doc_ids,
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

        loss = diff.mean()
        return loss

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        tape_pos_emb: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # mask = attn_mask
        # if mask is not None:
        #     mask = mask.to(tokens.device).bool()
        data, noise, _, pred_dict = self.noise_denoise(
            tokens,
            labels,
            t=t,
            # attn_mask=mask,
            tape_pos_emb=tape_pos_emb,
            doc_ids=doc_ids,
            offsets=offsets,
        )
        loss = self.compute_loss(data, noise, pred_dict)
        return loss

    def _apply_cond_dropout(self, labels: torch.Tensor | None, force_drop: bool = False):
        if labels is None:
            return None
        if force_drop:
            return torch.zeros_like(labels)
        if self._cond_dropout <= 0.0:
            return labels
        drop_mask = (torch.rand(labels.size(0), device=labels.device) > self._cond_dropout).float().unsqueeze(-1)
        return labels * drop_mask
