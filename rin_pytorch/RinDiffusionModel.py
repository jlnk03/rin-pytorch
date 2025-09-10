import torch
from tqdm import tqdm

from .Rin import Rin
from .utils import diffusion_utils
from einops import rearrange
from .utils.mask import downsample_mask
from .utils.pos_embedding import create_2d_sin_cos_pos_emb
from .utils.ragged_tensor import ragged_list_to_tensor, get_document_ids
from torch.nn.functional import pad

from rin_pytorch.utils.logging_utils import log_first_tensor, log_first_document

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
        # x: torch.Tensor,
        # gamma: torch.Tensor,
        # cond: torch.Tensor | None,
        # masks: torch.Tensor | None = None,
        # pos_embs: torch.Tensor | None = None,
        # nmh: torch.Tensor | None = None,
        # nmw: torch.Tensor | None = None,
        # latent_prev: torch.Tensor | None = None,
        # tape_prev: torch.Tensor | None = None,
        x, gamma, cond, pos_embs, offsets, offsets_pos_embs, document_ids,
        latent_prev, tape_prev,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = gamma.squeeze()
        assert gamma.ndim == 1
        # print(f'pos_embs_denoise: {pos_embs.shape}')
        output, latent, tape = self.denoiser(x, gamma, cond, pos_embs, offsets, offsets_pos_embs, document_ids, latent_prev, tape_prev)
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
        image_height: int = 32,
        image_width: int = 32,
    ):
        """
        Generate samples using the given diffusion model.

        Args:
            num_samples: Number of samples to generate.
            tape_dim: Dimension of the tape embeddings.
            iterations: Number of sampling steps (diffusion timesteps).
            method: Sampler name (e.g. "ddim", "ddpm", etc.).
            seed: Optional random seed for reproducible sampling.
            class_override: If set, use this single class for all samples.
            image_height: Height of the generated images.
            image_width: Width of the generated images.

        Returns:
            (samples) Generated images in [0, 1].
        """
        device = self.denoiser.device
        image_shape = [self.denoiser._image_channels, image_height, image_width]  # [C, H, W]
        patch_size = self.denoiser._patch_size
        nmh = image_shape[1] // patch_size
        nmw = image_shape[2] // patch_size

        # print(f'num classes: {self._num_classes}')

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

        # Prepare shape of initial noise
        samples_shape = [num_samples, *image_shape]  # (B, C, H, W)
        samples = self.scheduler.sample_noise(samples_shape, device=device, seed=seed)
        samples = patchify(samples, patch_size)  # [num_samples, num_patches, patch_dim]
        
        # Store original batch dimensions for reshaping later
        batch_size, num_patches, patch_dim = samples.shape
        
        # Convert to ragged tensor format - flatten all samples
        samples_list = [samples[i] for i in range(num_samples)]  # List of [num_patches, patch_dim] tensors
        samples_flat, offsets = ragged_list_to_tensor(samples_list)  # [total_patches, patch_dim]
        offsets = offsets.to(device)  # Move offsets to the same device as samples
        offsets_pos_embs = offsets.clone()  # Same offsets for positional embeddings
        
        # Create document IDs
        document_ids = get_document_ids(offsets)  # Now this will be on the correct device
        
        # Create positional embeddings for all patches (repeated for each sample)
        height = image_height // patch_size
        width = image_width // patch_size
        single_pos_emb = create_2d_sin_cos_pos_emb(height, width, tape_dim)
        pos_embs_list = [single_pos_emb for _ in range(num_samples)]
        pos_embs_flat, _ = ragged_list_to_tensor(pos_embs_list)  # [total_patches, tape_dim]
        pos_embs_flat = pos_embs_flat.to(device)  # Also ensure pos_embs are on correct device

        # print(f'samples_flat: {samples_flat.shape}')
        # print(f'pos_embs_flat: {pos_embs_flat.shape}')
        # print(f'offsets: {offsets.shape}')
        # print(f'offsets_pos_embs: {offsets_pos_embs.shape}')
        # print(f'document_ids: {document_ids.shape}')
        
        # Note: Mask functionality removed since all samples have the same shape

        # Prepare schedule transforms
        if self._inference_schedule is None:
            time_transform = self.scheduler.time_transform
        else:
            time_transform = self.scheduler.get_time_transform(
                self._inference_schedule
            )

        # Helper to get the (1 - t/iterations) step
        def get_step(t):
            return torch.full(
                [num_samples],  # Still using original num_samples for batch dimension
                1.0 - t / iterations,
                device=device,
            )

        data_pred = torch.zeros_like(samples_flat, device=device)
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
            # Calculate the correct expansion ratio from actual tensor sizes
            patches_per_sample = samples_flat.shape[0] // num_samples
            gamma = torch.repeat_interleave(gamma, patches_per_sample, dim=0)
            gamma_prev = torch.repeat_interleave(gamma_prev, patches_per_sample, dim=0)

            # Add extra dimension for proper broadcasting with [N, D] tensors
            gamma = gamma.unsqueeze(-1)  # [16384] -> [16384, 1]
            gamma_prev = gamma_prev.unsqueeze(-1)  # [16384] -> [16384, 1]

            # Denoise with current samples using new signature
            pred_out, latent_prev, tape_prev = self.denoise(
                samples_flat, gamma, cond, pos_embs_flat, offsets, offsets_pos_embs, document_ids, latent_prev, tape_prev
            )

            # print(f'pred_out: {pred_out.shape}')
            # print(f'samples_flat: {samples_flat.shape}')
            # print(f'gamma: {gamma.shape}')

            # Convert model output to x0 and eps
            x0_eps = diffusion_utils.get_x0_eps(
                samples_flat,
                gamma,
                pred_out,
                self._pred_type,
                truncate_noise=True,
                clip_x0=True,
            )
            noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]

            # Take one sampling step
            samples_flat = self.scheduler.transition_step(
                samples=samples_flat,
                data_pred=data_pred,
                noise_pred=noise_pred,
                gamma_now=gamma,
                gamma_prev=gamma_prev,
                sampler_name=method,
            )

            # print(f'samples_t: {samples.shape}')

        # Map final samples from [-1, 1] into [0, 1], clamp, and return
        samples_final = data_pred * 0.5 + 0.5
        samples_final.clamp_(0.0, 1.0)

        # Reshape back to batch format: [total_patches, patch_dim] -> [num_samples, num_patches, patch_dim]
        samples_final = samples_final.view(num_samples, num_patches, patch_dim)

        samples = unpatchify(samples_final, nmh, nmw, patch_size, image_shape[0])
        return samples

    def noise_denoise(
        self,
        images, pos_embs, labels, offsets, offsets_pos_embs, document_ids,
        t: torch.Tensor | None = None,
    ):

        # Log entire first packed sample (document 0)
        log_first_document("diff.images_in", images, document_ids)
        images = images * 2.0 - 1.0
        # Log scaled image for first document
        log_first_document("diff.images_scaled", images, document_ids)
        # Ensure timestep t is per-document and broadcast to tokens when packing
        if t is None:
            num_docs = labels.shape[0]
            t_per_doc = torch.rand(num_docs, device=images.device)
            t_tokens = t_per_doc[document_ids]
        elif isinstance(t, torch.Tensor) and t.ndim == 1 and t.shape[0] == labels.shape[0]:
            # t provided per-document; expand to tokens
            t_tokens = t.to(device=images.device)[document_ids]
        else:
            # t is either a float or already token-aligned/broadcastable
            t_tokens = t
        images_noised, noise, _, gamma = self.scheduler.add_noise(images, t=t_tokens)
        log_first_document("diff.noise", noise, document_ids)
        log_first_document("diff.images_noised", images_noised, document_ids)

        bsz = labels.shape[0]

        tape_length = images_noised.shape[0]
        tape_prev = torch.zeros((tape_length, self.denoiser.tape_dim), device=images.device)
        latent_length = bsz * self.denoiser._latent_slots
        latent_prev = torch.zeros((latent_length, self.denoiser.latent_dim), device=images.device)

        if self._self_cond != "none" and self._self_cond_rate > 0.0:
            # Create document-level mask
            doc_mask = torch.rand(bsz, device=images.device) < self._self_cond_rate

            log_first_tensor("diff.doc_mask", doc_mask)
            log_first_document("diff.latent_prev", latent_prev, document_ids)
            log_first_document("diff.tape_prev", tape_prev, document_ids)

            if torch.any(doc_mask):
                
                with torch.no_grad():
                    _, latent_prev_out, tape_prev_out = self.denoise(

                        images_noised,
                        gamma,
                        labels,
                        pos_embs,
                        offsets,
                        offsets_pos_embs,
                        document_ids,
                        latent_prev,
                        tape_prev
                    )

                latent_prev = latent_prev_out.detach()
                tape_prev = tape_prev_out.detach()
                # print(f'latent_prev: {latent_prev.shape}')
                log_first_document("diff.latent_prev_out", latent_prev_out, document_ids)
                log_first_document("diff.tape_prev_out", tape_prev_out, document_ids)

        # pass masks to denoise
        denoise_out, _, _ = self.denoise(images_noised, gamma, labels, pos_embs, offsets, offsets_pos_embs, document_ids, latent_prev, tape_prev)

        pred_dict = diffusion_utils.get_x0_eps(
            images_noised, gamma, denoise_out, self._pred_type, truncate_noise=False, clip_x0=True
        )

        # Log predicted outputs on first item
        if isinstance(pred_dict, dict):
            if "data_pred" in pred_dict:
                log_first_document("diff.data_pred", pred_dict["data_pred"], document_ids)
            if "noise_pred" in pred_dict:
                log_first_document("diff.noise_pred", pred_dict["noise_pred"], document_ids)

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
        images, pos_embs, labels, offsets, offsets_pos_embs, document_ids
    ) -> torch.Tensor:

        images, noise, _, pred_dict = self.noise_denoise(images, pos_embs, labels, offsets, offsets_pos_embs, document_ids)

        log_first_document("diff.images_pre_mask", images, document_ids)
        log_first_document("diff.images_masked", images, document_ids)
        log_first_document("diff.data_pred_masked", pred_dict["data_pred"], document_ids)
        loss = self.compute_loss(images, noise, pred_dict)
        return loss