import torch
import torch.nn as nn
from einops import rearrange

from .modules import (
    LambdaModule,
    MLP,
    ScalarEmbedding,
    TransformerDecoderLayer,
    TransformerEncoder,
)
from .utils.logging_utils import log_first_document, log_first_tensor
from .utils.pos_embedding import create_2d_sin_cos_pos_emb
from .utils.ragged_tensor import get_document_ids


def _concat_tokens(*tokens: torch.Tensor | None) -> torch.Tensor:
    return torch.cat([t for t in tokens if t is not None], dim=-2)


def _concat_tokens_interleave(
    latent: torch.Tensor,
    extra_tokens: torch.Tensor,
    latent_slots_per_sample: int,
) -> torch.Tensor:
    batch_size = extra_tokens.shape[0]
    latent_dim = latent.shape[-1]
    latent = latent.view(batch_size, latent_slots_per_sample, latent_dim)
    latent = torch.cat([latent, extra_tokens], dim=1)
    return latent.view(-1, latent_dim)


class Rin(nn.Module):
    def __init__(
        self,
        num_layers: str,
        latent_slots: int,
        latent_dim: int,
        latent_mlp_ratio: int,
        latent_num_heads: int,
        tape_dim: int,
        tape_mlp_ratio: int,
        rw_num_heads: int,
        image_height: int,
        image_width: int,
        image_channels: int,
        patch_size: int,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        drop_path=0.0,
        drop_units=0.1,
        drop_att=0.0,
        time_scaling=1e4,
        self_cond="none",
        time_on_latent=False,
        cond_on_latent_n=0,
        cond_tape_writable=False,
        cond_dim=0,
        cond_proj=True,
        cond_decoupled_read=False,
        xattn_enc_ln=False,
        num_classes=None,
        pre_tokenized: bool = False,
        mask_ratio: float = 0.0,
    ):
        super().__init__()

        self._image_height = image_height
        self._image_width = image_width
        self._image_channels = image_channels
        self._n_rows = image_height // patch_size
        self._n_cols = image_width // patch_size
        self._num_tokens = int(self._n_rows * self._n_cols * max(1.0 - mask_ratio, 1e-3))
        self._patch_size = patch_size
        self._patch_dim = patch_size ** 2 * image_channels
        self._output_dim = self._patch_dim

        self._num_layers = [int(i) for i in num_layers.split(",")]
        self._latent_slots = latent_slots
        self._cond_on_latent_n = cond_on_latent_n
        self._time_on_latent = time_on_latent
        self._cond_on_latent = cond_on_latent_n > 0
        if self._time_on_latent:
            latent_slots -= 1
        latent_slots -= cond_on_latent_n
        self._latent_dim = latent_dim
        self._tape_slots = self._num_tokens
        self._tape_dim = tape_dim
        self._cond_dim = cond_dim if cond_dim > 0 else tape_dim
        self._latent_pos_encoding = latent_pos_encoding
        self._tape_pos_encoding = tape_pos_encoding
        assert self_cond in ["none", "latent", "latent+tape", "tape"]
        self._self_cond = self_cond
        self._cond_tape_writable = cond_tape_writable
        self._cond_decoupled_read = cond_decoupled_read
        self.pre_tokenized = pre_tokenized

        self.stem = nn.Linear(self._output_dim, tape_dim)
        self.stem_ln = nn.LayerNorm(tape_dim, eps=1e-6)
        self.time_emb = ScalarEmbedding(
            dim=(latent_dim if self._time_on_latent else self._cond_dim) // 4,
            scaling=time_scaling,
            expansion=4,
        )
        if cond_proj:
            if num_classes is None:
                raise ValueError("num_classes must be provided when cond_proj=True")
            self.cond_proj = nn.Linear(num_classes, latent_dim if self._cond_on_latent else self._cond_dim)
        else:
            self.cond_proj = nn.Identity()

        self.make_latent_pos(latent_slots, latent_dim, latent_pos_encoding, time_scaling)
        self.make_tape_pos(tape_dim, tape_pos_encoding, time_scaling)

        if self_cond in ["latent", "latent+tape"]:
            self.latent_prev_proj = MLP(
                num_layers=1,
                dim=latent_dim,
                mlp_ratio=latent_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.latent_prev_ln = nn.LayerNorm(latent_dim, eps=1e-6)
            nn.init.zeros_(self.latent_prev_ln.weight)

        if self_cond in ["tape", "latent+tape"]:
            self.tape_prev_proj = MLP(
                num_layers=1,
                dim=tape_dim,
                mlp_ratio=tape_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.tape_prev_ln = nn.LayerNorm(tape_dim, eps=1e-6)
            nn.init.zeros_(self.tape_prev_ln.weight)

        self.read_units = nn.ModuleList()
        self.read_cond_units = nn.ModuleList()
        self.write_units = nn.ModuleList()
        self.latent_processing_units = nn.ModuleList()

        for num_layers_per_readwrite in self._num_layers:
            self.read_units.append(
                TransformerDecoderLayer(
                    dim=latent_dim,
                    mlp_ratio=latent_mlp_ratio,
                    num_heads=rw_num_heads,
                    drop_path=0.0,
                    drop_units=0.0,
                    drop_att=0.0,
                    dim_x_att=tape_dim,
                    self_attention=False,
                    cross_attention=True,
                    use_mlp=True,
                    use_enc_ln=xattn_enc_ln,
                )
            )
            if cond_decoupled_read:
                self.read_cond_units.append(
                    TransformerDecoderLayer(
                        dim=latent_dim,
                        mlp_ratio=latent_mlp_ratio,
                        num_heads=rw_num_heads,
                        drop_path=0.0,
                        drop_units=0.0,
                        drop_att=0.0,
                        dim_x_att=self._cond_dim,
                        self_attention=False,
                        cross_attention=True,
                        use_mlp=True,
                        use_enc_ln=xattn_enc_ln,
                    )
                )
            if num_layers_per_readwrite == 0:
                self.write_units.append(LambdaModule(lambda x: x))
                self.latent_processing_units.append(LambdaModule(lambda x, _: x))
            else:
                self.write_units.append(
                    TransformerDecoderLayer(
                        dim=tape_dim,
                        mlp_ratio=tape_mlp_ratio,
                        num_heads=rw_num_heads,
                        drop_path=0.0,
                        drop_units=0.0,
                        drop_att=0.0,
                        dim_x_att=latent_dim,
                        self_attention=False,
                        cross_attention=True,
                        use_mlp=tape_mlp_ratio > 0,
                        use_enc_ln=xattn_enc_ln,
                    )
                )
                self.latent_processing_units.append(
                    TransformerEncoder(
                        num_layers=num_layers_per_readwrite,
                        dim=latent_dim,
                        mlp_ratio=latent_mlp_ratio,
                        num_heads=latent_num_heads,
                        drop_path=drop_path,
                        drop_units=drop_units,
                        drop_att=drop_att,
                    )
                )

        self.output_ln = nn.LayerNorm(tape_dim, eps=1e-6)
        self.output_linear = nn.Linear(tape_dim, self._output_dim)

    def make_latent_pos(
        self,
        latent_slots: int,
        latent_dim: int,
        latent_pos_encoding: str,
        time_scaling: float,
    ) -> None:
        if latent_pos_encoding in ["sin_cos", "sin_cos_plus_learned"]:
            self.register_buffer(
                "latent_pos_emb",
                create_2d_sin_cos_pos_emb(
                    n_rows=latent_slots,
                    n_cols=1,
                    dim=latent_dim,
                    normalization_max=time_scaling,
                ),
            )
        if latent_pos_encoding == "learned":
            self.latent_pos_emb = nn.Parameter(torch.zeros(latent_slots, latent_dim))
            nn.init.trunc_normal_(self.latent_pos_emb, std=0.02)
        elif latent_pos_encoding == "sin_cos_plus_learned":
            self.latent_pos_emb_res = nn.Parameter(torch.zeros(latent_slots, latent_dim))
        else:
            raise ValueError(f"Unknown latent_pos_encoding `{latent_pos_encoding}`")

    def make_tape_pos(
        self,
        tape_dim: int,
        tape_pos_encoding: str,
        time_scaling: float,
    ) -> None:
        if tape_pos_encoding in ["sin_cos", "sin_cos_plus_learned"]:
            self.register_buffer(
                "tape_pos_emb",
                create_2d_sin_cos_pos_emb(
                    n_rows=self._n_rows,
                    n_cols=self._n_cols,
                    dim=tape_dim,
                    normalization_max=time_scaling,
                ),
            )
        if tape_pos_encoding == "learned":
            self.tape_pos_emb = nn.Parameter(torch.zeros(self._n_rows * self._n_cols, tape_dim))
            nn.init.trunc_normal_(self.tape_pos_emb, std=0.02)
        elif tape_pos_encoding == "sin_cos_plus_learned":
            self.tape_pos_emb_res = nn.Parameter(torch.zeros(self._n_rows * self._n_cols, tape_dim))
        else:
            raise ValueError(f"Unknown tape_pos_encoding `{tape_pos_encoding}`")

    def initialize_cond(
        self,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        batch_size = int(offsets.shape[0] - 1)
        if t.ndim > 1:
            t = t.view(t.shape[0], -1).squeeze(-1)
        if t.shape[0] != batch_size:
            doc_starts = offsets[:-1].to(t.device)
            t = t[doc_starts]
        time_emb = self.time_emb(t, last_swish=False, normalize=True)
        time_emb = rearrange(time_emb, "b d -> b 1 d")

        if cond is not None:
            cond = self.cond_proj(cond)
            if cond.ndim == 2:
                cond = rearrange(cond, "b d -> b 1 d")
        return time_emb, cond

    def initialize_tape(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        pos_embs: torch.Tensor,
        offsets: torch.Tensor,
        offsets_pos_embs: torch.Tensor,
        document_ids: torch.Tensor,
        tape_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del offsets, offsets_pos_embs
        tape_r = None
        log_first_document("rin.tokens_in", tokens, document_ids)
        tape = self.stem(tokens)
        tape = self.stem_ln(tape) + pos_embs.to(tape.device)
        if self._self_cond in ["tape", "latent+tape"] and tape_prev is not None:
            tape = tape + self.tape_prev_ln(self.tape_prev_proj(tape_prev))
        log_first_document("rin.tape_projected", tape, document_ids)
        if self._cond_tape_writable and tape_r is not None:
            tape = _concat_tokens(tape, tape_r)
            tape_r = None
        return tape, tape_r

    def initialize_latent(
        self,
        batch_size: int,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.latent_pos_emb
        if self._latent_pos_encoding in ["sin_cos_plus_learned"]:
            latent = latent + self.latent_pos_emb_res
        latent = latent.repeat(batch_size, 1, 1)

        aux_tokens = None
        if self._time_on_latent and time_emb is not None:
            aux_tokens = time_emb if aux_tokens is None else torch.cat((aux_tokens, time_emb), dim=1)
        if self._cond_on_latent and cond is not None:
            aux_tokens = cond if aux_tokens is None else torch.cat((aux_tokens, cond), dim=1)

        tokens_per_sample = latent.shape[1]
        latent = latent.view(-1, self._latent_dim)
        if aux_tokens is not None:
            latent = _concat_tokens_interleave(latent, aux_tokens, tokens_per_sample)
            tokens_per_sample += aux_tokens.shape[1]

        latent_offsets = torch.arange(
            0,
            batch_size + 1,
            device=latent.device,
            dtype=torch.int64,
        ) * tokens_per_sample
        latent_document_ids = get_document_ids(latent_offsets)

        if self._self_cond in ["latent", "latent+tape"] and latent_prev is not None:
            latent = latent + self.latent_prev_ln(self.latent_prev_proj(latent_prev))

        return latent, latent_document_ids

    def compute(
        self,
        latent: torch.Tensor,
        tape: torch.Tensor,
        tape_r: torch.Tensor | None,
        offsets: torch.Tensor,
        offsets_pos_embs: torch.Tensor,
        document_ids: torch.Tensor,
        latent_document_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del offsets, offsets_pos_embs
        for i in range(len(self._num_layers)):
            if self._cond_decoupled_read and tape_r is not None:
                latent = self.read_cond_units[i](latent, tape_r, latent_document_ids, document_ids)
            latent = self.read_units[i](latent, tape, latent_document_ids, document_ids)
            latent = self.latent_processing_units[i](latent, latent_document_ids)
            tape = self.write_units[i](tape, latent, document_ids, latent_document_ids)
        return latent, tape

    def readout_tape(self, tape: torch.Tensor, document_ids: torch.Tensor) -> torch.Tensor:
        tokens = self.output_linear(self.output_ln(tape))
        log_first_document("rin.readout_tokens", tokens, document_ids)
        return tokens

    @property
    def latent_shape(self) -> list[int]:
        return [self._latent_slots, self._latent_dim]

    @property
    def tape_shape(self) -> list[int]:
        return [self._tape_slots, self._tape_dim]

    @property
    def tape_dim(self) -> int:
        return self._tape_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def patch_dim(self) -> int:
        return self._patch_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def image_shape(self) -> list[int]:
        return [self._image_channels, self._image_height, self._image_width]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def pass_dummy_data(self, num_classes: int | None = None) -> None:
        was_training = self.training
        self.eval()

        test_image_size = max(64, self._patch_size * 4)
        dummy_image = torch.zeros([1, self._image_channels, test_image_size, test_image_size], device=self.device)
        dummy_tokens = (
            dummy_image.unfold(2, self._patch_size, self._patch_size)
            .unfold(3, self._patch_size, self._patch_size)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(-1, self._patch_dim)
        )

        dummy_pos_embs = create_2d_sin_cos_pos_emb(
            n_rows=test_image_size // self._patch_size,
            n_cols=test_image_size // self._patch_size,
            dim=self._tape_dim,
        ).to(self.device)

        total_patches = dummy_tokens.shape[0]
        split = max(1, total_patches // 2)
        offsets = torch.tensor([0, split, total_patches], dtype=torch.int64, device=self.device)
        offsets_pos_embs = offsets.clone()
        document_ids = get_document_ids(offsets)

        dummy_label = None
        if num_classes is not None:
            dummy_label = torch.zeros([offsets.shape[0] - 1, num_classes], device=self.device)

        self(
            x=dummy_tokens,
            t=torch.zeros(offsets.shape[0] - 1, device=self.device),
            cond=dummy_label,
            pos_embs=dummy_pos_embs,
            offsets=offsets,
            offsets_pos_embs=offsets_pos_embs,
            document_ids=document_ids,
            latent_prev=None,
            tape_prev=None,
        )
        self.train(was_training)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        cond: torch.Tensor | None,
        pos_embs: torch.Tensor,
        offsets: torch.Tensor,
        offsets_pos_embs: torch.Tensor,
        document_ids: torch.Tensor,
        latent_prev: torch.Tensor | None,
        tape_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError("Input tokens must be a [total_tokens, patch_dim] tensor.")
        num_docs = int(offsets.shape[0] - 1)
        if isinstance(t, float):
            t_tensor = torch.full((num_docs,), t, device=x.device, dtype=torch.float32)
        elif torch.is_tensor(t):
            t_tensor = t.to(x.device)
        else:
            raise TypeError("t must be a float or tensor")

        time_emb, cond_tokens = self.initialize_cond(t_tensor, cond, offsets)
        tape, tape_r = self.initialize_tape(
            x,
            time_emb,
            cond_tokens,
            pos_embs,
            offsets,
            offsets_pos_embs,
            document_ids,
            tape_prev,
        )
        latent, latent_document_ids = self.initialize_latent(num_docs, time_emb, cond_tokens, latent_prev)
        latent, tape = self.compute(latent, tape, tape_r, offsets, offsets_pos_embs, document_ids, latent_document_ids)
        tokens = self.readout_tape(tape, document_ids)
        return tokens, latent, tape
