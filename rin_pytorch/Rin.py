import numpy as np
import torch
from einops import rearrange

from .modules import MLP, LambdaModule, ScalarEmbedding, TransformerDecoderLayer, TransformerEncoder
from .utils.pos_embedding import create_2d_sin_cos_pos_emb


def _concat_tokens(*tokens: torch.Tensor | None) -> torch.Tensor:
    # tokens in shape [..., n, d]
    return torch.cat([t for t in tokens if t is not None], -2)


class Rin(torch.nn.Module):
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
    ):
        super().__init__()

        self._image_height = image_height
        self._image_width = image_width
        self._image_channels = image_channels
        self._n_rows = image_height // patch_size
        self._n_cols = image_width // patch_size
        self._num_tokens = self._n_rows * self._n_cols
        self._patch_size = patch_size
        self._patch_dim = patch_size**2 * image_channels
        self._output_dim = self._patch_dim

        self._num_layers = [int(i) for i in num_layers.split(",")]
        self._latent_slots = latent_slots
        self._time_on_latent = time_on_latent
        self._cond_on_latent = cond_on_latent_n > 0
        if self._time_on_latent:  # replace 1 latent with time emb.
            latent_slots -= 1
        latent_slots -= cond_on_latent_n
        self._latent_dim = latent_dim
        self._tape_slots = self._num_tokens
        self._tape_dim = tape_dim
        self._cond_dim = cond_dim = cond_dim if cond_dim > 0 else tape_dim
        self._latent_pos_encoding = latent_pos_encoding
        self._tape_pos_encoding = tape_pos_encoding
        assert self_cond in ["none", "latent", "latent+tape", "tape"]
        self._self_cond = self_cond
        self._cond_tape_writable = cond_tape_writable
        self._cond_decoupled_read = cond_decoupled_read
        self.token_proj = torch.nn.Linear(self._patch_dim, tape_dim)
        self.stem_ln = torch.nn.LayerNorm(tape_dim, eps=1e-6)
        self.time_emb = ScalarEmbedding(
            dim=(latent_dim if self._time_on_latent else cond_dim) // 4,
            scaling=time_scaling,
            expansion=4,
        )
        if cond_proj:
            if num_classes is None:
                raise ValueError("num_classes must be provided when cond_proj=True")
            self.cond_proj = torch.nn.Linear(num_classes, latent_dim if self._cond_on_latent else cond_dim)
        else:
            self.cond_proj = torch.nn.Identity()

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
            self.latent_prev_ln = torch.nn.LayerNorm(latent_dim, eps=1e-6)
            torch.nn.init.zeros_(self.latent_prev_ln.weight)

        if self_cond in ["tape", "latent+tape"]:
            self.tape_prev_proj = MLP(
                num_layers=1,
                dim=tape_dim,
                mlp_ratio=tape_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.tape_prev_ln = torch.nn.LayerNorm(tape_dim, eps=1e-6)
            torch.nn.init.zeros_(self.tape_prev_ln.weight)

        self.read_units = torch.nn.ModuleList()
        self.read_cond_units = torch.nn.ModuleList()
        self.write_units = torch.nn.ModuleList()
        self.latent_processing_units = torch.nn.ModuleList()

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
                        dim_x_att=cond_dim,
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
                        use_mlp=True if tape_mlp_ratio > 0 else False,
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

        self.output_ln = torch.nn.LayerNorm(tape_dim, eps=1e-6)
        self.output_linear = torch.nn.Linear(tape_dim, self._output_dim)

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
            self.latent_pos_emb = torch.nn.Parameter(torch.zeros(latent_slots, latent_dim))
            torch.nn.init.trunc_normal_(self.latent_pos_emb, std=0.02)
        elif latent_pos_encoding == "sin_cos_plus_learned":
            self.latent_pos_emb_res = torch.nn.Parameter(torch.zeros(latent_slots, latent_dim))
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
            self.tape_pos_emb = torch.nn.Parameter(torch.zeros(self._n_rows * self._n_cols, tape_dim))
            torch.nn.init.trunc_normal_(self.tape_pos_emb, std=0.02)
        elif tape_pos_encoding == "sin_cos_plus_learned":
            self.tape_pos_emb_res = torch.nn.Parameter(torch.zeros(self._n_rows * self._n_cols, tape_dim))
        else:
            raise ValueError(f"Unknown tape_pos_encoding `{tape_pos_encoding}`")

    def _get_base_tape_pos(self, length: int) -> torch.Tensor:
        tape_pos = rearrange(self.tape_pos_emb, "n d -> 1 n d")
        if self._tape_pos_encoding in ["sin_cos_plus_learned"]:
            tape_pos = tape_pos + rearrange(self.tape_pos_emb_res, "n d -> 1 n d")
        if tape_pos.size(1) < length:
            raise ValueError("Base positional embeddings shorter than requested length.")
        return tape_pos[:, :length]

    def initialize_cond(
        self,
        t: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if t is not None:
            t = self.time_emb(t, last_swish=False, normalize=True)
            t = rearrange(t, "b d -> b 1 d")
        if cond is not None:
            cond = self.cond_proj(cond)
            if cond.ndim == 2:
                cond = rearrange(cond, "b d -> b 1 d")
        return t, cond

    def initialize_tape(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        tape_prev: torch.Tensor | None,
        external_tape_pos: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tape_r = None
        # NOTE: With current config (time_on_latent=true, cond_on_latent_n=1),
        # these conditions are false, so tape_r stays None
        # if not self._time_on_latent and time_emb is not None:
        #     tape_r = time_emb
        # if not self._cond_on_latent and cond is not None:
        #     tape_r = _concat_tokens(tape_r, cond)

        tape = self.token_proj(tokens)
        tape_pos_emb = external_tape_pos
        tape_pos_emb = tape_pos_emb.to(tape.device)
        tape = self.stem_ln(tape) + tape_pos_emb

        # NOTE: With current config (self_cond="latent"), this is not hit
        # if self._self_cond in ["tape", "latent+tape"] and tape_prev is not None:
        #     tape = tape + self.tape_prev_ln(self.tape_prev_proj(tape_prev))
        # if self._cond_tape_writable and tape_r is not None:
        #     tape, tape_r = _concat_tokens(tape, tape_r), None

        return tape, tape_r

    def initialize_latent(
        self,
        num_docs: int,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize latent as a packed sequence.
        
        Args:
            num_docs: Number of documents in batch.
            time_emb: Time embeddings [num_docs, 1, latent_dim] or None.
            cond: Condition embeddings [num_docs, 1, latent_dim] or None.
            latent_prev: Previous latent, packed [num_docs * latent_slots, latent_dim] or None.
            
        Returns:
            latent: Packed latent sequence [total_latent_tokens, latent_dim]
            latent_doc_ids: Document ID per latent token [total_latent_tokens]
            latent_offsets: Cumulative offsets [num_docs + 1]
        """
        device = self.latent_pos_emb.device
        
        # Base latent positional embedding [latent_slots, latent_dim]
        latent = self.latent_pos_emb
        if self._latent_pos_encoding in ["sin_cos_plus_learned"]:
            latent = latent + self.latent_pos_emb_res
        
        # Expand for each document: [num_docs, latent_slots, latent_dim]
        latent = latent.unsqueeze(0).expand(num_docs, -1, -1).clone()
        
        # print(latent.shape)
        # print(latent_prev.shape)

        # Concat time_emb and cond if needed (they're [num_docs, 1, latent_dim])
        if self._time_on_latent and time_emb is not None:
            # print("time_emb.shape", time_emb.shape)
            # print("latent.shape", latent.shape)
            latent = _concat_tokens(latent, time_emb)
        if self._cond_on_latent and cond is not None:
            # print("cond.shape", cond.shape)
            # print("latent.shape", latent.shape)
            latent = _concat_tokens(latent, cond)
        
        # Add latent_prev if provided (reshape from packed to batched first)
        if self._self_cond in ["latent", "latent+tape"] and latent_prev is not None:
            # latent_prev is packed [num_docs * latent_slots, latent_dim], reshape to batched
            latent_prev_batched = latent_prev.view(num_docs, self._latent_slots, self._latent_dim)
            latent = latent + self.latent_prev_ln(self.latent_prev_proj(latent_prev_batched))
        
        # Get slots per document (may include time/cond tokens)
        slots_per_doc = latent.shape[1]
        
        # Reshape to packed sequence: [num_docs * slots_per_doc, latent_dim]
        latent = latent.reshape(-1, self._latent_dim)
        
        # Create doc_ids: [0,0,...,1,1,...,2,2,...]
        latent_doc_ids = torch.arange(num_docs, device=device).repeat_interleave(slots_per_doc)
        
        # Create offsets: [0, slots, 2*slots, ..., num_docs*slots]
        latent_offsets = torch.arange(num_docs + 1, device=device) * slots_per_doc

        return latent, latent_doc_ids, latent_offsets

    def compute(
        self,
        latent: torch.Tensor,
        tape: torch.Tensor,
        tape_r: torch.Tensor | None,
        # tape_key_padding_mask: torch.Tensor | None = None,
        tape_document_ids: torch.Tensor | None = None,
        latent_document_ids: torch.Tensor | None = None,
        tape_offsets: torch.Tensor | None = None,
        latent_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PRE-CREATE ALL MASKS ONCE to avoid repeated mask creation overhead
        # This is critical for performance with xformers
        from .modules.TransformerEncoderLayer import create_document_block_mask
        from .modules.TransformerDecoderLayer import create_cross_document_block_mask
        
        # Create latent self-attention mask (used in latent_processing_units)
        latent_self_mask = create_document_block_mask(latent_document_ids, offsets=latent_offsets)
        
        # Create cross-attention masks
        # Read: latent queries tape (latent_to_tape)
        read_cross_mask = create_cross_document_block_mask(
            latent_document_ids, tape_document_ids,
            q_offsets=latent_offsets, kv_offsets=tape_offsets
        )
        
        # Write: tape queries latent (tape_to_latent)
        write_cross_mask = create_cross_document_block_mask(
            tape_document_ids, latent_document_ids,
            q_offsets=tape_offsets, kv_offsets=latent_offsets
        )
        
        # Tape self-attention mask (for write units self-attention)
        tape_self_mask = create_document_block_mask(tape_document_ids, offsets=tape_offsets)

        for i in range(len(self._num_layers)):
            if self._cond_decoupled_read:
                latent = self.read_cond_units[i](latent, tape_r, enc_key_padding_mask=None)
                latent = self.read_units[i](
                    latent,
                    tape,
                    cross_attn_mask=read_cross_mask,
                    self_attn_mask=latent_self_mask,
                )
            else:
                tape_merged = _concat_tokens(tape, tape_r)
                latent = self.read_units[i](
                    latent,
                    tape_merged,
                    cross_attn_mask=read_cross_mask,
                    self_attn_mask=latent_self_mask,
                )
            latent = self.latent_processing_units[i](
                latent, 
                self_attn_mask=latent_self_mask,
            )
            tape = self.write_units[i](
                tape, 
                latent, 
                cross_attn_mask=write_cross_mask,
                self_attn_mask=tape_self_mask,
            )
        return latent, tape

    def readout_tape(self, tape: torch.Tensor) -> torch.Tensor:
        tokens = self.output_linear(self.output_ln(tape))
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
    def image_shape(self) -> list[int]:
        return [self._image_channels, self._image_height, self._image_width]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def pass_dummy_data(self, num_classes: int | None = None) -> None:
        # pass dummy data to initialize weights
        was_training = self.training
        self.eval()

        num_docs = 1
        seq_len = self._num_tokens
        
        # Packed format: [num_docs * seq_len, patch_dim]
        dummy_tokens = torch.zeros([num_docs * seq_len, self._patch_dim], device=self.device)
        
        # doc_ids and offsets for packed sequence
        doc_ids = torch.zeros(num_docs * seq_len, dtype=torch.long, device=self.device)
        offsets = torch.tensor([0, seq_len], dtype=torch.long, device=self.device)
        
        # Packed positional embeddings: [num_docs * seq_len, tape_dim]
        tape_pos = self._get_base_tape_pos(seq_len).squeeze(0)  # [seq_len, tape_dim]
        
        self(
            x=dummy_tokens,
            t=0.0,
            cond=None if num_classes is None else torch.zeros([num_docs, num_classes], device=self.device),
            tape_pos_emb=tape_pos,
            offsets=offsets,
            doc_ids=doc_ids,
        )

        self.train(was_training)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        cond: torch.Tensor | None = None,
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
        # tape_padding_mask: torch.Tensor | None = None,
        tape_pos_emb: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # if x.ndim != 3:
        #     raise ValueError("Input tokens must be a B x T x D tensor.")
        # bs = x.shape[0]
        # seq_len = x.shape[1]
        num_docs = int(offsets.shape[0] - 1)
        seq_len = x.shape[1]
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((num_docs,), t, device=x.device, dtype=torch.float32)

        if latent_prev is None:
            latent_prev = torch.zeros(num_docs * self._latent_slots, self._latent_dim, device=x.device)

        # if tape_prev is None:
        #     # tape_prev = torch.zeros(bs, seq_len, self._tape_dim, device=x.device)
        #     tape_prev = torch.zeros(num_docs, seq_len, self._tape_dim, device=x.device)

        if self._cond_on_latent and cond is None:
            raise ValueError("cond is None but cond_on_latent is True")

        time_emb, cond = self.initialize_cond(t, cond)
        # print("time_emb.shape_init", time_emb.shape)
        # print("cond.shape", cond.shape)
        tape_pos_emb = tape_pos_emb.to(x.device) if tape_pos_emb is not None else None
        tape, tape_r = self.initialize_tape(x, time_emb, cond, tape_prev, tape_pos_emb, offsets, doc_ids)
        latent, latent_doc_ids, latent_offsets = self.initialize_latent(num_docs, time_emb, cond, latent_prev)
        latent, tape = self.compute(
            latent, tape, tape_r, 
            tape_document_ids=doc_ids, 
            latent_document_ids=latent_doc_ids,
            tape_offsets=offsets,
            latent_offsets=latent_offsets,
        )
        x = self.readout_tape(tape)
        return x, latent, tape

    def load_weights_numpy(self, np_file):
        # load weights from numpy file relying on the order of parameters
        weights_np = list(np.load(np_file, allow_pickle=True).item().values())
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        for weight_np, param in zip(weights_np, trainable_params):
            data = torch.from_numpy(weight_np).to(param.device)
            param.data.copy_(data)

    def save_weights_numpy(self, np_file):
        # save weights to numpy file relying on the order of parameters
        weights_np = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                weights_np[name] = param.detach().cpu().numpy()

        np.save(np_file, weights_np)
