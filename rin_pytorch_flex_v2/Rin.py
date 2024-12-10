import numpy as np
import torch
from einops import rearrange
from typing import Optional

from .modules import MLP, LambdaModule, ScalarEmbedding, TransformerDecoderLayer, TransformerEncoder
from .utils.pos_embedding import create_2d_sin_cos_pos_emb
from .utils.mask_utils import create_padding_mask


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
        max_seq_len=256
    ):
        super().__init__()

        self._image_height = image_height
        self._image_width = image_width
        self._image_channels = image_channels
        self._n_rows = image_height // patch_size
        self._n_cols = image_width // patch_size
        self._num_tokens = self._n_rows * self._n_cols
        self._patch_size = patch_size
        self._output_dim = patch_size**2 * image_channels
        self._max_seq_len = max_seq_len

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

        self.stem = torch.nn.Conv2d(
            in_channels=image_channels,
            out_channels=tape_dim, 
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True
        )

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
        x: torch.Tensor,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        tape_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Initialize tape with proper padding and attention mask
        Returns:
            tuple of (tape, conditional_tape, attention_mask)
        """
        # Convert image to patches
        tape = self.stem(x)
        b, c, h, w = tape.shape
        tape = rearrange(tape, 'b c h w -> b (h w) c')
        
        # Get actual sequence length
        seq_len = tape.size(1)
        
        # Create attention mask (False = attend, True = ignore)
        attention_mask = create_padding_mask(
            batch_size=b,
            seq_len=seq_len,
            max_len=self._max_seq_len,
            device=x.device
        )
        
        # Add positional embeddings before padding
        if self._tape_pos_encoding == "learned":
            tape = tape + self.tape_pos_emb[:seq_len]
        elif self._tape_pos_encoding in ["sin_cos", "sin_cos_plus_learned"]:
            tape = tape + self.tape_pos_emb[:seq_len]
            if self._tape_pos_encoding == "sin_cos_plus_learned":
                tape = tape + self.tape_pos_emb_res[:seq_len]
        
        # Pad tape to max sequence length
        if seq_len < self._max_seq_len:
            tape = F.pad(tape, (0, 0, 0, self._max_seq_len - seq_len))
        
        # Apply layer norm
        tape = self.stem_ln(tape)
        
        # Handle conditional tape
        tape_r = None
        if time_emb is not None and not self._time_on_latent:
            tape_r = time_emb
        if cond is not None and not self._cond_on_latent:
            tape_r = _concat_tokens(tape_r, cond)
        
        # Handle self-conditioning
        if self._self_cond in ["tape", "latent+tape"] and tape_prev is not None:
            tape = tape + self.tape_prev_ln(self.tape_prev_proj(tape_prev))
        
        return tape, tape_r, attention_mask

    def initialize_latent(
        self,
        batch_size: int,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        latent = self.latent_pos_emb
        if self._latent_pos_encoding in ["sin_cos_plus_learned"]:
            latent = latent + self.latent_pos_emb_res
        latent = latent.repeat(batch_size, 1, 1)
        if self._time_on_latent and time_emb is not None:
            latent = _concat_tokens(latent, time_emb)
        if self._cond_on_latent and cond is not None:
            latent = _concat_tokens(latent, cond)
        if self._self_cond in ["latent", "latent+tape"] and latent_prev is not None:
            latent = latent + self.latent_prev_ln(self.latent_prev_proj(latent_prev))
        return latent

    def compute(
        self,
        latent: torch.Tensor,
        tape: torch.Tensor,
        tape_r: torch.Tensor | None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(len(self._num_layers)):
            if self._cond_decoupled_read:
                # For conditional reads, no masking needed
                latent = self.read_cond_units[i](latent, tape_r)
                # For main reads, apply mask
                latent = self.read_units[i](latent, tape, attention_mask=attention_mask)
            else:
                # Combined reads
                tape_merged = _concat_tokens(tape, tape_r)
                # Create merged attention mask
                if attention_mask is not None:
                    batch_size = attention_mask.size(0)
                    total_seq_len = tape_merged.size(1)
                    merged_mask = torch.ones((batch_size, total_seq_len), device=attention_mask.device, dtype=torch.bool)
                    # Copy original mask for tape portion
                    merged_mask[:, :self._max_seq_len] = attention_mask
                    # Set False for conditional tokens
                    if tape_r is not None:
                        merged_mask[:, self._max_seq_len:] = False
                    attention_mask_merged = merged_mask
                else:
                    attention_mask_merged = None
                    
                latent = self.read_units[i](latent, tape_merged, attention_mask=attention_mask_merged)
                
            # No masking needed for latent processing
            latent = self.latent_processing_units[i](latent)
            
            # Apply mask for writing back to tape
            tape = self.write_units[i](
                tape[:, :self._max_seq_len],
                latent,
                attention_mask=attention_mask[:, :self._max_seq_len] if attention_mask is not None else None
            )
        return latent, tape

    def readout_tape(self, tape: torch.Tensor) -> torch.Tensor:
        """Convert tape tokens back to image space"""
        # Get the actual dimensions from the input
        bs = tape.size(0)
        n_tokens = min(self._num_tokens, self._max_seq_len)
        
        tokens = self.output_linear(self.output_ln(tape[:, :n_tokens]))
        
        # Get the actual input dimensions
        _, _, h, w = self._input_shape
        n_rows = h // self._patch_size
        n_cols = w // self._patch_size
        
        # Ensure we have enough tokens
        expected_tokens = n_rows * n_cols
        if tokens.size(1) < expected_tokens:
            tokens = F.pad(tokens, (0, 0, 0, expected_tokens - tokens.size(1)))
        else:
            tokens = tokens[:, :expected_tokens]
        
        # Reshape to match input dimensions
        tokens = rearrange(
            tokens,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=n_rows,
            w=n_cols,
            p1=self._patch_size,
            p2=self._patch_size,
        )
        
        return tokens

    @property
    def latent_shape(self) -> list[int]:
        return [self._latent_slots, self._latent_dim]

    @property
    def tape_shape(self) -> list[int]:
        return [self._tape_slots, self._tape_dim]

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

        self(
            x=torch.zeros([1, *self.image_shape], device=self.device),
            t=0.0,
            cond=None if num_classes is None else torch.zeros([1, num_classes], device=self.device),
        )

        self.train(was_training)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        cond: torch.Tensor | None = None,
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 4
        bs = x.shape[0]

        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((bs,), t, device=x.device, dtype=torch.float32)

        if latent_prev is None:
            latent_prev = torch.zeros(bs, *self.latent_shape, device=x.device)

        if tape_prev is None:
            tape_prev = torch.zeros(bs, *self.tape_shape, device=x.device)

        if self._cond_on_latent and cond is None:
            raise ValueError("cond is None but cond_on_latent is True")

        time_emb, cond = self.initialize_cond(t, cond)
        tape, tape_r, attention_mask = self.initialize_tape(x, time_emb, cond, tape_prev)
        latent = self.initialize_latent(bs, time_emb, cond, latent_prev)
        latent, tape = self.compute(latent, tape, tape_r)
        x = self.readout_tape(tape)
        return x, latent, tape[:, : self._tape_slots]

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
