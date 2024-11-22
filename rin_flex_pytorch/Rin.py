import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F
import math
from typing import Optional

from .modules import MLP, LambdaModule, ScalarEmbedding, TransformerDecoderLayer, TransformerEncoder
from .utils.pos_embedding import create_2d_sin_cos_pos_emb


def _concat_tokens(*tokens: torch.Tensor | None) -> torch.Tensor:
    # tokens in shape [..., n, d]
    return torch.cat([t for t in tokens if t is not None], -2)


def create_attention_mask(seq_len: int, max_len: int, device: torch.device) -> torch.Tensor:
    """Creates 2D attention mask for padding tokens"""
    # Create mask of shape [max_len, max_len] where valid tokens are True and padding is False
    mask = torch.zeros((max_len, max_len), device=device, dtype=torch.bool)
    mask[:seq_len, :seq_len] = True
    return mask


def create_padding_mask(batch_size: int, seq_len: int, max_len: int, device: torch.device) -> torch.Tensor:
    """Creates padding mask for tokens beyond seq_len
    Args:
        batch_size: Batch size
        seq_len: Number of valid tokens
        max_len: Maximum sequence length
        device: Device to create tensor on
    Returns:
        Tensor of shape (batch_size, max_len) where False indicates valid tokens
        and True indicates padding tokens
    """
    mask = torch.ones((batch_size, max_len), device=device, dtype=torch.bool)
    mask[:, :seq_len] = False
    return mask


class RotaryPositionalEmbeddings2D(torch.nn.Module):
    def __init__(self, dim: int, max_height: int = 64, max_width: int = 64, base: int = 10000):
        super().__init__()
        
        # Ensure dim is divisible by 4 for pairs of sin/cos in both dimensions
        if dim % 4 != 0:
            raise ValueError(f"Dimension {dim} must be divisible by 4 for 2D rotary embeddings")
        
        # Split dimension for height and width components
        dim_per_direction = dim // 2
        
        # Create position indices for height and width
        pos_h = torch.arange(max_height).unsqueeze(1)
        pos_w = torch.arange(max_width).unsqueeze(1)
        
        # Create dimension indices
        div_term = torch.exp(
            torch.arange(0, dim_per_direction, 2) * (-math.log(base) / dim_per_direction)
        )
        
        # Calculate sin and cos terms for height
        pe_h = torch.zeros(max_height, dim_per_direction)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)
        
        # Calculate sin and cos terms for width
        pe_w = torch.zeros(max_width, dim_per_direction)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)
        
        # Register buffers
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
        
        self.max_height = max_height
        self.max_width = max_width
        
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [..., h*w, dim]
            h: Height of the feature map
            w: Width of the feature map
        Returns:
            Tensor of same shape with 2D positional embeddings added
        """
        # Get height embeddings for each row
        pos_emb_h = self.pe_h[:h]  # [h, dim//2]
        pos_emb_h = pos_emb_h.unsqueeze(1).repeat(1, w, 1)  # [h, w, dim//2]
        
        # Get width embeddings for each column
        pos_emb_w = self.pe_w[:w]  # [w, dim//2]
        pos_emb_w = pos_emb_w.unsqueeze(0).repeat(h, 1, 1)  # [h, w, dim//2]
        
        # Combine height and width embeddings
        pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=-1)  # [h, w, dim]
        pos_emb = pos_emb.reshape(h * w, -1)  # [h*w, dim]
        
        return pos_emb


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
        max_seq_len: int = 256,
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
        self._output_dim = patch_size**2 * image_channels

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

        self._max_seq_len = max_seq_len

    def make_latent_pos(
        self,
        latent_slots: int,
        latent_dim: int,
        latent_pos_encoding: str,
        time_scaling: float,
    ) -> None:
        if latent_pos_encoding == "rotary":
            # For latent, treat it as 1D sequence (latent_slots x 1)
            self.latent_pos_emb = RotaryPositionalEmbeddings2D(
                dim=latent_dim,
                max_height=latent_slots,
                max_width=1,
                base=time_scaling
            )
        elif latent_pos_encoding in ["sin_cos", "sin_cos_plus_learned"]:
            self.register_buffer(
                "latent_pos_emb",
                create_2d_sin_cos_pos_emb(
                    n_rows=latent_slots,
                    n_cols=1,
                    dim=latent_dim,
                    normalization_max=time_scaling,
                ),
            )
        elif latent_pos_encoding == "learned":
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
        if tape_pos_encoding == "rotary":
            self.tape_pos_emb = RotaryPositionalEmbeddings2D(
                dim=tape_dim,
                max_height=self._n_rows,
                max_width=self._n_cols,
                base=time_scaling
            )
        elif tape_pos_encoding in ["sin_cos", "sin_cos_plus_learned"]:
            self.register_buffer(
                "tape_pos_emb",
                create_2d_sin_cos_pos_emb(
                    n_rows=self._n_rows,
                    n_cols=self._n_cols,
                    dim=tape_dim,
                    normalization_max=time_scaling,
                ),
            )
        elif tape_pos_encoding == "learned":
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, Optional[torch.Tensor]]:
        tape_r = None
        if not self._time_on_latent and time_emb is not None:
            tape_r = time_emb
        if not self._cond_on_latent and cond is not None:
            tape_r = _concat_tokens(tape_r, cond)

        # Handle variable input sizes
        _, _, h, w = x.shape
        n_rows = h // self._patch_size
        n_cols = w // self._patch_size
        n_patches = n_rows * n_cols
        
        tape = self.stem(x)
        tape = rearrange(tape, "b d h w -> b (h w) d")
        
        # Get actual sequence length before padding
        seq_len = min(n_patches, self._max_seq_len)
        
        # Create position embeddings for actual sequence length first
        if self._tape_pos_encoding == "rotary":
            h_tokens = min(n_rows, int(np.sqrt(seq_len)))
            w_tokens = min(n_cols, seq_len // h_tokens)
            pos_emb = self.tape_pos_emb(
                torch.zeros_like(tape[:, :seq_len]), 
                h=h_tokens,
                w=w_tokens
            )
        else:
            pos_emb = self.tape_pos_emb[:seq_len]
            if self._tape_pos_encoding in ["sin_cos_plus_learned"]:
                pos_emb = pos_emb + self.tape_pos_emb_res[:seq_len]
        
        # Pad both tape and position embeddings to max_seq_len
        if tape.size(1) > self._max_seq_len:
            tape = tape[:, :self._max_seq_len]
            pos_emb = pos_emb[:self._max_seq_len]
        elif tape.size(1) < self._max_seq_len:
            tape = F.pad(tape, (0, 0, 0, self._max_seq_len - tape.size(1)))
            pos_emb = F.pad(pos_emb, (0, 0, 0, self._max_seq_len - pos_emb.size(0)))
        
        # Add batch dimension to position embeddings
        tape_pos_emb = rearrange(pos_emb, "n d -> 1 n d")
        
        # Calculate total sequence length including conditioning tokens
        cond_len = 0
        if time_emb is not None and not self._time_on_latent:
            cond_len += 1
        if cond is not None and not self._cond_on_latent:
            cond_len += 1
        total_seq_len = self._max_seq_len + cond_len
        
        # Create padding mask
        batch_size = x.shape[0]
        padding_mask = torch.ones((batch_size, total_seq_len), device=x.device, dtype=torch.bool)
        padding_mask[:, :seq_len] = False  # Valid tokens in tape
        if cond_len > 0:
            padding_mask[:, self._max_seq_len:] = False  # Valid conditioning tokens
        
        # Apply layer norm and add position embeddings
        tape = self.stem_ln(tape) + tape_pos_emb

        if self._self_cond in ["tape", "latent+tape"] and tape_prev is not None:
            tape = tape + self.tape_prev_ln(self.tape_prev_proj(tape_prev))
        if self._cond_tape_writable and tape_r is not None:
            tape, tape_r = _concat_tokens(tape, tape_r), None

        return tape, tape_r, padding_mask

    def initialize_latent(
        self,
        batch_size: int,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._latent_pos_encoding == "rotary":
            latent = torch.zeros(batch_size, self._latent_slots, self._latent_dim, device=self.device)
            latent_pos_emb = self.latent_pos_emb(
                torch.zeros_like(latent), 
                h=self._latent_slots, 
                w=1
            )
            latent = latent + latent_pos_emb
        else:
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
                latent = self.read_cond_units[i](latent, tape_r, attention_mask=attention_mask)
                latent = self.read_units[i](latent, tape, attention_mask=attention_mask)
            else:
                # First get the merged tape
                tape_merged = _concat_tokens(tape, tape_r)
                
                # Calculate the total sequence length after concatenation
                total_seq_len = tape_merged.size(1)
                
                # Create a new padding mask for the concatenated sequence
                if attention_mask is not None:
                    batch_size = attention_mask.size(0)
                    new_mask = torch.ones((batch_size, total_seq_len), device=attention_mask.device, dtype=torch.bool)
                    # Copy the original mask for the tape portion
                    new_mask[:, :self._max_seq_len] = attention_mask[:, :self._max_seq_len]
                    # Set False (valid tokens) for the tape_r portion if it exists
                    if tape_r is not None:
                        new_mask[:, self._max_seq_len:] = False
                    attention_mask_merged = new_mask
                else:
                    attention_mask_merged = None
                
                latent = self.read_units[i](latent, tape_merged, attention_mask=attention_mask_merged)
                
            latent = self.latent_processing_units[i](latent)
            # For writing, use only the original tape portion of the mask
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
        
        # Store input shape for readout
        self._input_shape = x.shape

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
        latent, tape = self.compute(latent, tape, tape_r, attention_mask)
        x = self.readout_tape(tape)
        return x, latent, tape[:, :self._tape_slots]

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
