import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import torch

from rin_pytorch import Rin


def _load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def _extract_configs_from_ckpt(ckpt: dict) -> Tuple[dict | None, dict | None]:
    """Return (rin_config, diffusion_config) if present in Lightning hyper_parameters."""
    rin_cfg = None
    diff_cfg = None
    if isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
        hp = ckpt["hyper_parameters"]
        rin_cfg = hp.get("rin")
        diff_cfg = hp.get("diffusion")
    return rin_cfg, diff_cfg


def _extract_rin_state_dict_from_ckpt(ckpt: dict) -> Dict[str, torch.Tensor]:
    """
    Normalize a variety of checkpoint layouts to a Rin-compatible state_dict.

    Handles these cases:
    - Lightning 'state_dict' with keys prefixed by 'ema_diffusion_model.' or 'diffusion_model.'
      → extracts the '...denoiser.' sub-tree and strips that prefix for Rin.
    - 'ema_model' (saved via on_save_checkpoint) containing a diffusion state-dict
      → extracts 'denoiser.' keys and strips prefix.
    - Direct state-dict (either diffusion or Rin) → auto-detects and maps.
    """
    def _strip_prefix(d: dict, prefix: str) -> dict:
        out = {}
        for k, v in d.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    # 1) Lightning 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        # Prefer EMA weights if present
        ema_den = _strip_prefix(sd, "ema_diffusion_model.denoiser.")
        if ema_den:
            return ema_den
        # Fallback to main model
        main_den = _strip_prefix(sd, "diffusion_model.denoiser.")
        if main_den:
            return main_den
        # Last resort: maybe entire Rin was saved directly in state_dict
        rin_direct = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
        return rin_direct

    # 2) EMA model nested
    if isinstance(ckpt, dict) and "ema_model" in ckpt:
        sd = ckpt["ema_model"]
        den = _strip_prefix(sd, "denoiser.")
        if den:
            return den
        return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    # 3) Assume direct state-dict
    if isinstance(ckpt, dict):
        # Try diffusion → Rin mapping
        den = _strip_prefix(ckpt, "denoiser.")
        if den:
            return den
        # Maybe it is already Rin
        return {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    raise ValueError("Unsupported checkpoint format")


def _build_rin_from_config(rin_cfg: dict) -> Rin:
    # Create the model using rin_config; ensure required fields are present
    model = Rin(**rin_cfg)
    # Initialize lazy params with dummy data
    model.pass_dummy_data(num_classes=rin_cfg.get("num_classes"))
    return model


def _make_synthetic_ragged_input(
    rin: Rin,
    *,
    num_docs: int = 3,
    image_height: int | None = None,
    image_width: int | None = None,
    num_classes: int | None = None,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct a deterministic ragged-token batch for Rin.forward.

    Returns:
      x, t, cond, pos_embs, offsets, offsets_pos_embs, document_ids
    """
    from rin_pytorch.Rin import patchify  # reuse helper
    from rin_pytorch.utils.pos_embedding import create_2d_sin_cos_pos_emb
    from rin_pytorch.utils.ragged_tensor import ragged_list_to_tensor, get_document_ids

    device = rin.device if device is None else device

    c = rin.image_shape[0]
    p = rin._patch_size
    H = image_height if image_height is not None else max(64, p * 4)
    W = image_width if image_width is not None else max(64, p * 4)

    # Create per-document images of varying sizes (to test ragged handling)
    hs = [H, max(p * 3, H - p), max(p * 2, H - 2 * p)]
    ws = [W, max(p * 3, W - p), max(p * 2, W - 2 * p)]
    hs = hs[:num_docs]
    ws = ws[:num_docs]

    x_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []

    for h, w in zip(hs, ws):
        # Make size divisible by patch
        h = h - (h % p)
        w = w - (w % p)
        img = torch.zeros((1, c, h, w), device=device)  # zeros for determinism
        tokens = patchify(img, p).squeeze(0)  # [T, D]
        pos = create_2d_sin_cos_pos_emb(h // p, w // p, rin.tape_dim).to(device)
        x_list.append(tokens)
        pos_list.append(pos)

    x, offsets = ragged_list_to_tensor(x_list)  # [sumT, D]
    pos_embs, offsets_pos = ragged_list_to_tensor(pos_list)
    doc_ids = get_document_ids(offsets).to(device)

    # Conditioning – one class per doc
    bsz = len(hs)
    n_classes = num_classes if num_classes is not None else rin._cond_dim
    if rin._cond_on_latent:
        # Use class-conditioning as in training
        if n_classes is None or isinstance(n_classes, bool):
            raise ValueError("num_classes must be provided for conditional model")
        labels = torch.arange(bsz, device=device) % int(n_classes)
        cond = torch.nn.functional.one_hot(labels, int(n_classes)).float()
    else:
        cond = None

    # Scalar time
    t = torch.tensor(0.0, device=device)

    return x, t, cond, pos_embs, offsets, offsets_pos, doc_ids


@torch.no_grad()
def _collect_activations(rin: Rin, inputs: Tuple[Any, ...]) -> Dict[str, torch.Tensor]:
    """Register forward hooks on key blocks and collect their outputs (detached CPU)."""
    hook_handles = []
    activations: Dict[str, torch.Tensor] = {}

    def _save(name: str):
        def _hook(module, inp, out):
            # Some layers return tuples (e.g., TransformerEncoderLayer)
            tensor = out[0] if isinstance(out, tuple) else out
            if isinstance(tensor, torch.Tensor):
                activations[name] = tensor.detach().to("cpu")
        return _hook

    # Register on read/write/encoder blocks and a few key linears
    for name, module in rin.named_modules():
        if any(name.startswith(prefix) for prefix in [
            "read_units.", "write_units.", "latent_processing_units.",
        ]):
            hook_handles.append(module.register_forward_hook(_save(name)))
        elif name in {"stem", "output_linear"}:
            hook_handles.append(module.register_forward_hook(_save(name)))

    # Run the forward pass
    rin.eval()
    _ = rin(*inputs)

    # Cleanup hooks
    for h in hook_handles:
        h.remove()

    return activations


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a_flat = a.reshape(-1).to(torch.float32)
    b_flat = b.reshape(-1).to(torch.float32)
    n = min(a_flat.numel(), b_flat.numel())
    a_flat = a_flat[:n]
    b_flat = b_flat[:n]
    diff = (a_flat - b_flat).abs()
    mean_abs = float(diff.mean().item())
    max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
    # Cosine similarity can be NaN if vectors are all-zero; handle safely
    denom = (a_flat.norm() * b_flat.norm()).item()
    cosine = float((torch.dot(a_flat, b_flat) / denom).item()) if denom > 0 else float('nan')
    return {"mean_abs": mean_abs, "max_abs": max_abs, "cosine": cosine}


def main():
    parser = argparse.ArgumentParser(description="Compare intermediate activations between two checkpoints (Rin)")
    parser.add_argument("ckpt_a", type=str, help="Path to checkpoint A (e.g., FlexAttention)")
    parser.add_argument("ckpt_b", type=str, help="Path to checkpoint B (e.g., baseline SDPA)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--num_docs", type=int, default=3)
    parser.add_argument("--topk", type=int, default=15, help="Top-K layers by mean |Δ| to print")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save full per-layer stats as JSON")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoints and configs
    ckpt_a = _load_checkpoint(args.ckpt_a)
    ckpt_b = _load_checkpoint(args.ckpt_b)

    rin_cfg_a, _ = _extract_configs_from_ckpt(ckpt_a)
    rin_cfg_b, _ = _extract_configs_from_ckpt(ckpt_b)

    if rin_cfg_a is None and rin_cfg_b is None:
        raise RuntimeError("Could not find Rin config in either checkpoint hyper_parameters.")

    # Prefer config from A if both present; otherwise take whichever is available
    rin_cfg = rin_cfg_a if rin_cfg_a is not None else rin_cfg_b

    # Build two Rin models with the SAME architecture to ensure hook names align
    rin_a = _build_rin_from_config(rin_cfg).to(device)
    rin_b = _build_rin_from_config(rin_cfg).to(device)

    # Load state dicts (Rin-only) from both checkpoints
    sd_a = _extract_rin_state_dict_from_ckpt(ckpt_a)
    sd_b = _extract_rin_state_dict_from_ckpt(ckpt_b)

    missing_a = rin_a.load_state_dict(sd_a, strict=False)
    missing_b = rin_b.load_state_dict(sd_b, strict=False)
    if missing_a.missing_keys or missing_b.missing_keys:
        print("Warning: Some keys were missing when loading (strict=False)")
        print(f"A missing: {missing_a.missing_keys}")
        print(f"B missing: {missing_b.missing_keys}")

    rin_a.eval()
    rin_b.eval()

    # Build the same synthetic input once and feed to both models
    x, t, cond, pos_embs, offsets, offsets_pos, doc_ids = _make_synthetic_ragged_input(
        rin_a,
        num_docs=args.num_docs,
        image_height=args.image_height,
        image_width=args.image_width,
        num_classes=rin_cfg.get("num_classes"),
        device=device,
    )

    inputs: Tuple[Any, ...] = (x, t, cond, pos_embs, offsets, offsets_pos, doc_ids, None, None)

    # Collect activations
    acts_a = _collect_activations(rin_a, inputs)
    acts_b = _collect_activations(rin_b, inputs)

    # Intersect layers
    common = sorted(set(acts_a.keys()) & set(acts_b.keys()))
    if not common:
        raise RuntimeError("No common activation points found. Check hook registration.")

    # Compute stats
    stats = {}
    for name in common:
        stats[name] = _compare_tensors(acts_a[name], acts_b[name])

    # Rank by mean |Δ|
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["mean_abs"], reverse=True)

    print("\nPer-layer activation differences (top-K):")
    for name, s in ranked[: args.topk]:
        print(f"  {name:40s}  mean|Δ|={s['mean_abs']:.6e}  max|Δ|={s['max_abs']:.6e}  cos={s['cosine']:.6f}")

    # Optionally save full JSON
    if args.save_json:
        out = {
            "ckpt_a": str(args.ckpt_a),
            "ckpt_b": str(args.ckpt_b),
            "device": str(device),
            "stats": stats,
        }
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"Saved full stats → {args.save_json}")


if __name__ == "__main__":
    main()


