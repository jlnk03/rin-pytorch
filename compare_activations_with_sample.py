import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import torch

from rin_pytorch import RinDiffusionModel
from sample import load_config, create_model_from_config


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    return ckpt


def _override_config_with_ckpt_hparams(config: dict, ckpt: dict) -> dict:
    if 'hyper_parameters' in ckpt:
        hyper_params = ckpt['hyper_parameters']
        if 'rin' in hyper_params:
            for key, value in hyper_params['rin'].items():
                config[key] = value
        if 'diffusion' in hyper_params:
            for key, value in hyper_params['diffusion'].items():
                config[key] = value
    return config


def _load_into_diffusion_model_from_ckpt(model: RinDiffusionModel, ckpt: dict) -> None:
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        ema_state_dict = {}
        for k, v in sd.items():
            if k.startswith('ema_diffusion_model.'):
                new_key = k[len('ema_diffusion_model.') :]
                ema_state_dict[new_key] = v
        if ema_state_dict:
            model.load_state_dict(ema_state_dict, strict=False)
            return

        main_state_dict = {}
        for k, v in sd.items():
            if k.startswith('diffusion_model.'):
                new_key = k[len('diffusion_model.') :]
                main_state_dict[new_key] = v
        if main_state_dict:
            model.load_state_dict(main_state_dict, strict=False)
            return

        raise KeyError("Could not find diffusion_model weights in checkpoint state_dict")

    if 'ema_model' in ckpt:
        model.load_state_dict(ckpt['ema_model'], strict=False)
        return

    # Assume direct diffusion model state dict
    model.load_state_dict(ckpt, strict=False)


@torch.no_grad()
def _collect_rin_activations(rin, run_callable) -> Dict[str, torch.Tensor]:
    hooks = []
    acts: Dict[str, torch.Tensor] = {}

    def once(name: str):
        def _hook(m, inp, out):
            if name not in acts:
                tensor = out[0] if isinstance(out, tuple) else out
                if isinstance(tensor, torch.Tensor):
                    acts[name] = tensor.detach().to('cpu')
        return _hook

    for name, module in rin.named_modules():
        if any(name.startswith(prefix) for prefix in (
            'read_units.', 'write_units.', 'latent_processing_units.'
        )) or name in {'stem', 'output_linear'}:
            hooks.append(module.register_forward_hook(once(name)))

    run_callable()

    for h in hooks:
        h.remove()
    return acts


def _compare(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    diff = (a - b).abs()
    mean_abs = float(diff.mean().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = (a.norm() * b.norm()).item()
    cosine = float((torch.dot(a, b) / denom).item()) if denom > 0 else float('nan')
    return {"mean_abs": mean_abs, "max_abs": max_abs, "cosine": cosine}


def main():
    parser = argparse.ArgumentParser(description='Compare activations using the sampling pipeline')
    parser.add_argument('ckpt_a', type=str)
    parser.add_argument('ckpt_b', type=str)
    parser.add_argument('--config', type=str, default='sample_config.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--method', type=str, default='ddim')
    parser.add_argument('--class_label', type=int, default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--save_json', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load base config
    cfg = load_config(args.config)

    # Load checkpoints and override with their hyperparameters so we mirror training setup
    ckpt_a = _load_checkpoint(args.ckpt_a, device)
    ckpt_b = _load_checkpoint(args.ckpt_b, device)

    cfg_a = dict(cfg)
    cfg_b = dict(cfg)
    cfg_a = _override_config_with_ckpt_hparams(cfg_a, ckpt_a)
    cfg_b = _override_config_with_ckpt_hparams(cfg_b, ckpt_b)

    # Build models from their respective (possibly different) configs
    # Important: We want identical architecture to compare activations meaningfully.
    # Prefer config A, but assert critical fields match; if not, halt.
    critical = [
        'num_layers','latent_slots','latent_dim','latent_mlp_ratio','latent_num_heads',
        'tape_dim','tape_mlp_ratio','rw_num_heads','image_height','image_width','image_channels',
        'patch_size','latent_pos_encoding','tape_pos_encoding','self_cond','cond_on_latent_n',
        'cond_proj','cond_decoupled_read','xattn_enc_ln','num_classes'
    ]
    for k in critical:
        if k in cfg_a and k in cfg_b and cfg_a[k] != cfg_b[k]:
            raise RuntimeError(f"Model architectures differ in '{k}': {cfg_a[k]} vs {cfg_b[k]}")

    final_cfg = cfg_a

    model_a = create_model_from_config(final_cfg).to(device)
    model_b = create_model_from_config(final_cfg).to(device)

    # Initialize lazy params
    model_a.denoiser.pass_dummy_data(num_classes=final_cfg["num_classes"])
    model_b.denoiser.pass_dummy_data(num_classes=final_cfg["num_classes"])

    _load_into_diffusion_model_from_ckpt(model_a, ckpt_a)
    _load_into_diffusion_model_from_ckpt(model_b, ckpt_b)

    model_a.eval()
    model_b.eval()

    # Sampling kwargs
    sample_kwargs = dict(
        num_samples=args.num_samples,
        iterations=args.iterations,
        method=args.method,
        class_override=args.class_label,
        image_height=final_cfg.get('image_height', 32),
        image_width=final_cfg.get('image_width', 32),
        tape_dim=final_cfg.get('tape_dim', 256),
        seed=args.seed,
    )

    # Define the callable that runs one sampling pass
    def run_a():
        _ = model_a.sample(**sample_kwargs)

    def run_b():
        _ = model_b.sample(**sample_kwargs)

    acts_a = _collect_rin_activations(model_a.denoiser, run_a)
    acts_b = _collect_rin_activations(model_b.denoiser, run_b)

    common = sorted(set(acts_a.keys()) & set(acts_b.keys()))
    if not common:
        raise RuntimeError('No common activation points captured')

    stats = {name: _compare(acts_a[name], acts_b[name]) for name in common}
    ranked = sorted(stats.items(), key=lambda kv: kv[1]['mean_abs'], reverse=True)

    print('\nPer-layer activation differences (top-K):')
    for name, s in ranked[: args.topk]:
        print(f"  {name:40s}  mean|Δ|={s['mean_abs']:.6e}  max|Δ|={s['max_abs']:.6e}  cos={s['cosine']:.6f}")

    if args.save_json:
        out = {
            'ckpt_a': str(args.ckpt_a),
            'ckpt_b': str(args.ckpt_b),
            'stats': stats,
            'num_samples': args.num_samples,
            'iterations': args.iterations,
            'method': args.method,
            'seed': args.seed,
        }
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"Saved full stats → {args.save_json}")


if __name__ == '__main__':
    main()


