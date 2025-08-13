#!/usr/bin/env python
"""
compare_weights.py
==================

Quick utility to inspect how much two PyTorch checkpoints differ.

Usage
-----
    python compare_weights.py /path/to/ckpt_A.pt /path/to/ckpt_B.pt [--topk 10]

If the two paths are not supplied on the command line the script will fall
back to the hard-coded paths in HARD_CODED_CHECKPOINTS – handy when you are
iterating on the same pair of files again and again.

What you get
------------
1) Global metrics
   • L2 distance between the flattened parameter vectors.
   • Cosine similarity between the two vectors.

2) Per-parameter drill-down
   • Mean absolute difference for every tensor.
   • The top-k tensors with the largest drift are printed (k is configurable).

Additionally, the script writes:
   • keys_A.txt / keys_B.txt                – full key lists per checkpoint
   • missing_in_B.txt / missing_in_A.txt    – keys only present in one side
   • shape_mismatch.txt                     – overlapping keys with shape diffs
   • ckptA_overlap.pt / ckptB_overlap.pt    – tensors for overlapping same-shaped keys

Author: ChatGPT (generated on request)
"""
import sys
from pathlib import Path
from typing import Tuple, List

import torch
import numpy as np


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _flatten_state_dict(state_dict: dict) -> torch.Tensor:
    """Flatten only tensor entries of a state_dict into a single 1-D vector.

    Non-tensor items (ints, strings, lists) are silently ignored to avoid
    attribute errors like `'int' object has no attribute 'reshape'`.
    """
    flat_tensors = [p.reshape(-1) for p in state_dict.values() if isinstance(p, torch.Tensor)]
    if len(flat_tensors) == 0:
        raise ValueError("State-dict contains no tensors to compare.")
    return torch.nn.utils.parameters_to_vector(flat_tensors)


def _load_state_dict(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    # For PyTorch Lightning checkpoints grab the nested state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return ckpt


# -----------------------------------------------------------------------------
# Core comparison logic
# -----------------------------------------------------------------------------

def compare_checkpoints(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    topk: int = 5,
    csv_dir: str | None = None,
    key_filter: str | None = None,
    quantize_int4: bool = False,
) -> Tuple[float, float]:
    """Compare two checkpoints and print summary statistics.

    Returns
    -------
    l2 : float
        L2 distance between parameter vectors.
    cosine : float
        Cosine similarity between parameter vectors.
    """
    sd_a = _load_state_dict(ckpt_a)
    sd_b = _load_state_dict(ckpt_b)

    # ------------------------------------------------------------------
    # 1) Handle key mismatches gracefully
    # ------------------------------------------------------------------
    keys_a_full = sorted(sd_a.keys())
    keys_b_full = sorted(sd_b.keys())
    Path("keys_A.txt").write_text("\n".join(keys_a_full))
    Path("keys_B.txt").write_text("\n".join(keys_b_full))
    print("Saved full key lists → keys_A.txt / keys_B.txt")

    keys_a = set(keys_a_full)
    keys_b = set(keys_b_full)

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    intersect = sorted(keys_a & keys_b)

    if only_a or only_b:
        print("WARNING: State-dict keys differ – continuing with intersection only.")
        if only_a:
            print(f"  • {len(only_a)} keys only in A → saved to missing_in_B.txt")
            Path("missing_in_B.txt").write_text("\n".join(only_a))
        if only_b:
            print(f"  • {len(only_b)} keys only in B → saved to missing_in_A.txt")
            Path("missing_in_A.txt").write_text("\n".join(only_b))

    # Keep only overlapping tensors and further restrict to identical shapes
    sd_a_tensors = {k: v for k, v in sd_a.items() if isinstance(v, torch.Tensor)}
    sd_b_tensors = {k: v for k, v in sd_b.items() if isinstance(v, torch.Tensor)}

    shape_mismatch: List[str] = []
    matched_keys: List[str] = []
    for k in intersect:
        if k in sd_a_tensors and k in sd_b_tensors:
            if sd_a_tensors[k].shape == sd_b_tensors[k].shape:
                matched_keys.append(k)
            else:
                shape_mismatch.append(
                    f"{k}: {tuple(sd_a_tensors[k].shape)} vs {tuple(sd_b_tensors[k].shape)}"
                )

    if shape_mismatch:
        Path("shape_mismatch.txt").write_text("\n".join(shape_mismatch))
        print(
            f"Found {len(shape_mismatch)} overlapping keys with different shapes → shape_mismatch.txt"
        )

    sd_a = {k: sd_a_tensors[k] for k in matched_keys}
    sd_b = {k: sd_b_tensors[k] for k in matched_keys}

    # Optional: filter keys
    if key_filter:
        keys = [k for k in sd_a.keys() if key_filter in k]
        sd_a = {k: sd_a[k] for k in keys}
        sd_b = {k: sd_b[k] for k in keys}

    # Optionally dump the overlapping tensors so the user can inspect
    torch.save(sd_a, "ckptA_overlap.pt")
    torch.save(sd_b, "ckptB_overlap.pt")
    print("Saved overlapping tensor weights → ckptA_overlap.pt / ckptB_overlap.pt")

    # Optional: dump CSVs for overlapping tensors
    if csv_dir is not None:
        out_dir = Path(csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _sanitize(name: str) -> str:
            return name.replace("/", "_").replace(".", "_")

        def _quantize_to_int4(x: torch.Tensor) -> np.ndarray:
            x = x.detach().to(torch.float32)
            scale = x.abs().max().clamp(min=1e-12) / 7.0
            q = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
            return q.view(-1).cpu().numpy()

        for name in sd_a:
            a = sd_a[name]
            b = sd_b[name]
            safe = _sanitize(name)

            if quantize_int4:
                a_arr = _quantize_to_int4(a)
                b_arr = _quantize_to_int4(b)
            else:
                a_arr = a.view(-1).detach().cpu().numpy()
                b_arr = b.view(-1).detach().cpu().numpy()

            n = min(a_arr.shape[0], b_arr.shape[0])
            merged = np.stack([a_arr[:n], b_arr[:n]], axis=1)
            header = "A_int4,B_int4" if quantize_int4 else "A_float,B_float"
            np.savetxt(out_dir / f"{safe}.csv", merged, delimiter=",", header=header, comments="")
        print(f"Saved CSV dumps for {len(sd_a)} tensors → {out_dir}")

    # Compute global metrics if we still have comparable tensors
    vec_a = _flatten_state_dict(sd_a)
    vec_b = _flatten_state_dict(sd_b)

    if vec_a.numel() != vec_b.numel():
        print("WARNING: Overlapping tensors (same-shaped subset) have different total number of elements.")
        print(f"  • A: {vec_a.numel():,} elements\n  • B: {vec_b.numel():,} elements")
        print("Skipping global L2 / cosine stats – raw weights saved for manual inspection.")
        return float('nan'), float('nan')

    diff_vec = vec_a - vec_b
    l2 = diff_vec.norm().item()
    cosine = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).item()

    print("\nGlobal difference:")
    print(f"  L2 distance      : {l2:.6e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    # Per-parameter mean|Δ|
    per_layer: List[Tuple[float, str]] = []
    for name in sd_a:
        delta = (sd_a[name] - sd_b[name]).abs().mean().item()
        per_layer.append((delta, name))
    per_layer.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop {topk} layers by mean |Δ|:")
    for delta, name in per_layer[:topk]:
        print(f"  {delta:.6e}  {name}")

    return l2, cosine


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Simple manual arg-parsing so we keep the script dependency-free
    topk = 5
    csv_dir: str | None = None
    key_filter: str | None = None
    quantize_int4 = False
    if "--topk" in argv:
        idx = argv.index("--topk")
        try:
            topk = int(argv[idx + 1])
        except (IndexError, ValueError):
            raise SystemExit("Error: --topk must be followed by an integer")
        # Remove the two consumed args
        del argv[idx : idx + 2]

    if "--csv-dir" in argv:
        idx = argv.index("--csv-dir")
        try:
            csv_dir = argv[idx + 1]
        except (IndexError, ValueError):
            raise SystemExit("Error: --csv-dir must be followed by a directory path")
        del argv[idx : idx + 2]

    if "--filter" in argv:
        idx = argv.index("--filter")
        try:
            key_filter = argv[idx + 1]
        except (IndexError, ValueError):
            raise SystemExit("Error: --filter must be followed by a substring")
        del argv[idx : idx + 2]

    if "--quantize-int4" in argv:
        quantize_int4 = True
        argv.remove("--quantize-int4")

    # Hard-coded fallback (edit to taste)
    HARD_CODED_CHECKPOINTS = [
        # "/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar/masked/cifar_flex_nested_20250807_134133/model-step=2000.ckpt",
        "/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar/masked/cifar_flex_nested_20250813_104132/last.ckpt",
        # "/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar/cifar_flex_no_self_cond_20250807_140552/model-step=2000.ckpt",
        "/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar/cifar_flex_no_self_cond_20250813_115536/last.ckpt",
    ]

    if len(argv) == 2:
        ckpt_a, ckpt_b = argv
    elif len(argv) == 0:
        print("No checkpoint paths provided on CLI; falling back to hard-coded paths.")
        ckpt_a, ckpt_b = HARD_CODED_CHECKPOINTS
    else:
        raise SystemExit("Usage: python compare_weights.py ckpt_A ckpt_B [--topk 10]")

    compare_checkpoints(
        ckpt_a,
        ckpt_b,
        topk=topk,
        csv_dir=csv_dir,
        key_filter=key_filter,
        quantize_int4=quantize_int4,
    )


if __name__ == "__main__":
    main()

