#!/usr/bin/env python
"""
compare_weights.py
==================

Quick utility to inspect how much **two** PyTorch checkpoints differ.

Usage
-----
    python compare_weights.py /path/to/ckpt_A.pt /path/to/ckpt_B.pt [--topk 10]

If the two paths are *not* supplied on the command line the script will fall
back to the hard-coded paths in `HARD_CODED_CHECKPOINTS` – handy when you are
iterating on the same pair of files again and again.

What you get
------------
1) Global metrics
   • L2 distance between the flattened parameter vectors.
   • Cosine similarity between the two vectors.

2) Per-parameter drill-down
   • Mean absolute difference for every tensor.
   • The top-k tensors with the largest drift are printed (k is configurable).

This makes it trivial to locate the first layer that has started diverging.

Author: ChatGPT (generated on request)
"""
import sys
from pathlib import Path
from typing import Tuple, List

import torch

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _flatten_state_dict(state_dict: dict) -> torch.Tensor:
    """Flatten a state_dict into a single 1-D tensor for fast global diffs."""
    return torch.nn.utils.parameters_to_vector([p.reshape(-1) for p in state_dict.values()])


def _load_state_dict(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"Loading checkpoint: {path}")
    return torch.load(path, map_location="cpu")


# -----------------------------------------------------------------------------
# Core comparison logic
# -----------------------------------------------------------------------------

def compare_checkpoints(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    topk: int = 5,
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

    if sd_a.keys() != sd_b.keys():
        diff_keys = sd_a.keys() ^ sd_b.keys()
        raise ValueError(
            f"State-dict keys mismatch! Differing keys ({len(diff_keys)}):\n" + "\n".join(sorted(diff_keys))
        )

    vec_a = _flatten_state_dict(sd_a)
    vec_b = _flatten_state_dict(sd_b)

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
    if "--topk" in argv:
        idx = argv.index("--topk")
        try:
            topk = int(argv[idx + 1])
        except (IndexError, ValueError):
            raise SystemExit("Error: --topk must be followed by an integer")
        # Remove the two consumed args
        del argv[idx : idx + 2]

    # Hard-coded fallback (edit to taste)
    HARD_CODED_CHECKPOINTS = [
        "/absolute/path/to/checkpoint_A.pt",  # <- edit me
        "/absolute/path/to/checkpoint_B.pt",  # <- edit me
    ]

    if len(argv) == 2:
        ckpt_a, ckpt_b = argv
    elif len(argv) == 0:
        print("No checkpoint paths provided on CLI; falling back to hard-coded paths.")
        ckpt_a, ckpt_b = HARD_CODED_CHECKPOINTS
    else:
        raise SystemExit("Usage: python compare_weights.py ckpt_A ckpt_B [--topk 10]")

    compare_checkpoints(ckpt_a, ckpt_b, topk=topk)


if __name__ == "__main__":
    main()
