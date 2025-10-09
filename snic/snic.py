"""
Utilities for SNIC-based patchify/depatchify.

This module provides two primary functions:
- snic_patchify: segment an image into superpixels (clusters) using SNIC and
  compute a per-cluster merged color token (mean BGR).
- snic_depatchify: reconstruct an image by expanding cluster tokens back to
  the original spatial resolution using a label map.

It also exposes a CLI to run an end-to-end test: given an input image, it
produces labels, tokens, overlay visualization, and a reconstructed image.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

# Robust import so this file works both as part of the package (python -m snic.snic)
# and when executed directly as a script (python snic/snic.py).
try:
    from .snic_segmentation import snic_segment, labels_to_boundary_overlay
except Exception:
    try:
        from snic.snic_segmentation import snic_segment, labels_to_boundary_overlay  # type: ignore
    except Exception:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.append(str(_Path(__file__).resolve().parent))
        from snic_segmentation import snic_segment, labels_to_boundary_overlay  # type: ignore


def snic_patchify(
    bgr: np.ndarray,
    num_superpixels: int = 400,
    compactness: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SNIC to obtain cluster labels and per-cluster color tokens.

    Args:
        bgr: Input image in BGR color order, dtype uint8, shape (H, W, 3).
        num_superpixels: Target number of superpixels.
        compactness: Compactness parameter controlling color vs spatial tradeoff.

    Returns:
        tokens_bgr: Array of shape (K, 3) with mean BGR uint8 per cluster id.
        labels: Array of shape (H, W) with cluster ids in [0, K-1].
        overlay: BGR image with superpixel boundaries overlaid (for visualization).
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3)")
    if bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")

    labels, _ = snic_segment(bgr, num_superpixels=num_superpixels, compactness=compactness)
    overlay = labels_to_boundary_overlay(bgr, labels)
    tokens_bgr, _, _ = compute_cluster_stats(labels=labels, bgr=bgr)
    return tokens_bgr, labels, overlay


def snic_depatchify(tokens_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reconstruct an image from cluster tokens and labels.

    Args:
        tokens_bgr: Array of shape (K, 3), dtype uint8, mean BGR per cluster id.
        labels: Array of shape (H, W), ints in [0, K-1].

    Returns:
        reconstructed: BGR image of shape (H, W, 3), dtype uint8.
    """
    if tokens_bgr.ndim != 2 or tokens_bgr.shape[1] != 3:
        raise ValueError("tokens_bgr must have shape (K, 3)")
    if labels.ndim != 2:
        raise ValueError("labels must have shape (H, W)")

    num_labels = int(np.max(labels)) + 1
    if num_labels > tokens_bgr.shape[0]:
        raise ValueError("labels contain ids beyond tokens_bgr length")

    reconstructed = tokens_bgr[labels]
    return reconstructed.astype(np.uint8)


def compute_cluster_stats(
    labels: np.ndarray,
    bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cluster mean color tokens, centroids, and pixel counts.

    Args:
        labels: (H, W) int32/64 array of cluster ids in [0, K-1].
        bgr: (H, W, 3) uint8 image used to compute mean color per cluster.

    Returns:
        tokens_bgr: (K, 3) uint8 mean BGR color per cluster id.
        centroids_yx: (K, 2) float32 centroid coordinates [y, x] per cluster.
        counts: (K,) int64 number of pixels per cluster.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be (H, W)")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must be (H, W, 3)")

    h, w = labels.shape
    num_labels = int(np.max(labels)) + 1

    labels_flat = labels.reshape(-1)
    counts = np.bincount(labels_flat, minlength=num_labels)
    counts_safe = counts.astype(np.float64)
    counts_safe[counts_safe == 0.0] = 1.0

    # Mean colors
    b_flat = bgr[..., 0].reshape(-1).astype(np.float64)
    g_flat = bgr[..., 1].reshape(-1).astype(np.float64)
    r_flat = bgr[..., 2].reshape(-1).astype(np.float64)
    sum_b = np.bincount(labels_flat, weights=b_flat, minlength=num_labels)
    sum_g = np.bincount(labels_flat, weights=g_flat, minlength=num_labels)
    sum_r = np.bincount(labels_flat, weights=r_flat, minlength=num_labels)
    mean_b = (sum_b / counts_safe).round().astype(np.uint8)
    mean_g = (sum_g / counts_safe).round().astype(np.uint8)
    mean_r = (sum_r / counts_safe).round().astype(np.uint8)
    tokens_bgr = np.stack([mean_b, mean_g, mean_r], axis=1)

    # Centroids [y, x]
    ys = np.repeat(np.arange(h, dtype=np.float64), w)
    xs = np.tile(np.arange(w, dtype=np.float64), h)
    sum_y = np.bincount(labels_flat, weights=ys, minlength=num_labels)
    sum_x = np.bincount(labels_flat, weights=xs, minlength=num_labels)
    centroid_y = (sum_y / counts_safe).astype(np.float32)
    centroid_x = (sum_x / counts_safe).astype(np.float32)
    centroids_yx = np.stack([centroid_y, centroid_x], axis=1)

    return tokens_bgr, centroids_yx, counts.astype(np.int64)


def snic_patchify_with_stats(
    bgr: np.ndarray,
    num_superpixels: int = 400,
    compactness: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SNIC patchify that also returns centroids and counts.

    Returns:
        tokens_bgr, labels, overlay, centroids_yx, counts
    """
    tokens_bgr, labels, overlay = snic_patchify(
        bgr=bgr, num_superpixels=num_superpixels, compactness=compactness
    )
    tokens_bgr, centroids_yx, counts = compute_cluster_stats(labels=labels, bgr=bgr)
    return tokens_bgr, labels, overlay, centroids_yx, counts


def _ensure_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ImportError("PyTorch is required for the *torch functions*") from e


def snic_patchify_torch(
    bgr_t: "np.ndarray | 'torch.Tensor'",
    num_superpixels: int = 400,
    compactness: float = 10.0,
    device: Optional[str] = None,
) -> Tuple["'torch.Tensor'", "'torch.Tensor'", "'torch.Tensor'", "'torch.Tensor'"]:
    """Torch API: returns tokens, labels, centroids, counts as torch tensors.

    Args:
        bgr_t: (H, W, 3) uint8 image as torch.Tensor or numpy array.
        num_superpixels: target K.
        compactness: SNIC compactness.
        device: optional torch device string for outputs.

    Returns:
        tokens_t: (K, 3) uint8
        labels_t: (H, W) int64
        centroids_t: (K, 2) float32 [y, x]
        counts_t: (K,) int64
    """
    _ensure_torch()
    import torch

    if hasattr(bgr_t, "numpy") and isinstance(bgr_t, torch.Tensor):
        bgr_np = bgr_t.detach().cpu().numpy()
    else:
        bgr_np = np.asarray(bgr_t)
    if bgr_np.dtype != np.uint8:
        raise ValueError("bgr_t must be uint8 with range [0, 255]")

    tokens_bgr, labels, _ = snic_patchify(bgr=bgr_np, num_superpixels=num_superpixels, compactness=compactness)
    _, centroids_yx, counts = compute_cluster_stats(labels=labels, bgr=bgr_np)

    dev = torch.device(device) if device is not None else None
    tokens_t = torch.from_numpy(tokens_bgr)
    labels_t = torch.from_numpy(labels.astype(np.int64))
    centroids_t = torch.from_numpy(centroids_yx.astype(np.float32))
    counts_t = torch.from_numpy(counts.astype(np.int64))
    if dev is not None:
        tokens_t = tokens_t.to(dev)
        labels_t = labels_t.to(dev)
        centroids_t = centroids_t.to(dev)
        counts_t = counts_t.to(dev)
    return tokens_t, labels_t, centroids_t, counts_t


def snic_depatchify_torch(
    tokens_t: "'torch.Tensor'",
    labels_t: "'torch.Tensor'",
) -> "'torch.Tensor'":
    """Torch API: reconstruct an image from tokens and labels.

    Args:
        tokens_t: (K, 3) uint8 or float tensor.
        labels_t: (H, W) int64 tensor.
    Returns:
        recon_t: (H, W, 3) tensor of same dtype/device as tokens_t.
    """
    _ensure_torch()
    import torch

    if labels_t.dtype != torch.long:
        labels_t = labels_t.long()
    recon_t = tokens_t[labels_t]
    return recon_t

def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="SNIC patchify/depatchify utilities")
    parser.add_argument("--image", type=str, help="Path to input image (BGR)")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "snic_output"), help="Output directory")
    parser.add_argument("--k", type=int, default=400, help="Target number of superpixels")
    parser.add_argument("--m", type=float, default=10.0, help="Compactness parameter")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    parser.add_argument("--labels", type=str, default="", help="Optional path to labels .npy for depatchify-only mode")
    parser.add_argument("--tokens", type=str, default="", help="Optional path to tokens .npy for depatchify-only mode")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If labels and tokens are provided, run depatchify-only mode
    if args.labels and args.tokens:
        labels_path = Path(args.labels)
        tokens_path = Path(args.tokens)
        logging.info("Depatchify-only mode: labels=%s tokens=%s", labels_path, tokens_path)
        labels = np.load(str(labels_path))
        tokens_bgr = np.load(str(tokens_path))
        recon = snic_depatchify(tokens_bgr=tokens_bgr, labels=labels)
        recon_path = out_dir / "snic_reconstructed.png"
        cv2.imwrite(str(recon_path), recon)
        logging.info("Saved reconstructed image: %s", recon_path)
        return

    if not args.image:
        raise SystemExit("--image is required unless --labels and --tokens are provided")

    image_path = Path(args.image)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    logging.info("Running SNIC patchify: image=%s k=%d m=%.3f", image_path, args.k, args.m)
    tokens_bgr, labels, overlay = snic_patchify(bgr, num_superpixels=args.k, compactness=args.m)

    # Save outputs
    overlay_path = out_dir / "snic_overlay.png"
    labels_path = out_dir / "snic_labels.npy"
    tokens_path = out_dir / "snic_tokens_bgr.npy"
    cv2.imwrite(str(overlay_path), overlay)
    np.save(str(labels_path), labels)
    np.save(str(tokens_path), tokens_bgr)
    logging.info("Saved overlay: %s", overlay_path)
    logging.info("Saved labels: %s", labels_path)
    logging.info("Saved tokens: %s", tokens_path)

    # Reconstruct immediately as a sanity check
    recon = snic_depatchify(tokens_bgr=tokens_bgr, labels=labels)
    recon_path = out_dir / "snic_reconstructed.png"
    cv2.imwrite(str(recon_path), recon)
    logging.info("Saved reconstructed image: %s", recon_path)


if __name__ == "__main__":
    _run_cli()


