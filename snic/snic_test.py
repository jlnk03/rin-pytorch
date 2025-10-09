"""
Simple end-to-end test for SNIC patchify/depatchify.

Usage examples:
  python -m snic.snic_test --image path/to/img.png --out snic/snic_output --k 400 --m 10
  python snic/snic_test.py --image path/to/img.png --out snic/snic_output --k 400 --m 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Robust imports whether run as a module or a script
try:
    from .snic import snic_patchify, snic_depatchify
except Exception:  # pragma: no cover - fallback for direct script execution
    try:
        from snic.snic import snic_patchify, snic_depatchify  # type: ignore
    except Exception as e:  # noqa: F401
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNIC E2E test: patchify + depatchify")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "snic_output"),
        help="Output directory",
    )
    parser.add_argument("--k", type=int, default=400, help="Target number of superpixels")
    parser.add_argument("--m", type=float, default=10.0, help="Compactness parameter")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    return parser.parse_args()


def save_palette(tokens_bgr: np.ndarray, out_path: Path, tile_size: int = 20, tiles_per_row: Optional[int] = None) -> None:
    """Save a small palette image visualizing cluster mean colors.

    This is optional for quick visual inspection of the tokens.
    """
    k = tokens_bgr.shape[0]
    if tiles_per_row is None:
        tiles_per_row = int(np.ceil(np.sqrt(k)))
    num_rows = int(np.ceil(k / tiles_per_row))

    h = num_rows * tile_size
    w = tiles_per_row * tile_size
    palette = np.zeros((h, w, 3), dtype=np.uint8)

    for idx in range(k):
        row = idx // tiles_per_row
        col = idx % tiles_per_row
        y0 = row * tile_size
        x0 = col * tile_size
        palette[y0 : y0 + tile_size, x0 : x0 + tile_size] = tokens_bgr[idx]

    cv2.imwrite(str(out_path), palette)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    logging.info("Reading image: %s", image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    logging.info("Patchify: k=%d m=%.3f", args.k, args.m)
    tokens_bgr, labels, overlay = snic_patchify(bgr, num_superpixels=args.k, compactness=args.m)

    overlay_path = out_dir / "test_overlay.png"
    labels_path = out_dir / "test_labels.npy"
    tokens_path = out_dir / "test_tokens_bgr.npy"
    palette_path = out_dir / "test_tokens_palette.png"

    cv2.imwrite(str(overlay_path), overlay)
    np.save(str(labels_path), labels)
    np.save(str(tokens_path), tokens_bgr)
    save_palette(tokens_bgr, palette_path)

    logging.info("Depatchify (reconstruction)")
    recon = snic_depatchify(tokens_bgr=tokens_bgr, labels=labels)
    recon_path = out_dir / "test_reconstructed.png"
    cv2.imwrite(str(recon_path), recon)

    logging.info("Saved:\n overlay=%s\n labels=%s\n tokens=%s\n palette=%s\n recon=%s", overlay_path, labels_path, tokens_path, palette_path, recon_path)


if __name__ == "__main__":
    main()


