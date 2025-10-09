"""
SNIC Superpixel Segmentation (non-iterative) implementation inspired by:

Achanta, R., & Süsstrunk, S. (2017).
"Superpixels and Polygons using Simple Non-Iterative Clustering (SNIC)".
CVPR 2017.

This script loads an input image, performs SNIC-style segmentation, and
outputs a boundary overlay and the label map. Default input is the app icon.

Note: This is a straightforward educational implementation intended to be
simple and readable. It follows the core idea of priority-queue-based region
growing with SLIC-like distance while enforcing connectivity from the start.
"""

from __future__ import annotations

import argparse
import heapq
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Cluster:
    id: int
    mean_l: float
    mean_a: float
    mean_b: float
    mean_x: float
    mean_y: float
    count: int


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    return mag


def find_low_gradient_seed(grad: np.ndarray, y: int, x: int) -> Tuple[int, int]:
    h, w = grad.shape
    y0, y1 = max(1, y - 1), min(h - 2, y + 1)
    x0, x1 = max(1, x - 1), min(w - 2, x + 1)
    patch = grad[y0 : y1 + 1, x0 : x1 + 1]
    min_idx = int(np.argmin(patch))
    dy, dx = divmod(min_idx, patch.shape[1])
    return y0 + dy, x0 + dx


def initialize_seeds(
    lab: np.ndarray, num_superpixels: int
) -> Tuple[List[Cluster], float]:
    h, w, _ = lab.shape
    num_pixels = h * w
    s = max(1.0, np.sqrt(float(num_pixels) / float(num_superpixels)))

    gray = cv2.cvtColor(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2GRAY)
    grad = compute_gradient_magnitude(gray)

    clusters: List[Cluster] = []
    grid_y = int(np.floor(h / s))
    grid_x = int(np.floor(w / s))
    if grid_y < 1:
        grid_y = 1
    if grid_x < 1:
        grid_x = 1

    id_counter = 0
    for gy in range(grid_y):
        cy = int((gy + 0.5) * s)
        if cy >= h:
            cy = h - 1
        for gx in range(grid_x):
            cx = int((gx + 0.5) * s)
            if cx >= w:
                cx = w - 1
            sy, sx = find_low_gradient_seed(grad, cy, cx)
            L, A, B = lab[sy, sx].astype(np.float32)
            clusters.append(
                Cluster(
                    id=id_counter,
                    mean_l=float(L),
                    mean_a=float(A),
                    mean_b=float(B),
                    mean_x=float(sx),
                    mean_y=float(sy),
                    count=1,
                )
            )
            id_counter += 1

    return clusters, float(s)


def snic_segment(
    bgr: np.ndarray,
    num_superpixels: int = 400,
    compactness: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = bgr.shape
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    clusters, s = initialize_seeds(lab, num_superpixels)
    labels = -np.ones((h, w), dtype=np.int32)

    # Priority queue of (distance, tie_breaker, y, x, cluster_id)
    heap: List[Tuple[float, int, int, int, int]] = []
    tie = 0
    for c in clusters:
        y = int(round(c.mean_y))
        x = int(round(c.mean_x))
        y = min(max(0, y), h - 1)
        x = min(max(0, x), w - 1)
        heapq.heappush(heap, (0.0, tie, y, x, c.id))
        tie += 1

    id_to_cluster = {c.id: c for c in clusters}
    ms_over_s = (compactness / s) ** 2

    def pixel_distance(ci: Cluster, yy: int, xx: int) -> float:
        L, A, B = lab[yy, xx].astype(np.float32)
        dc2 = (L - ci.mean_l) ** 2 + (A - ci.mean_a) ** 2 + (B - ci.mean_b) ** 2
        ds2 = (xx - ci.mean_x) ** 2 + (yy - ci.mean_y) ** 2
        return float(np.sqrt(dc2 + ms_over_s * ds2))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        dist, _, y, x, cid = heapq.heappop(heap)
        if labels[y, x] != -1:
            continue

        labels[y, x] = cid
        c = id_to_cluster[cid]
        L, A, B = lab[y, x].astype(np.float32)

        # Update cluster running means incrementally
        new_count = c.count + 1
        c.mean_l = (c.mean_l * c.count + float(L)) / new_count
        c.mean_a = (c.mean_a * c.count + float(A)) / new_count
        c.mean_b = (c.mean_b * c.count + float(B)) / new_count
        c.mean_x = (c.mean_x * c.count + float(x)) / new_count
        c.mean_y = (c.mean_y * c.count + float(y)) / new_count
        c.count = new_count

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if labels[ny, nx] != -1:
                continue
            d = pixel_distance(c, ny, nx)
            heapq.heappush(heap, (d, tie, ny, nx, cid))
            tie += 1

    return labels, lab


def labels_to_boundary_overlay(bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            l = labels[y, x]
            if y + 1 < h and labels[y + 1, x] != l:
                boundary[y, x] = 255
            if x + 1 < w and labels[y, x + 1] != l:
                boundary[y, x] = 255
    overlay = bgr.copy()
    overlay[boundary == 255] = (0, 0, 255)  # red boundaries
    return overlay


def run(
    image_path: Path,
    output_dir: Path,
    num_superpixels: int,
    compactness: float,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    labels, _ = snic_segment(bgr, num_superpixels=num_superpixels, compactness=compactness)
    overlay = labels_to_boundary_overlay(bgr, labels)

    overlay_path = output_dir / "snic_overlay.png"
    labels_path = output_dir / "snic_labels.npy"
    cv2.imwrite(str(overlay_path), overlay)
    np.save(str(labels_path), labels)
    return overlay_path, labels_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNIC superpixel segmentation test script")
    parser.add_argument(
        "--image",
        type=str,
        default=str(
            Path(
                "/Users/julianlink/Documents/muunai/avin-web/next/public/assets/muunai_app_icon.png"
            )
        ),
        help="Path to input image",
    )
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    image_path = Path(args.image)
    output_dir = Path(args.out)
    logging.info("Running SNIC on image: %s", image_path)
    overlay_path, labels_path = run(
        image_path=image_path,
        output_dir=output_dir,
        num_superpixels=args.k,
        compactness=args.m,
    )
    logging.info("Saved overlay to: %s", overlay_path)
    logging.info("Saved labels to: %s", labels_path)


if __name__ == "__main__":
    main()


