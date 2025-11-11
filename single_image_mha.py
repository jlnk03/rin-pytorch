import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from rin_pytorch.Rin import patchify as rin_patchify


class ResizeMaxSide:
    def __init__(self, max_side, interpolation=Image.BILINEAR):
        self.max_side = max_side
        self.interpolation = interpolation

    def __call__(self, img):
        width, height = img.size
        max_dim = max(width, height)
        if max_dim > self.max_side:
            scale = self.max_side / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), self.interpolation)
        return img


def load_single_imagenet_tensor(max_side=128):
    ds = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
    ex = ds[0]
    img = ex["image"]
    if img.mode == "RGBA":
        img = img.convert("RGB")
    transform = transforms.Compose([
        ResizeMaxSide(max_side),
        transforms.ToTensor(),
    ])
    img_t = transform(img)  # [C,H,W] in [0,1]
    # Ensure H,W divisible by 16 for 16x16 pooling
    C, H, W = img_t.shape
    H_crop = H - (H // 16) * 16
    W_crop = W - (W // 16) * 16
    if H_crop > 0 or W_crop > 0:
        img_t = img_t[:, :H - H_crop, :W - W_crop]
    return img_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_side", type=int, default=128, help="Longest image side after resize")
    parser.add_argument("--latent_index", type=int, default=-1, help="Which latent token to visualize (-1 = center)")
    parser.add_argument("--colormap", type=str, default="magma", help="Matplotlib colormap for heatmap")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--out_prefix", type=str, default="attention_overlay", help="Output filename prefix")
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="pooled_self",
        choices=["pooled_self", "pooled_to_dense"],
        help="pooled_self: attention among pooled tokens, then upsample; pooled_to_dense: pooled queries over dense pixels",
    )
    parser.add_argument("--pixel_refine", action="store_true", help="Refine selected latent with full pixel-level attention over top pooled blocks")
    parser.add_argument("--top_percent", type=float, default=10.0, help="Top percent pooled blocks to keep for refinement")
    parser.add_argument(
        "--aggregate_mode",
        type=str,
        default="none",
        choices=["none", "mean", "max"],
        help="Aggregate attention across all pooled queries; saves pooled and pixel-level overlays",
    )
    parser.add_argument(
        "--roi_from",
        type=str,
        default="aggregate",
        choices=["aggregate", "latent"],
        help="ROI selection based on aggregated pooled keys or a single latent row",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load single image (similar to TrainerLightning)
    img = load_single_imagenet_tensor(max_side=args.max_side).to(device)  # [C,H,W], 3-channel RGB
    C, H, W = img.shape

    # 2) Patchify with patch size = 1 using existing function (batch version)
    img_b = img.unsqueeze(0)  # [1,C,H,W]
    tokens_dense = rin_patchify(img_b, p=1)  # [1, H*W, C]

    # 3) Pool into 16x16 blocks to create latent tokens (queries)
    pooled = F.avg_pool2d(img_b, kernel_size=16, stride=16)  # [1,C,H/16,W/16]
    tokens_latent = rin_patchify(pooled, p=1)  # [1, (H/16)*(W/16), C]

    # 4) Attention
    nh = H // 16
    nw = W // 16
    if args.attn_mode == "pooled_self":
        # Self-attention over pooled tokens (queries=keys=values = pooled tokens)
        mha = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True).to(device)
        attn_out, attn_weights = mha(
            query=tokens_latent,  # [B,Lp,C]
            key=tokens_latent,    # [B,Lp,C]
            value=tokens_latent,  # [B,Lp,C]
            need_weights=True,
            average_attn_weights=False,  # -> [B, num_heads, Lp, Lp]
        )
    else:
        # Cross-attention (pooled queries -> dense keys/values)
        mha = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True).to(device)
        attn_out, attn_weights = mha(
            query=tokens_latent,  # [B,Lq,C]
            key=tokens_dense,     # [B,Lk,C]
            value=tokens_dense,   # [B,Lk,C]
            need_weights=True,
            average_attn_weights=False,  # -> [B, num_heads, Lq, Lk]
        )

    # 5) Print shapes and a small attention sample
    print(f"Image shape: {tuple(img.shape)}")
    print(f"Dense tokens: {tuple(tokens_dense.shape)}  (T={H*W})")
    print(f"Latent tokens (16x16 pooled): {tuple(tokens_latent.shape)}  (T={(H//16)*(W//16)})")
    print(f"Attention output shape: {tuple(attn_out.shape)}")
    print(f"Attention weights shape: {tuple(attn_weights.shape)}")  # [1,1,L*,L*]

    # Show top-5 attended dense tokens for the first latent token
    with torch.no_grad():
        first_head_weights = attn_weights[0, 0]
        lq0 = first_head_weights[0]
        top_vals, top_idx = torch.topk(lq0, k=min(5, lq0.numel()))
        print("Top-5 attention indices for first query:", top_idx.tolist())
        print("Top-5 attention weights for first query:", top_vals.tolist())

    # 6) Visualize attention heatmap for a selected latent token (overlay on original image)
    with torch.no_grad():
        Lq = nh * nw
        if args.latent_index < 0:
            # center latent
            r = nh // 2
            c = nw // 2
            lq_index = int(r * nw + c)
        else:
            lq_index = int(min(max(args.latent_index, 0), Lq - 1))
        print(f"Visualizing latent index {lq_index} (grid {nh}x{nw})")

        if args.attn_mode == "pooled_self":
            # Attention over pooled tokens, then upsample to image resolution
            weights_row = attn_weights[0, 0, lq_index]  # [Lp]
            heat_small = weights_row.reshape(nh, nw).unsqueeze(0).unsqueeze(0)  # [1,1,nh,nw]
            heat_up = F.interpolate(heat_small, size=(H, W), mode="nearest")   # [1,1,H,W]
            heat = heat_up.squeeze(0).squeeze(0)
        else:
            # Cross-attention over dense tokens: one weight per pixel (p=1)
            weights_row = attn_weights[0, 0, lq_index]  # [Lk]
            heat = weights_row.reshape(H, W)
        # Normalize to [0,1]
        heat = heat - heat.min()
        denom = heat.max().clamp(min=1e-8)
        heat = heat / denom

        # Convert image and heatmap to numpy
        img_np = img.detach().cpu().numpy()  # [C,H,W]
        img_np = np.transpose(img_np, (1, 2, 0))  # [H,W,C]
        heat_np = heat.detach().cpu().numpy()  # [H,W]

        # Colorize heatmap
        if not hasattr(cm, args.colormap):
            cmap = cm.get_cmap("magma")
        else:
            cmap = cm.get_cmap(args.colormap)
        heat_color = cmap(heat_np)[..., :3]  # [H,W,3] in [0,1]

        # Overlay
        alpha = float(np.clip(args.alpha, 0.0, 1.0))
        overlay = (1.0 - alpha) * img_np + alpha * heat_color
        overlay = np.clip(overlay, 0.0, 1.0)

        # Save figures
        out_img = f"{args.out_prefix}_img.png"
        out_heat = f"{args.out_prefix}_heatmap_lq{lq_index}.png"
        out_overlay = f"{args.out_prefix}_overlay_lq{lq_index}.png"

        plt.imsave(out_img, np.clip(img_np, 0, 1))
        plt.imsave(out_heat, heat_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
        plt.imsave(out_overlay, overlay)
        print(f"Saved:\n- {out_img}\n- {out_heat}\n- {out_overlay}")

        # ROI-based pixel refinement: compute rough pooled attention -> select top-k pooled blocks -> pixel attention only in ROI
        if args.pixel_refine:
            # Ensure we have pooled self-attention to pick ROI
            if args.attn_mode == "pooled_self":
                pooled_attn = attn_weights[0, 0]  # [Lp, Lp]
            else:
                mha_p = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True).to(device)
                _, attn_weights_p = mha_p(
                    query=tokens_latent,
                    key=tokens_latent,
                    value=tokens_latent,
                    need_weights=True,
                    average_attn_weights=False,
                )
                pooled_attn = attn_weights_p[0, 0]  # [Lp, Lp]

            Lp = nh * nw
            # ROI scores: aggregated across queries or single latent row
            if args.roi_from == "aggregate":
                reduce_fn = torch.mean if args.aggregate_mode in ["mean", "none"] else torch.amax
                roi_scores = reduce_fn(pooled_attn, dim=0)  # [Lp]
            else:
                roi_scores = pooled_attn[lq_index]  # [Lp]

            k = max(1, int((float(args.top_percent) / 100.0) * Lp))
            _, top_idx_pooled = torch.topk(roi_scores, k=k, largest=True, sorted=False)

            # Build allowed pixel mask from pooled blocks (each pooled cell -> 16x16 pixels)
            allowed = torch.zeros((H, W), dtype=torch.bool, device=device)
            for idx in top_idx_pooled.tolist():
                r = idx // nw
                c = idx % nw
                rs, re = int(r * 16), int((r + 1) * 16)
                cs, ce = int(c * 16), int((c + 1) * 16)
                allowed[rs:re, cs:ce] = True

            # Cross-attend ALL pooled queries to pixels, mask outside ROI
            # Build attn_mask of shape [Lq, Lk] (True = masked)
            not_allowed_flat = (~allowed).reshape(1, H * W)  # [1, Lk]
            attn_mask_bool_full = not_allowed_flat.expand(tokens_latent.shape[1], -1)  # [Lq, Lk]

            mha_px = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True).to(device)
            attn_out_px, attn_w_px = mha_px(
                query=tokens_latent,  # [1,Lq,C]
                key=tokens_dense,     # [1,Lk,C]
                value=tokens_dense,   # [1,Lk,C]
                need_weights=True,
                average_attn_weights=False,  # [1,1,Lq,Lk]
                attn_mask=attn_mask_bool_full,
            )

            # Aggregate pixel attention across queries, mask outside ROI to black
            reduce_fn_px = torch.mean if args.aggregate_mode in ["mean", "none"] else torch.amax
            pixel_scores = reduce_fn_px(attn_w_px[0, 0], dim=0)  # [Lk]
            heat_px = pixel_scores.reshape(H, W)
            # Normalize within ROI only
            roi_vals = heat_px[allowed]
            if roi_vals.numel() > 0:
                roi_vals = roi_vals - roi_vals.min()
                denom_px = roi_vals.max().clamp(min=1e-8)
                roi_vals = roi_vals / denom_px
                heat_px_norm = torch.zeros_like(heat_px)
                heat_px_norm[allowed] = roi_vals
            else:
                heat_px_norm = torch.zeros_like(heat_px)

            heat_px_np = heat_px_norm.detach().cpu().numpy()
            heat_px_color = cmap(heat_px_np)[..., :3]
            # Overlay only in ROI (outside = original image)
            overlay_px = img_np.copy()
            overlay_px = (1.0 - alpha) * overlay_px + alpha * heat_px_color
            overlay_px = np.clip(overlay_px, 0.0, 1.0)

            out_heat_px = f"{args.out_prefix}_roi_pixel_heatmap.png"
            out_overlay_px = f"{args.out_prefix}_roi_pixel_overlay.png"
            plt.imsave(out_heat_px, heat_px_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
            plt.imsave(out_overlay_px, overlay_px)
            print(f"ROI pixel refinement saved:\n- {out_heat_px}\n- {out_overlay_px}")

        # Aggregate across all pooled queries
        if args.aggregate_mode != "none":
            reduce_fn = torch.mean if args.aggregate_mode == "mean" else torch.amax

            # 1) Aggregate over pooled keys (pooled self-attn), then upsample to pixels
            if args.attn_mode == "pooled_self":
                pooled_attn = attn_weights[0, 0]  # [Lp, Lp] (queries x keys)
            else:
                # If we didn't compute pooled self-attention this run, compute it now for pooled aggregation
                mha_p = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True).to(device)
                _, attn_weights_p = mha_p(
                    query=tokens_latent,
                    key=tokens_latent,
                    value=tokens_latent,
                    need_weights=True,
                    average_attn_weights=False,
                )
                pooled_attn = attn_weights_p[0, 0]  # [Lp, Lp]

            pooled_keys_agg = reduce_fn(pooled_attn, dim=0)  # [Lp] aggregate across queries
            heat_small = pooled_keys_agg.reshape(nh, nw).unsqueeze(0).unsqueeze(0)  # [1,1,nh,nw]
            heat_up = F.interpolate(heat_small, size=(H, W), mode="nearest")       # [1,1,H,W]
            heat_pooled = heat_up.squeeze(0).squeeze(0)
            heat_pooled = heat_pooled - heat_pooled.min()
            denom_pooled = heat_pooled.max().clamp(min=1e-8)
            heat_pooled = heat_pooled / denom_pooled

            heat_pooled_np = heat_pooled.detach().cpu().numpy()
            heat_pooled_color = cmap(heat_pooled_np)[..., :3]
            overlay_pooled = (1.0 - alpha) * img_np + alpha * heat_pooled_color
            overlay_pooled = np.clip(overlay_pooled, 0.0, 1.0)

            out_heat_pooled = f"{args.out_prefix}_aggregate_pooled_heatmap.png"
            out_overlay_pooled = f"{args.out_prefix}_aggregate_pooled_overlay.png"
            plt.imsave(out_heat_pooled, heat_pooled_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
            plt.imsave(out_overlay_pooled, overlay_pooled)
            print(f"Aggregate pooled saved:\n- {out_heat_pooled}\n- {out_overlay_pooled}")

            # 2) Aggregate to pixel level (pooled→dense cross-attn), save pixel heatmap and overlay
            if args.attn_mode == "pooled_to_dense":
                dense_attn = attn_weights[0, 0]  # [Lp, Lk]
            else:
                mha_d = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True).to(device)
                _, attn_weights_d = mha_d(
                    query=tokens_latent,  # [B,Lp,C]
                    key=tokens_dense,     # [B,Lk,C]
                    value=tokens_dense,   # [B,Lk,C]
                    need_weights=True,
                    average_attn_weights=False,  # -> [B,1,Lp,Lk]
                )
                dense_attn = attn_weights_d[0, 0]  # [Lp, Lk]

            pixel_scores = reduce_fn(dense_attn, dim=0)  # [Lk]
            heat_pixel = pixel_scores.reshape(H, W)
            heat_pixel = heat_pixel - heat_pixel.min()
            denom_px2 = heat_pixel.max().clamp(min=1e-8)
            heat_pixel = heat_pixel / denom_px2

            heat_pixel_np = heat_pixel.detach().cpu().numpy()
            heat_pixel_color = cmap(heat_pixel_np)[..., :3]
            overlay_pixel = (1.0 - alpha) * img_np + alpha * heat_pixel_color
            overlay_pixel = np.clip(overlay_pixel, 0.0, 1.0)

            out_heat_pixel = f"{args.out_prefix}_aggregate_pixel_heatmap.png"
            out_overlay_pixel = f"{args.out_prefix}_aggregate_pixel_overlay.png"
            plt.imsave(out_heat_pixel, heat_pixel_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
            plt.imsave(out_overlay_pixel, overlay_pixel)
            print(f"Aggregate pixel saved:\n- {out_heat_pixel}\n- {out_overlay_pixel}")


if __name__ == "__main__":
    main()


