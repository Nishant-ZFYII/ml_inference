#!/usr/bin/env python3
"""
Create 7-panel Gazebo fusion figures (corridor_fusion style) using
gazebo_captured_frames.

Layout per figure:
  Row 0: RGB | GT (Gazebo) | Noise-infused (77% dead) | DA3-Small
  Row 1: V9 Student | Fused (Noise + DA3) | Fused (Noise + V9)

Source data: /home/nishant/MS_Project/NCHSB/gazebo_captured_frames/

Usage:
  cd ~/MS_Project/ml_pipeline
  python create_gazebo_fusion_figures.py [--device cpu]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from models.student import build_student

FRAMES_DIR = Path("/home/nishant/MS_Project/NCHSB/gazebo_captured_frames")
OUTPUT_DIR = Path("/home/nishant/maps/paper_figures/corridor_fusion/gazebo")
V9_CKPT = Path("/home/nishant/MS_Project/ml_pipeline/hpc_outputs/best_depth_v9.pt")

MIN_DEPTH = 0.1
MAX_DEPTH = 10.0
DEAD_PIXEL_RATIO = 0.77
SEED = 42


def _make_dead_mask(h, w, ratio, rng):
    bh, bw = (h + 7) // 8, (w + 7) // 8
    block_dead = rng.random((bh, bw)) < ratio
    mask = np.repeat(np.repeat(block_dead, 8, axis=0), 8, axis=1)
    return mask[:h, :w]


def inject_sensor_failure(gt_depth, rng):
    """Apply 77% dead pixels (sensor failure) to GT depth."""
    result = gt_depth.copy()
    dead = _make_dead_mask(gt_depth.shape[0], gt_depth.shape[1],
                           DEAD_PIXEL_RATIO, rng)
    valid = np.isfinite(gt_depth) & (gt_depth >= MIN_DEPTH) & (gt_depth <= MAX_DEPTH)
    result[dead] = np.inf
    return result


def load_da3(device):
    from transformers import pipeline as hf_pipeline
    dev = 0 if device.type == "cuda" else -1
    return hf_pipeline(task="depth-estimation",
                       model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                       device=dev)


def load_v9(device):
    cfg = Config()
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False,
                          backbone=cfg.BACKBONE)
    ckpt = torch.load(str(V9_CKPT), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


def median_scale(pred, gt, valid_mask):
    p = pred[valid_mask]
    g = gt[valid_mask]
    if len(p) > 0 and np.median(p) > 1e-8:
        return pred * (np.median(g) / np.median(p))
    return pred


def make_figure(rgb, gt, noise_infused, da3, v9, fused_da3, fused_v9,
                frame_name, out_dir):
    vmax = 5.0
    depth_norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="white")

    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05],
                          wspace=0.08, hspace=0.15,
                          left=0.02, right=0.95, top=0.92, bottom=0.02)

    def to_display(d):
        return np.where(np.isfinite(d) & (d >= MIN_DEPTH) & (d <= MAX_DEPTH), d, np.nan)

    noise_valid_pct = np.isfinite(noise_infused) & (noise_infused >= MIN_DEPTH)
    valid_pct = noise_valid_pct.mean() * 100

    panels = [
        (0, 0, rgb, "RGB", False),
        (0, 1, to_display(gt), "GT Depth (Gazebo)", True),
        (0, 2, to_display(noise_infused), f"Sensor Failure ({100-valid_pct:.0f}% dead)", True),
        (0, 3, to_display(da3), "DA3-Small", True),
        (1, 0, to_display(v9), "V9 Student", True),
        (1, 1, to_display(fused_da3), "Fused (Sensor + DA3)", True),
        (1, 2, to_display(fused_v9), "Fused (Sensor + V9)", True),
    ]

    for row, col, data, title, use_cmap in panels:
        ax = fig.add_subplot(gs[row, col])
        if use_cmap:
            ax.imshow(data, cmap=cmap, norm=depth_norm)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    cax = fig.add_subplot(gs[:, 4])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=depth_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Depth (m)", fontsize=12)

    fig.suptitle(f"Gazebo Corridor — {frame_name}: Depth Fusion Under Sensor Failure",
                 fontsize=15, fontweight="bold", y=0.98)

    for ext in ("png", "pdf"):
        path = out_dir / f"{frame_name}.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {frame_name}.png + .pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="Frame indices to process (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb_dir = FRAMES_DIR / "rgb"
    depth_dir = FRAMES_DIR / "depth"
    frame_ids = sorted([p.stem for p in rgb_dir.glob("*.png")])
    if not frame_ids:
        print(f"No frames in {rgb_dir}")
        return

    print(f"Frames: {len(frame_ids)} from {FRAMES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    print("Loading DA3-Small...")
    da3_pipe = load_da3(device)

    print("Loading V9 Student...")
    v9_model, v9_cfg = load_v9(device)
    H, W = v9_cfg.INPUT_HEIGHT, v9_cfg.INPUT_WIDTH
    print(f"  V9 resolution: {W}x{H}")
    print()

    indices = args.frames if args.frames else list(range(len(frame_ids)))

    for i in indices:
        if i >= len(frame_ids):
            print(f"  [WARN] Index {i} out of range, skipping")
            continue

        fname = frame_ids[i]
        print(f"[{i}] {fname} ...")

        rgb_path = rgb_dir / f"{fname}.png"
        depth_path = depth_dir / f"{fname}.npy"
        if not rgb_path.exists() or not depth_path.exists():
            print(f"  [WARN] Missing files, skipping")
            continue

        rgb_pil = Image.open(rgb_path).convert("RGB")
        gt_full = np.load(str(depth_path)).astype(np.float32)
        gt_full[~np.isfinite(gt_full)] = 0.0

        rgb_resized = rgb_pil.resize((W, H), Image.BILINEAR)
        rgb_np = np.array(rgb_resized, dtype=np.float32) / 255.0
        gt = np.array(Image.fromarray(gt_full).resize((W, H), Image.NEAREST),
                      dtype=np.float32)

        rng = np.random.default_rng(SEED)
        noise_infused = inject_sensor_failure(gt, rng)
        gt_valid = np.isfinite(gt) & (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
        noise_valid = np.isfinite(noise_infused) & (noise_infused >= MIN_DEPTH) & (noise_infused <= MAX_DEPTH)

        # DA3 — scale against full GT
        da3_result = da3_pipe(rgb_pil)
        da3_metric = da3_result["predicted_depth"].squeeze().numpy().astype(np.float32)
        da3_d = np.array(Image.fromarray(da3_metric).resize((W, H), Image.BILINEAR),
                         dtype=np.float32)
        da3_d = median_scale(da3_d, gt, gt_valid)

        # V9 — scale against full GT
        rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            v9_pred, _ = v9_model(rgb_t)
        v9_d = v9_pred.squeeze().cpu().numpy()
        v9_d = median_scale(v9_d, gt, gt_valid)

        # Fused: keep sensor where valid, fill dead pixels with model prediction
        fused_da3 = np.where(noise_valid, noise_infused, da3_d)
        fused_v9 = np.where(noise_valid, noise_infused, v9_d)

        rgb_display = np.array(rgb_resized)

        make_figure(rgb_display, gt, noise_infused, da3_d, v9_d,
                    fused_da3, fused_v9, fname, OUTPUT_DIR)

    del v9_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone! Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
