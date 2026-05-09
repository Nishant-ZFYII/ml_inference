#!/usr/bin/env python3
"""
Create 6-panel fusion figures (matching fusion_figures/ style) using the
corridor_key_frames data extracted from bag 172822.

Layout per figure:
  Row 0: RGB | Sensor Depth (with valid%) | DA3-Small (median-scaled)
  Row 1: V9 Student | Fused (Sensor + DA3) | Fused (Sensor + V9)

Source data:
  RGB + sensor depth: /home/nishant/maps/corridor_key_frames/
  Extracted from bag: rgbd_imu_20260302_172822_0.db3 (263s corridor)

Usage:
  cd ~/MS_Project/ml_pipeline
  python create_corridor_fusion_figures.py [--device cpu]
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

FRAMES_DIR = Path("/home/nishant/maps/corridor_key_frames")
OUTPUT_DIR = Path("/home/nishant/maps/paper_figures/corridor_fusion")
V9_CKPT = Path("/home/nishant/MS_Project/ml_pipeline/hpc_outputs/best_depth_v9.pt")

MIN_DEPTH = 0.1
MAX_DEPTH = 5.0


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


def make_figure(rgb, sensor, da3, v9, fused_da3, fused_v9,
                valid_mask, frame_name, elapsed_s, out_dir):
    depth_norm = Normalize(vmin=0.0, vmax=MAX_DEPTH)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="white")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05],
                          wspace=0.08, hspace=0.12,
                          left=0.02, right=0.95, top=0.92, bottom=0.02)

    panels = [
        (0, 0, rgb, "RGB", False),
        (0, 1, np.where(valid_mask, sensor, np.nan),
         f"Sensor Depth ({valid_mask.mean()*100:.0f}% valid)", True),
        (0, 2, da3, "DA3-Small (median-scaled)", True),
        (1, 0, v9, "V9 Student", True),
        (1, 1, fused_da3, "Fused (Sensor + DA3)", True),
        (1, 2, fused_v9, "Fused (Sensor + V9)", True),
    ]

    for row, col, data, title, use_cmap in panels:
        ax = fig.add_subplot(gs[row, col])
        if use_cmap:
            ax.imshow(data, cmap=cmap, norm=depth_norm)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    cax = fig.add_subplot(gs[:, 3])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=depth_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Depth (m)", fontsize=12)

    fig.suptitle(f"Bag 172822 — t={elapsed_s:.0f}s: Pixel-Level Depth Fusion",
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
                        help="Indices into manifest to process (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(FRAMES_DIR / "manifest.json"))
    print(f"Manifest: {len(manifest)} frames from bag 172822")
    print(f"Output:   {OUTPUT_DIR}")
    print()

    print("Loading DA3-Small...")
    da3_pipe = load_da3(device)

    print("Loading V9 Student...")
    v9_model, v9_cfg = load_v9(device)
    H, W = v9_cfg.INPUT_HEIGHT, v9_cfg.INPUT_WIDTH
    print(f"  V9 resolution: {W}x{H}")
    print()

    indices = args.frames if args.frames else list(range(len(manifest)))

    for i in indices:
        if i >= len(manifest):
            print(f"  [WARN] Index {i} out of range, skipping")
            continue

        entry = manifest[i]
        fname = entry["frame"]
        elapsed = entry["elapsed_s"]
        print(f"[{i}] {fname} (t={elapsed:.0f}s) ...")

        rgb_path = FRAMES_DIR / "rgb" / f"{fname}.png"
        depth_path = FRAMES_DIR / "depth_sensor" / f"{fname}.npy"

        rgb_pil = Image.open(rgb_path).convert("RGB")
        sensor_full = np.load(str(depth_path)).astype(np.float32)

        # Resize everything to model resolution (matching eval_pixel_fusion.py)
        rgb_resized = rgb_pil.resize((W, H), Image.BILINEAR)
        rgb_np = np.array(rgb_resized, dtype=np.float32) / 255.0
        sensor = np.array(Image.fromarray(sensor_full).resize((W, H), Image.NEAREST),
                          dtype=np.float32)
        valid_mask = (sensor >= MIN_DEPTH) & (sensor <= MAX_DEPTH)

        # DA3 metric model outputs metric depth directly (larger = farther)
        da3_result = da3_pipe(rgb_pil)
        da3_metric = da3_result["predicted_depth"].squeeze().numpy().astype(np.float32)
        da3_d = np.array(Image.fromarray(da3_metric).resize((W, H), Image.BILINEAR),
                         dtype=np.float32)
        da3_d = median_scale(da3_d, sensor, valid_mask)

        # V9 inference (input normalized to 0-1, matching original)
        rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            v9_pred, _ = v9_model(rgb_t)
        v9_d = v9_pred.squeeze().cpu().numpy()

        # Per-frame median scaling for V9
        v9_d = median_scale(v9_d, sensor, valid_mask)

        # Fused depth: sensor where valid, model prediction where dead
        fused_da3 = np.where(valid_mask, sensor, da3_d)
        fused_v9 = np.where(valid_mask, sensor, v9_d)

        # For display, use uint8 RGB (0-255)
        rgb_display = np.array(rgb_resized)

        make_figure(rgb_display, sensor, da3_d, v9_d,
                    fused_da3, fused_v9, valid_mask,
                    fname, elapsed, OUTPUT_DIR)

    del v9_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone! Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
