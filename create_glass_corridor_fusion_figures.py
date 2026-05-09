#!/usr/bin/env python3
"""
Create fusion-style comparison figures for the glass corridor dataset,
updated to include all student models through V9.

Layout per figure (2x4 grid):
  Row 0: RGB | Sensor Depth (dead%) | DA3-Small | V9 Student
  Row 1: V5 Student | V6 Student | V7 Student | V8 Student

Source data:
  RGB + sensor depth + DA3 depth:  /home/nishant/maps/glass_corridor_frames/
  Pre-computed V5/V6/V7 .npy:      /home/nishant/maps/glass_corridor_student_results/
  V8/V9 inference:                  live from checkpoints
  Extracted from bag:               rgbd_imu_20260302_173610 (glass corridor)

Usage:
  cd ~/MS_Project/ml_pipeline
  python create_glass_corridor_fusion_figures.py [--device cpu] [--frames 0 40 80]
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

FRAMES_DIR = Path("/home/nishant/maps/glass_corridor_frames")
STUDENT_DIR = Path("/home/nishant/maps/glass_corridor_student_results")
OUTPUT_DIR = Path("/home/nishant/maps/paper_figures/glass_corridor_v9")
CKPT_DIR = Path("/home/nishant/MS_Project/ml_pipeline/hpc_outputs")
DA3_METRIC_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

MIN_DEPTH = 0.1
MAX_DEPTH = 5.0


def load_student_model(version, device):
    cfg = Config()
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False,
                          backbone=cfg.BACKBONE)
    ckpt_path = CKPT_DIR / f"best_depth_{version}.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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


def run_student_inference(model, rgb_np_01, device):
    """rgb_np_01: float32 array in [0, 1] range, shape (H, W, 3)."""
    rgb_t = torch.from_numpy(
        rgb_np_01.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(device)
    with torch.no_grad():
        depth_pred, _ = model(rgb_t)
    return depth_pred.squeeze().cpu().numpy()


def make_figure(rgb, sensor, valid_mask, da3, students, frame_name,
                elapsed_s, dead_pct, out_dir):
    """
    students: dict of {label: depth_array}, e.g. {"V5": ..., "V9": ...}
    """
    depth_norm = Normalize(vmin=0.0, vmax=MAX_DEPTH)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#333333")

    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05],
                          wspace=0.08, hspace=0.12,
                          left=0.02, right=0.95, top=0.92, bottom=0.02)

    # Row 0: RGB, Sensor, DA3, V9
    row0 = [
        (rgb, "RGB", False),
        (np.where(valid_mask, sensor, np.nan),
         f"Sensor Depth ({dead_pct:.0f}% dead)", True),
        (da3, "DA3-Small (teacher)", True),
        (students.get("V9"), "V9 Student", True),
    ]
    for col, (data, title, use_cmap) in enumerate(row0):
        ax = fig.add_subplot(gs[0, col])
        if data is not None:
            if use_cmap:
                ax.imshow(data, cmap=cmap, norm=depth_norm)
            else:
                ax.imshow(data)
            ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    # Row 1: V5, V6, V7, V8
    for col, label in enumerate(["V5", "V6", "V7", "V8"]):
        ax = fig.add_subplot(gs[1, col])
        if label in students:
            ax.imshow(students[label], cmap=cmap, norm=depth_norm)
            ax.set_title(f"{label} Student", fontsize=13, fontweight="bold")
        ax.axis("off")

    cax = fig.add_subplot(gs[:, 4])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=depth_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Depth (m)", fontsize=12)

    fig.suptitle(f"Glass Corridor — t={elapsed_s:.1f}s: Model Comparison (DA3 + V5–V9)",
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
                        help="Frame indices to process (default: evenly spaced 10)")
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(FRAMES_DIR / "manifest.json"))
    n_frames = len(manifest)
    print(f"Manifest: {n_frames} frames from glass corridor bag")
    print(f"Output:   {OUTPUT_DIR}")
    print()

    print("Loading DA3 metric model...")
    from transformers import pipeline as hf_pipeline
    da3_dev = 0 if device.type == "cuda" else -1
    da3_pipe = hf_pipeline(task="depth-estimation", model=DA3_METRIC_MODEL,
                           device=da3_dev)

    print("Loading V8 Student...")
    v8_model, v8_cfg = load_student_model("v8", device)
    H, W = v8_cfg.INPUT_HEIGHT, v8_cfg.INPUT_WIDTH
    print(f"  Resolution: {W}x{H}")

    print("Loading V9 Student...")
    v9_model, _ = load_student_model("v9", device)
    print()

    # Pre-computed student depth directories
    precomputed = {
        "V5": STUDENT_DIR / "depth_v5_vivek",
        "V6": STUDENT_DIR / "depth_v6",
        "V7": STUDENT_DIR / "depth_v7",
    }

    if args.frames is not None:
        indices = args.frames
    else:
        indices = list(range(0, n_frames, max(1, n_frames // 10)))[:10]

    for i in indices:
        if i >= n_frames:
            print(f"  [WARN] Index {i} out of range, skipping")
            continue

        entry = manifest[i]
        fname = entry["frame"]
        elapsed = entry.get("elapsed_s", i * 0.5 + 30.0)
        print(f"[{i}] {fname} (t={elapsed:.1f}s) ...")

        rgb_path = FRAMES_DIR / "rgb" / f"{fname}.png"
        depth_path = FRAMES_DIR / "depth" / f"{fname}.npy"
        da3_path = FRAMES_DIR / "da3_depth" / f"{fname}.npy"

        rgb_pil = Image.open(rgb_path).convert("RGB")

        # Resize to model resolution (matching eval_pixel_fusion.py)
        rgb_resized = rgb_pil.resize((W, H), Image.BILINEAR)
        rgb_np_01 = np.array(rgb_resized, dtype=np.float32) / 255.0
        rgb_display = np.array(rgb_resized)

        sensor_full = np.load(str(depth_path)).astype(np.float32)
        sensor = np.array(Image.fromarray(sensor_full).resize((W, H), Image.NEAREST),
                          dtype=np.float32)
        valid_mask = (sensor >= MIN_DEPTH) & (sensor <= MAX_DEPTH)
        dead_pct = (~valid_mask).sum() * 100.0 / sensor.size

        # DA3 metric model outputs metric depth directly (larger = farther)
        da3_result = da3_pipe(rgb_pil)
        da3_metric = da3_result["predicted_depth"].squeeze().numpy().astype(np.float32)
        da3_d = np.array(Image.fromarray(da3_metric).resize((W, H), Image.BILINEAR),
                         dtype=np.float32)
        da3_d = median_scale(da3_d, sensor, valid_mask)

        students = {}

        # Pre-computed V5/V6/V7: resize to model resolution
        for label, sdir in precomputed.items():
            npy_path = sdir / f"{fname}.npy"
            if npy_path.exists():
                pred_full = np.load(str(npy_path)).astype(np.float32)
                if pred_full.shape != sensor.shape:
                    pred = np.array(Image.fromarray(pred_full).resize(
                        (W, H), Image.BILINEAR), dtype=np.float32)
                else:
                    pred = pred_full
                students[label] = median_scale(pred, sensor, valid_mask)

        # Live inference for V8 and V9 (input normalized to 0-1)
        v8_d = run_student_inference(v8_model, rgb_np_01, device)
        students["V8"] = median_scale(v8_d, sensor, valid_mask)

        v9_d = run_student_inference(v9_model, rgb_np_01, device)
        students["V9"] = median_scale(v9_d, sensor, valid_mask)

        make_figure(rgb_display, sensor, valid_mask, da3_d, students,
                    fname, elapsed, dead_pct, OUTPUT_DIR)

    del v8_model, v9_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone! Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
