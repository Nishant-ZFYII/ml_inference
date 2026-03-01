#!/usr/bin/env python3
"""
Evaluate student depth predictions against real sensor depth from the
Orbbec Femto Bolt in the deployment corridor.

Computes:
  - Overall depth metrics (RMSE, MAE, AbsRel, delta<1.25) on valid pixels
  - Per-bin breakdown (near 0.3-1m, mid 1-2m, far 2-4m)
  - Top/bottom image split (walls+ceiling vs floor)
  - Sensor failure pixel report (zero or >5m readings)

The manifest must contain 'sensor_depth' fields (float32 .npy, metres)
produced by extract_corridor_bag.py.

Usage:
    python eval_corridor_depth.py \
        --checkpoint best_depth.pt \
        --manifest corridor_eval_data/manifest.jsonl

    python eval_corridor_depth.py \
        --checkpoint best_depth.pt \
        --manifest corridor_eval_data/manifest.jsonl \
        --backbone efficientvit_b2.r224_in1k
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import Config
from models.student import build_student


def load_model(checkpoint_path: str, device: torch.device, cfg):
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False,
                          backbone=cfg.BACKBONE)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys (model lacks these): {missing}")
    if unexpected:
        print(f"  [INFO] Unexpected keys (checkpoint extras, ignored): "
              f"{[k for k in unexpected]}")
    model.to(device)
    model.eval()
    return model


def compute_depth_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(p) == 0:
        return {}
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))
    abs_rel = float(np.mean(np.abs(p - g) / (g + 1e-8)))
    ratio = np.maximum(p / (g + 1e-8), g / (p + 1e-8))
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)
    return {"rmse": rmse, "mae": mae, "abs_rel": abs_rel,
            "delta_125": delta_125, "n_pixels": int(len(p))}


def print_metrics(label, metrics):
    if not metrics or "rmse" not in metrics:
        print(f"  {label:30s}: no valid pixels")
        return
    print(f"  {label:30s}: RMSE {metrics['rmse']:.4f} m  "
          f"MAE {metrics['mae']:.4f} m  "
          f"AbsRel {metrics['abs_rel']:.4f}  "
          f"d<1.25 {metrics['delta_125']:.1f}%  "
          f"(n={metrics['n_pixels']:,})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate student depth vs corridor sensor depth")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None,
                        help="timm backbone name (must match checkpoint)")
    parser.add_argument("--min-depth", type=float, default=0.1,
                        help="Min valid sensor depth in metres (default: 0.1)")
    parser.add_argument("--max-depth", type=float, default=5.0,
                        help="Max valid sensor depth in metres (default: 5.0)")
    args = parser.parse_args()

    cfg = Config()
    if args.backbone:
        cfg.BACKBONE = args.backbone

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Backbone: {cfg.BACKBONE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Valid depth range: [{args.min_depth}, {args.max_depth}] m")

    model = load_model(args.checkpoint, device, cfg)

    manifest_dir = Path(args.manifest).parent
    with open(args.manifest) as f:
        samples = [json.loads(line) for line in f]
    print(f"Samples: {len(samples)}")

    H, W = cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH  # 240, 320

    # Accumulators
    all_pred = []
    all_gt = []
    all_valid = []
    total_sensor_zero = 0
    total_sensor_far = 0
    total_pixels = 0

    for entry in tqdm(samples, desc="Evaluating"):
        rgb_path = manifest_dir / entry["rgb"]
        depth_path = manifest_dir / entry["sensor_depth"]

        rgb = Image.open(rgb_path).convert("RGB")
        rgb = rgb.resize((W, H), Image.BILINEAR)
        rgb_np = np.array(rgb, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        sensor_depth = np.load(depth_path).astype(np.float32)
        # Resize sensor depth to model resolution using nearest (no interpolation artifacts)
        sensor_depth_pil = Image.fromarray(sensor_depth)
        sensor_depth = np.array(
            sensor_depth_pil.resize((W, H), Image.NEAREST), dtype=np.float32
        )

        with torch.no_grad():
            pred_depth, _ = model(rgb_t)
        pred_d = pred_depth.squeeze().cpu().numpy()

        valid = (sensor_depth >= args.min_depth) & (sensor_depth <= args.max_depth)
        sensor_zero = sensor_depth == 0
        sensor_far = sensor_depth > args.max_depth

        total_sensor_zero += sensor_zero.sum()
        total_sensor_far += sensor_far.sum()
        total_pixels += sensor_depth.size

        all_pred.append(pred_d)
        all_gt.append(sensor_depth)
        all_valid.append(valid)

    # Stack
    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)
    all_valid = np.array(all_valid)

    # Overall metrics (valid pixels only)
    overall = compute_depth_metrics(all_pred, all_gt, all_valid)

    print()
    print("=" * 75)
    print("Corridor Depth Evaluation (Student vs Femto Bolt Sensor)")
    print("=" * 75)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Frames: {len(samples)}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Valid pixels (sensor {args.min_depth}-{args.max_depth}m): "
          f"{all_valid.sum():,} ({all_valid.sum()*100/total_pixels:.1f}%)")
    print(f"Sensor zero (dead): {total_sensor_zero:,} "
          f"({total_sensor_zero*100/total_pixels:.1f}%)")
    print(f"Sensor far (>{args.max_depth}m): {total_sensor_far:,} "
          f"({total_sensor_far*100/total_pixels:.1f}%)")
    print()

    # Overall
    print("--- Overall (valid pixels) ---")
    print_metrics("All valid", overall)
    print()

    # Per-bin breakdown
    bins = [(0.3, 1.0, "Near (0.3-1.0m)"),
            (1.0, 2.0, "Mid  (1.0-2.0m)"),
            (2.0, 4.0, "Far  (2.0-4.0m)")]

    print("--- Per-depth-bin breakdown ---")
    for lo, hi, label in bins:
        bin_mask = all_valid & (all_gt >= lo) & (all_gt < hi)
        metrics = compute_depth_metrics(all_pred, all_gt, bin_mask)
        print_metrics(label, metrics)
    print()

    # Top/bottom split (walls+ceiling vs floor)
    # Bottom 1/3 of image ≈ floor; top 2/3 ≈ walls + ceiling
    split_row = H * 2 // 3  # row 160 out of 240
    top_mask = np.zeros_like(all_valid)
    top_mask[:, :split_row, :] = True
    bot_mask = np.zeros_like(all_valid)
    bot_mask[:, split_row:, :] = True

    print("--- Spatial split (top=walls/ceiling, bottom=floor) ---")
    top_valid = all_valid & top_mask
    bot_valid = all_valid & bot_mask
    print_metrics("Top 2/3 (walls/ceiling)", compute_depth_metrics(all_pred, all_gt, top_valid))
    print_metrics("Bottom 1/3 (floor)", compute_depth_metrics(all_pred, all_gt, bot_valid))
    print()

    # Sensor failure pixels: what does the student predict there?
    failure_mask = (all_gt == 0) | (all_gt > args.max_depth)
    n_failure = failure_mask.sum()
    if n_failure > 0:
        failure_pred = all_pred[failure_mask]
        print("--- Sensor failure pixels (zero/far readings) ---")
        print(f"  Count: {n_failure:,} ({n_failure*100/total_pixels:.1f}%)")
        print(f"  Student prediction range: "
              f"{failure_pred.min():.3f} - {failure_pred.max():.3f} m")
        print(f"  Student prediction mean:  {failure_pred.mean():.3f} m, "
              f"median: {np.median(failure_pred):.3f} m")
        print(f"  (No GT available — these are where the sensor failed)")

    print()
    print("=" * 75)
    print("NYU reference (B1 best_depth.pt): RMSE 0.998m, MAE 0.780m, "
          "AbsRel 0.255, d<1.25 57.9%")
    print("=" * 75)


if __name__ == "__main__":
    main()
