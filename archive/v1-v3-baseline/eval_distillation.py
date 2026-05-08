#!/usr/bin/env python3
"""
Evaluate student model against teacher predictions (Table IV).

Compares student outputs to DA2 depth and YOLO+SAM2 segmentation labels
from the teacher inference pipeline.

Usage:
    python eval_distillation.py \
        --checkpoint checkpoints/best.pt \
        --manifest path/to/manifest.jsonl \
        --data-root ./data
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import Config
from models.student import build_student


def compute_depth_metrics(pred: np.ndarray, target: np.ndarray):
    """
    Compute depth evaluation metrics.

    Returns dict with: rmse, abs_rel, delta_1.25
    """
    valid = (target > 0) & (pred > 0)
    if valid.sum() == 0:
        return {"rmse": 0, "abs_rel": 0, "delta_125": 0}

    p, t = pred[valid], target[valid]
    rmse = float(np.sqrt(((p - t) ** 2).mean()))
    abs_rel = float((np.abs(p - t) / t).mean())

    ratio = np.maximum(p / t, t / p)
    delta_125 = float((ratio < 1.25).mean())

    return {"rmse": rmse, "abs_rel": abs_rel, "delta_125": delta_125}


def compute_seg_metrics(pred: np.ndarray, target: np.ndarray,
                        num_classes: int = 6):
    """
    Compute segmentation mIoU and per-class IoU.

    Returns dict with: miou, per_class_iou (list)
    """
    per_class = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union > 0:
            per_class.append(float(intersection / union))
        else:
            per_class.append(float("nan"))

    valid_ious = [x for x in per_class if not np.isnan(x)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {"miou": miou, "per_class_iou": per_class}


def main():
    p = argparse.ArgumentParser(description="Evaluate student vs teachers")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--manifest", type=str, required=True,
                   help="Path to manifest.jsonl from teacher inference")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of samples (0 = all)")
    args = p.parse_args()

    cfg = Config()

    if not os.path.exists(args.manifest):
        print(f"ERROR: Manifest not found: {args.manifest}")
        print("\nTo generate teacher predictions, run:")
        print("  sbatch teacher_infer/teacher_infer.slurm")
        print("Then run this script with --manifest pointing to the output.")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    # Load manifest
    samples = []
    with open(args.manifest) as f:
        for line in f:
            samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Evaluating {len(samples)} samples...")
    base_dir = Path(os.path.dirname(args.manifest))

    all_depth_metrics = []
    all_seg_metrics = []

    for sample in tqdm(samples, desc="Evaluating"):
        rgb_path = base_dir / sample["rgb"]
        if not rgb_path.exists():
            continue

        # Load and preprocess RGB
        img = Image.open(rgb_path).convert("RGB")
        img = img.resize((cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT))
        rgb_np = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)

        # Run student inference
        with torch.no_grad():
            pred_depth, pred_seg = model(rgb_t)

        pred_depth_np = pred_depth.squeeze().cpu().numpy()
        pred_seg_np = pred_seg.argmax(dim=1).squeeze().cpu().numpy()

        # Compare to DA2 teacher depth
        da2_path = sample.get("da2_depth")
        if da2_path:
            da2_full = base_dir / da2_path
            if da2_full.exists():
                da2_depth = np.load(da2_full).astype(np.float32)
                da2_resized = np.array(
                    Image.fromarray(da2_depth).resize(
                        (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT)),
                    dtype=np.float32)
                metrics = compute_depth_metrics(pred_depth_np, da2_resized)
                all_depth_metrics.append(metrics)

        # Compare to SAM2 teacher segmentation
        sam2_path = sample.get("sam2_seg")
        if sam2_path:
            sam2_full = base_dir / sam2_path
            if sam2_full.exists():
                sam2_seg = np.load(sam2_full).astype(np.int64)
                sam2_resized = np.array(
                    Image.fromarray(sam2_seg.astype(np.uint8)).resize(
                        (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT),
                        Image.NEAREST),
                    dtype=np.int64)
                metrics = compute_seg_metrics(pred_seg_np, sam2_resized,
                                              cfg.NUM_CLASSES)
                all_seg_metrics.append(metrics)

    # ── Print Table IV ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Table IV: Student vs Teacher Distillation Quality")
    print(f"{'='*55}")

    if all_depth_metrics:
        avg_rmse = np.mean([m["rmse"] for m in all_depth_metrics])
        avg_absrel = np.mean([m["abs_rel"] for m in all_depth_metrics])
        avg_delta = np.mean([m["delta_125"] for m in all_depth_metrics])
        print(f"\nDepth (Student vs DA2-Large):")
        print(f"  RMSE:           {avg_rmse:.4f} m")
        print(f"  AbsRel:         {avg_absrel:.4f}")
        print(f"  delta < 1.25:   {avg_delta*100:.1f}%")
    else:
        print("\nDepth: No DA2 teacher predictions available.")

    if all_seg_metrics:
        avg_miou = np.mean([m["miou"] for m in all_seg_metrics])
        print(f"\nSegmentation (Student vs YOLO+SAM2):")
        print(f"  mIoU:           {avg_miou*100:.1f}%")

        avg_per_class = np.nanmean(
            [m["per_class_iou"] for m in all_seg_metrics], axis=0)
        class_names = cfg.CLASS_NAMES
        print(f"  Per-class IoU:")
        for i, (name, iou) in enumerate(zip(class_names, avg_per_class)):
            if not np.isnan(iou):
                print(f"    {name:12s}: {iou*100:.1f}%")
    else:
        print("\nSegmentation: No SAM2 teacher predictions available.")

    print(f"\n{'='*55}")


if __name__ == "__main__":
    main()
