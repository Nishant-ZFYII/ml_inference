#!/usr/bin/env python3
"""
Evaluate student model against DA3-Metric-Large depth and YOLO+SAM2
segmentation teacher labels (Table IV in the paper).

Computes:
  Depth:  RMSE, MAE, AbsRel, delta < 1.25
  Seg:    mIoU, per-class IoU

Usage:
    python eval_distillation.py \
        --checkpoint best.pt \
        --manifest  $SCRATCH/nyu_teacher_data/manifest.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import Config
from models.student import build_student


def load_model(checkpoint_path: str, device: torch.device, cfg: Config):
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def resize_np(arr, h, w, order=1):
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    mode = Image.NEAREST if order == 0 else Image.BILINEAR
    return np.array(Image.fromarray(arr).resize((w, h), mode), dtype=arr.dtype)


def compute_depth_metrics(pred, gt, mask):
    """Compute depth metrics on valid pixels."""
    p, g = pred[mask], gt[mask]
    if len(p) == 0:
        return {}

    rmse = np.sqrt(np.mean((p - g) ** 2))
    mae = np.mean(np.abs(p - g))
    abs_rel = np.mean(np.abs(p - g) / (g + 1e-8))

    ratio = np.maximum(p / (g + 1e-8), g / (p + 1e-8))
    delta_125 = np.mean(ratio < 1.25) * 100.0

    return {
        "rmse": rmse,
        "mae": mae,
        "abs_rel": abs_rel,
        "delta_125": delta_125,
    }


def compute_seg_metrics(pred, gt, num_classes):
    """Compute per-class IoU and mIoU."""
    ious = {}
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union > 0:
            ious[c] = intersection / union
    miou = np.mean(list(ious.values())) if ious else 0.0
    return miou, ious


def main():
    parser = argparse.ArgumentParser(description="Evaluate student vs teacher")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device, cfg)

    manifest_dir = Path(args.manifest).parent
    with open(args.manifest) as f:
        samples = [json.loads(line) for line in f]

    print(f"Evaluating {len(samples)} samples...")

    all_depth = {"rmse": [], "mae": [], "abs_rel": [], "delta_125": []}
    all_seg_pred = []
    all_seg_gt = []

    for entry in tqdm(samples, desc="Evaluating"):
        rgb_path = manifest_dir / entry["rgb"]
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = rgb.resize((cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT), Image.BILINEAR)
        rgb_np = np.array(rgb, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_depth, pred_seg = model(rgb_t)

        pred_d = pred_depth.squeeze().cpu().numpy()
        pred_s = pred_seg.argmax(dim=1).squeeze().cpu().numpy()

        # Depth: compare vs DA3 teacher
        da3_path = entry.get("da3_depth")
        if da3_path:
            da3_full = manifest_dir / da3_path
            if da3_full.exists():
                da3_depth = np.load(da3_full).astype(np.float32)
                da3_depth = resize_np(da3_depth, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH)
                mask = da3_depth > 0
                metrics = compute_depth_metrics(pred_d, da3_depth, mask)
                for k in all_depth:
                    if k in metrics:
                        all_depth[k].append(metrics[k])

        # Seg: compare vs YOLO+SAM2 teacher
        seg_path = entry.get("sam2_seg")
        if seg_path:
            seg_full = manifest_dir / seg_path
            if seg_full.exists():
                teacher_seg = np.load(seg_full).astype(np.int64)
                teacher_seg = resize_np(
                    teacher_seg.astype(np.float32),
                    cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, order=0
                ).astype(np.int64)
                valid = teacher_seg < cfg.NUM_CLASSES
                if valid.any():
                    all_seg_pred.append(pred_s[valid])
                    all_seg_gt.append(teacher_seg[valid])

    # Aggregate depth metrics
    print()
    print("=" * 60)
    print("Table IV: Student vs Teacher Distillation Quality")
    print("=" * 60)

    print(f"\nDepth (Student vs DA3-Metric-Large):")
    for k in ["rmse", "mae", "abs_rel", "delta_125"]:
        vals = all_depth[k]
        if vals:
            mean_val = np.mean(vals)
            label = {
                "rmse": "RMSE",
                "mae": "MAE",
                "abs_rel": "AbsRel",
                "delta_125": "delta < 1.25",
            }[k]
            unit = "%" if k == "delta_125" else "m"
            if k == "abs_rel":
                unit = ""
                print(f"  {label:20s}: {mean_val:.4f}")
            else:
                print(f"  {label:20s}: {mean_val:.4f} {unit}")

    # Aggregate seg metrics
    if all_seg_pred:
        combined_pred = np.concatenate(all_seg_pred)
        combined_gt = np.concatenate(all_seg_gt)
        miou, per_class = compute_seg_metrics(
            combined_pred, combined_gt, cfg.NUM_CLASSES
        )
        print(f"\nSegmentation (Student vs YOLO+SAM2):")
        print(f"  {'mIoU':20s}: {miou*100:.1f}%")
        print(f"\n  Per-class IoU:")
        for c in range(cfg.NUM_CLASSES):
            name = cfg.CLASS_NAMES[c] if c < len(cfg.CLASS_NAMES) else f"class_{c}"
            iou = per_class.get(c, float("nan"))
            print(f"    {name:15s}: {iou*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
