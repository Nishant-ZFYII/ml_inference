#!/usr/bin/env python3
"""
Evaluate DA3 depth predictions directly against real sensor depth from the
Orbbec Femto Bolt in the deployment corridor.

This answers: what is DA3's actual RMSE in the corridor?
That number determines whether direct Jetson deployment is viable.

For DA3METRIC models: uses focal * raw / 300 for metric conversion.
For DA3-SMALL/BASE/LARGE (relative): applies median-scaling alignment
to show best-case accuracy if properly calibrated.

Usage:
    python eval_corridor_da3.py \
        --model-id depth-anything/DA3METRIC-LARGE \
        --manifest $SCRATCH/corridor_eval_data/manifest.jsonl

    python eval_corridor_da3.py \
        --model-id depth-anything/DA3-SMALL --relative \
        --manifest $SCRATCH/corridor_eval_data/manifest.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_da3_model(model_id: str, device: torch.device):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    model.eval()
    return model


def infer_single(model, img_path: str, focal: float,
                 is_relative: bool) -> np.ndarray:
    rgb = Image.open(img_path).convert("RGB")
    orig_w, orig_h = rgb.size

    prediction = model.inference([img_path])
    raw_depth = prediction.depth[0]

    if is_relative:
        depth_out = raw_depth.astype(np.float32)
    else:
        depth_out = (focal * raw_depth / 300.0).astype(np.float32)

    if depth_out.shape != (orig_h, orig_w):
        depth_pil = Image.fromarray(depth_out)
        depth_pil = depth_pil.resize((orig_w, orig_h), Image.BILINEAR)
        depth_out = np.array(depth_pil, dtype=np.float32)

    return depth_out


def median_scale_align(pred, gt, valid_mask):
    """Scale relative depth so its median matches GT median on valid pixels."""
    p_valid = pred[valid_mask]
    g_valid = gt[valid_mask]
    if len(p_valid) == 0 or np.median(p_valid) < 1e-8:
        return pred
    scale = np.median(g_valid) / np.median(p_valid)
    return pred * scale


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
        description="Evaluate DA3 depth vs corridor sensor depth")
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID "
                             "(e.g. depth-anything/DA3METRIC-LARGE)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--relative", action="store_true",
                        help="Model outputs relative depth (DA3-SMALL/BASE/LARGE). "
                             "Applies per-frame median-scaling to align to sensor.")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    manifest_dir = Path(args.manifest).parent

    focal = 0.0
    if not args.relative:
        intr_path = manifest_dir / "intrinsics.json"
        if intr_path.exists():
            with open(intr_path) as f:
                intr = json.load(f)
            focal = (intr["fx"] + intr["fy"]) / 2.0
            print(f"Intrinsics: fx={intr['fx']:.2f} fy={intr['fy']:.2f} → "
                  f"focal={focal:.2f}")
        else:
            focal = 605.0
            print(f"WARNING: intrinsics.json not found, using default focal={focal}")

    mode_str = "RELATIVE (median-scaled)" if args.relative else "METRIC"
    print(f"Model: {args.model_id}")
    print(f"Mode: {mode_str}")
    print(f"Device: {device}")
    print(f"Valid depth range: [{args.min_depth}, {args.max_depth}] m")

    print(f"Loading {args.model_id}...")
    model = load_da3_model(args.model_id, device)
    print("DA3 model loaded.")

    with open(args.manifest) as f:
        samples = [json.loads(line) for line in f]
    print(f"Samples: {len(samples)}")

    all_pred = []
    all_gt = []
    all_valid = []
    total_sensor_zero = 0
    total_sensor_far = 0
    total_pixels = 0
    latencies = []

    for entry in tqdm(samples, desc="DA3 corridor eval"):
        rgb_path = str(manifest_dir / entry["rgb"])
        depth_path = manifest_dir / entry["sensor_depth"]

        t0 = time.time()
        pred_d = infer_single(model, rgb_path, focal, args.relative)
        latencies.append(time.time() - t0)

        sensor_depth = np.load(depth_path).astype(np.float32)

        if pred_d.shape != sensor_depth.shape:
            pred_pil = Image.fromarray(pred_d)
            pred_pil = pred_pil.resize(
                (sensor_depth.shape[1], sensor_depth.shape[0]), Image.BILINEAR)
            pred_d = np.array(pred_pil, dtype=np.float32)

        valid = (sensor_depth >= args.min_depth) & (sensor_depth <= args.max_depth)

        if args.relative:
            pred_d = median_scale_align(pred_d, sensor_depth, valid)

        sensor_zero = sensor_depth == 0
        sensor_far = sensor_depth > args.max_depth

        total_sensor_zero += sensor_zero.sum()
        total_sensor_far += sensor_far.sum()
        total_pixels += sensor_depth.size

        all_pred.append(pred_d)
        all_gt.append(sensor_depth)
        all_valid.append(valid)

    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)
    all_valid = np.array(all_valid)

    overall = compute_depth_metrics(all_pred, all_gt, all_valid)

    H = all_pred.shape[1]

    model_short = args.model_id.split("/")[-1]
    print()
    print("=" * 75)
    print(f"Corridor Depth Evaluation — {model_short} "
          f"({'Relative+MedianScale' if args.relative else 'Metric'})")
    print("=" * 75)
    print(f"Model: {args.model_id}")
    print(f"Frames: {len(samples)}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Valid pixels (sensor {args.min_depth}-{args.max_depth}m): "
          f"{all_valid.sum():,} ({all_valid.sum()*100/total_pixels:.1f}%)")
    print(f"Sensor zero (dead): {total_sensor_zero:,} "
          f"({total_sensor_zero*100/total_pixels:.1f}%)")
    print(f"Sensor far (>{args.max_depth}m): {total_sensor_far:,} "
          f"({total_sensor_far*100/total_pixels:.1f}%)")
    if latencies:
        mean_lat = np.mean(latencies)
        print(f"Mean latency: {mean_lat*1000:.0f} ms  "
              f"({1.0/mean_lat:.1f} FPS)")
    print()

    print("--- Overall (valid pixels) ---")
    print_metrics("All valid", overall)
    print()

    bins = [(0.3, 1.0, "Near (0.3-1.0m)"),
            (1.0, 2.0, "Mid  (1.0-2.0m)"),
            (2.0, 4.0, "Far  (2.0-4.0m)")]

    print("--- Per-depth-bin breakdown ---")
    for lo, hi, label in bins:
        bin_mask = all_valid & (all_gt >= lo) & (all_gt < hi)
        metrics = compute_depth_metrics(all_pred, all_gt, bin_mask)
        print_metrics(label, metrics)
    print()

    split_row = H * 2 // 3
    top_mask = np.zeros_like(all_valid)
    top_mask[:, :split_row, :] = True
    bot_mask = np.zeros_like(all_valid)
    bot_mask[:, split_row:, :] = True

    print("--- Spatial split (top=walls/ceiling, bottom=floor) ---")
    top_valid = all_valid & top_mask
    bot_valid = all_valid & bot_mask
    print_metrics("Top 2/3 (walls/ceiling)",
                  compute_depth_metrics(all_pred, all_gt, top_valid))
    print_metrics("Bottom 1/3 (floor)",
                  compute_depth_metrics(all_pred, all_gt, bot_valid))
    print()

    failure_mask = (all_gt == 0) | (all_gt > args.max_depth)
    n_failure = failure_mask.sum()
    if n_failure > 0:
        failure_pred = all_pred[failure_mask]
        print("--- Sensor failure pixels (zero/far readings) ---")
        print(f"  Count: {n_failure:,} ({n_failure*100/total_pixels:.1f}%)")
        print(f"  DA3 prediction range: "
              f"{failure_pred.min():.3f} - {failure_pred.max():.3f} m")
        print(f"  DA3 prediction mean:  {failure_pred.mean():.3f} m, "
              f"median: {np.median(failure_pred):.3f} m")
        print(f"  (No GT available — these are where the sensor failed)")

    print()
    print("=" * 75)
    print("Reference — Student models on this corridor data:")
    print("  B1 (Iter 6): RMSE 2.777m  B2 (Iter 7b): RMSE 2.356m")
    print("Reference — NYU Depth V2 benchmarks:")
    print("  DA3-Large: RMSE ~0.250m   DA3-Small: RMSE ~0.296m")
    print("=" * 75)


if __name__ == "__main__":
    main()
