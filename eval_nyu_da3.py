#!/usr/bin/env python3
"""
Evaluate DA3 depth models on NYU Depth V2 val split against GT depth.

Produces the DA3-Small teacher row for Table VII (tab:distillation).
For relative models (DA3-SMALL/BASE/LARGE): per-image median scaling.

Usage (HPC):
    python eval_nyu_da3.py \
        --model-id depth-anything/DA3-SMALL --relative \
        --data-root $SCRATCH/nyu_depth_v2

    python eval_nyu_da3.py \
        --model-id depth-anything/DA3METRIC-LARGE \
        --data-root $SCRATCH/nyu_depth_v2
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

NYU_FX = 518.8579
NYU_FY = 519.4696


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

    prediction = model.inference([str(img_path)])
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
        return pred, 1.0
    scale = float(np.median(g_valid) / np.median(p_valid))
    return pred * scale, scale


def compute_depth_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(p) == 0:
        return {}
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))
    abs_rel = float(np.mean(np.abs(p - g) / (g + 1e-8)))
    ratio = np.maximum(p / (g + 1e-8), g / (p + 1e-8))
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)
    delta_125_2 = float(np.mean(ratio < 1.25 ** 2) * 100.0)
    delta_125_3 = float(np.mean(ratio < 1.25 ** 3) * 100.0)
    return {
        "rmse": rmse, "mae": mae, "abs_rel": abs_rel,
        "delta_125": delta_125, "delta_125_2": delta_125_2,
        "delta_125_3": delta_125_3, "n_pixels": int(len(p)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DA3 depth on NYU Depth V2 vs GT")
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID "
                             "(e.g. depth-anything/DA3-SMALL)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path containing nyu_cache/ directory "
                             "(e.g. $SCRATCH/nyu_depth_v2)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--relative", action="store_true",
                        help="Model outputs relative depth; applies "
                             "per-image median scaling")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "train", "all"],
                        help="Which split to evaluate (default: val)")
    parser.add_argument("--min-depth", type=float, default=1e-3,
                        help="Min valid GT depth in meters")
    parser.add_argument("--max-depth", type=float, default=10.0,
                        help="Max valid GT depth in meters")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Results directory (default: "
                             "<data-root>/nyu_da3_eval)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cache_dir = Path(args.data_root) / "nyu_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"nyu_cache not found at {cache_dir}. Run training first "
            f"to download NYU data, or check --data-root path."
        )

    rgb_dir = cache_dir / "rgb"
    depth_dir = cache_dir / "depth"

    if args.split == "all":
        n_images = len(list(rgb_dir.glob("*.png")))
        indices = list(range(n_images))
        split_name = "all"
    else:
        split_file = cache_dir / f"{args.split}_indices.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file {split_file} not found. "
                f"Available: {list(cache_dir.glob('*_indices.txt'))}"
            )
        with open(split_file) as f:
            indices = [int(line.strip()) for line in f]
        split_name = args.split

    focal = (NYU_FX + NYU_FY) / 2.0
    mode_str = "RELATIVE (per-image median scaling)" if args.relative \
        else f"METRIC (focal={focal:.2f})"
    model_short = args.model_id.split("/")[-1]

    print(f"{'=' * 70}")
    print(f"  DA3 NYU Depth V2 Evaluation")
    print(f"{'=' * 70}")
    print(f"  Model:      {args.model_id}")
    print(f"  Mode:       {mode_str}")
    print(f"  Device:     {device}")
    print(f"  Split:      {split_name} ({len(indices)} images)")
    print(f"  Depth range:[{args.min_depth}, {args.max_depth}] m")
    print(f"  Cache:      {cache_dir}")
    print()

    print(f"Loading {args.model_id}...")
    model = load_da3_model(args.model_id, device)
    print("Model loaded.\n")

    per_image = []
    latencies = []

    for idx in tqdm(indices, desc=f"{model_short} NYU eval"):
        stem = f"{idx:05d}"
        rgb_path = rgb_dir / f"{stem}.png"
        depth_path = depth_dir / f"{stem}.npy"

        if not rgb_path.exists() or not depth_path.exists():
            print(f"WARNING: missing {stem}, skipping")
            continue

        gt_depth = np.load(depth_path).astype(np.float32)

        t0 = time.time()
        pred_d = infer_single(model, str(rgb_path), focal, args.relative)
        latencies.append(time.time() - t0)

        if pred_d.shape != gt_depth.shape:
            pred_pil = Image.fromarray(pred_d)
            pred_pil = pred_pil.resize(
                (gt_depth.shape[1], gt_depth.shape[0]), Image.BILINEAR)
            pred_d = np.array(pred_pil, dtype=np.float32)

        valid = ((gt_depth >= args.min_depth) &
                 (gt_depth <= args.max_depth))

        scale = 1.0
        if args.relative:
            pred_d, scale = median_scale_align(pred_d, gt_depth, valid)

        metrics = compute_depth_metrics(pred_d, gt_depth, valid)
        metrics["index"] = idx
        metrics["stem"] = stem
        metrics["scale"] = scale
        per_image.append(metrics)

    n = len(per_image)
    if n == 0:
        print("ERROR: no images evaluated")
        return

    rmse_vals = np.array([m["rmse"] for m in per_image])
    mae_vals = np.array([m["mae"] for m in per_image])
    abs_rel_vals = np.array([m["abs_rel"] for m in per_image])
    d125_vals = np.array([m["delta_125"] for m in per_image])
    d125_2_vals = np.array([m["delta_125_2"] for m in per_image])
    d125_3_vals = np.array([m["delta_125_3"] for m in per_image])

    def stat(arr):
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se = std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
        ci95 = 1.96 * se
        return {"mean": mean, "std": std, "se": se, "ci95": ci95,
                "min": float(np.min(arr)), "max": float(np.max(arr))}

    results = {
        "model_id": args.model_id,
        "model_short": model_short,
        "mode": "relative" if args.relative else "metric",
        "split": split_name,
        "n_images": n,
        "depth_range": [args.min_depth, args.max_depth],
        "rmse": stat(rmse_vals),
        "mae": stat(mae_vals),
        "abs_rel": stat(abs_rel_vals),
        "delta_125": stat(d125_vals),
        "delta_125_2": stat(d125_2_vals),
        "delta_125_3": stat(d125_3_vals),
        "mean_latency_ms": float(np.mean(latencies) * 1000) if latencies else 0,
        "fps": float(1.0 / np.mean(latencies)) if latencies else 0,
    }

    print()
    print(f"{'=' * 70}")
    print(f"  Results: {model_short} on NYU {split_name} "
          f"({n} images)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':20s} {'Mean':>10s} {'± Std':>12s} "
          f"{'95% CI':>12s}")
    print(f"  {'-' * 56}")
    for name, key, unit in [
        ("RMSE", "rmse", " m"),
        ("MAE", "mae", " m"),
        ("AbsRel", "abs_rel", ""),
        ("δ<1.25", "delta_125", "%"),
        ("δ<1.25²", "delta_125_2", "%"),
        ("δ<1.25³", "delta_125_3", "%"),
    ]:
        s = results[key]
        mean_s = f"{s['mean']:.4f}{unit}"
        std_s = f"± {s['std']:.4f}"
        ci_s = f"± {s['ci95']:.4f}"
        print(f"  {name:20s} {mean_s:>10s} {std_s:>12s} {ci_s:>12s}")

    if latencies:
        print(f"\n  Latency: {results['mean_latency_ms']:.0f} ms/img "
              f"({results['fps']:.1f} FPS)")

    print(f"\n{'=' * 70}")
    print(f"  LaTeX row for Table VII (tab:distillation)")
    print(f"{'=' * 70}")
    r = results["rmse"]["mean"]
    print(f"  DA3-Small & ---  & {r:.3f}\\,m & --- & --- "
          f"& Teacher (rel.~depth) \\\\")
    print(f"  DA3-Small & ---  & {r:.3f} $\\pm$ "
          f"{results['rmse']['ci95']:.3f}\\,m & --- & --- "
          f"& Teacher (rel.~depth) \\\\")

    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.data_root) / "nyu_da3_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / f"nyu_da3_{model_short.lower()}_{split_name}.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Aggregate results: {json_out}")

    csv_out = out_dir / f"nyu_da3_{model_short.lower()}_{split_name}_per_image.csv"
    with open(csv_out, "w") as f:
        f.write("index,stem,rmse,mae,abs_rel,delta_125,delta_125_2,"
                "delta_125_3,scale,n_pixels\n")
        for m in per_image:
            f.write(f"{m['index']},{m['stem']},{m['rmse']:.6f},"
                    f"{m['mae']:.6f},{m['abs_rel']:.6f},"
                    f"{m['delta_125']:.2f},{m['delta_125_2']:.2f},"
                    f"{m['delta_125_3']:.2f},{m['scale']:.6f},"
                    f"{m['n_pixels']}\n")
    print(f"  Per-image CSV:     {csv_out}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
