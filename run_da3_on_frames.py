#!/usr/bin/env python3
"""
Run DA3-Small baseline on extracted frames and save depth predictions.
Creates same output structure as student evaluation for comparison.
"""
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import pipeline


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_metrics(pred_m, gt_m, valid_mask):
    if valid_mask.sum() < 100:
        return {"rmse": float('nan'), "mae": float('nan'),
                "absrel": float('nan'), "delta1": float('nan'),
                "valid_pixels": int(valid_mask.sum())}

    p = pred_m[valid_mask]
    g = gt_m[valid_mask]

    diff = np.abs(p - g)
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(diff)
    absrel = np.mean(diff / np.clip(g, 0.01, None))
    ratio = np.maximum(p / np.clip(g, 0.01, None),
                       g / np.clip(p, 0.01, None))
    delta1 = (ratio < 1.25).mean() * 100.0

    return {"rmse": float(rmse), "mae": float(mae),
            "absrel": float(absrel), "delta1": float(delta1),
            "valid_pixels": int(valid_mask.sum())}


def median_scale(pred, gt, valid_mask):
    """Scale DA3 relative depth to metric using median scaling."""
    if valid_mask.sum() < 50:
        return pred
    scale = np.median(gt[valid_mask]) / np.median(pred[valid_mask])
    return pred * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    da3_dir = output_dir / "depth_da3"
    da3_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DA3 model: {args.model}")
    pipe = pipeline("depth-estimation", model=args.model, device=args.device)
    print("  Model loaded")

    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        frame_names = [m["frame"] for m in manifest]
    else:
        frame_names = sorted([p.stem for p in (frames_dir / "rgb").glob("*.png")])

    print(f"\nProcessing {len(frame_names)} frames")

    results = []

    for fi, fname in enumerate(frame_names):
        rgb_path = frames_dir / "rgb" / f"{fname}.png"
        depth_path = frames_dir / "depth" / f"{fname}.npy"

        if not rgb_path.exists():
            continue

        from PIL import Image
        rgb_pil = Image.open(str(rgb_path)).convert("RGB")

        pred = pipe(rgb_pil)
        da3_raw = np.array(pred["depth"]).astype(np.float32)

        frame_result = {"frame": fname}

        if depth_path.exists():
            sensor_depth = np.load(str(depth_path))
            # Resize DA3 to match sensor depth
            da3_resized = cv2.resize(da3_raw,
                                     (sensor_depth.shape[1], sensor_depth.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
            valid = (sensor_depth > 0.1) & (sensor_depth < 5.0) & ~np.isnan(sensor_depth)
            da3_metric = median_scale(da3_resized, sensor_depth, valid)
            metrics = compute_metrics(da3_metric, sensor_depth, valid)
            frame_result["da3"] = metrics
            frame_result["sensor_dead_pct"] = round(
                (~valid).sum() * 100.0 / sensor_depth.size, 1)
            np.save(str(da3_dir / f"{fname}.npy"), da3_metric)
        else:
            np.save(str(da3_dir / f"{fname}.npy"), da3_raw)

        results.append(frame_result)

        if (fi + 1) % 20 == 0:
            print(f"  Processed {fi + 1}/{len(frame_names)}")

    with open(output_dir / "da3_results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Summary
    if results and "da3" in results[0]:
        rmses = [r["da3"]["rmse"] for r in results if "da3" in r and not np.isnan(r["da3"]["rmse"])]
        maes = [r["da3"]["mae"] for r in results if "da3" in r and not np.isnan(r["da3"]["mae"])]
        absrels = [r["da3"]["absrel"] for r in results if "da3" in r and not np.isnan(r["da3"]["absrel"])]
        d1s = [r["da3"]["delta1"] for r in results if "da3" in r and not np.isnan(r["da3"]["delta1"])]

        print(f"\n{'='*60}")
        print(f"DA3 Baseline Summary: {len(results)} frames")
        print(f"{'='*60}")
        if rmses:
            print(f"  RMSE:   {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
            print(f"  MAE:    {np.mean(maes):.4f} +/- {np.std(maes):.4f}")
            print(f"  AbsRel: {np.mean(absrels):.4f} +/- {np.std(absrels):.4f}")
            print(f"  d<1.25: {np.mean(d1s):.1f}% +/- {np.std(d1s):.1f}%")


if __name__ == "__main__":
    main()
