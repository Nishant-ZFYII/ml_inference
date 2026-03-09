#!/usr/bin/env python3
"""
Run DA3-Small on extracted glass corridor frames and create
side-by-side comparison images: RGB | Sensor Depth | DA3 Depth.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def run_da3_on_frames(frames_dir: str, output_dir: str):
    frames = Path(frames_dir)
    out = Path(output_dir)
    (out / "da3_depth").mkdir(parents=True, exist_ok=True)
    (out / "da3_viz").mkdir(parents=True, exist_ok=True)
    (out / "comparison").mkdir(parents=True, exist_ok=True)

    with open(frames / "manifest.json") as f:
        manifest = json.load(f)

    print(f"Loading DA3-Small model...")
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1,
    )
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    results = []

    for i, entry in enumerate(manifest):
        fname = entry["frame"]
        rgb_path = frames / "rgb" / f"{fname}.png"
        depth_path = frames / "depth" / f"{fname}.npy"
        sensor_viz_path = frames / "depth_viz" / f"{fname}.png"

        rgb_pil = Image.open(rgb_path).convert("RGB")

        da3_result = pipe(rgb_pil)
        da3_raw = np.array(da3_result["depth"], dtype=np.float32)

        sensor_depth = np.load(depth_path)
        valid = (sensor_depth > 0.1) & (sensor_depth < 5.0)

        if valid.sum() > 100:
            da3_resized = cv2.resize(
                da3_raw, (sensor_depth.shape[1], sensor_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR)

            sensor_median = np.median(sensor_depth[valid])
            da3_at_valid = da3_resized[valid]
            da3_median = np.median(da3_at_valid)

            if da3_median > 0:
                scale = sensor_median / da3_median
                da3_metric = da3_resized * scale
            else:
                da3_metric = da3_resized
                scale = 1.0

            rmse = float(np.sqrt(np.mean(
                (da3_metric[valid] - sensor_depth[valid]) ** 2)))
            mae = float(np.mean(
                np.abs(da3_metric[valid] - sensor_depth[valid])))
            abs_rel = float(np.mean(
                np.abs(da3_metric[valid] - sensor_depth[valid])
                / (sensor_depth[valid] + 1e-8)))
        else:
            da3_metric = da3_raw
            rmse = mae = abs_rel = -1.0
            scale = 1.0

        np.save(out / "da3_depth" / f"{fname}.npy", da3_metric)

        da3_viz = np.zeros_like(da3_metric, dtype=np.uint8)
        da3_valid = (da3_metric > 0.05) & (da3_metric < 8.0)
        if da3_valid.any():
            dmin = da3_metric[da3_valid].min()
            dmax = da3_metric[da3_valid].max()
            if dmax > dmin:
                da3_viz[da3_valid] = (
                    255 * (1.0 - (da3_metric[da3_valid] - dmin)
                           / (dmax - dmin))).astype(np.uint8)
        da3_color = cv2.applyColorMap(da3_viz, cv2.COLORMAP_INFERNO)
        da3_color[~da3_valid] = [0, 0, 0]
        cv2.imwrite(str(out / "da3_viz" / f"{fname}.png"), da3_color)

        # Create side-by-side comparison
        rgb_cv = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
        sensor_viz = cv2.imread(str(sensor_viz_path))

        target_h = 360
        aspect = rgb_cv.shape[1] / rgb_cv.shape[0]
        target_w = int(target_h * aspect)

        rgb_resized = cv2.resize(rgb_cv, (target_w, target_h))
        sensor_resized = cv2.resize(sensor_viz, (target_w, target_h))
        da3_resized_viz = cv2.resize(da3_color, (target_w, target_h))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for img, label in [(rgb_resized, "RGB"),
                           (sensor_resized, f"Sensor (valid={entry['depth_valid_pct']:.0f}%)"),
                           (da3_resized_viz, f"DA3 (RMSE={rmse:.3f}m)" if rmse > 0 else "DA3")]:
            cv2.putText(img, label, (10, 30), font, 0.8, (255, 255, 255), 2)
            cv2.putText(img, label, (10, 30), font, 0.8, (0, 0, 0), 1)

        comparison = np.hstack([rgb_resized, sensor_resized, da3_resized_viz])
        cv2.imwrite(str(out / "comparison" / f"{fname}.png"), comparison)

        entry_result = {
            **entry,
            "da3_rmse_m": round(rmse, 4) if rmse > 0 else None,
            "da3_mae_m": round(mae, 4) if mae > 0 else None,
            "da3_abs_rel": round(abs_rel, 4) if abs_rel > 0 else None,
            "da3_scale": round(scale, 4),
        }
        results.append(entry_result)

        print(f"  [{i}] {fname}  RMSE={rmse:.3f}m  MAE={mae:.3f}m  "
              f"scale={scale:.3f}  sensor_valid={entry['depth_valid_pct']:.0f}%")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return super().default(obj)

    with open(out / "da3_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    valid_results = [r for r in results if r["da3_rmse_m"] is not None]
    if valid_results:
        avg_rmse = np.mean([r["da3_rmse_m"] for r in valid_results])
        avg_mae = np.mean([r["da3_mae_m"] for r in valid_results])
        avg_abs_rel = np.mean([r["da3_abs_rel"] for r in valid_results])
        avg_sensor_valid = np.mean([r["depth_valid_pct"] for r in valid_results])

        print(f"\n{'='*60}")
        print(f"DA3 Glass Corridor Summary ({len(valid_results)} frames)")
        print(f"  Avg RMSE:        {avg_rmse:.4f} m")
        print(f"  Avg MAE:         {avg_mae:.4f} m")
        print(f"  Avg AbsRel:      {avg_abs_rel:.4f}")
        print(f"  Avg sensor valid: {avg_sensor_valid:.1f}%")
        print(f"  Avg sensor dead:  {100-avg_sensor_valid:.1f}%")
        print(f"{'='*60}")

    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    run_da3_on_frames(
        "/home/nishant/maps/glass_corridor_frames",
        "/home/nishant/maps/glass_corridor_frames",
    )
