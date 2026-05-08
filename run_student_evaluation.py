#!/usr/bin/env python3
"""
Run student depth models on extracted frames and generate comparison outputs.

For each dataset:
  - Loads RGB frames and sensor depth
  - Runs each student model (v5, v6, v7) for depth prediction
  - Creates side-by-side comparison images: RGB | Sensor | StudentV5 | StudentV6 | StudentV7
  - Calculates per-frame and aggregate metrics (RMSE, MAE, AbsRel, delta<1.25)
  - Saves all results as JSON
"""
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.student import build_student


STUDENT_MODELS = {
    "v5_vivek": {
        "path": "hpc_outputs/best_depth_v5_vivek.pt",
        "label": "Student V5",
    },
    "v6": {
        "path": "hpc_outputs/best_depth_v6.pt",
        "label": "Student V6",
    },
    "v7": {
        "path": "hpc_outputs/best_depth_v7.pt",
        "label": "Student V7",
    },
}

INPUT_H, INPUT_W = 240, 320


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_student_model(model_key, device):
    info = STUDENT_MODELS[model_key]
    base = Path(__file__).parent
    ckpt_path = base / info["path"]

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = build_student(backbone="efficientvit_b1", pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()

    rmse = ckpt.get("best_depth_rmse", None)
    epoch = ckpt.get("epoch", None)
    print(f"  Loaded {info['label']}: epoch={epoch}, train_rmse={rmse:.4f}" if rmse else
          f"  Loaded {info['label']}: epoch={epoch}")
    return model, info["label"]


def colorize_depth(depth_m, valid_mask, vmin=0.0, vmax=4.0,
                   colormap=cv2.COLORMAP_INFERNO):
    h, w = depth_m.shape
    norm = np.zeros((h, w), dtype=np.uint8)
    if valid_mask.any():
        d = np.clip(depth_m[valid_mask], vmin, vmax)
        norm_vals = (255 * (1.0 - (d - vmin) / (vmax - vmin))).clip(0, 255)
        norm[valid_mask] = norm_vals.astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    colored[~valid_mask] = [40, 40, 40]
    return colored


def compute_metrics(pred_m, gt_m, valid_mask):
    """Compute depth metrics on valid pixels only."""
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

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "absrel": float(absrel),
        "delta1": float(delta1),
        "valid_pixels": int(valid_mask.sum()),
    }


def run_student_inference(model, rgb_bgr, device):
    """Run student model on a single RGB frame. Returns depth in meters."""
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input
    rgb_resized = cv2.resize(rgb, (INPUT_W, INPUT_H),
                             interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 1]
    tensor = torch.from_numpy(rgb_resized).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        depth_pred, seg_pred = model(tensor)

    depth_np = depth_pred.squeeze().cpu().numpy()
    return depth_np


def process_dataset(frames_dir, output_dir, models, device):
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    comp_dir = output_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for model_key in models:
        (output_dir / f"depth_{model_key}").mkdir(parents=True, exist_ok=True)

    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        frame_names = [m["frame"] for m in manifest]
    else:
        frame_names = sorted([p.stem for p in (frames_dir / "rgb").glob("*.png")])

    print(f"\nProcessing {len(frame_names)} frames from {frames_dir}")

    all_results = []

    for fi, fname in enumerate(frame_names):
        rgb_path = frames_dir / "rgb" / f"{fname}.png"
        depth_path = frames_dir / "depth" / f"{fname}.npy"

        if not rgb_path.exists() or not depth_path.exists():
            continue

        rgb_bgr = cv2.imread(str(rgb_path))
        sensor_depth = np.load(str(depth_path))

        orig_h, orig_w = sensor_depth.shape[:2]
        valid_sensor = (sensor_depth > 0.1) & (sensor_depth < 5.0) & ~np.isnan(sensor_depth)

        # Downscale sensor depth to model resolution for metric comparison
        sensor_small = cv2.resize(sensor_depth, (INPUT_W, INPUT_H),
                                  interpolation=cv2.INTER_NEAREST)
        valid_small = (sensor_small > 0.1) & (sensor_small < 5.0) & ~np.isnan(sensor_small)

        frame_result = {
            "frame": fname,
            "sensor_dead_pct": round((~valid_sensor).sum() * 100.0 / sensor_depth.size, 1),
        }

        # Run each student model
        student_depths = {}
        for model_key, (model, label) in models.items():
            pred = run_student_inference(model, rgb_bgr, device)
            student_depths[model_key] = pred

            # Save raw prediction
            np.save(str(output_dir / f"depth_{model_key}" / f"{fname}.npy"), pred)

            metrics = compute_metrics(pred, sensor_small, valid_small)
            frame_result[model_key] = metrics

        all_results.append(frame_result)

        # Create comparison image
        panel_h = 240
        rgb_small = cv2.resize(rgb_bgr, (INPUT_W, INPUT_H))

        panels = [rgb_small]
        labels = ["RGB"]

        # Sensor depth panel
        sensor_color = colorize_depth(sensor_small, valid_small)
        dead_pct = frame_result["sensor_dead_pct"]
        panels.append(sensor_color)
        labels.append(f"Sensor ({dead_pct:.0f}% dead)")

        # Student model panels
        for model_key in models:
            pred = student_depths[model_key]
            pred_valid = (pred > 0.01) & (pred < 8.0)
            pred_color = colorize_depth(pred, pred_valid)
            rmse = frame_result[model_key]["rmse"]
            panels.append(pred_color)
            labels.append(f"{models[model_key][1]} (RMSE={rmse:.2f})")

        # Compose side-by-side
        gap = 2
        n = len(panels)
        total_w = INPUT_W * n + gap * (n - 1)
        canvas = np.ones((INPUT_H, total_w, 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (panel, label) in enumerate(zip(panels, labels)):
            x0 = i * (INPUT_W + gap)
            canvas[:, x0:x0 + INPUT_W] = panel
            # Label
            cv2.rectangle(canvas, (x0, 0), (x0 + INPUT_W, 28), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x0 + 4, 20), font, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(comp_dir / f"{fname}.png"), canvas,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        if (fi + 1) % 20 == 0:
            print(f"  Processed {fi + 1}/{len(frame_names)}")

    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(all_results)} frames")
    print(f"{'='*60}")

    for model_key in models:
        label = models[model_key][1]
        rmses = [r[model_key]["rmse"] for r in all_results
                 if not np.isnan(r[model_key]["rmse"])]
        maes = [r[model_key]["mae"] for r in all_results
                if not np.isnan(r[model_key]["mae"])]
        absrels = [r[model_key]["absrel"] for r in all_results
                   if not np.isnan(r[model_key]["absrel"])]
        d1s = [r[model_key]["delta1"] for r in all_results
               if not np.isnan(r[model_key]["delta1"])]

        print(f"\n{label}:")
        if rmses:
            print(f"  RMSE:   {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
            print(f"  MAE:    {np.mean(maes):.4f} +/- {np.std(maes):.4f}")
            print(f"  AbsRel: {np.mean(absrels):.4f} +/- {np.std(absrels):.4f}")
            print(f"  d<1.25: {np.mean(d1s):.1f}% +/- {np.std(d1s):.1f}%")
        else:
            print(f"  No valid results")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", help="Directory with rgb/ and depth/ subdirs")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--models", nargs='+',
                        default=list(STUDENT_MODELS.keys()),
                        help="Which student models to run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load models
    loaded_models = {}
    for key in args.models:
        if key not in STUDENT_MODELS:
            print(f"WARNING: Unknown model key '{key}', skipping")
            continue
        model, label = load_student_model(key, device)
        loaded_models[key] = (model, label)

    process_dataset(args.frames_dir, args.output_dir, loaded_models, device)


if __name__ == "__main__":
    main()
