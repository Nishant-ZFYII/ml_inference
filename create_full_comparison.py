#!/usr/bin/env python3
"""
Create full comparison images: RGB | Sensor | DA3 (teacher) | Student V5 | V6 | V7
for both glass corridor and bag 213831 datasets.
Also generates a final metrics summary table.
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

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


def colorize_depth(depth_m, valid_mask, vmin=None, vmax=None,
                   colormap=cv2.COLORMAP_INFERNO):
    """Colorize depth map. If vmin/vmax are None, auto-range from data."""
    h, w = depth_m.shape
    if vmin is None or vmax is None:
        if valid_mask.any():
            vd = depth_m[valid_mask]
            vmin = vmin if vmin is not None else max(0.0, float(np.percentile(vd, 2)))
            vmax = vmax if vmax is not None else float(np.percentile(vd, 98))
            vmax = max(vmax, vmin + 0.3)
        else:
            vmin, vmax = 0.0, 4.0
    norm = np.zeros((h, w), dtype=np.uint8)
    if valid_mask.any():
        d = np.clip(depth_m[valid_mask], vmin, vmax)
        norm_vals = (255 * (1.0 - (d - vmin) / (vmax - vmin))).clip(0, 255)
        norm[valid_mask] = norm_vals.astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    colored[~valid_mask] = [40, 40, 40]
    return colored


def compute_metrics(pred, gt, valid):
    if valid.sum() < 100:
        return {"rmse": float('nan'), "mae": float('nan'),
                "absrel": float('nan'), "delta1": float('nan')}
    p, g = pred[valid], gt[valid]
    diff = np.abs(p - g)
    ratio = np.maximum(p / np.clip(g, 0.01, None),
                       g / np.clip(p, 0.01, None))
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(diff)),
        "absrel": float(np.mean(diff / np.clip(g, 0.01, None))),
        "delta1": float((ratio < 1.25).mean() * 100),
    }


def process_dataset(dataset_name, frames_dir, da3_depth_dir,
                    student_results_dir, output_dir):
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    comp_dir = output_dir / "full_comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        frame_names = [m["frame"] for m in manifest]
    else:
        frame_names = sorted([p.stem for p in (frames_dir / "rgb").glob("*.png")])

    student_dirs = {
        "v5_vivek": Path(student_results_dir) / "depth_v5_vivek",
        "v6": Path(student_results_dir) / "depth_v6",
        "v7": Path(student_results_dir) / "depth_v7",
    }
    da3_dir = Path(da3_depth_dir)

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} ({len(frame_names)} frames)")
    print(f"{'='*70}")

    all_metrics = {"da3": [], "v5_vivek": [], "v6": [], "v7": []}

    for fi, fname in enumerate(frame_names):
        rgb_path = frames_dir / "rgb" / f"{fname}.png"
        depth_path = frames_dir / "depth" / f"{fname}.npy"

        if not rgb_path.exists() or not depth_path.exists():
            continue

        rgb_bgr = cv2.imread(str(rgb_path))
        sensor_depth = np.load(str(depth_path))

        # Resize to common size
        rgb_small = cv2.resize(rgb_bgr, (INPUT_W, INPUT_H))
        sensor_small = cv2.resize(sensor_depth, (INPUT_W, INPUT_H),
                                  interpolation=cv2.INTER_NEAREST)
        valid = (sensor_small > 0.1) & (sensor_small < 5.0) & ~np.isnan(sensor_small)
        dead_pct = (~valid).sum() * 100.0 / sensor_small.size

        # Collect all depth arrays first to compute shared colormap range
        depth_arrays = {}
        depth_arrays["sensor"] = (sensor_small, valid)

        da3_depth = None
        da3_path = da3_dir / f"{fname}.npy"
        if da3_path.exists():
            da3_depth = np.load(str(da3_path))
            if da3_depth.shape != (INPUT_H, INPUT_W):
                da3_depth = cv2.resize(da3_depth, (INPUT_W, INPUT_H),
                                       interpolation=cv2.INTER_LINEAR)
            da3_valid = (da3_depth > 0.05) & (da3_depth < 8.0)
            depth_arrays["da3"] = (da3_depth, da3_valid)

        student_data = {}
        for key, sdir in student_dirs.items():
            spath = sdir / f"{fname}.npy"
            if spath.exists():
                pred = np.load(str(spath))
                if pred.shape != (INPUT_H, INPUT_W):
                    pred = cv2.resize(pred, (INPUT_W, INPUT_H),
                                      interpolation=cv2.INTER_LINEAR)
                pred_valid = (pred > 0.01) & (pred < 8.0)
                student_data[key] = (pred, pred_valid)
                depth_arrays[key] = (pred, pred_valid)

        # Shared colormap range across all depth sources for this frame
        all_vals = []
        for _, (darr, dmask) in depth_arrays.items():
            if dmask.any():
                all_vals.append(darr[dmask])
        if all_vals:
            combined = np.concatenate(all_vals)
            shared_vmin = max(0.0, float(np.percentile(combined, 2)))
            shared_vmax = float(np.percentile(combined, 98))
            shared_vmax = max(shared_vmax, shared_vmin + 0.3)
        else:
            shared_vmin, shared_vmax = 0.0, 4.0

        panels = [rgb_small]
        labels = ["RGB"]

        sensor_color = colorize_depth(sensor_small, valid,
                                      vmin=shared_vmin, vmax=shared_vmax)
        panels.append(sensor_color)
        labels.append(f"Sensor ({dead_pct:.0f}% dead)")

        if da3_depth is not None:
            da3_m = compute_metrics(da3_depth, sensor_small, valid)
            all_metrics["da3"].append(da3_m)
            da3_color = colorize_depth(da3_depth, da3_valid,
                                       vmin=shared_vmin, vmax=shared_vmax)
            panels.append(da3_color)
            labels.append(f"DA3 ({da3_m['rmse']:.2f})")
        else:
            panels.append(np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8))
            labels.append("DA3 (N/A)")

        for key, sdir in student_dirs.items():
            label_prefix = {"v5_vivek": "V5", "v6": "V6", "v7": "V7"}[key]
            if key in student_data:
                pred, pred_valid = student_data[key]
                m = compute_metrics(pred, sensor_small, valid)
                all_metrics[key].append(m)
                pred_color = colorize_depth(pred, pred_valid,
                                            vmin=shared_vmin, vmax=shared_vmax)
                panels.append(pred_color)
                labels.append(f"{label_prefix} ({m['rmse']:.2f})")
            else:
                panels.append(np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8))
                labels.append(f"{label_prefix} (N/A)")

        # Compose
        gap = 2
        n = len(panels)
        total_w = INPUT_W * n + gap * (n - 1)
        canvas = np.ones((INPUT_H, total_w, 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (panel, label) in enumerate(zip(panels, labels)):
            x0 = i * (INPUT_W + gap)
            canvas[:, x0:x0 + INPUT_W] = panel
            cv2.rectangle(canvas, (x0, 0), (x0 + INPUT_W, 28), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x0 + 4, 20), font, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(comp_dir / f"{fname}.png"), canvas,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        if (fi + 1) % 20 == 0:
            print(f"  Processed {fi + 1}/{len(frame_names)}")

    # Summary table
    print(f"\n{'Model':<12} {'RMSE':>8} {'MAE':>8} {'AbsRel':>8} {'d<1.25':>8}")
    print("-" * 48)
    model_names = {"da3": "DA3 (T)", "v5_vivek": "Stud V5",
                   "v6": "Stud V6", "v7": "Stud V7"}

    summary = {}
    for key in ["da3", "v5_vivek", "v6", "v7"]:
        metrics = all_metrics[key]
        if not metrics:
            continue
        rmses = [m["rmse"] for m in metrics if not np.isnan(m["rmse"])]
        maes = [m["mae"] for m in metrics if not np.isnan(m["mae"])]
        absrels = [m["absrel"] for m in metrics if not np.isnan(m["absrel"])]
        d1s = [m["delta1"] for m in metrics if not np.isnan(m["delta1"])]
        if rmses:
            avg = {
                "rmse": np.mean(rmses), "rmse_std": np.std(rmses),
                "mae": np.mean(maes), "mae_std": np.std(maes),
                "absrel": np.mean(absrels), "absrel_std": np.std(absrels),
                "delta1": np.mean(d1s), "delta1_std": np.std(d1s),
                "n_frames": len(rmses),
            }
            summary[key] = avg
            print(f"{model_names[key]:<12} {avg['rmse']:>8.4f} {avg['mae']:>8.4f} "
                  f"{avg['absrel']:>8.4f} {avg['delta1']:>7.1f}%")

    with open(output_dir / f"comparison_summary_{dataset_name}.json", 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    return summary


def main():
    output_base = Path("/home/nishant/maps/paper_figures")
    output_base.mkdir(parents=True, exist_ok=True)

    # Dataset 1: Glass corridor (173610)
    s1 = process_dataset(
        "glass_corridor",
        frames_dir="/home/nishant/maps/glass_corridor_frames",
        da3_depth_dir="/home/nishant/maps/glass_corridor_frames/da3_depth",
        student_results_dir="/home/nishant/maps/glass_corridor_student_results",
        output_dir=output_base / "glass_corridor",
    )

    # Dataset 2: New bag (213831)
    s2 = process_dataset(
        "bag_213831",
        frames_dir="/home/nishant/maps/bag_213831_frames",
        da3_depth_dir="/home/nishant/maps/bag_213831_student_results/depth_da3",
        student_results_dir="/home/nishant/maps/bag_213831_student_results",
        output_dir=output_base / "bag_213831",
    )

    # Print combined table
    print(f"\n{'='*70}")
    print("COMBINED RESULTS ACROSS BOTH DATASETS")
    print(f"{'='*70}")
    print(f"\n{'Model':<12} {'Glass RMSE':>12} {'213831 RMSE':>12} {'Glass d<1.25':>14} {'213831 d<1.25':>14}")
    print("-" * 66)
    names = {"da3": "DA3 (T)", "v5_vivek": "Stud V5",
             "v6": "Stud V6", "v7": "Stud V7"}
    for key in ["da3", "v5_vivek", "v6", "v7"]:
        g = s1.get(key, {})
        n = s2.get(key, {})
        gr = f"{g.get('rmse', float('nan')):.4f}" if g else "N/A"
        nr = f"{n.get('rmse', float('nan')):.4f}" if n else "N/A"
        gd = f"{g.get('delta1', float('nan')):.1f}%" if g else "N/A"
        nd = f"{n.get('delta1', float('nan')):.1f}%" if n else "N/A"
        print(f"{names[key]:<12} {gr:>12} {nr:>12} {gd:>14} {nd:>14}")

    # LaTeX table
    print(f"\n% LaTeX table for paper")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Model & RMSE (Glass) & RMSE (Corridor) & $\delta_{1.25}$ (Glass) & $\delta_{1.25}$ (Corridor) \\")
    print(r"\midrule")
    for key in ["da3", "v5_vivek", "v6", "v7"]:
        g = s1.get(key, {})
        n = s2.get(key, {})
        name = {"da3": "DA3-Small (Teacher)", "v5_vivek": "Student V5",
                "v6": "Student V6", "v7": "Student V7 (Best)"}[key]
        gr = f"{g['rmse']:.3f}" if g else "--"
        nr = f"{n['rmse']:.3f}" if n else "--"
        gd = f"{g['delta1']:.1f}\\%" if g else "--"
        nd = f"{n['delta1']:.1f}\\%" if n else "--"
        print(f"{name} & {gr} & {nr} & {gd} & {nd} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
