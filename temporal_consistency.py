#!/usr/bin/env python3
"""
Compute costmap temporal stability metrics from per-frame CSVs.

Metrics (computed between consecutive frames):
  1. Occupied-cell jitter: |occ[t] - occ[t-1]|
     - mean, std, max of absolute frame-to-frame change
     - normalized jitter: mean_jitter / mean_occupied (% of mean)
  2. Centroid drift: sqrt((cx[t]-cx[t-1])^2 + (cy[t]-cy[t-1])^2)
     - mean, std, max of per-frame Euclidean drift (metres)

Datasets:
  - Corridor live replay: maps/ablation_results/{L,L+S,L+D,D,L+D+dyn}/
    Source rosbag: rgbd_imu_20260302_174745 (81s, 8GB), GTX 1060, 0.3x
  - LILocBench dynamics_0: maps/lilocbench_ablation_results/ + benchmark/

Outputs:
  ml_pipeline/results/temporal_consistency.json
  stdout: LaTeX-ready table rows
"""

import csv
import json
import math
import pathlib

import numpy as np

MAPS = pathlib.Path.home() / "maps"
ROOT = pathlib.Path(__file__).resolve().parent
RESOLUTION = 0.05  # m/cell for converting centroid grid coords to metres


def load_csv(path: pathlib.Path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def compute_stability(rows):
    """Compute temporal stability from per-frame costmap rows."""
    occ = np.array([int(r["occupied_cells"]) for r in rows], dtype=np.float64)
    cx = np.array([float(r["obstacle_centroid_x"]) for r in rows])
    cy = np.array([float(r["obstacle_centroid_y"]) for r in rows])
    ts = np.array([float(r["timestamp"]) for r in rows])

    n = len(occ)
    if n < 2:
        return None

    occ_diff = np.abs(np.diff(occ))
    dt = np.diff(ts)
    centroid_drift = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)

    valid_drift = np.isfinite(centroid_drift)
    centroid_drift_clean = centroid_drift[valid_drift]

    duration = ts[-1] - ts[0]
    mean_dt = float(np.mean(dt))
    update_hz = 1.0 / mean_dt if mean_dt > 0 else 0

    mean_occ = float(np.mean(occ))
    norm_jitter = (float(np.mean(occ_diff)) / mean_occ * 100
                   if mean_occ > 0 else 0)

    return {
        "n_frames": n,
        "duration_s": round(duration, 1),
        "update_hz": round(update_hz, 1),
        "mean_occupied": round(mean_occ, 1),
        "occ_jitter_mean": round(float(np.mean(occ_diff)), 1),
        "occ_jitter_std": round(float(np.std(occ_diff, ddof=1)), 1),
        "occ_jitter_max": int(np.max(occ_diff)),
        "occ_jitter_normalized_pct": round(norm_jitter, 2),
        "centroid_drift_mean_m": round(
            float(np.mean(centroid_drift_clean)) * RESOLUTION, 4),
        "centroid_drift_std_m": round(
            float(np.std(centroid_drift_clean, ddof=1)) * RESOLUTION, 4),
        "centroid_drift_max_m": round(
            float(np.max(centroid_drift_clean)) * RESOLUTION, 4),
    }


def process_dataset(name, config_paths):
    """Process all configs in a dataset, return results dict."""
    results = {}
    print(f"\n{'=' * 78}")
    print(f"  {name}")
    print(f"{'=' * 78}")
    print(f"{'Config':<10s} {'Frames':>6s} {'Hz':>5s} "
          f"{'MeanOcc':>8s} {'Jitter':>8s} {'Jit%':>6s} "
          f"{'Drift(m)':>9s} {'MaxDrift':>9s}")
    print("-" * 78)

    for cfg, csv_path in config_paths:
        if not csv_path.exists():
            print(f"{cfg:<10s}  --- file not found: {csv_path}")
            continue
        rows = load_csv(csv_path)
        stats = compute_stability(rows)
        if stats is None:
            print(f"{cfg:<10s}  --- too few frames")
            continue
        results[cfg] = stats
        print(
            f"{cfg:<10s} {stats['n_frames']:>6d} "
            f"{stats['update_hz']:>5.1f} "
            f"{stats['mean_occupied']:>8.0f} "
            f"{stats['occ_jitter_mean']:>7.1f} "
            f"{stats['occ_jitter_normalized_pct']:>5.1f}% "
            f"{stats['centroid_drift_mean_m']:>8.4f} "
            f"{stats['centroid_drift_max_m']:>8.4f}"
        )
    return results


def print_latex(corridor, lilocbench):
    """Print LaTeX-ready table rows."""
    print(f"\n{'=' * 78}")
    print("  LaTeX table: Costmap Temporal Stability")
    print(f"{'=' * 78}")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Costmap temporal stability (frame-to-frame)}")
    print("\\label{tab:temporal}")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\footnotesize")
    print("\\begin{tabular}{llrrrrr}")
    print("\\toprule")
    print(" & & \\textbf{Frames} & \\textbf{Mean Occ.}"
          " & \\textbf{Jitter} & \\textbf{Jitter\\%}"
          " & \\textbf{Drift} \\\\")
    print("\\midrule")

    for scene_label, data in [("Corridor", corridor),
                               ("LILocBench", lilocbench)]:
        config_order = ["L", "L+S", "L+D", "D", "L+D+dyn"]
        for i, cfg in enumerate(config_order):
            if cfg not in data:
                continue
            s = data[cfg]
            prefix = f"\\multirow{{5}}{{*}}{{{scene_label}}}" if i == 0 else ""
            jit = s["occ_jitter_mean"]
            jit_pct = s["occ_jitter_normalized_pct"]
            drift = s["centroid_drift_mean_m"]
            print(
                f"{prefix} & {cfg} & {s['n_frames']}"
                f" & {s['mean_occupied']:.0f}"
                f" & {jit:.1f}"
                f" & {jit_pct:.1f}\\%"
                f" & {drift:.4f}\\,m \\\\"
            )
        if scene_label == "Corridor":
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    corridor_configs = [
        ("L",       MAPS / "ablation_results" / "L" / "costmap_stats.csv"),
        ("L+S",     MAPS / "ablation_results" / "L+S" / "costmap_stats.csv"),
        ("L+D",     MAPS / "ablation_results" / "L+D" / "costmap_stats.csv"),
        ("D",       MAPS / "ablation_results" / "D" / "costmap_stats.csv"),
        ("L+D+dyn", MAPS / "ablation_results" / "L+D+dyn" / "costmap_stats.csv"),
    ]

    lilocbench_abl = MAPS / "lilocbench_ablation_results"
    lilocbench_bench = MAPS / "lilocbench_benchmark" / "dynamics_0"
    lilocbench_configs = [
        ("L",       lilocbench_bench / "L" / "costmap_stats.csv"),
        ("L+S",     lilocbench_abl / "L+S" / "costmap_stats.csv"),
        ("L+D",     lilocbench_abl / "L+D" / "costmap_stats.csv"),
        ("D",       lilocbench_abl / "D" / "costmap_stats.csv"),
        ("L+D+dyn", lilocbench_abl / "L+D+dyn" / "costmap_stats.csv"),
    ]

    corridor = process_dataset(
        "CORRIDOR LIVE REPLAY (GTX 1060, 0.3x, bag: rgbd_imu_20260302_174745)",
        corridor_configs)
    lilocbench = process_dataset(
        "LILOCBENCH dynamics_0 (10 pedestrians)",
        lilocbench_configs)

    print_latex(corridor, lilocbench)

    # Interpretation
    print(f"\n{'=' * 78}")
    print("  INTERPRETATION")
    print(f"{'=' * 78}")
    if "L" in corridor and "L+D" in corridor:
        l = corridor["L"]
        ld = corridor["L+D"]
        print(f"  Corridor jitter: L = {l['occ_jitter_normalized_pct']:.1f}%, "
              f"L+D = {ld['occ_jitter_normalized_pct']:.1f}%")
        print(f"  Corridor drift:  L = {l['centroid_drift_mean_m']:.4f}m, "
              f"L+D = {ld['centroid_drift_mean_m']:.4f}m")
    if "L" in lilocbench and "L+D" in lilocbench:
        l = lilocbench["L"]
        ld = lilocbench["L+D"]
        print(f"  LILocBench jitter: L = {l['occ_jitter_normalized_pct']:.1f}%, "
              f"L+D = {ld['occ_jitter_normalized_pct']:.1f}%")
        print(f"  LILocBench drift:  L = {l['centroid_drift_mean_m']:.4f}m, "
              f"L+D = {ld['centroid_drift_mean_m']:.4f}m")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "temporal_consistency.json"
    with open(out_path, "w") as f:
        json.dump({
            "corridor": corridor,
            "lilocbench": lilocbench,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
