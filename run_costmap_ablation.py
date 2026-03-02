#!/usr/bin/env python3
"""
Costmap ablation runner: orchestrates all 7 experiments on a dataset of
RGB + sensor depth frames.

Experiment matrix:
    Baseline : Sensor depth -> occupancy grid + fixed inflation (0.09m)
    A1       : Sensor depth -> occupancy grid + dynamic inflation (corridor_width)
    A2       : Sensor depth + DA3 depth -> occupancy grid + fixed inflation
    A3       : Sensor depth + DA3 depth -> occupancy grid + dynamic inflation
    A4       : DA3 depth only -> occupancy grid + dynamic inflation (no LiDAR/sensor)
    A5       : Sensor + DA3 depth -> occupancy grid + class-aware inflation (YOLO+SAM2)
    A6       : DA3 depth only -> occupancy grid + class-aware inflation (YOLO+SAM2)

For each frame:
    1. Load RGB + sensor depth
    2. Run DA3-Small on RGB (or load pre-computed DA3 depth)
    3. Median-scale DA3 output against sensor depth
    4. Load YOLO+SAM2 segmentation mask (if available)
    5. Build occupancy grids + class grids
    6. Apply inflation variants
    7. Compute metrics

Output: per-frame CSV + per-sequence summary + overall summary table.

Usage:
    # With pre-computed DA3 depth + segmentation (no GPU needed):
    python run_costmap_ablation.py \
        --manifest corridor_eval_data/manifest.jsonl \
        --da3-dir corridor_eval_data/da3_depth/ \
        --seg-dir corridor_eval_data/sam2_seg/ \
        --output-dir results/costmap_ablation/corridor

    # With live DA3 inference (GPU required):
    python run_costmap_ablation.py \
        --manifest corridor_eval_data/manifest.jsonl \
        --run-da3 --model-id depth-anything/DA3-SMALL \
        --output-dir results/costmap_ablation/corridor
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from costmap_builder import (
    depth_to_occupancy_grid,
    build_class_grid,
    compute_costmap_metrics,
    LETHAL_COST,
    SEG_UNKNOWN,
)
from inflation import (
    apply_fixed_inflation,
    apply_dynamic_inflation,
    apply_class_aware_inflation,
    compute_dynamic_radius,
    estimate_corridor_width,
)


EXPERIMENTS = ["Baseline", "A1", "A2", "A3", "A4", "A5", "A6"]

EXPERIMENT_DESCRIPTIONS = {
    "Baseline": "Sensor depth + fixed inflation (0.09m)",
    "A1": "Sensor depth + dynamic inflation (corridor_width)",
    "A2": "Sensor+DA3 depth + fixed inflation (0.09m)",
    "A3": "Sensor+DA3 depth + dynamic inflation (corridor_width)",
    "A4": "DA3 depth only + dynamic inflation (no sensor/LiDAR)",
    "A5": "Sensor+DA3 + class-aware inflation (YOLO+SAM2)",
    "A6": "DA3 only + class-aware inflation (YOLO+SAM2)",
}


def median_scale_align(pred: np.ndarray, gt: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    """Scale relative DA3 depth so median matches sensor depth on valid pixels."""
    valid = (gt >= min_d) & (gt <= max_d)
    p_valid = pred[valid]
    g_valid = gt[valid]
    if len(p_valid) == 0 or np.median(p_valid) < 1e-8:
        return pred
    scale = np.median(g_valid) / np.median(p_valid)
    return pred * scale


def load_da3_depth(
    rgb_path: str,
    da3_dir: Optional[Path],
    da3_model=None,
    is_relative: bool = True,
    focal: float = 300.0,
) -> Optional[np.ndarray]:
    """Load pre-computed DA3 depth or run live inference."""
    stem = Path(rgb_path).stem

    if da3_dir is not None:
        npy_path = da3_dir / f"{stem}.npy"
        if npy_path.exists():
            return np.load(npy_path).astype(np.float32)

    if da3_model is not None:
        from eval_corridor_da3 import infer_single
        return infer_single(da3_model, rgb_path, focal, is_relative)

    return None


def merge_occupancy_grids(grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
    """Merge two occupancy grids: a cell is lethal if either grid marks it lethal."""
    return np.maximum(grid_a, grid_b)


def run_single_frame(
    sensor_depth: np.ndarray,
    da3_depth: Optional[np.ndarray],
    intrinsics: Dict[str, float],
    grid_params: Dict,
    seg_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Run all experiments on a single frame.

    Returns dict[experiment_name] -> dict with keys:
        grid, inflated_grid, radius, metrics, timing_ms
    """
    gp = grid_params
    results = {}

    grid_kw = dict(
        grid_size=gp["grid_size"], resolution=gp["resolution"],
        min_height=gp["min_height"], max_height=gp["max_height"],
        min_depth=gp["min_depth"], max_depth=gp["max_depth"],
        camera_height=gp["camera_height"],
    )

    # --- Sensor-based occupancy grid (Baseline, A1) ---
    t0 = time.time()
    grid_sensor = depth_to_occupancy_grid(sensor_depth, intrinsics, **grid_kw)
    t_sensor_grid = (time.time() - t0) * 1000

    # --- DA3-based occupancy grid (A2-A6) ---
    grid_da3 = None
    t_da3_grid = 0.0
    if da3_depth is not None:
        t0 = time.time()
        grid_da3 = depth_to_occupancy_grid(da3_depth, intrinsics, **grid_kw)
        t_da3_grid = (time.time() - t0) * 1000

    # --- Merged grid for A2, A3, A5 ---
    if grid_da3 is not None:
        grid_merged = merge_occupancy_grids(grid_sensor, grid_da3)
    else:
        grid_merged = grid_sensor.copy()

    depth_for_dyn = sensor_depth

    # --- Build class grids for A5/A6 (only if seg mask available) ---
    class_grid_merged = None
    class_grid_da3 = None
    t_seg = 0.0
    if seg_mask is not None and da3_depth is not None:
        t0 = time.time()
        class_grid_merged = build_class_grid(
            da3_depth, seg_mask, intrinsics, **grid_kw,
        )
        class_grid_da3 = class_grid_merged  # DA3-only uses same seg
        t_seg = (time.time() - t0) * 1000

    # --- Baseline: sensor grid + fixed inflation ---
    t0 = time.time()
    inflated_baseline = apply_fixed_inflation(
        grid_sensor, radius=gp["fixed_radius"], resolution=gp["resolution"],
    )
    t_baseline = t_sensor_grid + (time.time() - t0) * 1000
    results["Baseline"] = {
        "grid": grid_sensor,
        "inflated_grid": inflated_baseline,
        "radius": gp["fixed_radius"],
        "timing_ms": t_baseline,
    }

    # --- A1: sensor grid + dynamic inflation ---
    t0 = time.time()
    inflated_a1, radius_a1 = apply_dynamic_inflation(
        grid_sensor, depth_for_dyn,
        strategy=gp["dynamic_strategy"], resolution=gp["resolution"],
    )
    t_a1 = t_sensor_grid + (time.time() - t0) * 1000
    results["A1"] = {
        "grid": grid_sensor,
        "inflated_grid": inflated_a1,
        "radius": radius_a1,
        "timing_ms": t_a1,
    }

    if da3_depth is not None:
        # --- A2: merged grid + fixed inflation ---
        t0 = time.time()
        inflated_a2 = apply_fixed_inflation(
            grid_merged, radius=gp["fixed_radius"], resolution=gp["resolution"],
        )
        t_a2 = t_sensor_grid + t_da3_grid + (time.time() - t0) * 1000
        results["A2"] = {
            "grid": grid_merged,
            "inflated_grid": inflated_a2,
            "radius": gp["fixed_radius"],
            "timing_ms": t_a2,
        }

        # --- A3: merged grid + dynamic inflation ---
        t0 = time.time()
        inflated_a3, radius_a3 = apply_dynamic_inflation(
            grid_merged, depth_for_dyn,
            strategy=gp["dynamic_strategy"], resolution=gp["resolution"],
        )
        t_a3 = t_sensor_grid + t_da3_grid + (time.time() - t0) * 1000
        results["A3"] = {
            "grid": grid_merged,
            "inflated_grid": inflated_a3,
            "radius": radius_a3,
            "timing_ms": t_a3,
        }

        # --- A4: DA3-only grid + dynamic inflation ---
        t0 = time.time()
        inflated_a4, radius_a4 = apply_dynamic_inflation(
            grid_da3, da3_depth,
            strategy=gp["dynamic_strategy"], resolution=gp["resolution"],
        )
        t_a4 = t_da3_grid + (time.time() - t0) * 1000
        results["A4"] = {
            "grid": grid_da3,
            "inflated_grid": inflated_a4,
            "radius": radius_a4,
            "timing_ms": t_a4,
        }

        # --- A5: merged grid + class-aware inflation ---
        if class_grid_merged is not None:
            t0 = time.time()
            inflated_a5, radius_a5 = apply_class_aware_inflation(
                grid_merged, class_grid_merged, resolution=gp["resolution"],
            )
            t_a5 = t_sensor_grid + t_da3_grid + t_seg + (time.time() - t0) * 1000
            results["A5"] = {
                "grid": grid_merged,
                "inflated_grid": inflated_a5,
                "radius": radius_a5,
                "timing_ms": t_a5,
            }

        # --- A6: DA3-only grid + class-aware inflation ---
        if class_grid_da3 is not None:
            t0 = time.time()
            inflated_a6, radius_a6 = apply_class_aware_inflation(
                grid_da3, class_grid_da3, resolution=gp["resolution"],
            )
            t_a6 = t_da3_grid + t_seg + (time.time() - t0) * 1000
            results["A6"] = {
                "grid": grid_da3,
                "inflated_grid": inflated_a6,
                "radius": radius_a6,
                "timing_ms": t_a6,
            }

    # --- Compute metrics (compare all experiments against Baseline) ---
    baseline_grid = results["Baseline"]["inflated_grid"]
    for name, res in results.items():
        if name == "Baseline":
            res["metrics"] = compute_costmap_metrics(res["inflated_grid"], res["inflated_grid"])
        else:
            res["metrics"] = compute_costmap_metrics(res["inflated_grid"], baseline_grid)

    return results


def write_per_frame_csv(
    all_results: List[Dict],
    stems: List[str],
    output_path: Path,
):
    """Write per-frame metrics to CSV."""
    fieldnames = [
        "frame", "experiment",
        "iou", "lethal_iou", "detection_rate", "false_positive_rate",
        "clearance_pred_m", "clearance_gt_m",
        "occupied_cells_pred", "occupied_cells_gt",
        "inflation_radius_m", "timing_ms",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stem, frame_results in zip(stems, all_results):
            for exp_name in EXPERIMENTS:
                if exp_name not in frame_results:
                    continue
                res = frame_results[exp_name]
                row = {
                    "frame": stem,
                    "experiment": exp_name,
                    "inflation_radius_m": f"{res['radius']:.4f}",
                    "timing_ms": f"{res['timing_ms']:.2f}",
                }
                for k, v in res["metrics"].items():
                    row[k] = f"{v:.6f}" if isinstance(v, float) else str(v)
                writer.writerow(row)


def compute_summary(
    all_results: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-frame metrics into per-experiment summary."""
    summary = {}

    for exp_name in EXPERIMENTS:
        metrics_lists = {
            "iou": [], "lethal_iou": [], "detection_rate": [],
            "false_positive_rate": [],
            "clearance_pred_m": [], "clearance_gt_m": [],
            "radius": [], "timing_ms": [],
        }

        for frame_results in all_results:
            if exp_name not in frame_results:
                continue
            res = frame_results[exp_name]
            for k in ["iou", "lethal_iou", "detection_rate",
                       "false_positive_rate",
                       "clearance_pred_m", "clearance_gt_m"]:
                metrics_lists[k].append(res["metrics"][k])
            metrics_lists["radius"].append(res["radius"])
            metrics_lists["timing_ms"].append(res["timing_ms"])

        if not metrics_lists["iou"]:
            continue

        summary[exp_name] = {
            k: float(np.mean(v)) for k, v in metrics_lists.items()
        }
        summary[exp_name]["radius_std"] = float(np.std(metrics_lists["radius"]))
        summary[exp_name]["n_frames"] = len(metrics_lists["iou"])

    return summary


def print_summary_table(summary: Dict[str, Dict[str, float]]):
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 120)
    print(f"{'Experiment':<12} {'Description':<45} {'IoU':>6} {'L-IoU':>6} {'DetRate':>8} "
          f"{'FPR':>6} {'Clearance':>10} {'Radius':>8} {'Time':>8}")
    print("-" * 120)

    for exp_name in EXPERIMENTS:
        if exp_name not in summary:
            continue
        s = summary[exp_name]
        desc = EXPERIMENT_DESCRIPTIONS[exp_name][:44]
        print(f"{exp_name:<12} {desc:<45} {s['iou']:.3f} {s['lethal_iou']:.3f} "
              f"{s['detection_rate']:.4f} "
              f"{s['false_positive_rate']:.4f} {s['clearance_pred_m']:.3f}m "
              f"{s['radius']:.4f}m {s['timing_ms']:.1f}ms")

    print("=" * 120)
    print()


def write_summary_json(summary: Dict, output_path: Path):
    """Write summary to JSON for changelog integration."""
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run costmap ablation study (Baseline + A1-A6)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="results/costmap_ablation",
                        help="Output directory for CSVs and summaries")

    # DA3 depth source (mutually exclusive)
    da3_group = parser.add_mutually_exclusive_group()
    da3_group.add_argument("--da3-dir", type=str, default=None,
                           help="Directory with pre-computed DA3 depth .npy files")
    da3_group.add_argument("--run-da3", action="store_true",
                           help="Run DA3 inference live (requires GPU)")

    parser.add_argument("--seg-dir", type=str, default=None,
                        help="Directory with YOLO+SAM2 segmentation .npy files "
                             "(enables A5/A6 class-aware experiments)")

    parser.add_argument("--model-id", type=str, default="depth-anything/DA3-SMALL",
                        help="DA3 model ID for live inference")
    parser.add_argument("--device", type=str, default=None)

    # Grid parameters (defaults from nav2_hardware.yaml)
    parser.add_argument("--grid-size", type=float, default=6.0)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--min-height", type=float, default=0.05)
    parser.add_argument("--max-height", type=float, default=0.50)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--camera-height", type=float, default=0.25,
                        help="Camera mount height above ground (metres)")
    parser.add_argument("--fixed-radius", type=float, default=0.09,
                        help="Fixed inflation radius (metres)")
    parser.add_argument("--dynamic-strategy", type=str, default="corridor_width",
                        choices=["corridor_width", "min_depth", "class_aware"])

    args = parser.parse_args()

    manifest_dir = Path(args.manifest).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load intrinsics
    intr_path = manifest_dir / "intrinsics.json"
    if intr_path.exists():
        with open(intr_path) as f:
            intrinsics = json.load(f)
        print(f"Intrinsics: fx={intrinsics['fx']:.2f} fy={intrinsics['fy']:.2f}")
    else:
        intrinsics = {"fx": 605.0, "fy": 605.0, "cx": 320.0, "cy": 240.0}
        print(f"WARNING: intrinsics.json not found, using defaults: {intrinsics}")

    # Load manifest
    with open(args.manifest) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} frames from manifest")

    # DA3 source
    da3_dir = Path(args.da3_dir) if args.da3_dir else None
    da3_model = None

    if args.run_da3:
        import torch
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Loading DA3 model {args.model_id} on {device}...")
        from eval_corridor_da3 import load_da3_model
        da3_model = load_da3_model(args.model_id, device)
        print("DA3 model loaded.")
    elif da3_dir:
        print(f"Using pre-computed DA3 depth from {da3_dir}")
    else:
        print("WARNING: No DA3 depth source. Only Baseline and A1 will run.")

    seg_dir = Path(args.seg_dir) if args.seg_dir else None
    if seg_dir:
        print(f"Using YOLO+SAM2 segmentation from {seg_dir} (A5/A6 enabled)")
    else:
        print("No segmentation data. A5/A6 class-aware experiments disabled.")

    grid_params = {
        "grid_size": args.grid_size,
        "resolution": args.resolution,
        "min_height": args.min_height,
        "max_height": args.max_height,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "camera_height": args.camera_height,
        "fixed_radius": args.fixed_radius,
        "dynamic_strategy": args.dynamic_strategy,
    }

    print(f"\nGrid params: {grid_params}")
    print(f"Experiments: {EXPERIMENTS}")
    print(f"Output: {output_dir}")
    print()

    all_results = []
    stems = []
    da3_latencies = []
    corridor_widths = []

    for entry in tqdm(samples, desc="Costmap ablation"):
        rgb_path = str(manifest_dir / entry["rgb"])
        sensor_depth_path = manifest_dir / entry["sensor_depth"]
        stem = Path(entry["rgb"]).stem

        sensor_depth = np.load(sensor_depth_path).astype(np.float32)

        # Load or compute DA3 depth
        t0 = time.time()
        da3_depth = load_da3_depth(
            rgb_path, da3_dir, da3_model, is_relative=True,
        )
        if da3_depth is not None:
            da3_latencies.append((time.time() - t0) * 1000)

            # Resize DA3 output to match sensor depth if needed
            if da3_depth.shape != sensor_depth.shape:
                from PIL import Image as PILImage
                da3_pil = PILImage.fromarray(da3_depth)
                da3_pil = da3_pil.resize(
                    (sensor_depth.shape[1], sensor_depth.shape[0]),
                    PILImage.BILINEAR,
                )
                da3_depth = np.array(da3_pil, dtype=np.float32)

            # Median-scale alignment
            da3_depth = median_scale_align(
                da3_depth, sensor_depth,
                args.min_depth, args.max_depth,
            )

        # Load segmentation mask if available
        seg_mask = None
        if seg_dir is not None:
            # run_sam2.py saves as {stem}_sam2_seg.npy
            seg_path = seg_dir / f"{stem}_sam2_seg.npy"
            if not seg_path.exists():
                seg_path = seg_dir / f"{stem}.npy"
            if seg_path.exists():
                seg_mask = np.load(seg_path).astype(np.uint8)
                if seg_mask.shape[:2] != sensor_depth.shape:
                    from PIL import Image as PILImage
                    seg_pil = PILImage.fromarray(seg_mask)
                    seg_pil = seg_pil.resize(
                        (sensor_depth.shape[1], sensor_depth.shape[0]),
                        PILImage.NEAREST,
                    )
                    seg_mask = np.array(seg_pil, dtype=np.uint8)

        # Track corridor width estimates
        width = estimate_corridor_width(sensor_depth, args.min_depth, args.max_depth)
        corridor_widths.append(width)

        # Run all experiments
        frame_results = run_single_frame(
            sensor_depth, da3_depth, intrinsics, grid_params,
            seg_mask=seg_mask,
        )

        all_results.append(frame_results)
        stems.append(stem)

    # --- Output ---
    print(f"\nProcessed {len(all_results)} frames")

    # Per-frame CSV
    csv_path = output_dir / "per_frame_metrics.csv"
    write_per_frame_csv(all_results, stems, csv_path)
    print(f"Per-frame CSV: {csv_path}")

    # Summary
    summary = compute_summary(all_results)
    print_summary_table(summary)

    summary_path = output_dir / "summary.json"
    write_summary_json(summary, summary_path)
    print(f"Summary JSON: {summary_path}")

    # Corridor width histogram data
    if corridor_widths:
        widths_path = output_dir / "corridor_widths.npy"
        np.save(widths_path, np.array(corridor_widths))
        print(f"Corridor widths: mean={np.mean(corridor_widths):.2f}m, "
              f"std={np.std(corridor_widths):.2f}m, "
              f"min={np.min(corridor_widths):.2f}m, "
              f"max={np.max(corridor_widths):.2f}m")

    # DA3 latency stats
    if da3_latencies:
        print(f"DA3 depth loading/inference: mean={np.mean(da3_latencies):.1f}ms, "
              f"n={len(da3_latencies)}")

    # Seg mask coverage stats
    if seg_dir:
        seg_count = sum(
            1 for r in all_results if "A5" in r
        )
        print(f"Segmentation masks loaded: {seg_count}/{len(all_results)} frames")

    # Inflation radius histogram (for dynamic experiments)
    for exp_name in ["A1", "A3", "A4", "A5", "A6"]:
        if exp_name not in summary:
            continue
        radii = [r[exp_name]["radius"] for r in all_results if exp_name in r]
        radii_path = output_dir / f"inflation_radii_{exp_name}.npy"
        np.save(radii_path, np.array(radii))
        print(f"{exp_name} inflation radius: mean={np.mean(radii):.4f}m, "
              f"std={np.std(radii):.4f}m, "
              f"range=[{np.min(radii):.4f}, {np.max(radii):.4f}]m")


if __name__ == "__main__":
    main()
