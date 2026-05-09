#!/usr/bin/env python3
"""
Pixel-level depth fusion study: sensor where valid + model fills dead pixels.

Evaluates five depth sources on 459 corridor frames:
  1. Sensor only        -- GT where valid (23% coverage)
  2. DA3-Small only     -- pre-computed DA3 w/ median scaling (100% coverage)
  3. V9 student only    -- live inference (100% coverage)
  4. Fused(Sensor+DA3)  -- sensor where valid, DA3 fills dead pixels
  5. Fused(Sensor+V9)   -- sensor where valid, V9 fills dead pixels

Metrics are evaluated on the 23% sensor-valid pixels (where we have GT).
For dead pixels (77%), cross-model agreement is reported (V9 vs DA3).
Per-frame metrics enable std/CI computation.

Optionally generates 6-panel comparison figures for representative frames.

Dataset: 459 corridor frames from Orbbec Femto Bolt rosbag
         (rgbd_imu_20260228_003828_0.mcap extraction)

Usage:
    python eval_pixel_fusion.py
    python eval_pixel_fusion.py --save-figures --figure-frames 50 150 250 350
    python eval_pixel_fusion.py --device cuda
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import Config
from models.student import build_student

MANIFEST = "corridor_eval_data/manifest.jsonl"
DA3_DEPTH_DIR = "corridor_eval_data/da3_depth"
RESULTS_DIR = "results"

MIN_DEPTH = 0.1
MAX_DEPTH = 5.0

BINS = [
    (0.3, 1.0, "near"),
    (1.0, 2.0, "mid"),
    (2.0, 4.0, "far"),
]


def compute_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(p) == 0:
        return {"rmse": float("nan"), "mae": float("nan"),
                "abs_rel": float("nan"), "delta_125": float("nan"),
                "n_pixels": 0}
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))
    abs_rel = float(np.mean(np.abs(p - g) / (g + 1e-8)))
    ratio = np.maximum(p / (g + 1e-8), g / (p + 1e-8))
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)
    return {"rmse": rmse, "mae": mae, "abs_rel": abs_rel,
            "delta_125": delta_125, "n_pixels": int(len(p))}


def per_frame_metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(p) == 0:
        return None
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))
    abs_rel = float(np.mean(np.abs(p - g) / (g + 1e-8)))
    ratio = np.maximum(p / (g + 1e-8), g / (p + 1e-8))
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)
    return {"rmse": rmse, "mae": mae, "abs_rel": abs_rel,
            "delta_125": delta_125, "n_pixels": int(len(p))}


def load_student(checkpoint_path, device):
    cfg = Config()
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False,
                          backbone=cfg.BACKBONE)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def aggregate_frame_stats(frame_list):
    """Compute mean, std, 95% CI from a list of per-frame metric dicts."""
    if not frame_list:
        return {}
    keys = frame_list[0].keys()
    stats = {}
    for k in keys:
        if k == "n_pixels":
            continue
        vals = [f[k] for f in frame_list if f is not None and not math.isnan(f[k])]
        n = len(vals)
        if n == 0:
            stats[f"{k}_mean"] = float("nan")
            stats[f"{k}_std"] = float("nan")
            stats[f"{k}_ci95"] = float("nan")
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        stats[f"{k}_mean"] = mean
        stats[f"{k}_std"] = std
        stats[f"{k}_ci95"] = ci95
    stats["n_frames"] = len(frame_list)
    return stats


def generate_figures(frame_data, out_dir, frame_indices):
    """Generate 6-panel comparison figures for selected frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig_dir = out_dir / "fusion_figures"
    fig_dir.mkdir(exist_ok=True)

    depth_norm = Normalize(vmin=0.0, vmax=5.0)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="white")

    from matplotlib.gridspec import GridSpec

    for idx in frame_indices:
        if idx >= len(frame_data):
            print(f"  [WARN] Frame index {idx} out of range, skipping")
            continue

        fd = frame_data[idx]
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05],
                      hspace=0.15, wspace=0.08)

        panels_spec = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
        ]
        axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
        cax = fig.add_subplot(gs[:, 3])

        # Row 0: RGB, Sensor, DA3
        axes[0][0].imshow(fd["rgb"])
        axes[0][0].set_title("RGB", fontsize=13, fontweight="bold")
        axes[0][0].axis("off")

        sensor_vis = np.where(fd["sensor_valid"], fd["sensor"], np.nan)
        axes[0][1].imshow(sensor_vis, cmap=cmap, norm=depth_norm)
        valid_pct = fd["sensor_valid"].mean() * 100
        axes[0][1].set_title(f"Sensor Depth ({valid_pct:.0f}% valid)",
                             fontsize=13, fontweight="bold")
        axes[0][1].axis("off")

        axes[0][2].imshow(fd["da3"], cmap=cmap, norm=depth_norm)
        axes[0][2].set_title("DA3-Small (median-scaled)", fontsize=13,
                             fontweight="bold")
        axes[0][2].axis("off")

        # Row 1: V9, Fused(S+DA3), Fused(S+V9)
        axes[1][0].imshow(fd["v9"], cmap=cmap, norm=depth_norm)
        axes[1][0].set_title("V9 Student", fontsize=13, fontweight="bold")
        axes[1][0].axis("off")

        axes[1][1].imshow(fd["fused_da3"], cmap=cmap, norm=depth_norm)
        axes[1][1].set_title("Fused (Sensor + DA3)", fontsize=13,
                             fontweight="bold")
        axes[1][1].axis("off")

        axes[1][2].imshow(fd["fused_v9"], cmap=cmap, norm=depth_norm)
        axes[1][2].set_title("Fused (Sensor + V9)", fontsize=13,
                             fontweight="bold")
        axes[1][2].axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=depth_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Depth (m)", fontsize=12)

        frame_id = fd["frame_id"]
        fig.suptitle(f"Frame {frame_id}: Pixel-Level Depth Fusion",
                     fontsize=15, fontweight="bold", y=0.98)

        png_path = fig_dir / f"fusion_frame_{frame_id}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        pdf_path = fig_dir / f"fusion_frame_{frame_id}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {png_path.name} + {pdf_path.name}")

    print(f"\nFigures saved to {fig_dir}/")
    return fig_dir


def main():
    parser = argparse.ArgumentParser(
        description="Pixel-level depth fusion: sensor + DA3/V9")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--v9-checkpoint", type=str,
                        default="hpc_outputs/best_depth_v9.pt")
    parser.add_argument("--save-figures", action="store_true",
                        help="Generate 6-panel comparison PNGs/PDFs")
    parser.add_argument("--figure-frames", type=int, nargs="+",
                        default=None,
                        help="Frame indices for figures (auto-selected if omitted)")
    args = parser.parse_args()

    manifest_dir = Path(MANIFEST).parent
    da3_dir = Path(DA3_DEPTH_DIR)
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)

    with open(MANIFEST) as f:
        samples = [json.loads(line) for line in f]
    n_frames = len(samples)
    print(f"Corridor frames: {n_frames}")
    print(f"Device: {args.device}")
    print(f"V9 checkpoint: {args.v9_checkpoint}")

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load V9 student model
    # ------------------------------------------------------------------
    print("\nLoading V9 student model...")
    v9_model, v9_cfg = load_student(args.v9_checkpoint, device)
    H, W = v9_cfg.INPUT_HEIGHT, v9_cfg.INPUT_WIDTH
    print(f"  Model resolution: {W}x{H}")

    # ------------------------------------------------------------------
    # Process all frames
    # ------------------------------------------------------------------
    print(f"\nProcessing {n_frames} frames...\n")

    source_names = ["sensor", "da3", "v9", "fused_da3", "fused_v9"]

    per_frame_all = {s: [] for s in source_names}
    per_frame_bins = {s: {b[2]: [] for b in BINS} for s in source_names}

    coverage_stats = {"sensor_valid_pct": [], "dead_pct": [],
                      "da3_fill_pct": [], "v9_fill_pct": []}

    dead_pixel_agreement = []

    frame_data_for_figures = [] if args.save_figures else None

    total_pixels = 0
    total_valid = 0
    total_dead = 0

    for i, entry in enumerate(tqdm(samples, desc="Fusion eval")):
        frame_id = Path(entry["rgb"]).stem

        # --- Load RGB ---
        rgb_path = manifest_dir / entry["rgb"]
        rgb_full = Image.open(rgb_path).convert("RGB")
        rgb = rgb_full.resize((W, H), Image.BILINEAR)
        rgb_np = np.array(rgb, dtype=np.float32) / 255.0

        # --- Load sensor depth ---
        sd_full = np.load(manifest_dir / entry["sensor_depth"]).astype(np.float32)
        sd = np.array(Image.fromarray(sd_full).resize((W, H), Image.NEAREST),
                      dtype=np.float32)

        # --- Load DA3 pre-computed depth ---
        da3_path = da3_dir / f"{frame_id}.npy"
        da3_full = np.load(da3_path).astype(np.float32)
        if da3_full.shape != sd.shape:
            da3_d = np.array(Image.fromarray(da3_full).resize(
                (W, H), Image.BILINEAR), dtype=np.float32)
        else:
            da3_d = da3_full.copy()

        # Per-frame median scaling for DA3 (relative -> metric)
        valid_mask = (sd >= MIN_DEPTH) & (sd <= MAX_DEPTH)
        p_valid = da3_d[valid_mask]
        g_valid = sd[valid_mask]
        if len(p_valid) > 0 and np.median(p_valid) > 1e-8:
            scale = np.median(g_valid) / np.median(p_valid)
            da3_d = da3_d * scale

        # --- V9 inference ---
        rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            v9_pred, _ = v9_model(rgb_t)
        v9_d = v9_pred.squeeze().cpu().numpy()

        # --- Masks ---
        dead_mask = ~valid_mask
        n_pix = sd.size
        n_valid = int(valid_mask.sum())
        n_dead = int(dead_mask.sum())
        total_pixels += n_pix
        total_valid += n_valid
        total_dead += n_dead

        # --- Build fused depth maps ---
        fused_da3 = np.where(valid_mask, sd, da3_d)
        fused_v9 = np.where(valid_mask, sd, v9_d)

        # --- Coverage stats ---
        da3_coverage = (da3_d > 0).mean() * 100
        v9_coverage = (v9_d > 0).mean() * 100
        coverage_stats["sensor_valid_pct"].append(n_valid / n_pix * 100)
        coverage_stats["dead_pct"].append(n_dead / n_pix * 100)
        coverage_stats["da3_fill_pct"].append(da3_coverage)
        coverage_stats["v9_fill_pct"].append(v9_coverage)

        # --- Dead-pixel cross-model agreement (V9 vs DA3 on dead pixels) ---
        if n_dead > 0:
            da3_dead = da3_d[dead_mask]
            v9_dead = v9_d[dead_mask]
            both_positive = (da3_dead > 0) & (v9_dead > 0)
            if both_positive.sum() > 0:
                d_d = da3_dead[both_positive]
                v_d = v9_dead[both_positive]
                cross_rmse = float(np.sqrt(np.mean((d_d - v_d) ** 2)))
                cross_mae = float(np.mean(np.abs(d_d - v_d)))
                dead_pixel_agreement.append({
                    "frame": frame_id, "cross_rmse": cross_rmse,
                    "cross_mae": cross_mae, "n_dead_both_valid": int(both_positive.sum())
                })

        # --- Per-source metrics on VALID pixels (where we have GT) ---
        sources = {
            "sensor": sd,
            "da3": da3_d,
            "v9": v9_d,
            "fused_da3": fused_da3,
            "fused_v9": fused_v9,
        }

        for sname, depth_map in sources.items():
            fm = per_frame_metrics(depth_map, sd, valid_mask)
            if fm is not None:
                per_frame_all[sname].append(fm)

            for lo, hi, blabel in BINS:
                bin_mask = valid_mask & (sd >= lo) & (sd < hi)
                bm = per_frame_metrics(depth_map, sd, bin_mask)
                if bm is not None:
                    per_frame_bins[sname][blabel].append(bm)

        # --- Collect data for figure generation ---
        if frame_data_for_figures is not None:
            frame_data_for_figures.append({
                "frame_id": frame_id,
                "rgb": np.array(rgb),
                "sensor": sd,
                "sensor_valid": valid_mask,
                "da3": da3_d,
                "v9": v9_d,
                "fused_da3": fused_da3,
                "fused_v9": fused_v9,
            })

    del v9_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==================================================================
    # Aggregate results
    # ==================================================================
    print("\n" + "=" * 80)
    print("  PIXEL-LEVEL DEPTH FUSION RESULTS")
    print("=" * 80)

    print(f"\n  Total pixels: {total_pixels:,}")
    print(f"  Valid (sensor GT): {total_valid:,} "
          f"({total_valid / total_pixels * 100:.1f}%)")
    print(f"  Dead (no GT):      {total_dead:,} "
          f"({total_dead / total_pixels * 100:.1f}%)")

    # --- Overall metrics (evaluated on valid pixels) ---
    print(f"\n{'─' * 80}")
    print("  OVERALL METRICS (on sensor-valid pixels, where GT exists)")
    print(f"{'─' * 80}")

    all_results = {}
    header = (f"  {'Source':25s} {'RMSE (m)':>12s} {'MAE (m)':>12s} "
              f"{'AbsRel':>10s} {'δ<1.25':>10s} {'Coverage':>10s}")
    print(header)
    print("  " + "─" * 77)

    source_labels = {
        "sensor": "Sensor (GT)",
        "da3": "DA3-Small (median)",
        "v9": "V9 student",
        "fused_da3": "Fused (Sensor+DA3)",
        "fused_v9": "Fused (Sensor+V9)",
    }
    coverage_map = {
        "sensor": f"{np.mean(coverage_stats['sensor_valid_pct']):.1f}%",
        "da3": "100%",
        "v9": "100%",
        "fused_da3": "100%",
        "fused_v9": "100%",
    }

    for sname in source_names:
        stats = aggregate_frame_stats(per_frame_all[sname])
        all_results[sname] = {"overall": stats}

        rmse_s = f"{stats['rmse_mean']:.4f} ± {stats['rmse_ci95']:.4f}"
        mae_s = f"{stats['mae_mean']:.4f}"
        arel_s = f"{stats['abs_rel_mean']:.4f}"
        d125_s = f"{stats['delta_125_mean']:.1f}%"
        cov_s = coverage_map[sname]
        print(f"  {source_labels[sname]:25s} {rmse_s:>14s} {mae_s:>12s} "
              f"{arel_s:>10s} {d125_s:>10s} {cov_s:>10s}")

    # --- Per-bin breakdown ---
    for lo, hi, blabel in BINS:
        name = {"near": "NEAR (0.3-1.0m)", "mid": "MID (1.0-2.0m)",
                "far": "FAR (2.0-4.0m)"}[blabel]
        print(f"\n{'─' * 80}")
        print(f"  {name}")
        print(f"{'─' * 80}")
        print(f"  {'Source':25s} {'RMSE (m)':>14s} {'MAE (m)':>12s} "
              f"{'AbsRel':>10s} {'δ<1.25':>10s}")
        print("  " + "─" * 70)

        for sname in source_names:
            stats = aggregate_frame_stats(per_frame_bins[sname][blabel])
            all_results[sname][blabel] = stats
            if not stats:
                print(f"  {source_labels[sname]:25s}  (no data)")
                continue

            rmse_s = f"{stats['rmse_mean']:.4f} ± {stats['rmse_ci95']:.4f}"
            mae_s = f"{stats.get('mae_mean', float('nan')):.4f}"
            arel_s = f"{stats.get('abs_rel_mean', float('nan')):.4f}"
            d125_s = f"{stats.get('delta_125_mean', float('nan')):.1f}%"
            print(f"  {source_labels[sname]:25s} {rmse_s:>14s} {mae_s:>12s} "
                  f"{arel_s:>10s} {d125_s:>10s}")

    # --- Dead-pixel cross-model agreement ---
    print(f"\n{'─' * 80}")
    print("  DEAD-PIXEL ANALYSIS (77% of image -- no sensor GT)")
    print(f"{'─' * 80}")
    if dead_pixel_agreement:
        cross_rmse_vals = [d["cross_rmse"] for d in dead_pixel_agreement]
        cross_mae_vals = [d["cross_mae"] for d in dead_pixel_agreement]
        n = len(cross_rmse_vals)
        mean_cross_rmse = np.mean(cross_rmse_vals)
        std_cross_rmse = np.std(cross_rmse_vals, ddof=1) if n > 1 else 0
        ci_cross_rmse = 1.96 * std_cross_rmse / math.sqrt(n) if n > 1 else 0
        mean_cross_mae = np.mean(cross_mae_vals)

        print(f"  V9-vs-DA3 disagreement on dead pixels (no GT available):")
        print(f"    Cross-model RMSE: {mean_cross_rmse:.4f} ± {ci_cross_rmse:.4f} m")
        print(f"    Cross-model MAE:  {mean_cross_mae:.4f} m")
        print(f"    Interpretation: where sensor has no data, V9 and DA3")
        print(f"    disagree by ~{mean_cross_rmse:.2f}m. Lower = more consistent fill.")

        all_results["dead_pixel_analysis"] = {
            "cross_rmse_mean": float(mean_cross_rmse),
            "cross_rmse_std": float(std_cross_rmse),
            "cross_rmse_ci95": float(ci_cross_rmse),
            "cross_mae_mean": float(mean_cross_mae),
            "n_frames": n,
        }

    # --- Coverage summary ---
    print(f"\n{'─' * 80}")
    print("  COVERAGE SUMMARY")
    print(f"{'─' * 80}")
    print(f"  {'Source':25s} {'Coverage':>10s}")
    print("  " + "─" * 35)
    cov_rows = [
        ("Sensor only", f"{np.mean(coverage_stats['sensor_valid_pct']):.1f}%"),
        ("DA3-Small", f"{np.mean(coverage_stats['da3_fill_pct']):.1f}%"),
        ("V9 student", f"{np.mean(coverage_stats['v9_fill_pct']):.1f}%"),
        ("Fused (Sensor+DA3)", "100.0%"),
        ("Fused (Sensor+V9)", "100.0%"),
    ]
    for label, cov in cov_rows:
        print(f"  {label:25s} {cov:>10s}")

    all_results["coverage"] = {
        "sensor_valid_pct_mean": float(np.mean(coverage_stats["sensor_valid_pct"])),
        "dead_pct_mean": float(np.mean(coverage_stats["dead_pct"])),
    }

    # ==================================================================
    # LaTeX table
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("  LaTeX table rows (tab:pixel-fusion)")
    print(f"{'=' * 80}")

    for sname in source_names:
        stats = all_results[sname]["overall"]
        label = source_labels[sname].replace("δ", r"$\delta$")
        rmse = stats["rmse_mean"]
        mae = stats["mae_mean"]
        arel = stats["abs_rel_mean"]
        d125 = stats["delta_125_mean"]
        cov = coverage_map[sname]
        print(f"  {label} & {rmse:.3f} & {mae:.3f} & {arel:.3f} "
              f"& {d125:.1f}\\% & {cov} \\\\")

    # ==================================================================
    # Save JSON
    # ==================================================================
    json_path = out_dir / "pixel_fusion.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nJSON results: {json_path}")

    # ==================================================================
    # Save per-frame CSV
    # ==================================================================
    csv_path = out_dir / "pixel_fusion_per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_cols = ["frame_idx"]
        for sname in source_names:
            for metric in ["rmse", "mae", "abs_rel", "delta_125"]:
                header_cols.append(f"{sname}_{metric}")
        header_cols.extend(["sensor_valid_pct", "dead_pct"])
        writer.writerow(header_cols)

        for i in range(n_frames):
            row = [i]
            for sname in source_names:
                if i < len(per_frame_all[sname]):
                    fm = per_frame_all[sname][i]
                    row.extend([fm["rmse"], fm["mae"],
                                fm["abs_rel"], fm["delta_125"]])
                else:
                    row.extend([None] * 4)
            row.append(coverage_stats["sensor_valid_pct"][i])
            row.append(coverage_stats["dead_pct"][i])
            writer.writerow(row)
    print(f"Per-frame CSV: {csv_path}")

    # ==================================================================
    # Generate figures
    # ==================================================================
    if args.save_figures and frame_data_for_figures:
        print(f"\n{'=' * 80}")
        print("  GENERATING COMPARISON FIGURES")
        print(f"{'=' * 80}")

        if args.figure_frames:
            indices = args.figure_frames
        else:
            indices = auto_select_frames(frame_data_for_figures)

        print(f"  Selected frames: {indices}")
        generate_figures(frame_data_for_figures, out_dir, indices)

    print(f"\n{'=' * 80}")
    print("  FUSION STUDY COMPLETE")
    print(f"{'=' * 80}")


def auto_select_frames(frame_data):
    """Pick 5 representative frames: low/mid/high dead-pixel %, plus extremes."""
    dead_pcts = []
    for fd in frame_data:
        dead_pcts.append((~fd["sensor_valid"]).mean())
    dead_pcts = np.array(dead_pcts)

    indices = []
    # Frame with lowest dead pixel % (best sensor coverage)
    indices.append(int(np.argmin(dead_pcts)))
    # 25th percentile
    indices.append(int(np.argsort(dead_pcts)[len(dead_pcts) // 4]))
    # Median
    indices.append(int(np.argsort(dead_pcts)[len(dead_pcts) // 2]))
    # 75th percentile
    indices.append(int(np.argsort(dead_pcts)[3 * len(dead_pcts) // 4]))
    # Frame with highest dead pixel % (worst sensor coverage)
    indices.append(int(np.argmax(dead_pcts)))

    return sorted(set(indices))


if __name__ == "__main__":
    main()
