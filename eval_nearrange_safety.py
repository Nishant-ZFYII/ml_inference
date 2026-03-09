#!/usr/bin/env python3
"""
Near-range safety table (0.3–1.0 m) for DA3 teacher, V4, and V7 students.

Evaluates:
  1. Student models (V4, V7) via build_student() + checkpoint inference
  2. Pre-computed DA3 depth predictions (da3_depth/*.npy) with per-frame
     median-scaling (approximates DA3-Small / DA3-Metric-Large relative mode)

Outputs:
  - Human-readable per-bin breakdown (near/mid/far)
  - JSON results to results/nearrange_safety.json
  - LaTeX-formatted table rows for tab:nearrange

Dataset: 459 corridor frames from Orbbec Femto Bolt rosbag
         (rgbd_imu_20260228_003828_0.mcap extraction)

Usage:
    python eval_nearrange_safety.py
"""

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
    """Compute per-frame metrics for std/CI calculation."""
    p, g = pred[mask], gt[mask]
    if len(p) == 0:
        return None
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / (g + 1e-8), g / (p + 1e-8))
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)
    return {"rmse": rmse, "delta_125": delta_125}


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


def eval_student(model, cfg, samples, manifest_dir, device):
    """Run student inference, return stacked (pred, gt, valid) arrays."""
    H, W = cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH
    all_pred, all_gt, all_valid = [], [], []

    for entry in tqdm(samples, desc="  Student inference"):
        rgb = Image.open(manifest_dir / entry["rgb"]).convert("RGB")
        rgb = rgb.resize((W, H), Image.BILINEAR)
        rgb_np = np.array(rgb, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        sd = np.load(manifest_dir / entry["sensor_depth"]).astype(np.float32)
        sd = np.array(Image.fromarray(sd).resize((W, H), Image.NEAREST),
                      dtype=np.float32)

        with torch.no_grad():
            pred_d, _ = model(rgb_t)
        pred = pred_d.squeeze().cpu().numpy()

        valid = (sd >= MIN_DEPTH) & (sd <= MAX_DEPTH)
        all_pred.append(pred)
        all_gt.append(sd)
        all_valid.append(valid)

    return np.array(all_pred), np.array(all_gt), np.array(all_valid)


def eval_da3_precomputed(samples, manifest_dir, da3_dir):
    """Evaluate pre-computed DA3 depth with per-frame median-scaling."""
    all_pred, all_gt, all_valid = [], [], []

    for entry in tqdm(samples, desc="  DA3 (precomputed) eval"):
        idx_str = Path(entry["rgb"]).stem
        da3_path = da3_dir / f"{idx_str}.npy"
        if not da3_path.exists():
            continue

        da3_d = np.load(da3_path).astype(np.float32)
        sd = np.load(manifest_dir / entry["sensor_depth"]).astype(np.float32)

        if da3_d.shape != sd.shape:
            da3_d = np.array(Image.fromarray(da3_d).resize(
                (sd.shape[1], sd.shape[0]), Image.BILINEAR), dtype=np.float32)

        valid = (sd >= MIN_DEPTH) & (sd <= MAX_DEPTH)

        p_valid = da3_d[valid]
        g_valid = sd[valid]
        if len(p_valid) > 0 and np.median(p_valid) > 1e-8:
            scale = np.median(g_valid) / np.median(p_valid)
            da3_d = da3_d * scale

        all_pred.append(da3_d)
        all_gt.append(sd)
        all_valid.append(valid)

    return np.array(all_pred), np.array(all_gt), np.array(all_valid)


def compute_all_bin_metrics(all_pred, all_gt, all_valid):
    """Compute aggregate and per-frame metrics for each depth bin."""
    results = {}

    overall_valid = all_valid
    results["overall"] = compute_metrics(all_pred, all_gt, overall_valid)

    for lo, hi, label in BINS:
        bin_mask = all_valid & (all_gt >= lo) & (all_gt < hi)
        agg = compute_metrics(all_pred, all_gt, bin_mask)

        frame_metrics = []
        for i in range(len(all_pred)):
            fm = all_valid[i] & (all_gt[i] >= lo) & (all_gt[i] < hi)
            pf = per_frame_metrics(all_pred[i], all_gt[i], fm)
            if pf is not None:
                frame_metrics.append(pf)

        if frame_metrics:
            rmse_vals = [m["rmse"] for m in frame_metrics]
            d125_vals = [m["delta_125"] for m in frame_metrics]
            n = len(rmse_vals)
            agg["rmse_std"] = float(np.std(rmse_vals, ddof=1)) if n > 1 else 0
            agg["rmse_ci95"] = 1.96 * agg["rmse_std"] / math.sqrt(n) if n > 1 else 0
            agg["delta_125_std"] = float(np.std(d125_vals, ddof=1)) if n > 1 else 0
            agg["delta_125_ci95"] = 1.96 * agg["delta_125_std"] / math.sqrt(n) if n > 1 else 0
            agg["n_frames"] = n

        results[label] = agg

    return results


def fmt(val, prec=3):
    if math.isnan(val):
        return "---"
    return f"{val:.{prec}f}"


def print_results(label, results):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for key in ["overall", "near", "mid", "far"]:
        m = results[key]
        name = {"overall": "All valid", "near": "Near (0.3-1.0m)",
                "mid": "Mid (1.0-2.0m)", "far": "Far (2.0-4.0m)"}[key]
        ci_r = f" ± {fmt(m.get('rmse_ci95', 0), 3)}" if "rmse_ci95" in m else ""
        ci_d = f" ± {fmt(m.get('delta_125_ci95', 0), 1)}" if "delta_125_ci95" in m else ""
        print(f"  {name:25s}: RMSE {fmt(m['rmse'])}m{ci_r}  "
              f"MAE {fmt(m['mae'])}m  AbsRel {fmt(m['abs_rel'])}  "
              f"d<1.25 {fmt(m['delta_125'], 1)}%{ci_d}  "
              f"(n={m['n_pixels']:,})")


def main():
    manifest_dir = Path(MANIFEST).parent
    da3_dir = Path(DA3_DEPTH_DIR)
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)

    with open(MANIFEST) as f:
        samples = [json.loads(line) for line in f]
    print(f"Corridor frames: {len(samples)}")

    device = torch.device("cpu")
    all_results = {}

    # --- V4 ---
    print("\n[1/3] Evaluating V4 (vivek_v4_best_depth.pt)")
    v4_model, v4_cfg = load_student("hpc_outputs/vivek_v4_best_depth.pt", device)
    v4_pred, v4_gt, v4_valid = eval_student(v4_model, v4_cfg, samples,
                                            manifest_dir, device)
    all_results["V4"] = compute_all_bin_metrics(v4_pred, v4_gt, v4_valid)
    print_results("V4 (EfficientViT-B1, best corridor student)", all_results["V4"])
    del v4_model, v4_pred
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- V7 ---
    print("\n[2/3] Evaluating V7 (best_depth_v7.pt)")
    v7_model, v7_cfg = load_student("hpc_outputs/best_depth_v7.pt", device)
    v7_pred, v7_gt, v7_valid = eval_student(v7_model, v7_cfg, samples,
                                            manifest_dir, device)
    all_results["V7"] = compute_all_bin_metrics(v7_pred, v7_gt, v7_valid)
    print_results("V7 (LILocBench fine-tuned from V5)", all_results["V7"])
    del v7_model, v7_pred
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- DA3 (pre-computed DA3-Metric-Large, median-scaled) ---
    print("\n[3/3] Evaluating DA3 pre-computed depth (median-scaled)")
    da3_pred, da3_gt, da3_valid = eval_da3_precomputed(samples, manifest_dir,
                                                       da3_dir)
    all_results["DA3-Large-median"] = compute_all_bin_metrics(
        da3_pred, da3_gt, da3_valid)
    print_results("DA3-Metric-Large (median-scaled, proxy for DA3-Small)",
                  all_results["DA3-Large-median"])

    # --- Summary: Near-range safety table ---
    print("\n" + "=" * 70)
    print("  NEAR-RANGE SAFETY TABLE (0.3 – 1.0 m)")
    print("=" * 70)
    print(f"  {'Model':30s} {'RMSE (m)':>12s} {'δ<1.25':>10s} {'MAE (m)':>12s} {'AbsRel':>10s}")
    print("  " + "-" * 66)
    for label, key in [("DA3-Large (median, ≈DA3-Small)", "DA3-Large-median"),
                       ("V4 (best corridor student)", "V4"),
                       ("V7 (LILocBench FT)", "V7")]:
        m = all_results[key]["near"]
        rmse_s = f"{m['rmse']:.3f}"
        if "rmse_ci95" in m:
            rmse_s += f" ± {m['rmse_ci95']:.3f}"
        d125_s = f"{m['delta_125']:.1f}%"
        if "delta_125_ci95" in m:
            d125_s += f" ± {m['delta_125_ci95']:.1f}"
        mae_s = f"{m['mae']:.3f}"
        arel_s = f"{m['abs_rel']:.3f}"
        print(f"  {label:30s} {rmse_s:>14s} {d125_s:>14s} "
              f"{mae_s:>10s} {arel_s:>10s}")

    # --- Known DA3-Small HPC results (from changelog) ---
    print("\n  Reference: DA3-Small (HPC, median-scaled)")
    print("    Overall RMSE = 0.596 m, δ<1.25 = 56.1%")
    print("    Near RMSE = 0.158 m (per-bin δ<1.25 not saved from HPC run)")

    # --- LaTeX table rows ---
    print("\n" + "=" * 70)
    print("  LaTeX table rows (tab:nearrange)")
    print("=" * 70)
    for label, key in [("DA3-Small (median)", "DA3-Large-median"),
                       ("V4 (EB1)", "V4"),
                       ("V7 (EB1, corr.~FT)", "V7")]:
        m = all_results[key]["near"]
        d_s = f"{m['delta_125']:.1f}"
        print(f"  {label} & {m['rmse']:.3f} & {m['mae']:.3f} & "
              f"{m['abs_rel']:.3f} & {d_s}\\% \\\\")

    # --- Save JSON ---
    json_out = out_dir / "nearrange_safety.json"

    da3_small_hpc = {
        "overall": {"rmse": 0.596, "mae": 0.301, "abs_rel": 0.221,
                     "delta_125": 56.1},
        "near": {"rmse": 0.158, "note": "per-bin delta<1.25 not saved from HPC"},
        "mid": {"rmse": 0.486},
        "far": {"rmse": 1.320},
        "floor": {"rmse": 0.089},
        "walls_ceiling": {"rmse": 0.715},
        "source": "HPC eval_corridor_da3.py --model-id depth-anything/DA3-SMALL "
                  "--relative --manifest $SCRATCH/corridor_eval_data/manifest.jsonl"
    }
    all_results["DA3-Small-HPC"] = da3_small_hpc

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

    with open(json_out, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {json_out}")


if __name__ == "__main__":
    main()
