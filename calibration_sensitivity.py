#!/usr/bin/env python3
"""
Calibration Sensitivity Experiment
===================================
Addresses reviewer concern: if affine alignment parameters (α, β) are
estimated on the same frames being tested, accuracy numbers are
optimistically biased.

This script:
  1. Within-scene: splits each dataset into a calibration set (N frames)
     and a disjoint test set. Computes α, β via least-squares on the
     calibration set, applies the FIXED pair to the test set. Sweeps
     N ∈ {1, 3, 5, 10, 25, 50, 100, 200} and reports how AbsRel / RMSE
     degrade relative to per-frame (oracle) alignment.
  2. Cross-scene: calibrates on one dataset, tests on another.
  3. Temporal stability: measures how stable α, β are across frames.

Datasets used:
  - corridor_eval_data (459 frames, LILocBench corridor)
  - glass_corridor_frames (121 frames, glass corridor bag 173610)
  - bag_213831_frames (150 frames, corridor bag 213831)

Output: JSON results + LaTeX-ready table.

Usage:
    python calibration_sensitivity.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

MIN_DEPTH = 0.1
MAX_DEPTH = 5.0


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Data Loading ─────────────────────────────────────────────────────────────
# Frames are loaded lazily via file paths and downsampled to 240x320
# to keep peak memory under ~500 MB even for the 459-frame corridor set.

DS_H, DS_W = 240, 320


class LazyFrame:
    """Stores paths, loads and downsamples on demand."""
    __slots__ = ("sensor_path", "da3_path", "_sensor", "_da3")

    def __init__(self, sensor_path, da3_path):
        self.sensor_path = sensor_path
        self.da3_path = da3_path
        self._sensor = None
        self._da3 = None

    @property
    def sensor_depth(self):
        if self._sensor is None:
            raw = np.load(str(self.sensor_path))
            self._sensor = cv2.resize(raw, (DS_W, DS_H),
                                      interpolation=cv2.INTER_NEAREST)
        return self._sensor

    @property
    def da3_depth(self):
        if self._da3 is None:
            raw = np.load(str(self.da3_path))
            self._da3 = cv2.resize(raw, (DS_W, DS_H),
                                   interpolation=cv2.INTER_LINEAR)
        return self._da3

    def evict(self):
        self._sensor = None
        self._da3 = None


def _frame_dict(lf):
    return {"sensor_depth": lf.sensor_depth, "da3_depth": lf.da3_depth}


def load_corridor_eval():
    """Load 459-frame corridor_eval_data (LILocBench corridor) lazily."""
    base = Path(__file__).parent / "corridor_eval_data"
    with open(base / "manifest.jsonl") as f:
        entries = [json.loads(line) for line in f]
    frames = []
    for e in entries:
        depth_path = base / e["sensor_depth"]
        da3_path = base / "da3_depth" / (Path(e["sensor_depth"]).name)
        if not da3_path.exists() or not depth_path.exists():
            continue
        frames.append(LazyFrame(depth_path, da3_path))
    return frames


def load_npy_dataset(frames_dir, da3_dir):
    """Load a dataset with depth/ and da3_depth/ subdirs lazily."""
    frames_dir = Path(frames_dir)
    da3_dir = Path(da3_dir)

    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        frame_names = [m["frame"] for m in manifest]
    else:
        frame_names = sorted([p.stem for p in (frames_dir / "rgb").glob("*.png")])

    frames = []
    for fname in frame_names:
        depth_path = frames_dir / "depth" / f"{fname}.npy"
        da3_path = da3_dir / f"{fname}.npy"
        if not depth_path.exists() or not da3_path.exists():
            continue
        frames.append(LazyFrame(depth_path, da3_path))
    return frames


# ── Alignment Methods ────────────────────────────────────────────────────────

def get_valid_mask(sensor, da3):
    return ((sensor >= MIN_DEPTH) & (sensor <= MAX_DEPTH) &
            ~np.isnan(sensor) & (da3 > 1e-6) & ~np.isnan(da3))


def per_frame_median_scale(da3, sensor, valid):
    """Per-frame median scaling (what the code actually does)."""
    if valid.sum() < 50:
        return da3.copy()
    scale = np.median(sensor[valid]) / np.median(da3[valid])
    return da3 * scale


def least_squares_affine(da3_vals, sensor_vals):
    """Compute α, β via least-squares: sensor ≈ α·da3 + β."""
    A = np.column_stack([da3_vals, np.ones(len(da3_vals))])
    result = np.linalg.lstsq(A, sensor_vals, rcond=None)
    alpha, beta = result[0]
    return alpha, beta


def compute_calibration_params(frames):
    """Compute α, β from a set of frames (pooled least-squares)."""
    all_da3 = []
    all_sensor = []
    subsample_rng = np.random.default_rng(42)
    for f in frames:
        sensor = f.sensor_depth if isinstance(f, LazyFrame) else f["sensor_depth"]
        da3 = f.da3_depth if isinstance(f, LazyFrame) else f["da3_depth"]
        valid = get_valid_mask(sensor, da3)
        if valid.sum() < 100:
            continue
        valid_idx = np.where(valid.ravel())[0]
        if len(valid_idx) > 3000:
            valid_idx = subsample_rng.choice(valid_idx, 3000, replace=False)
        all_da3.extend(da3.ravel()[valid_idx])
        all_sensor.extend(sensor.ravel()[valid_idx])

    all_da3 = np.array(all_da3)
    all_sensor = np.array(all_sensor)
    alpha, beta = least_squares_affine(all_da3, all_sensor)
    return alpha, beta


def apply_affine(da3, alpha, beta):
    return alpha * da3 + beta


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, valid):
    if valid.sum() < 50:
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


def evaluate_frames(frames, alpha=None, beta=None, per_frame=False):
    """Evaluate depth accuracy. If per_frame=True, use per-frame median
    scaling (oracle). Otherwise use fixed α, β."""
    all_m = []
    for f in frames:
        sensor = f.sensor_depth if isinstance(f, LazyFrame) else f["sensor_depth"]
        da3 = f.da3_depth if isinstance(f, LazyFrame) else f["da3_depth"]
        valid = get_valid_mask(sensor, da3)

        if per_frame:
            aligned = per_frame_median_scale(da3, sensor, valid)
        else:
            aligned = apply_affine(da3, alpha, beta)

        m = compute_metrics(aligned, sensor, valid)
        all_m.append(m)

    # Aggregate
    def avg(key):
        vals = [m[key] for m in all_m if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float('nan')

    def std(key):
        vals = [m[key] for m in all_m if not np.isnan(m[key])]
        return float(np.std(vals)) if vals else float('nan')

    return {
        "rmse": avg("rmse"), "rmse_std": std("rmse"),
        "mae": avg("mae"), "mae_std": std("mae"),
        "absrel": avg("absrel"), "absrel_std": std("absrel"),
        "delta1": avg("delta1"), "delta1_std": std("delta1"),
        "n_frames": len(all_m),
    }


# ── Experiment 1: Within-Scene Calibration Sweep ─────────────────────────────

def within_scene_sweep(name, frames, n_values, n_trials=5):
    """Sweep calibration set size N, report test-set metrics."""
    rng = np.random.default_rng(42)
    n_total = len(frames)

    # Oracle: per-frame alignment
    oracle = evaluate_frames(frames, per_frame=True)
    print(f"\n  Oracle (per-frame): RMSE={oracle['rmse']:.4f}, "
          f"AbsRel={oracle['absrel']:.4f}, d<1.25={oracle['delta1']:.1f}%")

    results = {"oracle": oracle, "sweeps": []}

    for N in n_values:
        if N >= n_total:
            continue

        trial_metrics = []
        trial_alphas = []
        trial_betas = []

        for trial in range(n_trials):
            # Random split
            idx = rng.permutation(n_total)
            cal_idx = idx[:N]
            test_idx = idx[N:]

            cal_frames = [frames[i] for i in cal_idx]
            test_frames = [frames[i] for i in test_idx]

            alpha, beta = compute_calibration_params(cal_frames)
            trial_alphas.append(alpha)
            trial_betas.append(beta)

            m = evaluate_frames(test_frames, alpha=alpha, beta=beta)
            trial_metrics.append(m)

        # Average across trials
        avg_rmse = np.mean([m["rmse"] for m in trial_metrics])
        avg_absrel = np.mean([m["absrel"] for m in trial_metrics])
        avg_d1 = np.mean([m["delta1"] for m in trial_metrics])
        std_rmse = np.std([m["rmse"] for m in trial_metrics])

        rmse_degradation = (avg_rmse - oracle["rmse"]) / oracle["rmse"] * 100

        entry = {
            "N": N,
            "n_test": n_total - N,
            "n_trials": n_trials,
            "rmse": float(avg_rmse),
            "rmse_std_across_trials": float(std_rmse),
            "absrel": float(avg_absrel),
            "delta1": float(avg_d1),
            "alpha_mean": float(np.mean(trial_alphas)),
            "alpha_std": float(np.std(trial_alphas)),
            "beta_mean": float(np.mean(trial_betas)),
            "beta_std": float(np.std(trial_betas)),
            "rmse_degradation_pct": float(rmse_degradation),
        }
        results["sweeps"].append(entry)

        print(f"  N={N:>3}: RMSE={avg_rmse:.4f} (+{rmse_degradation:+.1f}%), "
              f"AbsRel={avg_absrel:.4f}, d<1.25={avg_d1:.1f}%, "
              f"α={np.mean(trial_alphas):.3f}±{np.std(trial_alphas):.3f}, "
              f"β={np.mean(trial_betas):.3f}±{np.std(trial_betas):.3f}")

    return results


# ── Experiment 2: Cross-Scene Test ───────────────────────────────────────────

def cross_scene_test(datasets):
    """Calibrate on each dataset, test on all others."""
    results = {}
    names = list(datasets.keys())

    for cal_name in names:
        cal_frames = datasets[cal_name]
        alpha, beta = compute_calibration_params(cal_frames)
        print(f"\n  Calibrated on {cal_name}: α={alpha:.4f}, β={beta:.4f}")

        results[cal_name] = {"alpha": float(alpha), "beta": float(beta), "tests": {}}

        for test_name in names:
            test_frames = datasets[test_name]
            m = evaluate_frames(test_frames, alpha=alpha, beta=beta)
            oracle = evaluate_frames(test_frames, per_frame=True)
            degradation = ((m["rmse"] - oracle["rmse"]) / oracle["rmse"] * 100
                           if not np.isnan(oracle["rmse"]) else float('nan'))

            results[cal_name]["tests"][test_name] = {
                "fixed": m,
                "oracle": oracle,
                "rmse_degradation_pct": float(degradation),
            }

            marker = " <-- SAME" if cal_name == test_name else ""
            print(f"    → test {test_name}: "
                  f"RMSE={m['rmse']:.4f} (oracle={oracle['rmse']:.4f}, "
                  f"degrad={degradation:+.1f}%){marker}")

    return results


# ── Experiment 3: Temporal Stability ─────────────────────────────────────────

def temporal_stability(name, frames):
    """Per-frame α, β to show how stable alignment is."""
    alphas = []
    betas = []
    scales = []
    for f in frames:
        sensor = f.sensor_depth if isinstance(f, LazyFrame) else f["sensor_depth"]
        da3 = f.da3_depth if isinstance(f, LazyFrame) else f["da3_depth"]
        valid = get_valid_mask(sensor, da3)
        if valid.sum() < 200:
            continue

        alpha, beta = least_squares_affine(da3[valid], sensor[valid])
        alphas.append(alpha)
        betas.append(beta)

        median_scale = np.median(sensor[valid]) / np.median(da3[valid])
        scales.append(median_scale)

    result = {
        "n_frames": len(alphas),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "alpha_cv": float(np.std(alphas) / np.mean(alphas)) if np.mean(alphas) != 0 else 0,
        "beta_mean": float(np.mean(betas)),
        "beta_std": float(np.std(betas)),
        "median_scale_mean": float(np.mean(scales)),
        "median_scale_std": float(np.std(scales)),
        "median_scale_cv": float(np.std(scales) / np.mean(scales)) if np.mean(scales) != 0 else 0,
    }

    print(f"\n  {name} ({len(alphas)} frames):")
    print(f"    α = {result['alpha_mean']:.4f} ± {result['alpha_std']:.4f} "
          f"(CV = {result['alpha_cv']:.3f})")
    print(f"    β = {result['beta_mean']:.4f} ± {result['beta_std']:.4f}")
    print(f"    median_scale = {result['median_scale_mean']:.4f} "
          f"± {result['median_scale_std']:.4f} "
          f"(CV = {result['median_scale_cv']:.3f})")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("/home/nishant/maps/calibration_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    datasets = {}

    print("  Loading corridor_eval_data (LILocBench, 459 frames)...")
    datasets["corridor_459"] = load_corridor_eval()
    print(f"    Loaded {len(datasets['corridor_459'])} frames")

    print("  Loading glass_corridor (bag 173610, 121 frames)...")
    datasets["glass_121"] = load_npy_dataset(
        "/home/nishant/maps/glass_corridor_frames",
        "/home/nishant/maps/glass_corridor_frames/da3_depth")
    print(f"    Loaded {len(datasets['glass_121'])} frames")

    print("  Loading bag_213831 (150 frames)...")
    datasets["bag213831_150"] = load_npy_dataset(
        "/home/nishant/maps/bag_213831_frames",
        "/home/nishant/maps/bag_213831_student_results/depth_da3")
    print(f"    Loaded {len(datasets['bag213831_150'])} frames")

    all_results = {}

    # Experiment 1: Within-scene calibration sweep
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Within-Scene Calibration Set Size Sweep")
    print("=" * 70)

    n_values = [1, 3, 5, 10, 25, 50, 100, 200]

    for name, frames in datasets.items():
        print(f"\n--- {name} ({len(frames)} frames) ---")
        applicable_n = [n for n in n_values if n < len(frames)]
        all_results[f"within_{name}"] = within_scene_sweep(
            name, frames, applicable_n)

    # Experiment 2: Cross-scene
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cross-Scene Calibration Transfer")
    print("=" * 70)
    all_results["cross_scene"] = cross_scene_test(datasets)

    # Experiment 3: Temporal stability
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Temporal Stability of α, β")
    print("=" * 70)
    stability = {}
    for name, frames in datasets.items():
        stability[name] = temporal_stability(name, frames)
    all_results["temporal_stability"] = stability

    # Save JSON
    with open(output_dir / "calibration_sensitivity_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # Print LaTeX tables
    print("\n" + "=" * 70)
    print("LATEX OUTPUT")
    print("=" * 70)

    # Table 1: Within-scene sweep (corridor_459 as primary)
    print("\n% Within-scene calibration sensitivity (corridor, 459 frames)")
    print(r"\begin{tabular}{rcccc}")
    print(r"\toprule")
    print(r"$N_\text{cal}$ & RMSE (m) & AbsRel & $\delta_{1.25}$ & Degrad. \\")
    print(r"\midrule")
    key = "within_corridor_459"
    if key in all_results:
        oracle = all_results[key]["oracle"]
        print(f"Oracle & {oracle['rmse']:.3f} & {oracle['absrel']:.3f} "
              f"& {oracle['delta1']:.1f}\\% & --- \\\\")
        for s in all_results[key]["sweeps"]:
            print(f"{s['N']} & {s['rmse']:.3f} & {s['absrel']:.3f} "
                  f"& {s['delta1']:.1f}\\% & {s['rmse_degradation_pct']:+.1f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Table 2: Cross-scene
    print("\n% Cross-scene calibration transfer (RMSE)")
    cs = all_results.get("cross_scene", {})
    names = list(cs.keys())
    print(r"\begin{tabular}{l" + "c" * len(names) + "}")
    print(r"\toprule")
    header = " & ".join([r"\textbf{Test: " + n + "}" for n in names])
    print(r"\textbf{Cal.\ set} & " + header + r" \\")
    print(r"\midrule")
    for cal in names:
        vals = []
        for test in names:
            t = cs[cal]["tests"][test]
            rmse = t["fixed"]["rmse"]
            degrad = t["rmse_degradation_pct"]
            if cal == test:
                vals.append(f"\\underline{{{rmse:.3f}}} ({degrad:+.0f}\\%)")
            else:
                vals.append(f"{rmse:.3f} ({degrad:+.0f}\\%)")
        print(f"{cal} & " + " & ".join(vals) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print(f"\nResults saved to {output_dir / 'calibration_sensitivity_results.json'}")


if __name__ == "__main__":
    main()
