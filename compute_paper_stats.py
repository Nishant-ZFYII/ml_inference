#!/usr/bin/env python3
"""
Compute mean +/- std and 95% CI for all four core paper tables.

Outputs LaTeX-ready table rows to stdout and a JSON summary to
results/paper_stats.json for downstream insertion into the .tex file.

Tables:
  III  -- Offline corridor costmap ablation (459 frames x 7 configs)
  IV   -- Live corridor replay (5 configs)
  V    -- LILocBench dynamics_0 costmap (5 configs)
  VI   -- Localization on LILocBench dynamics_0 (5 configs)
"""

import json
import math
import pathlib
import sys

import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
MAPS = pathlib.Path.home() / "maps"

# Paper's current L reference for Table V (711 frames, 1582 mean).
# Source data not found in either lilocbench_ablation_results or
# lilocbench_benchmark; provenance must be resolved before updating.
PAPER_TABLE_V_L = {"frames": 711, "mean": 1582}


def ci95(std: float, n: int) -> float:
    """95% CI half-width for the mean."""
    if n <= 1:
        return float("nan")
    return 1.96 * std / math.sqrt(n)


def delta_pct_with_ci(mean_x, std_x, n_x, mean_y, std_y, n_y):
    """
    Returns (delta%, ci_half_width%) for delta = (X - Y) / Y * 100.
    Uses first-order error propagation (independent samples).
    """
    if mean_y == 0:
        return float("nan"), float("nan")
    delta = (mean_x - mean_y) / mean_y * 100
    se_x = std_x / math.sqrt(n_x) if n_x > 1 else 0
    se_y = std_y / math.sqrt(n_y) if n_y > 1 else 0
    var_delta = (se_x ** 2 + (mean_x / mean_y) ** 2 * se_y ** 2) / mean_y ** 2
    ci = 1.96 * math.sqrt(var_delta) * 100 if var_delta > 0 else 0
    return delta, ci


def fmt_delta(delta, ci):
    sign = "+" if delta >= 0 else "$-$"
    return f"{sign}{abs(delta):.0f}\\% $\\pm$ {ci:.0f}\\%"


# ═══════════════════════════════════════════════════════════════════════
#  TABLE III — Offline Corridor Costmap Ablation (459 frames)
# ═══════════════════════════════════════════════════════════════════════

def table_iii():
    csv_path = (ROOT / "results" / "costmap_ablation" / "corridor"
                / "per_frame_metrics.csv")
    df = pd.read_csv(csv_path)

    config_order = ["Baseline", "A1", "A2", "A3", "A4", "A5", "A6"]
    display_names = {"Baseline": "Base"}

    metrics = [
        # (csv_col,      latex_hdr,   decimals, suffix, scale)
        ("iou",                 "IoU",       3, "",     1),
        ("detection_rate",      "Det.\\%",   1, "",     100),
        ("false_positive_rate", "FPR",       1, "\\%",  100),
        ("clearance_pred_m",    "Clear.",    3, "\\,m", 1),
        ("inflation_radius_m",  "Rad.",      3, "\\,m", 1),
        ("timing_ms",           "ms",        0, "",     1),
    ]

    results = {}
    print("=" * 80)
    print("TABLE III — Offline Corridor Costmap Ablation")
    print("=" * 80)

    header = " & ".join([""] + [m[1] for m in metrics]) + " \\\\"
    print(header)
    print("\\midrule")

    for cfg in config_order:
        sub = df[df["experiment"] == cfg]
        n = len(sub)
        label = display_names.get(cfg, cfg)
        row = {"n": n}
        parts = [f"{label:<4s}"]

        for col, _, dec, suf, scale in metrics:
            vals = sub[col].values * scale
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            row[col] = {
                "mean": round(m, dec + 2),
                "std": round(s, dec + 2),
                "ci95": round(ci95(s, n), dec + 2),
            }
            parts.append(f"{m:.{dec}f} $\\pm$ {s:.{dec}f}{suf}")

        results[cfg] = row
        print(" & ".join(parts) + " \\\\")

    # Paper-comparison note
    print()
    print("  NOTE: Paper reports FPR = 5.3%; computed mean = 5.2%.")
    print("  Difference is rounding: raw mean FPR ≈ 0.0524 → 5.24%")
    print("  → 5.2% at 1 dp. The 5.3% in the paper likely used a")
    print("  different rounding or a per-frame median.")
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  TABLE IV — Live Corridor Replay (full Nav2 stack)
# ═══════════════════════════════════════════════════════════════════════

def table_iv():
    base_dir = MAPS / "ablation_results"
    config_order = ["L", "L+S", "L+D", "D", "L+D+dyn"]
    display_names = {
        "L": "L (LiDAR)", "L+S": "L+S (+ sensor)",
        "L+D": "L+D (+ DA3)", "D": "D (DA3 only)",
        "L+D+dyn": "L+D+dyn",
    }

    summaries = {}
    for cfg in config_order:
        with open(base_dir / cfg / "summary.json") as f:
            summaries[cfg] = json.load(f)

    ref = summaries["L"]

    results = {}
    print("=" * 80)
    print("TABLE IV — Live Corridor Replay (full Nav2 stack, GTX 1060)")
    print("=" * 80)
    print("Config & Frames & Mean Occ. & $\\Delta$ vs L \\\\")
    print("\\midrule")

    for cfg in config_order:
        s = summaries[cfg]
        n = s["total_frames"]
        m = s["occupied_cells_mean"]
        sd = s["occupied_cells_std"]
        label = display_names[cfg]

        if cfg == "L":
            delta_str = "---"
        else:
            delta, dci = delta_pct_with_ci(
                m, sd, n,
                ref["occupied_cells_mean"], ref["occupied_cells_std"],
                ref["total_frames"],
            )
            delta_str = fmt_delta(delta, dci)

        results[cfg] = {
            "n": n,
            "mean": round(m, 1),
            "std": round(sd, 1),
            "ci95": round(ci95(sd, n), 1),
        }
        occ_str = f"{m:.0f} $\\pm$ {sd:.0f}"
        print(f"{label:<15s} & {n:<4d} & {occ_str:<24s} & {delta_str} \\\\")

    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  TABLE V — LILocBench dynamics_0 Costmap Ablation
# ═══════════════════════════════════════════════════════════════════════

def table_v():
    ablation_dir = MAPS / "lilocbench_ablation_results"
    benchmark_dir = MAPS / "lilocbench_benchmark" / "dynamics_0"
    config_order = ["L", "L+S", "L+D", "D", "L+D+dyn"]

    summaries = {}
    provenance = {}

    for cfg in config_order:
        abl_path = ablation_dir / cfg / "summary.json"
        bench_path = benchmark_dir / cfg / "summary.json"

        if cfg == "L":
            # Known provenance issue: paper says 711/1582, ablation has
            # 33/543.8, benchmark has 749/953.3.  Neither matches.
            if bench_path.exists():
                with open(bench_path) as f:
                    summaries[cfg] = json.load(f)
                provenance[cfg] = (
                    f"benchmark ({summaries[cfg]['total_frames']} frames, "
                    f"mean={summaries[cfg]['occupied_cells_mean']:.0f})"
                    f" — MISMATCH: paper has {PAPER_TABLE_V_L['frames']} "
                    f"frames, mean={PAPER_TABLE_V_L['mean']}"
                )
            continue

        if abl_path.exists():
            with open(abl_path) as f:
                s = json.load(f)
            summaries[cfg] = s
            provenance[cfg] = f"ablation ({s['total_frames']} frames)"

    results = {}
    warnings = []

    print("=" * 80)
    print("TABLE V — LILocBench dynamics_0 Costmap Ablation (10 pedestrians)")
    print("=" * 80)

    # --- Section A: std values from ablation data (matches paper) ---
    print()
    print("--- (A) std for the 4 configs that MATCH the paper ---")
    print("Config & Frames & Mean Occ. $\\pm$ std \\\\")
    print("\\midrule")

    for cfg in ["L+S", "L+D", "D", "L+D+dyn"]:
        s = summaries[cfg]
        n = s["total_frames"]
        m = s["occupied_cells_mean"]
        sd = s["occupied_cells_std"]
        results[cfg] = {
            "n": n,
            "mean": round(m, 1),
            "std": round(sd, 1),
            "ci95": round(ci95(sd, n), 1),
            "source": provenance[cfg],
        }
        occ_str = f"{m:.0f} $\\pm$ {sd:.0f}"
        print(f"{cfg:<8s} & {n:<4d} & {occ_str} \\\\")

    # --- Section B: delta% using paper's L reference ---
    print()
    print("--- (B) Delta% vs L using PAPER reference (L = 1582 mean) ---")
    print("       (Use these for consistency with the paper until")
    print("        L-config provenance is resolved)")
    print("Config & $\\Delta$ vs L \\\\")
    print("\\midrule")

    for cfg in ["L+S", "L+D", "D", "L+D+dyn"]:
        s = summaries[cfg]
        m = s["occupied_cells_mean"]
        delta = (m - PAPER_TABLE_V_L["mean"]) / PAPER_TABLE_V_L["mean"] * 100
        sign = "+" if delta >= 0 else "$-$"
        # Can't compute CI on delta without L's std
        print(f"{cfg:<8s} & {sign}{abs(delta):.0f}\\% (no CI — L std unknown) \\\\")

    # --- Section C: provenance warning ---
    print()
    print("--- (C) PROVENANCE WARNING ---")
    print("  L-config discrepancy:")
    print(f"    Paper:     {PAPER_TABLE_V_L['frames']} frames, "
          f"mean = {PAPER_TABLE_V_L['mean']}")
    if "L" in summaries:
        s = summaries["L"]
        print(f"    Benchmark: {s['total_frames']} frames, "
              f"mean = {s['occupied_cells_mean']:.0f}")
    print(f"    Ablation:  33 frames, mean = 544")
    print("  Action: resolve in table5-provenance task before updating LaTeX.")
    print()

    results["L"] = {
        "n": PAPER_TABLE_V_L["frames"],
        "mean": PAPER_TABLE_V_L["mean"],
        "std": None,
        "ci95": None,
        "source": "paper (original data not found)",
    }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  TABLE VI — Localization on LILocBench dynamics_0
# ═══════════════════════════════════════════════════════════════════════

def table_vi():
    base_dir = MAPS / "lilocbench_benchmark" / "dynamics_0"
    config_order = ["L", "L+S", "L+D", "D", "L+D+dyn"]

    results = {}
    print("=" * 80)
    print("TABLE VI — Localization on LILocBench dynamics_0 (Umeyama SE(2))")
    print("=" * 80)

    # Two formats: mean ± std (spread), and mean ± CI (precision of estimate)
    print()
    print("--- (A) Point estimates ± std (shows spread of per-pose errors) ---")
    print(" & APE Med. & APE Mean & RPE Trans & RPE Rot \\\\")
    print("\\midrule")

    for cfg in config_order:
        loc_path = base_dir / cfg / "localization_results.json"
        with open(loc_path) as f:
            loc = json.load(f)

        n = loc["n_poses"]
        results[cfg] = {
            "ape_median": loc["ape_median"],
            "ape_mean": loc["ape_mean"],
            "ape_std": loc["ape_std"],
            "ape_ci95": ci95(loc["ape_std"], n),
            "rpe_trans_mean": loc["rpe_trans_mean"],
            "rpe_rot_rmse_deg": loc["rpe_rot_rmse_deg"],
            "n_poses": n,
        }

        print(
            f"{cfg:<8s}"
            f" & {loc['ape_median']:.2f} $\\pm$ {loc['ape_std']:.2f}\\,m"
            f" & {loc['ape_mean']:.2f} $\\pm$ {loc['ape_std']:.2f}\\,m"
            f" & {loc['rpe_trans_mean']:.2f}\\,m"
            f" & {loc['rpe_rot_rmse_deg']:.1f}$^\\circ$"
            " \\\\"
        )

    # CI on the mean (very tight with n=3194)
    print()
    print("--- (B) 95% CI on the mean (n=3194 poses) ---")
    print(" & APE Mean ± CI95 \\\\")
    print("\\midrule")
    for cfg in config_order:
        r = results[cfg]
        c = r["ape_ci95"]
        print(f"{cfg:<8s} & {r['ape_mean']:.2f} $\\pm$ {c:.2f}\\,m \\\\")

    print()
    print("  NOTE: std reflects per-pose error spread; CI reflects")
    print("  precision of the mean estimate. With n=3194, CIs are tight.")
    print("  L+D's large std (3.18m) despite low median (1.23m) reflects")
    print("  occasional SLAM jumps (ape_max ≈ 159m). Consider also")
    print("  reporting median + IQR for robustness.")
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  PAPER VALUE CROSS-CHECK
# ═══════════════════════════════════════════════════════════════════════

def cross_check(t4_results, t6_results):
    """Compare computed values against paper's current values."""
    print("=" * 80)
    print("PAPER VALUE CROSS-CHECK")
    print("=" * 80)

    paper_t4 = {
        "L": (252, 2295), "L+S": (258, 3061),
        "L+D": (909, 3546), "D": (883, 1461),
        "L+D+dyn": (904, 3740),
    }
    print("\nTable IV (corridor live):")
    for cfg, (pn, pm) in paper_t4.items():
        cn = t4_results[cfg]["n"]
        cm = t4_results[cfg]["mean"]
        match = "OK" if cn == pn and abs(cm - pm) < 1 else "MISMATCH"
        print(f"  {cfg:<8s}: paper=({pn}, {pm})  computed=({cn}, {cm:.0f})  {match}")

    paper_t6 = {
        "L": (4.63, 4.93, 0.36, 7.8),
        "L+S": (4.84, 5.08, 0.17, 8.3),
        "L+D": (1.23, 1.82, 0.19, 9.0),
        "D": (4.52, 4.97, 0.34, 8.5),
        "L+D+dyn": (4.40, 4.62, 0.23, 8.7),
    }
    print("\nTable VI (localization):")
    for cfg, (pmed, pmean, prpe, prot) in paper_t6.items():
        r = t6_results[cfg]
        checks = [
            abs(r["ape_median"] - pmed) < 0.015,
            abs(r["ape_mean"] - pmean) < 0.015,
            abs(r["rpe_trans_mean"] - prpe) < 0.015,
            abs(r["rpe_rot_rmse_deg"] - prot) < 0.15,
        ]
        match = "OK" if all(checks) else "MISMATCH"
        print(f"  {cfg:<8s}: {match}")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    all_results = {}

    all_results["table_iii"] = table_iii()
    all_results["table_iv"] = table_iv()
    all_results["table_v"] = table_v()
    all_results["table_vi"] = table_vi()

    cross_check(all_results["table_iv"], all_results["table_vi"])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "paper_stats.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON summary written to {out_path}")


if __name__ == "__main__":
    main()
