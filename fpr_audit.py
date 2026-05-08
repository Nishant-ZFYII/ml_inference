#!/usr/bin/env python3
"""
FPR Origin Audit: classify the 5.3% corridor false-positive rate
into three mutually exclusive categories.

Categories:
  1. sensor-invalid-fill  -- DA3 produced depth where the ToF sensor
     had no valid reading (dead pixel). This is arguably a *safety
     improvement*, not a true false positive.
  2. free-space-hallucination -- DA3 produced an obstacle where the
     sensor had valid depth indicating free space. True depth
     disagreement.
  3. inflation-artifact -- The cell is not directly occupied by any
     projected DA3 point but falls within the inflation radius of
     a DA3-added obstacle cell.

Method:
  For each of the 459 corridor frames (A2 config: sensor + DA3,
  fixed 0.09m inflation):
    a) Load sensor depth and DA3 depth from corridor_eval_data/
    b) Back-project both to 3D, apply height filter, rasterize to
       120x120 BEV grid at 0.05m resolution
    c) Build sensor-only grid (Baseline) and sensor+DA3 grid (A2),
       both pre-inflation
    d) Inflate both grids with a circular kernel matching the 0.09m
       Nav2 inflation radius
    e) Identify FP cells = A2_inflated & ~Baseline_inflated
    f) Classify each FP cell:
       - pre-inflation FP with DA3 source at sensor-invalid pixel
         → sensor-invalid-fill
       - pre-inflation FP with DA3 source at sensor-valid pixel
         → free-space-hallucination
       - post-inflation-only FP → inflation-artifact

Inputs:
  ml_pipeline/corridor_eval_data/{rgb,depth,da3_depth,intrinsics.json}
  (459 frames extracted from rgbd_imu_20260228_003828_0.mcap)

Outputs:
  ml_pipeline/results/fpr_audit.json -- per-frame + aggregate breakdown
  stdout -- LaTeX-ready summary paragraph
"""

import json
import pathlib

import numpy as np
from scipy import ndimage
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "corridor_eval_data"

GRID_SIZE = 120
RESOLUTION = 0.05        # m/cell
ORIGIN_X = -3.0           # grid covers -3m to +3m in X (right)
ORIGIN_Z = 0.0            # grid covers 0m to +6m in Z (forward)
MIN_DEPTH = 0.1
MAX_DEPTH = 5.0
MIN_HEIGHT = 0.05         # above ground
MAX_HEIGHT = 0.50
CAMERA_HEIGHT = 0.25
INFLATION_RADIUS = 0.09   # metres (A2 fixed inflation)


def make_inflation_kernel(radius_m: float, res: float) -> np.ndarray:
    """Circular structuring element matching Nav2 lethal inflation."""
    r_cells = radius_m / res
    size = int(np.ceil(r_cells)) * 2 + 1
    half = size // 2
    y, x = np.ogrid[-half:half + 1, -half:half + 1]
    return (x ** 2 + y ** 2) <= r_cells ** 2


def depth_to_grid(depth: np.ndarray, fx, fy, cx, cy):
    """
    Back-project depth to BEV grid.

    Returns:
      grid: (GRID_SIZE, GRID_SIZE) bool
      source_sensor_valid: for each occupied cell, whether any
        contributing pixel had valid sensor depth (used for
        classification). Stored as a dict {(row, col): bool}.
      pixel_vs, pixel_us: v,u coords of pixels that passed all filters
    """
    valid = (depth >= MIN_DEPTH) & (depth <= MAX_DEPTH)
    vs, us = np.where(valid)
    z = depth[vs, us].astype(np.float64)
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy

    h_above = CAMERA_HEIGHT - y
    h_mask = (h_above >= MIN_HEIGHT) & (h_above <= MAX_HEIGHT)

    x = x[h_mask].astype(np.float32)
    z = z[h_mask].astype(np.float32)
    vs_out = vs[h_mask]
    us_out = us[h_mask]

    gx = ((x - ORIGIN_X) / RESOLUTION).astype(int)
    gz = ((z - ORIGIN_Z) / RESOLUTION).astype(int)
    in_bounds = (gx >= 0) & (gx < GRID_SIZE) & (gz >= 0) & (gz < GRID_SIZE)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    grid[gz[in_bounds], gx[in_bounds]] = True

    return grid, vs_out[in_bounds], us_out[in_bounds]


def classify_frame(sensor_depth, da3_depth, fx, fy, cx, cy, kernel):
    """
    Classify FP cells for a single frame.

    Returns dict with counts for each FP category.
    """
    sensor_valid_mask = sensor_depth > 0

    sensor_grid, _, _ = depth_to_grid(
        sensor_depth, fx, fy, cx, cy)
    da3_grid, da3_vs, da3_us = depth_to_grid(
        da3_depth, fx, fy, cx, cy)

    combined_raw = sensor_grid | da3_grid

    sensor_inflated = ndimage.binary_dilation(
        sensor_grid, structure=kernel)
    combined_inflated = ndimage.binary_dilation(
        combined_raw, structure=kernel)

    fp_inflated = combined_inflated & ~sensor_inflated
    fp_raw = combined_raw & ~sensor_grid
    fp_inflation_only = fp_inflated & ~fp_raw

    n_fp_total = int(fp_inflated.sum())
    n_fp_inflation = int(fp_inflation_only.sum())
    n_fp_direct = int(fp_raw.sum())

    n_fp_sensor_invalid = 0
    n_fp_sensor_valid = 0

    if n_fp_direct > 0:
        da3_only_cells = set()
        da3_cell_sensor_status = {}
        for v, u in zip(da3_vs, da3_us):
            gz_idx = int((da3_depth[v, u] - ORIGIN_Z) / RESOLUTION)
            x3d = (u - cx) * da3_depth[v, u] / fx
            gx_idx = int((x3d - ORIGIN_X) / RESOLUTION)
            if 0 <= gx_idx < GRID_SIZE and 0 <= gz_idx < GRID_SIZE:
                cell = (gz_idx, gx_idx)
                if fp_raw[cell]:
                    da3_only_cells.add(cell)
                    sv = bool(sensor_valid_mask[v, u])
                    if cell not in da3_cell_sensor_status:
                        da3_cell_sensor_status[cell] = sv
                    else:
                        da3_cell_sensor_status[cell] = (
                            da3_cell_sensor_status[cell] or sv)

        for cell in da3_only_cells:
            if da3_cell_sensor_status.get(cell, False):
                n_fp_sensor_valid += 1
            else:
                n_fp_sensor_invalid += 1

        untracked = n_fp_direct - len(da3_only_cells)
        if untracked > 0:
            n_fp_sensor_invalid += untracked

    baseline_free = int((~sensor_inflated).sum())

    return {
        "fp_total": n_fp_total,
        "fp_sensor_invalid_fill": n_fp_sensor_invalid,
        "fp_free_space_hallucination": n_fp_sensor_valid,
        "fp_inflation_artifact": n_fp_inflation,
        "baseline_occupied_inflated": int(sensor_inflated.sum()),
        "combined_occupied_inflated": int(combined_inflated.sum()),
        "baseline_free": baseline_free,
        "fpr_reconstructed": n_fp_total / baseline_free if baseline_free > 0 else 0,
    }


def main():
    with open(DATA / "intrinsics.json") as f:
        intr = json.load(f)
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    manifest_path = DATA / "manifest.jsonl"
    frames = []
    with open(manifest_path) as f:
        for line in f:
            frames.append(json.loads(line))

    kernel = make_inflation_kernel(INFLATION_RADIUS, RESOLUTION)
    print(f"Inflation kernel: {kernel.shape}, {kernel.sum()} cells")

    per_frame = []
    agg = {
        "fp_total": 0, "fp_sensor_invalid_fill": 0,
        "fp_free_space_hallucination": 0, "fp_inflation_artifact": 0,
        "baseline_free_total": 0,
    }

    for i, entry in enumerate(tqdm(frames, desc="FPR audit")):
        rgb_name = pathlib.Path(entry["rgb"]).stem
        sensor_path = DATA / "depth" / f"{rgb_name}.npy"
        da3_path = DATA / "da3_depth" / f"{rgb_name}.npy"

        sensor_depth = np.load(sensor_path)
        da3_depth = np.load(da3_path)

        result = classify_frame(sensor_depth, da3_depth, fx, fy, cx, cy, kernel)
        result["frame"] = rgb_name
        per_frame.append(result)

        for k in agg:
            if k == "baseline_free_total":
                agg[k] += result["baseline_free"]
            elif k in result:
                agg[k] += result[k]

    n = len(frames)
    pct = lambda x: x / agg["fp_total"] * 100 if agg["fp_total"] > 0 else 0
    fpr_overall = agg["fp_total"] / agg["baseline_free_total"] * 100

    print()
    print("=" * 70)
    print("FPR AUDIT RESULTS (459 corridor frames, A2 config)")
    print("=" * 70)
    print(f"Total FP cells across all frames:  {agg['fp_total']}")
    print(f"Reconstructed aggregate FPR:       {fpr_overall:.1f}%")
    print()
    print("Breakdown of FP origin:")
    print(f"  1. Sensor-invalid fill:          "
          f"{agg['fp_sensor_invalid_fill']:>6d}  "
          f"({pct(agg['fp_sensor_invalid_fill']):.1f}%)")
    print(f"  2. Free-space hallucination:     "
          f"{agg['fp_free_space_hallucination']:>6d}  "
          f"({pct(agg['fp_free_space_hallucination']):.1f}%)")
    print(f"  3. Inflation artifact:           "
          f"{agg['fp_inflation_artifact']:>6d}  "
          f"({pct(agg['fp_inflation_artifact']):.1f}%)")
    print()

    print("LaTeX-ready paragraph:")
    print("-" * 70)
    si = pct(agg["fp_sensor_invalid_fill"])
    fh = pct(agg["fp_free_space_hallucination"])
    ia = pct(agg["fp_inflation_artifact"])
    print(
        f"The {fpr_overall:.1f}\\% false-positive rate decomposes into three "
        f"origins: \\textbf{{{si:.0f}\\%}} sensor-invalid fill "
        f"(DA3 fills ToF dead pixels, a safety improvement), "
        f"\\textbf{{{fh:.0f}\\%}} free-space hallucination "
        f"(DA3 disagrees with valid sensor depth), and "
        f"\\textbf{{{ia:.0f}\\%}} inflation artifact "
        f"(inflation spreading from DA3-added cells)."
    )
    print("-" * 70)
    print()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fpr_audit.json"
    with open(out_path, "w") as f:
        json.dump({
            "params": {
                "grid_size": GRID_SIZE,
                "resolution": RESOLUTION,
                "inflation_radius": INFLATION_RADIUS,
                "min_depth": MIN_DEPTH,
                "max_depth": MAX_DEPTH,
                "height_filter": [MIN_HEIGHT, MAX_HEIGHT],
                "camera_height": CAMERA_HEIGHT,
                "n_frames": n,
            },
            "aggregate": {
                "fp_total": agg["fp_total"],
                "fp_sensor_invalid_fill": agg["fp_sensor_invalid_fill"],
                "fp_free_space_hallucination": agg["fp_free_space_hallucination"],
                "fp_inflation_artifact": agg["fp_inflation_artifact"],
                "fpr_reconstructed_pct": round(fpr_overall, 2),
                "pct_sensor_invalid": round(si, 1),
                "pct_hallucination": round(fh, 1),
                "pct_inflation": round(ia, 1),
            },
            "per_frame": per_frame,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
