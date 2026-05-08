#!/usr/bin/env python3
"""
Extract training frames from LILocBench individual-files format.

LILocBench uses RealSense D455 cameras at 640x480 @15Hz. This script:
  1. Walks the camera_XXX folders to find color + depth images
  2. Parses intrinsics from intrinsics.yaml
  3. Matches color/depth by nearest timestamp
  4. Subsamples to target FPS (default 2Hz → keeps every 7th-8th frame)
  5. Saves RGB PNGs + depth .npy (float32, metres) to output dir
  6. Writes manifest.jsonl compatible with our training pipeline

Usage:
    python extract_lilocbench.py \
        --input /scratch/$USER/lilocbench/lt_changes_3 \
        --output /scratch/$USER/lilocbench_extracted \
        --target-fps 2

Data structure expected (per-sequence):
    sequence_name/
        camera_XXX/
            color/images/*.png   OR   color_images/*.png
            depth/images/*.png   OR   depth_images/*.png
            intrinsics.yaml
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def find_image_dirs(camera_dir: Path):
    """Auto-detect color and depth image directories under a camera folder."""
    candidates_color = [
        camera_dir / "color" / "images",
        camera_dir / "color_images",
        camera_dir / "color",
        camera_dir / "rgb" / "images",
        camera_dir / "rgb",
    ]
    candidates_depth = [
        camera_dir / "depth" / "images",
        camera_dir / "depth_images",
        camera_dir / "depth",
    ]

    color_dir = None
    for d in candidates_color:
        if d.is_dir() and any(d.glob("*.png")):
            color_dir = d
            break

    depth_dir = None
    for d in candidates_depth:
        if d.is_dir() and any(d.glob("*.png")):
            depth_dir = d
            break

    return color_dir, depth_dir


def parse_timestamp(filename: str) -> float:
    """Extract timestamp (seconds) from PNG filename like '1733375656.696981258.png'."""
    stem = Path(filename).stem
    try:
        return float(stem)
    except ValueError:
        parts = stem.split(".")
        if len(parts) >= 2:
            return float(f"{parts[0]}.{parts[1]}")
        raise ValueError(f"Cannot parse timestamp from: {filename}")


def parse_intrinsics(yaml_path: Path) -> dict:
    """Parse camera intrinsics from LILocBench intrinsics.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        if "K" in data:
            K = data["K"]
            if isinstance(K, list) and len(K) == 9:
                return {"fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5]}
        if "camera_matrix" in data:
            cm = data["camera_matrix"]
            if "data" in cm:
                K = cm["data"]
                return {"fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5]}
        for key in ["fx", "width"]:
            if key in data:
                return {
                    "fx": data.get("fx", 386.0),
                    "fy": data.get("fy", 386.0),
                    "cx": data.get("cx", 320.0),
                    "cy": data.get("cy", 240.0),
                }

    # D455 defaults at 640x480
    print(f"  [WARN] Could not parse {yaml_path}, using D455 defaults")
    return {"fx": 386.0, "fy": 386.0, "cx": 320.0, "cy": 240.0}


def match_timestamps(color_files, depth_files, max_diff_ms=50):
    """Match color and depth images by nearest timestamp within threshold."""
    color_ts = [(parse_timestamp(f.name), f) for f in color_files]
    depth_ts = [(parse_timestamp(f.name), f) for f in depth_files]

    color_ts.sort(key=lambda x: x[0])
    depth_ts.sort(key=lambda x: x[0])

    pairs = []
    d_idx = 0
    for c_t, c_f in color_ts:
        best_diff = float("inf")
        best_d = None
        for j in range(max(0, d_idx - 2), len(depth_ts)):
            diff = abs(depth_ts[j][0] - c_t)
            if diff < best_diff:
                best_diff = diff
                best_d = depth_ts[j]
                d_idx = j
            elif diff > best_diff:
                break

        if best_d is not None and best_diff * 1000 < max_diff_ms:
            pairs.append((c_t, c_f, best_d[1]))

    return pairs


def subsample(pairs, target_fps=2.0):
    """Subsample pairs to approximately target FPS."""
    if not pairs:
        return []

    min_interval = 1.0 / target_fps
    selected = [pairs[0]]
    last_t = pairs[0][0]

    for t, c, d in pairs[1:]:
        if t - last_t >= min_interval:
            selected.append((t, c, d))
            last_t = t

    return selected


def main():
    parser = argparse.ArgumentParser(description="Extract LILocBench frames")
    parser.add_argument("--input", required=True,
                        help="Path to LILocBench sequence folder (individual files)")
    parser.add_argument("--output", required=True,
                        help="Output directory for extracted data")
    parser.add_argument("--target-fps", type=float, default=2.0,
                        help="Target frame rate after subsampling (default: 2 Hz)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated camera names to use (default: all)")
    parser.add_argument("--resize", type=str, default=None,
                        help="Resize to WxH (e.g. 640x480). Default: keep original")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="Depth scale: multiply PNG value by this to get metres "
                             "(default: 0.001 for RealSense D455 mm→m)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    rgb_out = output_dir / "rgb"
    depth_out = output_dir / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    camera_dirs = sorted(input_dir.glob("camera_*"))
    if args.cameras:
        keep = set(args.cameras.split(","))
        camera_dirs = [d for d in camera_dirs if d.name in keep]

    if not camera_dirs:
        print(f"ERROR: No camera_* directories found in {input_dir}")
        sys.exit(1)

    print(f"=== LILocBench Extraction ===")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Cameras:    {[d.name for d in camera_dirs]}")
    print(f"Target FPS: {args.target_fps}")
    print()

    manifest_entries = []
    global_idx = 0
    intrinsics_cache = {}

    for cam_dir in camera_dirs:
        cam_name = cam_dir.name
        print(f"Processing {cam_name}...")

        color_dir, depth_dir = find_image_dirs(cam_dir)
        if color_dir is None or depth_dir is None:
            print(f"  [SKIP] Could not find color/depth dirs in {cam_dir}")
            print(f"  Checked subdirs: {[d.name for d in cam_dir.iterdir() if d.is_dir()]}")
            continue

        print(f"  Color: {color_dir}")
        print(f"  Depth: {depth_dir}")

        intrinsics_file = cam_dir / "intrinsics.yaml"
        intrinsics = parse_intrinsics(intrinsics_file) if intrinsics_file.exists() else {
            "fx": 386.0, "fy": 386.0, "cx": 320.0, "cy": 240.0
        }
        intrinsics_cache[cam_name] = intrinsics
        print(f"  Intrinsics: fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f}")

        color_files = sorted(color_dir.glob("*.png"))
        depth_files = sorted(depth_dir.glob("*.png"))
        print(f"  Color frames: {len(color_files)}, Depth frames: {len(depth_files)}")

        pairs = match_timestamps(color_files, depth_files)
        print(f"  Matched pairs: {len(pairs)}")

        sampled = subsample(pairs, args.target_fps)
        print(f"  After {args.target_fps}Hz subsampling: {len(sampled)}")

        for ts, color_path, depth_path in sampled:
            stem = f"{global_idx:06d}"

            rgb = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if rgb is None:
                continue

            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            if depth_raw is None:
                continue

            if resize:
                rgb = cv2.resize(rgb, resize, interpolation=cv2.INTER_LINEAR)
                depth_raw = cv2.resize(depth_raw, resize, interpolation=cv2.INTER_NEAREST)

            depth_m = depth_raw.astype(np.float32) * args.depth_scale

            rgb_path = rgb_out / f"{stem}.png"
            depth_path_out = depth_out / f"{stem}.npy"

            cv2.imwrite(str(rgb_path), rgb)
            np.save(str(depth_path_out), depth_m)

            entry = {
                "stem": stem,
                "rgb": f"rgb/{stem}.png",
                "sensor_depth": f"depth/{stem}.npy",
                "source": "lilocbench",
                "sequence": input_dir.name,
                "camera": cam_name,
                "timestamp": ts,
                "intrinsics": intrinsics,
            }
            manifest_entries.append(entry)
            global_idx += 1

    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n=== Extraction Complete ===")
    print(f"Total frames:  {global_idx}")
    print(f"Manifest:      {manifest_path}")
    print(f"RGB dir:       {rgb_out}")
    print(f"Depth dir:     {depth_out}")

    if intrinsics_cache:
        first_cam = next(iter(intrinsics_cache.values()))
        print(f"Focal length:  {(first_cam['fx'] + first_cam['fy'])/2:.1f} px")


if __name__ == "__main__":
    main()
