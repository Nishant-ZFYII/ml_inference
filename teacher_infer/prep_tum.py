#!/usr/bin/env python3
"""
Prepare TUM RGB-D sequences for the teacher inference pipeline.

TUM RGB-D stores data as timestamped files in per-sequence directories:
    rgbd_dataset_freiburg1_room/
        rgb/        1305031102.175304.png   (640x480 8-bit RGB)
        depth/      1305031102.160407.png   (640x480 16-bit, millimetres, factor 5000)
        rgb.txt     timestamp filename
        depth.txt   timestamp filename

This script:
  1. Reads rgb.txt and depth.txt from each sequence
  2. Associates RGB-depth pairs by closest timestamp (< 20ms threshold)
  3. Converts depth PNGs from 16-bit/5000 to float32 metres (.npy)
  4. Copies RGB PNGs with sequential stem names (00000.png, 00001.png, ...)
  5. Creates train/val index splits (80/20)

Output layout (matches what run_da3.py / run_sam2.py / build_manifest.py expect):
    $OUTPUT_DIR/
        rgb/        00000.png, 00001.png, ...
        depth/      00000.npy, 00001.npy, ...
        train_indices.txt
        val_indices.txt

Usage:
    python -m teacher_infer.prep_tum \\
        --sequences /scratch/np3129/tum_rgbd/rgbd_dataset_freiburg1_room \\
                    /scratch/np3129/tum_rgbd/rgbd_dataset_freiburg1_desk \\
                    /scratch/np3129/tum_rgbd/rgbd_dataset_freiburg2_large_no_loop \\
        --output-dir /scratch/np3129/tum_teacher_data
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

TUM_DEPTH_FACTOR = 5000.0
ASSOC_MAX_DT = 0.02  # 20 ms


def read_file_list(path: Path) -> list:
    """Read a TUM-format file list (timestamp filename)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ts = float(parts[0])
                fname = parts[1]
                entries.append((ts, fname))
    return entries


def associate(rgb_list, depth_list, max_dt=ASSOC_MAX_DT):
    """Associate RGB and depth frames by closest timestamps."""
    pairs = []
    depth_idx = 0
    for rgb_ts, rgb_file in rgb_list:
        best_dt = float("inf")
        best_depth = None
        for i in range(max(0, depth_idx - 5), len(depth_list)):
            dt = abs(depth_list[i][0] - rgb_ts)
            if dt < best_dt:
                best_dt = dt
                best_depth = (depth_list[i][0], depth_list[i][1])
                depth_idx = i
            elif dt > best_dt:
                break
        if best_depth and best_dt < max_dt:
            pairs.append((rgb_ts, rgb_file, best_depth[0], best_depth[1]))
    return pairs


def convert_depth_png_to_npy(depth_path: Path) -> np.ndarray:
    """Convert TUM 16-bit depth PNG to float32 metres."""
    depth_img = Image.open(depth_path)
    depth_raw = np.array(depth_img, dtype=np.float32)
    depth_m = depth_raw / TUM_DEPTH_FACTOR
    depth_m[depth_raw == 0] = 0.0
    return depth_m


def main():
    p = argparse.ArgumentParser(description="Prepare TUM RGB-D for teacher inference")
    p.add_argument("--sequences", type=str, nargs="+", required=True,
                   help="Paths to extracted TUM sequence directories")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=3,
                   help="Take every Nth frame to reduce temporal redundancy (default: 3)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    rgb_out = output_dir / "rgb"
    depth_out = output_dir / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    seq_stats = []

    for seq_path_str in args.sequences:
        seq_path = Path(seq_path_str)
        seq_name = seq_path.name
        print(f"\n=== Processing: {seq_name} ===")

        rgb_txt = seq_path / "rgb.txt"
        depth_txt = seq_path / "depth.txt"

        if not rgb_txt.exists() or not depth_txt.exists():
            print(f"  SKIP: missing rgb.txt or depth.txt in {seq_path}")
            continue

        rgb_list = read_file_list(rgb_txt)
        depth_list = read_file_list(depth_txt)
        pairs = associate(rgb_list, depth_list)
        print(f"  RGB frames: {len(rgb_list)}, Depth frames: {len(depth_list)}, "
              f"Associated pairs: {len(pairs)}")

        pairs = pairs[::args.subsample]
        print(f"  After subsampling (1/{args.subsample}): {len(pairs)} frames")

        seq_start = global_idx
        for rgb_ts, rgb_file, depth_ts, depth_file in tqdm(pairs, desc=f"  {seq_name}"):
            stem = f"{global_idx:05d}"

            src_rgb = seq_path / rgb_file
            if src_rgb.exists():
                shutil.copy2(src_rgb, rgb_out / f"{stem}.png")
            else:
                print(f"  WARNING: missing {src_rgb}")
                continue

            src_depth = seq_path / depth_file
            if src_depth.exists():
                depth_m = convert_depth_png_to_npy(src_depth)
                np.save(depth_out / f"{stem}.npy", depth_m)
            else:
                np.save(depth_out / f"{stem}.npy",
                        np.zeros((480, 640), dtype=np.float32))
                print(f"  WARNING: missing {src_depth}, saved zeros")

            global_idx += 1

        seq_stats.append((seq_name, len(pairs), seq_start, global_idx - 1))

    total = global_idx
    print(f"\n=== Total frames extracted: {total} ===")
    for name, count, start, end in seq_stats:
        print(f"  {name}: {count} frames (indices {start:05d}..{end:05d})")

    all_indices = list(range(total))
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)
    split = int((1.0 - args.val_ratio) * total)

    train_indices = sorted(all_indices[:split])
    val_indices = sorted(all_indices[split:])

    with open(output_dir / "train_indices.txt", "w") as f:
        for idx in train_indices:
            f.write(f"{idx}\n")
    with open(output_dir / "val_indices.txt", "w") as f:
        for idx in val_indices:
            f.write(f"{idx}\n")

    print(f"\nSplit: {len(train_indices)} train / {len(val_indices)} val")
    print(f"Output: {output_dir}")
    print(f"\nNext step: run teacher inference:")
    print(f"  sbatch teacher_infer/teacher_infer_tum.slurm")


if __name__ == "__main__":
    main()
