#!/usr/bin/env python3
"""
Run DA3-Metric-Large on a directory of RGB images.

Produces per-image .npy depth predictions in metric metres using the
official DA3 API: metric_depth = focal * raw_output / 300.0

Usage (NYU -- defaults to NYU intrinsics):
    python -m teacher_infer.run_da3 \
        --input-dir $SCRATCH/nyu_teacher_data/rgb/ \
        --output-dir $SCRATCH/teacher_output/da3_depth/

Usage (corridor -- pass Femto Bolt intrinsics):
    python -m teacher_infer.run_da3 \
        --input-dir corridor_data/rgb/ \
        --output-dir corridor_data/da3_depth/ \
        --fx 605.0 --fy 605.0

Works on NYU images now, corridor images later -- same script, different
--input-dir and --fx/--fy.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_da3_model(model_id: str, device: torch.device):
    """Load DA3-Metric-Large via the depth_anything_3 API."""
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    model.eval()
    return model


def infer_single(model, img_path: str, focal: float) -> np.ndarray:
    """Run DA3 inference on a single image, return metric depth in metres."""
    prediction = model.inference([img_path])
    raw_depth = prediction.depth[0]  # [H, W] float32

    metric_depth = focal * raw_depth / 300.0
    return metric_depth.astype(np.float32)


def sanity_check(depth: np.ndarray, stem: str, failed_log: Path) -> bool:
    """Per-frame sanity check. Returns True if OK."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        with open(failed_log, "a") as f:
            f.write(f"{stem}: all-zero depth\n")
        return False

    if valid.max() > 50.0:
        with open(failed_log, "a") as f:
            f.write(f"{stem}: max {valid.max():.1f}m -- likely not metric\n")
        return False

    if valid.mean() > 20.0:
        with open(failed_log, "a") as f:
            f.write(f"{stem}: mean {valid.mean():.1f}m -- likely not metric\n")
        return False

    return True


def main():
    p = argparse.ArgumentParser(description="Run DA3-Metric-Large depth inference")
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-id", type=str, default="depth-anything/da3metric-large")
    p.add_argument("--fx", type=float, default=518.8579, help="Camera focal length x (px)")
    p.add_argument("--fy", type=float, default=519.4696, help="Camera focal length y (px)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    focal = (args.fx + args.fy) / 2.0

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not available. DA3 inference will be very slow on CPU.")
        device = torch.device("cpu")

    extensions = {".png", ".jpg", ".jpeg"}
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in extensions
    )
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Focal length: {focal:.2f} px (fx={args.fx}, fy={args.fy})")

    print(f"Loading {args.model_id}...")
    model = load_da3_model(args.model_id, device)
    print("DA3 model loaded.")

    failed_log = output_dir / "failed_frames.txt"
    n_ok, n_fail, n_skip = 0, 0, 0
    times = []

    for img_path in tqdm(image_files, desc="DA3 inference"):
        stem = img_path.stem
        out_path = output_dir / f"{stem}.npy"

        if out_path.exists():
            n_skip += 1
            continue

        t0 = time.time()
        try:
            depth = infer_single(model, str(img_path), focal)
        except Exception as e:
            with open(failed_log, "a") as f:
                f.write(f"{stem}: inference error -- {e}\n")
            n_fail += 1
            continue
        elapsed = time.time() - t0
        times.append(elapsed)

        if sanity_check(depth, stem, failed_log):
            np.save(out_path, depth)
            n_ok += 1
        else:
            n_fail += 1

    total = len(image_files)
    mean_fps = 1.0 / np.mean(times) if times else 0.0
    print(f"\nDA3 inference complete.")
    print(f"  Total: {total}, OK: {n_ok}, Failed: {n_fail}, Skipped: {n_skip}")
    if times:
        print(f"  Mean FPS: {mean_fps:.2f}, Mean latency: {np.mean(times)*1000:.0f} ms")
    if n_fail > 0:
        print(f"  See {failed_log} for failure details.")


if __name__ == "__main__":
    main()
