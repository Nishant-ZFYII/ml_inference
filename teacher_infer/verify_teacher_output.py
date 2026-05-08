#!/usr/bin/env python3
"""
Verify DA3 teacher output on a small number of frames before full HPC job.

Checks:
  1. Shape: DA3 output (H,W) matches RGB input (H,W)
  2. Dtype: must be float32
  3. Scale: mean < 20m, max < 50m (flags non-metric output)
  4. Correlation: Pearson r vs GT depth where GT > 0
     - r > 0.7: OK
     - r < 0.5: STRONG WARNING, exit 1
  5. Visual: saves side-by-side PNG (RGB | DA3 depth | GT depth)

Exit codes:
  0 -- all checks pass
  1 -- scale or correlation failure (SLURM job should abort)

Usage:
    python -m teacher_infer.verify_teacher_output \
        --input-dir $SCRATCH/nyu_teacher_data/rgb/ \
        --gt-depth-dir $SCRATCH/nyu_teacher_data/depth/ \
        --output-dir $SCRATCH/verify_output/ \
        --model-id depth-anything/da3metric-large \
        --fx 518.8579 --fy 519.4696 \
        --n-frames 5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr


def main():
    p = argparse.ArgumentParser(description="Verify DA3 teacher depth output")
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--gt-depth-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-id", type=str, default="depth-anything/da3metric-large")
    p.add_argument("--fx", type=float, default=518.8579)
    p.add_argument("--fy", type=float, default=519.4696)
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_depth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    focal = (args.fx + args.fy) / 2.0

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    extensions = {".png", ".jpg", ".jpeg"}
    image_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in extensions
    )[:args.n_frames]

    if not image_files:
        print("ERROR: No images found in", input_dir)
        sys.exit(1)

    print(f"Verifying DA3 output on {len(image_files)} frames...")
    print(f"Model: {args.model_id}, focal: {focal:.2f}")

    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(args.model_id)
    model = model.to(device=device)
    model.eval()

    any_fail = False

    for img_path in image_files:
        stem = img_path.stem
        print(f"\n--- Frame: {stem} ---")

        from PIL import Image
        rgb = Image.open(img_path).convert("RGB")
        rgb_np = np.array(rgb)

        prediction = model.inference([str(img_path)])
        raw = prediction.depth[0]
        da3_depth = (focal * raw / 300.0).astype(np.float32)

        # Check 1: dtype
        print(f"  dtype: {da3_depth.dtype}", end="")
        if da3_depth.dtype != np.float32:
            print(" FAIL")
            any_fail = True
        else:
            print(" OK")

        # Check 2: shape vs RGB
        print(f"  shape: DA3={da3_depth.shape}, RGB={rgb_np.shape[:2]}", end="")
        if da3_depth.shape != rgb_np.shape[:2]:
            print(f" MISMATCH (DA3 processes at different res, will be resized)")
        else:
            print(" OK")

        # Check 3: scale
        valid = da3_depth[da3_depth > 0]
        if len(valid) > 0:
            print(f"  scale: min={valid.min():.2f}m, max={valid.max():.2f}m, "
                  f"mean={valid.mean():.2f}m, std={valid.std():.2f}m")
            if valid.mean() > 20.0 or valid.max() > 50.0:
                print("  FAIL: depth values out of indoor range -- likely not metric")
                any_fail = True
        else:
            print("  FAIL: all-zero depth")
            any_fail = True

        # Check 4: correlation with GT
        gt_path = gt_dir / f"{stem}.npy"
        if gt_path.exists():
            gt_depth = np.load(gt_path).astype(np.float32)

            if gt_depth.shape != da3_depth.shape:
                from PIL import Image as PILImage
                gt_resized = np.array(
                    PILImage.fromarray(gt_depth).resize(
                        (da3_depth.shape[1], da3_depth.shape[0])),
                    dtype=np.float32)
            else:
                gt_resized = gt_depth

            mask = gt_resized > 0
            if mask.sum() > 100:
                r, _ = pearsonr(da3_depth[mask], gt_resized[mask])
                print(f"  Pearson r vs GT: {r:.4f}", end="")
                if r < 0.5:
                    print(" FAIL (r < 0.5 -- scale or inversion issue)")
                    any_fail = True
                elif r < 0.7:
                    print(" WARNING (r < 0.7)")
                else:
                    print(" OK")
            else:
                print("  Pearson r: insufficient valid GT pixels")

            # Check 5: visual
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(rgb_np)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            im1 = axes[1].imshow(da3_depth, cmap="viridis", vmin=0, vmax=10)
            axes[1].set_title(f"DA3 depth (mean={valid.mean():.2f}m)")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            im2 = axes[2].imshow(gt_resized, cmap="viridis", vmin=0, vmax=10)
            axes[2].set_title("GT depth")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            plt.tight_layout()
            out_png = output_dir / f"verify_{stem}.png"
            plt.savefig(out_png, dpi=100)
            plt.close()
            print(f"  Saved: {out_png}")
        else:
            print(f"  GT depth not found at {gt_path}, skipping correlation check")

    print(f"\n{'='*50}")
    if any_fail:
        print("VERIFICATION FAILED -- do NOT proceed with full teacher inference.")
        print("Check the saved PNGs and fix the issue.")
        sys.exit(1)
    else:
        print("All checks passed. Safe to proceed with full DA3 inference.")
        sys.exit(0)


if __name__ == "__main__":
    main()
