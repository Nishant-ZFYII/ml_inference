#!/usr/bin/env python3
"""
Regenerate DA3 depth predictions for corridor_eval_data using
Depth-Anything-V2-Small (HuggingFace) on CPU.

Overwrites corridor_eval_data/da3_depth/*.npy with predictions
that are correctly aligned to the RGB frames.
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import pipeline


def main():
    base = Path(__file__).parent / "corridor_eval_data"
    rgb_dir = base / "rgb"
    out_dir = base / "da3_depth"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    n = len(rgb_files)
    print(f"Found {n} RGB frames in {rgb_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Loading Depth-Anything-V2-Small on CPU...")
    sys.stdout.flush()

    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu",
    )
    print("Model loaded. Starting inference...\n")
    sys.stdout.flush()

    t0 = time.time()
    for i, rgb_path in enumerate(rgb_files):
        frame_start = time.time()

        img = Image.open(rgb_path).convert("RGB")
        result = pipe(img)
        depth_pil = result["depth"]
        raw = np.array(depth_pil, dtype=np.float32)
        depth_np = 255.0 - raw  # invert: pipeline outputs disparity (close=high), we need depth (close=low)

        out_path = out_dir / f"{rgb_path.stem}.npy"
        np.save(str(out_path), depth_np)

        elapsed = time.time() - frame_start
        if (i + 1) % 10 == 0 or i == 0:
            total_elapsed = time.time() - t0
            avg = total_elapsed / (i + 1)
            remaining = avg * (n - i - 1)
            print(f"  [{i+1:3d}/{n}] {rgb_path.stem}  "
                  f"{elapsed:.1f}s  (avg {avg:.1f}s/frame, "
                  f"~{remaining/60:.1f} min remaining)")
            sys.stdout.flush()

        if (i + 1) % 50 == 0:
            gc.collect()

    total = time.time() - t0
    print(f"\nDone. {n} frames in {total/60:.1f} minutes "
          f"({total/n:.1f}s avg per frame)")
    print(f"Output: {out_dir}")

    existing = list(out_dir.glob("*.npy"))
    print(f"Files in output dir: {len(existing)}")


if __name__ == "__main__":
    main()
