#!/usr/bin/env python3
"""
Run DA2-Large (Depth Anything V2 Large) on a directory of RGB images.

Produces per-image .npy depth predictions in metric meters.

Usage:
    python -m teacher_infer.run_da2 \
        --input-dir $SCRATCH/nyu_teacher_input/rgb/ \
        --output-dir $SCRATCH/nyu_teacher_output/da2_depth/

Works on NYU images now, corridor images later — same script, different
--input-dir. No code changes needed for the swap.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_da2_model(device: torch.device):
    """
    Load Depth Anything V2 Large model.

    Tries torch.hub first, falls back to HuggingFace transformers.
    """
    try:
        model = torch.hub.load(
            "LiheYoung/Depth-Anything-V2",
            "depth_anything_v2_vitl",
            pretrained=True,
        )
        model = model.to(device).eval()
        return model, "torchhub"
    except Exception:
        pass

    try:
        from transformers import pipeline
        pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=device,
        )
        return pipe, "transformers"
    except Exception as e:
        raise RuntimeError(
            "Could not load DA2-Large via torch.hub or transformers. "
            f"Install depth-anything-v2 or transformers. Error: {e}"
        )


def infer_torchhub(model, img: Image.Image, device: torch.device) -> np.ndarray:
    """Run inference using torch.hub model."""
    img_np = np.array(img)
    with torch.no_grad():
        depth = model.infer_image(img_np)
    return depth.astype(np.float32)


def infer_transformers(pipe, img: Image.Image) -> np.ndarray:
    """Run inference using HuggingFace transformers pipeline."""
    result = pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)
    return depth


def main():
    p = argparse.ArgumentParser(description="Run DA2-Large depth inference")
    p.add_argument("--input-dir", type=str, required=True,
                   help="Directory of RGB .png images")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory for .npy depth files")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA required for DA2-Large inference.")

    # Collect image files
    extensions = {".png", ".jpg", ".jpeg"}
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in extensions
    )
    print(f"Found {len(image_files)} images in {input_dir}")

    # Load model
    print("Loading DA2-Large...")
    model, backend = load_da2_model(device)
    print(f"Loaded via {backend}")

    # Run inference
    for img_path in tqdm(image_files, desc="DA2 inference"):
        stem = img_path.stem
        out_path = output_dir / f"{stem}_da2_depth.npy"

        if out_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")

        if backend == "torchhub":
            depth = infer_torchhub(model, img, device)
        else:
            depth = infer_transformers(model, img)

        np.save(out_path, depth)

    print(f"\nDA2 inference complete. {len(image_files)} depth maps saved "
          f"to {output_dir}")


if __name__ == "__main__":
    main()
