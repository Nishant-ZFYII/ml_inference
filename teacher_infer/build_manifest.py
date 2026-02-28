#!/usr/bin/env python3
"""
Build manifest.jsonl linking all per-sample files across directories.

The manifest ties together RGB, depth, DA3 predictions, and YOLO+SAM2 labels
for each sample.

Usage:
    python -m teacher_infer.build_manifest \
        --rgb-dir   data/rgb/ \
        --depth-dir data/depth/ \
        --da3-dir   data/da3_depth/ \
        --sam2-dir  data/sam2_seg/ \
        --output    data/manifest.jsonl

For NYU:  confidence is synthesised in the loader (not in the manifest).
For corridor data: add --confidence-dir corridor_data/confidence/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def find_stems(directory: Path, extensions=None) -> dict:
    """Find all file stems in a directory, mapping stem -> filename."""
    if extensions is None:
        extensions = {".png", ".jpg", ".jpeg", ".npy"}
    stems = {}
    if not directory.exists():
        return stems
    for f in directory.iterdir():
        if f.suffix.lower() in extensions:
            stems[f.stem] = f.name
    return stems


def main():
    p = argparse.ArgumentParser(description="Build sample manifest")
    p.add_argument("--rgb-dir", type=str, required=True)
    p.add_argument("--depth-dir", type=str, required=True)
    p.add_argument("--da3-dir", type=str, default=None,
                   help="Directory of DA3 metric depth .npy files")
    p.add_argument("--sam2-dir", type=str, default=None,
                   help="Directory of YOLO+SAM2 merged 6-class seg .npy files")
    p.add_argument("--confidence-dir", type=str, default=None)
    p.add_argument("--output", type=str, default="manifest.jsonl")
    p.add_argument("--allow-missing-teachers", action="store_true",
                   help="Don't fail if DA3/SAM2 files are missing")
    args = p.parse_args()

    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    da3_dir = Path(args.da3_dir) if args.da3_dir else None
    sam2_dir = Path(args.sam2_dir) if args.sam2_dir else None
    conf_dir = Path(args.confidence_dir) if args.confidence_dir else None

    rgb_stems = find_stems(rgb_dir, {".png", ".jpg", ".jpeg"})
    if not rgb_stems:
        print(f"ERROR: No images found in {rgb_dir}")
        sys.exit(1)

    depth_stems = find_stems(depth_dir, {".npy"})

    da3_stems = {}
    if da3_dir:
        da3_stems = find_stems(da3_dir, {".npy"})

    sam2_stems = {}
    if sam2_dir:
        sam2_stems = find_stems(sam2_dir, {".npy"})

    conf_stems = {}
    if conf_dir:
        conf_stems = find_stems(conf_dir, {".npy"})

    entries = []
    missing = []

    for stem in sorted(rgb_stems.keys()):
        entry = {
            "stem": stem,
            "rgb": f"rgb/{rgb_stems[stem]}",
        }

        # Depth (required)
        depth_key = stem
        if depth_key not in depth_stems:
            for dk in depth_stems:
                if dk.startswith(stem):
                    depth_key = dk
                    break
        if depth_key in depth_stems:
            entry["depth"] = f"depth/{depth_stems[depth_key]}"
        else:
            missing.append(f"  MISSING depth for {stem}")
            continue

        # DA3 metric depth (from run_da3.py output: {stem}.npy)
        if da3_dir:
            da3_key = stem
            if da3_key in da3_stems:
                entry["da3_depth"] = f"da3_depth/{da3_stems[da3_key]}"
            else:
                entry["da3_depth"] = None
                if not args.allow_missing_teachers:
                    missing.append(f"  MISSING DA3 for {stem}")
                    continue
        else:
            entry["da3_depth"] = None

        # YOLO+SAM2 segmentation: final merged 6-class map from YOLO+SAM2+geometric pipeline
        if sam2_dir:
            sam2_key = f"{stem}_sam2_seg"
            if sam2_key not in sam2_stems:
                sam2_key = stem
            if sam2_key in sam2_stems:
                entry["sam2_seg"] = f"sam2_seg/{sam2_stems[sam2_key]}"
            else:
                entry["sam2_seg"] = None
                if not args.allow_missing_teachers:
                    missing.append(f"  MISSING SAM2 for {stem}")
                    continue
        else:
            entry["sam2_seg"] = None

        # Confidence (optional, corridor data only)
        if conf_dir:
            if stem in conf_stems:
                entry["confidence"] = f"confidence/{conf_stems[stem]}"

        entries.append(entry)

    # Report
    print(f"RGB images found:   {len(rgb_stems)}")
    print(f"Valid samples:      {len(entries)}")
    if da3_dir:
        da3_count = sum(1 for e in entries if e.get("da3_depth") is not None)
        print(f"DA3 depth matched:  {da3_count}")
    if sam2_dir:
        sam2_count = sum(1 for e in entries if e.get("sam2_seg") is not None)
        print(f"SAM2 seg matched:   {sam2_count}")

    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for m in missing[:20]:
            print(m)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        if not args.allow_missing_teachers:
            print("\nERROR: Missing files detected. Use --allow-missing-teachers "
                  "to skip incomplete samples.")
            sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nManifest written: {output_path} ({len(entries)} samples)")


if __name__ == "__main__":
    main()
