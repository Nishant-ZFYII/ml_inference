#!/usr/bin/env python3
"""
Build manifest.jsonl linking all per-sample files across directories.

The manifest ties together RGB, depth, DA2 predictions, and SAM2 labels
for each sample.  If any file is missing for a sample, the script fails
loudly (silent corruption guard).

Usage:
    python -m teacher_infer.build_manifest \
        --rgb-dir   data/rgb/ \
        --depth-dir data/depth/ \
        --da2-dir   data/da2_depth/ \
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
    p.add_argument("--da2-dir", type=str, default=None)
    p.add_argument("--sam2-dir", type=str, default=None)
    p.add_argument("--confidence-dir", type=str, default=None)
    p.add_argument("--output", type=str, default="manifest.jsonl")
    p.add_argument("--allow-missing-teachers", action="store_true",
                   help="Don't fail if DA2/SAM2 files are missing")
    args = p.parse_args()

    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    da2_dir = Path(args.da2_dir) if args.da2_dir else None
    sam2_dir = Path(args.sam2_dir) if args.sam2_dir else None
    conf_dir = Path(args.confidence_dir) if args.confidence_dir else None

    # Find all RGB stems as the reference set
    rgb_stems = find_stems(rgb_dir, {".png", ".jpg", ".jpeg"})
    if not rgb_stems:
        print(f"ERROR: No images found in {rgb_dir}")
        sys.exit(1)

    depth_stems = find_stems(depth_dir, {".npy"})

    da2_stems = {}
    if da2_dir:
        da2_stems = find_stems(da2_dir, {".npy"})

    sam2_stems = {}
    if sam2_dir:
        sam2_stems = find_stems(sam2_dir, {".npy"})

    conf_stems = {}
    if conf_dir:
        conf_stems = find_stems(conf_dir, {".npy"})

    # Build manifest
    entries = []
    missing = []
    orphaned_da2 = set(da2_stems.keys()) - set(rgb_stems.keys())
    orphaned_sam2 = set(sam2_stems.keys()) - set(rgb_stems.keys())

    for stem in sorted(rgb_stems.keys()):
        entry = {
            "stem": stem,
            "rgb": f"rgb/{rgb_stems[stem]}",
        }

        # Depth (required)
        depth_key = stem
        if depth_key not in depth_stems:
            # Try with _depth suffix removed
            for dk in depth_stems:
                if dk.startswith(stem):
                    depth_key = dk
                    break
        if depth_key in depth_stems:
            entry["depth"] = f"depth/{depth_stems[depth_key]}"
        else:
            missing.append(f"  MISSING depth for {stem}")
            continue

        # DA2 depth (optional unless --allow-missing-teachers is off)
        if da2_dir:
            da2_key = f"{stem}_da2_depth"
            if da2_key not in da2_stems:
                da2_key = stem  # try bare stem
            if da2_key in da2_stems:
                entry["da2_depth"] = f"da2_depth/{da2_stems[da2_key]}"
            elif not args.allow_missing_teachers:
                missing.append(f"  MISSING DA2 for {stem}")
                continue

        # SAM2 segmentation (optional unless --allow-missing-teachers is off)
        if sam2_dir:
            sam2_key = f"{stem}_sam2_seg"
            if sam2_key not in sam2_stems:
                sam2_key = stem
            if sam2_key in sam2_stems:
                entry["sam2_seg"] = f"sam2_seg/{sam2_stems[sam2_key]}"
            elif not args.allow_missing_teachers:
                missing.append(f"  MISSING SAM2 for {stem}")
                continue

        # Confidence (optional)
        if conf_dir:
            conf_key = stem
            if conf_key in conf_stems:
                entry["confidence"] = f"confidence/{conf_stems[conf_key]}"

        entries.append(entry)

    # Report
    print(f"RGB images found:   {len(rgb_stems)}")
    print(f"Valid samples:      {len(entries)}")

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

    if orphaned_da2:
        print(f"\nOrphaned DA2 files (no matching RGB): {len(orphaned_da2)}")
    if orphaned_sam2:
        print(f"Orphaned SAM2 files (no matching RGB): {len(orphaned_sam2)}")

    # Write manifest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nManifest written: {output_path} ({len(entries)} samples)")


if __name__ == "__main__":
    main()
