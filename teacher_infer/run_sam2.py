#!/usr/bin/env python3
"""
YOLO-seeded SAM2 semantic labeling pipeline.

SAM2 produces instance masks (no class labels). This pipeline:
  1. YOLO detects objects with class labels (person, furniture items)
  2. SAM2 refines each YOLO box to a pixel-accurate mask
  3. Geometric heuristics label floor/wall/glass from the depth map
  4. All masks merge into a single 6-class label map

Output: per-image .npy segmentation label (uint8, HxW, values 0-5).

Usage:
    python -m teacher_infer.run_sam2 \
        --input-dir $SCRATCH/nyu_teacher_input/rgb/ \
        --depth-dir $SCRATCH/nyu_teacher_input/depth/ \
        --output-dir $SCRATCH/nyu_teacher_output/sam2_seg/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Our 6-class mapping:  0=floor, 1=wall, 2=person, 3=furniture, 4=glass, 5=other
PERSON_LABEL = 2
FURNITURE_LABEL = 3
FLOOR_LABEL = 0
WALL_LABEL = 1
GLASS_LABEL = 4
OTHER_LABEL = 5

# YOLO COCO class IDs that map to our classes
YOLO_PERSON_IDS = {0}  # COCO: person
YOLO_FURNITURE_IDS = {
    56,  # chair
    57,  # couch/sofa
    58,  # potted plant (close enough)
    59,  # bed
    60,  # dining table
    61,  # toilet
    62,  # tv/monitor
    63,  # laptop
    69,  # oven
    71,  # sink
    72,  # refrigerator
    73,  # book
    74,  # clock
}


def load_yolo():
    """Load YOLOv8 for object detection."""
    from ultralytics import YOLO
    model = YOLO("yolov8x.pt")
    return model


def load_sam2(device: torch.device):
    """
    Load SAM2 model for mask refinement.

    Tries sam2 package first, falls back to segment-anything.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(
            "sam2_hiera_large",
            "sam2_hiera_large.pt",
            device=device,
        )
        predictor = SAM2ImagePredictor(sam2_model)
        return predictor, "sam2"
    except Exception:
        pass

    try:
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_h"](
            checkpoint="sam_vit_h_4b8939.pth")
        sam = sam.to(device)
        predictor = SamPredictor(sam)
        return predictor, "sam1"
    except Exception as e:
        raise RuntimeError(
            f"Could not load SAM2 or SAM. Install sam2 package. Error: {e}")


def refine_box_with_sam(predictor, image_np: np.ndarray,
                        box: np.ndarray, backend: str) -> np.ndarray:
    """Use SAM to refine a YOLO bounding box into a pixel-accurate mask."""
    predictor.set_image(image_np)

    if backend == "sam2":
        masks, scores, _ = predictor.predict(
            box=box, multimask_output=True)
    else:
        masks, scores, _ = predictor.predict(
            box=box, multimask_output=True)

    # Pick highest-confidence mask
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(bool)


def geometric_floor_wall_glass(depth: np.ndarray, rgb: np.ndarray,
                                existing_mask: np.ndarray) -> np.ndarray:
    """
    Label floor, wall, and glass using geometric heuristics from the depth map.

    Only labels pixels not already claimed by YOLO+SAM2.

    Args:
        depth: float32 depth map (H, W) in meters
        rgb: uint8 RGB image (H, W, 3)
        existing_mask: uint8 label map (H, W), 255 = unlabeled

    Returns:
        Updated label map with floor/wall/glass filled in.
    """
    h, w = depth.shape
    labels = existing_mask.copy()
    unlabeled = labels == 255

    if depth.max() == 0:
        # No depth available, mark all unlabeled as "other"
        labels[unlabeled] = OTHER_LABEL
        return labels

    # ── Floor heuristic: bottom portion of image + close depth ──────
    # Assume camera is roughly horizontal; floor is in the lower half
    # and has moderate, smoothly varying depth
    floor_region = np.zeros((h, w), dtype=bool)
    bottom_half = np.zeros((h, w), dtype=bool)
    bottom_half[h // 2:, :] = True

    valid_depth = depth > 0
    if valid_depth.sum() > 0:
        median_depth = np.median(depth[valid_depth])
        close_depth = (depth > 0) & (depth < median_depth * 1.5)
        floor_region = bottom_half & close_depth & unlabeled
    labels[floor_region] = FLOOR_LABEL
    unlabeled = labels == 255

    # ── Wall heuristic: upper portion or large uniform-depth regions ─
    # Compute depth gradient magnitude as proxy for surface orientation
    grad_y = np.gradient(depth, axis=0)
    grad_x = np.gradient(depth, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Walls have low depth gradient (flat surfaces) and aren't floor
    low_gradient = grad_mag < np.percentile(grad_mag[valid_depth], 30) if valid_depth.sum() > 0 else np.zeros_like(grad_mag, dtype=bool)
    wall_region = low_gradient & unlabeled & valid_depth
    labels[wall_region] = WALL_LABEL
    unlabeled = labels == 255

    # ── Glass heuristic: invalid depth + high RGB gradient ──────────
    invalid_depth = depth == 0
    rgb_gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb.astype(float)
    rgb_grad_y = np.gradient(rgb_gray, axis=0)
    rgb_grad_x = np.gradient(rgb_gray, axis=1)
    rgb_grad = np.sqrt(rgb_grad_x**2 + rgb_grad_y**2)

    high_rgb_grad = rgb_grad > np.percentile(rgb_grad, 70)
    glass_region = invalid_depth & high_rgb_grad & unlabeled
    labels[glass_region] = GLASS_LABEL
    unlabeled = labels == 255

    # ── Everything else → other ────────────────────────────────────
    labels[unlabeled] = OTHER_LABEL

    return labels


def process_image(yolo_model, sam_predictor, sam_backend: str,
                  rgb_path: Path, depth_path: Path | None) -> np.ndarray:
    """Process a single image through the full YOLO+SAM2+geometric pipeline."""
    img = Image.open(rgb_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # Start with all pixels unlabeled
    labels = np.full((h, w), 255, dtype=np.uint8)

    # ── Step 1: YOLO detection ─────────────────────────────────────
    results = yolo_model(img_np, verbose=False)
    detections = results[0]

    # ── Step 2: SAM2 refinement of each detection ──────────────────
    if len(detections.boxes) > 0:
        for box, cls_id, conf in zip(
            detections.boxes.xyxy.cpu().numpy(),
            detections.boxes.cls.cpu().numpy().astype(int),
            detections.boxes.conf.cpu().numpy(),
        ):
            if conf < 0.3:
                continue

            # Map YOLO class to our 6-class scheme
            if cls_id in YOLO_PERSON_IDS:
                our_label = PERSON_LABEL
            elif cls_id in YOLO_FURNITURE_IDS:
                our_label = FURNITURE_LABEL
            else:
                continue  # skip classes we don't care about

            try:
                mask = refine_box_with_sam(sam_predictor, img_np,
                                          box, sam_backend)
                labels[mask & (labels == 255)] = our_label
            except Exception:
                # Fall back to bounding box fill
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                region = labels[y1:y2, x1:x2] == 255
                labels[y1:y2, x1:x2][region] = our_label

    # ── Steps 3-4: Geometric heuristics + merge ────────────────────
    depth = np.zeros((h, w), dtype=np.float32)
    if depth_path and depth_path.exists():
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (h, w):
            depth_img = Image.fromarray(depth)
            depth = np.array(depth_img.resize((w, h)), dtype=np.float32)

    labels = geometric_floor_wall_glass(depth, img_np, labels)

    return labels


def main():
    p = argparse.ArgumentParser(
        description="YOLO-seeded SAM2 segmentation labeling")
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--depth-dir", type=str, default=None,
                   help="Directory of depth .npy files for geometric heuristics")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA required for SAM2 inference.")

    extensions = {".png", ".jpg", ".jpeg"}
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in extensions
    )
    print(f"Found {len(image_files)} images in {input_dir}")

    # Load models
    print("Loading YOLOv8...")
    yolo_model = load_yolo()
    print("Loading SAM2...")
    sam_predictor, sam_backend = load_sam2(device)
    print(f"SAM backend: {sam_backend}")

    for img_path in tqdm(image_files, desc="YOLO+SAM2 labeling"):
        stem = img_path.stem
        out_path = output_dir / f"{stem}_sam2_seg.npy"

        if out_path.exists():
            continue

        depth_path = None
        if depth_dir:
            for ext in [".npy"]:
                candidate = depth_dir / f"{stem}{ext}"
                if candidate.exists():
                    depth_path = candidate
                    break

        labels = process_image(yolo_model, sam_predictor, sam_backend,
                               img_path, depth_path)
        np.save(out_path, labels)

    print(f"\nSegmentation labeling complete. {len(image_files)} label maps "
          f"saved to {output_dir}")


if __name__ == "__main__":
    main()
