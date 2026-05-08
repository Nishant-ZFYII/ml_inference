"""
Corridor dataset loader (stub).

This loader activates when config.USE_CORRIDOR_DATA = True.
It expects data collected from the physical Ackermann robot with:
    - RGB images from the Orbbec Femto Bolt
    - ToF depth maps (.npy, float32, meters)
    - ToF confidence maps (.npy, float32, 0-1)
    - DA2-Large teacher depth predictions (.npy, float32, meters)
    - YOLO+SAM2 teacher segmentation labels (.npy, uint8, 6-class)

All file associations come from a manifest.jsonl produced by
teacher_infer/build_manifest.py.

Expected directory layout:
    corridor_data/
        rgb/            *.png    (320x240 or larger, will be resized)
        depth/          *.npy    (float32, metric meters)
        confidence/     *.npy    (float32, 0.0 – 1.0)
        da2_depth/      *.npy    (float32, metric meters)
        sam2_seg/       *.npy    (uint8, values 0–5)
        manifest.jsonl           (one JSON object per line)

To populate this:
    1. Collect RGB + ToF data on the robot
    2. Run teacher_infer/run_da2.py on HPC
    3. Run teacher_infer/run_sam2.py on HPC
    4. Run teacher_infer/build_manifest.py
    5. Set USE_CORRIDOR_DATA = True in config.py
    6. Re-run train.py
"""

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ────────────────────────────────────────────────────────────────────────────
# TODO: When corridor data is ready, verify:
#   - Depth values are in meters (not mm)
#   - Confidence maps are normalized to [0, 1]
#   - SAM2 seg labels use our 6-class scheme (0-5)
#   - All files for each sample exist (build_manifest.py checks this)
# ────────────────────────────────────────────────────────────────────────────


class CorridorDataset(Dataset):
    """
    PyTorch Dataset for corridor data with teacher labels.

    Returns the same dict format as NYUDepthV2Dataset for drop-in
    compatibility with train.py and losses.py.
    """

    def __init__(
        self,
        manifest_path: str,
        base_dir: str,
        height: int = 240,
        width: int = 320,
        augment: bool = False,
    ):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.height = height
        self.width = width
        self.augment = augment

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Corridor manifest not found: {manifest_path}\n"
                "Run teacher_infer/build_manifest.py first."
            )

        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        entry = self.samples[idx]

        rgb = Image.open(self.base_dir / entry["rgb"]).convert("RGB")
        depth = np.load(self.base_dir / entry["depth"]).astype(np.float32)

        # ToF confidence from the Femto Bolt sensor
        conf_path = entry.get("confidence")
        if conf_path and os.path.exists(self.base_dir / conf_path):
            confidence = np.load(self.base_dir / conf_path).astype(np.float32)
        else:
            confidence = (depth > 0).astype(np.float32)

        # DA2 teacher depth
        da2_path = entry.get("da2_depth")
        has_da2 = False
        da2_depth = np.zeros_like(depth)
        if da2_path and os.path.exists(self.base_dir / da2_path):
            da2_depth = np.load(self.base_dir / da2_path).astype(np.float32)
            has_da2 = True

        # YOLO+SAM2 teacher segmentation
        seg_path = entry.get("sam2_seg")
        if seg_path and os.path.exists(self.base_dir / seg_path):
            seg = np.load(self.base_dir / seg_path).astype(np.int64)
        else:
            # TODO: handle missing seg labels gracefully
            seg = np.full(depth.shape, 255, dtype=np.int64)

        # Resize all to target resolution
        rgb = rgb.resize((self.width, self.height), Image.BILINEAR)
        depth = _resize_np(depth, self.height, self.width)
        confidence = _resize_np(confidence, self.height, self.width)
        da2_depth = _resize_np(da2_depth, self.height, self.width)
        seg = _resize_np(seg.astype(np.float32), self.height, self.width, order=0).astype(np.int64)

        # Augmentation
        if self.augment:
            import random
            if random.random() > 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = np.fliplr(depth).copy()
                seg = np.fliplr(seg).copy()
                confidence = np.fliplr(confidence).copy()
                da2_depth = np.fliplr(da2_depth).copy()

            from torchvision import transforms as T
            rgb = T.ColorJitter(0.2, 0.2, 0.2, 0.1)(rgb)

        # To tensors
        rgb_t = torch.from_numpy(
            np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        depth_t = torch.from_numpy(depth[np.newaxis].astype(np.float32))
        seg_t = torch.from_numpy(seg.astype(np.int64))
        conf_t = torch.from_numpy(confidence[np.newaxis].astype(np.float32))
        da2_t = torch.from_numpy(da2_depth[np.newaxis].astype(np.float32))

        return {
            "rgb": rgb_t,
            "depth": depth_t,
            "seg": seg_t,
            "confidence": conf_t,
            "da2_depth": da2_t,
            "has_da2": has_da2,
        }


def _resize_np(arr: np.ndarray, h: int, w: int, order: int = 1) -> np.ndarray:
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    mode = "NEAREST" if order == 0 else "BILINEAR"
    pil_img = Image.fromarray(arr)
    resized = pil_img.resize((w, h), getattr(Image, mode, Image.BILINEAR))
    return np.array(resized, dtype=arr.dtype)


def get_corridor_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for corridor data.

    Splits the manifest 80/20 using the same seed as NYU for consistency.
    """
    import random as _random

    manifest_path = cfg.MANIFEST_PATH
    if not manifest_path:
        manifest_path = os.path.join(cfg.CORRIDOR_RGB_DIR, "..", "manifest.jsonl")

    full_ds = CorridorDataset(
        manifest_path=manifest_path,
        base_dir=os.path.dirname(manifest_path),
        height=cfg.INPUT_HEIGHT,
        width=cfg.INPUT_WIDTH,
        augment=False,
    )

    n = len(full_ds)
    indices = list(range(n))
    rng = _random.Random(cfg.SEED)
    rng.shuffle(indices)
    split = int(0.8 * n)

    from torch.utils.data import Subset

    train_ds = Subset(full_ds, indices[:split])
    val_ds = Subset(full_ds, indices[split:])

    # Enable augmentation for training subset
    # (Subset wraps the dataset, augmentation is controlled per-sample)
    full_ds.augment = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
