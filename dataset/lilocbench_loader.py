"""
LILocBench dataset loader for training.

Loads extracted LILocBench frames (RGB PNGs + depth NPY + DA3 teacher labels)
from the output of extract_lilocbench.py + teacher inference.

Each sample matches the NYU loader dict format:
    rgb          : float32 tensor [3, H, W]  in [0, 1]
    depth        : float32 tensor [1, H, W]  in meters (sensor depth)
    seg          : int64   tensor [H, W]      class ids {0..5, 255}
    confidence   : float32 tensor [1, H, W]  synthetic confidence
    da3_depth    : float32 tensor [1, H, W]  DA3 teacher prediction
    has_da3      : bool
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LILocBenchDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        height: int = 240,
        width: int = 320,
        augment: bool = False,
        data_limit: int = 0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.height = height
        self.width = width
        self.augment = augment

        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        if data_limit > 0:
            self.samples = self.samples[:data_limit]

        da3_count = sum(1 for s in self.samples if "da3_depth" in s)
        seg_count = sum(1 for s in self.samples if "seg" in s)
        print(f"LILocBench: {len(self.samples)} samples, "
              f"{da3_count} with DA3 depth, {seg_count} with seg labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        rgb_path = self.data_root / entry["rgb"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Cannot read: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Sensor depth (from RealSense D455)
        depth_path = self.data_root / entry["sensor_depth"]
        if depth_path.suffix == ".npy":
            depth = np.load(str(depth_path))
        else:
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            depth = depth_raw.astype(np.float32) * 0.001
        depth = cv2.resize(depth, (self.width, self.height),
                           interpolation=cv2.INTER_NEAREST)

        # DA3 teacher depth (if available)
        has_da3 = False
        da3_depth = np.zeros((self.height, self.width), dtype=np.float32)
        if "da3_depth" in entry:
            da3_path = self.data_root / entry["da3_depth"]
            if os.path.exists(da3_path):
                da3_depth = np.load(str(da3_path))
                da3_depth = cv2.resize(da3_depth, (self.width, self.height),
                                       interpolation=cv2.INTER_LINEAR)
                has_da3 = True

        # Segmentation labels (if available from YOLO+SAM2)
        seg = np.full((self.height, self.width), 255, dtype=np.int64)
        if "seg" in entry:
            seg_path = self.data_root / entry["seg"]
            if os.path.exists(seg_path):
                seg = np.load(str(seg_path)).astype(np.int64)
                seg = cv2.resize(seg, (self.width, self.height),
                                 interpolation=cv2.INTER_NEAREST)

        # Confidence: 1.0 where sensor depth is valid (0.2-10m), 0.0 elsewhere
        confidence = ((depth > 0.2) & (depth < 10.0)).astype(np.float32)

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                rgb = np.fliplr(rgb).copy()
                depth = np.fliplr(depth).copy()
                da3_depth = np.fliplr(da3_depth).copy()
                seg = np.fliplr(seg).copy()
                confidence = np.fliplr(confidence).copy()

        # To tensors
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        da3_t = torch.from_numpy(da3_depth).unsqueeze(0).float()
        seg_t = torch.from_numpy(seg.copy()).long()
        conf_t = torch.from_numpy(confidence).unsqueeze(0).float()

        return {
            "rgb": rgb_t,
            "depth": depth_t,
            "seg": seg_t,
            "confidence": conf_t,
            "da3_depth": da3_t,
            "has_da3": has_da3,
        }


def get_lilocbench_dataloaders(cfg, manifest_path=None):
    """Build train/val dataloaders for LILocBench data."""
    manifest = manifest_path or cfg.MANIFEST_PATH
    if manifest is None:
        raise ValueError("LILocBench manifest path required. "
                         "Use --lilocbench-manifest or --manifest")

    data_root = str(Path(manifest).parent)

    full_dataset = LILocBenchDataset(
        data_root=data_root,
        manifest_path=manifest,
        height=cfg.INPUT_HEIGHT,
        width=cfg.INPUT_WIDTH,
        augment=True,
        data_limit=cfg.DATA_LIMIT,
    )

    # 80/20 train/val split
    n = len(full_dataset)
    n_val = max(1, int(n * cfg.VAL_SPLIT))
    n_train = n - n_val

    g = torch.Generator()
    g.manual_seed(cfg.SEED)
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader
