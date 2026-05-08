"""
TUM RGB-D dataset loader.

Loads frames prepared by teacher_infer/prep_tum.py (flat directory of
sequential PNGs + NPYs) with teacher predictions from DA3 + YOLO+SAM2.

Returns the same dict format as NYUDepthV2Dataset so train.py and
losses.py work without changes.

No GT segmentation fallback — TUM has no semantic labels.  If teacher
seg is missing for a sample, it is filled with IGNORE_LABEL (255) so
cross-entropy skips it.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TUMRGBDDataset(Dataset):
    """
    PyTorch Dataset for TUM RGB-D data prepped by prep_tum.py.

    Each sample dict matches the NYU loader interface:
        rgb          : float32 tensor [3, H, W]  in [0, 1]
        depth        : float32 tensor [1, H, W]  in metres (Kinect GT)
        seg          : int64   tensor [H, W]      class ids {0..5, 255}
        confidence   : float32 tensor [1, H, W]  (depth > 0)
        da3_depth    : float32 tensor [1, H, W]  (or zeros if unavailable)
        has_da3      : bool
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        height: int = 240,
        width: int = 320,
        augment: bool = False,
        manifest_path: Optional[str] = None,
        data_limit: int = 0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.height = height
        self.width = width
        self.augment = augment

        split_file = self.data_root / f"{split}_indices.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"TUM split file not found: {split_file}\n"
                "Run: python -m teacher_infer.prep_tum --sequences ... "
                f"--output-dir {data_root}"
            )

        with open(split_file) as f:
            self.indices = [int(line.strip()) for line in f if line.strip()]

        if data_limit > 0:
            self.indices = self.indices[:data_limit]

        self._teacher_map: Dict[str, dict] = {}
        self._manifest_base: Optional[str] = None
        if manifest_path and os.path.exists(manifest_path):
            self._manifest_base = str(Path(manifest_path).parent)
            with open(manifest_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self._teacher_map[entry["stem"]] = entry
            covered = sum(1 for i in self.indices
                          if f"{i:05d}" in self._teacher_map)
            print(f"[TUM {split}] Manifest covers {covered}/{len(self.indices)} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        sample_idx = self.indices[idx]
        stem = f"{sample_idx:05d}"

        rgb = Image.open(self.data_root / "rgb" / f"{stem}.png").convert("RGB")
        depth = np.load(self.data_root / "depth" / f"{stem}.npy").astype(np.float32)

        confidence = (depth > 0).astype(np.float32)

        has_da3 = False
        da3_depth = np.zeros_like(depth)
        seg = np.full(depth.shape, 255, dtype=np.int32)

        if self._manifest_base and stem in self._teacher_map:
            entry = self._teacher_map[stem]

            da3_rel = entry.get("da3_depth")
            if da3_rel:
                da3_path = os.path.join(self._manifest_base, da3_rel)
                if os.path.exists(da3_path):
                    da3_depth = np.load(da3_path).astype(np.float32)
                    has_da3 = True

            sam2_rel = entry.get("sam2_seg")
            if sam2_rel:
                sam2_path = os.path.join(self._manifest_base, sam2_rel)
                if os.path.exists(sam2_path):
                    seg = np.load(sam2_path).astype(np.int32)

        rgb, depth, seg, confidence, da3_depth = self._transform(
            rgb, depth, seg, confidence, da3_depth
        )

        return {
            "rgb": rgb,
            "depth": depth,
            "seg": seg,
            "confidence": confidence,
            "da3_depth": da3_depth,
            "has_da3": has_da3,
        }

    def _transform(
        self,
        rgb: Image.Image,
        depth: np.ndarray,
        seg: np.ndarray,
        confidence: np.ndarray,
        da3_depth: np.ndarray,
    ) -> Tuple[torch.Tensor, ...]:
        w, h = rgb.size

        if self.augment:
            if random.random() > 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = np.fliplr(depth).copy()
                seg = np.fliplr(seg).copy()
                confidence = np.fliplr(confidence).copy()
                da3_depth = np.fliplr(da3_depth).copy()

            from torchvision import transforms as T
            jitter = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            rgb = jitter(rgb)

            if h > self.height and w > self.width:
                top = random.randint(0, h - self.height)
                left = random.randint(0, w - self.width)
                rgb = rgb.crop((left, top, left + self.width, top + self.height))
                depth = depth[top:top + self.height, left:left + self.width]
                seg = seg[top:top + self.height, left:left + self.width]
                confidence = confidence[top:top + self.height, left:left + self.width]
                da3_depth = da3_depth[top:top + self.height, left:left + self.width]
            else:
                rgb = rgb.resize((self.width, self.height), Image.BILINEAR)
                depth = _resize_np(depth, self.height, self.width)
                seg = _resize_np(seg, self.height, self.width, order=0)
                confidence = _resize_np(confidence, self.height, self.width)
                da3_depth = _resize_np(da3_depth, self.height, self.width)
        else:
            rgb = rgb.resize((self.width, self.height), Image.BILINEAR)
            depth = _resize_np(depth, self.height, self.width)
            seg = _resize_np(seg, self.height, self.width, order=0)
            confidence = _resize_np(confidence, self.height, self.width)
            da3_depth = _resize_np(da3_depth, self.height, self.width)

        rgb_t = torch.from_numpy(
            np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        depth_t = torch.from_numpy(depth[np.newaxis].astype(np.float32))
        seg_t = torch.from_numpy(seg.astype(np.int64))
        conf_t = torch.from_numpy(confidence[np.newaxis].astype(np.float32))
        da3_t = torch.from_numpy(da3_depth[np.newaxis].astype(np.float32))

        return rgb_t, depth_t, seg_t, conf_t, da3_t


def _resize_np(arr: np.ndarray, h: int, w: int, order: int = 1) -> np.ndarray:
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    mode = "NEAREST" if order == 0 else "BILINEAR"
    pil_img = Image.fromarray(arr)
    resized = pil_img.resize((w, h), getattr(Image, mode, Image.BILINEAR))
    return np.array(resized, dtype=arr.dtype)


def get_tum_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders for TUM RGB-D."""
    train_ds = TUMRGBDDataset(
        data_root=cfg.DATA_ROOT,
        split="train",
        height=cfg.INPUT_HEIGHT,
        width=cfg.INPUT_WIDTH,
        augment=True,
        manifest_path=cfg.MANIFEST_PATH,
        data_limit=cfg.DATA_LIMIT,
    )
    val_ds = TUMRGBDDataset(
        data_root=cfg.DATA_ROOT,
        split="val",
        height=cfg.INPUT_HEIGHT,
        width=cfg.INPUT_WIDTH,
        augment=False,
        manifest_path=cfg.MANIFEST_PATH,
        data_limit=cfg.DATA_LIMIT,
    )

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
