"""
NYU Depth V2 dataset loader.

Downloads the labeled split (1449 RGBD pairs) from HuggingFace,
caches to disk, and provides PyTorch Dataset / DataLoader objects.

When teacher predictions (DA2 depth, YOLO+SAM2 seg) are available via a
manifest.jsonl file, those are loaded alongside NYU ground truth so the
hybrid depth loss can use them.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .label_remapper import remap_labels

# ────────────────────────────────────────────────────────────────────────────
# ▶ CORRIDOR DATA SWAP-IN HOOK
# ────────────────────────────────────────────────────────────────────────────
# When config.USE_CORRIDOR_DATA is True, get_dataloaders() returns
# corridor loaders instead. See corridor_loader.py.
# ────────────────────────────────────────────────────────────────────────────


class NYUDepthV2Dataset(Dataset):
    """
    PyTorch Dataset wrapping the NYU Depth V2 labeled split.

    Each sample is a dict:
        rgb          : float32 tensor [3, H, W]  in [0, 1]
        depth        : float32 tensor [1, H, W]  in meters
        seg          : int64   tensor [H, W]      class ids {0..5, 255}
        confidence   : float32 tensor [1, H, W]  synthetic confidence
        da2_depth    : float32 tensor [1, H, W]  (or zeros if unavailable)
        has_da2      : bool                        whether da2_depth is real
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
        self.manifest_path = manifest_path

        self._cache_dir = self.data_root / "nyu_cache"
        self._ensure_cached()

        split_file = self._cache_dir / f"{split}_indices.txt"
        with open(split_file) as f:
            self.indices = [int(line.strip()) for line in f]

        if data_limit > 0:
            self.indices = self.indices[:data_limit]

        # Load teacher manifest if available
        self._teacher_map: Dict[str, dict] = {}
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self._teacher_map[entry["stem"]] = entry

    # ── Data caching ───────────────────────────────────────────────────

    def _ensure_cached(self):
        """Download NYU Depth V2 from HuggingFace and cache as individual files."""
        marker = self._cache_dir / ".download_complete"
        if marker.exists():
            return

        print("Downloading NYU Depth V2 from HuggingFace (0jl/NYUv2)...")
        print("This will download ~2.8 GB and may take a few minutes.")

        from datasets import load_dataset

        ds = load_dataset("0jl/NYUv2", trust_remote_code=True, split="train")

        n_samples = len(ds)
        print(f"Dataset has {n_samples} samples.")
        if n_samples < 1400:
            raise RuntimeError(
                f"Expected ~1449 samples, got {n_samples}. "
                "Check that the dataset ID '0jl/NYUv2' is correct."
            )

        rgb_dir = self._cache_dir / "rgb"
        depth_dir = self._cache_dir / "depth"
        seg_dir = self._cache_dir / "seg"
        for d in [rgb_dir, depth_dir, seg_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(ds):
            stem = f"{i:05d}"
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img.save(rgb_dir / f"{stem}.png")

            depth = np.array(sample["depth"], dtype=np.float32)
            np.save(depth_dir / f"{stem}.npy", depth)

            label = np.array(sample["label"], dtype=np.int32)
            np.save(seg_dir / f"{stem}.npy", label)

        # Create 80/20 train/val split
        all_indices = list(range(n_samples))
        rng = random.Random(42)
        rng.shuffle(all_indices)
        split_point = int(0.8 * n_samples)

        with open(self._cache_dir / "train_indices.txt", "w") as f:
            for idx in all_indices[:split_point]:
                f.write(f"{idx}\n")
        with open(self._cache_dir / "val_indices.txt", "w") as f:
            for idx in all_indices[split_point:]:
                f.write(f"{idx}\n")

        marker.touch()
        print(f"Cached {n_samples} samples to {self._cache_dir}")

    # ── Sample loading ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        sample_idx = self.indices[idx]
        stem = f"{sample_idx:05d}"

        rgb = Image.open(self._cache_dir / "rgb" / f"{stem}.png").convert("RGB")
        depth = np.load(self._cache_dir / "depth" / f"{stem}.npy")
        seg_raw = np.load(self._cache_dir / "seg" / f"{stem}.npy")

        seg = remap_labels(seg_raw)

        # Synthesize confidence from depth validity (NYU stand-in for ToF confidence)
        confidence = (depth > 0).astype(np.float32)

        # Load DA2 teacher depth if available
        has_da2 = False
        da2_depth = np.zeros_like(depth)
        if stem in self._teacher_map:
            entry = self._teacher_map[stem]
            da2_path = entry.get("da2_depth")
            if da2_path and os.path.exists(da2_path):
                da2_depth = np.load(da2_path).astype(np.float32)
                has_da2 = True

        rgb, depth, seg, confidence, da2_depth = self._transform(
            rgb, depth, seg, confidence, da2_depth
        )

        return {
            "rgb": rgb,
            "depth": depth,
            "seg": seg,
            "confidence": confidence,
            "da2_depth": da2_depth,
            "has_da2": has_da2,
        }

    # ── Transforms ─────────────────────────────────────────────────────

    def _transform(
        self,
        rgb: Image.Image,
        depth: np.ndarray,
        seg: np.ndarray,
        confidence: np.ndarray,
        da2_depth: np.ndarray,
    ) -> Tuple[torch.Tensor, ...]:
        """Resize, augment, and convert to tensors."""
        w, h = rgb.size

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = np.fliplr(depth).copy()
                seg = np.fliplr(seg).copy()
                confidence = np.fliplr(confidence).copy()
                da2_depth = np.fliplr(da2_depth).copy()

            # Color jitter (RGB only)
            from torchvision import transforms as T
            jitter = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            rgb = jitter(rgb)

            # Random crop (if image is larger than target)
            if h > self.height and w > self.width:
                top = random.randint(0, h - self.height)
                left = random.randint(0, w - self.width)
                rgb = rgb.crop((left, top, left + self.width, top + self.height))
                depth = depth[top:top + self.height, left:left + self.width]
                seg = seg[top:top + self.height, left:left + self.width]
                confidence = confidence[top:top + self.height, left:left + self.width]
                da2_depth = da2_depth[top:top + self.height, left:left + self.width]
            else:
                rgb = rgb.resize((self.width, self.height), Image.BILINEAR)
                depth = _resize_np(depth, self.height, self.width)
                seg = _resize_np(seg, self.height, self.width, order=0)
                confidence = _resize_np(confidence, self.height, self.width)
                da2_depth = _resize_np(da2_depth, self.height, self.width)
        else:
            # Validation: simple resize
            rgb = rgb.resize((self.width, self.height), Image.BILINEAR)
            depth = _resize_np(depth, self.height, self.width)
            seg = _resize_np(seg, self.height, self.width, order=0)
            confidence = _resize_np(confidence, self.height, self.width)
            da2_depth = _resize_np(da2_depth, self.height, self.width)

        # To tensors
        rgb_t = torch.from_numpy(
            np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        depth_t = torch.from_numpy(depth[np.newaxis].astype(np.float32))
        seg_t = torch.from_numpy(seg.astype(np.int64))
        conf_t = torch.from_numpy(confidence[np.newaxis].astype(np.float32))
        da2_t = torch.from_numpy(da2_depth[np.newaxis].astype(np.float32))

        return rgb_t, depth_t, seg_t, conf_t, da2_t


def _resize_np(arr: np.ndarray, h: int, w: int, order: int = 1) -> np.ndarray:
    """Resize a 2D numpy array using PIL (avoids skimage dependency)."""
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    mode = "NEAREST" if order == 0 else "BILINEAR"
    pil_img = Image.fromarray(arr)
    resized = pil_img.resize((w, h), getattr(Image, mode, Image.BILINEAR))
    return np.array(resized, dtype=arr.dtype)


# ────────────────────────────────────────────────────────────────────────────
# ▶ FACTORY: get_dataloaders
# ────────────────────────────────────────────────────────────────────────────


def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders based on config.

    If cfg.USE_CORRIDOR_DATA is True, delegates to corridor_loader.
    """
    if cfg.USE_CORRIDOR_DATA:
        from .corridor_loader import get_corridor_dataloaders
        return get_corridor_dataloaders(cfg)

    train_ds = NYUDepthV2Dataset(
        data_root=cfg.DATA_ROOT,
        split="train",
        height=cfg.INPUT_HEIGHT,
        width=cfg.INPUT_WIDTH,
        augment=True,
        manifest_path=cfg.MANIFEST_PATH,
        data_limit=cfg.DATA_LIMIT,
    )
    val_ds = NYUDepthV2Dataset(
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
