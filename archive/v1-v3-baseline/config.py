"""
Central configuration for the multi-task student model training pipeline.

All hyperparameters, paths, loss weights, and dataset flags live here.
Override any value via command-line args in train.py or environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Dataset selection ──────────────────────────────────────────────
    # Set True when corridor RGB/ToF/DA2/SAM2 data is available.
    # When False, uses NYU Depth V2 as a stand-in dataset.
    USE_CORRIDOR_DATA: bool = False

    # ── Paths ──────────────────────────────────────────────────────────
    DATA_ROOT: str = os.environ.get("DATA_ROOT", "./data")
    CHECKPOINT_DIR: str = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
    LOG_DIR: str = os.environ.get("LOG_DIR", "./runs")

    # Teacher prediction manifest (produced by teacher_infer/build_manifest.py)
    MANIFEST_PATH: Optional[str] = None

    # Corridor data layout (used when USE_CORRIDOR_DATA=True)
    CORRIDOR_RGB_DIR: str = "corridor_data/rgb"
    CORRIDOR_DEPTH_DIR: str = "corridor_data/depth"
    CORRIDOR_DA2_DIR: str = "corridor_data/da2_depth"
    CORRIDOR_SAM2_DIR: str = "corridor_data/sam2_seg"
    CORRIDOR_CONFIDENCE_DIR: str = "corridor_data/confidence"

    # ── Model ──────────────────────────────────────────────────────────
    NUM_CLASSES: int = 6  # floor, wall, person, furniture, glass, other
    INPUT_HEIGHT: int = 240
    INPUT_WIDTH: int = 320

    # ── Training ───────────────────────────────────────────────────────
    EPOCHS: int = 100
    BATCH_SIZE: int = 16
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 1.0

    # ── Loss weights ───────────────────────────────────────────────────
    LAMBDA_DEPTH: float = 1.0
    LAMBDA_SEG: float = 0.5
    LAMBDA_EDGE: float = 0.1

    # Confidence threshold for hybrid depth target (mirrors runtime fusion)
    CONFIDENCE_THRESHOLD: float = 0.5

    # ── Data loading ───────────────────────────────────────────────────
    NUM_WORKERS: int = 4
    VAL_SPLIT: float = 0.2
    SEED: int = 42

    # Limit dataset size for debugging (0 = use all)
    DATA_LIMIT: int = 0

    # ── Augmentation (train only) ──────────────────────────────────────
    AUG_HFLIP: bool = True
    AUG_COLOR_JITTER: bool = True
    COLOR_JITTER_BRIGHTNESS: float = 0.2
    COLOR_JITTER_CONTRAST: float = 0.2
    COLOR_JITTER_SATURATION: float = 0.2
    COLOR_JITTER_HUE: float = 0.1

    # ── Checkpointing ─────────────────────────────────────────────────
    SAVE_EVERY: int = 5  # save checkpoint every N epochs
    KEEP_BEST: bool = True  # always keep best val-loss checkpoint

    # ── Export ─────────────────────────────────────────────────────────
    ONNX_OPSET: int = 17
    TRT_INT8_CALIB_IMAGES: int = 500

    # ── Class names (index = label id) ─────────────────────────────────
    CLASS_NAMES: tuple = ("floor", "wall", "person", "furniture", "glass", "other")
