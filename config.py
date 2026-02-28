"""
Central configuration for the multi-task student model training pipeline.

All hyperparameters, paths, loss weights, and dataset flags live here.
Override any value via command-line args in train.py or environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ── Dataset selection ──────────────────────────────────────────────
    USE_CORRIDOR_DATA: bool = False

    # ── Paths ──────────────────────────────────────────────────────────
    DATA_ROOT: str = os.environ.get("DATA_ROOT", "./data")
    CHECKPOINT_DIR: str = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
    LOG_DIR: str = os.environ.get("LOG_DIR", "./runs")
    TEACHER_OUT_ROOT: str = os.environ.get("TEACHER_OUT_ROOT", "./data/teacher_output")

    MANIFEST_PATH: Optional[str] = None

    # Corridor data layout (used when USE_CORRIDOR_DATA=True)
    CORRIDOR_RGB_DIR: str = "corridor_data/rgb"
    CORRIDOR_DEPTH_DIR: str = "corridor_data/depth"
    CORRIDOR_DA3_DIR: str = "corridor_data/da3_depth"
    CORRIDOR_SAM2_DIR: str = "corridor_data/sam2_seg"
    CORRIDOR_CONFIDENCE_DIR: str = "corridor_data/confidence"

    # ── Model ──────────────────────────────────────────────────────────
    NUM_CLASSES: int = 6  # floor, wall, person, furniture, glass, other
    INPUT_HEIGHT: int = 240
    INPUT_WIDTH: int = 320

    # Student backbone (EfficientViT-B1 via timm)
    BACKBONE: str = "efficientvit_b1.r288_in1k"
    # Verified by print_model_shapes.py: stages [32, 64, 128, 256] at 1/4, 1/8, 1/16, 1/32
    SKIP_CHANNELS: tuple = (32, 64, 128)  # stages 0, 1, 2
    BOTTLENECK_CH: int = 256               # stage 3

    # ── Teacher models ─────────────────────────────────────────────────
    DA3_MODEL: str = "depth-anything/da3metric-large"
    DA3_CANONICAL_FOCAL: float = 300.0

    # NYU camera intrinsics (for DA3 metric conversion: metric = focal * raw / 300)
    NYU_FX: float = 518.8579
    NYU_FY: float = 519.4696
    NYU_CX: float = 325.5824
    NYU_CY: float = 253.7362
    NYU_W: int = 640
    NYU_H: int = 480

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
    CONFIDENCE_THRESHOLD: float = 0.5

    # ── Data loading ───────────────────────────────────────────────────
    NUM_WORKERS: int = 4
    VAL_SPLIT: float = 0.2
    SEED: int = 42
    DATA_LIMIT: int = 0

    # ── Augmentation (train only) ──────────────────────────────────────
    AUG_HFLIP: bool = True
    AUG_COLOR_JITTER: bool = True
    COLOR_JITTER_BRIGHTNESS: float = 0.2
    COLOR_JITTER_CONTRAST: float = 0.2
    COLOR_JITTER_SATURATION: float = 0.2
    COLOR_JITTER_HUE: float = 0.1

    # ── Checkpointing ─────────────────────────────────────────────────
    SAVE_EVERY: int = 5
    KEEP_BEST: bool = True

    # ── Export ─────────────────────────────────────────────────────────
    ONNX_OPSET: int = 17
    TRT_INT8_CALIB_IMAGES: int = 500

    # ── Class names (index = label id) ─────────────────────────────────
    CLASS_NAMES: tuple = ("floor", "wall", "person", "furniture", "glass", "other")
