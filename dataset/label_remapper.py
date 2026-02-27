"""
Remap NYU Depth V2 semantic labels to our 6-class scheme.

Two-stage mapping:
    894 raw NYU classes  -->  40 standard NYU classes  -->  6 project classes

The 894->40 mapping comes from the NYUv2 metadata (classMapping40.mat).
The 40->6 mapping is defined here based on our project's class taxonomy:

    0: floor      (NYU40: floor, floor_mat)
    1: wall       (NYU40: wall, ceiling, window, door)
    2: person     (NYU40: person)
    3: furniture  (NYU40: cabinet, bed, chair, sofa, table, bookshelf,
                   counter, desk, shelves, dresser, refrigerator,
                   night_stand, toilet, sink, bathtub, otherfurniture)
    4: glass      (NYU40: mirror)
    5: other      (everything else)
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# ── NYU40 class names (1-indexed: id 1 = wall, id 2 = floor, ...) ──────────
NYU40_NAMES = [
    "void",           # 0 – unlabeled / background
    "wall",           # 1
    "floor",          # 2
    "cabinet",        # 3
    "bed",            # 4
    "chair",          # 5
    "sofa",           # 6
    "table",          # 7
    "door",           # 8
    "window",         # 9
    "bookshelf",      # 10
    "picture",        # 11
    "counter",        # 12
    "blinds",         # 13
    "desk",           # 14
    "shelves",        # 15
    "curtain",        # 16
    "dresser",        # 17
    "pillow",         # 18
    "mirror",         # 19
    "floor_mat",      # 20
    "clothes",        # 21
    "ceiling",        # 22
    "books",          # 23
    "refrigerator",   # 24
    "television",     # 25
    "paper",          # 26
    "towel",          # 27
    "shower_curtain", # 28
    "box",            # 29
    "whiteboard",     # 30
    "person",         # 31
    "night_stand",    # 32
    "toilet",         # 33
    "sink",           # 34
    "lamp",           # 35
    "bathtub",        # 36
    "bag",            # 37
    "otherstructure", # 38
    "otherfurniture", # 39
    "otherprop",      # 40
]

# ── Our 6-class scheme ─────────────────────────────────────────────────────
OUR_CLASS_NAMES = ("floor", "wall", "person", "furniture", "glass", "other")
IGNORE_LABEL = 255

# NYU40 id  -->  our 6-class id
# 0 (void/unlabeled) maps to IGNORE_LABEL so cross-entropy ignores it.
_NYU40_TO_6 = np.full(41, 5, dtype=np.uint8)  # default = "other"
_NYU40_TO_6[0] = IGNORE_LABEL                 # unlabeled
_NYU40_TO_6[2] = 0    # floor
_NYU40_TO_6[20] = 0   # floor_mat
_NYU40_TO_6[1] = 1    # wall
_NYU40_TO_6[22] = 1   # ceiling
_NYU40_TO_6[9] = 1    # window
_NYU40_TO_6[8] = 1    # door
_NYU40_TO_6[31] = 2   # person
for fid in [3, 4, 5, 6, 7, 10, 12, 14, 15, 17, 24, 32, 33, 34, 36, 39]:
    _NYU40_TO_6[fid] = 3   # furniture
_NYU40_TO_6[19] = 4   # mirror -> glass


def _get_894_to_40_lut(cache_dir: Optional[str] = None) -> np.ndarray:
    """
    Build a lookup table mapping raw NYU 894-class ids to NYU40 ids.

    Downloads classMapping40.mat from the nyuv2-meta-data repo if not cached.
    Returns an int array of shape (895,) where lut[raw_id] = nyu40_id.
    Index 0 (unlabeled) maps to 0.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    npy_path = os.path.join(cache_dir, "nyu894_to_40.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)

    mat_path = os.path.join(cache_dir, "classMapping40.mat")
    if not os.path.exists(mat_path):
        url = (
            "https://raw.githubusercontent.com/ankurhanda/"
            "nyuv2-meta-data/master/classMapping40.mat"
        )
        print(f"Downloading 894->40 class mapping from {url} ...")
        urllib.request.urlretrieve(url, mat_path)

    from scipy.io import loadmat
    mapping = loadmat(mat_path)["mapClass"][0]   # shape (894,), 1-indexed values
    lut = np.zeros(len(mapping) + 1, dtype=np.int32)
    lut[1:] = mapping  # raw id 1..894 -> nyu40 id
    lut[0] = 0         # unlabeled stays 0

    np.save(npy_path, lut)
    return lut


class LabelRemapper:
    """
    Converts NYU Depth V2 labels to our 6-class scheme.

    Handles two input formats:
        - Raw 894-class labels (max value > 40): applies 894->40->6
        - Pre-mapped 40-class labels (max value <= 40): applies 40->6 only
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._nyu40_to_6 = _NYU40_TO_6.copy()
        self._lut_894_to_40: Optional[np.ndarray] = None
        self._cache_dir = cache_dir

    def _ensure_894_lut(self) -> np.ndarray:
        if self._lut_894_to_40 is None:
            self._lut_894_to_40 = _get_894_to_40_lut(self._cache_dir)
        return self._lut_894_to_40

    def remap(self, labels: np.ndarray) -> np.ndarray:
        """
        Remap a label array to our 6-class scheme.

        Args:
            labels: integer array (H, W) with NYU class ids (raw 894 or 40).

        Returns:
            uint8 array (H, W) with values in {0..5, 255}.
            255 = ignore (unlabeled pixels).
        """
        max_val = int(labels.max())

        if max_val > 40:
            lut_894 = self._ensure_894_lut()
            safe_labels = np.clip(labels, 0, len(lut_894) - 1)
            labels_40 = lut_894[safe_labels]
        else:
            labels_40 = labels.astype(np.int32)

        labels_40 = np.clip(labels_40, 0, 40)
        return self._nyu40_to_6[labels_40]

    def remap_single(self, nyu40_id: int) -> int:
        """Remap a single NYU40 class id to our 6-class id."""
        if 0 <= nyu40_id <= 40:
            return int(self._nyu40_to_6[nyu40_id])
        return int(IGNORE_LABEL)


# Module-level convenience instance
_default_remapper: Optional[LabelRemapper] = None


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """Convenience function: remap NYU labels to 6-class scheme."""
    global _default_remapper
    if _default_remapper is None:
        _default_remapper = LabelRemapper()
    return _default_remapper.remap(labels)
