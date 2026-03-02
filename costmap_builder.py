#!/usr/bin/env python3
"""
Depth-to-occupancy-grid converter for costmap ablation study.

Converts a single depth image (float32, metres) into a 2D occupancy grid
that simulates Nav2's ObstacleLayer. Parameters default to the values in
NCHSB/rc_hardware_bringup/config/nav2_hardware.yaml.

Pipeline per frame:
    1. Back-project depth pixels to 3D points using camera intrinsics
    2. Filter by height band (keep robot-relevant obstacles)
    3. Filter by depth range
    4. Rasterise surviving points into a bird's-eye-view 2D grid
    5. Return uint8 grid: 0 = free, 254 = lethal (Nav2 convention)

Usage:
    from costmap_builder import depth_to_occupancy_grid

    grid = depth_to_occupancy_grid(depth_m, intrinsics)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union


# Nav2 cost values
LETHAL_COST = 254
FREE_COST = 0

# Segmentation class labels (matches run_sam2.py output)
SEG_FLOOR = 0
SEG_WALL = 1
SEG_PERSON = 2
SEG_FURNITURE = 3
SEG_GLASS = 4
SEG_OTHER = 5
SEG_UNKNOWN = 255


def back_project_depth(
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> np.ndarray:
    """
    Back-project a depth image to 3D points in the camera optical frame.

    Args:
        depth_m: (H, W) float32 depth in metres.
        intrinsics: dict with keys fx, fy, cx, cy.
        min_depth: discard points closer than this.
        max_depth: discard points farther than this.

    Returns:
        (N, 3) float64 array of [X, Y, Z] points in camera optical frame.
        In optical frame convention: X=right, Y=down, Z=forward.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    valid = (depth_m >= min_depth) & (depth_m <= max_depth)
    vs, us = np.where(valid)
    ds = depth_m[vs, us].astype(np.float64)

    X = (us.astype(np.float64) - cx) * ds / fx
    Y = (vs.astype(np.float64) - cy) * ds / fy
    Z = ds

    return np.stack([X, Y, Z], axis=1)


def back_project_depth_with_pixels(
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth to 3D and also return original pixel coordinates.

    Returns:
        Tuple of ((N, 3) 3D points, (N, 2) pixel coords as [row, col]).
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    valid = (depth_m >= min_depth) & (depth_m <= max_depth)
    vs, us = np.where(valid)
    ds = depth_m[vs, us].astype(np.float64)

    X = (us.astype(np.float64) - cx) * ds / fx
    Y = (vs.astype(np.float64) - cy) * ds / fy
    Z = ds

    points = np.stack([X, Y, Z], axis=1)
    pixel_coords = np.stack([vs, us], axis=1)
    return points, pixel_coords


def filter_by_height(
    points: np.ndarray,
    min_height: float = 0.05,
    max_height: float = 0.50,
    camera_height: float = 0.25,
) -> np.ndarray:
    """
    Filter 3D points by height relative to the ground plane.

    In the camera optical frame, Y points downward. A point at ground level
    has Y ≈ +camera_height (camera looks forward, mounted at camera_height
    above ground). An obstacle at height h has Y ≈ camera_height - h.

    We keep points whose estimated height above ground is in
    [min_height, max_height].

    Args:
        points: (N, 3) array in camera optical frame.
        min_height: minimum obstacle height above ground (metres).
        max_height: maximum obstacle height above ground (metres).
        camera_height: camera mount height above ground (metres).

    Returns:
        Filtered (M, 3) array.
    """
    height_above_ground = camera_height - points[:, 1]
    mask = (height_above_ground >= min_height) & (height_above_ground <= max_height)
    return points[mask]


def rasterise_to_grid(
    points: np.ndarray,
    grid_size: float = 6.0,
    resolution: float = 0.05,
) -> np.ndarray:
    """
    Rasterise 3D points into a bird's-eye-view occupancy grid.

    The robot is at the centre of the grid. Forward (camera Z) maps to
    grid rows decreasing (top of grid = far away), right (camera X) maps
    to grid columns increasing.

    Args:
        points: (N, 3) array in camera optical frame [X_right, Y_down, Z_fwd].
        grid_size: side length of square grid in metres.
        resolution: metres per cell.

    Returns:
        (grid_cells, grid_cells) uint8 array. 0=free, 254=lethal.
    """
    grid_cells = int(grid_size / resolution)
    grid = np.zeros((grid_cells, grid_cells), dtype=np.uint8)

    if len(points) == 0:
        return grid

    half = grid_size / 2.0

    # Camera X (right) -> grid column
    col = ((points[:, 0] + half) / resolution).astype(np.int32)
    # Camera Z (forward) -> grid row (invert so forward = top)
    row = ((half - points[:, 2]) / resolution).astype(np.int32)

    # Bounds check
    valid = (col >= 0) & (col < grid_cells) & (row >= 0) & (row < grid_cells)
    col, row = col[valid], row[valid]

    grid[row, col] = LETHAL_COST

    return grid


def depth_to_occupancy_grid(
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    grid_size: float = 6.0,
    resolution: float = 0.05,
    min_height: float = 0.05,
    max_height: float = 0.50,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
    camera_height: float = 0.25,
) -> np.ndarray:
    """
    Full pipeline: depth image -> 2D occupancy grid.

    Args:
        depth_m: (H, W) float32 depth in metres.
        intrinsics: dict with keys fx, fy, cx, cy.
        grid_size: local costmap side length (metres). Default 6.0 from nav2_hardware.yaml.
        resolution: metres per cell. Default 0.05 from nav2_hardware.yaml.
        min_height: minimum obstacle height above ground (metres).
        max_height: maximum obstacle height. Default 0.5 from nav2_hardware.yaml.
        min_depth: minimum valid depth (metres).
        max_depth: maximum valid depth (metres).
        camera_height: camera mount height above ground (metres).

    Returns:
        (grid_cells, grid_cells) uint8 grid. 0=free, 254=lethal.
    """
    points = back_project_depth(depth_m, intrinsics, min_depth, max_depth)

    if len(points) == 0:
        grid_cells = int(grid_size / resolution)
        return np.zeros((grid_cells, grid_cells), dtype=np.uint8)

    points = filter_by_height(points, min_height, max_height, camera_height)
    grid = rasterise_to_grid(points, grid_size, resolution)

    return grid


def build_class_grid(
    depth_m: np.ndarray,
    seg_mask: np.ndarray,
    intrinsics: Dict[str, float],
    grid_size: float = 6.0,
    resolution: float = 0.05,
    min_height: float = 0.05,
    max_height: float = 0.50,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
    camera_height: float = 0.25,
) -> np.ndarray:
    """
    Build a bird's-eye-view class grid from depth + segmentation mask.

    Each cell in the output grid contains the semantic class (0-5) of the
    obstacle at that location, or SEG_UNKNOWN (255) if empty. When multiple
    classes map to the same cell, the most safety-critical class wins
    (person > glass > furniture > other > wall > floor).

    Args:
        depth_m: (H, W) float32 depth in metres.
        seg_mask: (H, W) uint8 segmentation labels (0-5 from YOLO+SAM2).
        intrinsics: dict with keys fx, fy, cx, cy.
        grid_size: side length of square grid in metres.
        resolution: metres per cell.
        min_height: minimum obstacle height above ground (metres).
        max_height: maximum obstacle height above ground (metres).
        min_depth: minimum valid depth (metres).
        max_depth: maximum valid depth (metres).
        camera_height: camera mount height above ground (metres).

    Returns:
        (grid_cells, grid_cells) uint8 class grid. 255 = no obstacle.
    """
    grid_cells = int(grid_size / resolution)

    points, pixel_coords = back_project_depth_with_pixels(
        depth_m, intrinsics, min_depth, max_depth,
    )
    if len(points) == 0:
        return np.full((grid_cells, grid_cells), SEG_UNKNOWN, dtype=np.uint8)

    height_above_ground = camera_height - points[:, 1]
    h_mask = (height_above_ground >= min_height) & (height_above_ground <= max_height)
    points = points[h_mask]
    pixel_coords = pixel_coords[h_mask]

    if len(points) == 0:
        return np.full((grid_cells, grid_cells), SEG_UNKNOWN, dtype=np.uint8)

    seg_h, seg_w = seg_mask.shape[:2]
    rows = np.clip(pixel_coords[:, 0], 0, seg_h - 1)
    cols = np.clip(pixel_coords[:, 1], 0, seg_w - 1)
    classes = seg_mask[rows, cols]

    half = grid_size / 2.0
    gc = ((points[:, 0] + half) / resolution).astype(np.int32)
    gr = ((half - points[:, 2]) / resolution).astype(np.int32)
    valid = (gc >= 0) & (gc < grid_cells) & (gr >= 0) & (gr < grid_cells)
    gc, gr = gc[valid], gr[valid]
    classes = classes[valid]

    class_grid = np.full((grid_cells, grid_cells), SEG_UNKNOWN, dtype=np.uint8)

    # Assign in increasing priority order so highest-priority class wins
    priority_order = [SEG_FLOOR, SEG_WALL, SEG_OTHER, SEG_FURNITURE, SEG_GLASS, SEG_PERSON]
    for cls in priority_order:
        cls_mask = classes == cls
        if np.any(cls_mask):
            class_grid[gr[cls_mask], gc[cls_mask]] = cls

    return class_grid


def compute_costmap_metrics(
    grid_pred: np.ndarray,
    grid_gt: np.ndarray,
) -> Dict[str, float]:
    """
    Compare two occupancy grids and return ablation metrics.

    Args:
        grid_pred: predicted occupancy grid (e.g. from DA3 depth).
        grid_gt: ground truth occupancy grid (from sensor depth).

    Returns:
        dict with keys: iou, detection_rate, false_positive_rate,
        clearance_pred, clearance_gt, occupied_cells_pred, occupied_cells_gt.
    """
    # Cost threshold: cells at or above this are considered "occupied"
    # for metric purposes. This captures both lethal cells (254) and
    # significantly inflated cells, making the metric sensitive to
    # inflation radius differences.
    COST_THRESHOLD = 128

    pred_occ_lethal = grid_pred == LETHAL_COST
    gt_occ_lethal = grid_gt == LETHAL_COST

    pred_occ = grid_pred >= COST_THRESHOLD
    gt_occ = grid_gt >= COST_THRESHOLD
    gt_free = ~gt_occ

    # IoU of occupied cells (includes inflated region)
    intersection = np.sum(pred_occ & gt_occ)
    union = np.sum(pred_occ | gt_occ)
    iou = float(intersection / union) if union > 0 else 1.0

    # Lethal IoU: agreement on raw obstacle cells only (no inflation)
    lethal_inter = np.sum(pred_occ_lethal & gt_occ_lethal)
    lethal_union = np.sum(pred_occ_lethal | gt_occ_lethal)
    lethal_iou = float(lethal_inter / lethal_union) if lethal_union > 0 else 1.0

    # Obstacle detection rate: % of GT lethal cells also detected by pred
    gt_lethal_total = np.sum(gt_occ_lethal)
    detection_rate = float(
        np.sum(pred_occ_lethal & gt_occ_lethal) / gt_lethal_total
    ) if gt_lethal_total > 0 else 1.0

    # False positive rate: % of GT free cells incorrectly marked occupied
    gt_free_total = np.sum(gt_free)
    false_pos = float(np.sum(pred_occ & gt_free) / gt_free_total) if gt_free_total > 0 else 0.0

    # Clearance: min distance from grid centre to nearest occupied cell
    # (uses cost threshold, so inflation radius affects this metric)
    grid_h, grid_w = grid_pred.shape
    centre = np.array([grid_h / 2.0, grid_w / 2.0])
    cell_to_m = 6.0 / grid_h

    pred_occ_coords = np.argwhere(pred_occ)
    if len(pred_occ_coords) > 0:
        dists = np.linalg.norm(pred_occ_coords - centre, axis=1)
        clearance_pred = float(np.min(dists)) * cell_to_m
    else:
        clearance_pred = float(grid_h / 2.0) * cell_to_m

    gt_occ_coords = np.argwhere(gt_occ)
    if len(gt_occ_coords) > 0:
        dists = np.linalg.norm(gt_occ_coords - centre, axis=1)
        clearance_gt = float(np.min(dists)) * cell_to_m
    else:
        clearance_gt = float(grid_h / 2.0) * cell_to_m

    return {
        "iou": iou,
        "lethal_iou": lethal_iou,
        "detection_rate": detection_rate,
        "false_positive_rate": false_pos,
        "clearance_pred_m": clearance_pred,
        "clearance_gt_m": clearance_gt,
        "occupied_cells_pred": int(np.sum(pred_occ)),
        "occupied_cells_gt": int(np.sum(gt_occ)),
    }
