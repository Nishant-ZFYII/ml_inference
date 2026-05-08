#!/usr/bin/env python3
"""
Inflation module for costmap ablation study.

Replicates Nav2's InflationLayer behaviour (fixed radius with exponential
cost decay) and adds three dynamic inflation strategies that adapt the
radius per frame based on depth information.

Strategies:
    - fixed: constant radius (Nav2 default, 0.09m)
    - corridor_width: estimates passage width from horizontal depth profile,
      sets radius proportional to available clearance
    - min_depth: radius inversely proportional to closest obstacle distance
    - class_aware: per-class radius using YOLO detections (stub, requires
      detection data)

Usage:
    from inflation import apply_fixed_inflation, apply_dynamic_inflation

    inflated = apply_fixed_inflation(grid, radius=0.09)
    inflated, radius = apply_dynamic_inflation(grid, depth_m, strategy="corridor_width")
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, Optional, Tuple

from costmap_builder import (
    LETHAL_COST, FREE_COST,
    SEG_FLOOR, SEG_WALL, SEG_PERSON, SEG_FURNITURE,
    SEG_GLASS, SEG_OTHER, SEG_UNKNOWN,
)


# Nav2 inflation cost model:
#   cost(d) = (INSCRIBED - 1) * exp(-cost_scaling * (d - inscribed_radius))
# We simplify: cells within inflation_radius get decaying cost.
INSCRIBED_COST = 253
NO_INFORMATION = 255


def _exponential_decay_costs(
    grid: np.ndarray,
    radius_cells: float,
    cost_scaling: float = 3.0,
) -> np.ndarray:
    """
    Apply Nav2-style exponential cost decay around lethal cells.

    For each cell, cost decreases exponentially with distance from the
    nearest lethal cell, dropping to zero beyond radius_cells.
    """
    lethal_mask = grid == LETHAL_COST
    if not np.any(lethal_mask):
        return grid.copy()

    # Distance transform: distance of each cell to nearest lethal cell
    dist = distance_transform_edt(~lethal_mask)

    inflated = grid.copy()

    within_radius = (dist > 0) & (dist <= radius_cells)
    decay = (INSCRIBED_COST - 1) * np.exp(-cost_scaling * (dist[within_radius] - 1.0))
    decay = np.clip(decay, 1, INSCRIBED_COST - 1).astype(np.uint8)

    # Only inflate cells that are currently free (don't downgrade lethal)
    free_and_inflatable = within_radius & (inflated < LETHAL_COST)
    dist_free = dist[free_and_inflatable]
    decay_free = (INSCRIBED_COST - 1) * np.exp(-cost_scaling * (dist_free - 1.0))
    decay_free = np.clip(decay_free, 1, INSCRIBED_COST - 1).astype(np.uint8)
    inflated[free_and_inflatable] = np.maximum(
        inflated[free_and_inflatable], decay_free
    )

    return inflated


def apply_fixed_inflation(
    grid: np.ndarray,
    radius: float = 0.09,
    resolution: float = 0.05,
    cost_scaling: float = 3.0,
) -> np.ndarray:
    """
    Replicate Nav2 InflationLayer with a fixed radius.

    Args:
        grid: (H, W) uint8 occupancy grid (0=free, 254=lethal).
        radius: inflation radius in metres. Default 0.09m from nav2_hardware.yaml.
        resolution: grid resolution in m/cell. Default 0.05m.
        cost_scaling: exponential decay factor. Default 3.0 from nav2_hardware.yaml.

    Returns:
        (H, W) uint8 inflated grid.
    """
    radius_cells = radius / resolution
    return _exponential_decay_costs(grid, radius_cells, cost_scaling)


def estimate_corridor_width(
    depth_m: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> float:
    """
    Estimate passage width from the horizontal depth profile at the
    middle row of the image.

    Takes the middle band (40-60% height) of the depth image, computes
    the median depth at each column, and estimates width as the distance
    between the leftmost and rightmost columns with depth below a
    threshold (walls).

    Args:
        depth_m: (H, W) float32 depth in metres.
        min_depth: minimum valid depth.
        max_depth: maximum valid depth.

    Returns:
        Estimated corridor width in metres. Returns max_depth * 2 if
        no corridor walls are detected.
    """
    h, w = depth_m.shape

    # Use the middle 20% of rows for stability
    row_start = int(h * 0.4)
    row_end = int(h * 0.6)
    band = depth_m[row_start:row_end, :]

    # Median depth per column (ignoring invalid pixels)
    col_depths = np.full(w, np.nan, dtype=np.float64)
    for c in range(w):
        col = band[:, c]
        valid = (col >= min_depth) & (col <= max_depth)
        if np.sum(valid) > 0:
            col_depths[c] = np.median(col[valid])

    valid_cols = ~np.isnan(col_depths)
    if np.sum(valid_cols) < 2:
        return max_depth * 2.0

    # Wall detection: columns where depth is below the 30th percentile
    # are considered "wall" (close obstacles)
    valid_depths = col_depths[valid_cols]
    wall_threshold = np.percentile(valid_depths, 30)

    wall_cols = np.where(valid_cols & (col_depths <= wall_threshold))[0]
    if len(wall_cols) < 2:
        return max_depth * 2.0

    # Angular width between leftmost and rightmost wall columns
    left_wall_col = wall_cols[0]
    right_wall_col = wall_cols[-1]
    left_depth = col_depths[left_wall_col]
    right_depth = col_depths[right_wall_col]

    # Approximate width using the depth at each wall and the angular span
    # Simple triangle approximation: width ≈ 2 * depth * tan(half_angle)
    # For a more direct estimate: use the mean wall depth and pixel span
    mean_wall_depth = (left_depth + right_depth) / 2.0
    pixel_span = right_wall_col - left_wall_col

    # Rough FOV-based width estimate (assuming ~80° HFOV for Femto Bolt color)
    hfov_rad = np.radians(80.0)
    width_at_depth = 2.0 * mean_wall_depth * np.tan(hfov_rad / 2.0) * (pixel_span / w)

    return float(max(0.5, width_at_depth))


def compute_dynamic_radius(
    depth_m: np.ndarray,
    strategy: str = "corridor_width",
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> float:
    """
    Compute an adaptive inflation radius based on depth information.

    Args:
        depth_m: (H, W) float32 depth in metres.
        strategy: one of 'corridor_width', 'min_depth', 'class_aware'.
        min_depth: minimum valid depth for filtering.
        max_depth: maximum valid depth for filtering.

    Returns:
        Inflation radius in metres.
    """
    if strategy == "corridor_width":
        width = estimate_corridor_width(depth_m, min_depth, max_depth)
        # Narrower corridor -> larger inflation for safety margin
        # Wider corridor -> smaller inflation for manoeuvrability
        #   width <= 0.5m  -> 0.18m radius (tight, max safety)
        #   width  = 1.0m  -> 0.12m radius (moderate)
        #   width  = 2.0m  -> 0.06m radius (wide, max manoeuvre)
        #   width >= 3.0m  -> 0.05m radius (open space)
        if width <= 0.5:
            radius = 0.18
        elif width >= 3.0:
            radius = 0.05
        else:
            # Linear interpolation: 0.18 at 0.5m, 0.05 at 3.0m
            radius = 0.18 - (width - 0.5) * (0.13 / 2.5)
        return float(np.clip(radius, 0.05, 0.25))

    elif strategy == "min_depth":
        valid = (depth_m >= min_depth) & (depth_m <= max_depth)
        if not np.any(valid):
            return 0.09

        # Use 5th percentile of valid depths (robust minimum)
        min_d = np.percentile(depth_m[valid], 5)

        # Closer obstacles -> larger inflation for safety
        # Far obstacles -> smaller inflation for manoeuvrability
        #   < 0.3m  -> 0.25m radius (imminent collision zone)
        #   0.3-1m  -> linear 0.25 -> 0.12
        #   1.0-3m  -> linear 0.12 -> 0.05
        #   > 3.0m  -> 0.05m radius (open space)
        if min_d < 0.3:
            radius = 0.25
        elif min_d < 1.0:
            radius = 0.25 - (min_d - 0.3) * (0.13 / 0.7)
        elif min_d < 3.0:
            radius = 0.12 - (min_d - 1.0) * (0.07 / 2.0)
        else:
            radius = 0.05
        return float(np.clip(radius, 0.05, 0.25))

    elif strategy == "class_aware":
        # Falls back to min_depth for scalar radius reporting.
        # Actual per-class inflation uses apply_class_aware_inflation().
        return compute_dynamic_radius(depth_m, "min_depth", min_depth, max_depth)

    else:
        raise ValueError(f"Unknown inflation strategy: {strategy}")


# Per-class inflation radii for class-aware strategy
CLASS_RADII: Dict[int, float] = {
    SEG_FLOOR:     0.00,   # not an obstacle
    SEG_WALL:      0.06,   # static, well-localised
    SEG_PERSON:    0.30,   # unpredictable movement, high safety margin
    SEG_FURNITURE: 0.15,   # static but may protrude
    SEG_GLASS:     0.20,   # invisible to LiDAR/ToF, needs safety buffer
    SEG_OTHER:     0.12,   # default
}


def apply_class_aware_inflation(
    grid: np.ndarray,
    class_grid: np.ndarray,
    resolution: float = 0.05,
    cost_scaling: float = 3.0,
) -> Tuple[np.ndarray, float]:
    """
    Apply per-class inflation: each obstacle cell is inflated according to
    its semantic class (person gets 0.30m, wall gets 0.06m, etc.).

    Runs one inflation pass per class present in the grid, then merges
    by taking the maximum cost at each cell.

    Args:
        grid: (H, W) uint8 occupancy grid (0=free, 254=lethal).
        class_grid: (H, W) uint8 class labels (0-5, 255=unknown).
        resolution: grid resolution in m/cell.
        cost_scaling: exponential decay factor.

    Returns:
        Tuple of (inflated_grid, effective_mean_radius_metres).
    """
    inflated = grid.copy()
    radius_sum = 0.0
    cell_count = 0

    all_classes = list(CLASS_RADII.keys()) + [SEG_UNKNOWN]
    for cls in all_classes:
        radius = CLASS_RADII.get(cls, CLASS_RADII[SEG_OTHER])
        if radius <= 0:
            continue

        cls_lethal = (grid == LETHAL_COST) & (class_grid == cls)
        n = int(np.sum(cls_lethal))
        if n == 0:
            continue

        temp_grid = np.where(cls_lethal, LETHAL_COST, FREE_COST).astype(np.uint8)
        radius_cells = radius / resolution
        temp_inflated = _exponential_decay_costs(temp_grid, radius_cells, cost_scaling)
        inflated = np.maximum(inflated, temp_inflated)

        radius_sum += radius * n
        cell_count += n

    effective_radius = radius_sum / cell_count if cell_count > 0 else 0.12
    return inflated, float(effective_radius)


def apply_dynamic_inflation(
    grid: np.ndarray,
    depth_m: np.ndarray,
    strategy: str = "corridor_width",
    resolution: float = 0.05,
    cost_scaling: float = 3.0,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> Tuple[np.ndarray, float]:
    """
    Apply inflation with a per-frame adaptive radius.

    Args:
        grid: (H, W) uint8 occupancy grid.
        depth_m: (H, W) float32 depth image used to compute the radius.
        strategy: inflation strategy name.
        resolution: grid resolution in m/cell.
        cost_scaling: exponential decay factor.
        min_depth: minimum valid depth.
        max_depth: maximum valid depth.

    Returns:
        Tuple of (inflated_grid, computed_radius_metres).
    """
    radius = compute_dynamic_radius(depth_m, strategy, min_depth, max_depth)
    radius_cells = radius / resolution
    inflated = _exponential_decay_costs(grid, radius_cells, cost_scaling)
    return inflated, radius
