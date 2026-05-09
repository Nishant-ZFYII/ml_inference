#!/usr/bin/env python3
"""Generate a comparison grid showing all 12 depth error profiles
applied to a single Gazebo GT depth frame.

Directly supports the navigation results in Table VIII by showing
exactly what depth input the robot received during each trial.

Usage:
    python generate_noise_profile_figure.py \
        --frame-dir ../NCHSB/gazebo_captured_frames \
        --frame-id frame_0005 \
        --output-dir /home/nishant/maps/paper_figures/noise_profiles \
        --vmax 5.5
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image


# ── Profiles (copied from depth_error_injector.py to avoid ROS deps) ──

@dataclass(frozen=True)
class ErrorProfile:
    name: str
    near_rmse: float
    mid_rmse: float
    far_rmse: float
    beyond_rmse: float
    dead_pixel_ratio: float
    description: str
    noise_scale: float = 0.3
    fusion: bool = False


NOISE_SCALE = 0.3

PROFILES = {
    0: ErrorProfile(
        name='GT', near_rmse=0.0, mid_rmse=0.0, far_rmse=0.0, beyond_rmse=0.0,
        dead_pixel_ratio=0.0,
        description='Ground truth (no error)',
    ),
    1: ErrorProfile(
        name='DA3-Small', near_rmse=0.158, mid_rmse=0.486, far_rmse=1.320,
        beyond_rmse=1.5, dead_pixel_ratio=0.0, noise_scale=NOISE_SCALE,
        description='DA3-Small teacher (RMSE 0.596 m)',
    ),
    2: ErrorProfile(
        name='V4', near_rmse=1.434, mid_rmse=1.182, far_rmse=1.283,
        beyond_rmse=1.4, dead_pixel_ratio=0.0, noise_scale=NOISE_SCALE,
        description='V4 student (RMSE 1.374 m)',
    ),
    3: ErrorProfile(
        name='V5', near_rmse=2.361, mid_rmse=2.0, far_rmse=1.469,
        beyond_rmse=1.5, dead_pixel_ratio=0.0, noise_scale=NOISE_SCALE,
        description='V5 student (RMSE 2.186 m)',
    ),
    4: ErrorProfile(
        name='V6', near_rmse=2.262, mid_rmse=2.0, far_rmse=1.5,
        beyond_rmse=1.6, dead_pixel_ratio=0.0, noise_scale=NOISE_SCALE,
        description='V6 student (RMSE 2.158 m)',
    ),
    5: ErrorProfile(
        name='V7', near_rmse=1.982, mid_rmse=1.453, far_rmse=0.732,
        beyond_rmse=0.9, dead_pixel_ratio=0.0, noise_scale=NOISE_SCALE,
        description='V7 specialist (RMSE 1.712 m)',
    ),
    6: ErrorProfile(
        name='V8', near_rmse=2.368, mid_rmse=2.1, far_rmse=1.6,
        beyond_rmse=1.7, dead_pixel_ratio=0.0, noise_scale=NOISE_SCALE,
        description='V8 mixed (RMSE 2.266 m)',
    ),
    7: ErrorProfile(
        name='V9', near_rmse=1.782, mid_rmse=1.3, far_rmse=1.0,
        beyond_rmse=1.1, dead_pixel_ratio=0.0, noise_scale=NOISE_SCALE,
        description='V9 best student (RMSE 1.589 m)',
    ),
    8: ErrorProfile(
        name='Sensor-fail', near_rmse=0.0, mid_rmse=0.0, far_rmse=0.0,
        beyond_rmse=0.0, dead_pixel_ratio=0.77,
        description='Sensor failure (77% dead)',
    ),
    9: ErrorProfile(
        name='Sensor+DA3', near_rmse=0.158, mid_rmse=0.486, far_rmse=1.320,
        beyond_rmse=1.5, dead_pixel_ratio=0.77, noise_scale=NOISE_SCALE,
        fusion=True,
        description='Sensor + DA3 fusion',
    ),
    10: ErrorProfile(
        name='Sensor+V9', near_rmse=1.782, mid_rmse=1.3, far_rmse=1.0,
        beyond_rmse=1.1, dead_pixel_ratio=0.77, noise_scale=NOISE_SCALE,
        fusion=True,
        description='Sensor + V9 fusion',
    ),
    11: ErrorProfile(
        name='Sensor+V4', near_rmse=1.434, mid_rmse=1.182, far_rmse=1.283,
        beyond_rmse=1.4, dead_pixel_ratio=0.77, noise_scale=NOISE_SCALE,
        fusion=True,
        description='Sensor + V4 fusion',
    ),
}

DEPTH_BIN_EDGES = (0.3, 1.0, 2.0, 4.0)


def inject_error(depth: np.ndarray, profile: ErrorProfile,
                 rng: np.random.Generator,
                 min_depth=0.1, max_depth=10.0) -> np.ndarray:
    """Apply a noise profile to a GT depth frame. Pure numpy, no ROS."""
    if profile.name == 'GT':
        return depth.copy()

    result = depth.copy()
    p = profile

    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    has_noise = (p.near_rmse > 0 or p.mid_rmse > 0
                 or p.far_rmse > 0 or p.beyond_rmse > 0)

    noise_img = None
    if has_noise:
        noise_std = np.zeros_like(depth)
        d_near, d_mid, d_far, d_beyond = DEPTH_BIN_EDGES

        very_near = valid & (depth < d_near)
        near  = valid & (depth >= d_near)  & (depth < d_mid)
        mid   = valid & (depth >= d_mid)   & (depth < d_far)
        far   = valid & (depth >= d_far)   & (depth < d_beyond)
        beynd = valid & (depth >= d_beyond)

        noise_std[very_near] = p.near_rmse * 0.5
        noise_std[near]  = p.near_rmse
        noise_std[mid]   = p.mid_rmse
        noise_std[far]   = p.far_rmse
        noise_std[beynd] = p.beyond_rmse

        noise_std *= p.noise_scale

        h, w = depth.shape
        lo_h, lo_w = max(1, h // 8), max(1, w // 8)
        lo_noise = rng.standard_normal((lo_h, lo_w)).astype(np.float32)
        noise = cv2.resize(lo_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        ns = noise.std()
        if ns > 1e-6:
            noise /= ns
        noise_img = noise * noise_std

    if p.fusion:
        dead = _make_dead_mask(depth.shape[0], depth.shape[1],
                               p.dead_pixel_ratio, rng)
        if noise_img is not None:
            fill_depth = np.clip(depth + noise_img, min_depth, max_depth)
            result[dead & valid] = fill_depth[dead & valid]
        originally_invalid = ~np.isfinite(depth)
        result[dead & originally_invalid] = np.inf
    else:
        if valid.any() and noise_img is not None:
            result[valid] = depth[valid] + noise_img[valid]
            result = np.clip(result, min_depth, max_depth)
        if p.dead_pixel_ratio > 0:
            dead = _make_dead_mask(depth.shape[0], depth.shape[1],
                                   p.dead_pixel_ratio, rng)
            result[dead] = np.inf
        originally_invalid = ~np.isfinite(depth)
        result[originally_invalid] = np.inf

    return result


def _make_dead_mask(h, w, ratio, rng):
    bh = (h + 7) // 8
    bw = (w + 7) // 8
    block_dead = rng.random((bh, bw)) < ratio
    mask = np.repeat(np.repeat(block_dead, 8, axis=0), 8, axis=1)
    return mask[:h, :w]


# ── Grid labels: profile ID → display label ──
GRID_LABELS = {
    0:  'P0: GT Depth\n(no error)',
    1:  'P1: DA3-Small\nRMSE 0.596 m',
    2:  'P2: V4\nRMSE 1.374 m',
    3:  'P3: V5\nRMSE 2.186 m',
    4:  'P4: V6\nRMSE 2.158 m',
    5:  'P5: V7\nRMSE 1.712 m',
    6:  'P6: V8\nRMSE 2.266 m',
    7:  'P7: V9\nRMSE 1.589 m',
    8:  'P8: Sensor-fail\n77% dead pixels',
    9:  'P9: Sensor+DA3\n23% GT + 77% DA3',
    10: 'P10: Sensor+V9\n23% GT + 77% V9',
    11: 'P11: Sensor+V4\n23% GT + 77% V4',
}


def create_noise_comparison_grid(rgb, gt_depth, injected_depths,
                                 output_path, vmax=5.5,
                                 frame_id='frame_0005'):
    """Create a 4x4 grid: RGB + 12 noise profiles + colorbar."""
    ncols = 4
    nrows = 4
    fig = plt.figure(figsize=(ncols * 4.5, nrows * 3.2), dpi=150)
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.22, wspace=0.08)

    panels = [('RGB Input', rgb, 'rgb')]

    profile_order = list(range(12))
    for pid in profile_order:
        depth = injected_depths[pid]
        panels.append((GRID_LABELS[pid], depth, 'depth'))

    for i, (title, data, dtype) in enumerate(panels):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        if dtype == 'rgb':
            ax.imshow(data)
        else:
            vis = data.copy()
            vis[~np.isfinite(vis)] = np.nan
            vis[vis == 0] = np.nan
            ax.imshow(vis, cmap='inferno', vmin=0.0, vmax=vmax,
                      interpolation='bilinear')

        ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
        ax.axis('off')

    # Empty cells (we have 13 panels in a 4x4 = 16 grid)
    for i in range(len(panels), nrows * ncols):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap='inferno', norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Depth (metres)', fontsize=10)

    fig.suptitle(
        f'{frame_id} — Depth Error Profiles for Navigation Experiment',
        fontsize=13, fontweight='bold', y=0.98)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate noise profile comparison figure')
    parser.add_argument('--frame-dir', type=str,
                        default='../NCHSB/gazebo_captured_frames',
                        help='Directory with rgb/ and depth/ subdirs')
    parser.add_argument('--frame-id', type=str, default='frame_0005',
                        help='Frame ID to use (e.g. frame_0005)')
    parser.add_argument('--output-dir', type=str,
                        default='/home/nishant/maps/paper_figures/noise_profiles',
                        help='Output directory')
    parser.add_argument('--vmax', type=float, default=5.5,
                        help='Max depth for colormap (metres)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-frames', type=int, default=None,
                        help='Generate grids for multiple frames (e.g. 5)')
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_frames:
        frame_ids = [f'frame_{i:04d}' for i in range(args.num_frames)]
    else:
        frame_ids = [args.frame_id]

    for frame_id in frame_ids:
        rgb_path = frame_dir / 'rgb' / f'{frame_id}.png'
        depth_path = frame_dir / 'depth' / f'{frame_id}.npy'

        if not rgb_path.exists() or not depth_path.exists():
            print(f'Skipping {frame_id}: missing files')
            continue

        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        gt_depth = np.load(depth_path).astype(np.float32)
        gt_depth[~np.isfinite(gt_depth)] = 0.0

        print(f'\n=== {frame_id} ===')
        print(f'  RGB: {rgb.shape}, Depth: {gt_depth.shape}, '
              f'range [{gt_depth[gt_depth > 0].min():.2f}, {gt_depth.max():.2f}]m')

        injected = {}
        for pid, profile in PROFILES.items():
            rng = np.random.default_rng(args.seed)
            noisy = inject_error(gt_depth, profile, rng)
            injected[pid] = noisy

            valid = np.isfinite(noisy) & (noisy > 0.05)
            dead_pct = 100 * (1 - valid.sum() / noisy.size)
            if valid.any():
                diff = np.abs(noisy[valid] - gt_depth[valid]).mean()
            else:
                diff = float('nan')
            print(f'  P{pid:2d} ({profile.name:12s}): '
                  f'dead={dead_pct:5.1f}%, mean_err={diff:.3f}m')

        grid_path = output_dir / f'{frame_id}_noise_profiles.png'
        create_noise_comparison_grid(
            rgb, gt_depth, injected, grid_path,
            vmax=args.vmax, frame_id=frame_id)

    print(f'\nDone. Output: {output_dir}')


if __name__ == '__main__':
    main()
