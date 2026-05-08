#!/usr/bin/env python3
"""
Create publication-quality Fig. 2 for the paper:
  (a) Corridor RGB frame
  (b) Femto Bolt ToF depth (with dead pixels highlighted)
  (c) DA3-Small depth (dense)
"""

import cv2
import numpy as np
from pathlib import Path

GLASS_DIR = Path("/home/nishant/maps/glass_corridor_frames")
CORRIDOR_DIR = Path("/home/nishant/maps/corridor_key_frames")
OUTPUT = Path("/home/nishant/maps/paper_figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Use the frame with the most dead pixels from glass corridor
FRAME = "frame_0040_t49.9s"


def colorize_depth_publication(depth_m, valid_mask, colormap=cv2.COLORMAP_INFERNO):
    """Create publication-quality depth colorization."""
    h, w = depth_m.shape
    viz = np.zeros((h, w), dtype=np.uint8)

    if valid_mask.any():
        d = depth_m[valid_mask]
        dmin, dmax = d.min(), d.max()
        if dmax > dmin:
            viz[valid_mask] = (255 * (1.0 - (d - dmin) / (dmax - dmin))).clip(0, 255).astype(np.uint8)

    colored = cv2.applyColorMap(viz, colormap)
    # Dead pixels in dark gray to make them visually distinct
    colored[~valid_mask] = [40, 40, 40]
    return colored


def add_dead_pixel_overlay(depth_color, valid_mask, dead_pct):
    """Add subtle hatching/marking on dead pixel regions."""
    overlay = depth_color.copy()
    # Dead regions already dark gray; add a slight red tint to emphasize
    dead_mask = ~valid_mask
    overlay[dead_mask, 2] = np.clip(overlay[dead_mask, 2].astype(int) + 30, 0, 255).astype(np.uint8)
    return overlay


def create_fig2():
    rgb_path = GLASS_DIR / "rgb" / f"{FRAME}.png"
    depth_path = GLASS_DIR / "depth" / f"{FRAME}.npy"
    da3_path = GLASS_DIR / "da3_depth" / f"{FRAME}.npy"

    rgb = cv2.imread(str(rgb_path))
    sensor_depth = np.load(str(depth_path))
    da3_depth = np.load(str(da3_path))

    valid_sensor = (sensor_depth > 0.1) & (sensor_depth < 5.0)
    valid_da3 = (da3_depth > 0.05) & (da3_depth < 8.0)

    dead_pct = (~valid_sensor).sum() * 100.0 / sensor_depth.size
    valid_pct = valid_sensor.sum() * 100.0 / sensor_depth.size

    # Colorize sensor depth
    sensor_color = colorize_depth_publication(sensor_depth, valid_sensor)
    sensor_color = add_dead_pixel_overlay(sensor_color, valid_sensor, dead_pct)

    # Colorize DA3 depth (same colormap range for comparison)
    da3_color = colorize_depth_publication(da3_depth, valid_da3)

    # Resize all to same dimensions
    target_h = 480
    aspect = rgb.shape[1] / rgb.shape[0]
    target_w = int(target_h * aspect)

    rgb_r = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    sensor_r = cv2.resize(sensor_color, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    da3_r = cv2.resize(da3_color, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_bg_h = 36

    panels = []
    labels_text = [
        "(a) RGB",
        f"(b) Sensor depth ({dead_pct:.0f}% dead)",
        "(c) DA3-Small (dense)",
    ]
    for img, label in zip([rgb_r, sensor_r, da3_r], labels_text):
        panel = img.copy()
        # Semi-transparent label background
        cv2.rectangle(panel, (0, 0), (target_w, label_bg_h), (0, 0, 0), -1)
        alpha = 0.6
        panel[:label_bg_h] = cv2.addWeighted(
            panel[:label_bg_h], alpha, img[:label_bg_h], 1 - alpha, 0)
        cv2.putText(panel, label, (8, 26), font, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        panels.append(panel)

    # Create 3-panel horizontal composition
    gap = 4
    total_w = target_w * 3 + gap * 2
    fig = np.ones((target_h, total_w, 3), dtype=np.uint8) * 255
    fig[:, :target_w] = panels[0]
    fig[:, target_w + gap:2 * target_w + gap] = panels[1]
    fig[:, 2 * (target_w + gap):] = panels[2]

    out_path = OUTPUT / "fig2_depth_comparison.png"
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Saved: {out_path}")
    print(f"  Size: {total_w}x{target_h}")
    print(f"  Dead pixels: {dead_pct:.1f}%")
    print(f"  Valid pixels: {valid_pct:.1f}%")

    # Also save individual panels at high resolution for LaTeX
    for panel, name in zip(panels, ["fig2a_rgb", "fig2b_sensor", "fig2c_da3"]):
        p = OUTPUT / f"{name}.png"
        cv2.imwrite(str(p), panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"  Panel: {p}")

    # Also create a version from the regular corridor (bag 172822)
    # using the frame with highest dead pixels
    corridor_frame = "frame_007_t210s"
    c_rgb = cv2.imread(str(CORRIDOR_DIR / "rgb" / f"{corridor_frame}.png"))
    c_depth = np.load(str(CORRIDOR_DIR / "depth_sensor" / f"{corridor_frame}.npy"))

    if c_rgb is not None:
        c_valid = (c_depth > 0.1) & (c_depth < 5.0)
        c_dead = (~c_valid).sum() * 100.0 / c_depth.size
        c_sensor_color = colorize_depth_publication(c_depth, c_valid)
        c_sensor_color = add_dead_pixel_overlay(c_sensor_color, c_valid, c_dead)

        # Check if DA3 was run on corridor
        c_da3_path = CORRIDOR_DIR / "comparison" / f"{corridor_frame}.png"
        if c_da3_path.exists():
            # Use the existing comparison
            comp = cv2.imread(str(c_da3_path))
            cv2.imwrite(str(OUTPUT / "fig2_corridor_comparison.png"), comp,
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"\n  Corridor comparison: {OUTPUT / 'fig2_corridor_comparison.png'}")
            print(f"  Corridor dead pixels: {c_dead:.1f}%")


if __name__ == "__main__":
    create_fig2()
