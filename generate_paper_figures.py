#!/usr/bin/env python3
"""
Generate paper-quality depth comparison figures from a corridor rosbag.

Extracts a smart subset of frames, runs all student models + DA3-Small,
and saves side-by-side comparison grids suitable for publication.

Output layout:
    <output_dir>/
        individual/       per-model depth maps for each frame
        comparisons/      6-panel grids (RGB, Sensor, DA3, V4, V5, V7)
        frames/           extracted RGB + sensor depth PNGs

Usage:
    python generate_paper_figures.py \
        --bag /home/nishant/rosbags/rgbd_imu_20260302_213831 \
        --output-dir /home/nishant/maps/paper_figures/model_comparison \
        --num-frames 20
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from models.student import build_student


MODELS = {
    "V4": {
        "checkpoint": "hpc_outputs/vivek_v4_best_depth.pt",
        "imagenet_norm": True,
        "label": "V4 (Best Corridor)\nRMSE 1.373m",
    },
    "V5": {
        "checkpoint": "hpc_outputs/best_depth_v5_vivek.pt",
        "imagenet_norm": True,
        "label": "V5 (Best NYU)\nRMSE 0.572m NYU",
    },
    "V7": {
        "checkpoint": "hpc_outputs/best_depth_v7.pt",
        "imagenet_norm": True,
        "label": "V7 (LILocBench FT)\nRMSE 0.445m LILoc",
    },
    "Iter6_B1": {
        "checkpoint": "hpc_outputs/best_depth.pt",
        "imagenet_norm": False,
        "label": "Iter6 B1 (Baseline)\nRMSE 0.998m NYU",
    },
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def extract_frames_from_db3(bag_path: str, num_frames: int):
    """Extract evenly-spaced RGB + depth frames from a db3 rosbag via sqlite3."""
    import sqlite3
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    bag_dir = Path(bag_path)

    db3_files = list(bag_dir.glob("*.db3"))
    if not db3_files:
        raise RuntimeError(f"No .db3 file found in {bag_dir}")
    db3_path = str(db3_files[0])

    print(f"Reading rosbag: {db3_path}")
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()

    # Find topic IDs
    cursor.execute("SELECT id, name FROM topics")
    topic_map = {name: tid for tid, name in cursor.fetchall()}
    rgb_tid = topic_map.get("/camera/color/image_raw")
    depth_tid = topic_map.get("/camera/depth/image_raw")
    if rgb_tid is None or depth_tid is None:
        raise RuntimeError(f"Topics not found. Available: {list(topic_map.keys())}")

    # Get all RGB timestamps (fast: just integers, no blob data)
    print("Fetching RGB timestamps...")
    cursor.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
        (rgb_tid,),
    )
    all_timestamps = [row[0] for row in cursor.fetchall()]
    total = len(all_timestamps)
    stride = max(1, total // num_frames)
    print(f"Total RGB frames: {total}, selecting {num_frames} (stride={stride})")

    # Pick evenly spaced timestamps
    selected_ts = [all_timestamps[i * stride]
                   for i in range(num_frames) if i * stride < total]

    # Fetch data for selected timestamps only (fast: indexed lookup)
    selected_rgb = []
    for ts in selected_ts:
        cursor.execute(
            "SELECT timestamp, data FROM messages "
            "WHERE topic_id=? AND timestamp=?",
            (rgb_tid, ts),
        )
        row = cursor.fetchone()
        if row:
            selected_rgb.append((row[0], row[1]))

    print(f"Selected {len(selected_rgb)} RGB frames")

    frames = []
    for idx, (rgb_ts, rgb_raw) in enumerate(selected_rgb):
        msg = typestore.deserialize_cdr(rgb_raw, "sensor_msgs/msg/Image")
        h, w = msg.height, msg.width
        encoding = msg.encoding
        if encoding in ("rgb8", "8UC3"):
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
        elif encoding == "bgr8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            rgb = rgb[:, :, ::-1].copy()
        else:
            print(f"  Frame {idx}: unknown RGB encoding {encoding}, skipping")
            continue

        # Find nearest depth frame (within 50ms)
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? "
            "AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp - ?) LIMIT 1",
            (depth_tid, rgb_ts - 50_000_000, rgb_ts + 50_000_000, rgb_ts),
        )
        depth_row = cursor.fetchone()
        depth = None
        if depth_row:
            dmsg = typestore.deserialize_cdr(depth_row[1], "sensor_msgs/msg/Image")
            dh, dw = dmsg.height, dmsg.width
            denc = dmsg.encoding
            if denc == "16UC1":
                depth = np.frombuffer(dmsg.data, dtype=np.uint16).reshape(dh, dw)
                depth = depth.astype(np.float32) / 1000.0
            elif denc == "32FC1":
                depth = np.frombuffer(dmsg.data, dtype=np.float32).reshape(dh, dw)

        frames.append({"timestamp": rgb_ts, "rgb": rgb, "depth": depth})
        print(f"  Frame {idx}: ts={rgb_ts}, rgb={w}x{h}, "
              f"depth={'yes' if depth is not None else 'no'}")

    conn.close()
    print(f"Extracted {len(frames)} frames")
    return frames


def load_student_model(checkpoint_path: str, cfg, device):
    """Load a student model checkpoint."""
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False,
                          backbone=cfg.BACKBONE)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def run_student_inference(model, rgb_pil, imagenet_norm, device, H=240, W=320):
    """Run student model on a single RGB PIL image."""
    rgb_resized = rgb_pil.resize((W, H), Image.BILINEAR)
    rgb_np = np.array(rgb_resized, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    if imagenet_norm:
        rgb_t = (rgb_t - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

    with torch.no_grad():
        pred_depth, _ = model(rgb_t)
    return pred_depth.squeeze().cpu().numpy()


def load_da3_model(device):
    """Load DA3-Small for relative depth estimation."""
    from transformers import pipeline
    print("Loading DA3-Small model (this may download weights)...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device if str(device) != "cpu" else -1,
    )
    print("DA3-Small loaded.")
    return pipe


def run_da3_inference(pipe, rgb_pil):
    """Run DA3-Small on a PIL image, return relative depth as numpy."""
    result = pipe(rgb_pil)
    depth = np.array(result["depth"], dtype=np.float32)
    return depth


def apply_colormap(depth, vmin=0.0, vmax=5.0, cmap="inferno"):
    """Apply a matplotlib colormap to a depth map, return RGB uint8 array."""
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.get_cmap(cmap)
    colored = mapper(norm(depth))[:, :, :3]
    return (colored * 255).astype(np.uint8)


def create_comparison_grid(rgb, sensor_depth, predictions, frame_idx, output_path,
                           vmin=0.0, vmax=5.0):
    """
    Create a publication-quality comparison grid.

    Layout: 2 rows x 3 columns
        Row 1: RGB | Sensor Depth | DA3-Small
        Row 2: V4  | V5           | V7

    Each depth panel is normalised to a shared range computed from
    all valid depth values across sensor + predictions, so that DA3
    and student models are visually comparable.
    """
    n_models = len(predictions)
    n_panels = 2 + n_models
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    # Compute a shared depth range from all panels for this frame
    all_valid = []
    if sensor_depth is not None:
        sv = sensor_depth[(sensor_depth > 0.05) & (sensor_depth < 10)]
        if len(sv) > 0:
            all_valid.append(sv)
    for pred in predictions.values():
        pv = pred[(pred > 0.05) & (pred < 10) & ~np.isnan(pred)]
        if len(pv) > 0:
            all_valid.append(pv)

    if all_valid:
        combined = np.concatenate(all_valid)
        shared_vmin = max(0.0, float(np.percentile(combined, 1)))
        shared_vmax = float(np.percentile(combined, 99))
        shared_vmax = max(shared_vmax, shared_vmin + 0.5)
    else:
        shared_vmin, shared_vmax = vmin, vmax

    fig = plt.figure(figsize=(ncols * 5, nrows * 3.5), dpi=150)
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.15, wspace=0.05)

    panels = []
    panels.append(("RGB Input", rgb, "rgb"))

    if sensor_depth is not None:
        sensor_vis = sensor_depth.copy()
        sensor_vis[sensor_depth == 0] = np.nan
        panels.append(("Sensor Depth (Femto Bolt)", sensor_vis, "depth"))
    else:
        panels.append(("Sensor Depth (N/A)", np.zeros((240, 320)), "depth"))

    for name, pred in predictions.items():
        label = MODELS.get(name, {}).get("label", name)
        if name == "DA3-Small":
            label = "DA3-Small (Teacher)\nRMSE 0.596m corridor"
        panels.append((label, pred, "depth"))

    for i, (title, data, dtype) in enumerate(panels):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        if dtype == "rgb":
            ax.imshow(data)
        else:
            im = ax.imshow(data, cmap="inferno",
                           vmin=shared_vmin, vmax=shared_vmax,
                           interpolation="bilinear")

        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap="inferno", norm=Normalize(vmin=shared_vmin, vmax=shared_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Depth (metres)", fontsize=10)

    fig.suptitle(f"Frame {frame_idx:04d} — Corridor Depth Comparison",
                 fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_individual_depth(depth, name, frame_idx, output_dir, vmin=0.0, vmax=5.0):
    """Save a single depth map as a colormapped PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=150)
    im = ax.imshow(depth, cmap="inferno", vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    ax.axis("off")
    ax.set_title(name, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Depth (m)", fraction=0.046, pad=0.04)
    out_path = output_dir / f"frame_{frame_idx:04d}_{name.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-quality depth comparison figures")
    parser.add_argument("--bag", type=str, required=True,
                        help="Path to rosbag directory")
    parser.add_argument("--output-dir", type=str,
                        default="/home/nishant/maps/paper_figures/model_comparison",
                        help="Output directory for figures")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Number of frames to extract")
    parser.add_argument("--vmax", type=float, default=5.0,
                        help="Max depth for colormap (metres)")
    parser.add_argument("--skip-da3", action="store_true",
                        help="Skip DA3-Small inference (slow on CPU)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    (output_dir / "individual").mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(parents=True, exist_ok=True)

    H, W = 240, 320
    cfg = Config()

    # --- Extract frames ---
    frames = extract_frames_from_db3(args.bag, args.num_frames)

    # Save extracted frames
    for i, f in enumerate(frames):
        rgb_pil = Image.fromarray(f["rgb"])
        rgb_resized = rgb_pil.resize((W, H), Image.BILINEAR)
        rgb_resized.save(output_dir / "frames" / f"frame_{i:04d}_rgb.png")

        if f["depth"] is not None:
            depth_pil = Image.fromarray(f["depth"])
            depth_resized = np.array(
                depth_pil.resize((W, H), Image.NEAREST), dtype=np.float32
            )
            np.save(output_dir / "frames" / f"frame_{i:04d}_depth.npy", depth_resized)
    print(f"Saved {len(frames)} extracted frames to {output_dir / 'frames'}")

    # --- Load student models ---
    print("\n=== Loading Student Models ===")
    loaded_models = {}
    for name, info in MODELS.items():
        ckpt_path = Path(__file__).parent / info["checkpoint"]
        if not ckpt_path.exists():
            print(f"  [{name}] Checkpoint not found: {ckpt_path}, skipping")
            continue
        print(f"  Loading {name} from {ckpt_path.name}...")
        model = load_student_model(str(ckpt_path), cfg, device)
        loaded_models[name] = (model, info["imagenet_norm"])
        print(f"  [{name}] Loaded OK")

    # --- Load DA3-Small ---
    da3_pipe = None
    if not args.skip_da3:
        try:
            da3_pipe = load_da3_model(device)
        except Exception as e:
            print(f"  DA3 loading failed: {e}. Skipping DA3.")

    # --- Run inference on each frame ---
    print(f"\n=== Running Inference on {len(frames)} Frames ===")
    for i, f in enumerate(frames):
        print(f"\n--- Frame {i:04d} / {len(frames)-1} ---")
        rgb_pil = Image.fromarray(f["rgb"]).resize((W, H), Image.BILINEAR)

        # Sensor depth (resized to model resolution)
        sensor_depth = None
        if f["depth"] is not None:
            depth_pil = Image.fromarray(f["depth"])
            sensor_depth = np.array(
                depth_pil.resize((W, H), Image.NEAREST), dtype=np.float32
            )

        predictions = {}

        # Student models
        for name, (model, imagenet_norm) in loaded_models.items():
            pred = run_student_inference(model, rgb_pil, imagenet_norm, device, H, W)
            predictions[name] = pred
            save_individual_depth(pred, name, i, output_dir / "individual",
                                 vmax=args.vmax)
            print(f"  {name}: range [{pred.min():.2f}, {pred.max():.2f}]m, "
                  f"mean {pred.mean():.2f}m")

        # DA3-Small
        if da3_pipe is not None:
            print(f"  Running DA3-Small...")
            da3_relative = run_da3_inference(da3_pipe, rgb_pil)
            da3_resized = np.array(
                Image.fromarray(da3_relative).resize((W, H), Image.BILINEAR),
                dtype=np.float32,
            )
            # Median-scale against sensor depth if available
            if sensor_depth is not None:
                valid = (sensor_depth >= 0.1) & (sensor_depth <= 5.0)
                if valid.sum() > 100 and np.median(da3_resized[valid]) > 1e-8:
                    scale = np.median(sensor_depth[valid]) / np.median(da3_resized[valid])
                    da3_metric = da3_resized * scale
                else:
                    da3_metric = da3_resized
            else:
                da3_metric = da3_resized

            predictions["DA3-Small"] = da3_metric
            save_individual_depth(da3_metric, "DA3-Small", i,
                                 output_dir / "individual", vmax=args.vmax)
            print(f"  DA3-Small: range [{da3_metric.min():.2f}, "
                  f"{da3_metric.max():.2f}]m, mean {da3_metric.mean():.2f}m")

        # Save sensor depth image
        if sensor_depth is not None:
            save_individual_depth(sensor_depth, "Sensor_Depth", i,
                                 output_dir / "individual", vmax=args.vmax)

        # Reorder predictions for the grid: DA3 first, then students
        ordered = {}
        if "DA3-Small" in predictions:
            ordered["DA3-Small"] = predictions["DA3-Small"]
        for name in ["V4", "V5", "V7"]:
            if name in predictions:
                ordered[name] = predictions[name]
        if "Iter6_B1" in predictions:
            ordered["Iter6_B1"] = predictions["Iter6_B1"]

        # Comparison grid
        rgb_display = np.array(rgb_pil)
        grid_path = output_dir / "comparisons" / f"frame_{i:04d}_comparison.png"
        create_comparison_grid(
            rgb_display, sensor_depth, ordered, i, grid_path, vmax=args.vmax
        )
        print(f"  Saved comparison grid: {grid_path.name}")

    print(f"\n=== Done ===")
    print(f"Output directory: {output_dir}")
    print(f"  comparisons/  - {len(frames)} side-by-side grids")
    print(f"  individual/   - per-model depth maps")
    print(f"  frames/       - extracted RGB + sensor depth")


if __name__ == "__main__":
    main()
