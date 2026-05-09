#!/usr/bin/env python3
"""
Generate paper-quality depth comparison figures from a corridor rosbag.

Extracts a smart subset of frames, runs all student models + DA3-Small,
and saves side-by-side comparison grids suitable for publication.

Output layout:
    <output_dir>/
        individual/       per-model depth maps for each frame
        comparisons/      4-col grids (RGB, Sensor, DA3, V4..V9, Fused(S+DA3), Fused(S+V9))
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


# All student models use internal ImageNet normalization (no external --imagenet-norm)
MODELS = {
    "V4": {
        "checkpoint": "hpc_outputs/vivek_v4_best_depth.pt",
        "imagenet_norm": False,
        "label": "V4 (Best Corridor)\nRMSE 1.373m",
    },
    "V5": {
        "checkpoint": "hpc_outputs/best_depth_v5_vivek.pt",
        "imagenet_norm": False,
        "label": "V5 (Best NYU)\nRMSE 0.572m NYU",
    },
    "V6": {
        "checkpoint": "hpc_outputs/best_depth_v6.pt",
        "imagenet_norm": False,
        "label": "V6 (SUN+DIODE→NYU)\nRMSE 2.158m corridor",
    },
    "V6.2": {
        "checkpoint": "hpc_outputs/best_depth_v6.2.pt",
        "imagenet_norm": False,
        "label": "V6.2 (V6 variant)\nRMSE 2.102m corridor",
    },
    "V7": {
        "checkpoint": "hpc_outputs/best_depth_v7.pt",
        "imagenet_norm": False,
        "label": "V7 (LILocBench FT)\nRMSE 0.445m LILoc",
    },
    "V8": {
        "checkpoint": "hpc_outputs/best_depth_v8.pt",
        "imagenet_norm": False,
        "label": "V8 (V5+mixed)\nRMSE 2.266m corridor",
    },
    "V9": {
        "checkpoint": "hpc_outputs/best_depth_v9.pt",
        "imagenet_norm": False,
        "label": "V9 (best corridor)\nRMSE 1.589m corridor",
    },
    "Iter6_B1": {
        "checkpoint": "hpc_outputs/best_depth.pt",
        "imagenet_norm": False,
        "label": "Iter6 B1 (Baseline)\nRMSE 0.998m NYU",
    },
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def extract_frames_from_db3(bag_path: str, num_frames: int, skip_seconds: float = 0.0):
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
    depth_tid = (topic_map.get("/camera/depth/image_raw")
                 or topic_map.get("/camera/depth"))
    if rgb_tid is None or depth_tid is None:
        raise RuntimeError(f"Topics not found. Available: {list(topic_map.keys())}")

    # Get all RGB timestamps (fast: just integers, no blob data)
    print("Fetching RGB timestamps...")
    cursor.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
        (rgb_tid,),
    )
    all_timestamps = [row[0] for row in cursor.fetchall()]

    if skip_seconds > 0 and len(all_timestamps) > 0:
        t0 = all_timestamps[0]
        cutoff = t0 + int(skip_seconds * 1e9)
        before = len(all_timestamps)
        all_timestamps = [t for t in all_timestamps if t >= cutoff]
        print(f"Skipped first {skip_seconds:.0f}s: {before} -> {len(all_timestamps)} frames")

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
                depth = np.frombuffer(dmsg.data, dtype=np.float32).reshape(dh, dw).copy()
                depth[~np.isfinite(depth)] = 0.0

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
    """Run DA3-Small on a PIL image, return float32 depth as numpy."""
    result = pipe(rgb_pil)
    # Use predicted_depth (float32 tensor) not depth (uint8 PIL image)
    raw = result["predicted_depth"]
    if hasattr(raw, "numpy"):
        depth = raw.squeeze().numpy().astype(np.float32)
    else:
        depth = np.array(raw, dtype=np.float32)
    return depth


def apply_colormap(depth, vmin=0.0, vmax=5.0, cmap="inferno"):
    """Apply a matplotlib colormap to a depth map, return RGB uint8 array."""
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.get_cmap(cmap)
    colored = mapper(norm(depth))[:, :, :3]
    return (colored * 255).astype(np.uint8)


# Labels for fused panels (not in MODELS)
FUSED_LABELS = {
    "Fused(Sensor+DA3)": "Fused (Sensor + DA3)\nsensor where valid, DA3 fills dead",
    "Fused(Sensor+V9)": "Fused (Sensor + V9)\nsensor where valid, V9 fills dead",
}


def create_comparison_grid(rgb, sensor_depth, predictions, frame_idx, output_path,
                           vmin=0.0, vmax=5.0,
                           sensor_label="Sensor Depth (Femto Bolt)",
                           title_prefix="Corridor Depth Comparison"):
    """
    Create a publication-quality comparison grid.

    Layout: 4 columns, multiple rows
        Row 1: RGB | Sensor Depth | DA3-Small | V4
        Row 2: V5 | V6 | V6.2 | V7
        Row 3: V8 | V9 | Fused(Sensor+DA3) | Fused(Sensor+V9)

    Each depth panel is normalised to a shared range computed from
    all valid depth values across sensor + predictions, so that DA3
    and student models are visually comparable.
    """
    n_models = len(predictions)
    n_panels = 2 + n_models
    ncols = 4
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

    fig = plt.figure(figsize=(ncols * 4.5, nrows * 3.2), dpi=150)
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.15, wspace=0.05)

    panels = []
    panels.append(("RGB Input", rgb, "rgb", None))

    if sensor_depth is not None:
        sensor_vis = sensor_depth.copy()
        sensor_vis[sensor_depth == 0] = np.nan
        panels.append((sensor_label, sensor_vis, "depth", None))
    else:
        panels.append(("Sensor Depth (N/A)", np.zeros((240, 320)), "depth", None))

    for name, pred in predictions.items():
        label = MODELS.get(name, {}).get("label", FUSED_LABELS.get(name, name))
        if name == "DA3-Small":
            label = "DA3-Small (Teacher)\nRMSE 0.596m corridor"
        panels.append((label, pred, "depth", name))

    for i, (title, data, dtype, panel_key) in enumerate(panels):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        if dtype == "rgb":
            ax.imshow(data)
        else:
            # DA3 outputs narrow range (0–1.5m); shared colormap (0–5m) crushes it
            # into a dark band. Use DA3-specific range so its variation is visible.
            if panel_key == "DA3-Small":
                dv = data[(data > 0.05) & (data < 10) & ~np.isnan(data)]
                if len(dv) > 100:
                    da3_vmin = max(0.0, float(np.percentile(dv, 1)))
                    da3_vmax = max(2.0, float(np.percentile(dv, 99)))
                else:
                    da3_vmin, da3_vmax = 0.0, 2.0
                im = ax.imshow(data, cmap="inferno",
                               vmin=da3_vmin, vmax=da3_vmax,
                               interpolation="bilinear")
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

    fig.suptitle(f"Frame {frame_idx:04d} — {title_prefix}",
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


def extract_frames_from_corridor_eval(manifest_path: Path, num_frames: int):
    """Load frames from corridor_eval_data (manifest.jsonl + rgb/depth/da3_depth)."""
    with open(manifest_path) as f:
        entries = [json.loads(line) for line in f]
    n = min(num_frames, len(entries))
    stride = max(1, len(entries) // n)
    selected = [entries[i * stride] for i in range(n) if i * stride < len(entries)]

    base = manifest_path.parent
    frames = []
    for idx, e in enumerate(selected):
        rgb_path = base / e["rgb"]
        depth_path = base / e["sensor_depth"]
        frame_id = Path(e["rgb"]).stem
        if not rgb_path.exists() or not depth_path.exists():
            print(f"  Frame {idx}: missing {rgb_path.name} or {depth_path.name}, skipping")
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path).astype(np.float32)
        frames.append({"frame_id": frame_id, "rgb": rgb, "depth": depth})
    print(f"Loaded {len(frames)} frames from corridor_eval_data")
    return frames, base


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-quality depth comparison figures")
    parser.add_argument("--bag", type=str, default=None,
                        help="Path to rosbag directory (required unless --corridor-eval/--manifest)")
    parser.add_argument("--corridor-eval", action="store_true",
                        help="Use corridor_eval_data instead of rosbag (precomputed DA3)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest.jsonl (e.g. from gazebo_frame_capture.py)")
    parser.add_argument("--gazebo", action="store_true",
                        help="Gazebo mode: relabel 'Sensor Depth' as 'GT Depth (Gazebo)', "
                             "skip fusion panels (GT has no dead pixels)")
    parser.add_argument("--output-dir", type=str,
                        default="/home/nishant/maps/paper_figures/model_comparison",
                        help="Output directory for figures")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Number of frames to extract")
    parser.add_argument("--vmax", type=float, default=5.0,
                        help="Max depth for colormap (metres)")
    parser.add_argument("--skip-da3", action="store_true",
                        help="Skip DA3-Small inference (slow on CPU)")
    parser.add_argument("--skip-seconds", type=float, default=0.0,
                        help="Skip this many seconds from the start of the bag "
                             "(useful when robot is static at the beginning)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.corridor_eval and not args.bag and not args.manifest:
        parser.error("Either --bag, --corridor-eval, or --manifest is required")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    (output_dir / "individual").mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(parents=True, exist_ok=True)

    H, W = 240, 320
    cfg = Config()
    corridor_base = None

    # --- Extract frames ---
    if args.manifest:
        manifest = Path(args.manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        frames, corridor_base = extract_frames_from_corridor_eval(manifest, args.num_frames)
    elif args.corridor_eval:
        manifest = Path(__file__).parent / "corridor_eval_data" / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Corridor manifest not found: {manifest}")
        frames, corridor_base = extract_frames_from_corridor_eval(manifest, args.num_frames)
    else:
        frames = extract_frames_from_db3(args.bag, args.num_frames,
                                         skip_seconds=args.skip_seconds)
        for i, f in enumerate(frames):
            f["frame_id"] = f"{i:04d}"

    # Save extracted frames
    for i, f in enumerate(frames):
        rgb_pil = Image.fromarray(f["rgb"])
        rgb_resized = rgb_pil.resize((W, H), Image.BILINEAR)
        rgb_resized.save(output_dir / "frames" / f"frame_{i:04d}_rgb.png")

        if f.get("depth") is not None:
            d = f["depth"]
            if d.dtype == np.uint16:
                d = d.astype(np.float32) / 1000.0
            depth_resized = np.array(
                Image.fromarray(d.astype(np.float32)).resize((W, H), Image.NEAREST),
                dtype=np.float32,
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

    # --- Load DA3-Small (only needed when not using corridor-eval precomputed) ---
    da3_pipe = None
    if not args.skip_da3 and not args.corridor_eval and not args.manifest:
        try:
            da3_pipe = load_da3_model(device)
        except Exception as e:
            print(f"  DA3 loading failed: {e}. Skipping DA3.")

    # --- Gazebo mode config ---
    sensor_label = "GT Depth (Gazebo)" if args.gazebo else "Sensor Depth (Femto Bolt)"
    title_prefix = ("Gazebo Simulated Corridor — Depth Comparison"
                    if args.gazebo else "Corridor Depth Comparison")
    skip_fusion = args.gazebo

    # --- Run inference on each frame ---
    print(f"\n=== Running Inference on {len(frames)} Frames ===")
    for i, f in enumerate(frames):
        print(f"\n--- Frame {i:04d} / {len(frames)-1} ---")
        rgb_pil = Image.fromarray(f["rgb"]).resize((W, H), Image.BILINEAR)

        # Sensor depth (resized to model resolution)
        sensor_depth = None
        if f.get("depth") is not None:
            d = f["depth"]
            if d.dtype == np.uint16:
                d = d.astype(np.float32) / 1000.0
            sensor_depth = np.array(
                Image.fromarray(d.astype(np.float32)).resize((W, H), Image.NEAREST),
                dtype=np.float32,
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

        # DA3-Small: precomputed (corridor-eval) or live inference
        if corridor_base is not None:
            frame_id = f.get("frame_id", f"{i:04d}")
            da3_path = corridor_base / "da3_depth" / f"{frame_id}.npy"
            if da3_path.exists():
                da3_full = np.load(da3_path).astype(np.float32)
                da3_resized = np.array(
                    Image.fromarray(da3_full).resize((W, H), Image.BILINEAR),
                    dtype=np.float32,
                )
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
                                     output_dir / "individual", vmax=2.0)
                print(f"  DA3-Small (precomputed): range [{da3_metric.min():.2f}, "
                      f"{da3_metric.max():.2f}]m")
        elif da3_pipe is not None:
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
                                 output_dir / "individual", vmax=2.0)
            print(f"  DA3-Small: range [{da3_metric.min():.2f}, "
                  f"{da3_metric.max():.2f}]m, mean {da3_metric.mean():.2f}m")

        # Save sensor depth image
        if sensor_depth is not None:
            save_individual_depth(sensor_depth, "Sensor_Depth", i,
                                 output_dir / "individual", vmax=args.vmax)

        # Compute fused depth: sensor where valid, model fills dead pixels
        # (skipped in Gazebo mode -- GT has no dead pixels)
        if not skip_fusion:
            valid_mask = None
            if sensor_depth is not None:
                valid_mask = (sensor_depth >= 0.1) & (sensor_depth <= 5.0)
            if valid_mask is not None and "DA3-Small" in predictions:
                fused_da3 = np.where(valid_mask, sensor_depth, predictions["DA3-Small"])
                predictions["Fused(Sensor+DA3)"] = fused_da3
                save_individual_depth(fused_da3, "Fused_Sensor_DA3", i,
                                     output_dir / "individual", vmax=args.vmax)
            if valid_mask is not None and "V9" in predictions:
                fused_v9 = np.where(valid_mask, sensor_depth, predictions["V9"])
                predictions["Fused(Sensor+V9)"] = fused_v9
                save_individual_depth(fused_v9, "Fused_Sensor_V9", i,
                                     output_dir / "individual", vmax=args.vmax)

        # Reorder predictions for the grid: DA3, students V4..V9, then fused
        student_order = ["V4", "V5", "V6", "V6.2", "V7", "V8", "V9", "Iter6_B1"]
        ordered = {}
        if "DA3-Small" in predictions:
            ordered["DA3-Small"] = predictions["DA3-Small"]
        for name in student_order:
            if name in predictions:
                ordered[name] = predictions[name]
        if "Fused(Sensor+DA3)" in predictions:
            ordered["Fused(Sensor+DA3)"] = predictions["Fused(Sensor+DA3)"]
        if "Fused(Sensor+V9)" in predictions:
            ordered["Fused(Sensor+V9)"] = predictions["Fused(Sensor+V9)"]

        # Comparison grid
        rgb_display = np.array(rgb_pil)
        grid_path = output_dir / "comparisons" / f"frame_{i:04d}_comparison.png"
        create_comparison_grid(
            rgb_display, sensor_depth, ordered, i, grid_path, vmax=args.vmax,
            sensor_label=sensor_label, title_prefix=title_prefix,
        )
        print(f"  Saved comparison grid: {grid_path.name}")

    print(f"\n=== Done ===")
    print(f"Output directory: {output_dir}")
    print(f"  comparisons/  - {len(frames)} side-by-side grids")
    print(f"  individual/   - per-model depth maps")
    print(f"  frames/       - extracted RGB + sensor depth")


if __name__ == "__main__":
    main()
