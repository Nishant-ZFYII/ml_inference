#!/usr/bin/env python3
"""
Generate individual depth-model demo videos from pre-extracted corridor frames.

Produces one MP4 per depth source (sensor, DA3, student, fused variants),
colorized with a shared percentile-based colormap per frame for visual clarity.
All videos are written at native source resolution (typically 1280x720).

Datasets supported:
  - corridor_eval_data (485 rgb / 459 da3 / 485 sensor depth)
  - glass_corridor_frames (121 frames, all models)
  - bag_213831_frames (150 frames, all models)

Usage:
    # All models, all datasets (native resolution)
    python generate_demo_videos.py \
        --dataset all --models all \
        --output-dir /media/nishant/SeeGayt2/demo_videos

    # Single model test
    python generate_demo_videos.py \
        --dataset corridor --models v9 \
        --output-dir /media/nishant/SeeGayt2/demo_videos
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from models.student import build_student

MODEL_H, MODEL_W = 240, 320
MIN_DEPTH, MAX_DEPTH = 0.1, 5.0
FPS = 10


def load_student_model(checkpoint_path, device):
    cfg = Config()
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False,
                          backbone=cfg.BACKBONE)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def colorize_depth(depth_m, valid_mask, vmin, vmax,
                   colormap=cv2.COLORMAP_INFERNO):
    h, w = depth_m.shape
    norm = np.zeros((h, w), dtype=np.uint8)
    if valid_mask.any():
        d = np.clip(depth_m[valid_mask], vmin, vmax)
        norm_vals = (255 * (1.0 - (d - vmin) / (vmax - vmin))).clip(0, 255)
        norm[valid_mask] = norm_vals.astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    colored[~valid_mask] = [40, 40, 40]
    return colored


def compute_shared_range(depth_arrays):
    all_vals = []
    for darr, dmask in depth_arrays:
        if dmask.any():
            all_vals.append(darr[dmask])
    if all_vals:
        combined = np.concatenate(all_vals)
        vmin = max(0.0, float(np.percentile(combined, 2)))
        vmax = float(np.percentile(combined, 98))
        vmax = max(vmax, vmin + 0.3)
        return vmin, vmax
    return 0.0, 4.0


def add_label(frame_bgr, label):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    bar_h = max(36, h // 20)
    font_scale = max(0.6, h / 500.0)
    thickness = 1 if h < 500 else 2
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(overlay, label, (10, int(bar_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return overlay


def median_scale(pred, sensor, valid_mask):
    p_valid = pred[valid_mask]
    g_valid = sensor[valid_mask]
    if len(p_valid) > 0 and np.median(p_valid) > 1e-8:
        scale = np.median(g_valid) / np.median(p_valid)
        return pred * scale
    return pred


def fuse_depth(sensor, model_depth, valid_mask):
    return np.where(valid_mask, sensor, model_depth)


def upscale_depth(depth_small, target_h, target_w):
    """Bilinear upscale a depth array from model res to native res."""
    if depth_small.shape == (target_h, target_w):
        return depth_small
    return cv2.resize(depth_small, (target_w, target_h),
                      interpolation=cv2.INTER_LINEAR)


# ── Dataset loaders ──────────────────────────────────────────────────

def load_corridor_eval_frames():
    base = Path(__file__).parent / "corridor_eval_data"
    manifest_path = base / "manifest.jsonl"

    with open(manifest_path) as f:
        entries = [json.loads(line) for line in f]

    frames = []
    for entry in entries:
        frame_id = Path(entry["rgb"]).stem
        rgb_path = base / entry["rgb"]
        depth_path = base / entry["sensor_depth"]
        da3_path = base / "da3_depth" / f"{frame_id}.npy"
        if not rgb_path.exists() or not depth_path.exists():
            continue
        frames.append({
            "id": frame_id,
            "rgb_path": rgb_path,
            "sensor_path": depth_path,
            "da3_path": da3_path if da3_path.exists() else None,
        })
    return frames, "corridor_eval"


def load_glass_corridor_frames():
    base = Path("/home/nishant/maps/glass_corridor_frames")
    student_base = Path("/home/nishant/maps/glass_corridor_student_results")

    rgbs = sorted((base / "rgb").glob("*.png"))
    frames = []
    for rgb_path in rgbs:
        stem = rgb_path.stem
        frames.append({
            "id": stem,
            "rgb_path": rgb_path,
            "sensor_path": base / "depth" / f"{stem}.npy",
            "da3_path": base / "da3_depth" / f"{stem}.npy",
            "v5_path": student_base / "depth_v5_vivek" / f"{stem}.npy",
            "v6_path": student_base / "depth_v6" / f"{stem}.npy",
            "v7_path": student_base / "depth_v7" / f"{stem}.npy",
        })
    return frames, "glass_corridor"


def load_bag_213831_frames():
    base = Path("/home/nishant/maps/bag_213831_frames")
    student_base = Path("/home/nishant/maps/bag_213831_student_results")

    rgbs = sorted((base / "rgb").glob("*.png"))
    frames = []
    for rgb_path in rgbs:
        stem = rgb_path.stem
        frames.append({
            "id": stem,
            "rgb_path": rgb_path,
            "sensor_path": base / "depth" / f"{stem}.npy",
            "da3_path": student_base / "depth_da3" / f"{stem}.npy",
            "v5_path": student_base / "depth_v5_vivek" / f"{stem}.npy",
            "v6_path": student_base / "depth_v6" / f"{stem}.npy",
            "v7_path": student_base / "depth_v7" / f"{stem}.npy",
        })
    return frames, "bag_213831"


# ── Video writer ─────────────────────────────────────────────────────

def open_writer(path, w, h, fps=FPS):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = out_path.with_suffix(".mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
    return writer, out_path


# ── Main pipeline ────────────────────────────────────────────────────

def infer_student(model, rgb_path, device):
    """Run a single student model forward pass, return (MODEL_H, MODEL_W) depth."""
    rgb_pil = Image.open(str(rgb_path)).convert("RGB")
    rgb_small = rgb_pil.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    rgb_np = np.array(rgb_small, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, _ = model(rgb_t)
    return pred.squeeze().cpu().numpy()


def process_dataset(frames, dataset_name, output_dir, models_to_run,
                    student_models=None, device=None):
    """Stream-based: opens all VideoWriters up front, writes each frame
    immediately, never buffers more than one frame per video in memory."""
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if student_models is None:
        student_models = {}

    active_keys = ["rgb", "sensor_depth"]
    if "da3" in models_to_run or "all" in models_to_run:
        active_keys += ["da3", "fused_sensor_da3"]
    for m in ["v5", "v6", "v7", "v9"]:
        if m in models_to_run or "all" in models_to_run:
            active_keys += [m, f"fused_sensor_{m}"]

    n = len(frames)
    print(f"\nProcessing {dataset_name}: {n} frames, models={models_to_run}")

    native_h, native_w = None, None
    writers = {}
    writer_paths = {}
    frame_counts = {k: 0 for k in active_keys}

    for i, frame in enumerate(frames):
        rgb_bgr = cv2.imread(str(frame["rgb_path"]))
        if native_h is None:
            native_h, native_w = rgb_bgr.shape[:2]
            print(f"  Native resolution: {native_w}x{native_h}")
            for k in active_keys:
                w, p = open_writer(output_dir / k, native_w, native_h)
                writers[k] = w
                writer_paths[k] = p

        if rgb_bgr.shape[:2] != (native_h, native_w):
            rgb_bgr = cv2.resize(rgb_bgr, (native_w, native_h))

        sensor_raw = np.load(str(frame["sensor_path"]))
        if sensor_raw.shape != (native_h, native_w):
            sensor_native = cv2.resize(sensor_raw, (native_w, native_h),
                                       interpolation=cv2.INTER_NEAREST)
        else:
            sensor_native = sensor_raw
        valid_native = ((sensor_native > MIN_DEPTH) & (sensor_native < MAX_DEPTH)
                        & ~np.isnan(sensor_native))
        dead_pct = (~valid_native).sum() * 100.0 / sensor_native.size

        depth_sources_native = [(sensor_native, valid_native)]

        # ── DA3 (processed at native resolution) ──
        da3_native = None
        da3_valid_native = None
        da3_path = frame.get("da3_path")
        if da3_path and Path(da3_path).exists() and "da3" in writers:
            da3_raw = np.load(str(da3_path))
            if da3_raw.shape != (native_h, native_w):
                da3_raw = cv2.resize(da3_raw, (native_w, native_h),
                                     interpolation=cv2.INTER_LINEAR)
            da3_native = median_scale(da3_raw, sensor_native, valid_native)
            da3_valid_native = (da3_native > 0.05) & (da3_native < 8.0)
            depth_sources_native.append((da3_native, da3_valid_native))

        # ── Student models (V5, V6, V7) ──
        student_native = {}
        for mk in ["v5", "v6", "v7"]:
            if mk not in writers:
                continue
            key = f"{mk}_path"
            has_precomputed = (key in frame and frame[key]
                               and Path(frame[key]).exists())
            if has_precomputed:
                pred = np.load(str(frame[key]))
                if pred.shape != (native_h, native_w):
                    pred_nat = cv2.resize(pred, (native_w, native_h),
                                          interpolation=cv2.INTER_LINEAR)
                else:
                    pred_nat = pred
            elif mk in student_models:
                pred_model = infer_student(
                    student_models[mk], frame["rgb_path"], device)
                pred_nat = upscale_depth(pred_model, native_h, native_w)
            else:
                continue
            pred_nat = median_scale(pred_nat, sensor_native, valid_native)
            pred_valid_nat = (pred_nat > 0.01) & (pred_nat < 8.0)
            depth_sources_native.append((pred_nat, pred_valid_nat))
            student_native[mk] = pred_nat

        # ── V9 (always live inference) ──
        v9_native = None
        if "v9" in writers and "v9" in student_models:
            v9_model = infer_student(
                student_models["v9"], frame["rgb_path"], device)
            v9_nat = upscale_depth(v9_model, native_h, native_w)
            v9_nat = median_scale(v9_nat, sensor_native, valid_native)
            v9_valid_nat = (v9_nat > 0.01) & (v9_nat < 8.0)
            depth_sources_native.append((v9_nat, v9_valid_nat))
            v9_native = v9_nat

        vmin, vmax = compute_shared_range(depth_sources_native)

        # ── Stream frames directly to writers ──

        writers["rgb"].write(add_label(rgb_bgr, f"RGB  [{i+1}/{n}]"))
        frame_counts["rgb"] += 1

        sensor_color = colorize_depth(sensor_native, valid_native, vmin, vmax)
        writers["sensor_depth"].write(
            add_label(sensor_color, f"Sensor Depth ({dead_pct:.0f}% dead)"))
        frame_counts["sensor_depth"] += 1

        if da3_native is not None and "da3" in writers:
            da3_color = colorize_depth(da3_native, da3_valid_native, vmin, vmax)
            writers["da3"].write(
                add_label(da3_color, "DA3-Small (median-scaled)"))
            frame_counts["da3"] += 1

            fused = fuse_depth(sensor_native, da3_native, valid_native)
            fused_v = valid_native | da3_valid_native
            writers["fused_sensor_da3"].write(
                add_label(colorize_depth(fused, fused_v, vmin, vmax),
                          "Fused (Sensor + DA3)"))
            frame_counts["fused_sensor_da3"] += 1

        label_map = {"v5": "V5 Student", "v6": "V6 Student", "v7": "V7 Student"}
        for mk in ["v5", "v6", "v7"]:
            if mk in student_native and mk in writers:
                pred_nat = student_native[mk]
                pred_v = (pred_nat > 0.01) & (pred_nat < 8.0)
                writers[mk].write(
                    add_label(colorize_depth(pred_nat, pred_v, vmin, vmax),
                              label_map[mk]))
                frame_counts[mk] += 1

                fused = fuse_depth(sensor_native, pred_nat, valid_native)
                fused_v = valid_native | pred_v
                writers[f"fused_sensor_{mk}"].write(
                    add_label(colorize_depth(fused, fused_v, vmin, vmax),
                              f"Fused (Sensor + {label_map[mk]})"))
                frame_counts[f"fused_sensor_{mk}"] += 1

        if v9_native is not None and "v9" in writers:
            v9_v = (v9_native > 0.01) & (v9_native < 8.0)
            writers["v9"].write(
                add_label(colorize_depth(v9_native, v9_v, vmin, vmax),
                          "V9 Student (corridor FT)"))
            frame_counts["v9"] += 1

            fused = fuse_depth(sensor_native, v9_native, valid_native)
            fused_v = valid_native | v9_v
            writers["fused_sensor_v9"].write(
                add_label(colorize_depth(fused, fused_v, vmin, vmax),
                          "Fused (Sensor + V9)"))
            frame_counts["fused_sensor_v9"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] frames processed")

    for k, w in writers.items():
        w.release()
        p = writer_paths[k]
        sz = p.stat().st_size / 1e6
        print(f"  Wrote {p.name} ({frame_counts[k]} frames, "
              f"{native_w}x{native_h}, {sz:.1f} MB)")

    print(f"Done: {dataset_name}\n")


CHECKPOINT_MAP = {
    "v5": "hpc_outputs/best_depth_v5_vivek.pt",
    "v6": "hpc_outputs/best_depth_v6.pt",
    "v7": "hpc_outputs/best_depth_v7.pt",
    "v9": "hpc_outputs/best_depth_v9.pt",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth model demo videos at native resolution")
    parser.add_argument("--dataset", type=str, default="corridor",
                        choices=["corridor", "glass", "bag213831", "all"])
    parser.add_argument("--models", type=str, nargs="+", default=["v9"],
                        help="Models to include: v5 v6 v7 v9 da3, or 'all'")
    parser.add_argument("--output-dir", type=str,
                        default="/media/nishant/SeeGayt2/demo_videos")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    global FPS
    FPS = args.fps

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = set(args.models)
    want_all = "all" in models_to_run

    student_models = {}
    for mk, ckpt_path in CHECKPOINT_MAP.items():
        if want_all or mk in models_to_run:
            ckpt_full = Path(__file__).parent / ckpt_path
            if ckpt_full.exists():
                print(f"Loading {mk.upper()} model from {ckpt_path}...")
                student_models[mk] = load_student_model(str(ckpt_full), device)
                print(f"  {mk.upper()} model loaded.")
            else:
                print(f"  [WARN] Checkpoint not found: {ckpt_full}")

    datasets = []
    if args.dataset in ("corridor", "all"):
        datasets.append(load_corridor_eval_frames)
    if args.dataset in ("glass", "all"):
        datasets.append(load_glass_corridor_frames)
    if args.dataset in ("bag213831", "all"):
        datasets.append(load_bag_213831_frames)

    for loader in datasets:
        frames, name = loader()
        process_dataset(frames, name, args.output_dir, models_to_run,
                        student_models=student_models, device=device)

    for model in student_models.values():
        del model
    torch.cuda.empty_cache()

    print("\nAll videos complete.")


if __name__ == "__main__":
    main()
