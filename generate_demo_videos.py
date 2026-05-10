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

def write_video(path, frame_list, fps=FPS):
    if not frame_list:
        print(f"  [SKIP] No frames for {path}")
        return
    h, w = frame_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = path.with_suffix(".mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
    for f in frame_list:
        writer.write(f)
    writer.release()
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {out_path.name} ({len(frame_list)} frames, {w}x{h}, {size_mb:.1f} MB)")


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
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if student_models is None:
        student_models = {}

    video_buffers = {}

    always_keys = ["rgb", "sensor_depth"]
    for k in always_keys:
        video_buffers[k] = []

    if "da3" in models_to_run or "all" in models_to_run:
        video_buffers["da3"] = []
        video_buffers["fused_sensor_da3"] = []

    for m in ["v5", "v6", "v7", "v9"]:
        if m in models_to_run or "all" in models_to_run:
            video_buffers[m] = []
            video_buffers[f"fused_sensor_{m}"] = []

    n = len(frames)
    print(f"\nProcessing {dataset_name}: {n} frames, models={models_to_run}")

    native_h, native_w = None, None

    for i, frame in enumerate(frames):
        rgb_bgr = cv2.imread(str(frame["rgb_path"]))
        if native_h is None:
            native_h, native_w = rgb_bgr.shape[:2]
            print(f"  Native resolution: {native_w}x{native_h}")

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

        sensor_model = cv2.resize(sensor_native, (MODEL_W, MODEL_H),
                                  interpolation=cv2.INTER_NEAREST)
        valid_model = ((sensor_model > MIN_DEPTH) & (sensor_model < MAX_DEPTH)
                       & ~np.isnan(sensor_model))

        depth_sources_model = [(sensor_model, valid_model)]

        # ── DA3 ──
        da3_native = None
        da3_path = frame.get("da3_path")
        if da3_path and Path(da3_path).exists() and "da3" in video_buffers:
            da3_raw = np.load(str(da3_path))
            da3_model = cv2.resize(da3_raw, (MODEL_W, MODEL_H),
                                   interpolation=cv2.INTER_LINEAR)
            da3_model = median_scale(da3_model, sensor_model, valid_model)
            da3_valid_model = (da3_model > 0.05) & (da3_model < 8.0)
            depth_sources_model.append((da3_model, da3_valid_model))
            da3_native = upscale_depth(da3_model, native_h, native_w)

        # ── Student models (V5, V6, V7) ──
        # Use pre-computed .npy if available, otherwise live inference
        student_native = {}
        for mk in ["v5", "v6", "v7"]:
            if mk not in video_buffers:
                continue
            key = f"{mk}_path"
            has_precomputed = (key in frame and frame[key]
                               and Path(frame[key]).exists())
            if has_precomputed:
                pred = np.load(str(frame[key]))
                pred_model = cv2.resize(pred, (MODEL_W, MODEL_H),
                                        interpolation=cv2.INTER_LINEAR)
            elif mk in student_models:
                pred_model = infer_student(
                    student_models[mk], frame["rgb_path"], device)
            else:
                continue
            pred_valid_model = (pred_model > 0.01) & (pred_model < 8.0)
            depth_sources_model.append((pred_model, pred_valid_model))
            student_native[mk] = upscale_depth(pred_model, native_h, native_w)

        # ── V9 (always live inference) ──
        v9_native = None
        if "v9" in video_buffers and "v9" in student_models:
            v9_model = infer_student(
                student_models["v9"], frame["rgb_path"], device)
            v9_valid_model = (v9_model > 0.01) & (v9_model < 8.0)
            depth_sources_model.append((v9_model, v9_valid_model))
            v9_native = upscale_depth(v9_model, native_h, native_w)

        vmin, vmax = compute_shared_range(depth_sources_model)

        # ── Write frames at native resolution ──

        video_buffers["rgb"].append(
            add_label(rgb_bgr, f"RGB  [{i+1}/{n}]"))

        sensor_color = colorize_depth(sensor_native, valid_native, vmin, vmax)
        video_buffers["sensor_depth"].append(
            add_label(sensor_color, f"Sensor Depth ({dead_pct:.0f}% dead)"))

        if da3_native is not None and "da3" in video_buffers:
            da3_valid_nat = (da3_native > 0.05) & (da3_native < 8.0)
            da3_color = colorize_depth(da3_native, da3_valid_nat, vmin, vmax)
            video_buffers["da3"].append(
                add_label(da3_color, "DA3-Small (median-scaled)"))

            fused = fuse_depth(sensor_native, da3_native, valid_native)
            fused_v = valid_native | da3_valid_nat
            video_buffers["fused_sensor_da3"].append(
                add_label(colorize_depth(fused, fused_v, vmin, vmax),
                          "Fused (Sensor + DA3)"))

        label_map = {"v5": "V5 Student", "v6": "V6 Student", "v7": "V7 Student"}
        for mk in ["v5", "v6", "v7"]:
            if mk in student_native and mk in video_buffers:
                pred_nat = student_native[mk]
                pred_v = (pred_nat > 0.01) & (pred_nat < 8.0)
                video_buffers[mk].append(
                    add_label(colorize_depth(pred_nat, pred_v, vmin, vmax),
                              label_map[mk]))

                fused = fuse_depth(sensor_native, pred_nat, valid_native)
                fused_v = valid_native | pred_v
                video_buffers[f"fused_sensor_{mk}"].append(
                    add_label(colorize_depth(fused, fused_v, vmin, vmax),
                              f"Fused (Sensor + {label_map[mk]})"))

        if v9_native is not None and "v9" in video_buffers:
            v9_valid_nat = (v9_native > 0.01) & (v9_native < 8.0)
            video_buffers["v9"].append(
                add_label(colorize_depth(v9_native, v9_valid_nat, vmin, vmax),
                          "V9 Student (corridor FT)"))

            fused = fuse_depth(sensor_native, v9_native, valid_native)
            fused_v = valid_native | v9_valid_nat
            video_buffers["fused_sensor_v9"].append(
                add_label(colorize_depth(fused, fused_v, vmin, vmax),
                          "Fused (Sensor + V9)"))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] frames processed")

    print(f"\nWriting videos to {output_dir}/")
    for key, buf in video_buffers.items():
        if buf:
            write_video(output_dir / key, buf)

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
