#!/usr/bin/env python3
"""
Generate the missing V5/V6/V7 corridor_eval videos one model at a time
to avoid OOM. Streams frames directly to VideoWriter (no buffering).
"""
import gc
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
OUTPUT_DIR = Path("/media/nishant/SeeGayt2/demo_videos/corridor_eval")

CHECKPOINTS = {
    "v5": "hpc_outputs/best_depth_v5_vivek.pt",
    "v6": "hpc_outputs/best_depth_v6.pt",
    "v7": "hpc_outputs/best_depth_v7.pt",
}

LABELS = {"v5": "V5 Student", "v6": "V6 Student", "v7": "V7 Student"}


def load_model(ckpt_path, device):
    cfg = Config()
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False,
                          backbone=cfg.BACKBONE)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def colorize_depth(depth_m, valid_mask, vmin, vmax):
    h, w = depth_m.shape
    norm = np.zeros((h, w), dtype=np.uint8)
    if valid_mask.any():
        d = np.clip(depth_m[valid_mask], vmin, vmax)
        norm_vals = (255 * (1.0 - (d - vmin) / (vmax - vmin))).clip(0, 255)
        norm[valid_mask] = norm_vals.astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    colored[~valid_mask] = [40, 40, 40]
    return colored


def add_label(frame_bgr, label):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    bar_h = max(36, h // 20)
    font_scale = max(0.6, h / 500.0)
    thickness = 2
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(overlay, label, (10, int(bar_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return overlay


def make_writer(name, w, h):
    path = OUTPUT_DIR / f"{name}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h), isColor=True)
    if not writer.isOpened():
        path = OUTPUT_DIR / f"{name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h), isColor=True)
    return writer, path


def load_frames():
    base = Path(__file__).parent / "corridor_eval_data"
    with open(base / "manifest.jsonl") as f:
        entries = [json.loads(line) for line in f]
    frames = []
    for entry in entries:
        fid = Path(entry["rgb"]).stem
        rgb_p = base / entry["rgb"]
        dep_p = base / entry["sensor_depth"]
        if rgb_p.exists() and dep_p.exists():
            frames.append({"id": fid, "rgb_path": rgb_p, "sensor_path": dep_p})
    return frames


def process_one_model(model_key, frames, device):
    ckpt = Path(__file__).parent / CHECKPOINTS[model_key]
    print(f"\n{'='*60}")
    print(f"Processing {model_key.upper()} ({len(frames)} frames)")
    print(f"Loading {ckpt.name}...")

    model = load_model(str(ckpt), device)
    print(f"  {model_key.upper()} loaded.")

    n = len(frames)
    native_h, native_w = None, None
    model_writer, model_path = None, None
    fused_writer, fused_path = None, None

    for i, frame in enumerate(frames):
        rgb_bgr = cv2.imread(str(frame["rgb_path"]))
        if native_h is None:
            native_h, native_w = rgb_bgr.shape[:2]
            print(f"  Resolution: {native_w}x{native_h}")
            model_writer, model_path = make_writer(model_key, native_w, native_h)
            fused_writer, fused_path = make_writer(
                f"fused_sensor_{model_key}", native_w, native_h)

        sensor = np.load(str(frame["sensor_path"]))
        if sensor.shape != (native_h, native_w):
            sensor = cv2.resize(sensor, (native_w, native_h),
                                interpolation=cv2.INTER_NEAREST)
        valid = ((sensor > MIN_DEPTH) & (sensor < MAX_DEPTH) & ~np.isnan(sensor))

        # Inference at model resolution, then upscale
        rgb_pil = Image.open(str(frame["rgb_path"])).convert("RGB")
        rgb_small = rgb_pil.resize((MODEL_W, MODEL_H), Image.BILINEAR)
        rgb_np = np.array(rgb_small, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = model(rgb_t)
        pred_model = pred.squeeze().cpu().numpy()
        del rgb_t, pred

        pred_native = cv2.resize(pred_model, (native_w, native_h),
                                 interpolation=cv2.INTER_LINEAR)
        # Median-scale against sensor at native resolution
        p_valid = pred_native[valid]
        g_valid = sensor[valid]
        if len(p_valid) > 0 and np.median(p_valid) > 1e-8:
            scale = np.median(g_valid) / np.median(p_valid)
            pred_native = pred_native * scale
        pred_v_nat = (pred_native > 0.01) & (pred_native < 8.0)

        # Shared range at native resolution
        all_vals = []
        if valid.any():
            all_vals.append(sensor[valid])
        if pred_v_nat.any():
            all_vals.append(pred_native[pred_v_nat])
        if all_vals:
            combined = np.concatenate(all_vals)
            vmin = max(0.0, float(np.percentile(combined, 2)))
            vmax = float(np.percentile(combined, 98))
            vmax = max(vmax, vmin + 0.3)
        else:
            vmin, vmax = 0.0, 4.0

        # Model-only video
        vis = colorize_depth(pred_native, pred_v_nat, vmin, vmax)
        model_writer.write(add_label(vis, LABELS[model_key]))

        # Fused video
        fused = np.where(valid, sensor, pred_native)
        fused_v = valid | pred_v_nat
        fused_vis = colorize_depth(fused, fused_v, vmin, vmax)
        fused_writer.write(
            add_label(fused_vis, f"Fused (Sensor + {LABELS[model_key]})"))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] frames done")

    model_writer.release()
    fused_writer.release()
    sz1 = model_path.stat().st_size / 1e6
    sz2 = fused_path.stat().st_size / 1e6
    print(f"  Wrote {model_path.name} ({n} frames, {sz1:.1f} MB)")
    print(f"  Wrote {fused_path.name} ({n} frames, {sz2:.1f} MB)")

    # Free model memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  {model_key.upper()} model unloaded, memory freed.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    print(f"Corridor eval: {len(frames)} frames")

    already = {p.stem for p in OUTPUT_DIR.glob("*.avi")}
    for mk in ["v5", "v6", "v7"]:
        if mk in already and f"fused_sensor_{mk}" in already:
            print(f"  {mk.upper()} already exists, skipping.")
            continue
        process_one_model(mk, frames, device)

    print("\nAll missing corridor videos complete.")


if __name__ == "__main__":
    main()
