#!/usr/bin/env python3
"""
Generate a synchronized 2x3 grid video showing all depth sources side by side.

Layout (2 rows x 3 cols):
  ┌──────────┬──────────────────┬────────────┐
  │   RGB    │  DA3-Small       │  V5 Student│
  ├──────────┼──────────────────┼────────────┤
  │V6 Student│  V7 Student      │  V9 Student│
  └──────────┴──────────────────┴────────────┘

Each cell is 640x360, total grid is 1920x720.
Streams one frame at a time to VideoWriter to stay memory-safe.
Models are loaded sequentially: for each frame, run all 4 student models,
compose the grid, write, and discard.
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

CHECKPOINTS = {
    "v5": "hpc_outputs/best_depth_v5_vivek.pt",
    "v6": "hpc_outputs/best_depth_v6.pt",
    "v7": "hpc_outputs/best_depth_v7.pt",
    "v9": "hpc_outputs/best_depth_v9.pt",
}

GRID_CONFIGS = {
    "models": {
        "cols": 3,
        "cell": (640, 360),
        "layout": [
            ["rgb", "da3", "v5"],
            ["v6", "v7", "v9"],
        ],
    },
    "fused": {
        "cols": 3,
        "cell": (640, 360),
        "layout": [
            ["sensor", "fused_da3", "fused_v5"],
            ["fused_v6", "fused_v7", "fused_v9"],
        ],
    },
    "combined": {
        "cols": 4,
        "cell": (480, 360),
        "layout": [
            ["rgb", "sensor", "fused_da3", "fused_v5"],
            ["fused_v6", "fused_v7", "fused_v9", "_stats"],
        ],
    },
}

LABELS = {
    "rgb": "RGB",
    "sensor": "Sensor Depth",
    "da3": "DA3-Small (median-scaled)",
    "v5": "V5 Student",
    "v6": "V6 Student",
    "v7": "V7 Student",
    "v9": "V9 Student (corridor FT)",
    "fused_da3": "Fused (Sensor + DA3)",
    "fused_v5": "Fused (Sensor + V5)",
    "fused_v6": "Fused (Sensor + V6)",
    "fused_v7": "Fused (Sensor + V7)",
    "fused_v9": "Fused (Sensor + V9)",
}


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
    bar_h = max(24, h // 15)
    font_scale = max(0.45, h / 700.0)
    thickness = 1
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(overlay, label, (6, int(bar_h * 0.72)),
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


def infer_student(model, rgb_path, device):
    rgb_pil = Image.open(str(rgb_path)).convert("RGB")
    rgb_small = rgb_pil.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    rgb_np = np.array(rgb_small, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, _ = model(rgb_t)
    result = pred.squeeze().cpu().numpy()
    del rgb_t, pred
    return result


def to_cell(img_bgr, cell_w, cell_h):
    return cv2.resize(img_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)


def make_stats_panel(cell_w, cell_h, frame_idx, n_frames, dead_pct,
                     vmin, vmax, dataset_name):
    panel = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    panel[:] = [30, 30, 30]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, cell_h / 900.0)
    color = (220, 220, 220)
    dim_color = (140, 140, 140)
    y = 30
    gap = int(28 * (cell_h / 360.0))

    cv2.putText(panel, dataset_name, (10, y), font, scale * 1.1,
                (255, 255, 255), 1, cv2.LINE_AA)
    y += gap + 8
    cv2.putText(panel, f"Frame {frame_idx+1} / {n_frames}", (10, y),
                font, scale, color, 1, cv2.LINE_AA)
    y += gap
    cv2.putText(panel, f"Dead pixels: {dead_pct:.1f}%", (10, y),
                font, scale, color, 1, cv2.LINE_AA)
    y += gap
    cv2.putText(panel, f"Depth range:", (10, y),
                font, scale, dim_color, 1, cv2.LINE_AA)
    y += gap
    cv2.putText(panel, f"  {vmin:.2f} - {vmax:.2f} m", (10, y),
                font, scale, color, 1, cv2.LINE_AA)
    y += gap + 10
    cv2.putText(panel, "Colormap: inferno", (10, y),
                font, scale * 0.8, dim_color, 1, cv2.LINE_AA)
    y += gap
    cv2.putText(panel, "Fusion: sensor where", (10, y),
                font, scale * 0.8, dim_color, 1, cv2.LINE_AA)
    y += int(gap * 0.8)
    cv2.putText(panel, "valid, model elsewhere", (10, y),
                font, scale * 0.8, dim_color, 1, cv2.LINE_AA)
    return panel


# ── Dataset loaders ──────────────────────────────────────────────────

def load_corridor_eval_frames():
    base = Path(__file__).parent / "corridor_eval_data"
    with open(base / "manifest.jsonl") as f:
        entries = [json.loads(line) for line in f]
    frames = []
    for entry in entries:
        fid = Path(entry["rgb"]).stem
        rgb_p = base / entry["rgb"]
        dep_p = base / entry["sensor_depth"]
        da3_p = base / "da3_depth" / f"{fid}.npy"
        if rgb_p.exists() and dep_p.exists():
            frames.append({
                "id": fid, "rgb_path": rgb_p, "sensor_path": dep_p,
                "da3_path": da3_p if da3_p.exists() else None,
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


# ── Grid assembly ────────────────────────────────────────────────────

def fuse_depth(sensor, model_depth, valid_mask):
    return np.where(valid_mask, sensor, model_depth)


def process_dataset(frames, dataset_name, output_dir, student_models, device,
                    variant="models"):
    cfg = GRID_CONFIGS[variant]
    grid_layout = cfg["layout"]
    cell_w, cell_h = cfg["cell"]
    n_cols = cfg["cols"]
    n_rows = len(grid_layout)
    grid_w = cell_w * n_cols
    grid_h = cell_h * n_rows

    suffix = "" if variant == "models" else f"_{variant}"
    out_path = Path(output_dir) / f"{dataset_name}_grid{suffix}.avi"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS,
                             (grid_w, grid_h), isColor=True)
    if not writer.isOpened():
        out_path = out_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS,
                                 (grid_w, grid_h), isColor=True)

    n = len(frames)
    print(f"\nProcessing {dataset_name} [{variant}]: {n} frames -> {out_path.name}")
    print(f"  Grid: {grid_w}x{grid_h} ({cell_w}x{cell_h} per cell, {n_cols}x{n_rows})")

    for i, frame in enumerate(frames):
        rgb_bgr = cv2.imread(str(frame["rgb_path"]))

        sensor = np.load(str(frame["sensor_path"]))
        sensor_model = cv2.resize(sensor, (MODEL_W, MODEL_H),
                                  interpolation=cv2.INTER_NEAREST)
        valid_model = ((sensor_model > MIN_DEPTH) & (sensor_model < MAX_DEPTH)
                       & ~np.isnan(sensor_model))
        dead_pct = (~valid_model).sum() * 100.0 / sensor_model.size

        depth_sources = [(sensor_model, valid_model)]

        # DA3
        da3_model = None
        da3_path = frame.get("da3_path")
        if da3_path and Path(da3_path).exists():
            da3_raw = np.load(str(da3_path))
            da3_model = cv2.resize(da3_raw, (MODEL_W, MODEL_H),
                                   interpolation=cv2.INTER_LINEAR)
            da3_model = median_scale(da3_model, sensor_model, valid_model)
            da3_valid = (da3_model > 0.05) & (da3_model < 8.0)
            depth_sources.append((da3_model, da3_valid))

        # Student predictions (pre-computed or live)
        student_preds = {}
        for mk in ["v5", "v6", "v7"]:
            key = f"{mk}_path"
            has_pre = (key in frame and frame[key]
                       and Path(frame[key]).exists())
            if has_pre:
                pred = np.load(str(frame[key]))
                pred_m = cv2.resize(pred, (MODEL_W, MODEL_H),
                                    interpolation=cv2.INTER_LINEAR)
            elif mk in student_models:
                pred_m = infer_student(student_models[mk],
                                       frame["rgb_path"], device)
            else:
                continue
            pred_v = (pred_m > 0.01) & (pred_m < 8.0)
            student_preds[mk] = (pred_m, pred_v)
            depth_sources.append((pred_m, pred_v))

        # V9 always live
        if "v9" in student_models:
            v9_m = infer_student(student_models["v9"],
                                  frame["rgb_path"], device)
            v9_v = (v9_m > 0.01) & (v9_m < 8.0)
            student_preds["v9"] = (v9_m, v9_v)
            depth_sources.append((v9_m, v9_v))

        # Shared color range
        all_vals = []
        for darr, dmask in depth_sources:
            if dmask.any():
                all_vals.append(darr[dmask])
        if all_vals:
            combined = np.concatenate(all_vals)
            vmin = max(0.0, float(np.percentile(combined, 2)))
            vmax = float(np.percentile(combined, 98))
            vmax = max(vmax, vmin + 0.3)
        else:
            vmin, vmax = 0.0, 4.0

        # Build cell images
        cells = {}
        placeholder = np.full((cell_h, cell_w, 3), 40, dtype=np.uint8)

        cells["rgb"] = to_cell(rgb_bgr, cell_w, cell_h)

        sensor_col = colorize_depth(sensor_model, valid_model, vmin, vmax)
        cells["sensor"] = cv2.resize(sensor_col, (cell_w, cell_h))

        if da3_model is not None:
            da3_col = colorize_depth(da3_model, da3_valid, vmin, vmax)
            cells["da3"] = cv2.resize(da3_col, (cell_w, cell_h))
            fused_da3 = fuse_depth(sensor_model, da3_model, valid_model)
            fused_da3_v = valid_model | da3_valid
            fused_col = colorize_depth(fused_da3, fused_da3_v, vmin, vmax)
            cells["fused_da3"] = cv2.resize(fused_col, (cell_w, cell_h))
        else:
            cells["da3"] = placeholder.copy()
            cells["fused_da3"] = placeholder.copy()

        for mk in ["v5", "v6", "v7", "v9"]:
            if mk in student_preds:
                pred_m, pred_v = student_preds[mk]
                col = colorize_depth(pred_m, pred_v, vmin, vmax)
                cells[mk] = cv2.resize(col, (cell_w, cell_h))
                fused = fuse_depth(sensor_model, pred_m, valid_model)
                fused_v = valid_model | pred_v
                fused_col = colorize_depth(fused, fused_v, vmin, vmax)
                cells[f"fused_{mk}"] = cv2.resize(fused_col, (cell_w, cell_h))
            else:
                cells[mk] = placeholder.copy()
                cells[f"fused_{mk}"] = placeholder.copy()

        cells["_stats"] = make_stats_panel(
            cell_w, cell_h, i, n, dead_pct, vmin, vmax, dataset_name)

        # Add labels (skip stats panel, it has its own text)
        frame_tag = f" [{i+1}/{n}]"
        first_cell = grid_layout[0][0]
        for key in cells:
            if key == "_stats":
                continue
            lbl = LABELS.get(key, key.upper())
            if key == first_cell:
                lbl += frame_tag
            if key == "sensor":
                lbl += f" ({dead_pct:.0f}% dead)"
            cells[key] = add_label(cells[key], lbl)

        # Assemble grid
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for r, row_keys in enumerate(grid_layout):
            for c, key in enumerate(row_keys):
                y0 = r * cell_h
                x0 = c * cell_w
                grid[y0:y0+cell_h, x0:x0+cell_w] = cells[key]

        writer.write(grid)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] frames done")

    writer.release()
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {out_path.name} ({n} frames, {grid_w}x{grid_h}, {size_mb:.1f} MB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate grid comparison video")
    parser.add_argument("--dataset", type=str, default="corridor",
                        choices=["corridor", "glass", "bag213831", "all"])
    parser.add_argument("--variant", type=str, nargs="+", default=["models"],
                        choices=["models", "fused", "combined"],
                        help="Grid variant(s): 'models' (RGB+DA3+students), "
                             "'fused' (sensor+fused), "
                             "'combined' (RGB+sensor+all fused, 2x4 grid)")
    parser.add_argument("--output-dir", type=str,
                        default="/media/nishant/SeeGayt2/demo_videos")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    global FPS
    FPS = args.fps

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    student_models = {}
    base = Path(__file__).parent
    for mk, ckpt in CHECKPOINTS.items():
        p = base / ckpt
        if p.exists():
            print(f"Loading {mk.upper()} from {ckpt}...")
            student_models[mk] = load_model(str(p), device)
            print(f"  {mk.upper()} loaded.")
    print(f"Models in memory: {list(student_models.keys())}")

    loaders = []
    if args.dataset in ("corridor", "all"):
        loaders.append(load_corridor_eval_frames)
    if args.dataset in ("glass", "all"):
        loaders.append(load_glass_corridor_frames)
    if args.dataset in ("bag213831", "all"):
        loaders.append(load_bag_213831_frames)

    for loader in loaders:
        frames, name = loader()
        for v in args.variant:
            process_dataset(frames, name, args.output_dir, student_models,
                            device, variant=v)
        gc.collect()

    for m in student_models.values():
        del m
    torch.cuda.empty_cache()
    print("\nAll grid videos complete.")


if __name__ == "__main__":
    main()
