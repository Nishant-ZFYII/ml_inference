#!/usr/bin/env python3
"""Generate model comparison grids for all 12 noise profiles × 20 frames.

For each noise profile (P0–P11 from depth_error_injector.py), applies
the profile's noise to Gazebo GT depth, then generates a comparison grid
showing: RGB | Noisy Sensor | DA3-Small | V4 | V5 | V6 | V6.2 | V7 |
V8 | V9 | Iter6_B1 | Fused(Sensor+DA3) | Fused(Sensor+V9).

Model predictions depend only on RGB and are computed ONCE, then reused
across all 12 profiles.  Only the sensor depth and fusion panels change.

Output structure:
    <output_dir>/
        profile_00_GT/
            comparisons/   20 side-by-side grids
            individual/    per-model depth maps
            frames/        RGB + noisy sensor depth
        profile_01_DA3-Small/
            ...
        ...
        profile_11_Sensor+V4/
            ...

Usage:
    python generate_noise_profile_comparisons.py \
        --frame-dir ../NCHSB/gazebo_captured_frames \
        --num-frames 20 \
        --output-dir /home/nishant/maps/paper_figures/noise_profile_comparisons \
        --vmax 5.5
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from models.student import build_student


# ── Student model checkpoints ──

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

STUDENT_ORDER = ["V4", "V5", "V6", "V6.2", "V7", "V8", "V9", "Iter6_B1"]

FUSED_LABELS = {
    "Fused(Sensor+DA3)": "Fused (Sensor + DA3)\nsensor where valid, DA3 fills dead",
    "Fused(Sensor+V9)": "Fused (Sensor + V9)\nsensor where valid, V9 fills dead",
}


# ── Noise profiles (from depth_error_injector.py, no ROS deps) ──

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


# ── Noise injection (pure numpy, copied from depth_error_injector.py) ──

def inject_error(depth, profile, rng, min_depth=0.1, max_depth=10.0):
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

        noise_std[valid & (depth < d_near)] = p.near_rmse * 0.5
        noise_std[valid & (depth >= d_near) & (depth < d_mid)] = p.near_rmse
        noise_std[valid & (depth >= d_mid) & (depth < d_far)] = p.mid_rmse
        noise_std[valid & (depth >= d_far) & (depth < d_beyond)] = p.far_rmse
        noise_std[valid & (depth >= d_beyond)] = p.beyond_rmse
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
        result[dead & ~np.isfinite(depth)] = np.inf
    else:
        if valid.any() and noise_img is not None:
            result[valid] = depth[valid] + noise_img[valid]
            result = np.clip(result, min_depth, max_depth)
        if p.dead_pixel_ratio > 0:
            dead = _make_dead_mask(depth.shape[0], depth.shape[1],
                                   p.dead_pixel_ratio, rng)
            result[dead] = np.inf
        result[~np.isfinite(depth)] = np.inf

    return result


def _make_dead_mask(h, w, ratio, rng):
    bh, bw = (h + 7) // 8, (w + 7) // 8
    block_dead = rng.random((bh, bw)) < ratio
    mask = np.repeat(np.repeat(block_dead, 8, axis=0), 8, axis=1)
    return mask[:h, :w]


# ── Model loading and inference ──

def load_student_model(checkpoint_path, cfg, device):
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
    rgb_resized = rgb_pil.resize((W, H), Image.BILINEAR)
    rgb_np = np.array(rgb_resized, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    if imagenet_norm:
        rgb_t = (rgb_t - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    with torch.no_grad():
        pred_depth, _ = model(rgb_t)
    return pred_depth.squeeze().cpu().numpy()


def load_da3_model(device):
    from transformers import pipeline
    print("Loading DA3-Small model...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device if str(device) != "cpu" else -1,
    )
    print("DA3-Small loaded.")
    return pipe


def run_da3_inference(pipe, rgb_pil, sensor_depth=None):
    """Run DA3-Small, return float32 depth median-scaled against sensor."""
    result = pipe(rgb_pil)
    raw = result["predicted_depth"]
    if hasattr(raw, "numpy"):
        depth = raw.squeeze().numpy().astype(np.float32)
    else:
        depth = np.array(raw, dtype=np.float32)

    H, W = 240, 320
    da3_resized = np.array(
        Image.fromarray(depth).resize((W, H), Image.BILINEAR),
        dtype=np.float32)

    if sensor_depth is not None:
        valid = (sensor_depth >= 0.1) & (sensor_depth <= 8.0)
        if valid.sum() > 100 and np.median(da3_resized[valid]) > 1e-8:
            scale = np.median(sensor_depth[valid]) / np.median(da3_resized[valid])
            da3_resized = da3_resized * scale

    return da3_resized


# ── Grid generation ──

def _panel_range(data):
    """Compute per-panel colormap range from data percentiles."""
    dv = data[(data > 0.05) & (data < 10) & ~np.isnan(data) & np.isfinite(data)]
    if len(dv) > 100:
        vmin = max(0.0, float(np.percentile(dv, 1)))
        vmax = float(np.percentile(dv, 99))
        vmax = max(vmax, vmin + 0.3)
    else:
        vmin, vmax = 0.0, 5.0
    return vmin, vmax


def create_comparison_grid(rgb, sensor_depth, predictions, frame_idx,
                           output_path, vmax=5.5,
                           sensor_label="Injected Depth",
                           title_prefix="Depth Comparison",
                           gt_depth=None):
    n_panels = 2 + len(predictions) + (1 if gt_depth is not None else 0)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 4.5, nrows * 3.6), dpi=150)
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.25, wspace=0.05)

    panels = [("RGB Input", rgb, "rgb", None)]

    if gt_depth is not None:
        panels.append(("GT Depth (Gazebo)", gt_depth, "depth", "GT"))

    panels.append((sensor_label, sensor_depth, "depth", None))

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
            p_vmin, p_vmax = _panel_range(data)
            ax.imshow(data, cmap="inferno",
                      vmin=p_vmin, vmax=p_vmax,
                      interpolation="bilinear")
            ax.text(0.02, 0.02, f"[{p_vmin:.1f}–{p_vmax:.1f} m]",
                    transform=ax.transAxes, fontsize=7,
                    color="white", backgroundcolor=(0, 0, 0, 0.5),
                    verticalalignment="bottom")

        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.axis("off")

    for i in range(len(panels), nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(gs[row, col]).axis("off")

    fig.suptitle(f"Frame {frame_idx:04d} — {title_prefix}",
                 fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_individual_depth(depth, name, frame_idx, output_dir, vmax=None):
    p_vmin, p_vmax = _panel_range(depth) if vmax is None else (0.0, vmax)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=150)
    im = ax.imshow(depth, cmap="inferno", vmin=p_vmin, vmax=p_vmax,
                   interpolation="bilinear")
    ax.axis("off")
    ax.set_title(name, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Depth (m)", fraction=0.046, pad=0.04)
    safe_name = name.replace(' ', '_').replace('/', '_').replace('+', '+')
    out_path = output_dir / f"frame_{frame_idx:04d}_{safe_name}.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate noise profile comparison grids (12 profiles × N frames)")
    parser.add_argument("--frame-dir", type=str,
                        default="../NCHSB/gazebo_captured_frames")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--output-dir", type=str,
                        default="/home/nishant/maps/paper_figures/noise_profile_comparisons")
    parser.add_argument("--vmax", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-da3", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    frame_dir = Path(args.frame_dir)
    output_dir = Path(args.output_dir)
    H, W = 240, 320
    cfg = Config()

    # ── 1. Load frames ──
    print("=== Loading Gazebo frames ===")
    frame_ids = [f"frame_{i:04d}" for i in range(args.num_frames)]
    frames = []
    for fid in frame_ids:
        rgb_path = frame_dir / "rgb" / f"{fid}.png"
        depth_path = frame_dir / "depth" / f"{fid}.npy"
        if not rgb_path.exists() or not depth_path.exists():
            print(f"  {fid}: missing, skipping")
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        gt = np.load(depth_path).astype(np.float32)
        # Sky/ceiling pixels (inf or 0) → far depth so they render as
        # yellow/orange on inferno rather than white NaN holes.
        far_depth = float(args.vmax)
        gt[~np.isfinite(gt)] = far_depth
        gt[gt < 0.01] = far_depth
        frames.append({"frame_id": fid, "rgb": rgb, "gt_depth": gt})
    print(f"Loaded {len(frames)} frames")

    # ── 2. Load student models (once) ──
    print("\n=== Loading Student Models ===")
    loaded_models = {}
    for name, info in MODELS.items():
        ckpt_path = Path(__file__).parent / info["checkpoint"]
        if not ckpt_path.exists():
            print(f"  [{name}] Checkpoint not found: {ckpt_path}, skipping")
            continue
        print(f"  Loading {name}...")
        model = load_student_model(str(ckpt_path), cfg, device)
        loaded_models[name] = (model, info["imagenet_norm"])
        print(f"  [{name}] OK")

    # ── 3. Load DA3 (once) ──
    da3_pipe = None
    if not args.skip_da3:
        try:
            da3_pipe = load_da3_model(device)
        except Exception as e:
            print(f"  DA3 loading failed: {e}. Skipping.")

    # ── 4. Run all models on all frames ONCE, cache results ──
    print(f"\n=== Running inference on {len(frames)} frames (cached for all profiles) ===")
    cached_predictions = []

    for i, f in enumerate(frames):
        print(f"\n--- Frame {i:04d} ({f['frame_id']}) ---")
        rgb_pil = Image.fromarray(f["rgb"]).resize((W, H), Image.BILINEAR)
        gt = f["gt_depth"]

        preds = {}
        for name, (model, imagenet_norm) in loaded_models.items():
            pred = run_student_inference(model, rgb_pil, imagenet_norm,
                                        device, H, W)
            preds[name] = pred
            print(f"  {name}: [{pred.min():.2f}, {pred.max():.2f}]m")

        if da3_pipe is not None:
            print(f"  Running DA3-Small...")
            da3 = run_da3_inference(da3_pipe, rgb_pil, sensor_depth=gt)
            preds["DA3-Small"] = da3
            print(f"  DA3-Small: [{da3.min():.2f}, {da3.max():.2f}]m")

        cached_predictions.append(preds)

    # ── 5. For each profile, apply noise + generate grids ──
    print(f"\n=== Generating grids: {len(PROFILES)} profiles × {len(frames)} frames ===")

    for pid, profile in PROFILES.items():
        profile_dir_name = f"profile_{pid:02d}_{profile.name.replace('+', '+')}"
        prof_dir = output_dir / profile_dir_name
        (prof_dir / "comparisons").mkdir(parents=True, exist_ok=True)
        (prof_dir / "individual").mkdir(parents=True, exist_ok=True)
        (prof_dir / "frames").mkdir(parents=True, exist_ok=True)

        sensor_label = f"P{pid}: {profile.name}\n{profile.description}"
        title_prefix = f"Profile {pid} ({profile.name}) — Gazebo Corridor"

        print(f"\n--- Profile {pid}: {profile.name} ---")

        for i, f in enumerate(frames):
            rgb = f["rgb"]
            gt = f["gt_depth"]
            rgb_pil = Image.fromarray(rgb).resize((W, H), Image.BILINEAR)
            rgb_display = np.array(rgb_pil)
            preds = cached_predictions[i]

            rng = np.random.default_rng(args.seed)
            noisy_sensor = inject_error(gt, profile, rng)

            # Save extracted frame data
            Image.fromarray(rgb).save(
                prof_dir / "frames" / f"frame_{i:04d}_rgb.png")
            np.save(prof_dir / "frames" / f"frame_{i:04d}_depth.npy",
                    noisy_sensor)

            # Save individual depth maps (sensor only varies per profile)
            sensor_vis = noisy_sensor.copy()
            sensor_vis[~np.isfinite(sensor_vis)] = args.vmax
            save_individual_depth(sensor_vis, f"P{pid}_{profile.name}",
                                 i, prof_dir / "individual")

            # Build ordered predictions for grid
            ordered = {}
            if "DA3-Small" in preds:
                ordered["DA3-Small"] = preds["DA3-Small"]
            for name in STUDENT_ORDER:
                if name in preds:
                    ordered[name] = preds[name]

            # Fusion panels: sensor where valid, model fills dead pixels.
            # Dead pixels (from noise injection) show as inf in noisy_sensor.
            sensor_valid = (np.isfinite(noisy_sensor)
                            & (noisy_sensor >= 0.1))
            dead = ~sensor_valid

            def _median_scale(model_pred, sensor, vmask):
                """Scale model predictions to match sensor's metric range."""
                scaled = model_pred.copy()
                if vmask.sum() > 100:
                    s_med = np.median(sensor[vmask])
                    m_med = np.median(model_pred[vmask])
                    if m_med > 1e-8:
                        scaled = model_pred * (s_med / m_med)
                return scaled

            if "DA3-Small" in preds:
                da3_scaled = _median_scale(preds["DA3-Small"],
                                           noisy_sensor, sensor_valid)
                fused_da3 = noisy_sensor.copy()
                fused_da3[dead] = da3_scaled[dead]
                ordered["Fused(Sensor+DA3)"] = fused_da3
            if "V9" in preds:
                v9_scaled = _median_scale(preds["V9"],
                                          noisy_sensor, sensor_valid)
                fused_v9 = noisy_sensor.copy()
                fused_v9[dead] = v9_scaled[dead]
                ordered["Fused(Sensor+V9)"] = fused_v9

            # Sensor depth for display: dead pixels → vmax (far/yellow)
            sensor_for_grid = noisy_sensor.copy()
            sensor_for_grid[~np.isfinite(sensor_for_grid)] = args.vmax

            grid_path = (prof_dir / "comparisons"
                         / f"frame_{i:04d}_comparison.png")
            create_comparison_grid(
                rgb_display, sensor_for_grid, ordered, i, grid_path,
                vmax=args.vmax,
                sensor_label=sensor_label,
                title_prefix=title_prefix,
                gt_depth=gt)

            print(f"  Frame {i:04d}: grid saved")

    print(f"\n=== Done ===")
    print(f"Output: {output_dir}")
    print(f"  {len(PROFILES)} profile directories × {len(frames)} frames each")
    print(f"  Total grids: {len(PROFILES) * len(frames)}")


if __name__ == "__main__":
    main()
