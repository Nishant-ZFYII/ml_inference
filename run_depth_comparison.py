#!/usr/bin/env python3
"""
Run DA3-Small and Student V6 on extracted corridor frames.
Create side-by-side comparison: RGB | Sensor | DA3 | Student V6.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_student_v6():
    """Load the student EfficientViT-B1 model."""
    import timm
    ckpt = torch.load(
        '/home/nishant/MS_Project/ml_pipeline/hpc_outputs/best_depth_v6.pt',
        map_location='cpu')

    model = timm.create_model('efficientvit_b1', pretrained=False, num_classes=0)
    model.head = torch.nn.Sequential(
        torch.nn.Conv2d(256, 128, 3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(128, 64, 3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 1, 1),
        torch.nn.ReLU(),
    )

    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, device


def load_da3():
    """Load DA3-Small via HuggingFace pipeline."""
    from transformers import pipeline as hf_pipeline
    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return pipe


def colorize_depth(depth, valid_mask=None, vmin=None, vmax=None):
    """Convert depth to colormap image."""
    if valid_mask is None:
        valid_mask = (depth > 0.05) & (depth < 10.0)

    viz = np.zeros_like(depth, dtype=np.uint8)
    if valid_mask.any():
        d = depth[valid_mask]
        if vmin is None:
            vmin = d.min()
        if vmax is None:
            vmax = d.max()
        if vmax > vmin:
            viz[valid_mask] = (255 * (1.0 - (d - vmin) / (vmax - vmin))).clip(0, 255).astype(np.uint8)

    colored = cv2.applyColorMap(viz, cv2.COLORMAP_INFERNO)
    colored[~valid_mask] = [0, 0, 0]
    return colored


def main():
    frames_dir = Path('/home/nishant/maps/corridor_key_frames')
    out_dir = frames_dir / 'comparison'
    out_dir.mkdir(exist_ok=True)

    manifest = json.load(open(frames_dir / 'manifest.json'))

    print("Loading DA3-Small...")
    da3_pipe = load_da3()

    print("Loading Student V6...")
    try:
        student_model, device = load_student_v6()
        has_student = True
        print("  Student V6 loaded on", device)
    except Exception as e:
        print(f"  Could not load student: {e}")
        has_student = False

    results = []

    for i, entry in enumerate(manifest):
        fname = entry['frame']
        rgb_path = frames_dir / 'rgb' / f'{fname}.png'
        sensor_path = frames_dir / 'depth_sensor' / f'{fname}.npy'

        rgb_pil = Image.open(rgb_path).convert('RGB')
        rgb_np = np.array(rgb_pil)
        sensor_depth = np.load(sensor_path)
        valid = (sensor_depth > 0.1) & (sensor_depth < 5.0)

        # DA3 inference
        da3_result = da3_pipe(rgb_pil)
        da3_raw = np.array(da3_result['depth'], dtype=np.float32)
        da3_resized = cv2.resize(da3_raw,
                                 (sensor_depth.shape[1], sensor_depth.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        if valid.sum() > 100:
            scale = np.median(sensor_depth[valid]) / (np.median(da3_resized[valid]) + 1e-8)
            da3_metric = da3_resized * scale
            da3_rmse = float(np.sqrt(np.mean((da3_metric[valid] - sensor_depth[valid])**2)))
        else:
            da3_metric = da3_resized
            da3_rmse = -1.0
            scale = 1.0

        # Student V6 inference
        student_rmse = -1.0
        student_metric = np.zeros_like(sensor_depth)
        if has_student:
            from torchvision import transforms
            img_t = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])(rgb_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = student_model(img_t)

            if pred.ndim == 4:
                pred = pred.squeeze(0).squeeze(0)
            elif pred.ndim == 3:
                pred = pred.squeeze(0)

            student_raw = pred.cpu().numpy()
            student_resized = cv2.resize(student_raw,
                                         (sensor_depth.shape[1], sensor_depth.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)

            if valid.sum() > 100:
                s_scale = np.median(sensor_depth[valid]) / (np.median(student_resized[valid]) + 1e-8)
                student_metric = student_resized * s_scale
                student_rmse = float(np.sqrt(np.mean(
                    (student_metric[valid] - sensor_depth[valid])**2)))

        # Colorize
        sensor_color = colorize_depth(sensor_depth, valid)
        da3_color = colorize_depth(da3_metric)
        student_color = colorize_depth(student_metric) if has_student else np.zeros_like(sensor_color)

        # Create comparison
        target_h = 360
        aspect = rgb_np.shape[1] / rgb_np.shape[0]
        tw = int(target_h * aspect)

        panels = [
            cv2.resize(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR), (tw, target_h)),
            cv2.resize(sensor_color, (tw, target_h)),
            cv2.resize(da3_color, (tw, target_h)),
        ]
        labels = [
            f"RGB (t={entry['elapsed_s']:.0f}s)",
            f"Sensor ({entry['valid_pct']:.0f}% valid)",
            f"DA3 (RMSE={da3_rmse:.3f}m)" if da3_rmse > 0 else "DA3",
        ]

        if has_student:
            panels.append(cv2.resize(student_color, (tw, target_h)))
            labels.append(f"StudentV6 (RMSE={student_rmse:.3f}m)" if student_rmse > 0 else "StudentV6")

        font = cv2.FONT_HERSHEY_SIMPLEX
        for panel, label in zip(panels, labels):
            cv2.putText(panel, label, (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, label, (10, 30), font, 0.7, (0, 0, 0), 1)

        comparison = np.hstack(panels)
        cv2.imwrite(str(out_dir / f'{fname}.png'), comparison)

        entry_r = {
            **entry,
            'da3_rmse': round(da3_rmse, 4) if da3_rmse > 0 else None,
            'student_rmse': round(student_rmse, 4) if student_rmse > 0 else None,
        }
        results.append(entry_r)
        print(f"  [{i}] {fname}  DA3={da3_rmse:.3f}m  "
              f"Student={student_rmse:.3f}m" if student_rmse > 0
              else f"  [{i}] {fname}  DA3={da3_rmse:.3f}m")

    with open(out_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    da3_rmses = [r['da3_rmse'] for r in results if r['da3_rmse']]
    student_rmses = [r['student_rmse'] for r in results if r.get('student_rmse')]

    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY ({len(results)} frames)")
    if da3_rmses:
        print(f"  DA3-Small  avg RMSE: {np.mean(da3_rmses):.4f}m")
    if student_rmses:
        print(f"  StudentV6  avg RMSE: {np.mean(student_rmses):.4f}m")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
