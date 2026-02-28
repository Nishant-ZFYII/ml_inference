#!/usr/bin/env python3
"""
Extract RGB + depth frames from an Orbbec Femto Bolt MCAP rosbag.

Reads synchronized color and depth images, subsamples every Nth frame,
converts depth from uint16 mm to float32 metres, and writes output in the
corridor_loader.py layout:

    <output_dir>/
        rgb/        00000.png, 00005.png, ...
        depth/      00000.npy, 00005.npy, ...
        intrinsics.json
        manifest.jsonl

Usage:
    python -m teacher_infer.extract_corridor_bag \
        --bag  ../NCHSB/rosbags/rgbd_imu_20260228_003828_0.mcap \
        --output-dir corridor_eval_data \
        --subsample 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def extract(bag_path: str, output_dir: str, subsample: int,
            color_topic: str, depth_topic: str, info_topic: str):
    from mcap_ros2.reader import read_ros2_messages

    out = Path(output_dir)
    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = None
    frame_count = 0
    saved = 0
    manifest = []

    pending_color = None
    pending_color_ts = None

    print(f"Streaming {bag_path} ...")
    print(f"  color: {color_topic}")
    print(f"  depth: {depth_topic}")
    print(f"  subsample: every {subsample}th pair")

    MAX_DT_NS = 50_000_000  # 50 ms

    for msg in read_ros2_messages(bag_path,
                                  topics=[color_topic, depth_topic, info_topic]):
        topic = msg.channel.topic
        ts = msg.log_time_ns

        if topic == info_topic and intrinsics is None:
            ros_msg = msg.ros_msg
            K = ros_msg.k
            intrinsics = {
                "fx": float(K[0]), "fy": float(K[4]),
                "cx": float(K[2]), "cy": float(K[5]),
                "width": int(ros_msg.width),
                "height": int(ros_msg.height),
                "distortion_model": str(ros_msg.distortion_model),
            }
            print(f"Intrinsics: fx={intrinsics['fx']:.2f} fy={intrinsics['fy']:.2f} "
                  f"cx={intrinsics['cx']:.2f} cy={intrinsics['cy']:.2f} "
                  f"({intrinsics['width']}x{intrinsics['height']})")
            continue

        if topic == color_topic:
            pending_color = msg.ros_msg
            pending_color_ts = ts
            continue

        if topic == depth_topic and pending_color is not None:
            dt = abs(ts - pending_color_ts)
            if dt > MAX_DT_NS:
                pending_color = None
                pending_color_ts = None
                continue

            frame_count += 1
            if (frame_count - 1) % subsample != 0:
                pending_color = None
                pending_color_ts = None
                continue

            stem = f"{frame_count - 1:05d}"

            # Decode color (rgb8)
            c_msg = pending_color
            h_c, w_c = c_msg.height, c_msg.width
            rgb_np = np.frombuffer(c_msg.data, dtype=np.uint8).reshape(h_c, w_c, 3)
            rgb_img = Image.fromarray(rgb_np, mode="RGB")
            rgb_path = rgb_dir / f"{stem}.png"
            rgb_img.save(rgb_path)

            # Decode depth (16UC1 mm → float32 metres)
            d_msg = msg.ros_msg
            h_d, w_d = d_msg.height, d_msg.width
            depth_raw = np.frombuffer(d_msg.data, dtype=np.uint16).reshape(h_d, w_d)
            depth_m = depth_raw.astype(np.float32) / 1000.0
            depth_m[depth_raw == 0] = 0.0
            depth_path = depth_dir / f"{stem}.npy"
            np.save(depth_path, depth_m)

            manifest.append({
                "rgb": f"rgb/{stem}.png",
                "sensor_depth": f"depth/{stem}.npy",
            })
            saved += 1

            if saved % 50 == 0:
                print(f"  Saved {saved} frames ...")

            pending_color = None
            pending_color_ts = None

    print(f"\nTotal paired frames: {frame_count}")
    print(f"Saved (subsample={subsample}): {saved}")

    # Write intrinsics
    if intrinsics:
        intr_path = out / "intrinsics.json"
        with open(intr_path, "w") as f:
            json.dump(intrinsics, f, indent=2)
        print(f"Intrinsics saved: {intr_path}")

    # Write manifest
    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"\nExtraction complete:")
    print(f"  RGB:      {rgb_dir}  ({saved} files)")
    print(f"  Depth:    {depth_dir}  ({saved} files)")
    print(f"  Manifest: {manifest_path}  ({saved} entries)")

    # Quick stats
    if saved > 0:
        all_valid = []
        total_px = 0
        n_check = min(10, saved)
        for entry in manifest[:n_check]:
            d = np.load(out / entry["sensor_depth"])
            valid = d[d > 0]
            all_valid.append(valid)
            total_px += d.size
        combined = np.concatenate(all_valid)
        print(f"\nDepth stats (first {n_check} frames):")
        print(f"  Valid pixels: {len(combined)} / {total_px} "
              f"({len(combined)*100/total_px:.1f}%)")
        print(f"  Range: {combined.min():.3f} - {combined.max():.3f} m")
        print(f"  Mean:  {combined.mean():.3f} m, "
              f"Median: {np.median(combined):.3f} m")


def main():
    parser = argparse.ArgumentParser(
        description="Extract corridor RGB+depth from MCAP rosbag")
    parser.add_argument("--bag", type=str, required=True,
                        help="Path to .mcap rosbag file")
    parser.add_argument("--output-dir", type=str, default="corridor_eval_data",
                        help="Output directory (default: corridor_eval_data)")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Keep every Nth frame (default: 5)")
    parser.add_argument("--color-topic", type=str,
                        default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", type=str,
                        default="/camera/depth/image_raw")
    parser.add_argument("--info-topic", type=str,
                        default="/camera/color/camera_info")
    args = parser.parse_args()

    extract(args.bag, args.output_dir, args.subsample,
            args.color_topic, args.depth_topic, args.info_topic)


if __name__ == "__main__":
    main()
