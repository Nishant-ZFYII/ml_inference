#!/usr/bin/env python3
"""
Extract RGB + depth frames from an Orbbec Femto Bolt MCAP rosbag.

Reads synchronized color and depth images, subsamples every Nth frame,
converts depth from uint16 mm to float32 metres, **reprojects depth into
the color camera frame** using extrinsics from /tf_static, and writes
output in the corridor_loader.py layout:

    <output_dir>/
        rgb/        00000.png, 00005.png, ...
        depth/      00000.npy, 00005.npy, ...  (aligned to color frame)
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


def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x, y, z, w) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])


def read_extrinsics(bag_path: str):
    """Read depth→color extrinsic from /tf_static in the bag."""
    from mcap_ros2.reader import read_ros2_messages

    depth_to_color_t = None
    depth_to_color_q = None
    optical_q = None

    for msg in read_ros2_messages(bag_path, topics=["/tf_static"]):
        for t in msg.ros_msg.transforms:
            parent = t.header.frame_id
            child = t.child_frame_id
            tr = t.transform.translation
            rot = t.transform.rotation

            if parent == "camera_depth_frame" and child == "camera_color_frame":
                depth_to_color_t = np.array([tr.x, tr.y, tr.z])
                depth_to_color_q = (rot.x, rot.y, rot.z, rot.w)

            if parent == "camera_depth_frame" and child == "camera_depth_optical_frame":
                optical_q = (rot.x, rot.y, rot.z, rot.w)

            if parent == "camera_color_frame" and child == "camera_color_optical_frame":
                color_optical_q = (rot.x, rot.y, rot.z, rot.w)
        break

    if depth_to_color_t is None:
        raise RuntimeError("Could not find depth→color transform in /tf_static")

    # Chain: depth_optical → depth_frame → color_frame → color_optical
    R_depth_opt = quat_to_rot(*optical_q)       # depth_frame → depth_optical
    R_depth_opt_inv = R_depth_opt.T              # depth_optical → depth_frame
    R_dc = quat_to_rot(*depth_to_color_q)       # depth_frame → color_frame
    R_co = quat_to_rot(*color_optical_q)         # color_frame → color_optical

    R = R_co @ R_dc @ R_depth_opt_inv
    t = R_co @ depth_to_color_t

    print(f"Extrinsics (depth_optical → color_optical):")
    print(f"  Translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}] "
          f"({np.linalg.norm(t)*1000:.1f} mm baseline)")
    return R, t


def align_depth_to_color(depth_m, depth_intr, color_intr, R, t):
    """
    Reproject depth image from depth camera frame into color camera frame.

    For each valid pixel in the depth image:
      1. Back-project to 3D in depth optical frame
      2. Transform to color optical frame using R, t
      3. Project onto color image plane
      4. Write depth value at the projected location
    """
    h_d, w_d = depth_m.shape
    h_c = color_intr["height"]
    w_c = color_intr["width"]

    fx_d, fy_d = depth_intr["fx"], depth_intr["fy"]
    cx_d, cy_d = depth_intr["cx"], depth_intr["cy"]
    fx_c, fy_c = color_intr["fx"], color_intr["fy"]
    cx_c, cy_c = color_intr["cx"], color_intr["cy"]

    aligned = np.zeros((h_c, w_c), dtype=np.float32)

    # Vectorised: get all valid pixel coordinates
    valid_mask = depth_m > 0
    vs, us = np.where(valid_mask)
    ds = depth_m[vs, us]

    # Back-project to 3D (depth optical frame)
    X_d = (us.astype(np.float64) - cx_d) * ds / fx_d
    Y_d = (vs.astype(np.float64) - cy_d) * ds / fy_d
    Z_d = ds.astype(np.float64)

    pts_d = np.stack([X_d, Y_d, Z_d], axis=1)  # (N, 3)

    # Transform to color optical frame
    pts_c = (R @ pts_d.T).T + t  # (N, 3)

    X_c, Y_c, Z_c = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    # Filter points behind camera
    in_front = Z_c > 0.01
    X_c, Y_c, Z_c = X_c[in_front], Y_c[in_front], Z_c[in_front]

    # Project to color image
    u_c = (fx_c * X_c / Z_c + cx_c).astype(np.int32)
    v_c = (fy_c * Y_c / Z_c + cy_c).astype(np.int32)
    depth_vals = Z_c.astype(np.float32)

    # Bounds check
    valid = (u_c >= 0) & (u_c < w_c) & (v_c >= 0) & (v_c < h_c)
    u_c, v_c, depth_vals = u_c[valid], v_c[valid], depth_vals[valid]

    # Z-buffer: keep closest depth at each pixel
    # Sort by depth descending so closest overwrites
    order = np.argsort(-depth_vals)
    u_c, v_c, depth_vals = u_c[order], v_c[order], depth_vals[order]
    aligned[v_c, u_c] = depth_vals

    return aligned


def extract(bag_path: str, output_dir: str, subsample: int,
            color_topic: str, depth_topic: str,
            color_info_topic: str, depth_info_topic: str):
    from mcap_ros2.reader import read_ros2_messages

    out = Path(output_dir)
    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Read extrinsics for depth alignment
    R, t = read_extrinsics(bag_path)

    color_intr = None
    depth_intr = None
    frame_count = 0
    saved = 0
    manifest = []

    pending_color = None
    pending_color_ts = None

    print(f"\nStreaming {bag_path} ...")
    print(f"  color: {color_topic}")
    print(f"  depth: {depth_topic}")
    print(f"  subsample: every {subsample}th pair")

    MAX_DT_NS = 50_000_000  # 50 ms

    all_topics = [color_topic, depth_topic, color_info_topic, depth_info_topic]
    for msg in read_ros2_messages(bag_path, topics=all_topics):
        topic = msg.channel.topic
        ts = msg.log_time_ns

        if topic == color_info_topic and color_intr is None:
            ros_msg = msg.ros_msg
            K = ros_msg.k
            color_intr = {
                "fx": float(K[0]), "fy": float(K[4]),
                "cx": float(K[2]), "cy": float(K[5]),
                "width": int(ros_msg.width),
                "height": int(ros_msg.height),
                "distortion_model": str(ros_msg.distortion_model),
            }
            print(f"Color intrinsics: fx={color_intr['fx']:.2f} fy={color_intr['fy']:.2f} "
                  f"cx={color_intr['cx']:.2f} cy={color_intr['cy']:.2f} "
                  f"({color_intr['width']}x{color_intr['height']})")
            continue

        if topic == depth_info_topic and depth_intr is None:
            ros_msg = msg.ros_msg
            K = ros_msg.k
            depth_intr = {
                "fx": float(K[0]), "fy": float(K[4]),
                "cx": float(K[2]), "cy": float(K[5]),
                "width": int(ros_msg.width),
                "height": int(ros_msg.height),
            }
            print(f"Depth intrinsics: fx={depth_intr['fx']:.2f} fy={depth_intr['fy']:.2f} "
                  f"cx={depth_intr['cx']:.2f} cy={depth_intr['cy']:.2f} "
                  f"({depth_intr['width']}x{depth_intr['height']})")
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

            if color_intr is None or depth_intr is None:
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

            # Reproject depth into color camera frame
            aligned = align_depth_to_color(depth_m, depth_intr, color_intr, R, t)

            depth_path = depth_dir / f"{stem}.npy"
            np.save(depth_path, aligned)

            manifest.append({
                "rgb": f"rgb/{stem}.png",
                "sensor_depth": f"depth/{stem}.npy",
            })
            saved += 1

            if saved % 50 == 0:
                print(f"  Saved {saved} frames (aligned) ...")

            pending_color = None
            pending_color_ts = None

    print(f"\nTotal paired frames: {frame_count}")
    print(f"Saved (subsample={subsample}): {saved}")

    # Write intrinsics (color camera, since depth is now aligned to color)
    if color_intr:
        intr_path = out / "intrinsics.json"
        with open(intr_path, "w") as f:
            json.dump(color_intr, f, indent=2)
        print(f"Intrinsics saved: {intr_path}")

    # Write manifest
    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"\nExtraction complete:")
    print(f"  RGB:      {rgb_dir}  ({saved} files)")
    print(f"  Depth:    {depth_dir}  ({saved} files, aligned to color)")
    print(f"  Manifest: {manifest_path}  ({saved} entries)")

    # Quick stats on aligned depth
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
        print(f"\nAligned depth stats (first {n_check} frames):")
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
    parser.add_argument("--color-info-topic", type=str,
                        default="/camera/color/camera_info")
    parser.add_argument("--depth-info-topic", type=str,
                        default="/camera/depth/camera_info")
    args = parser.parse_args()

    extract(args.bag, args.output_dir, args.subsample,
            args.color_topic, args.depth_topic,
            args.color_info_topic, args.depth_info_topic)


if __name__ == "__main__":
    main()
