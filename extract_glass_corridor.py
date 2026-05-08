#!/usr/bin/env python3
"""
Extract RGB + sensor depth frames from the glass corridor db3 bag
at key timestamps. Streams frames one at a time to avoid OOM.
"""

import sqlite3
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image as RosImage


def extract_frames(db3_path, output_dir, start_sec=0.0, end_sec=999.0,
                   subsample=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    (out / "depth_viz").mkdir(exist_ok=True)

    conn = sqlite3.connect(db3_path)
    cur = conn.cursor()

    cur.execute("SELECT id, name, type FROM topics")
    topic_map = {}
    for tid, name, mtype in cur.fetchall():
        topic_map[tid] = (name, mtype)

    color_tid = depth_tid = None
    for tid, (name, _) in topic_map.items():
        if name == "/camera/color/image_raw":
            color_tid = tid
        elif name == "/camera/depth/image_raw":
            depth_tid = tid

    if color_tid is None or depth_tid is None:
        print("ERROR: Missing color or depth topics")
        conn.close()
        return []

    cur.execute("SELECT MIN(timestamp) FROM messages")
    t0 = cur.fetchone()[0]

    start_ns = int(t0 + start_sec * 1e9)
    end_ns = int(t0 + end_sec * 1e9)

    # Get timestamps only first (lightweight)
    print("Indexing color frame timestamps...")
    cur.execute(
        "SELECT timestamp FROM messages "
        "WHERE topic_id = ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp",
        (color_tid, start_ns, end_ns))
    color_times = [r[0] for r in cur.fetchall()]
    print(f"  {len(color_times)} color frames in range")

    print("Indexing depth frame timestamps...")
    cur.execute(
        "SELECT timestamp FROM messages "
        "WHERE topic_id = ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp",
        (depth_tid, start_ns, end_ns))
    depth_times = [r[0] for r in cur.fetchall()]
    print(f"  {len(depth_times)} depth frames in range")

    # Select which frames to extract
    selected_color_ts = [color_times[i] for i in range(0, len(color_times), subsample)]
    print(f"  Extracting {len(selected_color_ts)} subsampled frames")

    manifest = []
    saved = 0

    for ct in selected_color_ts:
        # Find closest depth timestamp
        closest_dt = min(depth_times, key=lambda x: abs(x - ct))
        if abs(closest_dt - ct) > 50_000_000:
            continue

        elapsed = (ct - t0) / 1e9

        # Fetch color frame
        cur.execute(
            "SELECT data FROM messages WHERE topic_id = ? AND timestamp = ?",
            (color_tid, ct))
        color_data = bytes(cur.fetchone()[0])

        # Fetch depth frame
        cur.execute(
            "SELECT data FROM messages WHERE topic_id = ? AND timestamp = ?",
            (depth_tid, closest_dt))
        depth_data = bytes(cur.fetchone()[0])

        # Deserialize
        cmsg = deserialize_message(color_data, RosImage)
        dmsg = deserialize_message(depth_data, RosImage)

        ch, cw, cenc = cmsg.height, cmsg.width, cmsg.encoding
        dh, dw, denc = dmsg.height, dmsg.width, dmsg.encoding

        if cenc == "rgb8":
            rgb = np.frombuffer(cmsg.data, dtype=np.uint8).reshape(ch, cw, 3)
        elif cenc == "bgr8":
            bgr = np.frombuffer(cmsg.data, dtype=np.uint8).reshape(ch, cw, 3)
            rgb = bgr[:, :, ::-1].copy()
        else:
            print(f"  Skip: unknown encoding {cenc}")
            continue

        if denc == "16UC1":
            depth_m = (np.frombuffer(dmsg.data, dtype=np.uint16)
                       .reshape(dh, dw).astype(np.float32) / 1000.0)
        elif denc == "32FC1":
            depth_m = np.frombuffer(dmsg.data, dtype=np.float32).reshape(dh, dw)
        else:
            print(f"  Skip: unknown depth encoding {denc}")
            continue

        fname = f"frame_{saved:04d}_t{elapsed:.1f}s"

        Image.fromarray(rgb).save(out / "rgb" / f"{fname}.png")
        np.save(out / "depth" / f"{fname}.npy", depth_m)

        valid = (depth_m > 0.1) & (depth_m < 5.0)
        depth_norm = np.zeros((dh, dw), dtype=np.uint8)
        if valid.any():
            dmin, dmax = depth_m[valid].min(), depth_m[valid].max()
            if dmax > dmin:
                depth_norm[valid] = (255 * (1.0 - (depth_m[valid] - dmin)
                                    / (dmax - dmin))).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        depth_color[~valid] = [0, 0, 0]
        cv2.imwrite(str(out / "depth_viz" / f"{fname}.png"), depth_color)

        zero_pct = float((depth_m == 0).sum() * 100.0 / depth_m.size)
        valid_pct = float(valid.sum() * 100.0 / depth_m.size)
        mean_d = float(depth_m[valid].mean()) if valid.any() else 0.0

        manifest.append({
            "frame": fname,
            "elapsed_s": round(elapsed, 1),
            "depth_zero_pct": round(zero_pct, 1),
            "depth_valid_pct": round(valid_pct, 1),
            "depth_mean_m": round(mean_d, 2),
        })

        print(f"  [{saved}] t={elapsed:.1f}s  "
              f"valid={valid_pct:.1f}% zero={zero_pct:.1f}% "
              f"mean_d={mean_d:.2f}m")
        saved += 1

    conn.close()

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {saved} frames to {out}/")
    return manifest


if __name__ == "__main__":
    bag = ("/home/nishant/rosbags/rgbd_imu_20260302_173610/"
           "rgbd_imu_20260302_173610_0.db3")
    out = "/home/nishant/maps/glass_corridor_frames"

    extract_frames(bag, out, start_sec=30.0, end_sec=90.0, subsample=15)
