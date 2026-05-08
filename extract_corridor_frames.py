#!/usr/bin/env python3
"""
Extract representative RGB frames from bag 172822 for paper figures.
Also runs DA3 + student model (v6) on selected frames for visual comparison.
"""

import sqlite3
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image as RosImage


def extract_key_frames(db3_path, output_dir, timestamps_sec=None, subsample=60):
    """Extract frames at specific timestamps or subsampled."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(exist_ok=True)
    (out / "depth_sensor").mkdir(exist_ok=True)

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

    cur.execute("SELECT MIN(timestamp) FROM messages")
    t0 = cur.fetchone()[0]

    # Get all color timestamps
    cur.execute("SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp", (color_tid,))
    all_color_ts = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp", (depth_tid,))
    all_depth_ts = [r[0] for r in cur.fetchall()]

    if timestamps_sec is not None:
        target_ns = [int(t0 + t * 1e9) for t in timestamps_sec]
        selected = [min(all_color_ts, key=lambda x: abs(x - ns)) for ns in target_ns]
    else:
        selected = all_color_ts[::subsample]

    manifest = []
    for i, ct in enumerate(selected):
        elapsed = (ct - t0) / 1e9
        closest_dt = min(all_depth_ts, key=lambda x: abs(x - ct))

        cur.execute("SELECT data FROM messages WHERE topic_id = ? AND timestamp = ?", (color_tid, ct))
        cdata = bytes(cur.fetchone()[0])
        cur.execute("SELECT data FROM messages WHERE topic_id = ? AND timestamp = ?", (depth_tid, closest_dt))
        ddata = bytes(cur.fetchone()[0])

        cmsg = deserialize_message(cdata, RosImage)
        dmsg = deserialize_message(ddata, RosImage)

        h, w = cmsg.height, cmsg.width
        if cmsg.encoding == "rgb8":
            rgb = np.frombuffer(cmsg.data, dtype=np.uint8).reshape(h, w, 3)
        else:
            bgr = np.frombuffer(cmsg.data, dtype=np.uint8).reshape(h, w, 3)
            rgb = bgr[:, :, ::-1].copy()

        dh, dw = dmsg.height, dmsg.width
        if dmsg.encoding == "16UC1":
            depth_m = np.frombuffer(dmsg.data, dtype=np.uint16).reshape(dh, dw).astype(np.float32) / 1000.0
        else:
            depth_m = np.frombuffer(dmsg.data, dtype=np.float32).reshape(dh, dw)

        fname = f"frame_{i:03d}_t{elapsed:.0f}s"
        Image.fromarray(rgb).save(out / "rgb" / f"{fname}.png")
        np.save(out / "depth_sensor" / f"{fname}.npy", depth_m)

        valid = (depth_m > 0.1) & (depth_m < 5.0)
        manifest.append({
            "frame": fname,
            "elapsed_s": round(elapsed, 1),
            "valid_pct": round(float(valid.sum() * 100 / depth_m.size), 1),
        })
        print(f"  [{i}] t={elapsed:.0f}s  valid={manifest[-1]['valid_pct']:.0f}%")

    conn.close()

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Extracted {len(manifest)} frames")
    return manifest


if __name__ == "__main__":
    bag = ("/home/nishant/rosbags/rgbd_imu_20260302_172822/"
           "rgbd_imu_20260302_172822_0.db3")
    out = "/home/nishant/maps/corridor_key_frames"

    # Extract at key timestamps: start, mid, end, and every 30s
    timestamps = [10, 30, 60, 90, 120, 150, 180, 210, 240, 260]
    extract_key_frames(bag, out, timestamps_sec=timestamps)
