#!/usr/bin/env python3
"""
Extract RGB + sensor depth frames from a ROS2 sqlite3 (.db3) rosbag.
Samples at a configurable interval and saves PNG (rgb) + NPY (depth).
"""
import sqlite3
import struct
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_image_cdr(raw_bytes):
    """Parse a sensor_msgs/Image CDR blob, returning (height, width, encoding, img_data).

    Uses a robust approach: parse h/w/encoding from the header, then grab the
    known-size pixel buffer from the tail of the blob to avoid CDR alignment issues.
    """
    data = bytes(raw_bytes)
    offset = 4  # CDR encapsulation
    offset += 8  # Header.stamp
    fid_len = struct.unpack_from('<I', data, offset)[0]
    offset += 4 + fid_len
    offset = (offset + 3) & ~3
    h = struct.unpack_from('<I', data, offset)[0]; offset += 4
    w = struct.unpack_from('<I', data, offset)[0]; offset += 4
    enc_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    enc = data[offset:offset + enc_len - 1].decode()

    # Compute expected pixel data size from encoding
    bpp = {'rgb8': 3, 'bgr8': 3, '16UC1': 2, '32FC1': 4, 'mono8': 1}
    bytes_per_pixel = bpp.get(enc, 3)
    pixel_bytes = h * w * bytes_per_pixel
    step = w * bytes_per_pixel

    img_data = data[-pixel_bytes:]
    return h, w, enc, step, img_data


def extract_frames(bag_path, output_dir, sample_interval=1.0, max_frames=200):
    output_dir = Path(output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(bag_path)
    cur = conn.cursor()

    cur.execute("SELECT id, name, type FROM topics")
    topics = {tid: (name, ttype) for tid, name, ttype in cur.fetchall()}

    rgb_tid = depth_tid = None
    for tid, (name, ttype) in topics.items():
        if name == '/camera/color/image_raw' and 'Image' in ttype:
            rgb_tid = tid
        elif name == '/camera/depth/image_raw' and 'Image' in ttype:
            depth_tid = tid

    if rgb_tid is None or depth_tid is None:
        print(f"ERROR: Missing topics. rgb_tid={rgb_tid}, depth_tid={depth_tid}")
        conn.close()
        return

    print(f"RGB topic id: {rgb_tid}, Depth topic id: {depth_tid}")

    # Pre-load depth frames indexed by timestamp for matching
    print("Loading depth timestamps...")
    cur.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
                (depth_tid,))
    depth_timestamps = [row[0] for row in cur.fetchall()]
    print(f"  {len(depth_timestamps)} depth frames available")

    # Process RGB frames at sample_interval
    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (rgb_tid,))

    first_ts = None
    last_sample_t = -sample_interval
    frame_idx = 0
    manifest = []

    for ts, raw in cur:
        if first_ts is None:
            first_ts = ts

        elapsed = (ts - first_ts) / 1e9
        if elapsed - last_sample_t < sample_interval:
            continue

        last_sample_t = elapsed

        # Parse RGB
        h, w, enc, step, img_bytes = parse_image_cdr(raw)
        if enc == 'rgb8':
            rgb = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif enc == 'bgr8':
            rgb_bgr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)
        else:
            print(f"  Skipping unknown RGB encoding: {enc}")
            continue

        # Find closest depth frame
        best_dt = None
        best_depth_ts = None
        for dts in depth_timestamps:
            dt = abs(dts - ts)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_depth_ts = dts
            if dts > ts:
                break

        if best_depth_ts is None or best_dt > 100_000_000:  # >100ms apart
            continue

        # Fetch depth data
        cur2 = conn.cursor()
        cur2.execute("SELECT data FROM messages WHERE topic_id=? AND timestamp=?",
                     (depth_tid, best_depth_ts))
        depth_row = cur2.fetchone()
        if depth_row is None:
            continue

        dh, dw, denc, dstep, depth_bytes = parse_image_cdr(depth_row[0])
        if denc == '16UC1':
            depth_arr = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(dh, dw)
            depth_m = depth_arr.astype(np.float32) / 1000.0
        elif denc == '32FC1':
            depth_m = np.frombuffer(depth_bytes, dtype=np.float32).reshape(dh, dw)
        else:
            print(f"  Skipping unknown depth encoding: {denc}")
            continue

        # Save
        fname = f"frame_{frame_idx:04d}_t{elapsed:.1f}s"
        cv2.imwrite(str(rgb_dir / f"{fname}.png"), rgb_bgr)
        np.save(str(depth_dir / f"{fname}.npy"), depth_m)

        valid = (depth_m > 0.1) & (depth_m < 5.0) & ~np.isnan(depth_m)
        dead_pct = (~valid).sum() * 100.0 / depth_m.size

        manifest.append({
            "frame": fname,
            "elapsed_s": round(elapsed, 1),
            "rgb_size": [h, w],
            "depth_size": [dh, dw],
            "depth_zero_pct": round(dead_pct, 1),
            "depth_valid_pct": round(valid.sum() * 100.0 / depth_m.size, 1),
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  Extracted {frame_idx} frames (t={elapsed:.1f}s)")

        if frame_idx >= max_frames:
            break

    conn.close()

    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {frame_idx} frames saved to {output_dir}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to .db3 rosbag")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds")
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()
    extract_frames(args.bag_path, args.output_dir, args.interval, args.max_frames)
