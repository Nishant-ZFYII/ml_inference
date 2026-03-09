#!/usr/bin/env python3
"""
Scan entire rosbag for the frame with the highest dead-pixel percentage.
"""
import sqlite3
import numpy as np
import sys

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

BAG = "/home/nishant/rosbags/rgbd_imu_20260302_172822/rgbd_imu_20260302_172822_0.db3"

conn = sqlite3.connect(BAG)
cur = conn.cursor()

cur.execute("SELECT id, name, type FROM topics")
topic_map = {}
for tid, name, ttype in cur.fetchall():
    topic_map[tid] = (name, ttype)
    if 'depth' in name.lower():
        print(f"  Topic {tid}: {name} ({ttype})")

depth_tid = None
for tid, (name, ttype) in topic_map.items():
    if name == '/camera/depth/image_raw' and 'Image' in ttype:
        depth_tid = tid
        break

if depth_tid is None:
    print("No depth topic found")
    sys.exit(1)

cur.execute(
    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (depth_tid,))

worst_pct = 0
worst_ts = 0
first_ts = None
count = 0
all_pcts = []

for ts, data in cur:
    if first_ts is None:
        first_ts = ts

    try:
        msg = deserialize_message(bytes(data), Image)
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        depth_m = arr.astype(np.float32) / 1000.0

        dead = ((depth_m <= 0) | (depth_m > 4.0)).sum()
        pct = dead * 100.0 / depth_m.size
        elapsed = (ts - first_ts) / 1e9
        all_pcts.append((elapsed, pct))

        if pct > worst_pct:
            worst_pct = pct
            worst_ts = elapsed
            worst_data = (msg, depth_m)

        count += 1
        if count % 100 == 0:
            print(f"  Scanned {count} frames, worst so far: {worst_pct:.1f}% at t={worst_ts:.1f}s")
    except Exception as e:
        continue

conn.close()

print(f"\n=== RESULTS ===")
print(f"Total depth frames: {count}")
print(f"Worst dead pixel frame: t={worst_ts:.1f}s, dead={worst_pct:.1f}%")
print(f"Mean dead pixel %: {np.mean([p for _,p in all_pcts]):.1f}%")
print(f"Median dead pixel %: {np.median([p for _,p in all_pcts]):.1f}%")

top10 = sorted(all_pcts, key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 worst frames:")
for t, p in top10:
    print(f"  t={t:.1f}s  dead={p:.1f}%")
