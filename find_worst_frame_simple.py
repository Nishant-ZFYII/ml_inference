#!/usr/bin/env python3
"""
Find the frame with highest dead-pixel percentage using raw CDR parsing.
Avoids rclpy dependency for speed.
"""
import sqlite3
import numpy as np
import struct

BAG = "/home/nishant/rosbags/rgbd_imu_20260302_172822/rgbd_imu_20260302_172822_0.db3"

conn = sqlite3.connect(BAG)
cur = conn.cursor()

cur.execute("SELECT id, name, type FROM topics")
depth_tid = None
for tid, name, ttype in cur.fetchall():
    if name == '/camera/depth/image_raw' and 'Image' in ttype:
        depth_tid = tid
        print(f"Depth topic: id={tid}, name={name}, type={ttype}")
        break

if depth_tid is None:
    print("ERROR: No depth topic found!")
    exit(1)

cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (depth_tid,))
total_msgs = cur.fetchone()[0]
print(f"Total depth messages: {total_msgs}")

cur.execute(
    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (depth_tid,))

worst_pct = 0
worst_ts = 0
first_ts = None
count = 0
pcts = []

for ts, raw_data in cur:
    if first_ts is None:
        first_ts = ts

    data = bytes(raw_data)
    try:
        # CDR: 4 bytes encapsulation header, then std_msgs/Header, then Image fields
        # For 16UC1 depth images from RealSense: height*width*2 bytes at end
        # The data field starts after: header (stamp + frame_id) + height + width + encoding + ...
        # Easier: just grab the last height*width*2 bytes since we know the format
        # Standard approach: parse the Image message fields
        
        # Skip CDR header (4 bytes)
        offset = 4
        
        # Header.stamp (sec: uint32 + nanosec: uint32)
        offset += 8
        
        # Header.frame_id (string: 4-byte length + data + padding)
        frame_id_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4 + frame_id_len
        # Align to 4 bytes
        offset = (offset + 3) & ~3
        
        # height: uint32
        height = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # width: uint32
        width = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # encoding: string
        enc_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        encoding = data[offset:offset+enc_len-1].decode()  # -1 for null terminator
        offset += enc_len
        offset = (offset + 3) & ~3
        
        # is_bigendian: uint8
        offset += 1
        # padding to align step (uint32)
        offset = (offset + 3) & ~3
        
        # step: uint32
        step = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # data: sequence<uint8> (4-byte length prefix)
        data_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        img_bytes = data[offset:offset+data_len]
        
        if encoding == '16UC1':
            arr = np.frombuffer(img_bytes, dtype=np.uint16).reshape(height, width)
            depth_m = arr.astype(np.float32) / 1000.0
        elif encoding == '32FC1':
            arr = np.frombuffer(img_bytes, dtype=np.float32).reshape(height, width)
            depth_m = arr
        else:
            continue
        
        dead = ((depth_m <= 0) | (depth_m > 4.0) | np.isnan(depth_m)).sum()
        pct = dead * 100.0 / depth_m.size
        elapsed = (ts - first_ts) / 1e9
        pcts.append((elapsed, pct))
        
        if pct > worst_pct:
            worst_pct = pct
            worst_ts = elapsed
        
        count += 1
        if count % 200 == 0:
            print(f"  {count}/{total_msgs} frames, worst: {worst_pct:.1f}% at t={worst_ts:.1f}s")
            
    except Exception as e:
        count += 1
        continue

conn.close()

print(f"\n=== RESULTS (bag 172822) ===")
print(f"Scanned {count} depth frames")
print(f"Worst dead pixel frame: t={worst_ts:.1f}s, dead={worst_pct:.1f}%")
if pcts:
    all_p = [p for _, p in pcts]
    print(f"Mean dead pixels: {np.mean(all_p):.1f}%")
    print(f"Median dead pixels: {np.median(all_p):.1f}%")
    print(f"P90 dead pixels: {np.percentile(all_p, 90):.1f}%")
    print(f"P99 dead pixels: {np.percentile(all_p, 99):.1f}%")
    
    top10 = sorted(pcts, key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 worst frames:")
    for t, p in top10:
        print(f"  t={t:.1f}s  dead={p:.1f}%")
