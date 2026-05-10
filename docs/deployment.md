---
layout: default
title: Deployment
---

# Deployment: HPC to Jetson

The training pipeline runs on NYU HPC (A100/L40S GPUs, 64 GB VRAM). The deployment target is a Jetson Orin Nano 8GB strapped to a Traxxas Maxx 4S RC car. The gap between those two environments is the subject of this page.

---

## Export pipeline

`export_trt.py` takes a PyTorch checkpoint and produces three things:

1. **ONNX model** — portable, runs anywhere with `onnxruntime`. Used for debugging and CPU-only validation.
2. **TensorRT FP16 engine** — half-precision inference, 2× speedup over FP32 on Jetson, negligible accuracy loss.
3. **TensorRT INT8 engine** — 4× speedup over FP32, but requires a calibration dataset to maintain accuracy.

```bash
# Full export (ONNX + FP16 + INT8)
python export_trt.py --checkpoint hpc_outputs/best_depth_v9.pt

# ONNX-only (no TRT, runs on laptop)
python export_trt.py --checkpoint hpc_outputs/best_depth_v9.pt --skip-trt

# INT8 with real calibration images
python export_trt.py --checkpoint hpc_outputs/best_depth_v9.pt \
    --calib-images corridor_eval_data/rgb/
```

**INT8 caveat:** The default calibration set in `export_trt.py` is `np.random.rand(...)`. The export succeeds and the engine runs, but the quantization quality is poor. For production INT8, always supply `--calib-images` with real images from the target domain.

---

## Jetson benchmarks

`benchmark_jetson.py` measures latency, GPU memory, and depth RMSE across all exported engines.

| Engine | Resolution | Latency (ms) | GPU Mem (MB) | Depth RMSE (m) |
|---|---|---|---|---|
| DA3-Small FP16 | 308×308 | 4.6 | 2,700 | 0.522 |
| V9 Student FP16 | 240×320 | 2.1 | 890 | 0.382* |
| V9 Student INT8 (random calib) | 240×320 | 1.4 | 640 | — |

*V9 RMSE is on LILocBench, not the corridor set. On the corridor set, V9 is 1.418 m (see [Evaluation](evaluation)).

DA3-Small at 218 FPS (FP16, 308×308) is the production runtime model. The V9 student runs alongside it as a corridor-specialized companion. Both fit in the Orin Nano's 8 GB shared memory with room for the ROS 2 stack.

**What `benchmark_jetson.py` does not measure:** segmentation mIoU. The mIoU column is initialized but never populated (lines 150, 186). Latency and depth RMSE are real; mIoU is not.

---

## ROS 2 integration

The student model plugs into the NCHSB ROS 2 stack through three nodes:

**Student TRT Node** → Subscribes to `/camera/rgb`, runs the TensorRT engine, publishes `/student/depth` and `/student/segmentation`.

**Depth Fusion Node** → Subscribes to `/student/depth` and `/camera/depth` (ToF). For each pixel: if the ToF sensor reports valid depth above a confidence threshold, output the ToF reading. Otherwise, output the student prediction. Publishes `/perception/fused_depth`.

**Point Cloud XYZ Node** → Back-projects `/perception/fused_depth` into a 3D point cloud using the camera intrinsics. Publishes `/perception/fused_depth_points`, which Nav2's local costmap observes.

The fusion logic in the Depth Fusion Node matches the training target exactly: trust hardware where it works, fall back to learned depth where it does not.

---

## What the paper says vs what the code does

The paper formulates fusion as affine alignment: \\(d_{metric} = \alpha \cdot d_{pred} + \beta\\), where \\(\alpha, \beta\\) are estimated from a calibration set.

The live Depth Fusion Node does not implement affine alignment. It implements confidence-gated substitution: use ToF if valid, use student otherwise. No \\(\alpha\\), no \\(\beta\\), no per-pixel scaling.

This gap is documented in the obsidian vault under "Paper vs Runtime Truth." The distinction matters because the affine formulation is cleaner for the paper's math, while the substitution approach is simpler and more robust at runtime (no risk of a bad calibration multiplying errors across the frame).

Both approaches arrive at the same costmap: dense depth from a mixture of hardware and learned sources. The evaluation scripts (`run_costmap_ablation.py`) use per-frame median scaling (the affine flavor). The runtime uses substitution. The [Calibration Study](calibration) quantifies the gap.

---

## Reproducing on a Jetson

```bash
# On the Jetson (JetPack 5.x, TensorRT pre-installed)
scp best_depth_v9.pt vortex@<jetson-ip>:~/
ssh vortex@<jetson-ip>

python export_trt.py --checkpoint best_depth_v9.pt
python benchmark_jetson.py --engine-dir exported/
```

The exported engines are platform-specific. An engine built on an L40S will not run on a Jetson Orin Nano. Always export on the target device.
