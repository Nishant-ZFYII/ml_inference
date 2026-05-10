---
layout: default
title: Project Overview
---

# Bootstrap Perception Under Hardware Depth Failure

**Paper:** [arXiv:2603.28890](https://arxiv.org/abs/2603.28890) · **Venue:** IEEE Robotics and Automation Letters (RA-L)

A mobile robot navigating university corridors discovers that its Orbbec Femto Bolt ToF depth camera returns valid pixels on only 22.2% of the frame. The polished floor, glass walls, and out-of-range surfaces kill the rest. The LiDAR still works, but it scans a thin 2D plane — it cannot see tabletops, chairs, or people's torsos.

This project asks whether a single RGB camera, running a learned monocular depth model, can fill the gap left by the dead ToF pixels — not to replace structured-light depth, but to make the robot's costmap dense enough to navigate safely.

The answer: monocular depth alone is not accurate enough to replace the ToF sensor. But fusing it with the surviving ToF pixels recovers +55% occupied costmap cells in narrow corridors and turns a robot that cannot see most of its environment into one that can.

---

## What this site documents

| Page | What it covers |
|------|----------------|
| [Training Pipeline](training) | V1 through V9 student lineage, loss functions, backbone choices, what worked and what failed |
| [Evaluation](evaluation) | Corridor depth metrics, costmap ablation, glass corridor, cross-scene testing |
| [Calibration Study](calibration) | Reviewer-requested experiment: how calibration set size and scene-specificity affect affine alignment accuracy |
| [Deployment](deployment) | ONNX export, TensorRT quantization, Jetson Orin Nano benchmarks, ROS 2 integration |
| [Datasets](datasets) | Every frame set used in the paper, where it lives, how many frames, what's extracted |
| [Demo Videos](videos) | Full-length comparison videos across all models and datasets |
| [Docker](docker) | Containerized reproducibility: smoke test to full evaluation in one command |

---

## The system in one diagram

Two repositories, one perception loop. This site documents the left half — the off-board ML training and evaluation pipeline (`ml_inference`). The right half is the ROS 2 navigation stack (`NCHSB`), which runs on the Jetson and consumes the student model at runtime.

**Off-board (this repo, NYU HPC):**
1. DA3-Metric-Large produces dense metric depth per frame.
2. YOLO + SAM2-Large produce 6-class semantic segmentation labels.
3. An EfficientViT-B1 student (5.31M params) distills both into a single network.
4. ONNX → TensorRT export produces FP16/INT8 engines for the Jetson.

**On-board (NCHSB, Jetson Orin Nano):**
Student TRT Node publishes depth + segmentation. Depth Fusion Node combines student output with surviving ToF pixels. Point Cloud XYZ Node back-projects the fused depth into a 3D point cloud. Nav2's local costmap consumes it.

The training target and the runtime fusion encode the same prior: trust the ToF sensor where it reports valid depth, fall back to the learned model everywhere else.

---

## Headline results

| Capability | Number | Context |
|---|---|---|
| Best general indoor model (V5) | 0.572 m RMSE on NYU val | EfficientViT-B1, deployment augmentations |
| Best corridor specialist (V9) | 0.382 m RMSE on LILocBench | V6 → LILocBench fine-tune; 9/10 Gazebo closed-loop success |
| DA3-Small on Jetson | 218 FPS / 4.6 ms / 2.7 GB | TensorRT FP16, 308×308 |
| Costmap recovery (L+D vs L-only) | +55% occupied cells | Narrow corridor, 459 frames |
| ToF sensor dead pixel rate | 77.8% invalid | Polished floor + glass + out-of-range |

---

## Honest caveats

Some things you should know before using these results:

1. **The APE/SLAM claim was dropped.** An earlier draft showed 73% APE improvement. It was removed because rosbag playback rates differed between configurations — faster playback artificially flatters localization. The invalidated numbers are preserved in `paper_stats.json` for completeness but should not be cited.

2. **V9 is a corridor specialist, not a universal model.** Its NYU val RMSE (1.553 m) is much worse than V5 (0.572 m) or V6 (0.519 m). Specialization is real and useful, but this is not a strictly better model.

3. **The 5.2% FPR in L+D configurations is not free.** Decomposition: 49.3% model hallucinations, 34.6% sensor-invalid-fill artifacts, 18.1% inflation artifacts.

4. **INT8 calibration in `export_trt.py` defaults to random noise.** For production INT8, supply `--calib-images <dir>` with real corridor frames.

---

## Quick start (Docker)

```bash
# Pull or build
docker build -t ml-inference .

# Smoke test — verifies model loads and forward pass works
docker compose run smoke-test

# Corridor depth evaluation (needs model weights + eval data mounted)
docker compose run eval-corridor

# Calibration sensitivity experiment
docker compose run calibration

# Grid comparison videos
docker compose run grid-videos
```

Model weights and evaluation data are not baked into the Docker image. Download them from HuggingFace and place them in `hpc_outputs/` and `corridor_eval_data/` before running.

---

## Technical stack

- **Training:** PyTorch 2.0+, timm (EfficientViT-B1), HuggingFace datasets
- **Teachers:** DA3-Metric-Large, SAM2-Large, YOLOv8
- **Export:** ONNX, TensorRT (FP16/INT8)
- **Evaluation:** NumPy, OpenCV, Matplotlib
- **Runtime:** ROS 2 Humble, Nav2, SLAM Toolbox
- **Hardware:** Jetson Orin Nano 8GB, Orbbec Femto Bolt, RPLiDAR S2
- **Compute:** NYU Torch HPC (L40S GPUs)

---

NYU MS Mechatronics & Robotics — Nishant Pushparaju, Vivek Mattam, Prof. Aliasghar Arab
