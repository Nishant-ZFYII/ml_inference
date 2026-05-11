---
layout: default
title: Project Overview
---

# Bootstrap Perception Under Hardware Depth Failure

<video controls width="100%" preload="metadata" poster="{{ '/assets/architecture.png' | relative_url }}">
  <source src="{{ '/assets/videos/v9_6panel.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the embedded video. <a href="{{ '/assets/videos/v9_6panel.mp4' | relative_url }}">Download MP4</a>.
</video>

*Six-panel grid showing the V9 (Lighthouse) production deployment on the 459-frame corridor evaluation set. **Top row:** RGB input · raw Femto Bolt ToF depth (note the dead-pixel pattern across the floor) · zero-shot DA3-Small reference depth. **Bottom row:** V9 raw inference · confidence-gated fusion of ToF and DA3 (the foundation-model baseline) · confidence-gated fusion of ToF and V9 (the production deployment realization, consumed directly by the local costmap).*

A mobile robot navigating an indoor corridor finds that its Orbbec Femto Bolt ToF camera returns valid depth on **20.3% of pixels** (459-frame measurement, verified end-to-end inside the Docker container — see [Docker](docker)). The polished floor, glass walls, and out-of-range surfaces kill the rest. The LiDAR still works, but it scans a 2D plane — it does not see chairs, tabletops, or torsos.

This project asks whether a single RGB camera, running a learned monocular depth model, can fill the gap left by the dead ToF pixels. Not to replace structured-light depth — to make the robot's costmap dense enough to navigate safely.

The position this work defends, in one sentence: **monocular depth alone cannot replace ToF, but it is an unexpectedly strong fusion partner.** Fusing LiDAR with depth recovers +55 % occupied costmap cells in narrow corridors and turns a robot that cannot see most of its environment into one that can.

---

## What this site documents

| Page | Content |
|---|---|
| [Architecture](architecture) | Mermaid diagrams: two-repo split, student model, runtime fusion (with median-scale calibration), training loss |
| [Concepts](concepts/) | Bootstrap perception, scale calibration, confidence-gated fusion, four-layer sensing, specification and deployment, knowledge distillation — the ideas the rest of the site assumes |
| [Hardware](hardware) | Femto Bolt, RPLiDAR S2, Jetson Orin Nano, Traxxas Maxx 4S — the deployment platform |
| [Model lineage (V1 → V9)](models/) | One page per training iteration with what changed, why, results, and verdict |
| [Training Pipeline](training) | Loss functions, teacher ensemble, the V1-V9 narrative |
| [Evaluation](evaluation) | Corridor depth metrics, costmap ablation, FPR decomposition |
| [Calibration Study](calibration) | Reviewer-requested experiment on affine alignment sensitivity |
| [Decisions and tradeoffs](decisions) | Specification vs deployment realization, deferred APE evaluation, INT8 calibration scope, V9 specialist tradeoff |
| [Deployment](deployment) | ONNX export, TensorRT, Jetson benchmarks, ROS 2 integration |
| [Datasets](datasets) | Frame counts, formats, naming conventions, hosting status |
| [Demo Videos](videos) | Comparison videos across all models and datasets |
| [Docker](docker) | One-command reproducibility, verified build |

---

## Headline results

| Capability | Number | Source |
|---|---|---|
| Best general indoor model (V5) | 0.572 m NYU val RMSE | [V5](models/v5-deployment-aug) |
| Best NYU model (V6) | 0.519 m NYU val RMSE | [V6](models/v6-sun-diode-pretrain) |
| **Production corridor specialist (V9)** | **0.382 m LILocBench RMSE, 9 / 10 Gazebo success** | [V9](models/v9-corridor-specialist) |
| DA3-Small zero-shot on Jetson | 218 FPS / 4.6 ms / 2.7 GB | TensorRT FP16 at 308×308 |
| Costmap recovery (L+D vs L-only) | +55% occupied cells | [Evaluation](evaluation) |
| ToF sensor dead-pixel rate (corridor) | 79.7% (verified in container) | [Docker](docker) |
| End-to-end Docker reproducibility | smoke + 459-frame eval | [Docker](docker) |

---

## Honest caveats

1. **APE / SLAM evaluation deferred to future work.** A preliminary measurement was confounded by mismatched rosbag playback rates; matched-rate re-evaluation is identified as future work. [Details](decisions#2-ape--slam-result-deferred-to-future-work).
2. **V9 is a corridor specialist.** Worse than V3 on NYU. The tradeoff is intentional and disclosed. [Details](decisions#4-v9-is-a-corridor-specialist-not-a-strictly-better-model).
3. **The 5.2% FPR is not free.** Decomposed into model hallucinations (49%), sensor-fill artifacts (35%), inflation artifacts (18%). [Details](decisions#5-the-52-fpr-is-not-free).
4. **INT8 calibration in `export_trt.py` is a stub.** Defaults to random noise. FP16 numbers are real; INT8 numbers are not validated. [Details](decisions#3-int8-calibration-in-export_trtpy-is-a-stub).
5. **The fusion pipeline has two complementary implementations.** The formal specification (used for evaluation and reporting) computes affine alignment explicitly; the deployment realization (used for on-vehicle inference) applies the same principle via per-pixel substitution within the embedded compute budget. The two are operationally equivalent in the deployment regime. [Details](decisions#1-specification-and-deployment-realization).

---

## Quick start (Docker, verified)

The Dockerfile and compose stack were built and run end-to-end on a Linux host. The smoke test and 459-frame corridor evaluation both pass.

```bash
# Build (CPU image, ~6.3 GB; needs --network=host on networks with restricted DNS)
docker build --network=host -t ml-inference .

# Smoke — forward pass on random tensor, confirms model loads
docker compose run --rm smoke-test
# → "Model forward pass OK: depth (1, 1, 240, 320), seg (1, 6, 240, 320)"
# → "PyTorch 2.11.0+cu130, OpenCV 4.13.0, NumPy 2.2.6"

# Corridor depth evaluation — 459 frames, prints per-bin RMSE
docker compose run --rm eval-corridor
# → RMSE 1.366 m (raw, uncalibrated; consistent with the reported 1.418 m within run-to-run variance)
# → 79.7% sensor dead-pixel rate confirms the bootstrap-perception premise

# Calibration sensitivity sweep
docker compose run --rm calibration

# Comparison videos (CPU; takes ~20 min)
docker compose run --rm grid-videos
```

Model weights (`hpc_outputs/*.pt`) and evaluation data (`corridor_eval_data/`) are volume-mounted, not baked into the image. HuggingFace hosting is on the to-do list — see [Datasets](datasets) for the current status.

---

## Stack

PyTorch 2.0+ · timm (EfficientViT-B1) · ONNX · TensorRT FP16 · OpenCV-headless · Jetson Orin Nano 8GB · Orbbec Femto Bolt · RPLiDAR S2 · ROS 2 Humble · Nav2 · SLAM Toolbox · NYU Greene HPC (L40S)

---

NYU MS Mechatronics & Robotics — Nishant Pushparaju, Vivek Mattam, Prof. Aliasghar Arab
