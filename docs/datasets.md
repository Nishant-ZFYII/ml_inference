---
layout: default
title: Datasets
---

# Datasets

Every evaluation in this work runs on one of four frame sets. None of them are large by computer vision standards — the corridor evaluation has 459 frames, and the glass corridor has 121. The data scale is constrained by the project's deployment scope: the robot collects data in a single building, and the depth sensor's failure mode is specific to that building's materials.

---

## Corridor evaluation data (459 frames)

**Location:** `ml_pipeline/corridor_eval_data/`

**Source:** Orbbec Femto Bolt RGB-D + RPLiDAR S2, recorded on the Traxxas Maxx 4S platform in a university corridor. The original rosbag (`rgbd_imu_20260228_003828_0.mcap`, ~8.1 GB) was recorded at 30 FPS; frames were subsampled to ~3 FPS for evaluation.

**Contents:**

| Subdirectory | Count | Format | Notes |
|---|---|---|---|
| `rgb/` | 485 | `.png`, 1280×720 | Raw RGB frames |
| `depth/` | 485 | `.npy`, 720×1280, float32 | ToF depth in meters, 0 = invalid |
| `da3_depth/` | 459 | `.npy`, 720×1280, float32 | DA3-Small predictions (zero-shot) |

The 26-frame gap between RGB (485) and DA3 (459) comes from frames where DA3 inference was not run. All evaluations use the 459-frame DA3-aligned subset.

**Naming convention:** `00000.png` through `00484.png` (5-digit zero-padded).

**Student predictions:** Not pre-computed. V5, V6, V7, and V9 predictions are generated via live inference in the evaluation and video generation scripts.

---

## Glass corridor (121 frames)

**Location:** `~/maps/glass_corridor_frames/`

**Source:** Same platform, different corridor section with a full glass wall on one side. Extracted from rosbag `rgbd_imu_20260302_173610`.

**Contents:**

| Subdirectory | Count | Format | Notes |
|---|---|---|---|
| `rgb/` | 121 | `.png`, 1280×720 | Raw RGB frames |
| `depth/` | 121 | `.npy`, varies | ToF depth (some at 576×640 native) |
| `da3_depth/` | 121 | `.npy` | DA3-Small predictions |

**Student predictions:** Pre-computed V5, V6, V7 `.npy` files live in a separate directory: `~/maps/glass_corridor_student_results/{depth_v5_vivek,depth_v6,depth_v7}/`. V9 is always computed via live inference.

**Naming convention:** `frame_0000_t30.0s.png` (timestamped, starting at t=30s to skip the initial static period).

**Why this dataset matters:** The glass wall creates a different failure mode than the polished floor. The ToF sensor does not just lose pixels — it sometimes returns erroneous depth from specular reflections off the glass. This tests whether the fusion pipeline degrades gracefully or introduces false obstacles.

---

## bag_213831 (150 frames)

**Location:** `~/maps/bag_213831_frames/`

**Source:** Jetson-mounted recording, different corridor section, different lighting. Extracted from rosbag `rgbd_imu_20260302_213831`.

**Contents:**

| Subdirectory | Count | Format | Notes |
|---|---|---|---|
| `rgb/` | 150 | `.png`, 1280×720 | Raw RGB frames |
| `depth/` | 150 | `.npy` | ToF depth |
| `da3_depth/` | 150 | `.npy` | DA3-Small predictions |

**Student predictions:** Pre-computed V5, V6, V7 and DA3 in `~/maps/bag_213831_student_results/`. V9 via live inference.

**Naming convention:** `frame_0000_t0.0s.png` (timestamped, 2-second spacing).

---

## NYU Depth V2

**Source:** HuggingFace datasets (`sayakpaul/nyu_depth_v2`), loaded via `dataset/nyu_loader.py`.

**Size:** 47,584 training + 654 test RGB-D pairs (640×480).

**Usage:** Primary training set for V1–V6. Evaluation set for NYU RMSE numbers.

**Gotcha:** The HuggingFace loader uses a loading script, which `datasets >= 4.0` refuses to execute by default. `requirements.txt` pins `datasets < 4.0`.

---

## LILocBench

**Source:** Bonn Indoor Localization Benchmark (corridor subset). Extracted via `extract_lilocbench.py`.

**Usage:** Fine-tuning set for V7 and V9. Evaluation set for corridor RMSE.

---

## SUN RGB-D + DIODE

**Usage:** V6 pretraining (diverse indoor / outdoor depth). Not used directly in evaluation — only as the pretraining-stage corpus prior to NYU fine-tuning.

---

## Data not in the repository

The following are git-ignored and must be obtained separately for full reproducibility:

| Asset | Size | Location | How to get it |
|---|---|---|---|
| Model weights (`best_depth_v{5,6,7,9}.pt`) | ~61 MB each | `hpc_outputs/` | HuggingFace dataset repo |
| Corridor eval data | ~1.2 GB | `corridor_eval_data/` | HuggingFace dataset repo |
| Raw rosbags | ~10+ GB each | `~/rosbags/` | Available on request |
| Glass corridor frames | ~200 MB | `~/maps/glass_corridor_frames/` | HuggingFace dataset repo |
| bag_213831 frames | ~250 MB | `~/maps/bag_213831_frames/` | HuggingFace dataset repo |
| Demo videos | ~700 MB | `/media/nishant/SeeGayt2/demo_videos/` | Generated via scripts |

The HuggingFace dataset repository provides model weights and evaluation data. Video outputs can be regenerated from the evaluation data using the scripts in this repo.
