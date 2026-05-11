<div align="center">

# ml_inference

**Vortex training pipeline — knowledge distillation for monocular depth + 6-class semantic segmentation, designed for ROS 2 / Nav2 deployment on a Jetson Orin Nano.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-17-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16%20%7C%20INT8-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-reproducible-2496ED.svg?logo=docker&logoColor=white)](#docker-reproducibility)
[![Project Blog](https://img.shields.io/badge/blog-Bootstrap%20Perception-1f425f.svg)](https://nishant-zfyii.github.io/ml_inference/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.28890-555555.svg)](https://arxiv.org/abs/2603.28890)

<img src="docs/assets/gifs/v9_6panel.gif" alt="Six-panel video showing V9 deployment on the corridor evaluation set: RGB, raw ToF, DA3 reference, V9 inference, ToF+DA3 fusion, ToF+V9 fusion" width="92%"/>

*V9 (Lighthouse) production deployment on the 459-frame corridor evaluation set. **Top row:** RGB input · raw Femto Bolt ToF depth (showing the **79.7 % dead pixel rate** that motivates the fusion) · zero-shot DA3-Small reference depth. **Bottom row:** V9 raw inference · ToF + DA3 fusion (foundation-model baseline) · ToF + V9 fusion (production deployment realization, consumed by the local costmap). Verified end-to-end inside the Docker container — see [Docker Reproducibility](#docker-reproducibility).*

</div>

---

## Overview

This repository is the off-board half of a bootstrap-perception system for indoor robot navigation under hardware depth failure. It contains the training pipeline that produces compact monocular depth and 6-class semantic segmentation student models distilled from large foundation teachers, the offline evaluation scaffold that measures their accuracy, and the export tooling that converts trained checkpoints to TensorRT engines for on-vehicle deployment.

The system addresses a specific operational problem. The deployment camera (Orbbec Femto Bolt Time-of-Flight) returns valid depth on approximately 22 % of pixels in the target environment; the remaining 78 % are lost to reflective surfaces (polished floors, glass walls, mid-field returns beyond sensor range). The bootstrap-perception strategy uses the surviving valid pixels to anchor a learned monocular depth prediction to metric scale, then fuses the two signals per pixel to produce a dense depth signal that the local costmap can consume. The position this work supports:

> Monocular depth estimation cannot replace structured-light or Time-of-Flight depth in indoor navigation, but it is an effective fusion partner. Fusing LiDAR with confidence-gated learned depth recovers approximately 55 % more occupied costmap cells in narrow corridors than LiDAR alone, and produces dense geometry where the hardware sensor returns invalid pixels.

The runtime ROS 2 nodes (Depth Fusion, Class Costmap, Student TRT, YOLO TRT), the Nav2 configuration, and the live deployment harness on the Traxxas Maxx 4S testbed live in the sibling repository `NCHSB`. The two repositories are deliberately decoupled along the off-board / on-vehicle boundary; their dependency stacks, release cadences, and audiences differ.

| Component | This repository (`ml_inference`) | Sibling repository (`NCHSB`) |
|---|---|---|
| Teacher inference on HPC | DA3-Metric-Large + YOLOv8 + SAM2-Large | — |
| Student training (V1 → V9 lineage) | All 9 configurations | — |
| Offline evaluation pipelines | Per-pixel depth, costmap ablation, calibration sensitivity | — |
| Export toolchain | ONNX → TensorRT FP16 / INT8, Jetson micro-benchmarks | — |
| Runtime ROS 2 nodes | — | Depth Fusion, Class Costmap, Student TRT, YOLO TRT |
| Navigation configuration | — | `nav2_params_rc.yaml`, controllers, EKF, launch files |
| Deployment platform | — | Live integration on Traxxas Maxx 4S |
| Simulation harness | — | Gazebo Fortress worlds, rosbag replay |

---

## Project highlights

- **Bootstrap perception architecture.** The Orbbec Femto Bolt Time-of-Flight sensor returns valid depth on 22.21 % of pixels in the corridor evaluation set; the surviving pixels anchor a learned monocular prediction to metric scale via per-frame median alignment. The runtime fusion (per-pixel confidence-gated substitution) preserves hardware geometry where available and substitutes the calibrated learned signal where the sensor failed.
- **Corridor specialist with closed-loop validation.** The V9 student (Lighthouse) achieves 0.382 m LILocBench corridor RMSE and matches ground-truth-depth navigation performance in Gazebo Fortress closed-loop trials (9 / 10 success at 10 seeds, 0 collisions, time-to-goal within 0.22 s of the ground-truth baseline).
- **Foundation-model deployment headline.** DA3-Small zero-shot inference benchmarks at 218 FPS / 4.6 ms / 2.7 GB on the Jetson Orin Nano (TensorRT FP16, 308 × 308 input). The V-series students complement the foundation model as corridor-domain specialists rather than replacing it.
- **Specification-aligned training target.** The hybrid depth supervision in `models/losses.py:HybridDepthLoss` selects DA3 teacher depth where available and falls back to ToF measurement otherwise. The frame-level training rule and the per-pixel deployment fusion implement the same supervision principle (prefer hardware ground truth where available; fall back to the learned signal where not) at the granularity appropriate to each stage.
- **Auditable results.** Every reported number traces to a JSON file under `results/`. Quantitative claims are reproducible from the offline evaluation pipeline (see Quick start below). The deferred APE / SLAM evaluation is documented in [§ Limitations and disclosures](#limitations-and-disclosures).

---

## Architecture

<img src="assets/architecture.png" alt="System architecture: HPC training (left) and Jetson runtime (right)" width="92%"/>

Two halves of one system. The dashed boundary is where this repo ends and `NCHSB` begins.

**Off-board (this repo, on NYU HPC):**
1. `teacher_infer/run_da3.py` produces metric-depth supervision per frame from DA3-Metric-Large (`metric = focal · raw / 300`).
2. `teacher_infer/run_sam2.py` produces 6-class segmentation labels by combining YOLOv8 detections (person, furniture), SAM2-Large mask refinement, and geometric heuristics for floor / wall / glass.
3. `teacher_infer/build_manifest.py` emits a `manifest.jsonl` linking every RGB to its DA3 depth, SAM2 seg, ToF depth, and ToF confidence.
4. `train.py` trains an EfficientViT-B1 student (5.31 M params) with a hybrid depth loss + cross-entropy seg + edge-aware smoothness, optionally Kendall-uncertainty-weighted.
5. `export_trt.py` exports ONNX → TensorRT FP16 / INT8 engines for the Jetson.

**On-board (NCHSB, on the Jetson):** Student TRT Node consumes RGB and publishes `/student/depth` + `/student/segmentation`. Depth Fusion Node combines the student output with the surviving ToF pixels and publishes `/perception/fused_depth`. Point Cloud XYZ Node back-projects that into `/perception/fused_depth_points`, which Nav2's local costmap consumes.

---

## The student lineage (V1 → V9)

The repository's `main` reflects the V4-V9 production codebase. The earlier V1-V3 baseline (MobileNetV3-Small + DA2-Large) is preserved verbatim under [`archive/v1-v3-baseline/`](archive/v1-v3-baseline/) and reproducible exactly via the `v1-v3-baseline` git tag.

Corridor RMSE is reported on two different sets: **LILocBench** (Bonn corridor benchmark, used for fine-tuning V7 and V9) and **Femto Bolt** (our own indoor corridor recordings, used as the deployment-truth set). They are not interchangeable — LILocBench is shorter and structurally simpler.

| Version | Codename | Backbone | Teacher | NYU RMSE | LILocBench RMSE | Femto Bolt RMSE | Configuration change | Outcome |
|---|---|---|---|---|---|---|---|---|
| V1 | Compass | MobileNetV3-Small | DA2-Large | 75.37 m | — | — | Initial distillation | Baseline. Unit-space mismatch (DA2 outputs relative depth) dominates the result. |
| V2 | Sextant | MobileNetV3-Small | DA2-Large | — | — | — | Kendall log-σ² clamp experiments | Diagnostic; ruled out loss weighting as the dominant cause of V1's result. |
| V3 | Anchor | MobileNetV3-Small | DA3-Large | 1.160 m | — | — | berHu loss + Kendall weighting + two-rate optimizer | First metric-scale predictions. Recipe set at V3 and held constant through V9. |
| V4 | Pivot | **EfficientViT-B1** | DA3-Large | 0.774 m | — | 1.373 m | Encoder substitution | −33 % NYU RMSE at fixed recipe. Architecture inherited by V5–V9. |
| V5 | **Atlas** | EfficientViT-B1 | DA3-Large | **0.572 m** | — | 2.186 m | Deployment-targeted augmentation pipeline | **Production: general indoor.** Largest single-step NYU improvement (−26 %). |
| V6 | **Cornerstone** | EfficientViT-B1 | DA3-Large | **0.519 m** | — | 2.158 m | Multi-domain pretraining (SUN+DIODE) → NYU fine-tune | **Production: fine-tuning base.** Best NYU result in lineage. |
| V7 | Tunnel | EfficientViT-B1 | DA3-Large | 1.315 m | 0.445 m | 1.982 m | V5 → LILocBench fine-tune | First corridor specialization. NYU regression characteristic of single-domain fine-tuning. |
| V8 | Confluence | EfficientViT-B1 | DA3-Large | 0.592 m | — | 2.266 m | Joint NYU + LILocBench training from V5 | Joint-domain training does not improve corridor performance over the V5 baseline (Pareto-dominated by V9). |
| **V9** | **Lighthouse** | EfficientViT-B1 | DA3-Large | 1.553 m | **0.382 m** | 1.589 m | V6 → LILocBench fine-tune | **Production: corridor specialist.** Closed-loop validated against ground-truth depth in simulation. |

Recipe details and per-version provenance are on the [model lineage page](https://nishant-zfyii.github.io/ml_inference/models/) of the project blog. Each version (V1 through V9) has a dedicated page documenting the configuration change, the experimental outcome, and the design rationale.

### Selection guidance

| Use case | Recommended checkpoint |
|---|---|
| General-purpose indoor depth estimation | **V5 (Atlas)** — `NYU 0.572 m`, balanced segmentation mIoU 63.7 % |
| Fine-tuning base for additional domain specialists | **V6 (Cornerstone)** — `NYU 0.519 m`, best NYU result; recommended initialization |
| Production corridor specialist (closed-loop validated) | **V9 (Lighthouse)** — `LILocBench 0.382 m`, 9 / 10 Gazebo success matching ground-truth depth |
| Maximum-throughput foundation-model inference on Jetson | **DA3-Small** zero-shot (218 FPS / 4.6 ms / 2.7 GB) — production runtime; V-series students complement it as domain specialists |

---

## Results

Numbers below are reproduced from `paper_stats.json` and the per-experiment JSON files in [`results/`](results/). Every row has an `n` and a 95 % CI in the source files; only the headline is shown here.

### DA3-Small zero-shot on NYU val (Table VII, teacher row)

| Metric | Value | n |
|---|---|---|
| RMSE (m) | 0.513 ± 0.038 | 290 |
| AbsRel | 0.124 ± 0.008 | 290 |
| δ < 1.25 (%) | 85.2 ± 1.6 | 290 |
| δ < 1.25² (%) | 95.3 ± 0.7 | 290 |
| Latency (ms, PyTorch) | 65.3 | — |

Source: `results/nyu_da3_da3-small_val.json`. The 218 FPS / 4.6 ms / 2.7 GB headline is the same model under TensorRT FP16 at 308 × 308 on Jetson Orin Nano (separate Jetson benchmark; not from this PyTorch run).

### Pixel-level depth on the corridor (459 frames)

| Method | RMSE (overall) | RMSE (near 0-1.5 m) | RMSE (mid 1.5-3 m) | RMSE (far 3-6 m) | δ < 1.25 (%) |
|---|---|---|---|---|---|
| Sensor (ToF) | 0.000 | 0.000 | 0.000 | 0.000 | 100 |
| **DA3-Small** | **0.522** | 0.145 | 0.503 | 1.305 | 53.4 |
| V9 student | 1.418 | 1.642 | 1.461 | 1.012 | 17.2 |

Sensor RMSE is zero by construction — sensor pixels are the "ground truth" against which everything else is scored on the surviving 22.21 % valid mask. Source: `results/pixel_fusion.json`.

DA3-Small dominates V9 on this metric. V9 wins the *corridor specialist* benchmark on LILocBench (above) but is not the right pick for general per-pixel accuracy.

### Costmap recovery (Table III, n = 459 frames)

| Config | IoU | Detection rate (%) | FPR (%) | Inflation radius (m) | Timing (ms) |
|---|---|---|---|---|---|
| Baseline | 1.000 | 100.0 | 0.0 | 0.090 | 16.5 |
| A1 (depth only) | 1.000 | 100.0 | 0.0 | 0.177 | 63.3 |
| A3 (L+D, fixed inflation) | 0.379 | 100.0 | 5.2 | 0.177 | 133.8 |
| A4 (L+D, adaptive) | 0.279 | 76.7 | 5.2 | 0.165 | 132.6 |
| A5 (L+D, large inflation) | 0.379 | 100.0 | 5.2 | 0.192 | 206.2 |
| A6 (L+D, conservative) | 0.279 | 76.7 | 5.2 | 0.197 | 189.9 |

Source: `results/costmap_ablation/corridor/summary.json`. Headline result: **L → L+D adds +55 % occupied cells** in narrow corridors (2 295 → 3 546 mean occupied; `paper_stats.json:table_iv`).

<img src="assets/lilocbench_panel.png" alt="LILocBench dynamics_0: RGB, LiDAR-only costmap, LiDAR+depth costmap" width="92%"/>

LILocBench `dynamics_0`: 10 pedestrians moving through the scene. L+D recovers pedestrian bodies that L misses entirely.

<img src="assets/costmap_replay.png" alt="Live Nav2 costmap during corridor replay with green DA3 contributions and blue V9 contributions" width="92%"/>

Live Nav2 costmap during corridor replay. Green = pixels filled by DA3, blue = pixels filled by V9, white = surviving ToF.

---

## Limitations and disclosures

<img src="assets/fpr_decomposition.png" alt="False positive rate decomposition: 49.3% hallucination, 34.6% sensor-invalid fill, 18.1% inflation artifacts" width="60%"/>

The 5.2 % false-positive rate measured in the L + D configuration is a real operational cost of fusion. Decomposition (`results/fpr_audit.json`):

- **49.3 %** model hallucinations (depth model predicts an obstacle where none exists)
- **34.6 %** sensor-invalid fill (depth model assigns obstacle status to a pixel where ToF would have reported invalid)
- **18.1 %** costmap inflation artifacts (geometry correct, inflation radius too aggressive)

Additional disclosures relevant to reproducing or quoting reported numbers:

1. **APE / SLAM evaluation deferred to future work.** A preliminary measurement reported a 73 % APE improvement (1.23 m vs 4.63 m) but used asymmetric rosbag playback rates between configurations (1.0× for LiDAR-only, 0.5× for fused-depth to allow inference completion). Slower playback runs SLAM Toolbox loop closure more aggressively, which reduces APE independently of the depth-fusion contribution. The preliminary numbers are retained in `paper_stats.json:table_vi` for traceability but are not cited as a reported result. Matched-playback re-evaluation on the deployment hardware is identified as future work.
2. **INT8 calibration in `export_trt.py` defaults to random tensors.** The plumbing for INT8 quantization is implemented and functional; the *accuracy* of the resulting INT8 engine without calibration data is not validated. For deployment INT8, supply `--calib-images <dir>` pointing at representative corridor frames. All reported Jetson runtime numbers (218 FPS / 4.6 ms / 2.7 GB on DA3-Small) use FP16, not INT8.
3. **`benchmark_jetson.py` reports depth RMSE only.** The segmentation mIoU column is initialized but not populated. Latency and depth-RMSE measurements from this script are valid; mIoU values from this script should not be used.
4. **V9 is specialized for corridor environments, not a general-purpose model.** NYU val RMSE (1.553 m) substantially exceeds V5 (0.572 m) and V6 (0.519 m) — a documented consequence of single-domain fine-tuning under standard catastrophic-forgetting dynamics. The tradeoff is intentional: V9 is the recommended checkpoint when the deployment domain is restricted to corridors; for general indoor scenes, V5 or V6 is preferred.

---

## Quick start

### Local CPU validation (laptop)

Validates the full pipeline end-to-end on a tiny subset before pushing to HPC. Downloads ~2.8 GB of NYU Depth V2 the first time.

```bash
conda create -n vortex_ml python=3.10 -y && conda activate vortex_ml
pip install -r requirements.txt

# Smoke test — 2 epochs, batch 4, 50 frames, CPU
python train.py --epochs 2 --batch-size 4 --device cpu --data-limit 50

# ONNX-only export (no TRT on laptop)
python export_trt.py --checkpoint checkpoints/best.pt --skip-trt
```

### HPC training (NYU Torch)

```bash
ssh <NetID>@login.torch.hpc.nyu.edu
cd $HOME && git clone https://github.com/Nishant-ZFYII/ml_inference.git ml_pipeline
bash ml_pipeline/setup_hpc.sh    # creates $SCRATCH/conda_envs/nchsb_ml

# Verify partitions for your account
sinfo
# Edit train.slurm + teacher_infer/teacher_infer.slurm if --partition or --gres differ

# Teacher inference on NYU val
sbatch ml_pipeline/teacher_infer/teacher_infer.slurm

# Train V4-V9-style student
sbatch ml_pipeline/train.slurm

# Distillation eval (Table IV equivalent)
python eval_distillation.py \
    --checkpoint $SCRATCH/checkpoints/best.pt \
    --manifest   $SCRATCH/nyu_teacher_data/manifest.jsonl
```

### Corridor specialisation (V7 / V9)

Once you have V5 or V6 weights, fine-tune on LILocBench:

```bash
# 1. Local: extract corridor frames from rosbag (Linux box where the bag lives)
python -m teacher_infer.extract_corridor_bag \
    --bag /home/<you>/rosbags/<your_corridor_bag>.mcap \
    --output-dir corridor_eval_data --subsample 5

# 2. tar + scp corridor_eval_data/ to $SCRATCH on HPC

# 3. Re-run teachers, build manifest, fine-tune
sbatch ml_pipeline/eval_corridor.slurm
```

### Jetson deployment

```bash
# Build engines on the Jetson (or any TRT-capable host)
python export_trt.py --checkpoint best.pt              # FP32 + FP16 + INT8

# Latency / GPU-mem / depth-RMSE micro-benchmark
python benchmark_jetson.py --engine-dir exported/
```

The engine then plugs into `Student TRT Node` in `NCHSB`.

---

## Repository layout

```
ml_inference/
├── README.md                       ← this file
├── config.py                       ← central Config dataclass
├── requirements.txt                ← Python deps
├── setup_hpc.sh                    ← one-time HPC env setup
│
├── train.py                        ← student training loop
├── train.slurm                     ← SLURM job for default training
├── train_iter6.slurm               ← Kendall uncertainty + per-task ckpts
├── train_iter7.slurm               ← TUM RGB-D experiment
├── train_iter7b_b2.slurm           ← EfficientViT-B2 ablation
│
├── eval_distillation.py            ← student vs teacher (RMSE / AbsRel / δ / mIoU)
├── eval_corridor_da3.py            ← DA3-Small zero-shot on the corridor
├── eval_corridor_depth.py          ← Student depth on the corridor
├── eval_corridor_v4.slurm          ← SLURM for V4-era corridor eval
├── eval_corridor.slurm             ← SLURM for B1/B2 corridor eval
├── eval_nyu_da3.{py,slurm}         ← DA3-Small zero-shot on NYU val
├── eval_nearrange_safety.py        ← Near-range (0-1.5 m) safety analysis
├── fpr_audit.py                    ← FPR origin classification
├── temporal_consistency.py         ← Costmap stability across frames
├── compute_paper_stats.py          ← Aggregates per-experiment JSONs into paper_stats.json
│
├── calibration_sensitivity.py      ← Reviewer-response calibration ablation
├── costmap_builder.py              ← Costmap construction for ablation
├── inflation.py                    ← Inflation logic
├── run_costmap_ablation.py         ← Full costmap ablation harness
├── extract_lilocbench.py           ← LILocBench frame extraction
├── corridor_sam2_seg.slurm         ← SAM2 corridor seg labels
├── costmap_ablation.slurm          ← SLURM for full ablation
│
├── export_trt.py                   ← ONNX + TensorRT FP32/FP16/INT8
├── benchmark_jetson.py             ← TRT engine micro-benchmark
├── print_model_shapes.py           ← Encoder feature-map verification utility
│
├── generate_paper_figures.py       ← Figures from rosbag + checkpoints
├── generate_demo_videos.py         ← Individual model comparison videos (1280×720)
├── generate_grid_video.py          ← Synchronized 2×3 / 2×4 grid comparison videos
├── generate_corridor_missing.py    ← OOM-safe sequential model inference for corridor
├── create_full_comparison.py       ← Side-by-side comparison panels
├── create_paper_fig2.py            ← Fig. 2 generator
├── extract_bag_frames.py           ← Frame extraction from rosbag2 (.db3)
├── extract_corridor_frames.py      ← Corridor-specific extraction
├── extract_glass_corridor.py       ← Glass-corridor scene extraction
├── find_worst_frame{,_simple}.py   ← Worst-case-frame finders
├── run_da3_on_frames.py            ← DA3 inference on raw frames
├── run_depth_comparison.py         ← Per-frame depth comparison
├── run_student_evaluation.py       ← Aggregate student eval
├── da3_glass_corridor.py           ← DA3 on glass corridor scene
│
├── pipeline_lilocbench.slurm       ← End-to-end LILocBench pipeline
│
├── dataset/
│   ├── nyu_loader.py               ← NYU Depth V2 (HuggingFace datasets, pinned <4)
│   ├── corridor_loader.py          ← Corridor data loader
│   ├── lilocbench_loader.py        ← Bonn LILocBench loader
│   ├── tum_loader.py               ← TUM RGB-D loader
│   └── label_remapper.py           ← 894 → 40 → 6 class remapping
│
├── models/
│   ├── student.py                  ← EfficientViT-B1 backbone-agnostic + dual decoders
│   └── losses.py                   ← Hybrid depth (ToF/DA3) + CE seg + edge smoothness
│
├── teacher_infer/
│   ├── run_da3.py                  ← DA3-Metric-Large depth teacher
│   ├── run_sam2.py                 ← YOLO-seeded SAM2-Large + geometric labeler
│   ├── verify_teacher_output.py    ← Pre-run scale/sanity gate
│   ├── build_manifest.py           ← Emits manifest.jsonl
│   ├── extract_corridor_bag.py     ← Local bag → frame folder + manifest
│   ├── prep_tum.py                 ← TUM RGB-D preparation
│   ├── teacher_infer.slurm         ← SLURM for NYU teachers
│   └── teacher_infer_tum.slurm     ← SLURM for TUM teachers
│
├── results/                        ← Versioned evaluation outputs
│   ├── paper_stats.json            ← Aggregated Tables III–VI
│   ├── nyu_da3_da3-small_val.json  ← DA3-Small NYU eval
│   ├── pixel_fusion.json           ← Per-frame fusion comparison
│   ├── nearrange_safety.json       ← Near-range RMSE breakdown
│   ├── fpr_audit.json              ← FPR origin decomposition
│   ├── temporal_consistency.json   ← Frame-to-frame stability
│   └── costmap_ablation/           ← Per-config inflation radii + per-frame metrics
│
├── Dockerfile                      ← ML inference / evaluation container
├── docker-compose.yml              ← Multi-service reproducibility harness
│
├── docs/                           ← Jekyll site (nishant-zfyii.github.io/ml_inference)
│   ├── _config.yml                 ← Jekyll config
│   ├── index.md                    ← Project overview
│   ├── training.md                 ← V1 → V9 training lineage
│   ├── evaluation.md               ← Depth metrics, costmap ablation
│   ├── calibration.md              ← Reviewer-requested calibration study
│   ├── deployment.md               ← ONNX / TRT export, Jetson benchmarks
│   ├── datasets.md                 ← Data inventory and provenance
│   ├── videos.md                   ← Demo video generation pipeline
│   └── docker.md                   ← Docker usage guide
│
├── archive/
│   ├── README.md                   ← Archive index
│   └── v1-v3-baseline/             ← Frozen V1-V3 codebase (MobileNetV3 + DA2)
│
└── assets/                         ← Figures referenced by this README
```

---

## Docker reproducibility

The fastest path from clone to results. Verified end-to-end on 2026-05-10 — `docker build`, `smoke-test`, and `eval-corridor` (459 frames, V9, CPU) all pass. The full verification log lives in [`docs/docker.md`](docs/docker.md).

Model weights and evaluation data are volume-mounted, not baked into the image. Image is ~6.3 GB (PyTorch wheel dominates).

```bash
# Build — pass --network=host on networks with restricted DNS (e.g. NYU campus)
docker build --network=host -t ml-inference .

# Smoke test (no data needed)
docker compose run --rm smoke-test
# → Model forward pass OK: depth (1, 1, 240, 320), seg (1, 6, 240, 320)

# Corridor depth evaluation (~3 min CPU on 459 frames)
docker compose run --rm eval-corridor
# → RMSE 1.366 m, sensor dead-pixel rate 79.7%

# Calibration sensitivity experiment
docker compose run --rm calibration

# Demo videos and grid comparisons
docker compose run --rm demo-videos
docker compose run --rm grid-videos
```

| Service | What it runs | GPU needed |
|---|---|---|
| `smoke-test` | Model architecture + forward pass check | No |
| `eval-corridor` | V9 depth on 459 corridor frames | No (CPU), faster with GPU |
| `calibration` | Calibration sensitivity sweep (N = 1–100 frames) | No |
| `demo-videos` | Individual model comparison videos (1280×720, XVID) | No |
| `demo-videos-gpu` | Same, GPU-accelerated inference | Yes |
| `grid-videos` | Synchronized 2×3 / 2×4 grid comparison videos | No |

For GPU passthrough: `docker compose run demo-videos-gpu` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

---

## Engineering blog

The [project blog](https://nishant-zfyii.github.io/ml_inference/) walks through each component in long-form: the [model lineage](https://nishant-zfyii.github.io/ml_inference/models/) with one page per training configuration (V1 through V9, what changed and why), the [architecture diagrams](https://nishant-zfyii.github.io/ml_inference/architecture) (training, student, runtime fusion as Mermaid), the [costmap ablation](https://nishant-zfyii.github.io/ml_inference/evaluation), the [calibration sensitivity study](https://nishant-zfyii.github.io/ml_inference/calibration), [deployment notes](https://nishant-zfyii.github.io/ml_inference/deployment) for the Jetson runtime, the [demo video generation pipeline](https://nishant-zfyii.github.io/ml_inference/videos), and a [decisions log](https://nishant-zfyii.github.io/ml_inference/decisions) documenting the relationship between the formal specification and the deployment realization. The README is the reference card; the blog is the technical journal.

---

## Reproducibility notes

**Bag location.** The corridor rosbag (`rgbd_imu_20260228_003828_0.mcap`, ~8.1 GB) lives on the local Linux extraction host, not on HPC. Frame extraction (`teacher_infer/extract_corridor_bag.py`) runs locally; the resulting `corridor_eval_data/` directory is `tar`/`scp`'d to `$SCRATCH/corridor_eval_data/` on HPC where the SLURM jobs read it. See `eval_corridor.slurm:21-29` for the exact handoff.

**HPC environment.**

| Setting | Value |
|---|---|
| Cluster | NYU Torch HPC (`login.torch.hpc.nyu.edu`) |
| Partition | `l40s_public` (default; verify with `sinfo`) |
| GPU | `gpu:l40s:1` |
| Account | `torch_pr_742_general` |
| Module | `anaconda3/2025.06` |
| Conda env | `$SCRATCH/conda_envs/nchsb_ml` (created by `setup_hpc.sh`) |

**Pinned dataset library.** The NYU Depth V2 HF dataset still uses a loading script, which means `datasets >= 4.0` will refuse to load it. `requirements.txt` pins `datasets >= 2.14, < 4.0`.

**Recovering the V1-V3 codebase.** Two paths, both reproducible:

```bash
# As browsable files at the top of main:
ls archive/v1-v3-baseline/

# As a complete checkout of the V1-V3 working tree:
git checkout v1-v3-baseline
```

---

## Acknowledgements

- **Vivekananda Swamy Mattam** — ROS 2 stack, hardware integration, training pipeline V3 onward, technical writing.
- **Nishant Pushparaju** — EfficientViT-B1 backbone (V4 turning point), HPC training infrastructure, Gazebo closed-loop validation, Jetson deployment.
- **Prof. Aliasghar Arab** — faculty advisor, NYU Tandon MAE.
- **NYU HPC** for compute on the Torch cluster (`torch_pr_742_general`).
- **Foundation models** used as teachers and runtime: Depth Anything V3 (DA3-Metric-Large + DA3-Small), SAM2-Large, YOLOv8.
- **Datasets**: NYU Depth V2 (Silberman et al.), SUN RGB-D, DIODE, LILocBench, TUM RGB-D.
- **Vortex project vault** (separate repository) — source of the project's design rationale and lessons.

---

## License

MIT — see [`LICENSE`](LICENSE).
