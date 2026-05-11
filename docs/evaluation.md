---
layout: default
title: Evaluation
---

# Evaluation

Every reported number in this work traces back to a JSON file in `results/`. This page documents each evaluation protocol, what it measures, and the methodological details required for reproduction.

---

## Pixel-level depth evaluation (459 corridor frames)

The core depth comparison: how accurately does each model predict metric depth, measured against the surviving ToF pixels as ground truth?

**Protocol:**
1. Extract 459 frames from the corridor rosbag where both RGB and DA3 predictions exist.
2. For each frame, compute the valid ToF mask (pixels where the sensor reports depth > 0 with confidence above threshold).
3. For each depth source (DA3-Small, V5, V6, V7, V9), apply median-scale alignment: \\(s = \text{median}(d_{tof}) / \text{median}(d_{pred})\\) on valid pixels, then \\(d_{aligned} = s \cdot d_{pred}\\).
4. Compute RMSE, MAE, AbsRel, and \\(\delta < 1.25\\) on the valid mask only.

**Results (per-pixel, median-scaled, 459 frames):**

| Method | RMSE | MAE | AbsRel | δ < 1.25 |
|---|---|---|---|---|
| DA3-Small | 0.522 m | 0.268 m | 0.087 | 53.4% |
| V5 student | 0.634 m | 0.371 m | 0.121 | 42.1% |
| V6 student | 0.601 m | 0.342 m | 0.112 | 45.8% |
| V7 student | 0.589 m | 0.331 m | 0.108 | 47.2% |
| V9 student | 1.418 m | 0.892 m | 0.291 | 17.2% |

> **Verification note (2026-05-10):** running V9 raw (uncalibrated, no median-scale alignment) inside the Docker container on the same 459 frames gives RMSE 1.366 m / MAE 0.929 m / AbsRel 1.457 / δ<1.25 13.5 %. The 1.418 m number above is with median scaling, which improves absolute scale but not per-pixel accuracy. Both numbers are reported transparently: the table reports the calibration-aligned metric used in formal evaluation, and the Docker number is the raw model output. See [calibration](calibration) for the full sensitivity analysis.

DA3-Small dominates on per-pixel accuracy. This is expected — DA3-Small is a 300M-parameter foundation model running zero-shot, while V9 is a 5.31M-parameter specialist trained on a different corridor benchmark. V9 wins on LILocBench RMSE (0.382 m vs DA3's ~0.5 m), but on our Femto Bolt corridor, the foundation model is stronger per-pixel.

The position this work defends is not that V9 outperforms DA3 on raw per-pixel depth accuracy. It is that either model, fused with the surviving ToF pixels, produces a costmap dense enough to navigate, and that the student model is small enough to run alongside DA3-Small on the deployment hardware.

Source: `results/pixel_fusion.json`

---

## Range-stratified depth analysis

Depth error is not uniform across distance. Near-range (0–1.5 m) matters most for obstacle avoidance. Far-range (3–6 m) matters for path planning.

| Method | Near (0–1.5 m) | Mid (1.5–3 m) | Far (3–6 m) |
|---|---|---|---|
| DA3-Small | 0.145 m | 0.503 m | 1.305 m |
| V9 student | 1.642 m | 1.461 m | 1.012 m |

DA3 degrades at distance (1.305 m far-range RMSE), while V9 is more uniform but worse everywhere. The near-range gap is the most operationally significant: 0.145 m vs 1.642 m for objects within arm's reach of the robot.

Source: `results/nearrange_safety.json`

---

## Costmap ablation (Table III)

The costmap ablation answers: does adding depth to the LiDAR-only costmap help the robot navigate? We configure Nav2's local costmap under six setups and replay the same corridor bag through each.

**Configurations:**

| Config | Sensors | Inflation | Notes |
|---|---|---|---|
| Baseline | LiDAR only | 0.090 m | Stock Nav2, robot barely fits |
| A1 | Depth only | 0.177 m | No LiDAR at all |
| A3 | LiDAR + Depth | 0.177 m (fixed) | Headline L+D configuration |
| A4 | LiDAR + Depth | Adaptive | Inflation shrinks in narrow passages |
| A5 | LiDAR + Depth | 0.192 m (large) | Conservative: wider safety margin |
| A6 | LiDAR + Depth | 0.197 m (conservative) | Most conservative tested |

**Results (459 frames):**

| Config | IoU vs Baseline | Detection Rate | FPR | Mean Occupied Cells |
|---|---|---|---|---|
| Baseline | 1.000 | 100% | 0.0% | 2,295 |
| A1 | 1.000 | 100% | 0.0% | — |
| A3 (L+D) | 0.379 | 100% | 5.2% | 3,546 |
| A4 | 0.279 | 76.7% | 5.2% | — |
| A5 | 0.379 | 100% | 5.2% | — |
| A6 | 0.279 | 76.7% | 5.2% | — |

The headline: **L → L+D adds +55 % occupied cells** (2,295 → 3,546). The IoU against baseline drops because the additional depth observations fill regions the LiDAR never saw — walls above the scan plane, chair legs, people's torsos. That is new geometry, not noise (mostly).

### LILocBench secondary evaluation

The same costmap-ablation protocol applied to the LILocBench `dynamics_0` corridor split (a different building, different camera, different scan plane height) produces a larger relative recovery:

| Configuration | Mean occupied cells | Δ vs LiDAR-only |
|---|---|---|
| L (LiDAR only) | baseline | — |
| L + D (LiDAR + Depth fusion) | +110 % | corridor depth filling |
| D only (Depth only, no LiDAR) | +219 % | dense depth dominates the costmap |

LILocBench corridors are taller and narrower than the deployment corridor, so the LiDAR-only baseline is sparser (the scan plane misses more vertical obstacles). Depth fusion therefore contributes more relative volume on LILocBench (+110 %) than on the deployment corridor (+55 %). The D-only configuration (+219 %) is reported for completeness; it is not the recommended deployment configuration because the LiDAR is the more reliable backbone where it can see.

Source: `results/costmap_ablation/lilocbench_dynamics_0/summary.json`.

**The 5.2% false positive rate is real.** Decomposition from `fpr_audit.py`:
- 49.3% are model hallucinations (predicting depth where nothing exists)
- 34.6% are sensor-invalid fills (model labels obstacle where ToF would have said invalid)
- 18.1% are inflation artifacts (geometry correct, inflation radius too aggressive)

Source: `results/costmap_ablation/corridor/summary.json`, `results/fpr_audit.json`

---

## Glass corridor evaluation (121 frames)

The glass corridor is a harder scene: one wall is entirely glass. The ToF sensor loses even more pixels on glass than on polished floor. This dataset tests whether the depth models degrade gracefully when the sensor failure mode shifts.

**What was evaluated:**
- RGB, Sensor depth, DA3-Small, V5, V6, V7 (V9 generated via live inference)
- Pixel-level fusion for each model
- All outputs saved as individual and grid comparison videos

**Key difference from corridor_eval_data:** Glass corridor has pre-computed DA3 `.npy` files and student V5/V6/V7 predictions saved as `.npy` in `~/maps/glass_corridor_student_results/`. V9 is always computed via live inference.

---

## bag_213831 evaluation (150 frames)

A third dataset from a Jetson-mounted recording session. Different lighting, different corridor section, 150 frames at 2-second intervals.

**What was evaluated:** Same protocol as glass corridor — all models, individual and grid comparison videos.

**Key difference:** This dataset has pre-computed student predictions in `~/maps/bag_213831_student_results/`. DA3 predictions also pre-computed.

---

## Cross-scene calibration sensitivity

See the dedicated [Calibration Study](calibration) page.

---

## Reproducing the evaluation

```bash
# Corridor depth evaluation (all models)
python eval_corridor_depth.py \
    --checkpoint hpc_outputs/best_depth_v9.pt \
    --manifest corridor_eval_data/manifest.jsonl \
    --imagenet-norm

# Costmap ablation
python run_costmap_ablation.py \
    --eval-dir corridor_eval_data/ \
    --output-dir results/costmap_ablation/corridor/

# FPR decomposition
python fpr_audit.py

# Paper statistics aggregation
python compute_paper_stats.py
```

Or via Docker:
```bash
docker compose run eval-corridor
docker compose run calibration
```
