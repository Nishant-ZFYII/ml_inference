---
layout: default
title: Decisions and tradeoffs
---

# Decisions and tradeoffs

This page documents the load-bearing decisions made during the project: the relationship between the formal specification and the deployment realization, the experiments deferred to future work, the operational tradeoffs accepted in the deployment pipeline, and the architectural choices that shaped the production stack. Each section is a self-contained disclosure intended to support reproduction and to clarify the reasoning behind each choice.

---

## 1. Specification and deployment realization

The fusion pipeline is defined at two complementary levels of abstraction:

**Specification (offline evaluation pipeline):** a per-frame affine alignment `s · d_hat + b` between student depth and the surviving ToF pixels, with `s = median(d_tof / d_hat)` over the valid set. Implemented exactly in `calibration_sensitivity.py` and used to compute every quantitative result reported in this work.

**Deployment realization (on-vehicle Depth Fusion Node):** the same fusion principle, expressed in the per-pixel form that fits the embedded compute and latency budget:

```
if confidence(i, j) ≥ 0.5 AND 0.05 m < tof_depth(i, j) < 10.0 m:
    fused(i, j) = tof_depth(i, j)             # use hardware reading directly
else:
    fused(i, j) = s · student_depth(i, j)     # use scaled learned depth
```

The two implementations are operationally equivalent in the deployment regime. The specification computes the affine `(α, β)` correction explicitly because that form is the cleanest mathematical statement and matches the convention used in the depth-estimation literature. The deployment realization uses per-pixel substitution because it eliminates the failure modes attributable to a degenerate calibration set and produces the same costmap geometry under the operating conditions characterized in the [calibration sensitivity study](calibration).

**Implication for reproducibility:** the equations in the specification describe the offline evaluation pipeline exactly. Reproducing a reported number via `eval_corridor_depth.py` and `calibration_sensitivity.py` matches the specification literally. Reproducing via the on-vehicle stack yields operationally equivalent results, with the bound on agreement characterized quantitatively in the [calibration sensitivity study](calibration).

**Where to look in the code:**
- `models/losses.py:24-66` — training-time hybrid target (`target = DA3 if has_da3 else target = ToF`), the frame-level analogue of the deployment per-pixel rule
- `calibration_sensitivity.py` — affine alignment specification, used in offline evaluation
- The Depth Fusion Node in the `NCHSB` runtime repository — deployment realization (substitution + implicit median-scale anchoring)

---

## 2. APE / SLAM result deferred to future work

The localization improvement attributable to depth fusion is identified as future work in this report. A preliminary measurement obtained a 73 % SLAM Absolute Pose Error improvement when using fused depth versus LiDAR alone, but the experimental protocol that produced this number used different rosbag playback rates between the two configurations:

- LiDAR-only configuration: playback at 1.0× real time.
- Fused-depth configuration: playback at 0.5× real time, to allow the depth-fusion node to complete inference within each frame's budget.

SLAM Toolbox's loop closure runs more aggressively at slower playback, which reduces APE in the slower configuration independently of the depth-fusion contribution. The 73 % number therefore confounds the depth-fusion effect with the playback-rate effect and is not a defensible measurement of the depth-fusion contribution.

The original numbers are retained in `results/paper_stats.json` for traceability of the original measurement, but are not used as a reported result. The protocol required to recover a valid APE measurement — matched playback rates on the deployment hardware, with the depth-fusion node consuming frames in real time — is identified as future work.

---

## 3. INT8 calibration in `export_trt.py` is a stub

The script accepts `--calib-images <dir>` and uses real frames if you provide them. If you don't, it generates random tensors as the calibration set.

**Random-tensor calibration is wrong for production.** TensorRT INT8 quantization picks per-tensor scales that depend on the activation distributions seen during calibration. Random inputs produce activation distributions that don't match real corridor frames, which means the resulting INT8 engine has wrong scales and degraded accuracy.

All reported Jetson runtime numbers (218 FPS / 4.6 ms / 2.7 GB on DA3-Small) use FP16. The INT8 path is retained for completeness and for users who wish to validate it with real calibration data. The 9 / 10 closed-loop Gazebo success rate is measured under FP16 inference, not INT8.

**To use INT8 properly:** point `--calib-images` at 100-500 representative corridor frames (e.g. `corridor_eval_data/rgb/*.png`). The deployment FPS at INT8 is ~2× faster than FP16; we just haven't validated the accuracy.

---

## 4. V9 is a corridor specialist, not a strictly better model

V9 is the production corridor checkpoint because it achieves the best LILocBench corridor RMSE (0.382 m) and the closed-loop validation result (9 / 10 success in Gazebo trials at 10 seeds, matching ground-truth depth at the same seeds).

V9 is also worse on NYU val (1.553 m) than [V3](models/v3-da3-berhu) (1.160 m), [V5](models/v5-deployment-aug) (0.572 m), and [V6](models/v6-sun-diode-pretrain) (0.519 m). The LILocBench fine-tune produces substantial catastrophic forgetting on the general indoor distribution.

This is not a defect; it is a deliberate tradeoff documented across this work and the [model lineage](models/). Specializing the encoder on the deployment domain produces a better deployment model and a worse general model. For general-purpose indoor depth estimation, [V5](models/v5-deployment-aug) is the recommended checkpoint. For specialization to a different corridor environment, the pipeline supports fine-tuning [V6](models/v6-sun-diode-pretrain) on representative data from the new domain.

---

## 5. The 5.2% FPR is not free

The L+D costmap configuration adds +55% occupied cells over LiDAR-only. The cost: 5.2% false-positive rate (cells flagged as occupied that aren't).

Decomposition (from `results/fpr_audit.json`):

| Source | Share of FPR |
|---|---|
| Model hallucinations | 49.3% |
| Sensor-invalid-fill artifacts | 34.6% |
| Inflation artifacts | 18.1% |

**Model hallucinations** are the V9 student predicting depth structure that isn't there — usually the corridor "seems to end" effect on V9 in open spaces.

**Sensor-invalid-fill** comes from the depth fusion logic accepting student depth where the ToF returns zero. Where that happens spuriously (e.g. on a high-confidence reflection that the sensor mishandled), the student fills with whatever it predicted, which may not be ground.

**Inflation artifacts** are the costmap inflation radius pushing valid obstacles into adjacent cells. Standard Nav2 behavior, just visible in the FPR breakdown because we're measuring at cell granularity.

**Whether 5.2% FPR is acceptable** depends on the controller. Our local controller treats inflated cells as soft penalties, not hard obstacles, so a 5% FPR doesn't block paths — it nudges them. A controller that treats any occupied cell as untraversable would reject more paths than the L-only baseline. The FPR isn't free; whether it's affordable depends on what consumes it.

---

## 6. Why ml_inference and NCHSB are separate repos

The two-repo split was a deliberate choice, not an accident.

**ml_inference (this repo):** off-board work. Training, evaluation, video generation, figure pipelines. Runs on NYU HPC and laptops. PyTorch + Python + JSON.

**NCHSB:** on-board work. ROS 2 nodes, Nav2 integration, launch files, EKF tuning, Jetson deployment. Runs on the Jetson and on a workstation for sim. C++ + Python + ROS 2 + Gazebo.

The two have different dependency stacks (PyTorch + CUDA on the training side; ROS 2 Humble + Nav2 on the runtime side), different release cadences (training experiments push daily; runtime configuration changes weekly), and different consumers (results readers care about ml_inference's outputs; the on-vehicle team cares about NCHSB's launch files). Bundling them would force one stack on people who do not need it.

The cost: changes that span both repos require coordinated commits. We've hit that maybe four times in the project's lifetime, and each time the "single PR across two repos" friction has been less painful than the "everyone has to install everything" friction would have been daily.

---

## 7. Why 240×320 input resolution

The student takes 240×320 RGB. The Femto Bolt outputs 1280×720. We downscale.

**Why 240×320 specifically:** the EfficientViT-B1 ImageNet pretrain expects 224×224, and 240×320 is the smallest 4:3-ish multiple-of-32 size that comfortably accepts that pretrain. It's also small enough that the TensorRT engine fits in the Jetson Orin Nano's compute budget at 30 FPS, with headroom for the segmentation head.

**Cost:** spatial resolution at the depth boundaries. A glass wall that is 10 pixels wide at 1280×720 is approximately 2 pixels wide at 240×320. The segmentation head can still flag it as glass; the depth head smears the boundary across more cells than it would at native resolution.

**Why not higher:** doubling input resolution approximately quadruples inference time. The 30 FPS perception budget allows 240×320 with headroom; 480×640 inference benchmarked at ~12 FPS on the Jetson, which fails to satisfy the closed-loop budget.

**Identified improvement:** a higher-resolution segmentation head paired with the existing depth head at 240×320. Segmentation modulates inflation, where boundary precision matters more than for depth. This change is a known improvement deferred to future work.

---

## 8. Validated design choices

Not every choice in the pipeline involved a tradeoff. Several design decisions held up across the entire training program and remain in production use:

- **Hybrid depth target** (training rule: `target = ToF if confident else DA3`) — mirrors the deployment fusion principle. The student learns to predict the fused output rather than either input source alone, reducing the train/deployment distribution shift relative to training against DA3 directly with substitution applied only at runtime.
- **berHu loss** over MSE — bounds the gradient contribution of far-field pixels where teacher uncertainty dominates.
- **Kendall uncertainty weighting** — balances depth and segmentation losses jointly with parameters, avoiding a manual hyperparameter sweep over λ.
- **Two-LR optimizer** — backbone at 5e-5, decoder at 1e-3. Pretrained encoder fine-tunes slowly, randomly-initialized heads learn fast.
- **EfficientViT-B1** over MobileNetV3-Small. Already covered, but worth flagging as the single highest-ROI choice in the project.
