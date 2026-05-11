---
layout: default
title: "Knowledge Distillation"
---

# Knowledge distillation

The training paradigm: large foundation models on HPC produce labels; a small student trained on those labels runs on the deployment hardware.

## The teacher / student split

| Role | Model | Where it runs | Why |
|---|---|---|---|
| **Depth teacher** | DA3-Metric-Large (Depth Anything 3) | NYU HPC L40S GPU, ~12 FPS | Generates dense metric depth labels per training frame |
| **Detection teacher** | YOLOv8-Large | NYU HPC, ~30 FPS | Bounding boxes for person, furniture, background |
| **Segmentation teacher** | SAM2-Large | NYU HPC, ~5 FPS | Instance masks from YOLO-prompted boxes; combined with geometric heuristics for floor / wall / glass |
| **Student** | EfficientViT-B1 (5.31M params) | Jetson Orin Nano, ~5 ms / frame | Mimics teacher outputs; the deployable model |

The teachers run once, offline, on HPC to produce a `manifest.jsonl` linking every RGB frame to its DA3 depth label, SAM2 segmentation label, ToF depth, and ToF confidence. The student trains against that manifest. None of the teacher models ever touch the deployment hardware.

## Why we don't run the teachers on the Jetson

Three constraints, all binding:

1. **VRAM.** DA3-Metric-Large is roughly 1B params. The Jetson Orin Nano has 8 GB total RAM (shared between system and GPU). DA3 alone exceeds that budget.
2. **Latency.** SAM2-Large runs at ~5 FPS on an A100. The Jetson would do worse. We need 30 FPS perception to match the camera frame rate.
3. **Power.** The Jetson runs on the robot's battery. Foundation-model inference is not a power budget we can afford to spend continuously.

The student is what we can afford. The teachers are what produces the supervision the student learns from.

## Why distillation works here

Distillation works because the teachers know more than they need to communicate. DA3-Metric-Large processes full RGB frames through a 1B-param network and outputs a 240×320 depth map. The student doesn't need DA3's intermediate representations; it just needs to reproduce the depth map. A small network can learn the input → output mapping without learning everything DA3 knows about depth in general.

The student also benefits from the teacher being *much better* than the data the student would otherwise train on. ToF depth is sparse and unreliable on 78% of pixels. Training on raw ToF would require the student to either ignore those pixels or learn to reproduce the sensor failure mode. Training on DA3 gives the student a dense, complete depth signal that doesn't have the failure mode in it — DA3 fills the gaps the way the student should learn to fill them.

## The hybrid depth target

The student doesn't train against pure DA3 depth. From `models/losses.py:HybridDepthLoss`:

```python
if has_da3 and da3_depth is not None:
    target = da3_depth
else:
    target = tof_depth   # fallback when DA3 inference is not available
```

When DA3 labels are present (which is the common case on HPC where teacher inference is cheap), the depth target is DA3's metric prediction. When DA3 isn't available for a particular frame, the target falls back to ToF.

The runtime fusion (see [Confidence-Gated Fusion](confidence-gated-fusion)) is per-pixel: trust ToF where valid, fall back to student elsewhere. The training target is per-frame: use DA3 if we have it, otherwise use ToF. These two policies aren't identical — they're aligned in spirit (prefer hardware ground truth where available, fall back to learned/teacher signal elsewhere) but they operate at different granularities.

## Loss formulation

The full multi-task loss combines three terms with Kendall uncertainty weighting:

$$\mathcal{L} = \frac{1}{2\sigma_d^2} \mathcal{L}_{\text{berHu}}(d_{\text{pred}}, d_{\text{target}}) + \log \sigma_d + \frac{1}{\sigma_s^2} \mathcal{L}_{\text{CE}}(s_{\text{pred}}, s_{\text{target}}) + \log \sigma_s + \lambda_e \mathcal{L}_{\text{edge}}$$

- **berHu** for depth — L1 in the small-error regime, L2 in the large-error regime, threshold = 0.2 × max-error per batch
- **Cross-entropy** for the 6-class segmentation
- **Edge-aware smoothness** as a regularizer ($\lambda_e = 0.1$, fixed)
- **Kendall log-σ** weighting on the depth and seg losses, clamped to `[-2, 2]`

Full discussion of these choices lives in the [training pipeline page](../training) and the [V3 page](../models/v3-da3-berhu) where the loss formulation was first introduced.

## What this distillation pipeline isn't trying to do

Two things this pipeline explicitly doesn't do, that other distillation pipelines often do:

1. **No soft-label distillation in the classical sense.** The student trains against the teacher's hard outputs (per-pixel metric depth value, per-pixel argmax class), not the teacher's logit distributions. For dense regression tasks, hard targets work fine; for the segmentation head, the 6-class output is small enough that argmax doesn't lose much.
2. **No KL divergence between teacher and student feature representations.** We tried it briefly. It added complexity without measurable improvement on the corridor metrics. The hard-label loss is simpler and produces deployable students.

## Why the student lineage has nine versions

Distillation has many free parameters: backbone, teacher choice, loss formulation, augmentation, dataset mixture, fine-tuning protocol. The [model lineage](../models/) walks through nine versions of varying these. The headline finding: teacher choice (V1 → V3) and backbone choice (V3 → V4) dominate. Augmentation (V4 → V5) is the next-largest single win. Pretrain order (V5 → V6 → V9) matters when specializing for a deployment domain.

## In one breath

Knowledge distillation in this project means that large teachers — too expensive to deploy on the embedded target — provided supervision for a small student that fits the on-vehicle compute budget. The student learns to reproduce the teachers' dense outputs from RGB alone, achieving ~5 ms per frame on the Jetson. The teachers remain on the HPC where their compute requirements are accommodated.
