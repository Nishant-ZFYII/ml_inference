---
layout: default
title: "V5: Deployment-Targeted Augmentation Pipeline"
version: 5
backbone: EfficientViT-B1
teacher: DA3-Metric-Large
nyu_rmse: 0.572
status: production-general
---

# V5: Deployment-Targeted Augmentation Pipeline

V5 introduces a deployment-targeted augmentation pipeline to the [V4](v4-efficientvit-swap) configuration, holding architecture, teacher, and loss formulation constant. The augmentation set yields a 26 % relative reduction in NYU validation RMSE (0.774 m → 0.572 m) — the largest single-step improvement in the lineage. The configuration is the recommended deployable model for general indoor depth estimation.

## Configuration

| Property | Value |
|---|---|
| Architecture | EfficientViT-B1 (unchanged from V4) |
| Trainable parameters | 5.31 × 10⁶ |
| Teacher | DA3-Metric-Large (unchanged) |
| Loss formulation | berHu + cross-entropy + edge-aware smoothness, Kendall-weighted (unchanged) |
| Optimizer | AdamW; encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴ (unchanged) |
| Augmentation pipeline | **Horizontal flip + ColorJitter + random crop / resize** (new) |
| Schedule | Cosine annealing, 200 epochs |
| Batch size | 16 |
| Training corpus | NYU Depth V2 |
| HPC job ID | 3070058 |
| **NYU val RMSE** | **0.572 m** (down from V4's 0.774 m, **−26 %**) |
| **NYU val mIoU (6-class)** | **63.7 %** (up from V4's 51.0 %, +12.7 pp) |
| Femto Bolt corridor RMSE | 2.186 m (see Distribution Shift Analysis below) |
| Checkpoint | `hpc_outputs/best_depth_v5_vivek.pt` |
| **HuggingFace identifier** | [`NishantPushparaju/vortex-depth-v5-general`](https://huggingface.co/NishantPushparaju/vortex-depth-v5-general) |
| **Codename** | **Atlas** (the comprehensive general-purpose map) |

## Method

The configuration introduces three augmentation operations, applied stochastically to RGB inputs at training time. Targets (depth, segmentation, confidence, DA3) are not perturbed photometrically; geometric transforms (flip, crop) are applied identically to RGB and target tensors to preserve pixel correspondence.

<div class="mermaid">
flowchart TB
    INPUT["Input frame<br/>(NYU Depth V2)"]
    INPUT --> RGB["RGB tensor"]
    INPUT --> TARGETS["Aligned targets:<br/>depth, segmentation,<br/>confidence, DA3 depth"]

    RGB --> FLIP{"Horizontal flip<br/>(p = 0.5)"}
    TARGETS --> FLIP_T{"Flip identically"}
    FLIP -->|"flip applied"| FLIP_T

    FLIP --> JITTER["ColorJitter<br/>brightness ± 0.2<br/>contrast ± 0.2<br/>saturation ± 0.2<br/>hue ± 0.1"]
    JITTER --> SIZE_CHECK{"Source dims<br/>> target?"}

    SIZE_CHECK -->|"yes"| RAND_CROP["Random crop<br/>240 × 320"]
    SIZE_CHECK -->|"no"| RESIZE["Bilinear resize<br/>240 × 320"]

    FLIP_T --> CROP_T{"Crop identically"}
    RAND_CROP --> CROP_T
    CROP_T --> RESIZED_T["Targets resized<br/>240 × 320"]

    RAND_CROP --> NORM["ImageNet normalization<br/>(within forward pass)"]
    RESIZE --> NORM

    NORM --> MODEL["EfficientViT-B1<br/>+ dual-head decoder"]
    RESIZED_T --> LOSS["Multi-task loss<br/>(berHu + CE + edge,<br/>Kendall-weighted)"]
    MODEL --> LOSS

    style FLIP fill:#fff3cd
    style JITTER fill:#fff3cd
    style RAND_CROP fill:#fff3cd
    style RESIZE fill:#fff3cd
    style MODEL fill:#e8f0ff
    style LOSS fill:#d4e7c5
</div>

*Diagram source: [`assets/diagrams/models/v5-augmentation-pipeline.mmd`](../assets/diagrams/models/v5-augmentation-pipeline.mmd).*

The implementation in `dataset/nyu_loader.py:222-249` consists of:

```python
if self.augment:
    # Horizontal flip (probability 0.5), applied identically to RGB and all targets
    if random.random() > 0.5:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        depth = np.fliplr(depth).copy()
        seg = np.fliplr(seg).copy()
        confidence = np.fliplr(confidence).copy()
        da3_depth = np.fliplr(da3_depth).copy()

    # Photometric jitter on RGB only
    jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    rgb = jitter(rgb)

    # Random crop if source dimensions exceed target; bilinear resize otherwise.
    # Crop is applied identically to all aligned tensors.
    if h > self.height and w > self.width:
        top = random.randint(0, h - self.height)
        left = random.randint(0, w - self.width)
        rgb = rgb.crop((left, top, left + self.width, top + self.height))
        depth = depth[top:top + self.height, left:left + self.width]
        # ... seg, confidence, da3_depth cropped identically
```

The augmentation set is intentionally minimal. Earlier ablation runs evaluated additional perturbations (gamma jitter, motion blur, rotation, random erasing); none produced statistically significant accuracy improvements over the three operations retained, and several increased training instability. The configuration shipped represents the smallest augmentation set that captured the deployment-relevant variance.

## Quantitative Results

| Metric | V5 | V4 | Δ |
|---|---|---|---|
| NYU val RMSE | 0.572 m | 0.774 m | −26 % |
| NYU val mIoU (6-class) | 63.7 % | 51.0 % | +12.7 pp |
| Femto Bolt corridor RMSE | 2.186 m | 1.373 m | +59 % (regression; see below) |

The 26 % NYU RMSE reduction and the 12.7 pp mIoU improvement are both attributable to the augmentation pipeline at fixed architecture and loss configuration. The mIoU improvement is the larger of the two in proportional terms; segmentation typically benefits more from training-time variability than regression, since class boundaries are categorically sharper than depth boundaries.

V5 represents the largest single-step improvement in the lineage by absolute and relative margin. The result indicates that V4's accuracy ceiling was overfitting-bound rather than capacity-bound: the 5.31 × 10⁶ parameter encoder was sufficient for the task, but the training distribution failed to span the variance encountered at evaluation time. Augmentation closed that gap.

## Distribution Shift Analysis

The Femto Bolt corridor RMSE increases from 1.373 m (V4) to 2.186 m (V5), despite the NYU val improvement. This regression has two contributing factors:

1. **Augmentation distribution mismatch.** The V5 augmentation set targets NYU-style variance (apartment-scale lighting, room geometry, indoor textures). The encoder fits the augmented NYU distribution more tightly than V4, which moves it further from the corridor distribution (long parallel walls, polished floors, mid-field-dominant depth, fluorescent lighting).

2. **Camera intrinsic mismatch.** NYU val measurements use the Microsoft Kinect (structured light, 640 × 480 RGB, ~28° vertical FOV). The Femto Bolt deployment camera uses Time-of-Flight at 1280 × 720 RGB with different focal length, baseline, and noise profile. Per-frame median-scale alignment partially corrects for the intrinsic difference but cannot compensate for distributional differences in the underlying scene structure.

The corridor regression at V5 motivated the corridor-specific fine-tuning configurations evaluated at [V7](v7-lilocbench-finetune) and [V9](v9-corridor-specialist).

## Deployment Profile

V5 is positioned as the recommended general-purpose indoor depth model in this lineage. Operational properties:

| Property | Value |
|---|---|
| Use case | General indoor depth estimation across diverse room geometries |
| Inference latency (Jetson Orin Nano, TensorRT FP16) | ~5 ms |
| Robustness across NYU val distribution | High (0.572 m RMSE, 63.7 % mIoU) |
| Robustness on corridor-class environments | Limited (use [V9](v9-corridor-specialist) instead) |
| Suitability as fine-tuning base | Demonstrated for [V7](v7-lilocbench-finetune); superseded by [V6](v6-sun-diode-pretrain) for [V9](v9-corridor-specialist) |

## Demonstration

<video controls width="100%" preload="metadata">
  <source src="{{ '/assets/videos/v5_6panel.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the embedded video. <a href="{{ '/assets/videos/v5_6panel.mp4' | relative_url }}">Download MP4</a>.
</video>

Six-panel video sequence over the 459-frame `corridor_eval` set. **Top row (raw inputs and reference)**: RGB input from the deployment camera · raw Femto Bolt ToF depth · zero-shot DA3-Small reference depth (median-scale aligned). **Bottom row (V5-specific predictions and fusions)**: V5 student raw inference · confidence-gated fusion of ToF and DA3 (the foundation-model fusion baseline) · confidence-gated fusion of ToF and V5 (the V5 deployment realization). The bottom-right panel is the depth signal that would be consumed by the local costmap if V5 (Atlas) were the deployed model; comparing it against the bottom-middle panel isolates the V5 contribution against the foundation-model fusion baseline.

A cross-version comparison on the same sequence is available on the [demo videos page](../videos).

## Findings

The V5 configuration establishes two operational principles inherited by subsequent training programs:

1. **Augmentation precedes capacity.** Before evaluating larger architectures, augmentation should be exercised against the V4 baseline. The V5 result demonstrates that the V4 accuracy plateau was attributable to overfitting rather than insufficient model capacity.

2. **NYU validation accuracy and corridor deployment accuracy are not monotonically correlated.** Training improvements measured on NYU val do not generalize to the corridor distribution. From V5 onward, both metrics are reported separately, and neither is treated as a proxy for the other.

→ [V6](v6-sun-diode-pretrain) reports the multi-domain pretraining stage that further improves NYU val accuracy and produces the fine-tuning base used for [V9](v9-corridor-specialist).
