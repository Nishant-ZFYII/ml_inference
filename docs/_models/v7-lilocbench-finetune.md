---
layout: default
title: "V7: Single-Domain Fine-Tuning from General-Purpose Base"
version: 7
backbone: EfficientViT-B1
teacher: DA3-Metric-Large
nyu_rmse: 1.315
lilocbench_rmse: 0.445
status: superseded
---

# V7: Single-Domain Fine-Tuning from General-Purpose Initialization

V7 evaluates corridor-domain specialization by fine-tuning the [V5](v5-deployment-aug) general-purpose checkpoint on the LILocBench corridor dataset. The configuration achieves 0.445 m corridor RMSE on the LILocBench distribution at the cost of substantial regression in general-domain capability (NYU val RMSE: 1.315 m, 130 % above V5). V7 establishes the corridor-specialization baseline against which [V9](v9-corridor-specialist) is compared and is superseded by V9 as the production corridor checkpoint.

## Configuration

| Property | Value |
|---|---|
| Architecture | EfficientViT-B1 (unchanged from V4) |
| Initialization | **V5 checkpoint** (NYU-only training, V5 augmentation pipeline) |
| Fine-tuning corpus | LILocBench `dynamics_0` split (Intel RealSense D455, ~5 × 10³ frames) |
| Loss formulation | berHu + cross-entropy + edge-aware smoothness, Kendall-weighted (unchanged) |
| Optimizer | AdamW; encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴ (unchanged) |
| Schedule | Cosine annealing, 50 epochs |
| Batch size | 16 |
| HPC job ID | 3092402 |
| **LILocBench corridor RMSE (D455)** | **0.445 m** |
| Femto Bolt corridor RMSE | 1.982 m (cross-camera evaluation) |
| **NYU val RMSE** | **1.315 m** (regression of 130 % vs V5's 0.572 m) |
| NYU val mIoU (6-class) | 47.5 % (regression of 16.2 pp vs V5's 63.7 %) |
| Codename | **Tunnel** (the first corridor specialist) |
| Checkpoint | `hpc_outputs/best_depth_v7.pt` |
| Status | Superseded by [V9](v9-corridor-specialist) |

## Method

V7 differs from V5 only in the training data and schedule. The architecture, loss formulation, optimizer settings, and augmentation pipeline are inherited unchanged. The configuration loads the V5 checkpoint and continues training on LILocBench `dynamics_0` for 50 epochs.

<div class="mermaid">
flowchart TB
    V5BASE["V5 checkpoint (initialization)<br/>NYU val RMSE: 0.572 m<br/>NYU val mIoU: 63.7 %<br/>(general indoor capability)"]

    V5BASE -->|"continued training, 50 epochs<br/>encoder LR 3 × 10⁻⁵<br/>decoder LR 3 × 10⁻⁴"| FT["LILocBench dynamics_0<br/>~5 × 10³ corridor frames<br/>Intel RealSense D455"]

    FT --> V7["V7 checkpoint"]

    V7 --> CORR_GAIN["Corridor performance<br/>LILocBench D455 RMSE: 0.445 m<br/>Femto Bolt RMSE: 1.982 m<br/>(↓ improvement vs V5)"]
    V7 --> NYU_LOSS["NYU general capability<br/>NYU val RMSE: 1.315 m (+130 %)<br/>NYU val mIoU: 47.5 % (−16.2 pp)<br/>(↑ catastrophic forgetting)"]

    style V5BASE fill:#e8f0ff
    style V7 fill:#fff3cd
    style CORR_GAIN fill:#d4e7c5
    style NYU_LOSS fill:#fde2e2
</div>

*Diagram source: [`assets/diagrams/models/v7-finetune-tradeoff.mmd`](../assets/diagrams/models/v7-finetune-tradeoff.mmd).*

LILocBench is a corridor-class indoor depth benchmark recorded with an Intel RealSense D455 active-stereo camera at the University of Bonn. The `dynamics_0` split contains static-scene frames (no walking pedestrians, no moving objects). The dataset structure differs from NYU in three ways relevant to model behavior:

| Property | NYU Depth V2 | LILocBench |
|---|---|---|
| Scene class | Apartments, offices, kitchens | University corridors |
| Depth range | ~1–4 m typical | ~1–15 m typical |
| Dominant geometry | Diverse, room-scale | Long parallel walls, mid-field-dominant |
| Sensor | Microsoft Kinect (structured light) | Intel RealSense D455 (active stereo) |
| Lighting | Mixed indoor (windows + ceiling) | Uniform fluorescent |
| RGB-depth alignment | Hardware-aligned | ~50 µs timestamp offset (nearest-timestamp bisection at load time) |

## Quantitative Results

| Metric | V7 | V5 (initialization) | Δ |
|---|---|---|---|
| LILocBench corridor RMSE (D455) | 0.445 m | not measured (out of training distribution) | — |
| Femto Bolt corridor RMSE | 1.982 m | 2.186 m | −9.3 % |
| NYU val RMSE | 1.315 m | 0.572 m | +130 % |
| NYU val mIoU (6-class) | 47.5 % | 63.7 % | −16.2 pp |

The corridor improvement (LILocBench RMSE 0.445 m, Femto Bolt RMSE 1.982 m) is consistent with the fine-tuning hypothesis: domain-specific training on corridor data produces a model competitive with published corridor-depth methods on the LILocBench benchmark. The NYU regression (RMSE +130 %, mIoU −16.2 pp) is consistent with standard catastrophic-forgetting dynamics under single-domain fine-tuning.

## Cross-Camera Evaluation

V7's corridor performance is reported separately on two cameras with distinct intrinsics, baseline geometry, and noise characteristics:

| Camera | Sensor type | Reported RMSE | Use |
|---|---|---|---|
| Intel RealSense D455 (LILocBench) | Active stereo | 0.445 m | In-domain accuracy; comparison against published corridor-depth methods on the LILocBench benchmark |
| Orbbec Femto Bolt (deployment) | Time-of-Flight | 1.982 m | Out-of-domain accuracy on the deployment camera; not directly comparable to LILocBench numbers |

These measurements are not directly comparable. Reported numbers throughout this technical report annotate the camera identifier (LILocBench D455 or Femto Bolt) to preserve the distinction.

## Catastrophic Forgetting Analysis

The 130 % NYU RMSE regression and 16.2 pp mIoU regression are consistent with the theoretical and empirical literature on catastrophic forgetting in transfer learning ([McCloskey & Cohen 1989](https://www.sciencedirect.com/science/article/pii/S0079742108605368)). The magnitude of forgetting depends on three factors:

1. **Distributional distance between source and target domains.** NYU (apartment-scale, diverse geometry) and LILocBench (corridor-scale, repetitive parallel structure) differ at the geometric level, producing high distance.
2. **Effective fine-tuning capacity.** Without parameter freezing or replay, all encoder weights are updated. Fifty epochs at the configured learning rate is sufficient to substantially overwrite the source-domain representations.
3. **Capacity-distribution coverage ratio.** A 5.31 × 10⁶ parameter encoder cannot maintain accurate predictions on both NYU and LILocBench simultaneously without an explicit retention mechanism.

Mitigation strategies considered for the V7 program:

| Strategy | Outcome | Production status |
|---|---|---|
| Lower encoder learning rate during fine-tune | Slower forgetting and slower corridor improvement at fixed cost; tradeoff did not change shape | Not adopted |
| Replay (joint NYU + LILocBench training) | Evaluated separately at [V8](v8-mixed-failed); produced regression on both metrics | Not adopted |
| Adapter layers (freeze encoder, learn task-specific heads) | Would require modifying the deployment architecture; deployment ABI constraints precluded | Not implemented |
| Initialize from a richer pretrain base | Adopted at [V9](v9-corridor-specialist) using V6 initialization; produced 14 % relative LILocBench RMSE improvement | **Adopted** |

The final mitigation strategy adopted at V9 — selecting a richer initialization checkpoint rather than altering the fine-tuning protocol — produced a measurable improvement at fixed protocol overhead.

## Demonstration

<video controls width="100%" preload="metadata">
  <source src="{{ '/assets/videos/v7_6panel.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the embedded video. <a href="{{ '/assets/videos/v7_6panel.mp4' | relative_url }}">Download MP4</a>.
</video>

Six-panel video sequence over the 459-frame `corridor_eval` set. **Top row (raw inputs and reference)**: RGB input · raw Femto Bolt ToF depth · zero-shot DA3-Small reference depth (median-scale aligned). **Bottom row (V7-specific predictions and fusions)**: V7 student raw inference · confidence-gated fusion of ToF and DA3 · confidence-gated fusion of ToF and V7. V7's predictions exhibit visibly higher temporal jitter than V6 or V9 — a consequence of catastrophic forgetting on general-domain features that previously stabilized predictions in regions outside the corridor specialization distribution. The V7 fusion (bottom-right) is the depth signal that would be consumed by the local costmap if V7 were the deployed corridor checkpoint.

## Findings

The V7 configuration supports two findings:

1. **Single-domain fine-tuning produces corridor-class specialists at the cost of general-domain capability.** The 130 % NYU RMSE regression quantifies the cost; the 0.445 m LILocBench RMSE quantifies the benefit. The tradeoff is intrinsic to the protocol and does not improve under standard mitigation strategies (LR scaling, additional epochs).

2. **Specialization quality depends on the initialization checkpoint, not solely on the fine-tuning protocol.** Comparing V7 (V5-initialized, 0.445 m LILocBench RMSE) and V9 (V6-initialized, 0.382 m LILocBench RMSE) at fixed fine-tuning protocol isolates the initialization effect at 14 % relative improvement.

→ [V8](v8-mixed-failed) reports the joint-training configuration evaluated as an alternative to single-domain fine-tuning. → [V9](v9-corridor-specialist) reports the production corridor checkpoint.
