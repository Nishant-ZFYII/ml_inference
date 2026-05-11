---
layout: default
title: "V8: Joint-Domain Training Ablation"
version: 8
backbone: EfficientViT-B1
teacher: DA3-Metric-Large
nyu_rmse: 0.592
status: ablation
---

# V8: Joint-Domain Training as an Alternative to Sequential Fine-Tuning

V8 evaluates joint-domain training (NYU Depth V2 + LILocBench in alternating batches) as an alternative to the sequential pretrain-then-fine-tune protocol used by [V6](v6-sun-diode-pretrain) and [V9](v9-corridor-specialist). The configuration produces a corridor RMSE of 2.266 m on the deployment camera — slightly worse than the V5 initialization (2.186 m). The result establishes that naive joint-domain mixing is not a viable substitute for sequential specialization when the source and target domains differ at the geometric level.

## Configuration

| Property | Value |
|---|---|
| Architecture | EfficientViT-B1 (unchanged from V4) |
| Initialization | V5 checkpoint |
| Training corpus | NYU Depth V2 + LILocBench `dynamics_0`, joint sampling at 50 / 50 ratio |
| Sampling strategy | NYU upsampled to balance the larger LILocBench split |
| Loss formulation | berHu + cross-entropy + edge-aware smoothness, Kendall-weighted (unchanged) |
| Optimizer | AdamW; encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴ (unchanged) |
| Schedule | Cosine annealing, 50 epochs |
| NYU val RMSE | 0.592 m (regression of 3.5 % vs V5's 0.572 m) |
| NYU val mIoU (6-class) | 62.9 % (regression of 0.8 pp vs V5's 63.7 %) |
| **Femto Bolt corridor RMSE** | **2.266 m** (regression of 3.7 % vs V5's 2.186 m) |
| Checkpoint | `hpc_outputs/best_depth_v8.pt` (retained for completeness; not in production use) |
| Codename | **Confluence** (the meeting of two distinct domains) |
| Status | Ablation; does not advance the deployment objective |

## Method

V8 implements naive replay: NYU and LILocBench frames are interleaved within each training batch at a 50 / 50 ratio. The objective is to evaluate whether joint exposure during training preserves NYU-domain capability while producing corridor specialization, eliminating the catastrophic forgetting observed at [V7](v7-lilocbench-finetune).

<div class="mermaid">
flowchart TB
    subgraph Sources["Source distributions"]
        direction LR
        NYU["NYU Depth V2<br/>Apartment-scale<br/>1–4 m typical depth<br/>Diverse room geometry<br/>Microsoft Kinect"]
        LIL["LILocBench dynamics_0<br/>Corridor-scale<br/>1–15 m typical depth<br/>Long parallel walls<br/>Intel RealSense D455"]
    end

    Sources -->|"50 / 50 joint sampling<br/>NYU upsampled to balance"| TRAIN["V5 initialization<br/>+ joint training<br/>50 epochs"]

    TRAIN --> V8["V8 checkpoint"]

    V8 --> NYU_R["NYU val RMSE: 0.592 m<br/>(slight regression vs V5: 0.572 m)<br/>NYU val mIoU: 62.9 %<br/>(comparable to V5: 63.7 %)"]
    V8 --> COR_R["Femto Bolt corridor RMSE: 2.266 m<br/>(regression vs V5: 2.186 m)<br/>↓ Pareto-dominated by V9: 1.589 m"]

    EXPLAIN["Failure mechanism:<br/>encoder converges to feature<br/>representations approximating<br/>the union of both distributions,<br/>optimally fitting neither"]
    V8 -.-> EXPLAIN

    style Sources fill:#e8f0ff
    style TRAIN fill:#fff3cd
    style V8 fill:#fde2e2
    style COR_R fill:#fde2e2
    style EXPLAIN fill:#fff3cd
</div>

*Diagram source: [`assets/diagrams/models/v8-joint-training-failure.mmd`](../assets/diagrams/models/v8-joint-training-failure.mmd).*

NYU contains 1.159 × 10³ training frames; LILocBench `dynamics_0` is substantially larger. To achieve 50 / 50 batch composition, the smaller dataset must be either upsampled (the configuration evaluated) or the larger dataset must be downsampled. The upsampling option preserves all corridor information at the cost of NYU overfitting from repeated frame exposure; the downsampling option preserves NYU diversity at the cost of corridor information loss. Neither configuration produced satisfactory results in early ablation; the upsampled variant is reported.

## Quantitative Results

| Metric | V8 | V5 (initialization) | V7 (sequential alternative) | V9 (sequential from V6) |
|---|---|---|---|---|
| Femto Bolt corridor RMSE | **2.266 m** | 2.186 m | 1.982 m | 1.589 m |
| NYU val RMSE | 0.592 m | 0.572 m | 1.315 m | 1.553 m |
| NYU val mIoU (6-class) | 62.9 % | 63.7 % | 47.5 % | 31.6 % |

V8 preserves NYU capability — both NYU val RMSE and mIoU remain within 1 percentage point of the V5 baseline. The replay hypothesis succeeds for the NYU axis. However, V8 fails to produce corridor improvement: the Femto Bolt corridor RMSE regresses by 3.7 % from the V5 starting point, and remains 43 % above the V9 corridor result (1.589 m).

The configuration is therefore characterized as Pareto-dominated: V5 produces strictly better general-domain performance with comparable corridor performance; V9 produces strictly better corridor performance through sequential specialization.

## Failure Mode Analysis

The corridor regression in V8 is attributable to the *average-geometry* effect. NYU and LILocBench differ along multiple geometric axes:

| Axis | NYU statistics | LILocBench statistics | Joint-training effect |
|---|---|---|---|
| Typical depth range | 1–4 m | 1–15 m | Encoder learns intermediate range distributions optimal for neither |
| Dominant geometry | Diverse, room-scale | Long parallel walls | Encoder loses scene-class specialization for both |
| Lighting structure | Mixed indoor (windows + ceiling) | Uniform fluorescent | Color-jitter augmentation effects compete across distributions |
| Camera-intrinsic-conditioned features | Kinect-tuned | RealSense-tuned | Encoder cannot specialize to either intrinsic profile |

The encoder converges to feature representations that approximate the union of both distributions but optimally fit neither. The corridor RMSE regression directly reflects this effect: V8 corridor predictions are worse than V5's because V5 was tuned to a narrower distribution that did not include corridor frames.

## Conditions Under Which Joint Training Could Succeed

The V8 result is specific to: (a) substantially distant source and target domains, (b) limited model capacity relative to the union of the two distributions, and (c) a heavily skewed source-vs-target dataset size requiring asymmetric resampling. Joint-domain training succeeds in other contexts:

- ImageNet → COCO transfer benefits from joint training because the source and target distributions share visual statistics
- Multilingual NLP models routinely train jointly across languages because the underlying linguistic representations transfer
- Multi-task learning with related tasks (depth + segmentation in this same project) succeeds because the tasks share encoder features

The V8 configuration violates all three conditions simultaneously. The result is specific to this project's data scale and architectural budget; it does not generalize to a universal claim against joint-domain training.

## Production Disposition

The V8 checkpoint is retained at `hpc_outputs/best_depth_v8.pt` for completeness but is not used in evaluation reporting, deployment, or comparison tables. The configuration is documented as a negative result that informed the production decision to adopt sequential specialization at [V9](v9-corridor-specialist).

## Findings

V8 establishes one finding for training-program design:

**Joint-domain training does not substitute for sequential pretrain-then-specialize when the source and target domains differ at the geometric level.** Under the conditions specific to this project (5.31 × 10⁶ parameter encoder, NYU vs corridor distributional distance, 6 % source-domain frame share without resampling), naive replay produces Pareto-dominated results: weaker general capability than the source-only baseline and weaker corridor capability than the sequentially-specialized alternative.

The result motivated the production decision to adopt [V9](v9-corridor-specialist) (V6 → corridor fine-tune) as the corridor specialist rather than further variants of joint-domain training.

→ [V9](v9-corridor-specialist) reports the production corridor checkpoint obtained via sequential specialization from the V6 multi-domain pretrained base.
