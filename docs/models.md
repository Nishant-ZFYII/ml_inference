---
layout: default
title: "Model Lineage"
permalink: /models/
---

# Model Lineage: V1 through V9

The student-model training program comprises nine configurations spanning two backbones, three dataset mixtures, and two loss formulations. Each configuration is documented on a dedicated page describing the experimental variable introduced, the quantitative results, the operational disposition, and the findings supported by the data.

## Configuration Summary

| # | Codename | Title | Backbone | NYU val RMSE | NYU val mIoU | Femto Bolt corridor | LILocBench corridor | Status |
|---|---|---|---|---|---|---|---|---|
| [V1]({{ '/models/v1-mobilenet-da2' | relative_url }}) | Compass | Initial Distillation Baseline | MobileNetV3-Small | 75.37 m | — | — | — | Baseline |
| [V2]({{ '/models/v2-kendall-clamp' | relative_url }}) | Sextant | Loss Weighting Diagnostic | MobileNetV3-Small | — | — | — | — | Diagnostic |
| [V3]({{ '/models/v3-da3-berhu' | relative_url }}) | Anchor | Recipe Rewrite (Metric Teacher) | MobileNetV3-Small | 1.160 m | 39.3 % | — | — | Working |
| [V4]({{ '/models/v4-efficientvit-swap' | relative_url }}) | Pivot | EfficientViT-B1 Encoder | EfficientViT-B1 | 0.774 m | 51.0 % | 1.373 m | — | Working |
| [V5]({{ '/models/v5-deployment-aug' | relative_url }}) | **Atlas** | Augmentation Pipeline | EfficientViT-B1 | **0.572 m** | **63.7 %** | 2.186 m | — | **Production (general)** |
| [V6]({{ '/models/v6-sun-diode-pretrain' | relative_url }}) | **Cornerstone** | Multi-Domain Pretraining | EfficientViT-B1 | **0.519 m** | 48.5 % | 2.158 m | — | **Production (fine-tune base)** |
| [V7]({{ '/models/v7-lilocbench-finetune' | relative_url }}) | Tunnel | Fine-Tuning from V5 | EfficientViT-B1 | 1.315 m | 47.5 % | 1.982 m | 0.445 m | Superseded |
| [V8]({{ '/models/v8-mixed-failed' | relative_url }}) | Confluence | Joint-Domain Training | EfficientViT-B1 | 0.592 m | 62.9 % | **2.266 m** ↑ | — | Ablation |
| [**V9**]({{ '/models/v9-corridor-specialist' | relative_url }}) | **Lighthouse** | **Corridor-Specialized Student** | EfficientViT-B1 | 1.553 m | 31.6 % | 1.589 m | **0.382 m** | **Production (corridor)** |

## HuggingFace releases

The three production checkpoints are released as separate HuggingFace model repositories under the `NishantPushparaju/vortex-depth-*` family. The pre-production configurations (V1-V4, V7, V8) are retained in the project's `hpc_outputs/` directory for paper-side reproducibility but are not released individually on HuggingFace.

| Production checkpoint | HuggingFace identifier | Use case |
|---|---|---|
| **V5 — Atlas** | [`NishantPushparaju/vortex-depth-v5-general`](https://huggingface.co/NishantPushparaju/vortex-depth-v5-general) | General-purpose indoor depth estimation across diverse room geometries |
| **V6 — Cornerstone** | [`NishantPushparaju/vortex-depth-v6-pretrained`](https://huggingface.co/NishantPushparaju/vortex-depth-v6-pretrained) | Fine-tuning base for additional domain specialists |
| **V9 — Lighthouse** | [`NishantPushparaju/vortex-depth-v9-corridor`](https://huggingface.co/NishantPushparaju/vortex-depth-v9-corridor) | Production corridor specialist; closed-loop validated against ground-truth depth in simulation |

LILocBench corridor RMSE is measured on the Intel RealSense D455 sensor (the LILocBench reference camera). Femto Bolt corridor RMSE is measured on the Orbbec Femto Bolt deployment camera. The two measurements use distinct intrinsics and noise profiles and are not directly comparable.

## Lineage Diagram

The nine configurations form a directed graph reflecting the initialization-checkpoint dependencies. Sequential improvements (V1 → V5) preceded the branching specialization phase (V5 → V6, V7, V8; V6 → V9).

<div class="mermaid">
flowchart TB
    V1["V1: Initial Distillation Baseline<br/>(relative-depth teacher)<br/>NYU RMSE: 75.37 m"]
    V2["V2: Loss Weighting Diagnostic"]
    V3["V3: Recipe Rewrite<br/>(metric-scale teacher)<br/>NYU RMSE: 1.160 m"]
    V4["V4: EfficientViT-B1 Encoder<br/>NYU RMSE: 0.774 m"]
    V5["V5: Augmentation Pipeline<br/>NYU RMSE: 0.572 m<br/>(production: general indoor)"]
    V6["V6: Multi-Domain Pretraining<br/>NYU RMSE: 0.519 m<br/>(production: fine-tune base)"]
    V7["V7: Fine-Tuning from V5<br/>LILocBench: 0.445 m<br/>NYU regression: +130 %"]
    V8["V8: Joint-Domain Training<br/>(Pareto-dominated by V5 and V9)"]
    V9["V9: Corridor-Specialized Student<br/>LILocBench: 0.382 m<br/>9 / 10 Gazebo success<br/>(production: corridor)"]

    V1 --> V2
    V2 --> V3
    V3 -->|"backbone substitution"| V4
    V4 -->|"augmentation pipeline added"| V5
    V5 -->|"recipe inherited<br/>+ multi-domain pretrain"| V6
    V5 -->|"checkpoint init<br/>+ corridor fine-tune"| V7
    V5 -->|"checkpoint init<br/>+ joint training"| V8
    V6 -->|"checkpoint init<br/>+ corridor fine-tune"| V9

    style V1 fill:#fde2e2
    style V2 fill:#fff3cd
    style V3 fill:#e8f0ff
    style V4 fill:#e8f0ff
    style V5 fill:#d4e7c5
    style V6 fill:#d4e7c5
    style V7 fill:#fff3cd
    style V8 fill:#fde2e2
    style V9 fill:#d4e7c5
</div>

## Outcomes by Category

| Category | Configurations |
|---|---|
| Production deployment | [V5]({{ '/models/v5-deployment-aug' | relative_url }}) (general indoor), [V6]({{ '/models/v6-sun-diode-pretrain' | relative_url }}) (fine-tune base), [V9]({{ '/models/v9-corridor-specialist' | relative_url }}) (corridor specialist) |
| Sequential improvement | [V3]({{ '/models/v3-da3-berhu' | relative_url }}), [V4]({{ '/models/v4-efficientvit-swap' | relative_url }}), [V5]({{ '/models/v5-deployment-aug' | relative_url }}) |
| Specialization branch | [V7]({{ '/models/v7-lilocbench-finetune' | relative_url }}), [V9]({{ '/models/v9-corridor-specialist' | relative_url }}) |
| Diagnostic / ablation (negative result) | [V2]({{ '/models/v2-kendall-clamp' | relative_url }}), [V8]({{ '/models/v8-mixed-failed' | relative_url }}) |
| Foundational baseline (superseded) | [V1]({{ '/models/v1-mobilenet-da2' | relative_url }}) |

## Findings by Configuration

The training program supports the following findings, each grounded in the experimental record of the corresponding configuration:

| # | Finding |
|---|---|
| [V1]({{ '/models/v1-mobilenet-da2' | relative_url }}) | Teacher output space must be expressed in the units used for evaluation. Unit-space inconsistency dominates any signal that loss formulation or model capacity could provide. |
| [V2]({{ '/models/v2-kendall-clamp' | relative_url }}) | Multi-task loss weighting cannot compensate for misaligned supervision. Bounded weighting (`log σ²` clamped to `[-2, 2]`) is necessary to prevent task-collapse but is not sufficient for accurate prediction. |
| [V3]({{ '/models/v3-da3-berhu' | relative_url }}) | The combination of metric-scale teacher, berHu loss, Kendall multi-task weighting, and two-rate optimizer constitutes a sufficient training recipe for metric depth estimation. The recipe was set at V3 and held constant through V9. |
| [V4]({{ '/models/v4-efficientvit-swap' | relative_url }}) | Encoder capacity remains a binding constraint at fixed training recipe. EfficientViT-B1 over MobileNetV3-Small produces a 33 % NYU RMSE reduction with no other change. |
| [V5]({{ '/models/v5-deployment-aug' | relative_url }}) | Overfitting precedes capacity. Before evaluating larger architectures, augmentation should be exercised against the existing baseline. NYU validation accuracy and corridor deployment accuracy are not monotonically correlated. |
| [V6]({{ '/models/v6-sun-diode-pretrain' | relative_url }}) | Multi-domain pretraining transfers to downstream specialization quality. Mixed-dataset training requires explicit handling of supervision gaps to avoid loss-function edge cases. |
| [V7]({{ '/models/v7-lilocbench-finetune' | relative_url }}) | Single-domain fine-tuning produces corridor-class specialists at the cost of general-domain capability. Specialization quality depends on the initialization checkpoint, not solely on the fine-tuning protocol. |
| [V8]({{ '/models/v8-mixed-failed' | relative_url }}) | Joint-domain training does not substitute for sequential pretrain-then-specialize when the source and target domains differ at the geometric level under the project's capacity-distribution coverage ratio. |
| [V9]({{ '/models/v9-corridor-specialist' | relative_url }}) | A 5.31 × 10⁶ parameter student produced via the V6 → corridor fine-tuning pipeline achieves closed-loop navigation parity with ground-truth depth in simulation (9 / 10 success at 10 seeds, 0 collisions). |

## Selection Guidance

| Use case | Recommended checkpoint |
|---|---|
| General-purpose indoor depth estimation | [V5]({{ '/models/v5-deployment-aug' | relative_url }}) |
| Production corridor specialist | [V9]({{ '/models/v9-corridor-specialist' | relative_url }}) |
| Fine-tuning base for additional domain specialists | [V6]({{ '/models/v6-sun-diode-pretrain' | relative_url }}) |
| Foundation-model zero-shot inference at maximum throughput on Jetson | DA3-Small (218 FPS / 4.6 ms / 2.7 GB on Jetson Orin Nano TensorRT FP16) — not part of this lineage |

The two production deployment models are V5 (general indoor) and V9 (corridor specialist). V9 is the recommended checkpoint for the deployment environment documented throughout this technical report; V5 is the recommended checkpoint for unconstrained indoor scenes.
