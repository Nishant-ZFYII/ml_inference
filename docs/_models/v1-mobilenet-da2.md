---
layout: default
title: "V1: Initial Distillation Baseline (Relative-Depth Teacher)"
version: 1
backbone: MobileNetV3-Small
teacher: DA2-Large (relative depth)
nyu_rmse: 75.37
status: baseline
---

# V1: Initial Distillation Baseline with Relative-Depth Teacher

V1 is the initial distillation configuration. It pairs a MobileNetV3-Small student with a Depth Anything 2 Large (DA2) teacher under standard MSE supervision. Evaluated against metric ground truth, the configuration produces an RMSE of 75.37 m on the NYU Depth V2 validation set — a result attributable to a unit-space mismatch between the teacher output and the evaluation metric, not to optimization or architecture failure.

## Configuration

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Small encoder, single-head depth decoder |
| Trainable parameters | ~3 × 10⁶ |
| Teacher | DA2-Large (relative depth) |
| Loss | Mean squared error against teacher output |
| Training corpus | NYU Depth V2 |
| Optimizer | AdamW with cosine annealing |
| Hardware | NVIDIA L40S 48 GB (NYU Greene HPC) |
| Status | Baseline; superseded by [V3](v3-da3-berhu) |

## Training Target Mismatch

DA2-Large produces *relative* depth: per-image depth ordering normalized to the unit interval [0, 1] without an absolute metric scale. The training procedure minimized MSE between the student prediction and these normalized targets. The student converged successfully against the teacher within the relative-depth space.

The evaluation metric, in contrast, was computed in metric units (meters) against the NYU Depth V2 ground truth, which spans approximately [0.1, 6.0] m for indoor scenes. Predictions in the [0, 1] range compared against targets in the [0.1, 6.0] range produced the observed 75.37 m RMSE.

<div class="mermaid">
flowchart LR
    RGB["RGB frame<br/>NYU Depth V2"] --> DA2["DA2-Large teacher<br/>(relative depth)"]
    DA2 --> RELDEPTH["Predicted depth<br/>values ∈ [0, 1]<br/>(normalized)"]

    RGB -.-> STUDENT["MobileNetV3-Small student"]
    STUDENT --> SPRED["Student prediction<br/>values ∈ [0, 1]"]

    RELDEPTH --> LOSS["MSE loss"]
    SPRED --> LOSS
    LOSS --> OPT["AdamW optimizer<br/>(unaware of unit space)"]
    OPT -.-> STUDENT

    GT["NYU ground truth<br/>values ∈ [0.1, 6.0] m<br/>(metric)"] --> EVAL["Evaluation metric<br/>RMSE in meters"]
    SPRED --> EVAL
    EVAL --> RESULT["RMSE = 75.37 m"]

    style DA2 fill:#fde2e2
    style RELDEPTH fill:#fde2e2
    style GT fill:#e8f0ff
    style EVAL fill:#fff3cd
    style RESULT fill:#fde2e2
</div>

*Diagram source: [`assets/diagrams/models/v1-unit-mismatch.mmd`](../assets/diagrams/models/v1-unit-mismatch.mmd) — refine in [mermaid.live](https://mermaid.live) and export as `v1-unit-mismatch.svg` to override the embedded version.*

The configuration therefore exhibits *consistent* internal optimization but *inconsistent* alignment between the supervision signal and the evaluation metric. The model learned the teacher's depth ordering correctly; the ordering had no metric anchor.

## Identification Procedure

The mismatch was identified during post-hoc evaluation of training output statistics. A single-line verification step in the dataloader at training initialization would have surfaced the inconsistency:

```python
# Verification at initialization
print(target.min().item(), target.max().item(), target.mean().item())
# DA2 output:        ~0.0,  ~1.0,  ~0.5         (normalized)
# NYU GT (m):        ~0.1,  ~6.0,  ~2.5         (metric)
```

Subsequent configurations include input-space verification at training initialization as a standard pre-flight check.

## Resolution Path

Three independent modifications were required to produce the first working configuration ([V3](v3-da3-berhu)). V1 had only the first issue; V2 confirmed the latter two were also limiting:

1. **Metric-scale teacher.** DA2-Large was replaced by DA3-Metric-Large, which produces depth in absolute meters by construction.
2. **Loss formulation.** MSE was replaced by berHu loss, which transitions from L1 to L2 at the 80th percentile of per-batch residual magnitude. This bounds the gradient contribution of far-field pixels where teacher uncertainty is highest.
3. **Multi-task balancing.** Kendall uncertainty weighting was added to balance the depth and segmentation losses without requiring a manual weight sweep.

V1 contained issue (1). Issues (2) and (3) became visible only after issue (1) was resolved, since unit-mismatched supervision dominates any signal that loss reweighting could provide.

## Reproducibility and Provenance

The V1 configuration is preserved under `archive/v1-v3-baseline/` and is reproducible via the `v1-v3-baseline` git tag. The MobileNetV3-Small encoder interface was maintained through V3, then deprecated at V4 when the architecture transitioned to EfficientViT-B1.

## Findings

The V1 configuration establishes a foundational requirement of the distillation pipeline: the teacher's output space must be expressed in the units used for evaluation. The 75.37 m RMSE reflects unit-space inconsistency, not a deficiency in optimization, architecture capacity, or training duration. This result motivated the input-space verification practice adopted in all subsequent configurations.

→ [V2](v2-kendall-clamp) reports the diagnostic study that ruled out loss-weighting as the dominant cause of the V1 result. → [V3](v3-da3-berhu) reports the first configuration to produce metric-scale predictions.
