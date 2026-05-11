---
layout: default
title: "V2: Multi-Task Loss Weighting Diagnostic"
version: 2
backbone: MobileNetV3-Small
teacher: DA2-Large (relative depth)
status: diagnostic
---

# V2: Diagnostic Study of Multi-Task Loss Weighting

V2 is a diagnostic configuration following [V1](v1-mobilenet-da2). It evaluates whether the V1 result was attributable to multi-task loss weighting rather than to teacher unit-space. Two paired training runs were conducted to isolate the loss-balancing variable while holding architecture and supervision constant. The result confirmed that loss weighting was not the dominant cause.

## Configuration

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Small (unchanged from V1) |
| Teacher | DA2-Large (relative depth, unchanged from V1) |
| Run A — `train_kendall_v2.slurm` | Kendall uncertainty weighting, log σ² clamped to [-2, 2] |
| Run B — `train_fixed.slurm` | Fixed loss weights (control) |
| Outcome | Both runs produced unusable depth predictions |
| Codename | **Sextant** (the diagnostic measurement instrument) |
| Status | Diagnostic; recipe rewrite occurred at [V3](v3-da3-berhu) |

## Hypothesis Tested

The V1 result (75.37 m RMSE on NYU val) admitted two candidate explanations: (a) the teacher unit-space mismatch documented on the [V1 page](v1-mobilenet-da2), or (b) multi-task loss imbalance, in which the depth and segmentation gradients compete for capacity in a manner the optimizer cannot resolve. V2 was designed to discriminate between these explanations.

## Multi-Task Loss Weighting

Kendall uncertainty weighting ([Kendall, Gal, Cipolla 2018](https://arxiv.org/abs/1705.07115)) introduces per-task learned log-variance parameters, with the loss formulation

$$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_d^2} \mathcal{L}_{\text{depth}} + \frac{1}{2\sigma_s^2} \mathcal{L}_{\text{seg}} + \log \sigma_d + \log \sigma_s$$

The network learns task weights jointly with parameters. Without bounds on log σ², one task can dominate the gradient budget while the other receives effectively zero supervision, producing a degenerate solution.

<div class="mermaid">
flowchart TB
    subgraph Unbounded["Unbounded log σ² (degenerate)"]
        direction TB
        U_INIT["Initialization<br/>log σ²_d ≈ 0<br/>log σ²_s ≈ 0"]
        U_DRIFT["Training drift<br/>log σ²_d → -∞<br/>log σ²_s → +∞"]
        U_RESULT["Effective behavior<br/>depth gradient → ∞<br/>seg gradient → 0<br/>(seg head untrained)"]
        U_INIT --> U_DRIFT --> U_RESULT
    end

    subgraph Clamped["log σ² clamped to [-2, 2]"]
        direction TB
        C_INIT["Initialization<br/>log σ²_d ≈ 0<br/>log σ²_s ≈ 0"]
        C_DRIFT["Bounded drift<br/>log σ²_d ∈ [-2, 2]<br/>log σ²_s ∈ [-2, 2]"]
        C_RESULT["Effective behavior<br/>both heads receive<br/>finite gradient<br/>(stable training)"]
        C_INIT --> C_DRIFT --> C_RESULT
    end

    style Unbounded fill:#fde2e2
    style Clamped fill:#d4e7c5
    style U_RESULT fill:#fde2e2
    style C_RESULT fill:#d4e7c5
</div>

*Diagram source: [`assets/diagrams/models/v2-kendall-clamp.mmd`](../assets/diagrams/models/v2-kendall-clamp.mmd).*

The clamping bound `[-2, 2]` constrains `σ²` to approximately `[0.018, 54.6]`, preventing collapse while permitting a 3000× spread in effective weighting. This bound is preserved across all subsequent configurations (V3 through V9).

## Outcome

Both runs produced unusable depth predictions. NYU val RMSE remained in the tens-of-meters range, with no significant divergence between the Kendall-weighted and fixed-weight conditions. The hypothesis that V1 had failed due to loss imbalance was therefore not supported. The remaining candidate explanation — teacher unit-space mismatch — was addressed in [V3](v3-da3-berhu).

## Components Carried Forward

Three V2 elements survived into V3 and subsequent configurations:

1. **The Kendall weighting module** with the `[-2, 2]` log-variance clamp. Implemented in `models/losses.py:MultiTaskLoss` with `use_kendall=True` as the default.
2. **The two-LR optimizer scaffolding** (separate parameter groups for encoder and decoder). The infrastructure was added during V2 and exercised at V3.
3. **Paired-control experimental discipline.** Subsequent ablation studies followed the V2 pattern of varying one experimental factor while holding the rest constant.

## Findings

V2 confirmed that multi-task loss weighting is not sufficient to compensate for misaligned supervision: bounded weighting cannot recover from a target signal expressed in the wrong units. The result narrowed the diagnostic search to teacher choice, addressed at V3. The Kendall weighting infrastructure built for this diagnostic remained in production use throughout the lineage.

→ [V3](v3-da3-berhu) reports the recipe rewrite that produced the first configuration with metric-scale predictions.
