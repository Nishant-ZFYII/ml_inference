---
layout: default
title: "V4: EfficientViT-B1 Encoder Substitution"
version: 4
backbone: EfficientViT-B1
teacher: DA3-Metric-Large
nyu_rmse: 0.774
status: working
---

# V4: Encoder Substitution at Fixed Training Recipe

V4 substitutes EfficientViT-B1 for MobileNetV3-Small as the encoder, holding the [V3](v3-da3-berhu) training recipe constant. The configuration achieves 0.774 m NYU val RMSE, a 33 % relative reduction from V3, and is the first configuration evaluated against the deployment camera (Femto Bolt corridor RMSE: 1.373 m). All subsequent configurations (V5 through V9) inherit the V4 encoder.

## Configuration

| Property | Value |
|---|---|
| Architecture | **EfficientViT-B1** encoder, dual-head decoder (depth + segmentation) |
| Trainable parameters | 5.31 × 10⁶ (3.5 × increase over V3) |
| Teacher | DA3-Metric-Large (unchanged from V3) |
| Loss formulation | berHu + cross-entropy + edge-aware smoothness, Kendall-weighted (unchanged from V3) |
| Optimizer | AdamW; encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴ (unchanged from V3) |
| Schedule | Cosine annealing, 200 epochs |
| Batch size | 16 |
| Training corpus | NYU Depth V2 |
| HPC job ID | 3043912 |
| **NYU val RMSE** | **0.774 m** (down from V3's 1.160 m, **−33 %**) |
| **NYU val mIoU (6-class)** | **51.0 %** (up from V3's 39.3 %, +11.7 pp) |
| **Femto Bolt corridor RMSE** | 1.373 m (first deployment-camera measurement) |
| Codename | **Pivot** (the architectural turning point of the lineage) |

## Method

The configuration differs from V3 by a single experimental variable: the encoder backbone. All other components — teacher selection, loss formulation, augmentation pipeline, optimizer settings, and training schedule — are held constant, isolating the effect of the encoder substitution.

EfficientViT-B1 ([Cai et al. 2023](https://arxiv.org/abs/2205.14756)) is a hybrid attention-convolution architecture with linear-complexity multi-head self-attention. The architectural difference relative to V3's encoder is shown below.

<div class="mermaid">
flowchart TB
    subgraph V3enc["V3: MobileNetV3-Small (~1.5 × 10⁶ params)"]
        direction TB
        M_IN["Input [B, 3, 240, 320]"]
        M1["Stage 1: 16 ch (H/2)"]
        M2["Stage 2: 24 ch (H/4)"]
        M3["Stage 3: 40 ch (H/8)"]
        M4["Stage 4: 96 ch (H/16)"]
        M_IN --> M1 --> M2 --> M3 --> M4
        M_NOTE["Pure depthwise-separable convolution<br/>Receptive field grows linearly with depth<br/>Limited capacity for long-range structure"]
        M4 -.-> M_NOTE
    end

    subgraph V4enc["V4: EfficientViT-B1 (5.31 × 10⁶ params)"]
        direction TB
        E_IN["Input [B, 3, 240, 320]"]
        E1["Stage 1: 40 ch (H/2)<br/>+ multi-head attention"]
        E2["Stage 2: 80 ch (H/4)<br/>+ multi-head attention"]
        E3["Stage 3: 160 ch (H/8)<br/>+ multi-head attention"]
        E4["Stage 4: 256 ch (H/16)<br/>+ multi-head attention"]
        E_IN --> E1 --> E2 --> E3 --> E4
        E_NOTE["Hybrid attention-convolution<br/>Linear-complexity attention<br/>Captures long-range dependencies<br/>(corridor lines, perspective)"]
        E4 -.-> E_NOTE
    end

    V3RES["NYU val RMSE: 1.160 m<br/>NYU val mIoU: 39.3 %"]
    V4RES["NYU val RMSE: 0.774 m (−33 %)<br/>NYU val mIoU: 51.0 % (+11.7 pp)"]
    V3enc -.->|"trained with V3 recipe"| V3RES
    V4enc -.->|"same V3 recipe, encoder swap"| V4RES

    style V3enc fill:#fde2e2
    style V4enc fill:#d4e7c5
    style V3RES fill:#fde2e2
    style V4RES fill:#d4e7c5
</div>

*Diagram source: [`assets/diagrams/models/v4-encoder-comparison.mmd`](../assets/diagrams/models/v4-encoder-comparison.mmd).*

The capacity comparison and selection rationale relative to neighboring options:

| Encoder | Parameters | Rationale |
|---|---|---|
| MobileNetV3-Small (V3) | ~1.5 × 10⁶ | Insufficient capacity for fine spatial structure at depth boundaries |
| **EfficientViT-B1 (V4)** | **5.31 × 10⁶** | Selected; meets Jetson Orin Nano latency budget at ~5 ms inference |
| EfficientViT-B2 | ~15.4 × 10⁶ | Evaluated in `train_iter7b_b2.slurm`; marginal accuracy gain insufficient to justify ~3 × inference cost |
| EfficientNet-B3 | ~12 × 10⁶ | Considered; rejected on inference latency |

EfficientViT-B1 occupies the operational sweet spot: sufficient capacity to capture long-range structure for indoor depth estimation, and inference latency compatible with the 30 Hz perception loop on the deployment hardware.

## Architectural Detail

The EfficientViT-B1 encoder produces feature maps at four resolution stages with channel widths `[40, 80, 160, 256]`. The decoder consumes these via skip connections to a 128-channel neck, with two parallel transposed-convolution paths producing depth and segmentation outputs at the input resolution.

Each `DecoderBlock(in, out)` implements:

```
Upsample(2×) → Conv2d(in + skip, out, 3×3) → BatchNorm → ReLU
```

The skip connection is drawn from the encoder stage of matching spatial resolution and concatenated with the upsampled feature map prior to convolution. ImageNet normalization is applied within the forward pass to ensure preprocessing parity between training and inference.

## Quantitative Results

| Metric | V4 | V3 | Δ |
|---|---|---|---|
| NYU val RMSE | 0.774 m | 1.160 m | −33 % |
| NYU val mIoU (6-class) | 51.0 % | 39.3 % | +11.7 pp |
| Femto Bolt corridor RMSE | 1.373 m | not measured | — |

The 33 % NYU RMSE reduction is attributable to the encoder substitution under fixed-recipe conditions. The 11.7 pp mIoU improvement is similarly encoder-attributable; segmentation typically benefits from increased encoder capacity due to the sharper class boundaries available with richer feature representations.

The Femto Bolt corridor RMSE (1.373 m) substantially exceeds the NYU val RMSE. This gap is attributable to distribution shift: V4 was trained on clean teacher labels over NYU's apartment-scale scenes, and the deployment recordings include exposure dynamics, motion blur, and corridor geometry not represented in the training distribution. This gap motivated the augmentation work introduced at [V5](v5-deployment-aug).

## Concurrent Contribution Pattern

The V4 result reflects the joint effect of two parallel contributions: the encoder substitution and the [V3](v3-da3-berhu) recipe rewrite. Either contribution in isolation produces a substantially weaker result. The same recipe applied to MobileNetV3-Small produced V3's 1.160 m RMSE; the same encoder applied to V1's MSE-against-relative-depth supervision would inherit V1's unit-space failure regardless of architectural improvements.

The V4 result is therefore not attributable to a single component but to the interaction between the encoder and the recipe — neither sufficient alone.

## Findings

V4 establishes the architectural template inherited by all subsequent configurations. The encoder, decoder structure, neck dimensionality, skip-connection topology, and forward-pass normalization remain unchanged through [V9](v9-corridor-specialist). All variation in V5 through V9 is restricted to the data pipeline (augmentation, dataset mixture) and the training schedule (pretraining stages, fine-tuning protocols).

→ [V5](v5-deployment-aug) reports the augmentation pipeline that closes the train/test distribution gap and produces the largest single-step accuracy improvement in the lineage.
