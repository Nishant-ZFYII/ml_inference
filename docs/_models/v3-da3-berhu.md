---
layout: default
title: "V3: Recipe Rewrite with Metric-Scale Teacher"
version: 3
backbone: MobileNetV3-Small
teacher: DA3-Metric-Large
nyu_rmse: 1.160
status: working
---

# V3: First Configuration with Metric-Scale Predictions

V3 is the first configuration in the lineage to produce metric-scale depth predictions. It introduces four changes relative to [V1](v1-mobilenet-da2) and [V2](v2-kendall-clamp): a metric-scale teacher, the berHu loss, Kendall uncertainty weighting (with the V2 clamp), and a two-rate optimizer. The configuration achieves 1.160 m RMSE on the NYU Depth V2 validation set.

## Configuration

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Small encoder, dual-head decoder (depth + 6-class segmentation) |
| Trainable parameters | ~1.5 × 10⁶ |
| Teacher | **DA3-Metric-Large** (metric depth, replaces DA2-Large) |
| Loss | berHu (depth) + cross-entropy (segmentation) + edge-aware smoothness, Kendall-weighted |
| Optimizer | AdamW; encoder LR 3 × 10⁻⁵ (= base × `ENCODER_LR_SCALE = 0.1`), decoder LR 3 × 10⁻⁴ |
| Encoder warm-up | 5 epochs frozen |
| Schedule | Cosine annealing, 200 epochs |
| Batch size | 16 |
| Training corpus | NYU Depth V2 (1,159 train frames) |
| Compute | NVIDIA L40S 48 GB, NYU Greene HPC, partition `l40s_public` |
| HPC job ID | 2981850 |
| **NYU val RMSE** | **1.160 m** |
| **NYU val mIoU (6-class)** | **39.3 %** |
| Codename | **Anchor** (the first stable metric-scale model) |

## Method

Four independent modifications were applied to the V1/V2 configuration. The component-level diagram below shows the resulting loss formulation.

<div class="mermaid">
flowchart TB
    INPUTS["Forward-pass outputs and targets<br/>RGB input, predicted depth, predicted seg logits<br/>Hybrid depth target (DA3 where available, else ToF)<br/>6-class segmentation ground truth (ignore_index = 255)"]

    INPUTS --> BERHU["berHu loss (depth)<br/>L1 if |e| ≤ c<br/>(e² + c²) / 2c if |e| > c<br/>c = 0.2 · max|e| per batch"]
    INPUTS --> CE["Cross-entropy loss (segmentation)<br/>standard reduction over valid pixels"]
    INPUTS --> EDGE["Edge-aware smoothness (regularizer)<br/>|∇d_pred| · exp(-|∇RGB|)"]

    BERHU --> KENDALL
    CE --> KENDALL
    EDGE --> KENDALL

    KENDALL["Kendall multi-task weighting + edge regularizer<br/>0.5 · exp(-log σ_d) · L_depth + 0.5 · log σ_d<br/>+ exp(-log σ_s) · L_seg + 0.5 · log σ_s<br/>+ λ_edge · L_edge<br/>(log σ clamped to [-2, 2], λ_edge = 0.1)"]

    KENDALL --> TOTAL["Total scalar loss<br/>(backpropagated via AdamW)"]

    style INPUTS fill:#f5f5f5
    style BERHU fill:#e8f0ff
    style CE fill:#e8f0ff
    style EDGE fill:#e8f0ff
    style KENDALL fill:#fff3cd
    style TOTAL fill:#d4e7c5
</div>

*Diagram source: [`assets/diagrams/models/v3-loss-composition.mmd`](../assets/diagrams/models/v3-loss-composition.mmd).*

### Metric-scale teacher

DA3-Metric-Large produces depth in absolute meters by construction (focal-aware, no per-image normalization). This eliminates the unit-space mismatch documented at V1 and reduces NYU val RMSE from the 75 m range to ~1.2 m without any other change.

### berHu loss

The reverse Huber loss ([Laina et al. 2016](https://arxiv.org/abs/1606.00373)) is implemented in `models/losses.py:HybridDepthLoss`:

$$\mathcal{L}_{\text{berHu}}(e) = \begin{cases} |e| & |e| \leq c \\ \dfrac{e^2 + c^2}{2c} & |e| > c \end{cases}, \quad c = 0.2 \cdot \max_{\text{batch}} |e|$$

The L1 regime preserves gradient signal in the small-error range; the L2 regime amplifies the penalty for large mispredictions. The threshold adapts per batch, set to the 80th percentile of residual magnitude. This formulation is appropriate for navigation-grade depth, where the operational cost of large errors (e.g., a 3 m hallucinated obstacle) substantially exceeds the cost of small errors (e.g., a 0.1 m boundary smear at 5 cm cell resolution).

### Kendall uncertainty weighting with clamp

The MultiTaskLoss in `models/losses.py:88` defaults to `use_kendall=True` with `log σ²` clamped to `[-2, 2]` (per the V2 finding):

$$\mathcal{L}_{\text{total}} = \frac{1}{2}e^{-\log \sigma_d}\mathcal{L}_{\text{depth}} + \frac{1}{2}\log \sigma_d + e^{-\log \sigma_s}\mathcal{L}_{\text{seg}} + \frac{1}{2}\log \sigma_s + \lambda_e \mathcal{L}_{\text{edge}}$$

with `λ_e = 0.1` fixed. The depth and segmentation losses balance jointly with parameters; the edge regularizer carries a small fixed weight.

### Two-rate optimizer with encoder warm-up

The encoder uses ImageNet pretraining and benefits from a fine-tuning rate one order of magnitude below the decoder rate. From `train.py:291`:

```python
encoder_lr = cfg.LR * cfg.ENCODER_LR_SCALE   # 3e-4 × 0.1 = 3e-5
param_groups = [
    {"params": encoder_params, "lr": encoder_lr},
    {"params": decoder_params, "lr": cfg.LR},   # 3e-4
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
```

The encoder is frozen for the first 5 epochs (`FREEZE_ENCODER_EPOCHS = 5`) to allow the decoder heads to stabilize before any encoder weights are updated.

## Quantitative Results

| Metric | V3 | V1 (initial baseline) |
|---|---|---|
| NYU val RMSE | **1.160 m** | 75.37 m (unit-mismatched) |
| NYU val mIoU (6-class) | 39.3 % | not measurable |

A 1.160 m RMSE on indoor scenes with typical depth range 1–6 m corresponds to approximately half-meter mean error, with substantially higher error on a tail of difficult frames. This level of accuracy is below the threshold required for navigation-grade obstacle inflation; obstacle predictions in the corridor environment would produce frequent collisions. The result is reported as the recipe-validation baseline rather than as a deployable model.

The 39.3 % segmentation mIoU exceeds the random-prediction baseline (~17 % for 6 classes) but remains below the level required for class-specific costmap inflation, particularly for the safety-critical glass class.

## Bottleneck Analysis

Per-pixel error analysis attributed the V3 RMSE primarily to the encoder. MobileNetV3-Small ([Howard et al. 2019](https://arxiv.org/abs/1905.02244)) provides ~1.5 × 10⁶ parameters tuned for ImageNet classification. The skip features available to the decoder originate from heavily downsampled intermediate stages, which limits the network's ability to recover fine spatial structure at depth boundaries. Errors concentrated at corridor edges where depth changes rapidly (floor-to-wall transitions, near-to-far horizon transitions) and dominated the aggregate RMSE.

This bottleneck motivated the encoder substitution evaluated at [V4](v4-efficientvit-swap), where the same recipe with EfficientViT-B1 in place of MobileNetV3-Small reduces NYU val RMSE by 33 %.

## Reproducibility and Provenance

The V3 configuration is preserved under `archive/v1-v3-baseline/` and is reproducible via the `v1-v3-baseline` git tag. The encoder interface was deprecated at V4; reproducing V3 against the current main branch requires the archived codebase.

## Findings

V3 establishes the loss formulation and optimizer configuration used for the remainder of the lineage. All subsequent configurations (V4 through V9) inherit the berHu loss, Kendall weighting with `[-2, 2]` clamp, two-rate optimizer with `ENCODER_LR_SCALE = 0.1`, encoder freeze for the first 5 epochs, AdamW optimizer, and cosine annealing schedule. The training recipe was set at V3 and did not change again.

→ [V4](v4-efficientvit-swap) reports the encoder substitution that reduced NYU val RMSE from 1.160 m to 0.774 m at fixed recipe.
