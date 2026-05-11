---
layout: default
title: "V9: Corridor-Specialized Student"
version: 9
backbone: EfficientViT-B1
teacher: DA3-Metric-Large
nyu_rmse: 1.553
lilocbench_rmse: 0.382
status: production
---

# V9: Corridor-Specialized Student via Sequential Domain Adaptation

V9 is the production checkpoint for corridor-class indoor environments. The configuration is obtained by fine-tuning the [V6 multi-domain pretrained model](v6-sun-diode-pretrain) on the LILocBench corridor dataset, yielding a 0.382 m RMSE specialist that achieves closed-loop navigation parity with ground-truth depth in simulation.

## Configuration

| Property | Value |
|---|---|
| Architecture | EfficientViT-B1 encoder, dual transposed-convolution decoder (depth + segmentation heads) |
| Trainable parameters | 5.31 × 10⁶ |
| Initialization | V6 checkpoint (SUN RGB-D + DIODE pretraining → NYU Depth V2 fine-tuning) |
| Fine-tuning corpus | LILocBench `dynamics_0` split, Intel RealSense D455 |
| Optimizer | AdamW; encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴ |
| Schedule | Cosine annealing, 50 epochs |
| Batch size | 16 |
| Loss formulation | berHu (depth) + cross-entropy (segmentation) + edge-aware smoothness, Kendall-weighted |
| Compute | NVIDIA L40S 48 GB (NYU Greene HPC, partition `l40s_public`) |
| Checkpoint artifact | `hpc_outputs/best_depth_v9.pt` |
| **HuggingFace identifier** | [`NishantPushparaju/vortex-depth-v9-corridor`](https://huggingface.co/NishantPushparaju/vortex-depth-v9-corridor) |
| **Codename** | **Lighthouse** (the production beacon for corridor environments) |

## Method

The V9 configuration differs from [V7](v7-lilocbench-finetune) by a single experimental variable: the source checkpoint used for fine-tuning. V7 initializes from [V5](v5-deployment-aug), which was trained on NYU Depth V2 alone. V9 initializes from V6, which received an additional pretraining stage on SUN RGB-D and DIODE Indoor before NYU fine-tuning. All other hyperparameters, augmentation pipelines, loss formulations, and training schedules are held constant between the two configurations, isolating the effect of the initialization choice.

The complete training pipeline that produces V9 spans four sequential stages, each progressively narrowing the data distribution:

<div class="mermaid">
flowchart LR
    S0["ImageNet pretraining<br/>(timm default)<br/>~1.3M classification labels"]
    S1["Stage 1: Multi-domain pretraining<br/>SUN RGB-D + DIODE Indoor<br/>~18k indoor depth frames<br/>(V6 pretrain checkpoint)"]
    S2["Stage 2: NYU fine-tuning<br/>NYU Depth V2<br/>1,159 train frames + augmentations<br/>(V6 final checkpoint)"]
    S3["Stage 3: Corridor specialization<br/>LILocBench dynamics_0<br/>~5k corridor frames<br/>(V9 final checkpoint)"]

    S0 --> S1 --> S2 --> S3

    style S0 fill:#f5f5f5
    style S1 fill:#e8f0ff
    style S2 fill:#e8f0ff
    style S3 fill:#d4e7c5
</div>

The first three stages are inherited from V6. Only the final corridor-specialization stage is unique to V9.

## Quantitative Results

| Metric | V9 | V7 | V5 (V7 init) | V6 (V9 init) |
|---|---|---|---|---|
| LILocBench corridor RMSE (D455) | **0.382 m** | 0.445 m | — | — |
| Femto Bolt corridor RMSE (deployment camera) | 1.589 m | 1.982 m | 2.186 m | 2.158 m |
| NYU val RMSE | 1.553 m | 1.315 m | 0.572 m | 0.519 m |
| NYU val mIoU (6-class) | 31.6 % | 47.5 % | 63.7 % | 48.5 % |

The 14 % relative improvement in LILocBench RMSE between V7 and V9, with all other hyperparameters held constant, is attributable to the initialization choice. This isolates the effect of multi-domain pretraining on subsequent specialization quality. The corresponding regression in NYU val RMSE — from V6's 0.519 m to V9's 1.553 m — is a documented consequence of single-domain fine-tuning under standard catastrophic-forgetting dynamics.

LILocBench and Femto Bolt corridor RMSE are not directly comparable: the two measurements use different cameras (Intel RealSense D455 for LILocBench, Orbbec Femto Bolt for the deployment recordings) with distinct intrinsics, baselines, and noise characteristics. Both numbers are reported throughout this technical report with the camera identifier annotated.

## Closed-Loop Validation

Closed-loop navigation experiments were conducted in a Gazebo Fortress corridor environment under two depth-source configurations:

(a) Ground-truth depth from the simulator's perfect depth sensor.
(b) V9 inference at runtime via TensorRT FP16, consumed through the deployment fusion pipeline.

Ten random seeds were evaluated per configuration with identical start pose, goal pose, and Nav2 controller settings.

| Depth source | Success rate | Time-to-goal (s) | Collisions |
|---|---|---|---|
| Ground-truth depth | 9 / 10 | 17.77 ± 0.90 | 0 |
| V9 (TensorRT FP16) | 9 / 10 | 17.99 ± 0.45 | 0 |

V9 achieves navigation parity with ground-truth depth under this protocol. The single failure case is shared between configurations (identical seed, equivalent final pose error), indicating a planner-level failure mode rather than a depth-quality difference attributable to V9.

## Runtime Integration

At deployment, V9 depth is consumed by a two-stage fusion procedure executed per frame:

<div class="mermaid">
flowchart LR
    RGB["RGB frame<br/>1280 × 720"] --> DOWN["Downsample<br/>240 × 320"]
    DOWN --> V9["V9 inference<br/>(TensorRT FP16, ~5 ms)"]
    V9 --> SDEPTH["Predicted depth<br/>240 × 320"]
    SDEPTH --> UP["Upsample<br/>1280 × 720"]

    TOF["ToF depth<br/>1280 × 720<br/>(~22% valid)"] --> CALIB
    CONF["ToF confidence"] --> CALIB
    UP --> CALIB

    CALIB["Median-scale calibration<br/>s = median(d_ToF / d_V9)<br/>over conf ≥ 0.5 pixels"]
    CALIB --> SCALED["Scaled prediction<br/>s · d_V9"]

    GATE{"Per-pixel gate:<br/>conf ≥ 0.5 AND<br/>0.05 ≤ d_ToF ≤ 10.0 m"}
    TOF --> GATE
    CONF --> GATE
    SCALED --> GATE

    GATE -->|"valid ToF"| FUSED["Fused depth"]
    GATE -->|"otherwise"| FUSED

    FUSED --> COSTMAP["Nav2 local costmap"]

    style V9 fill:#d4e7c5
    style CALIB fill:#fff3cd
    style GATE fill:#fff3cd
    style FUSED fill:#e8f0ff
</div>

The per-frame scale factor `s = median(d_ToF / d_V9)` is computed over pixels where the Femto Bolt confidence map exceeds 0.5. Substitution policy: a ToF pixel is used directly when its confidence exceeds 0.5 and its depth falls within [0.05 m, 10.0 m]; otherwise the scaled student prediction `s · d_V9` is used. This calibration corrects absolute scale per frame, requiring V9 to provide accurate depth structure rather than absolute depth magnitude.

Full runtime parameters and the rationale for hard substitution rather than weighted blending are documented in [Confidence-Gated Fusion](../concepts/confidence-gated-fusion).

## Operational Limitations

V9 is specialized for corridor-class indoor environments. Performance regressions occur predictably under the following conditions:

| Condition | Mechanism | Recommended alternative |
|---|---|---|
| Open rooms or large unbounded spaces | V9 expects depth compression against a back wall in the 6–15 m range; absent that geometry, predictions become structurally inconsistent | [V5](v5-deployment-aug) or [V6](v6-sun-diode-pretrain) |
| Reflective surfaces outside the training distribution (granite, polished metal, water, large mirrors) | V9 has been exposed to glass via LILocBench but has not generalized to other specular surfaces at comparable scale | DA3-Small zero-shot inference |
| Outdoor scenes | Out of training distribution | Not recommended |
| General room-scene segmentation | mIoU declines to 31.6 % from V5's 63.7 % as a consequence of fine-tuning prioritization | [V5](v5-deployment-aug) |

End-to-end verification within the [Docker reproducibility container](../docker) reports a raw V9 RMSE of 1.366 m on Femto Bolt sensor pixels (uncalibrated, no median-scale alignment) over 459 corridor frames. With per-frame median-scale alignment, the oracle RMSE reduces to 0.522 m. The full sensitivity analysis across calibration set sizes is reported in the [calibration study](../calibration).

## Architecture

V9 inherits the network architecture established at [V4](v4-efficientvit-swap). The encoder is EfficientViT-B1 from `timm` (5.31 × 10⁶ parameters); two parallel transposed-convolution decoders share skip connections from each encoder stage.

<div class="mermaid">
flowchart LR
    IN["RGB input<br/>[B, 3, 240, 320]"]
    NORM["ImageNet normalization<br/>(within forward pass)"]
    ENC["EfficientViT-B1<br/>encoder"]
    NECK["Neck<br/>Conv2d(256, 128, 1)"]

    DD["Depth decoder<br/>3 × DecoderBlock<br/>+ Conv2d(16, 1) + ReLU"]
    SD["Segmentation decoder<br/>3 × DecoderBlock<br/>+ Conv2d(16, 6)"]

    DEPTH["Depth output<br/>[B, 1, 240, 320]<br/>metric (m)"]
    SEG["Segmentation logits<br/>[B, 6, 240, 320]"]

    IN --> NORM --> ENC --> NECK
    NECK --> DD
    NECK --> SD
    DD --> DEPTH
    SD --> SEG
    ENC -.skip features.-> DD
    ENC -.skip features.-> SD

    style ENC fill:#e8f0ff
    style NECK fill:#fff3cd
    style DEPTH fill:#d4e7c5
    style SEG fill:#d4e7c5
</div>

`DecoderBlock(in, out)` implements `Upsample(2×) → Conv2d(in + skip, out, 3×3) → BatchNorm → ReLU`, with the skip connection drawn from the encoder stage of matching spatial resolution. ImageNet normalization is applied within the forward pass to ensure inference-time preprocessing parity with training.

## Demonstration

<video controls width="100%" preload="metadata">
  <source src="{{ '/assets/videos/v9_6panel.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the embedded video. <a href="{{ '/assets/videos/v9_6panel.mp4' | relative_url }}">Download MP4</a>.
</video>

Six-panel video sequence over the 459-frame `corridor_eval` set. **Top row (raw inputs and reference)**: RGB input from the deployment camera · raw Femto Bolt ToF depth (showing the dead-pixel pattern that motivates the bootstrap-perception design) · zero-shot DA3-Small reference depth (median-scale aligned, foundation-model baseline). **Bottom row (V9-specific predictions and fusions)**: V9 student raw inference · confidence-gated fusion of ToF and DA3 (the foundation-model deployment baseline) · confidence-gated fusion of ToF and V9 (the production deployment realization). The bottom-right panel is the depth signal consumed by the local costmap; comparing it against the bottom-middle panel isolates the V9 (Lighthouse) contribution against the DA3 fusion baseline.

A side-by-side comparison across all student configurations on the same sequence is available on the [demo videos page](../videos).

## Discussion

The V9 configuration supports three claims, each with corresponding evidence within this technical report:

1. **Initialization choice has measurable effect on specialization quality.** Comparing V9 to V7 at fixed protocol, multi-domain pretraining as initialization yields a 14 % relative reduction in LILocBench corridor RMSE (0.445 m → 0.382 m).
2. **Sequential pretraining followed by specialization outperforms simultaneous joint-domain training for distant-domain transfer.** V9 corridor RMSE (1.589 m on Femto Bolt) improves on the joint-training [V8](v8-mixed-failed) configuration (2.266 m) at comparable training cost.
3. **A 5.31 × 10⁶ parameter student trained via this pipeline achieves closed-loop navigation parity with ground-truth depth** in simulated corridor environments under matched seeds and controller configuration.

V9 is the recommended checkpoint for fixed-route corridor deployment. For general indoor navigation outside the corridor distribution, [V5](v5-deployment-aug) or [V6](v6-sun-diode-pretrain) is preferred. The pipeline supports re-specialization for additional target environments by fine-tuning V6 on representative data from the new domain.
