---
layout: default
title: "V6: Multi-Domain Pretraining with NYU Fine-Tuning"
version: 6
backbone: EfficientViT-B1
teacher: DA3-Metric-Large
nyu_rmse: 0.519
status: production-finetune-base
---

# V6: Multi-Domain Pretraining Followed by NYU Fine-Tuning

V6 introduces a two-stage training schedule: multi-domain pretraining on SUN RGB-D and DIODE Indoor, followed by NYU Depth V2 fine-tuning under the [V5](v5-deployment-aug) augmentation pipeline. The configuration achieves 0.519 m NYU validation RMSE (the lowest NYU result in the lineage) and serves as the fine-tuning base for the corridor specialist [V9](v9-corridor-specialist).

## Configuration

| Property | Value |
|---|---|
| Architecture | EfficientViT-B1 (unchanged from V4) |
| Stage 1 corpus | SUN RGB-D (~10 × 10³ frames) + DIODE Indoor (~8 × 10³ frames) |
| Stage 2 corpus | NYU Depth V2 (1,159 train frames) with V5 augmentation pipeline |
| Loss formulation | berHu + cross-entropy + edge-aware smoothness, Kendall-weighted |
| Optimizer | AdamW; encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴ |
| Stage 1 HPC job | 3093046 (38 / 50 epochs, terminated by scheduler — see Operational History) |
| Stage 2 HPC job | 3098656 (154 / 200 epochs, terminated at walltime) |
| Failed sibling jobs | 3084108, 3088530 |
| **NYU val RMSE** | **0.519 m** (lowest NYU result in lineage) |
| NYU val mIoU (6-class) | 48.5 % (regression vs V5's 63.7 %; see Mixed-Supervision Effects) |
| Femto Bolt corridor RMSE | 2.158 m (statistically equivalent to V5's 2.186 m) |
| **HuggingFace identifier** | [`NishantPushparaju/vortex-depth-v6-pretrained`](https://huggingface.co/NishantPushparaju/vortex-depth-v6-pretrained) |
| **Codename** | **Cornerstone** (the foundation other specialists build on) |
| Checkpoint | `hpc_outputs/best_depth_v6.pt` |

## Method

The configuration applies a two-stage training schedule designed to broaden the encoder's prior over indoor depth structure before specialization on the primary training corpus.

<div class="mermaid">
flowchart TB
    subgraph Stage1["Stage 1: Multi-domain pretraining"]
        direction TB
        SUN["SUN RGB-D<br/>~10 × 10³ frames<br/>(no segmentation labels)"]
        DIODE["DIODE Indoor<br/>~8 × 10³ frames<br/>(no segmentation labels)"]
        NYU1["NYU Depth V2<br/>1.159 × 10³ frames<br/>(with 6-class segmentation)"]
        S1MIX["Combined ~19 × 10³ frames<br/>Segmentation loss skipped<br/>where labels absent"]
        SUN --> S1MIX
        DIODE --> S1MIX
        NYU1 --> S1MIX
        S1OUT["Job 3093046<br/>38 / 50 epochs<br/>(scheduler termination)"]
        S1MIX --> S1OUT
    end

    subgraph Stage2["Stage 2: NYU fine-tuning"]
        direction TB
        NYU2["NYU Depth V2<br/>+ V5 augmentation pipeline"]
        S2OUT["Job 3098656<br/>154 / 200 epochs<br/>(walltime termination)"]
        NYU2 --> S2OUT
    end

    Stage1 --> CKPT["V6 pretrain checkpoint<br/>(encoder prior established)"]
    CKPT --> Stage2
    Stage2 --> RESULT["V6 final checkpoint<br/>NYU val RMSE: 0.519 m<br/>NYU val mIoU: 48.5 %"]

    FAIL1["Job 3084108<br/>NaN loss<br/>(ignore-index edge case)"]
    FAIL2["Job 3088530<br/>GPU utilization kill<br/>(dataloader bottleneck)"]
    FAIL1 -.->|"informed loss-function fix"| Stage1
    FAIL2 -.->|"informed dataloader profiling"| Stage1

    style Stage1 fill:#e8f0ff
    style Stage2 fill:#fff3cd
    style RESULT fill:#d4e7c5
    style FAIL1 fill:#fde2e2
    style FAIL2 fill:#fde2e2
</div>

*Diagram source: [`assets/diagrams/models/v6-pretrain-finetune-timeline.mmd`](../assets/diagrams/models/v6-pretrain-finetune-timeline.mmd).*

### Stage 1: Multi-domain pretraining

SUN RGB-D and DIODE Indoor are folded into the training mixture alongside NYU Depth V2. Both auxiliary datasets provide metric depth ground truth that does not require teacher inference. The objective is to expose the encoder to indoor *texture and lighting variance* not represented in NYU's apartment-scale scenes.

| Dataset | Approximate frame count | Segmentation labels | Role |
|---|---|---|---|
| SUN RGB-D | 10 × 10³ | None (all pixels = ignore index 255) | Texture diversity |
| DIODE Indoor | 8 × 10³ | None (all pixels = ignore index 255) | Texture diversity |
| NYU Depth V2 | 1.159 × 10³ | 6-class | Primary supervision |

The auxiliary datasets do not provide segmentation supervision; pixel labels are uniformly set to the ignore index. The segmentation head receives no gradient on auxiliary-dataset batches, requiring the cross-entropy loss to handle all-ignore conditions safely (see Loss-Function Edge Case below).

### Stage 2: NYU fine-tuning

The auxiliary datasets are removed. Training continues on NYU Depth V2 alone with the V5 augmentation pipeline (horizontal flip + ColorJitter + random crop / resize). The pretrain establishes a richer encoder prior; the fine-tune adapts the encoder to NYU's scene statistics and re-engages the segmentation head.

## Quantitative Results

| Metric | V6 | V5 | Δ |
|---|---|---|---|
| NYU val RMSE | **0.519 m** | 0.572 m | −9.3 % |
| NYU val mIoU (6-class) | 48.5 % | 63.7 % | −15.2 pp |
| Femto Bolt corridor RMSE | 2.158 m | 2.186 m | −1.3 % (statistically equivalent) |

The NYU val RMSE improvement of 9.3 % at fixed architecture is attributable to the multi-domain pretraining stage. The improvement is smaller than V4 → V5 (26 %) but is the second-largest improvement on the NYU validation metric in the lineage. The corridor RMSE is statistically unchanged from V5: the multi-domain pretrain does not improve corridor performance directly, since the auxiliary datasets share the apartment-scale geometry of NYU rather than the corridor-scale geometry of the deployment environment.

## Operational History

The V6 result of 0.519 m NYU val RMSE was produced by a training program that included two scheduler-terminated runs, one walltime termination, and one configuration error. The full sequence:

| HPC job | Outcome | Cause | Effect |
|---|---|---|---|
| 3084108 | Terminated | NaN loss propagation | Triggered investigation; produced the loss-function edge-case fix |
| 3088530 | Terminated by scheduler | GPU utilization below partition threshold | Identified as dataloader bottleneck on parallel filesystem |
| 3093046 | Terminated by scheduler at 38 / 50 epochs | Same GPU utilization issue | **Snapshotted as Stage 1 pretrain checkpoint** |
| 3098656 | Walltime timeout at 154 / 200 epochs | 12-hour partition limit | **Snapshotted as Stage 2 (V6 final) checkpoint** |

The final V6 checkpoint represents two terminated runs producing usable artifacts. Subsequent training programs include explicit dataloader profiling at job submission time and partition-aware walltime budgeting to reduce reliance on terminated-run snapshots.

## Loss-Function Edge Case

Job 3084108 was terminated by NaN loss propagation. The cause was a corner case in PyTorch's `nn.CrossEntropyLoss(ignore_index=255)`: when every pixel in a batch is set to the ignore index, the reduction divides by zero and returns NaN. The NaN propagates through the multi-task loss and corrupts the optimizer state within a small number of iterations.

The condition occurs whenever a batch consists entirely of frames from datasets without segmentation supervision (SUN RGB-D, DIODE), which becomes likely under randomized batch sampling. The fix:

```python
# Within MultiTaskLoss.forward
has_valid_seg = (seg_target != 255).any()
if has_valid_seg:
    L_seg = self.ce(pred_seg, seg_target)
else:
    L_seg = pred_seg.sum() * 0.0   # Returns zero with gradient connectivity preserved
```

The guard is now applied across all multi-domain training configurations.

## Mixed-Supervision Effects

The 15.2 pp regression in NYU val mIoU (V6: 48.5 % vs V5: 63.7 %) is attributable to mixed-supervision effects during Stage 1 pretraining. The segmentation head receives effective gradient on only the NYU subset of pretrain batches (~6 % of frames). By the start of Stage 2 fine-tuning, the segmentation decoder has drifted toward zero-output predictions on the auxiliary-dataset distribution and must re-learn the 6-class structure on NYU. The 154-epoch fine-tune (truncated by walltime) does not fully recover the V5 mIoU.

This is a documented tradeoff: the depth head benefits from the pretrain (richer encoder representations transfer); the segmentation head pays for the auxiliary-dataset supervision gap. The configuration accepts the segmentation regression because depth is the load-bearing output for the costmap fusion pipeline. For deployments where segmentation accuracy is primary, [V5](v5-deployment-aug) is the preferred checkpoint.

## Comparative Performance as Fine-Tuning Base

V6's primary value lies in its role as a fine-tuning base for domain-specialized configurations. The V6 vs V5 comparison as initialization for corridor specialization:

| Specialist | Initialization | LILocBench corridor RMSE | NYU val RMSE after fine-tune |
|---|---|---|---|
| [V7](v7-lilocbench-finetune) | V5 | 0.445 m | 1.315 m |
| [V9](v9-corridor-specialist) | **V6** | **0.382 m** | 1.553 m |

Initializing from V6 rather than V5 produces a 14 % relative improvement in LILocBench corridor RMSE at fixed fine-tuning protocol. The richer encoder prior established by the V6 pretrain transfers to corridor specialization more effectively than V5's NYU-only training.

## Demonstration

<video controls width="100%" preload="metadata">
  <source src="{{ '/assets/videos/v6_6panel.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the embedded video. <a href="{{ '/assets/videos/v6_6panel.mp4' | relative_url }}">Download MP4</a>.
</video>

Six-panel video sequence over the 459-frame `corridor_eval` set. **Top row (raw inputs and reference)**: RGB input · raw Femto Bolt ToF depth · zero-shot DA3-Small reference depth (median-scale aligned). **Bottom row (V6-specific predictions and fusions)**: V6 student raw inference · confidence-gated fusion of ToF and DA3 · confidence-gated fusion of ToF and V6. V6's prediction structure on the corridor distribution is comparable to V5 (both general-purpose models with similar Femto Bolt corridor RMSE: 2.158 m and 2.186 m, respectively). V6 (Cornerstone) is the recommended fine-tuning base for additional corridor specialists rather than the recommended deployment checkpoint for corridor environments — see [V9](v9-corridor-specialist) for the production corridor model.

## Findings

V6 supports two findings on training-program design:

1. **Multi-domain pretraining transfers to downstream specialization quality.** The 14 % LILocBench corridor RMSE improvement between V7 (V5-initialized) and V9 (V6-initialized) under fixed fine-tuning protocol isolates the effect of the pretraining stage on the specialization endpoint.

2. **Mixed-dataset training requires explicit handling of supervision gaps.** Datasets without complete annotation coverage produce loss-function edge cases (NaN propagation under ignore-index reductions) that surface only at runtime. Training programs combining sources with heterogeneous annotation must include explicit guards against zero-supervision conditions per loss term.

→ [V7](v7-lilocbench-finetune) reports the corridor specialization initialized from V5, establishing the baseline against which V9 is compared. → [V9](v9-corridor-specialist) reports the corridor specialization initialized from V6, the recommended deployment checkpoint for corridor environments.
