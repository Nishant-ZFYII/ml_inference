---
license: mit
library_name: pytorch
pipeline_tag: depth-estimation
tags:
- depth-estimation
- monocular-depth
- knowledge-distillation
- robotics
- indoor-navigation
- fine-tuning-base
- semantic-segmentation
- efficientvit
- pretraining
- multi-domain
- bootstrap-perception
- vortex-depth
datasets:
- sayakpaul/nyu_depth_v2
- sun_rgbd
- diode
metrics:
- rmse
- mae
- miou
model-index:
- name: vortex-depth-v6-pretrained
  results:
  - task:
      type: depth-estimation
      name: Monocular Indoor Depth Estimation
    dataset:
      name: NYU Depth V2 (val)
      type: nyu_depth_v2
    metrics:
    - type: rmse
      value: 0.519
      name: NYU val RMSE (m)
    - type: mIoU
      value: 48.5
      name: 6-class Segmentation mIoU (%)
---

# Vortex-Depth-V6-Pretrained (Cornerstone)

A 5.31 × 10⁶ parameter monocular depth + 6-class segmentation student trained with multi-domain pretraining on SUN RGB-D and DIODE Indoor, followed by NYU Depth V2 fine-tuning. The recommended fine-tuning base for additional domain specialists in the Vortex-Depth lineage.

| Property | Value |
|---|---|
| Codename | **Cornerstone** |
| Lineage version | V6 |
| Architecture | EfficientViT-B1 encoder + dual transposed-convolution decoder |
| Parameters | 5.31 × 10⁶ |
| Input | RGB, 240 × 320, ImageNet-normalized within forward pass |
| Output | depth `[B, 1, 240, 320]` in meters; segmentation `[B, 6, 240, 320]` logits |
| Stage 1 corpus | SUN RGB-D (~10 × 10³ frames) + DIODE Indoor (~8 × 10³ frames) |
| Stage 2 corpus | NYU Depth V2 (1.159 × 10³ train) with V5 augmentation pipeline |
| Teacher | DA3-Metric-Large |
| Inference latency | ~5 ms on Jetson Orin Nano (TensorRT FP16) |

## Use case

Recommended as the **fine-tuning base** for users developing additional domain-specialist depth models. The multi-domain pretraining stage establishes a richer encoder prior than NYU-only training, which transfers to subsequent domain specialization more effectively.

This is demonstrated empirically in the lineage: a corridor specialist fine-tuned from V6 ([vortex-depth-v9-corridor (Lighthouse)](https://huggingface.co/NishantPushparaju/vortex-depth-v9-corridor)) achieves 0.382 m LILocBench corridor RMSE, a 14 % relative improvement over the same fine-tuning protocol applied to the V5-initialized variant (V7: 0.445 m).

V6 itself is the lineage's best NYU result:
- NYU val RMSE: **0.519 m** (lowest in lineage)
- NYU val mIoU (6-class): 48.5 %

The mIoU regression relative to [vortex-depth-v5-general (Atlas)](https://huggingface.co/NishantPushparaju/vortex-depth-v5-general) (63.7 %) is an artifact of mixed-supervision effects during multi-domain pretraining (SUN RGB-D and DIODE Indoor have no segmentation annotations). For deployments where general-purpose segmentation accuracy matters, V5 (Atlas) is the recommended checkpoint.

## Fine-tuning template

To produce a corridor or room specialist from this base:

```python
import torch
from models.student import build_student
from config import Config

cfg = Config()
cfg.LR = 3e-4
cfg.ENCODER_LR_SCALE = 0.1   # encoder LR = 3e-5

model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False, backbone=cfg.BACKBONE)
state = torch.load("best_depth_v6.pt", map_location="cpu")
model.load_state_dict(state)

# Continue training on your domain-specific corpus for ~30-50 epochs
# Refer to train.py in the project codebase for the full training loop
```

## Training

Two-stage training schedule:

- **Stage 1 (multi-domain pretrain)**: 50 epochs across SUN RGB-D + DIODE Indoor + NYU Depth V2 with the V5 augmentation pipeline. HPC job 3093046, snapshotted at 38 / 50 epochs.
- **Stage 2 (NYU fine-tune)**: 200 epochs on NYU Depth V2 alone with the same augmentation pipeline. HPC job 3098656, snapshotted at 154 / 200 epochs (walltime).

Optimizer: AdamW, encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴, cosine annealing, batch size 16, encoder frozen for the first 5 epochs.

The Stage 1 pretrain required a loss-function guard for the cross-entropy term: SUN RGB-D and DIODE Indoor have no segmentation annotations, so the segmentation supervision is skipped on those batches. Without the guard, `nn.CrossEntropyLoss(ignore_index=255)` returns NaN on all-ignore batches and propagates through the multi-task loss, crashing the optimizer.

Training was performed on NVIDIA L40S 48 GB hardware (NYU Greene HPC, partition `l40s_public`).

## Bootstrap perception context

This checkpoint is one component of a three-checkpoint family released as part of the Vortex bootstrap-perception pipeline for indoor robot navigation under hardware depth failure. See [vortex-depth-v5-general (Atlas)](https://huggingface.co/NishantPushparaju/vortex-depth-v5-general) for the recommended general-purpose deployment checkpoint, and [vortex-depth-v9-corridor (Lighthouse)](https://huggingface.co/NishantPushparaju/vortex-depth-v9-corridor) for the production corridor specialist derived from this base.

## Project resources

- **Codebase**: [github.com/Nishant-ZFYII/ml_inference](https://github.com/Nishant-ZFYII/ml_inference)
- **Documentation**: [nishant-zfyii.github.io/ml_inference](https://nishant-zfyii.github.io/ml_inference/)
- **V6 model page**: [Cornerstone (V6)](https://nishant-zfyii.github.io/ml_inference/models/v6-sun-diode-pretrain)

## Citation

```bibtex
@misc{vortex-depth-2026,
  title={Vortex-Depth: Bootstrap Perception Under Hardware Depth Failure for Indoor Robot Navigation},
  author={Pushparaju, Nishant and Mattam, Vivekananda Swamy and Arab, Aliasghar},
  year={2026},
  url={https://github.com/Nishant-ZFYII/ml_inference}
}
```

## License

MIT.
