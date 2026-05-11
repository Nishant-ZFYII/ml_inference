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
- semantic-segmentation
- efficientvit
- bootstrap-perception
- vortex-depth
datasets:
- sayakpaul/nyu_depth_v2
metrics:
- rmse
- mae
- miou
model-index:
- name: vortex-depth-v5-general
  results:
  - task:
      type: depth-estimation
      name: Monocular Indoor Depth Estimation
    dataset:
      name: NYU Depth V2 (val)
      type: nyu_depth_v2
    metrics:
    - type: rmse
      value: 0.572
      name: NYU val RMSE (m)
    - type: mIoU
      value: 63.7
      name: 6-class Segmentation mIoU (%)
---

# Vortex-Depth-V5-General (Atlas)

A 5.31 × 10⁶ parameter monocular depth + 6-class segmentation student model for general-purpose indoor depth estimation. The recommended deployable checkpoint of the Vortex-Depth lineage for unconstrained indoor scenes (apartments, kitchens, offices, mixed room geometries).

| Property | Value |
|---|---|
| Codename | **Atlas** |
| Lineage version | V5 |
| Architecture | EfficientViT-B1 encoder + dual transposed-convolution decoder |
| Parameters | 5.31 × 10⁶ |
| Input | RGB, 240 × 320, ImageNet-normalized within forward pass |
| Output | depth `[B, 1, 240, 320]` in meters; segmentation `[B, 6, 240, 320]` logits |
| Training corpus | NYU Depth V2 with deployment-targeted augmentation pipeline |
| Teacher | DA3-Metric-Large |
| Loss | berHu (depth) + cross-entropy (segmentation) + edge-aware smoothness, Kendall-weighted |
| Inference latency | ~5 ms on Jetson Orin Nano (TensorRT FP16) |

## Use case

Recommended for general indoor depth estimation across diverse room geometries. This checkpoint is the lineage's most well-rounded model on standard indoor benchmarks:

- NYU val RMSE: **0.572 m**
- NYU val mIoU (6-class: floor, wall, person, furniture, glass, other): **63.7 %**

For corridor-class environments specifically, the [vortex-depth-v9-corridor (Lighthouse)](https://huggingface.co/NishantPushparaju/vortex-depth-v9-corridor) checkpoint achieves 0.382 m corridor RMSE and is the recommended choice when the deployment domain is restricted to corridors.

For users intending to fine-tune for additional domain specialists, the [vortex-depth-v6-pretrained (Cornerstone)](https://huggingface.co/NishantPushparaju/vortex-depth-v6-pretrained) checkpoint is the recommended initialization.

## Loading

```python
import torch
from models.student import build_student   # from the Vortex codebase
from config import Config

cfg = Config()
model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False, backbone=cfg.BACKBONE)
state = torch.load("best_depth_v5.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# Inference
with torch.no_grad():
    depth, seg_logits = model(rgb_tensor)  # rgb_tensor: [B, 3, 240, 320]
```

## Training

The configuration applies three augmentation operations to RGB inputs at training time, on top of the V4 baseline:

- Horizontal flip (probability 0.5)
- ColorJitter: brightness ± 0.2, contrast ± 0.2, saturation ± 0.2, hue ± 0.1
- Random crop or bilinear resize to 240 × 320

Training schedule: AdamW optimizer with encoder LR 3 × 10⁻⁵ and decoder LR 3 × 10⁻⁴ (10 × encoder LR), cosine annealing over 200 epochs, batch size 16. Encoder frozen for the first 5 epochs.

Training was performed on NVIDIA L40S 48 GB hardware (NYU Greene HPC, partition `l40s_public`), HPC job 3070058.

## Bootstrap perception context

This checkpoint is one component of a three-checkpoint family released as part of the Vortex bootstrap-perception pipeline for indoor robot navigation under hardware depth failure. The pipeline addresses the operational reality that Time-of-Flight depth sensors lose ~78 % of their pixels on reflective indoor surfaces (polished floors, glass walls). The student model fills the dead pixels with consistent learned geometry; runtime fusion combines surviving sensor pixels with the student output.

The deployment pipeline applies confidence-gated fusion: where the ToF confidence map exceeds 0.5 and depth lies in [0.05, 10.0] m, the sensor reading is used directly; elsewhere the student depth (median-scale aligned to surviving pixels per frame) is used.

## Project resources

- **Codebase**: [github.com/Nishant-ZFYII/ml_inference](https://github.com/Nishant-ZFYII/ml_inference)
- **Documentation**: [nishant-zfyii.github.io/ml_inference](https://nishant-zfyii.github.io/ml_inference/)
- **V5 model page**: [Atlas (V5)](https://nishant-zfyii.github.io/ml_inference/models/v5-deployment-aug)

## Reference

If you use this model in your work, please reference the project repository:

```
https://github.com/Nishant-ZFYII/ml_inference
```

## License

MIT.
