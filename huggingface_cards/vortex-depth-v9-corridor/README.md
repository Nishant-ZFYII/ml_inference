---
license: mit
library_name: pytorch
pipeline_tag: depth-estimation
base_model: NishantPushparaju/vortex-depth-v6-pretrained
tags:
- depth-estimation
- monocular-depth
- knowledge-distillation
- robotics
- indoor-navigation
- corridor
- ros2
- semantic-segmentation
- efficientvit
- fine-tuned
- domain-specialist
- bootstrap-perception
- vortex-depth
datasets:
- lilocbench
- sayakpaul/nyu_depth_v2
metrics:
- rmse
- mae
model-index:
- name: vortex-depth-v9-corridor
  results:
  - task:
      type: depth-estimation
      name: Corridor Depth Estimation
    dataset:
      name: LILocBench dynamics_0 (RealSense D455)
      type: lilocbench
    metrics:
    - type: rmse
      value: 0.382
      name: LILocBench Corridor RMSE (m)
  - task:
      type: depth-estimation
      name: Monocular Indoor Depth Estimation
    dataset:
      name: NYU Depth V2 (val)
      type: nyu_depth_v2
    metrics:
    - type: rmse
      value: 1.553
      name: NYU val RMSE (m)
---

# Vortex-Depth-V9-Corridor (Lighthouse)

A 5.31 × 10⁶ parameter monocular depth + 6-class segmentation student specialized for corridor-class indoor environments. The production corridor checkpoint of the Vortex-Depth lineage. Closed-loop validated against ground-truth depth in simulation.

| Property | Value |
|---|---|
| Codename | **Lighthouse** |
| Lineage version | V9 |
| Architecture | EfficientViT-B1 encoder + dual transposed-convolution decoder |
| Parameters | 5.31 × 10⁶ |
| Input | RGB, 240 × 320, ImageNet-normalized within forward pass |
| Output | depth `[B, 1, 240, 320]` in meters; segmentation `[B, 6, 240, 320]` logits |
| Initialization | [vortex-depth-v6-pretrained (Cornerstone)](https://huggingface.co/NishantPushparaju/vortex-depth-v6-pretrained) |
| Fine-tuning corpus | LILocBench `dynamics_0` (Intel RealSense D455, ~5 × 10³ corridor frames) |
| Teacher | DA3-Metric-Large |
| Inference latency | ~5 ms on Jetson Orin Nano (TensorRT FP16) |

## Headline result

**LILocBench corridor RMSE: 0.382 m** — the lineage's best corridor depth measurement.

**Closed-loop navigation parity with ground-truth depth** in Gazebo Fortress corridor experiments:

| Depth source | Success rate (10 seeds) | Time-to-goal | Collisions |
|---|---|---|---|
| Ground-truth depth | 9 / 10 | 17.77 ± 0.90 s | 0 |
| **V9 (TensorRT FP16)** | **9 / 10** | 17.99 ± 0.45 s | 0 |

The single failure is shared between configurations (identical seed, equivalent final pose error), indicating a planner-level failure mode rather than a depth-quality difference.

## Use case

Recommended for fixed-route corridor deployment in indoor robot navigation. The intended deployment context is corridors with the structural characteristics of typical institutional indoor spaces: long parallel walls, mid-field-dominant depth distributions (1–15 m), uniform or fluorescent lighting. The model has been fine-tuned on the LILocBench Bonn corridor dataset and achieves competitive corridor depth metrics on that distribution.

For environments outside the corridor distribution, see:

- [vortex-depth-v5-general (Atlas)](https://huggingface.co/NishantPushparaju/vortex-depth-v5-general) — general-purpose indoor depth across diverse room geometries
- [vortex-depth-v6-pretrained (Cornerstone)](https://huggingface.co/NishantPushparaju/vortex-depth-v6-pretrained) — fine-tuning base for additional corridor-class specialists

## Operational limitations

V9 is specialized for corridor environments. Predictable performance regressions occur under the following conditions:

| Condition | Mechanism | Recommendation |
|---|---|---|
| Open rooms or large unbounded spaces | The model expects depth to compress against a back wall in the 6–15 m range; absent that geometry, predictions become structurally inconsistent | Use V5 (Atlas) or V6 (Cornerstone) |
| Reflective surfaces outside training distribution (granite, polished metal, water, large mirrors) | The model has been exposed to glass via LILocBench but has not generalized to other specular surfaces at comparable scale | Use DA3-Small zero-shot |
| Outdoor scenes | Out of training distribution | Not recommended |
| General room-scene segmentation | NYU val mIoU degrades to 31.6 % from V5's 63.7 % as a consequence of single-domain fine-tuning | Use V5 (Atlas) |

## Loading

```python
import torch
from models.student import build_student
from config import Config

cfg = Config()
model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False, backbone=cfg.BACKBONE)
state = torch.load("best_depth_v9.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    depth, seg_logits = model(rgb_tensor)  # rgb_tensor: [B, 3, 240, 320]
```

## Deployment realization

At runtime, V9 depth is consumed by a confidence-gated fusion node combining ToF and student outputs per pixel:

```
s = median(d_tof / d_v9)        over pixels where ToF confidence ≥ 0.5

if confidence(i, j) ≥ 0.5 AND 0.05 m < tof_depth(i, j) < 10.0 m:
    fused(i, j) = tof_depth(i, j)            # use hardware reading directly
else:
    fused(i, j) = s · v9_depth(i, j)         # use scaled learned depth
```

The per-frame median-scale step anchors the V9 prediction to metric units using the surviving ToF pixels. The model is therefore not required to produce absolute scale accuracy — it must produce accurate depth *structure*, which the runtime then anchors per frame.

## Training

Fine-tuned from the [V6 multi-domain pretrained checkpoint](https://huggingface.co/NishantPushparaju/vortex-depth-v6-pretrained) on LILocBench `dynamics_0` for 50 epochs. AdamW optimizer, encoder LR 3 × 10⁻⁵, decoder LR 3 × 10⁻⁴, cosine annealing, batch size 16. Loss formulation unchanged from earlier lineage stages: berHu (depth) + cross-entropy (segmentation) + edge-aware smoothness, Kendall-weighted with `log σ²` clamped to `[-2, 2]`.

The complete training pipeline that produces V9 spans four sequential stages:

1. ImageNet pretraining (timm default, ~1.3 × 10⁶ classification labels).
2. Multi-domain pretraining on SUN RGB-D + DIODE Indoor + NYU Depth V2 (V6 pretrain checkpoint).
3. NYU Depth V2 fine-tuning with augmentation pipeline (V6 final checkpoint).
4. LILocBench `dynamics_0` fine-tuning (V9 final checkpoint).

The first three stages are inherited from V6; only the final corridor-specialization stage is unique to V9.

## Bootstrap perception context

This checkpoint is the production corridor model in a three-checkpoint family released as part of the Vortex bootstrap-perception pipeline. The pipeline addresses the operational reality that Time-of-Flight depth sensors lose ~78 % of their pixels on reflective indoor surfaces (polished floors, glass walls); the student fills the dead pixels with consistent learned geometry, and runtime fusion combines surviving sensor pixels with the student output.

## Project resources

- **Codebase**: [github.com/Nishant-ZFYII/ml_inference](https://github.com/Nishant-ZFYII/ml_inference)
- **Documentation**: [nishant-zfyii.github.io/ml_inference](https://nishant-zfyii.github.io/ml_inference/)
- **V9 model page**: [Lighthouse (V9)](https://nishant-zfyii.github.io/ml_inference/models/v9-corridor-specialist)

## Reference

If you use this model in your work, please reference the project repository:

```
https://github.com/Nishant-ZFYII/ml_inference
```

## License

MIT.
