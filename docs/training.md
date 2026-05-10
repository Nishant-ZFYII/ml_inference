---
layout: default
title: Training Pipeline
---

# Training Pipeline: V1 Through V9

This page documents the full student lineage — nine training iterations across two backbone architectures, three dataset mixtures, and two loss formulations. It is written as a story, not a changelog, because the decisions make no sense without the context that produced them.

---

## The problem

We need a small network that takes an RGB image and outputs:
1. Dense metric depth (240×320, meters)
2. 6-class semantic segmentation (floor, wall, glass, person, furniture, other)

The depth output will fuse with whatever ToF pixels survived hardware failure. The segmentation output will modulate costmap inflation (glass gets wider berth than wall). Both heads share a single encoder — we cannot afford two networks on a Jetson Orin Nano at 30 FPS.

---

## Teacher ensemble

Training labels come from three foundation models, none of which run at real time on embedded hardware:

| Teacher | What it produces | Speed on HPC (A100) |
|---|---|---|
| DA3-Metric-Large | Dense metric depth from RGB; scale comes from `focal · raw / 300` | ~12 FPS |
| YOLOv8-Large | Bounding boxes for person, furniture, background | ~30 FPS |
| SAM2-Large | Instance masks from YOLO-prompted boxes; fused with geometric floor/wall/glass heuristics | ~5 FPS |

The teacher pipeline runs on NYU HPC via SLURM. Per-frame outputs are written to `$SCRATCH/nyu_teacher_data/` and linked into a `manifest.jsonl` that the training script reads.

---

## The loss function

The depth head trains on a **hybrid target**: for each pixel, if the ToF sensor reports valid depth above a confidence threshold, the target is the ToF reading. Otherwise, the target is the DA3-Metric-Large prediction. This is not a design choice made for convenience — it mirrors the runtime fusion policy exactly. The student learns to predict what the fused output should look like, not what either sensor alone reports.

The full loss (from `models/losses.py`):

$$\mathcal{L} = w_d \cdot \text{berHu}(d_{pred}, d_{target}) + w_s \cdot \text{CE}(s_{pred}, s_{target}) + w_e \cdot \text{EdgeSmooth}(d_{pred}, I_{rgb})$$

where \\(w_d, w_s, w_e\\) are either fixed or learned via Kendall uncertainty weighting (V3+). The berHu loss transitions from L1 to L2 at the 80th percentile of the per-batch residual — punishes small errors linearly and large errors quadratically.

---

## V1: the disaster

**Backbone:** MobileNetV3-Small. **Teacher:** DA2-Large.

RMSE on NYU val: **75.37 m**. Not a typo.

DA2 outputs *relative* depth — values in [0, 1] with no metric scale. The training loop expected metric depth in meters. Every prediction was off by orders of magnitude. The model learned to predict normalized values, and the RMSE metric computed the gap against meter-scale targets.

The fix was not complicated: switch to DA3-Metric-Large, which has a proper scale anchor. But the lesson was important — relative depth models need an explicit alignment step before you can use them as supervision.

---

## V2: diagnostic only

Experiments with Kendall uncertainty clamping on V1's architecture. The backbone was still MobileNetV3-Small with DA2 targets. Not a useful result — just confirming that loss weighting cannot fix fundamentally wrong supervision.

---

## V3: first real model

**Backbone:** MobileNetV3-Small. **Teacher:** DA3-Large.

**Changes from V1:** berHu loss (replaces MSE), Kendall uncertainty weighting (learns task weights), two-LR optimizer (backbone and decoders at different rates).

**NYU RMSE:** 1.160 m.

This was the first iteration where the numbers started to make physical sense. A meter-scale error on indoor depth (typical range 1–6 m) is meaningful but not great. The model was underfitting — MobileNetV3-Small's encoder did not have enough capacity to capture the spatial structure the depth head needed.

---

## V4: the backbone swap

**Backbone:** EfficientViT-B1. **Teacher:** DA3-Large.

**NYU RMSE:** 0.774 m. **Femto Bolt corridor RMSE:** 1.373 m.

A 33% RMSE reduction just from swapping the encoder. EfficientViT-B1 has 5.31M parameters (vs ~2.5M for MobileNetV3-Small) but runs at comparable latency on the Jetson because its attention mechanism is hardware-friendly. The lesson: once the loss and supervision are right, the bottleneck shifts to backbone capacity. This is obvious in retrospect, but V1-V3 were too broken to expose it.

---

## V5: augmentations

**Backbone:** EfficientViT-B1. **Teacher:** DA3-Large.

**Changes from V4:** deployment augmentations (color jitter, horizontal flip, random crop).

**NYU RMSE:** 0.572 m. **Femto Bolt corridor RMSE:** 2.186 m.

The biggest single-version improvement: -26% NYU RMSE from augmentations alone. But corridor RMSE *increased* from 1.373 m to 2.186 m. The augmentations helped the model generalize on NYU's diverse indoor scenes but hurt on the narrow domain distribution of a single corridor. This is the augmentation-specialization tradeoff — it shows up again in V7/V9.

**V5 is the best general-purpose indoor student in the lineage.**

---

## V6: pretrain then finetune

**Backbone:** EfficientViT-B1. **Teacher:** DA3-Large.

**Changes from V5:** pretrained on SUN RGB-D + DIODE (diverse indoor/outdoor scenes), then fine-tuned on NYU.

**NYU RMSE:** 0.519 m. **Femto Bolt corridor RMSE:** 2.158 m.

Another step down on NYU. The diverse pretraining gave the encoder better feature representations before seeing NYU-specific supervision. Corridor RMSE stayed roughly where V5 left it — the model is more capable overall, but not corridor-specialized.

**V6 is the best NYU depth model and the recommended starting point for further specialization.**

---

## V7: corridor fine-tune from V5

**Backbone:** EfficientViT-B1. **Teacher:** DA3-Large.

**Changes from V5:** fine-tuned on LILocBench corridor frames.

**NYU RMSE:** 1.315 m. **LILocBench RMSE:** 0.445 m. **Femto Bolt RMSE:** 1.982 m.

NYU RMSE more than doubled — catastrophic forgetting. The model traded general indoor knowledge for corridor-specific accuracy. LILocBench RMSE dropped sharply. The corridor is a narrow distribution (one building, one lighting condition, one floor type), and the model overfits to it quickly.

---

## V8: mixed training (failed)

**Backbone:** EfficientViT-B1. **Teacher:** DA3-Large.

**Idea:** mix NYU + LILocBench frames during training to get the best of both worlds.

**NYU RMSE:** 0.592 m. **Femto Bolt RMSE:** 2.266 m.

Worse than V5 on corridor, barely better than V5 on NYU. The domain gap between NYU (diverse scenes, varying depth ranges) and LILocBench (one corridor, 1-6 m range) is too large for naive mixing to bridge. Multi-task training with explicit domain balancing might help, but we did not pursue it — the paper's story did not need a universal model, it needed a corridor specialist.

---

## V9: corridor fine-tune from V6

**Backbone:** EfficientViT-B1. **Teacher:** DA3-Large.

**Changes from V6:** fine-tuned on LILocBench corridor frames (same protocol as V7, better starting point).

**NYU RMSE:** 1.553 m. **LILocBench RMSE:** 0.382 m. **Femto Bolt RMSE:** 1.589 m.

Best corridor specialist. Starting from V6 (pretrained on SUN+DIODE) instead of V5 (augmented NYU only) gave a better LILocBench result (0.382 vs 0.445). The Gazebo closed-loop validation with V9's TensorRT engine achieved 9/10 success rate — matching the ground-truth depth reference run on the same seeds.

**V9 is the paper's headline corridor model.**

---

## Summary: when to use which checkpoint

| Checkpoint | Best for | RMSE to expect |
|---|---|---|
| `best_depth_v5_vivek.pt` | General indoor scenes | 0.572 m (NYU) |
| `best_depth_v6.pt` | NYU depth, or as a pretrain base | 0.519 m (NYU) |
| `best_depth_v7.pt` | Corridor (from V5 base) | 0.445 m (LILocBench) |
| `best_depth_v9.pt` | Corridor (from V6 base, paper headline) | 0.382 m (LILocBench) |

All checkpoints are EfficientViT-B1, 5.31M params, ImageNet normalization, 240×320 input.

---

## How to retrain

```bash
# Full training from scratch (V4-style)
python train.py --epochs 100 --batch-size 16 --device cuda \
    --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl

# Fine-tune on corridor data (V9-style)
python train.py --epochs 50 --batch-size 8 --device cuda \
    --manifest corridor_eval_data/manifest.jsonl \
    --resume hpc_outputs/best_depth_v6.pt \
    --lr 1e-5
```

Training on a single L40S GPU takes ~45 minutes per 100 epochs on NYU (4,845 frames). LILocBench fine-tuning converges in ~15 minutes.
