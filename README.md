# ML Pipeline -- Multi-Task Student Model Training

Multi-task knowledge distillation pipeline for indoor scene understanding.
Trains a lightweight EfficientViT-B1 student model to predict **metric depth**
and **6-class semantic segmentation** from a single RGB image, supervised by
DA3-Metric-Large (depth) and YOLO+SAM2 (segmentation) teacher models.

## Architecture

| Component | Details |
|-----------|---------|
| **Student backbone** | EfficientViT-B1 (5.3M params, via `timm`) |
| **Depth teacher** | DA3-Metric-Large (`depth-anything/da3metric-large`) |
| **Seg teacher** | YOLOv8 + SAM2-Large + geometric heuristics |
| **Input resolution** | 320 x 240 (matches Orbbec Femto Bolt) |
| **Depth output** | 1-channel metric depth map (meters) |
| **Seg output** | 6-channel logit map (floor, wall, person, furniture, glass, other) |

### DA3 Metric Depth Conversion

DA3-Metric-Large outputs are converted to metric metres using:

```
metric_depth = focal * raw_output / 300.0
```

where `focal = (fx + fy) / 2` (camera intrinsics, passed via `--fx`/`--fy`).

### Segmentation Pipeline (YOLO + SAM2 + Geometric)

The segmentation teacher (`run_sam2.py`) combines three strategies:

1. **YOLOv8 detection** -- bounding boxes with class labels (person, furniture)
2. **SAM2-Large refinement** -- pixel-accurate masks from YOLO box prompts
3. **Geometric heuristics** -- floor/wall/glass from depth map analysis
4. **Merge** -- YOLO+SAM2 masks override geometric labels

Output: single HxW uint8 array (values 0-5).

## Directory Structure

```
ml_pipeline/
├── config.py                   # Central configuration
├── train.py                    # Training script
├── train.slurm                 # SLURM job for training
├── eval_distillation.py        # Student vs teacher evaluation (Table IV)
├── export_trt.py               # ONNX + TensorRT export
├── benchmark_jetson.py         # Jetson inference benchmarking (Table III)
├── print_model_shapes.py       # EfficientViT-B1 feature map inspection
├── requirements.txt
├── setup_hpc.sh                # One-time HPC environment setup
├── dataset/
│   ├── nyu_loader.py           # NYU Depth V2 dataset loader
│   ├── corridor_loader.py      # Corridor dataset loader (stub)
│   └── label_remapper.py       # 894-class → 6-class mapping
├── models/
│   ├── student.py              # EfficientViT-B1 encoder + dual decoders
│   └── losses.py               # Hybrid depth + CE seg + edge losses
└── teacher_infer/
    ├── run_da3.py               # DA3-Metric-Large inference
    ├── verify_teacher_output.py # Pre-run DA3 output verification
    ├── run_sam2.py              # YOLO+SAM2+geometric segmentation
    ├── build_manifest.py        # Build manifest.jsonl
    └── teacher_infer.slurm      # SLURM job for teacher inference
```

## Quick Start

### 1. HPC Setup (one-time)

```bash
ssh <NetID>@login.torch.hpc.nyu.edu
cd $HOME
git clone https://github.com/Nishant-ZFYII/ml_inference.git ml_pipeline
bash ml_pipeline/setup_hpc.sh
```

This installs PyTorch, timm, DA3, SAM2, and downloads model checkpoints.

### 2. Run Teacher Inference

```bash
cd $HOME/ml_pipeline
sbatch teacher_infer/teacher_infer.slurm
```

This runs:
1. `verify_teacher_output.py` on 5 frames (gate -- aborts if scale is wrong)
2. `run_da3.py` -- DA3-Metric-Large on all NYU val images
3. `run_sam2.py` -- YOLO+SAM2 segmentation on all images
4. `build_manifest.py` -- builds `manifest.jsonl`

### 3. Train Student Model

```bash
sbatch train.slurm
```

### 4. Evaluate Distillation Quality

```bash
python eval_distillation.py \
    --checkpoint $SCRATCH/checkpoints/best.pt \
    --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl
```

### 5. Export to TensorRT

```bash
python export_trt.py --checkpoint $SCRATCH/checkpoints/best.pt
```

### 6. Benchmark on Jetson

Copy the `.engine` files and `benchmark_jetson.py` to the Jetson, then:

```bash
python benchmark_jetson.py --engine-dir ./engines/
```

## Local CPU Testing

Before pushing to HPC, validate locally:

```bash
python train.py --epochs 2 --batch-size 4 --device cpu --data-limit 50
python export_trt.py --checkpoint checkpoints/best.pt --skip-trt
```

## Switching to Corridor Data

1. Collect RGB + ToF data on the robot
2. Run `teacher_infer/run_da3.py` with Femto Bolt intrinsics (`--fx`, `--fy`)
3. Run `teacher_infer/run_sam2.py` on the corridor images
4. Run `teacher_infer/build_manifest.py`
5. Set `USE_CORRIDOR_DATA = True` in `config.py`
6. Re-run `train.py`

## HPC Configuration

| Setting | Value |
|---------|-------|
| Partition | `l40s_public` |
| GPU | `gpu:l40s:1` |
| Account | `torch_pr_742_general` |
| Module | `anaconda3/2025.06` |
| Conda env | `$SCRATCH/conda_envs/nchsb_ml` |
