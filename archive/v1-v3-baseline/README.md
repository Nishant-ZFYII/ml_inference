# Multi-Task Student Model Training Pipeline

Train a MobileNetV3-Small student model that predicts both metric depth and 6-class semantic segmentation from a single RGB image. Uses knowledge distillation from DA2-Large (depth) and YOLO+SAM2-Large (segmentation) teacher models.

## Quick Start (Local CPU Validation)

```bash
conda activate nchsb_ml
cd ml_pipeline

# Validate pipeline end-to-end (downloads ~2.8 GB NYU dataset on first run)
python train.py --epochs 2 --batch-size 4 --device cpu --data-limit 50

# Export to ONNX (skip TensorRT on laptop)
python export_trt.py --checkpoint checkpoints/best.pt --skip-trt
```

## Full Training on HPC (NYU Torch Cluster)

Following [NYU HPC conda best practices](https://services.rt.nyu.edu/docs/hpc/tools_and_software/conda_environments/).

### 1. Setup (one-time)

```bash
ssh <NetID>@login.torch.hpc.nyu.edu
cd $HOME
git clone git@github.com:Nishant-ZFYII/ml_inference.git ml_pipeline

# Run the setup script (creates prefix conda env in $SCRATCH)
bash ml_pipeline/setup_hpc.sh

# IMPORTANT: Check partition names before submitting jobs
sinfo
# Then update --gres and --partition in train.slurm and teacher_infer.slurm
```

The setup script creates a prefix conda environment at `$SCRATCH/conda_envs/nchsb_ml`
with CUDA-enabled PyTorch. This avoids filling up `$HOME` quota.

### 2. Run Teacher Inference (generates Table IV data)

```bash
# First, download NYU dataset (interactive, ~5 min)
module purge && module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
source activate $SCRATCH/conda_envs/nchsb_ml
cd $HOME/ml_pipeline
python train.py --epochs 1 --data-root $SCRATCH/nyu_depth_v2 --data-limit 10

# Then run teachers on NYU val set
sbatch teacher_infer/teacher_infer.slurm
```

### 3. Train Student Model

```bash
sbatch train.slurm
```

### 4. Evaluate Distillation Quality (Table IV)

```bash
python eval_distillation.py \
    --checkpoint $SCRATCH/checkpoints/best.pt \
    --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl
```

### 5. Export and Benchmark (on Jetson)

```bash
# Copy best.pt to Jetson, then:
python export_trt.py --checkpoint best.pt
python benchmark_jetson.py --engine-dir exported/
```

## Swapping to Corridor Data

When you have your own corridor RGB + ToF data:

1. **Collect data** on the robot (RGB + ToF depth + ToF confidence from Femto Bolt)

2. **Run teachers on HPC** (same scripts, different input directory):
   ```bash
   python -m teacher_infer.run_da2 --input-dir corridor_data/rgb/ --output-dir corridor_data/da2_depth/
   python -m teacher_infer.run_sam2 --input-dir corridor_data/rgb/ --depth-dir corridor_data/depth/ --output-dir corridor_data/sam2_seg/
   python -m teacher_infer.build_manifest --rgb-dir corridor_data/rgb/ --depth-dir corridor_data/depth/ --da2-dir corridor_data/da2_depth/ --sam2-dir corridor_data/sam2_seg/ --confidence-dir corridor_data/confidence/ --output corridor_data/manifest.jsonl
   ```

3. **Set the flag** in `config.py`:
   ```python
   USE_CORRIDOR_DATA = True
   ```

4. **Retrain**:
   ```bash
   python train.py --manifest corridor_data/manifest.jsonl
   ```

No other code changes needed.

## Architecture

```
Input RGB (320x240) → MobileNetV3-Small Encoder
                          ├── Depth Decoder (3 TransConv + bilinear) → Depth Map (320x240x1)
                          └── Seg Decoder   (3 TransConv + bilinear) → Seg Logits (320x240x6)
```

### Training Loss

```
L_total = λ_d · L_depth + λ_s · L_seg + λ_e · L_edge
```

- **L_depth** (hybrid): L1 loss against `where(confidence >= τ, ToF_depth, DA2_depth)` — mirrors runtime fusion
- **L_seg**: Cross-entropy against YOLO+SAM2 teacher labels
- **L_edge**: Edge-aware smoothness `|∇depth| · exp(-|∇RGB|)`

### 6-Class Scheme

| ID | Class     | Source                        |
|----|-----------|-------------------------------|
| 0  | floor     | Geometric heuristic           |
| 1  | wall      | Geometric heuristic           |
| 2  | person    | YOLO detect → SAM2 refine     |
| 3  | furniture | YOLO detect → SAM2 refine     |
| 4  | glass     | ToF confidence heuristic      |
| 5  | other     | Default                       |

## File Structure

```
ml_pipeline/
  config.py                 Central configuration
  dataset/
    nyu_loader.py            NYU Depth V2 dataset + dataloader
    corridor_loader.py       Corridor data loader (swap-in)
    label_remapper.py        894→40→6 class remapping
  models/
    student.py               MobileNetV3-Small + dual decoders
    losses.py                Hybrid depth loss + seg + edge
  teacher_infer/
    run_da2.py               DA2-Large inference
    run_sam2.py              YOLO+SAM2 semantic labeling
    build_manifest.py        Manifest builder
    teacher_infer.slurm      SLURM script for teacher inference
  train.py                   Training loop
  export_trt.py              ONNX + TensorRT export
  eval_distillation.py       Student vs teacher evaluation (Table IV)
  benchmark_jetson.py        TensorRT benchmark (Table III)
  print_model_shapes.py      Encoder shape verification utility
  train.slurm                SLURM script for student training
  setup_hpc.sh               One-time HPC environment setup
  requirements.txt           Python dependencies
  README.md                  This file
```
