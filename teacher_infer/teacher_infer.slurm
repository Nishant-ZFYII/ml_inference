#!/bin/bash
#SBATCH --job-name=teacher-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ============================================================================
# Teacher Inference: DA2-Large + YOLO+SAM2 on NYU val set
#
# BEFORE FIRST SUBMISSION:
#   1. Run `sinfo` to verify partition names and GPU spec strings
#   2. Update --gres and add --partition accordingly
#   3. Set up conda env (see setup_hpc.sh or README.md)
# ============================================================================

set -euo pipefail

# ── Environment (follows NYU HPC best practices) ──────────────────────────
module purge
module load anaconda3/2025.06

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Conda activation per NYU HPC docs (source activate, not conda activate)
source $(conda info --base)/etc/profile.d/conda.sh
source activate $SCRATCH/conda_envs/nchsb_ml
export PATH=$SCRATCH/conda_envs/nchsb_ml/bin:$PATH
export PYTHONNOUSERSITE=True

REPO_DIR=$HOME/ml_pipeline
DATA_DIR=$SCRATCH/nyu_teacher_data

# Install teacher-specific dependencies (only needed once)
pip install -q depth-anything-v2 ultralytics segment-anything-2 2>/dev/null || true

cd $REPO_DIR

# ── Step 1: Extract NYU images and depth to flat directories ───────────────
echo "=== Step 1: Preparing NYU data ==="
INPUT_RGB=$DATA_DIR/rgb
INPUT_DEPTH=$DATA_DIR/depth
mkdir -p $INPUT_RGB $INPUT_DEPTH

python -c "
import os, shutil
from pathlib import Path
cache = Path('$SCRATCH/nyu_depth_v2/nyu_cache')
if not cache.exists():
    print('NYU cache not found. Run train.py first to download NYU data.')
    exit(1)
val_file = cache / 'val_indices.txt'
indices = [int(l.strip()) for l in open(val_file)]
for idx in indices:
    stem = f'{idx:05d}'
    src_rgb = cache / 'rgb' / f'{stem}.png'
    src_depth = cache / 'depth' / f'{stem}.npy'
    if src_rgb.exists():
        shutil.copy2(src_rgb, Path('$INPUT_RGB') / f'{stem}.png')
    if src_depth.exists():
        shutil.copy2(src_depth, Path('$INPUT_DEPTH') / f'{stem}.npy')
print(f'Extracted {len(indices)} val samples')
"

# ── Step 2: DA2-Large depth predictions ────────────────────────────────────
echo "=== Step 2: Running DA2-Large ==="
OUTPUT_DA2=$DATA_DIR/da2_depth
python -m teacher_infer.run_da2 \
    --input-dir $INPUT_RGB \
    --output-dir $OUTPUT_DA2

# ── Step 3: YOLO+SAM2 segmentation labels ─────────────────────────────────
echo "=== Step 3: Running YOLO+SAM2 ==="
OUTPUT_SAM2=$DATA_DIR/sam2_seg
python -m teacher_infer.run_sam2 \
    --input-dir $INPUT_RGB \
    --depth-dir $INPUT_DEPTH \
    --output-dir $OUTPUT_SAM2

# ── Step 4: Build manifest ─────────────────────────────────────────────────
echo "=== Step 4: Building manifest ==="
python -m teacher_infer.build_manifest \
    --rgb-dir $INPUT_RGB \
    --depth-dir $INPUT_DEPTH \
    --da2-dir $OUTPUT_DA2 \
    --sam2-dir $OUTPUT_SAM2 \
    --output $DATA_DIR/manifest.jsonl

echo "=== Teacher inference complete ==="
echo "Manifest: $DATA_DIR/manifest.jsonl"
echo ""
echo "Next steps:"
echo "  1. Train student: sbatch train.slurm"
echo "  2. Evaluate:      python eval_distillation.py --manifest $DATA_DIR/manifest.jsonl"
