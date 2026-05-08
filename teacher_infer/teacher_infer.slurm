#!/bin/bash
#SBATCH --job-name=teacher-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ============================================================================
# Teacher Inference: DA3-Metric-Large + YOLO+SAM2 on ALL NYU images
# ============================================================================

set -euo pipefail

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.06

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $(conda info --base)/etc/profile.d/conda.sh
source activate $SCRATCH/conda_envs/nchsb_ml
export PATH=$SCRATCH/conda_envs/nchsb_ml/bin:$PATH
export PYTHONNOUSERSITE=True

REPO_DIR=$HOME/ml_pipeline
DATA_DIR=$SCRATCH/nyu_teacher_data
SAM2_CKPT=$SCRATCH/model_weights/sam2_hiera_large.pt
DA3_REPO=$SCRATCH/Depth-Anything-3

cd $REPO_DIR

# ── Step 0: Install DA3 if not already installed ────────────────────────────
if ! python -c "import depth_anything_3" 2>/dev/null; then
    echo "=== Installing DA3 ==="
    if [ ! -d "$DA3_REPO" ]; then
        git clone https://github.com/ByteDance-Seed/Depth-Anything-3 $DA3_REPO
    fi
    pip install -e $DA3_REPO
fi

# Install other teacher deps
pip install -q ultralytics 2>/dev/null || true

# ── Step 1: Extract NYU images and depth to flat directories ────────────────
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
# Extract ALL images (train + val) so every sample has teacher labels
count = 0
for f in sorted((cache / 'rgb').glob('*.png')):
    stem = f.stem
    shutil.copy2(f, Path('$INPUT_RGB') / f.name)
    depth_src = cache / 'depth' / f'{stem}.npy'
    if depth_src.exists():
        shutil.copy2(depth_src, Path('$INPUT_DEPTH') / f'{stem}.npy')
    count += 1
print(f'Extracted {count} samples (all train + val)')
"

# ── Step 2: Verify DA3 output on 5 frames (gate) ───────────────────────────
echo "=== Step 2: Verifying DA3 output ==="
python -m teacher_infer.verify_teacher_output \
    --input-dir $INPUT_RGB \
    --gt-depth-dir $INPUT_DEPTH \
    --output-dir $SCRATCH/verify_output \
    --n-frames 5

# If verify exits non-zero, set -e aborts the job here

# ── Step 3: DA3-Metric-Large depth predictions ─────────────────────────────
echo "=== Step 3: Running DA3-Metric-Large ==="
OUTPUT_DA3=$DATA_DIR/da3_depth
python -m teacher_infer.run_da3 \
    --input-dir $INPUT_RGB \
    --output-dir $OUTPUT_DA3

# ── Step 4: YOLO+SAM2 segmentation labels ──────────────────────────────────
echo "=== Step 4: Running YOLO+SAM2 ==="
OUTPUT_SAM2=$DATA_DIR/sam2_seg
python -m teacher_infer.run_sam2 \
    --input-dir $INPUT_RGB \
    --depth-dir $INPUT_DEPTH \
    --output-dir $OUTPUT_SAM2 \
    --sam2-checkpoint $SAM2_CKPT

# ── Step 5: Build manifest ─────────────────────────────────────────────────
echo "=== Step 5: Building manifest ==="
python -m teacher_infer.build_manifest \
    --rgb-dir $INPUT_RGB \
    --depth-dir $INPUT_DEPTH \
    --da3-dir $OUTPUT_DA3 \
    --sam2-dir $OUTPUT_SAM2 \
    --output $DATA_DIR/manifest.jsonl

echo "=== Teacher inference complete ==="
echo "Manifest: $DATA_DIR/manifest.jsonl"
echo ""
echo "Next steps:"
echo "  1. Train student: sbatch train.slurm"
echo "  2. Evaluate:      python eval_distillation.py --manifest $DATA_DIR/manifest.jsonl"
