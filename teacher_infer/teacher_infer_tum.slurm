#!/bin/bash
#SBATCH --job-name=teacher-infer-tum
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
# Teacher Inference: DA3-Metric-Large + YOLO+SAM2 on TUM RGB-D
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
TUM_RAW=$SCRATCH/tum_rgbd
DATA_DIR=$SCRATCH/tum_teacher_data
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

pip install -q ultralytics 2>/dev/null || true

# ── Step 1: Extract and prep TUM sequences ──────────────────────────────────
echo "=== Step 1: Extracting TUM sequences ==="

# Extract tarballs if not already done
for tgz in $TUM_RAW/*.tgz; do
    dirname=$(basename "$tgz" .tgz)
    if [ ! -d "$TUM_RAW/$dirname" ]; then
        echo "Extracting $tgz ..."
        tar -xzf "$tgz" -C "$TUM_RAW"
    else
        echo "Already extracted: $dirname"
    fi
done

# Prep: flatten to sequential stems with train/val split
if [ ! -f "$DATA_DIR/train_indices.txt" ]; then
    python -m teacher_infer.prep_tum \
        --sequences \
            $TUM_RAW/rgbd_dataset_freiburg1_room \
            $TUM_RAW/rgbd_dataset_freiburg1_desk \
            $TUM_RAW/rgbd_dataset_freiburg2_large_no_loop \
        --output-dir $DATA_DIR \
        --subsample 3
else
    echo "TUM data already prepped at $DATA_DIR"
    echo "  Train: $(wc -l < $DATA_DIR/train_indices.txt) samples"
    echo "  Val:   $(wc -l < $DATA_DIR/val_indices.txt) samples"
fi

INPUT_RGB=$DATA_DIR/rgb
INPUT_DEPTH=$DATA_DIR/depth

# ── Step 2: DA3-Metric-Large depth predictions ─────────────────────────────
echo "=== Step 2: Running DA3-Metric-Large ==="

# TUM Freiburg1/2 camera intrinsics (from calibration files)
# fr1: fx=517.3, fy=516.5     fr2: fx=520.9, fy=521.0
# Using average across sequences
TUM_FX=519.1
TUM_FY=518.75

OUTPUT_DA3=$DATA_DIR/da3_depth
python -m teacher_infer.run_da3 \
    --input-dir $INPUT_RGB \
    --output-dir $OUTPUT_DA3 \
    --fx $TUM_FX --fy $TUM_FY

# ── Step 3: YOLO+SAM2 segmentation labels ──────────────────────────────────
echo "=== Step 3: Running YOLO+SAM2 ==="
OUTPUT_SAM2=$DATA_DIR/sam2_seg
python -m teacher_infer.run_sam2 \
    --input-dir $INPUT_RGB \
    --depth-dir $INPUT_DEPTH \
    --output-dir $OUTPUT_SAM2 \
    --sam2-checkpoint $SAM2_CKPT

# ── Step 4: Build manifest ─────────────────────────────────────────────────
echo "=== Step 4: Building manifest ==="
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
echo "  1. Train student: sbatch train_iter7.slurm"
echo "  2. Evaluate:      python eval_distillation.py --checkpoint best.pt --manifest $DATA_DIR/manifest.jsonl"
