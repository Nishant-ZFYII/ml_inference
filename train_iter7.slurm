#!/bin/bash
#SBATCH --job-name=student-iter7-tum
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ============================================================================
# Iteration 7: Train on TUM RGB-D (larger dataset)
#
# Same architecture (EfficientViT-B1) and uncertainty weighting as Iter 6,
# but trained on TUM RGB-D (~2-3k frames from 3 sequences) instead of
# NYU Depth V2 (1159 training samples).
#
# Teacher labels (DA3 + YOLO+SAM2) must be generated first:
#   sbatch teacher_infer/teacher_infer_tum.slurm
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
cd $REPO_DIR

TUM_DATA=$SCRATCH/tum_teacher_data
CKPT_DIR=$SCRATCH/checkpoints_iter7
LOG_DIR=$SCRATCH/runs_iter7
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Verify teacher data exists ──────────────────────────────────────────────
if [ ! -f "$TUM_DATA/manifest.jsonl" ]; then
    echo "ERROR: TUM teacher data not found at $TUM_DATA/manifest.jsonl"
    echo "Run teacher inference first: sbatch teacher_infer/teacher_infer_tum.slurm"
    exit 1
fi

echo "=== Iteration 7: TUM RGB-D Training ==="
echo "Dataset:      TUM RGB-D"
echo "Data root:    $TUM_DATA"
echo "Manifest:     $TUM_DATA/manifest.jsonl"
echo "Checkpoints:  $CKPT_DIR"
echo "Logs:         $LOG_DIR"
echo "Train samples: $(wc -l < $TUM_DATA/train_indices.txt)"
echo "Val samples:   $(wc -l < $TUM_DATA/val_indices.txt)"

# ── Training ───────────────────────────────────────────────────────────────
python train.py \
    --dataset tum \
    --data-root "$TUM_DATA" \
    --checkpoint-dir "$CKPT_DIR" \
    --log-dir "$LOG_DIR" \
    --manifest "$TUM_DATA/manifest.jsonl" \
    --epochs 200 \
    --lr 3e-4 \
    --batch-size 16 \
    --num-workers 8 \
    --uncertainty-weighting

echo "=== Training complete ==="
echo "Best checkpoints:"
echo "  Combined: $CKPT_DIR/best.pt"
echo "  Depth:    $CKPT_DIR/best_depth.pt"
echo "  Seg:      $CKPT_DIR/best_seg.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate on TUM:  python eval_distillation.py --checkpoint $CKPT_DIR/best_depth.pt --manifest $TUM_DATA/manifest.jsonl"
echo "  2. Cross-eval on NYU: python eval_distillation.py --checkpoint $CKPT_DIR/best_depth.pt --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl"
