#!/bin/bash
#SBATCH --job-name=student-iter7b-b2
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
# Iteration 7b: EfficientViT-B2 on NYU (backbone comparison)
#
# Same training setup as Iter 6 (uncertainty weighting, per-task checkpoints)
# but with B2 backbone (15.4M params) instead of B1 (5.3M params).
# Uses existing NYU teacher labels (DA3 + YOLO+SAM2).
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

CKPT_DIR=$SCRATCH/checkpoints_iter7b_b2
LOG_DIR=$SCRATCH/runs_iter7b_b2
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Training ───────────────────────────────────────────────────────────────
echo "=== Iteration 7b: EfficientViT-B2 on NYU ==="
echo "Backbone:     efficientvit_b2.r224_in1k"
echo "Data root:    $SCRATCH/nyu_depth_v2"
echo "Checkpoints:  $CKPT_DIR"
echo "Logs:         $LOG_DIR"

python train.py \
    --backbone efficientvit_b2.r224_in1k \
    --data-root "$SCRATCH/nyu_depth_v2" \
    --checkpoint-dir "$CKPT_DIR" \
    --log-dir "$LOG_DIR" \
    --manifest "$SCRATCH/nyu_teacher_data/manifest.jsonl" \
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
echo "Evaluate (use --backbone to match):"
echo "  python eval_distillation.py --backbone efficientvit_b2.r224_in1k --checkpoint $CKPT_DIR/best_depth.pt --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl"
echo "  python eval_distillation.py --backbone efficientvit_b2.r224_in1k --checkpoint $CKPT_DIR/best_seg.pt --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl"
