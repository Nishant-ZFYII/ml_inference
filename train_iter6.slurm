#!/bin/bash
#SBATCH --job-name=student-iter6
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
# Iteration 6: Uncertainty Weighting + Per-Task Checkpoints
#
# Writes to separate directories so Iter 5 checkpoints are not overwritten.
# Can run in parallel with Iter 5 evaluation.
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
cd $REPO_DIR

CKPT_DIR=$SCRATCH/checkpoints_iter6
LOG_DIR=$SCRATCH/runs_iter6
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Training ───────────────────────────────────────────────────────────────
echo "=== Iteration 6: Uncertainty Weighting Training ==="
echo "Data root:    $SCRATCH/nyu_depth_v2"
echo "Checkpoints:  $CKPT_DIR"
echo "Logs:         $LOG_DIR"

python train.py \
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
echo "Next steps:"
echo "  1. Evaluate (combined): python eval_distillation.py --checkpoint $CKPT_DIR/best.pt --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl"
echo "  2. Evaluate (depth):    python eval_distillation.py --checkpoint $CKPT_DIR/best_depth.pt --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl"
echo "  3. Evaluate (seg):      python eval_distillation.py --checkpoint $CKPT_DIR/best_seg.pt --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl"
