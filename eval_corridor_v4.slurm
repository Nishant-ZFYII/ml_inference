#!/bin/bash
#SBATCH --job-name=eval-corridor-v4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ============================================================================
# Corridor Depth Evaluation — Vivek's V4 Checkpoint
#
# Tests whether V4's improved 0.774m NYU RMSE (vs Iter 6's 0.998m)
# translates to better corridor performance.
#
# Previous corridor results:
#   B1 (Iter 6):  RMSE 2.777m
#   B2 (Iter 7b): RMSE 2.356m
#   DA3-Small:    RMSE 0.596m (median-scaled)
#
# Prerequisites:
#   1. corridor_eval_data already on HPC at $SCRATCH/corridor_eval_data/
#   2. V4 checkpoint transferred:
#      scp vivek_v4_best.pt np3129@dtn.torch.hpc.nyu.edu:$SCRATCH/checkpoints_v4/
# ============================================================================

set -euo pipefail

module purge
module load anaconda3/2025.06

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $(conda info --base)/etc/profile.d/conda.sh
source activate $SCRATCH/conda_envs/nchsb_ml
export PATH=$SCRATCH/conda_envs/nchsb_ml/bin:$PATH
export PYTHONNOUSERSITE=True

REPO_DIR=$HOME/ml_pipeline
cd $REPO_DIR

CORRIDOR_DATA=$SCRATCH/corridor_eval_data
MANIFEST=$CORRIDOR_DATA/manifest.jsonl

echo "=== Corridor Depth Evaluation — Vivek V4 ==="
echo "Manifest: $MANIFEST"
echo ""

# ── V4 best.pt (EfficientViT-B1, Vivek's training recipe) ────────────────
CKPT_V4=$SCRATCH/checkpoints_v4/vivek_v4_best.pt
if [ -f "$CKPT_V4" ]; then
    echo "--- V4 best.pt (Vivek, EfficientViT-B1) ---"
    python eval_corridor_depth.py \
        --checkpoint "$CKPT_V4" \
        --manifest "$MANIFEST"
    echo ""
else
    echo "SKIP: $CKPT_V4 not found"
    echo "  Transfer it first:"
    echo "  scp vivek_v4_best.pt np3129@dtn.torch.hpc.nyu.edu:\$SCRATCH/checkpoints_v4/"
fi

# ── V4 best_depth.pt (if available) ──────────────────────────────────────
CKPT_V4_DEPTH=$SCRATCH/checkpoints_v4/vivek_v4_best_depth.pt
if [ -f "$CKPT_V4_DEPTH" ]; then
    echo "--- V4 best_depth.pt (Vivek, depth-specific) ---"
    python eval_corridor_depth.py \
        --checkpoint "$CKPT_V4_DEPTH" \
        --manifest "$MANIFEST"
    echo ""
else
    echo "INFO: $CKPT_V4_DEPTH not found (optional — skip)"
fi

# ── Comparison reference: re-run Iter 6 B1 for side-by-side ─────────────
CKPT_B1=$SCRATCH/checkpoints_iter6/best_depth.pt
if [ -f "$CKPT_B1" ]; then
    echo "--- B1 best_depth.pt (Iter 6, baseline comparison) ---"
    python eval_corridor_depth.py \
        --checkpoint "$CKPT_B1" \
        --manifest "$MANIFEST"
    echo ""
else
    echo "INFO: Iter 6 B1 checkpoint not found (skip comparison)"
fi

echo "=== Corridor V4 evaluation complete ==="
