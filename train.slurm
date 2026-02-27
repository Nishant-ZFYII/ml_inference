#!/bin/bash
#SBATCH --job-name=student-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ============================================================================
# Student Model Training on NYU Torch HPC
#
# BEFORE FIRST SUBMISSION:
#   1. Run `sinfo` to verify partition names and GPU spec strings
#   2. Update --gres and add --partition accordingly
#   3. Set up conda env (see setup_hpc.sh or README.md)
# ============================================================================

set -euo pipefail

# ── Environment (follows NYU HPC best practices) ──────────────────────────
module purge
module load anaconda3/2024.02

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Conda activation per NYU HPC docs (source activate, not conda activate)
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
source activate $SCRATCH/conda_envs/nchsb_ml
export PATH=$SCRATCH/conda_envs/nchsb_ml/bin:$PATH
export PYTHONNOUSERSITE=True

REPO_DIR=$HOME/ml_pipeline
cd $REPO_DIR

# ── Training ───────────────────────────────────────────────────────────────
echo "=== Starting student model training ==="
echo "Data root:    $SCRATCH/nyu_depth_v2"
echo "Checkpoints:  $SCRATCH/checkpoints"
echo "Logs:         $SCRATCH/runs"

python train.py \
    --data-root "$SCRATCH/nyu_depth_v2" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    --log-dir "$SCRATCH/runs" \
    --epochs 100 \
    --batch-size 16 \
    --num-workers 8

echo "=== Training complete ==="
echo "Best checkpoint: $SCRATCH/checkpoints/best.pt"
echo ""
echo "Next steps:"
echo "  1. Export:    python export_trt.py --checkpoint $SCRATCH/checkpoints/best.pt"
echo "  2. Evaluate:  python eval_distillation.py --checkpoint $SCRATCH/checkpoints/best.pt --manifest $SCRATCH/nyu_teacher_data/manifest.jsonl"
