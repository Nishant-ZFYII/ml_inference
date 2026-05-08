#!/bin/bash
#SBATCH --job-name=eval-da3-corridor
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
# DA3 Corridor Accuracy Evaluation
#
# Runs DA3-Small and DA3-Large directly on the 459 corridor RGB frames and
# compares depth predictions against aligned Femto Bolt sensor depth.
#
# This answers: "What is DA3-Small's actual RMSE in the corridor?"
# If DA3-Small achieves 0.3-0.5m → direct Jetson deployment is viable.
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

MANIFEST=$SCRATCH/corridor_eval_data/manifest.jsonl

echo "=== DA3 Corridor Depth Evaluation ==="
echo "Manifest: $MANIFEST"
echo ""

# ── DA3METRIC-LARGE (teacher — metric depth, accuracy ceiling) ─────────────
echo "=========================================="
echo "=== DA3METRIC-LARGE (teacher / ceiling) ==="
echo "=========================================="
python eval_corridor_da3.py \
    --model-id depth-anything/DA3METRIC-LARGE \
    --manifest "$MANIFEST"
echo ""

# ── DA3METRIC-LARGE (median-scaled — test if underlying depth is good) ─────
# The focal/300 metric conversion broke for Femto Bolt (748px focal).
# Median-scaling bypasses the conversion to isolate depth quality.
echo "=========================================="
echo "=== DA3METRIC-LARGE (median-scaled, bypass focal conversion) ==="
echo "=========================================="
python eval_corridor_da3.py \
    --model-id depth-anything/DA3METRIC-LARGE \
    --relative \
    --manifest "$MANIFEST"
echo ""

# ── DA3-SMALL (Jetson candidate — relative depth, median-scaled) ──────────
echo "=========================================="
echo "=== DA3-SMALL (Jetson candidate, median-scaled) ==="
echo "=========================================="
python eval_corridor_da3.py \
    --model-id depth-anything/DA3-SMALL \
    --relative \
    --manifest "$MANIFEST"
echo ""

echo "=== DA3 corridor evaluation complete ==="
