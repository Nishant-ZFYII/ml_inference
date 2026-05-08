#!/bin/bash
#SBATCH --job-name=eval-nyu-da3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ============================================================================
# DA3-Small NYU Depth V2 Evaluation  (Table VII teacher row)
#
# Runs DA3-Small (relative depth) on the NYU val split with per-image
# median scaling and computes RMSE / MAE / AbsRel / delta<1.25 vs GT.
#
# Prerequisites:
#   1. NYU data cached at $SCRATCH/nyu_depth_v2/nyu_cache/  (from training)
#   2. depth_anything_3 installed in $SCRATCH/conda_envs/nchsb_ml
#
# Output:
#   $SCRATCH/nyu_depth_v2/nyu_da3_eval/nyu_da3_da3-small_val.json
#   $SCRATCH/nyu_depth_v2/nyu_da3_eval/nyu_da3_da3-small_val_per_image.csv
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

DATA_ROOT=$SCRATCH/nyu_depth_v2

echo "=== DA3 NYU Depth V2 Evaluation ==="
echo "Data root: $DATA_ROOT"
echo "Cache:     $DATA_ROOT/nyu_cache"
echo ""

# Verify NYU cache exists
if [ ! -d "$DATA_ROOT/nyu_cache" ]; then
    echo "ERROR: NYU cache not found at $DATA_ROOT/nyu_cache"
    echo "  The cache should exist from training. Check path."
    exit 1
fi

echo "Val split indices:"
wc -l "$DATA_ROOT/nyu_cache/val_indices.txt"
echo ""

# ── DA3-SMALL (relative depth, per-image median scaling) ────────────────
echo "=========================================="
echo "=== DA3-SMALL (Table VII teacher row)  ==="
echo "=========================================="
python eval_nyu_da3.py \
    --model-id depth-anything/DA3-SMALL \
    --relative \
    --data-root "$DATA_ROOT" \
    --split val
echo ""

# ── (Optional) DA3METRIC-LARGE for comparison ──────────────────────────
# Uncomment to also evaluate the metric teacher on the same split.
# echo "=========================================="
# echo "=== DA3METRIC-LARGE (metric depth)     ==="
# echo "=========================================="
# python eval_nyu_da3.py \
#     --model-id depth-anything/DA3METRIC-LARGE \
#     --data-root "$DATA_ROOT" \
#     --split val
# echo ""

echo "=== DA3 NYU evaluation complete ==="
echo "Results saved to: $DATA_ROOT/nyu_da3_eval/"
echo ""
echo "To download results to local machine:"
echo "  scp np3129@dtn.torch.hpc.nyu.edu:\$SCRATCH/nyu_depth_v2/nyu_da3_eval/*.json ."
echo "  scp np3129@dtn.torch.hpc.nyu.edu:\$SCRATCH/nyu_depth_v2/nyu_da3_eval/*.csv ."
