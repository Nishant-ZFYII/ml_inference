#!/bin/bash
#SBATCH --job-name=eval-corridor
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
# Corridor Depth Evaluation
#
# Evaluate student depth predictions against real Femto Bolt sensor depth
# in the deployment corridor. This is the number that determines whether
# the distilled model is usable for corridor navigation.
#
# Prerequisites:
#   1. Extract corridor frames locally:
#      python -m teacher_infer.extract_corridor_bag \
#          --bag ../NCHSB/rosbags/rgbd_imu_20260228_003828_0.mcap \
#          --output-dir corridor_eval_data --subsample 5
#   2. Transfer to HPC:
#      tar czf corridor_eval_data.tar.gz corridor_eval_data/
#      scp corridor_eval_data.tar.gz np3129@dtn.torch.hpc.nyu.edu:/scratch/np3129/
#      ssh torch: cd $SCRATCH && tar xzf corridor_eval_data.tar.gz
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

CORRIDOR_DATA=$SCRATCH/corridor_eval_data
MANIFEST=$CORRIDOR_DATA/manifest.jsonl

echo "=== Corridor Depth Evaluation ==="
echo "Manifest:  $MANIFEST"
echo ""

# ── Eval 1: B1 best_depth.pt (Iter 6) ─────────────────────────────────────
CKPT_B1=$SCRATCH/checkpoints_iter6/best_depth.pt
if [ -f "$CKPT_B1" ]; then
    echo "--- B1 best_depth.pt (Iter 6) ---"
    python eval_corridor_depth.py \
        --checkpoint "$CKPT_B1" \
        --manifest "$MANIFEST"
    echo ""
else
    echo "SKIP: $CKPT_B1 not found"
fi

# ── Eval 2: B2 best_depth.pt (Iter 7b) ────────────────────────────────────
CKPT_B2=$SCRATCH/checkpoints_iter7b_b2/best_depth.pt
if [ -f "$CKPT_B2" ]; then
    echo "--- B2 best_depth.pt (Iter 7b) ---"
    python eval_corridor_depth.py \
        --backbone efficientvit_b2.r224_in1k \
        --checkpoint "$CKPT_B2" \
        --manifest "$MANIFEST"
    echo ""
else
    echo "SKIP: $CKPT_B2 not found"
fi

echo "=== Corridor evaluation complete ==="
