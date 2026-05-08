#!/bin/bash
# ============================================================================
# One-time HPC setup for NYU Torch cluster
#
# Run this interactively on a login node after cloning the repo:
#   ssh <NetID>@login.torch.hpc.nyu.edu
#   cd $HOME
#   git clone git@github.com:Nishant-ZFYII/ml_inference.git ml_pipeline
#   bash ml_pipeline/setup_hpc.sh
#
# This follows NYU HPC best practices:
#   https://services.rt.nyu.edu/docs/hpc/tools_and_software/conda_environments/
# ============================================================================

set -euo pipefail

echo "=== NYU Torch HPC Setup for ml_pipeline ==="

# ── Step 1: Configure conda to use $SCRATCH ────────────────────────────────
# Named envs default to $HOME which has limited space.
# Prefix envs in $SCRATCH are the recommended approach.
echo "Setting up conda directories in \$SCRATCH..."
mkdir -p $SCRATCH/conda_envs
mkdir -p $SCRATCH/conda_pkgs

# ── Step 2: Load anaconda module ───────────────────────────────────────────
module purge
module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh

# ── Step 3: Create prefix environment in $SCRATCH ──────────────────────────
ENV_PATH=$SCRATCH/conda_envs/nchsb_ml

if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH"
    echo "To recreate: rm -rf $ENV_PATH && bash setup_hpc.sh"
else
    echo "Creating conda environment at $ENV_PATH ..."
    conda create -p $ENV_PATH python=3.11 -y
fi

# ── Step 4: Activate and install dependencies ──────────────────────────────
source activate $ENV_PATH
export PATH=$ENV_PATH/bin:$PATH
export PYTHONNOUSERSITE=True

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
pip install -r $HOME/ml_pipeline/requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "Environment: $ENV_PATH"
echo ""
echo "To use in interactive sessions:"
echo "  module purge"
echo "  module load anaconda3/2025.06"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh"
echo "  source activate $ENV_PATH"
echo "  export PATH=$ENV_PATH/bin:\$PATH"
echo "  export PYTHONNOUSERSITE=True"
echo ""
echo "To submit jobs:"
echo "  cd $HOME/ml_pipeline"
echo "  sbatch train.slurm"
echo ""
echo "IMPORTANT: Before first sbatch, run 'sinfo' to check partition names"
echo "and update --gres/--partition in train.slurm and teacher_infer.slurm"
