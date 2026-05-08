#!/bin/bash
#SBATCH --job-name=costmap-ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ============================================================================
# Depth-Based Costmap Ablation Study
#
# Runs 5 experiments (Baseline, A1-A4) on corridor data and optionally
# LILocBench sequences. DA3-Small inference requires GPU.
#
# Phase 1: Corridor data (459 frames, ~30 min)
# Phase 2: LILocBench sequences (if downloaded, ~1 hr/sequence)
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

OUTPUT_BASE=$SCRATCH/hpc_outputs/costmap_ablation

# ── Step 1: Run DA3-Small on corridor RGB frames ────────────────────────────
# (generates da3_depth/*.npy files if they don't already exist)

CORRIDOR_DIR=$SCRATCH/corridor_eval_data
DA3_CORRIDOR_DIR=$CORRIDOR_DIR/da3_depth

if [ ! -d "$DA3_CORRIDOR_DIR" ] || [ -z "$(ls -A $DA3_CORRIDOR_DIR 2>/dev/null)" ]; then
    echo "=== Running DA3-Small on corridor RGB frames ==="
    python -m teacher_infer.run_da3 \
        --input-dir "$CORRIDOR_DIR/rgb/" \
        --output-dir "$DA3_CORRIDOR_DIR/" \
        --model-id depth-anything/DA3-SMALL \
        --fx 747.8 --fy 747.5
    echo ""
else
    echo "=== DA3 corridor depth already exists, skipping ==="
fi

# ── Step 2: Run YOLO+SAM2 on corridor frames if needed ───────────────────────
SAM2_CKPT=$SCRATCH/model_weights/sam2_hiera_large.pt
SAM2_CORRIDOR_DIR=$CORRIDOR_DIR/sam2_seg

if [ -f "$SAM2_CKPT" ]; then
    if [ ! -d "$SAM2_CORRIDOR_DIR" ] || [ -z "$(ls -A $SAM2_CORRIDOR_DIR 2>/dev/null)" ]; then
        echo "=== Running YOLO+SAM2 on corridor RGB frames ==="
        pip install -q ultralytics 2>/dev/null || true
        python -m teacher_infer.run_sam2 \
            --input-dir "$CORRIDOR_DIR/rgb/" \
            --depth-dir "$CORRIDOR_DIR/depth/" \
            --output-dir "$SAM2_CORRIDOR_DIR/" \
            --sam2-checkpoint "$SAM2_CKPT"
        echo ""
    else
        echo "=== YOLO+SAM2 corridor segmentation already exists, skipping ==="
    fi
else
    echo "=== SAM2 checkpoint not found, A5/A6 will be skipped ==="
fi

# ── Step 3: Corridor costmap ablation ────────────────────────────────────────
echo "=========================================="
echo "=== Corridor Costmap Ablation (459 frames) ==="
echo "=========================================="
SEG_FLAG=""
if [ -d "$SAM2_CORRIDOR_DIR" ] && [ -n "$(ls -A $SAM2_CORRIDOR_DIR 2>/dev/null)" ]; then
    SEG_FLAG="--seg-dir $SAM2_CORRIDOR_DIR"
fi
python run_costmap_ablation.py \
    --manifest "$CORRIDOR_DIR/manifest.jsonl" \
    --da3-dir "$DA3_CORRIDOR_DIR" \
    $SEG_FLAG \
    --output-dir "$OUTPUT_BASE/corridor" \
    --camera-height 0.25 \
    --dynamic-strategy corridor_width
echo ""

# ── Step 3: LILocBench sequences (if available) ─────────────────────────────
LILOCBENCH_BASE=$SCRATCH/lilocbench_data

if [ -d "$LILOCBENCH_BASE" ]; then
    for SEQ_DIR in "$LILOCBENCH_BASE"/*/; do
        SEQ_NAME=$(basename "$SEQ_DIR")
        MANIFEST="$SEQ_DIR/manifest.jsonl"

        if [ ! -f "$MANIFEST" ]; then
            echo "Skipping $SEQ_NAME (no manifest.jsonl)"
            continue
        fi

        # Run DA3 on LILocBench RGB frames if not done
        DA3_SEQ_DIR="$SEQ_DIR/da3_depth"
        if [ ! -d "$DA3_SEQ_DIR" ] || [ -z "$(ls -A $DA3_SEQ_DIR 2>/dev/null)" ]; then
            echo "=== Running DA3-Small on $SEQ_NAME ==="
            python -m teacher_infer.run_da3 \
                --input-dir "$SEQ_DIR/rgb/" \
                --output-dir "$DA3_SEQ_DIR/" \
                --model-id depth-anything/DA3-SMALL \
                --fx 386.0 --fy 386.0
            echo ""
        fi

        # Run YOLO+SAM2 on LILocBench frames if SAM2 checkpoint exists
        SAM2_SEQ_DIR="$SEQ_DIR/sam2_seg"
        if [ -f "$SAM2_CKPT" ]; then
            if [ ! -d "$SAM2_SEQ_DIR" ] || [ -z "$(ls -A $SAM2_SEQ_DIR 2>/dev/null)" ]; then
                echo "=== Running YOLO+SAM2 on $SEQ_NAME ==="
                python -m teacher_infer.run_sam2 \
                    --input-dir "$SEQ_DIR/rgb/" \
                    --depth-dir "$SEQ_DIR/depth/" \
                    --output-dir "$SAM2_SEQ_DIR/" \
                    --sam2-checkpoint "$SAM2_CKPT"
                echo ""
            fi
        fi

        echo "=========================================="
        echo "=== LILocBench: $SEQ_NAME ==="
        echo "=========================================="
        SEQ_SEG_FLAG=""
        if [ -d "$SAM2_SEQ_DIR" ] && [ -n "$(ls -A $SAM2_SEQ_DIR 2>/dev/null)" ]; then
            SEQ_SEG_FLAG="--seg-dir $SAM2_SEQ_DIR"
        fi
        python run_costmap_ablation.py \
            --manifest "$MANIFEST" \
            --da3-dir "$DA3_SEQ_DIR" \
            $SEQ_SEG_FLAG \
            --output-dir "$OUTPUT_BASE/lilocbench_$SEQ_NAME" \
            --camera-height 0.30 \
            --dynamic-strategy corridor_width
        echo ""
    done
else
    echo "=== LILocBench not found at $LILOCBENCH_BASE, skipping ==="
fi

echo "=== Costmap ablation complete ==="
echo "Results: $OUTPUT_BASE/"
ls -la "$OUTPUT_BASE"/
