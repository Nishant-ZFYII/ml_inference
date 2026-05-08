#!/bin/bash
#SBATCH --job-name=corridor-sam2-seg
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
# YOLO+SAM2 Segmentation on Corridor Frames
#
# Runs the YOLO-seeded SAM2 pipeline (teacher_infer/run_sam2.py) on the
# 459 corridor RGB frames to produce 6-class segmentation masks.
# Output: corridor_eval_data/sam2_seg/{stem}_sam2_seg.npy
#
# These masks are used by run_costmap_ablation.py --seg-dir for the
# A5/A6 class-aware inflation experiments.
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
SAM2_CKPT=$SCRATCH/model_weights/sam2_hiera_large.pt
DA3_REPO=$SCRATCH/Depth-Anything-3

CORRIDOR_DIR=$SCRATCH/corridor_eval_data
INPUT_RGB=$CORRIDOR_DIR/rgb
INPUT_DEPTH=$CORRIDOR_DIR/depth
OUTPUT_SAM2=$CORRIDOR_DIR/sam2_seg

cd $REPO_DIR

# ── Verify inputs exist ─────────────────────────────────────────────────────
echo "=== Corridor YOLO+SAM2 Segmentation ==="
echo "RGB dir:     $INPUT_RGB"
echo "Depth dir:   $INPUT_DEPTH"
echo "Output:      $OUTPUT_SAM2"
echo "SAM2 ckpt:   $SAM2_CKPT"

RGB_COUNT=$(ls -1 "$INPUT_RGB"/*.png 2>/dev/null | wc -l)
echo "RGB frames:  $RGB_COUNT"

if [ "$RGB_COUNT" -eq 0 ]; then
    echo "ERROR: No RGB frames found in $INPUT_RGB"
    echo "Copy corridor_eval_data to \$SCRATCH first:"
    echo "  cp -r ~/MS_Project/ml_pipeline/corridor_eval_data \$SCRATCH/"
    exit 1
fi

if [ ! -f "$SAM2_CKPT" ]; then
    echo "ERROR: SAM2 checkpoint not found at $SAM2_CKPT"
    echo "Download it: wget -O $SAM2_CKPT https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    exit 1
fi

# ── Install YOLO if needed ───────────────────────────────────────────────────
pip install -q ultralytics 2>/dev/null || true

# ── Run YOLO+SAM2 segmentation ──────────────────────────────────────────────
echo ""
echo "=== Running YOLO+SAM2 on $RGB_COUNT corridor frames ==="
python -m teacher_infer.run_sam2 \
    --input-dir "$INPUT_RGB" \
    --depth-dir "$INPUT_DEPTH" \
    --output-dir "$OUTPUT_SAM2" \
    --sam2-checkpoint "$SAM2_CKPT"

# ── Summary ──────────────────────────────────────────────────────────────────
SEG_COUNT=$(ls -1 "$OUTPUT_SAM2"/*_sam2_seg.npy 2>/dev/null | wc -l)
echo ""
echo "=== Segmentation Complete ==="
echo "Masks generated: $SEG_COUNT"
echo "Output dir:      $OUTPUT_SAM2"
echo ""
echo "Next: copy sam2_seg back to local machine and run ablation:"
echo "  scp -r torch:\$SCRATCH/corridor_eval_data/sam2_seg ~/MS_Project/ml_pipeline/corridor_eval_data/"
echo "  cd ~/MS_Project/ml_pipeline"
echo "  python3 run_costmap_ablation.py \\"
echo "      --manifest corridor_eval_data/manifest.jsonl \\"
echo "      --da3-dir corridor_eval_data/da3_depth/ \\"
echo "      --seg-dir corridor_eval_data/sam2_seg/ \\"
echo "      --output-dir results/costmap_ablation/corridor"
