#!/bin/bash
#SBATCH --job-name=lilocbench-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=lilocbench-pipeline-%j.out
#SBATCH --error=lilocbench-pipeline-%j.err

# ============================================================================
# LILocBench Full Pipeline: Download → Extract → Teacher Infer → Train → Eval
#
# Downloads lt_changes_3 (hallway-only, 10.1 GB), extracts frames at 2Hz,
# runs DA3-Metric-Large teacher, then trains on NYU + LILocBench combined.
#
# V5 recipe = Vivek V4 (berHu, LR split, freeze warmup) + corridor data
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

LILOCBENCH_RAW=$SCRATCH/lilocbench_raw
LILOCBENCH_EXTRACTED=$SCRATCH/lilocbench_extracted
LILOCBENCH_MANIFEST=$LILOCBENCH_EXTRACTED/manifest.jsonl
CKPT_DIR=$SCRATCH/checkpoints_v5
LOG_DIR=$SCRATCH/runs_v5
NYU_DATA=$SCRATCH/nyu_depth_v2
NYU_MANIFEST=$SCRATCH/nyu_teacher_data/manifest.jsonl
CORRIDOR_MANIFEST=$SCRATCH/corridor_eval_data/manifest.jsonl

# ============================================================================
# STEP 1: Download LILocBench lt_changes_3 (hallway-only sequence)
# ============================================================================
echo "=== STEP 1: Download LILocBench lt_changes_3 ==="

SEQUENCE_URL="https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/data/data_files_zipped/lt_changes_3.zip"
ZIP_FILE=$LILOCBENCH_RAW/lt_changes_3.zip

if [ -d "$LILOCBENCH_RAW/lt_changes_3" ]; then
    echo "Already downloaded and extracted, skipping."
else
    mkdir -p $LILOCBENCH_RAW
    if [ ! -f "$ZIP_FILE" ]; then
        echo "Downloading lt_changes_3.zip (10.1 GB)..."
        wget -q --show-progress -O "$ZIP_FILE" "$SEQUENCE_URL"
    fi
    echo "Extracting..."
    cd $LILOCBENCH_RAW
    unzip -q "$ZIP_FILE"
    cd $REPO_DIR
    echo "Download + extract complete."
fi

echo "Contents:"
ls -la $LILOCBENCH_RAW/lt_changes_3/

# ============================================================================
# STEP 2: Extract frames at 2Hz
# ============================================================================
echo ""
echo "=== STEP 2: Extract frames at 2Hz ==="

if [ -f "$LILOCBENCH_MANIFEST" ]; then
    FRAME_COUNT=$(wc -l < "$LILOCBENCH_MANIFEST")
    echo "Already extracted ($FRAME_COUNT frames), skipping."
else
    python extract_lilocbench.py \
        --input "$LILOCBENCH_RAW/lt_changes_3" \
        --output "$LILOCBENCH_EXTRACTED" \
        --target-fps 2
fi

FRAME_COUNT=$(wc -l < "$LILOCBENCH_MANIFEST")
echo "Extracted frames: $FRAME_COUNT"

# ============================================================================
# STEP 3: Run DA3-Metric-Large teacher on extracted frames
# ============================================================================
echo ""
echo "=== STEP 3: DA3 teacher inference on LILocBench ==="

DA3_OUT=$LILOCBENCH_EXTRACTED/da3_depth
if [ -d "$DA3_OUT" ] && [ "$(ls $DA3_OUT/*.npy 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "DA3 predictions already exist ($(ls $DA3_OUT/*.npy | wc -l) files), skipping."
else
    # Read focal length from first manifest entry
    FOCAL=$(python3 -c "
import json
with open('$LILOCBENCH_MANIFEST') as f:
    entry = json.loads(f.readline())
    intr = entry.get('intrinsics', {})
    fx = intr.get('fx', 386.0)
    fy = intr.get('fy', 386.0)
    print(f'{(fx+fy)/2:.2f}')
")
    echo "Using focal length: $FOCAL px"

    python -m teacher_infer.run_da3 \
        --input-dir "$LILOCBENCH_EXTRACTED/rgb/" \
        --output-dir "$DA3_OUT/" \
        --fx "$FOCAL" --fy "$FOCAL"
fi

# Update manifest with DA3 paths
echo "Updating manifest with DA3 depth paths..."
python3 -c "
import json, os
manifest_path = '$LILOCBENCH_MANIFEST'
entries = []
with open(manifest_path) as f:
    for line in f:
        entry = json.loads(line)
        stem = entry['stem']
        da3_path = f'da3_depth/{stem}.npy'
        if os.path.exists(os.path.join('$LILOCBENCH_EXTRACTED', da3_path)):
            entry['da3_depth'] = da3_path
        entries.append(entry)
with open(manifest_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')
da3_count = sum(1 for e in entries if 'da3_depth' in e)
print(f'Updated manifest: {da3_count}/{len(entries)} have DA3 depth')
"

# ============================================================================
# STEP 4: Train V5 (NYU + LILocBench combined)
# ============================================================================
echo ""
echo "=== STEP 4: Training V5 (NYU + LILocBench) ==="
echo "NYU data:           $NYU_DATA"
echo "NYU manifest:       $NYU_MANIFEST"
echo "LILocBench manifest: $LILOCBENCH_MANIFEST"
echo "Checkpoints:        $CKPT_DIR"
echo "Logs:               $LOG_DIR"

python train.py \
    --dataset nyu+lilocbench \
    --data-root "$NYU_DATA" \
    --manifest "$NYU_MANIFEST" \
    --lilocbench-manifest "$LILOCBENCH_MANIFEST" \
    --checkpoint-dir "$CKPT_DIR" \
    --log-dir "$LOG_DIR" \
    --backbone efficientvit_b1 \
    --epochs 100 \
    --lr 1e-3 \
    --batch-size 16 \
    --num-workers 8

# ============================================================================
# STEP 5: Evaluate on corridor data
# ============================================================================
echo ""
echo "=== STEP 5: Corridor evaluation ==="

if [ -f "$CORRIDOR_MANIFEST" ]; then
    echo "--- V5 best_depth.pt (NYU + LILocBench) ---"
    python eval_corridor_depth.py \
        --checkpoint "$CKPT_DIR/best_depth.pt" \
        --manifest "$CORRIDOR_MANIFEST" \
        --imagenet-norm

    echo ""
    echo "--- V4 baseline (NYU only, Vivek) ---"
    CKPT_V4=$SCRATCH/checkpoints_v4/vivek_v4_best_depth.pt
    if [ -f "$CKPT_V4" ]; then
        python eval_corridor_depth.py \
            --checkpoint "$CKPT_V4" \
            --manifest "$CORRIDOR_MANIFEST" \
            --imagenet-norm
    fi
else
    echo "[SKIP] Corridor manifest not found: $CORRIDOR_MANIFEST"
fi

echo ""
echo "=== Pipeline complete ==="
echo "V5 checkpoints: $CKPT_DIR/"
echo "  best.pt       - best combined loss"
echo "  best_depth.pt - best depth RMSE"
echo "  best_seg.pt   - best seg mIoU"
