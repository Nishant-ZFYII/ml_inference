#!/bin/bash
# Step 1: Download and extract LILocBench lt_changes_3 (hallway-only, 10.1 GB)
# Run this from a login node (no GPU needed, just network + disk)
#
# Usage: bash step1_download_lilocbench.sh

set -euo pipefail

LILOCBENCH_RAW=$SCRATCH/lilocbench_raw
SEQUENCE_URL="https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/data/data_files_zipped/lt_changes_3.zip"

mkdir -p $LILOCBENCH_RAW
cd $LILOCBENCH_RAW

if [ -d "lt_changes_3" ]; then
    echo "lt_changes_3 already exists, skipping download."
    echo "Contents:"
    ls -la lt_changes_3/
    exit 0
fi

echo "Downloading lt_changes_3.zip (10.1 GB)..."
echo "URL: $SEQUENCE_URL"
wget --show-progress -O lt_changes_3.zip "$SEQUENCE_URL"

echo "Extracting..."
unzip -q lt_changes_3.zip

echo "Cleaning up zip..."
rm lt_changes_3.zip

echo ""
echo "=== Download complete ==="
echo "Location: $LILOCBENCH_RAW/lt_changes_3"
echo "Contents:"
ls -la lt_changes_3/

echo ""
echo "Next: sbatch step2_extract_and_infer.slurm"
