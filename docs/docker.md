---
layout: default
title: Docker
---

# Docker Reproducibility

The Docker setup provides a self-contained environment for running evaluation, calibration experiments, and video generation without installing PyTorch, OpenCV, or managing conda environments.

---

## What Docker does and does not include

**Included in the image:**
- Python 3.10, PyTorch, timm, OpenCV, all dependencies from `requirements.txt`
- All evaluation scripts, model definitions, dataset loaders
- Paper results JSON files (for verification)

**Not included (volume-mounted at runtime):**
- Model weights (`hpc_outputs/*.pt`, ~61 MB each)
- Evaluation data (`corridor_eval_data/`, ~1.2 GB)
- Output directory for generated videos and results

This keeps the image small (~2 GB) and avoids baking 1.5 GB of data into a Docker layer.

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/Nishant-ZFYII/ml_inference.git
cd ml_inference

# 2. Download model weights and eval data (from HuggingFace)
# Place weights in hpc_outputs/, eval data in corridor_eval_data/

# 3. Build
docker compose build

# 4. Smoke test (no data needed)
docker compose run smoke-test

# 5. Run corridor evaluation
docker compose run eval-corridor

# 6. Run calibration sensitivity experiment
docker compose run calibration

# 7. Generate demo videos
docker compose run demo-videos

# 8. Generate grid comparison videos
docker compose run grid-videos
```

---

## Available services

| Service | What it runs | Data needed | GPU needed |
|---|---|---|---|
| `smoke-test` | Model forward pass on random tensor | None | No |
| `eval-corridor` | V9 depth evaluation on corridor frames | weights + eval data | No (CPU), faster with GPU |
| `calibration` | Calibration sensitivity sweep (N = 1–100) | eval data | No |
| `demo-videos` | Individual model videos (CPU) | weights + eval data | No (slow) |
| `demo-videos-gpu` | Individual model videos (GPU) | weights + eval data | Yes |
| `grid-videos` | Grid comparison videos | weights + eval data | No (CPU), faster with GPU |

---

## GPU passthrough

For GPU-accelerated inference:

```bash
# Single service with GPU
docker compose run demo-videos-gpu

# Or override any service
docker run --rm --gpus all \
    -v $(pwd)/hpc_outputs:/app/hpc_outputs:ro \
    -v $(pwd)/corridor_eval_data:/app/corridor_eval_data:ro \
    -v $(pwd)/output:/app/output \
    ml-inference generate_demo_videos.py --device cuda
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Building from scratch

```bash
# Build the image
docker build -t ml-inference .

# Or with compose (builds all services from the same image)
docker compose build
```

The build installs CPU-only PyTorch by default (via `requirements.txt`). For GPU support inside the container, either:
1. Use the `--gpus all` flag (Docker handles CUDA passthrough from host), or
2. Modify the Dockerfile to install CUDA-enabled PyTorch.

---

## Volume layout

```
host machine                    container
─────────────                   ─────────
./hpc_outputs/     ──(ro)──>    /app/hpc_outputs/
./corridor_eval_data/ ─(ro)──>  /app/corridor_eval_data/
./output/          ──(rw)──>    /app/output/
```

The `hpc_outputs/` and `corridor_eval_data/` volumes are mounted read-only. All generated outputs (results JSON, videos, figures) write to `/app/output/` inside the container, which maps to `./output/` on the host.

---

## Verifying results

After running `eval-corridor`, check that the output matches the paper:

```bash
# Inside output/ after eval-corridor
cat output/corridor_eval_results.json | python -m json.tool
```

Expected: V9 corridor RMSE ~1.418 m, DA3-Small ~0.522 m (on the 459-frame Femto Bolt corridor set).

The smoke test verifies that the model architecture loads correctly, a forward pass produces the expected tensor shapes (depth: 1×1×240×320, seg: 1×6×240×320), and all Python dependencies resolve.
