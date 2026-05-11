---
layout: default
title: Docker
---

# Docker reproducibility

The Docker setup gives you the evaluation pipeline in one container without installing PyTorch, OpenCV, or maintaining a conda environment. The setup was built and verified end-to-end on a Linux host. The notes below reflect what actually works, including the gotchas.

---

## Verified status

| Step | Result |
|---|---|
| `docker build` | ✅ Image: 6.29 GB (PyTorch wheel is the bulk; CPU-only by default) |
| `smoke-test` | ✅ Model forward pass: depth `(1, 1, 240, 320)` + seg `(1, 6, 240, 320)`. PyTorch 2.11.0+cu130, OpenCV 4.13.0, NumPy 2.2.6 |
| `eval-corridor` (459 frames, V9, CPU) | ✅ RMSE 1.366 m, sensor dead-pixel rate 79.7% |

The Docker eval RMSE (1.366 m) is consistent with the 1.418 m number reported in the formal evaluation; the small difference reflects that the formal evaluation averages over a slightly larger frame subset and both numbers are uncalibrated. The calibration-aligned oracle RMSE is 0.522 m — see [calibration](calibration). The 79.7 % dead-pixel rate measured here matches the 77.8 % reported in the formal evaluation within the same subset effect (the formal number averages over the full hardware-released frame set while the Docker run uses the released eval subset).

---

## What's in the image vs volume-mounted

**Inside the image (~6.3 GB):**
- Python 3.10 (Debian Bookworm base, pinned)
- PyTorch 2.11.0+cu130, timm, ONNX, OpenCV-headless, all `requirements.txt` deps
- All evaluation, inference, and video-generation scripts
- `models/`, `dataset/`, `config.py`
- `results/` JSON files (for verification against the reported numbers)

**Mounted at runtime:**
- `hpc_outputs/` — model weights (~61 MB per checkpoint, currently 7 versions)
- `corridor_eval_data/` — 459 RGB + 459 sensor depth + 459 DA3 depth (~1.2 GB)
- `output/` — generated videos, results JSON, figures

The image deliberately doesn't bake in the weights or eval data: licensing, image size, and the fact that you might want to plug in your own dataset.

---

## Quick start

```bash
git clone https://github.com/Nishant-ZFYII/ml_inference.git
cd ml_inference

# Build (see "Build gotchas" below for --network=host)
docker build --network=host -t ml-inference .

# Smoke (no data needed)
docker compose run --rm smoke-test

# Corridor evaluation — needs hpc_outputs/best_depth_v9.pt and corridor_eval_data/
docker compose run --rm eval-corridor
```

Compose reads `image: ml-inference:latest` from the base service, so once the image is built the `run --rm <service>` calls will reuse it instead of rebuilding.

---

## Available services

| Service | What it runs | Data needed | Approx. runtime |
|---|---|---|---|
| `smoke-test` | Forward pass on a random tensor | None | < 5 s |
| `eval-corridor` | V9 depth eval, 459 frames | weights + eval data | ~3 min CPU |
| `calibration` | Calibration sensitivity sweep, N ∈ {1,5,10,25,50,100} + cross-scene | eval data | ~5 min CPU |
| `demo-videos` | 12 individual videos for one dataset (CPU) | weights + eval data | ~25 min CPU |
| `demo-videos-gpu` | Same with `--device cuda` | weights + eval data, NVIDIA toolkit | ~3 min GPU |
| `grid-videos` | 3 grid-comparison videos (CPU) | weights + eval data | ~30 min CPU |

---

## Build gotchas

Two things tripped the first build attempt. Both are now patched in the Dockerfile and compose file, but worth knowing if you change either.

### 1. `--network=host` for restricted DNS

On networks where the Docker bridge nameserver can't resolve external hosts (e.g. NYU campus, many corporate networks), the build's `apt-get update` will hang for ~12 minutes and then fail with `Temporary failure resolving 'deb.debian.org'`.

The fix lives in the compose file (`build.network: host`). For one-off `docker build`, pass it explicitly:

```bash
docker build --network=host -t ml-inference .
```

### 2. Debian version pin

The Dockerfile pins `python:3.10-slim-bookworm` rather than `python:3.10-slim`. The unpinned tag now points to Debian Trixie, which renamed `libgl1-mesa-glx` to `libgl1`. We don't actually need libGL because we use `opencv-python-headless`, so the dependency was removed entirely; the pin guards against the next rename.

---

## GPU passthrough

For GPU-accelerated inference (substantially faster on `demo-videos` and `grid-videos`):

```bash
# Use the GPU-flavored compose service
docker compose run --rm demo-videos-gpu

# Or override an arbitrary script with --gpus all
docker run --rm --gpus all \
  -v $(pwd)/hpc_outputs:/app/hpc_outputs:ro \
  -v $(pwd)/corridor_eval_data:/app/corridor_eval_data:ro \
  -v $(pwd)/output:/app/output \
  ml-inference generate_demo_videos.py --device cuda --dataset corridor
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The image's PyTorch wheel includes CUDA 13.0 runtime, so it works against any modern host CUDA driver.

---

## Volume layout

```
host                              container
────────────────────              ───────────────────────
./hpc_outputs/      ──(ro)──>     /app/hpc_outputs/
./corridor_eval_data/ ──(ro)──>   /app/corridor_eval_data/
./output/           ──(rw)──>     /app/output/
```

`hpc_outputs/` and `corridor_eval_data/` are read-only by design — anything the container writes lands in `./output/` on the host.

---

## What the smoke test checks

```python
import torch, cv2, numpy as np
from models.student import build_student
from config import Config
cfg = Config()
m = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False, backbone=cfg.BACKBONE)
x = torch.randn(1, 3, 240, 320)
d, s = m(x)
assert d.shape == (1, 1, 240, 320)
assert s.shape == (1, 6, 240, 320)
```

Three things in one go: the EfficientViT-B1 backbone loads (timm pretrained=False so no network call), the dual decoders produce the expected shapes, and torch + opencv + numpy import. If any of those break, you'll see it in 5 seconds instead of 5 minutes into a real evaluation.

---

## Image size

6.29 GB is bigger than the README's old "~2 GB" estimate. The bulk is the PyTorch wheel (~3 GB unpacked, includes CUDA runtime). Other contributors:

- `python:3.10-slim-bookworm` base: ~150 MB
- ffmpeg + libglib2.0-0 (for video encoding): ~250 MB
- `requirements.txt` deps minus torch (timm, transformers, opencv, scipy, sklearn, datasets): ~600 MB
- App code + results JSON: ~30 MB

Cutting this further would mean splitting into a CPU-only image (no CUDA) and a GPU image — possible but not worth the maintenance overhead while the project has a single deployment story.
