# V1-V3 Baseline Archive

This folder is a frozen snapshot of the `ml_inference` repository at the V1-V3 era, before the V3 → V4 transition that swapped the teacher (DA2-Large → DA3-Large) and the student backbone (MobileNetV3-Small → EfficientViT-B1).

It exists so that the V1-V3 codebase remains directly browsable in the working tree, alongside the V4-V9 production code that lives at the top level of the repo.

## Provenance

- **Source commit:** `5ca9019` — "Try multiple SAM2 hydra config paths for compatibility" (tip of `main` before the V4-V9 lineage was merged)
- **Tag:** `v1-v3-baseline` (run `git checkout v1-v3-baseline` to reproduce this exact tree at the repo root)
- **Era:** V1, V2, V3 of the student lineage. V3 was the last "first real" version on this codebase (NYU RMSE 1.160 m, MobileNetV3-Small backbone).

## What's in this snapshot

| Component | Implementation at this point |
|---|---|
| Student backbone | MobileNetV3-Small (`models/student.py`) |
| Depth teacher | DA2-Large via torch.hub or HuggingFace (`teacher_infer/run_da2.py`) |
| Segmentation teacher | YOLO-seeded SAM2-Large (`teacher_infer/run_sam2.py`) |
| Depth loss | Hybrid L1 (ToF / DA2 by confidence) + edge-aware smoothness (`models/losses.py`) |
| Datasets | NYU Depth V2 + corridor swap-in (`dataset/`) |
| Export | ONNX + TensorRT FP32/FP16/INT8 (`export_trt.py`) |
| Benchmark | Jetson latency + GPU memory + RMSE (`benchmark_jetson.py`) |

## What changed after this archive

The V3 → V4 transition (commit `074c771`, "Replace DA2 + MobileNetV3 with DA3-Metric-Large + EfficientViT-B1") and subsequent V5–V9 iterations live at the top level of the repo. See the top-level README for the V4-V9 lineage and paper-relevant scripts.

## Why preserve as files (not just git history)

Git history alone preserves these commits — `git checkout v1-v3-baseline` reproduces this exact state. This folder duplicates that into the V4-V9-era working tree so the V1-V3 implementation can be read, diffed against, or referenced without checking out a different commit.

This snapshot is **not maintained** — it is a reference point. Bug fixes and improvements go to the V4-V9 code at the top level.
