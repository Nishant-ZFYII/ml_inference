---
layout: default
title: Demo Videos
---

# Demo videos

Full-length comparison videos showing every depth source side-by-side across all three evaluation datasets. Generated at native 1280×720 resolution, 10 FPS, with inferno colormap and shared per-frame percentile-based depth range.

---

## Per-version six-panel videos

The recommended way to compare model variants on the corridor evaluation set is the per-version six-panel video sequence, available on each production model's page:

- [V5 — Atlas]({{ '/models/v5-deployment-aug' | relative_url }}#demonstration) (general-purpose model)
- [V6 — Cornerstone]({{ '/models/v6-sun-diode-pretrain' | relative_url }}#demonstration) (fine-tuning base)
- [V7 — Tunnel]({{ '/models/v7-lilocbench-finetune' | relative_url }}#demonstration) (corridor specialist initialized from V5)
- [V9 — Lighthouse]({{ '/models/v9-corridor-specialist' | relative_url }}#demonstration) (production corridor specialist)

Each video shows: RGB input · raw ToF · DA3-Small reference depth · the model's raw inference · ToF+DA3 fusion · ToF+model fusion. The full-length source clips (per-channel `.avi` files) live on the external SSD at `/media/nishant/SeeGayt2/demo_videos/corridor_eval/`. The embedded MP4s on this site are H.264-encoded composites of those sources for browser playback.

---

## Glass corridor

<video controls width="100%" preload="metadata">
  <source src="{{ '/assets/videos/glass_corridor_combined.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the embedded video. <a href="{{ '/assets/videos/glass_corridor_combined.mp4' | relative_url }}">Download MP4</a>.
</video>

121 frames through a glass-walled section. The DA3 and V5/V6 panels mis-predict the geometry behind the glass (predicting the corridor "ends" at the glass surface). V7 and V9, which were fine-tuned on LILocBench corridor data, see through the glass and predict the actual room depth behind it.

---

## Bag 213831

<video controls width="100%" preload="metadata">
  <source src="{{ '/assets/videos/bag_213831_combined.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the embedded video. <a href="{{ '/assets/videos/bag_213831_combined.mp4' | relative_url }}">Download MP4</a>.
</video>

150-frame indoor recording outside the LILocBench corridor distribution. V5 and V6 produce smooth depth, V9 produces noticeably more noise (the corridor specialist generalizing poorly to a non-corridor scene). This is the [V9 generalization caveat](models/v9-corridor-specialist#operational-limitations) made visible.

---

## Individual model videos

Each dataset produces 12 videos: RGB, sensor depth, DA3-Small, four student models (V5, V6, V7, V9), and five fused variants (sensor + each model).

**Generation script:** `generate_demo_videos.py`

| Dataset | Frames | Videos | Total size |
|---|---|---|---|
| `corridor_eval/` | 459 | 12 `.avi` (XVID) | ~588 MB |
| `glass_corridor/` | 121 | 12 `.avi` (XVID) | ~46 MB |
| `bag_213831/` | 150 | 12 `.avi` (XVID) | ~73 MB |

**Video types per dataset:**
- `rgb.avi` — Raw RGB input
- `sensor_depth.avi` — ToF sensor depth (dead pixels shown in grey)
- `da3.avi` — DA3-Small median-scaled depth
- `v5.avi`, `v6.avi`, `v7.avi`, `v9.avi` — Student model depth
- `fused_sensor_da3.avi` — Pixel-level fusion: sensor where valid, DA3 where dead
- `fused_sensor_v5.avi` through `fused_sensor_v9.avi` — Same fusion, one per student model

**Fusion protocol per frame:**
1. Load ToF sensor depth and create valid mask (depth > 0).
2. Run model inference (or load pre-computed `.npy`).
3. Apply median-scale alignment to model prediction.
4. Fused output = sensor depth where valid, aligned model prediction where invalid.
5. Colorize both fused and individual using shared per-frame percentile range (2nd–98th percentile across all depth sources for that frame).

---

## Grid comparison videos

Synchronized multi-panel videos showing all sources in a single frame. Three layout variants:

### Models grid (2×3, 1920×720)

| RGB | DA3-Small | V5 |
|---|---|---|
| V6 | V7 | V9 |

**What it shows:** How each model interprets the same frame. Useful for spotting where students agree/disagree with DA3 and where hallucinations differ between model versions.

### Fused grid (2×3, 1920×720)

| Sensor Depth | Fused DA3 | Fused V5 |
|---|---|---|
| Fused V6 | Fused V7 | Fused V9 |

**What it shows:** How fusion quality varies by model. The sensor depth panel makes it obvious how many pixels are dead (grey) and how each model fills the gaps.

### Combined grid (2×4, 1920×720)

| RGB | Sensor | Fused DA3 | Fused V5 |
|---|---|---|---|
| Fused V6 | Fused V7 | Fused V9 | Stats |

**What it shows:** The full picture in one video. The stats panel displays frame index, dead pixel percentage, depth range, and dataset name — context you need when scrubbing through 459 frames at variable speeds.

**Generation script:** `generate_grid_video.py`

**Generated grid videos:**

| Dataset | Variant | File |
|---|---|---|
| corridor_eval | models | `corridor_eval_grid.avi` |
| corridor_eval | fused | `corridor_eval_grid_fused.avi` |
| corridor_eval | combined | `corridor_eval_grid_combined.avi` |
| glass_corridor | combined | `glass_corridor_grid_combined.avi` |
| bag_213831 | combined | `bag_213831_grid_combined.avi` |

---

## Memory management

Video generation on a 16 GB laptop is constrained. Three strategies keep memory stable:

1. **Stream-to-disk:** `generate_demo_videos.py` writes each frame to disk via `cv2.VideoWriter` immediately after compositing. No frame buffering.

2. **Sequential model processing:** `generate_corridor_missing.py` handles the case where corridor_eval needs live inference for V5/V6/V7 (no pre-computed `.npy`). It loads one model, processes all 459 frames, writes to disk, then explicitly unloads (`del model; torch.cuda.empty_cache(); gc.collect()`) before loading the next.

3. **All-models-loaded grid inference:** `generate_grid_video.py` loads all four student models at startup (combined GPU footprint ~1 GB) and processes frames one at a time. Per-frame inference for all models, compositing, writing, then discarding all per-frame data.

---

## Generating

```bash
# All models, all datasets at native 1280×720
python generate_demo_videos.py \
    --dataset all --models all \
    --output-dir /path/to/demo_videos

# Grid comparison videos
python generate_grid_video.py \
    --dataset all --variant combined \
    --output-dir /path/to/demo_videos

# Missing corridor V5/V6/V7 (if not pre-computed)
python generate_corridor_missing.py
```

Or via Docker:
```bash
docker compose run demo-videos
docker compose run grid-videos
```

Output videos land in the mounted `output/demo_videos/` directory.
