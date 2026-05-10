---
layout: default
title: Calibration Study
---

# Calibration Sensitivity Study

This experiment was prompted by a reviewer comment on the IEEE RA-L submission:

> *"The affine alignment calibration protocol is underspecified. Eq. (2) says α, β are estimated from 'a calibration set with known ground-truth depth.' How large is this set? Is it scene-specific? This is critical because if α, β are estimated from the same environment being tested, the accuracy results may be optimistic."*

The reviewer's concern is valid. Our evaluation code (`run_costmap_ablation.py`) uses per-frame median scaling: for each frame, compute \\(s = \text{median}(d_{tof}) / \text{median}(d_{DA3})\\) on valid pixels, then scale the DA3 prediction by \\(s\\). This means calibration and evaluation happen on the same frame — the most optimistic possible setup.

The question: how much do the numbers degrade when you use a fixed calibration from a separate set of frames, or from a different scene entirely?

---

## Experiment design

**Script:** `calibration_sensitivity.py`

**Protocol:**
1. Take the 459-frame corridor evaluation set.
2. Split into a calibration set (N frames, randomly sampled) and a test set (remaining 459 - N frames).
3. From the calibration set, compute fixed \\(\alpha, \beta\\) via least-squares: \\(d_{metric} = \alpha \cdot d_{DA3} + \beta\\).
4. Apply the fixed \\(\alpha, \beta\\) to every frame in the test set.
5. Compute AbsRel and RMSE on the test set.
6. Sweep N = {1, 5, 10, 25, 50, 100} frames.
7. Repeat each split 10 times (random seed) to get confidence intervals.

**Cross-scene test:** Calibrate on corridor → test on glass corridor, and vice versa.

---

## Results

### Within-scene: calibration set size

| Calibration frames (N) | Test AbsRel | Test RMSE (m) | Δ vs per-frame |
|---|---|---|---|
| Per-frame (oracle) | 0.087 | 0.522 | — |
| 100 | 0.091 ± 0.002 | 0.538 ± 0.008 | +3.1% |
| 50 | 0.093 ± 0.003 | 0.544 ± 0.011 | +4.2% |
| 25 | 0.097 ± 0.005 | 0.556 ± 0.016 | +6.5% |
| 10 | 0.104 ± 0.008 | 0.578 ± 0.024 | +10.7% |
| 5 | 0.112 ± 0.013 | 0.601 ± 0.035 | +15.1% |
| 1 | 0.138 ± 0.031 | 0.672 ± 0.068 | +28.7% |

The degradation is gradual. With just 10 calibration frames (a 30-second collection at 3 FPS), RMSE increases by ~10%. With 100 frames (~3 minutes of data), the gap is only 3%.

### Cross-scene transfer

| Calibration scene | Test scene | Test AbsRel | Test RMSE (m) |
|---|---|---|---|
| Corridor | Corridor (within-scene, N=100) | 0.091 | 0.538 |
| Corridor | Glass corridor (cross-scene) | 0.142 | 0.723 |
| Glass corridor | Corridor (cross-scene) | 0.119 | 0.621 |
| Glass corridor | Glass corridor (within-scene, N=100) | 0.098 | 0.554 |

Cross-scene calibration hurts. Calibrating on the standard corridor and testing on the glass corridor increases RMSE by ~34%. The scenes have different depth distributions (the glass corridor is wider, with more far-range pixels), which shifts the affine parameters.

---

## What this means for the paper

1. **Per-frame alignment is the most favorable condition**, but it is also standard practice (Eigen et al. 2014, Ranftl et al. 2020 both use per-image alignment for monocular depth evaluation).

2. **A fixed calibration from ~100 frames of the same scene is nearly as good** (+3% RMSE). For deployment, this means collecting 3 minutes of data in the target environment is sufficient.

3. **Cross-scene calibration degrades meaningfully** (~34% RMSE). The affine parameters are partially scene-specific. For a robot deploying in a new building, a brief calibration collection in that building is advisable.

4. The updated paper (Section III / V.B) now states this explicitly and cites this experiment.

---

## Reproducing

```bash
python calibration_sensitivity.py

# Or via Docker
docker compose run calibration
```

Output: `results/calibration_sensitivity.json` with per-N-per-seed breakdown.
