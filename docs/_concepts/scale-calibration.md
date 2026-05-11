---
layout: default
title: "Scale Calibration"
---

# Scale calibration

The "bootstrap" step — using surviving ToF pixels to find the correct metric scale for learned depth.

## The equation

Per frame, over pixels where ToF confidence ≥ 0.5:

$$s = \text{median}\left(\frac{d_{\text{tof}}}{\hat{d}}\right)$$

- $d_{\text{tof}}$ — hardware ToF depth at valid pixels
- $\hat{d}$ — raw student-predicted depth at the same pixels
- $s$ — scalar scale factor, applied to the entire student depth map

After computing `s`, the student depth gets re-scaled: `d_student_scaled = s · d_hat`. The scaled student depth then participates in [Confidence-Gated Fusion](confidence-gated-fusion).

## Why median, not mean

Three reasons:

1. **Robust to outliers.** A handful of misread ToF pixels (specular spike, edge case at close range) don't corrupt the estimate. Median is the natural robust statistic for ratio-of-noisy-signals.
2. **Symmetric in the ratio direction.** If the student systematically over-predicts on one half of the frame and under-predicts on the other, the median picks the central tendency without being pulled by either tail.
3. **Computationally trivial.** No iterative fitting, no convergence concerns — just a sort. Runs at >100 Hz on the Jetson without optimization.

## Why this works at 22% pixel coverage

The surviving 22% of ToF pixels per frame is more than enough sample size for a median estimate. Ten valid samples is enough; we typically have tens of thousands. The bottleneck isn't sample count, it's whether the surviving pixels span the depth range — and they generally do, because failures are surface-specific (see [ToF Failure Modes](tof-failure-modes)) rather than depth-specific.

The valid samples cluster on matte walls, fabric, and rough surfaces — but those surfaces appear at varying depths across a typical corridor frame. Near walls, mid-field surfaces, and far walls all contribute valid samples. The median ratio is robust across that depth range.

## Why this is needed in the first place

The student model (EfficientViT-B1, see the [V9 page](../models/v9-corridor-specialist) for the deployment checkpoint) outputs depth in a learned scale. That scale is approximately metric — the model trains against DA3-Metric-Large which is calibrated to meters — but per-frame absolute scale can drift due to:

- The training distribution's average focal length differing from the deployment camera
- The deployment camera's exact intrinsics not matching the training calibration
- Frame-to-frame variation in which structures are visible (a frame dominated by far-field gets a different absolute scale than a near-field frame)

The median-scale step corrects all three at once, per frame, using the surviving sensor pixels. Absolute scale doesn't have to be perfect because the calibration fixes it.

## What the model has to get right (since it doesn't have to get scale right)

If absolute scale gets corrected per frame, the model only has to get the *shape* of the depth map right:

- Spatial structure of obstacles (where the wall is, where the floor ends)
- Relative depth between regions (which obstacle is closer)
- Smoothness within regions (no holes or spikes within a flat surface)
- Sharpness at boundaries (depth discontinuity at object edges)

This is why the project doesn't lean too hard on raw RMSE numbers. The absolute meter-error matters less than the structural correctness, because median-scale alignment is part of the runtime.

## Specification and deployment

The median-scale equation as written is the formal specification, and is computed explicitly by the offline evaluation pipeline (`calibration_sensitivity.py`). The on-vehicle deployment realization applies the same principle implicitly: confidence-gated substitution achieves the same costmap geometry as scale-then-substitute under the deployment regime, which avoids an explicit per-frame median computation in the latency-critical path.

The two implementations are operationally equivalent within the calibration sensitivity bound reported in [the calibration study](../calibration). See [Specification and Deployment Realization](specification-and-deployment) for the full mapping between the two pipelines.

## Related

- [Bootstrap Perception](bootstrap-perception) — the project-level idea this calibration step is part of
- [Confidence-Gated Fusion](confidence-gated-fusion) — what happens after the scale is computed
- [Calibration sensitivity study](../calibration) — how the calibration set size affects accuracy
