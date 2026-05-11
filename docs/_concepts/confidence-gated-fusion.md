---
layout: default
title: "Confidence-Gated Fusion"
---

# Confidence-gated fusion

The per-pixel decision rule the runtime uses to combine ToF depth with student depth.

## The rule

For each pixel `(i, j)` in the output depth image:

```
if confidence(i, j) >= 0.5 AND 0.05 m < tof_depth(i, j) < 10.0 m:
    fused(i, j) = tof_depth(i, j)            # trust hardware
else:
    fused(i, j) = s · student_depth(i, j)    # use scaled learned depth
```

Where `s` is the per-frame scale factor from [Scale Calibration](scale-calibration). There's no blending — it's a hard switch per pixel.

## Parameters (verified against the runtime depth-fusion node)

| Parameter | Value | What it controls |
|---|---|---|
| `confidence_threshold` | **0.5** | Reject ToF pixels the sensor's own confidence map flags as unreliable |
| `tof_min_range` | **0.05 m** | Reject ToF pixels closer than 5 cm — too noisy at minimum range |
| `tof_max_range` | **10.0 m** | Reject ToF pixels beyond 10 m — beyond the sensor's reliable working distance |

These numbers come from the deployed runtime config and match the values in `Depth Fusion Node`.

## Why hard substitution and not blending

Three reasons the runtime uses substitution instead of weighted blending:

1. **The ToF confidence map is bimodal in practice.** Pixels are either confidently valid (>0.5) or clearly bad (<0.2). The middle band is small, so a soft-blend would only differ from a hard switch on a tiny fraction of pixels.
2. **Student depth has consistent local error structure** — within a region, it's biased the same way. Blending in some sensor depth on a per-pixel basis can introduce frame-to-frame jitter at boundary pixels where confidence hovers near threshold.
3. **It's strictly cheaper.** A hard mask is a single bit per pixel. Soft blending requires a multiply-add per pixel and adds 5-10% latency on the Jetson budget.

Hard substitution also has a neat operational property: the costmap that consumes the fused depth has *exactly* the same spatial obstacle structure as the surviving ToF pixels would, plus extra detail in the regions ToF lost. Adding student-only fill never *moves* a sensor-confirmed obstacle.

## What this gets you operationally

Because student depth fills the dead regions and the median-scale step (see [Scale Calibration](scale-calibration)) anchors absolute scale to the ToF, the fused depth is:

- **Dense** — every pixel has a depth value (no holes from sensor failure)
- **Metric** — anchored to real meters via the surviving sensor pixels
- **Temporally smoother than raw ToF** — student depth is more stable frame-to-frame than the noisy ToF readings the substitution would have used

That last property is counter-intuitive: adding more data shouldn't reduce noise, but it does, because the fill replaces noisy-or-zero ToF samples with consistent learned predictions. Measured temporal jitter on the corridor recordings:

| Configuration | Temporal jitter |
|---|---|
| LiDAR only | 0.61% |
| LiDAR + Depth fusion | **0.43%** (lower) |

Source: `results/01_paper_results/temporal_consistency.json`. The L+D fused output is operationally smoother, not noisier.

## Specification and deployment

The formal specification expresses the fusion as a per-frame affine alignment (`α · d + β`) between student and sensor over the valid set, followed by substitution. The deployment realization performs substitution combined with the implicit median-scale anchoring described in [Scale Calibration](scale-calibration). The two implementations are operationally equivalent in the regime where student bias is dominated by scale rather than offset, which is the operating regime characterized in [the calibration study](../calibration).

See [Specification and Deployment Realization](specification-and-deployment) for the full mapping between the two pipelines.

## Related

- [Bootstrap Perception](bootstrap-perception) — the project-level idea this fusion is part of
- [ToF Failure Modes](tof-failure-modes) — why so many pixels need filling in the first place
- [Scale Calibration](scale-calibration) — how the `s` factor is computed
- [Four-Layer Sensing Hierarchy](four-layer-sensing) — where this fusion sits in the larger sensing stack
