---
layout: default
title: "ToF Failure Modes"
---

# ToF failure modes

Why the Orbbec Femto Bolt loses most of its depth pixels in our corridor.

## The numbers

From `results/01_paper_results/pixel_fusion.json`, averaged over 459 corridor frames:

| Statistic | Value |
|---|---|
| Mean dead-pixel rate | **77.79%** |
| Mean valid-pixel rate | **22.21%** |
| Worst-case dead rate (glass corridor) | ~80%+ |
| Confidence threshold for "valid" | 0.5 |

We re-verified this end-to-end inside the Docker container on the same 459 frames:

```
Total pixels: 35,251,200
Valid pixels (sensor 0.1-5.0m): 7,172,823 (20.3%)
Sensor zero (dead): 27,942,585 (79.3%)
Sensor far (>5.0m): 135,792 (0.4%)
```

Slightly different filtering produces slightly different numbers, but both methods land in the same range — the formal evaluation reports 77.8 %, the Docker run measures 79.7 %, and both correspond to the same operational claim: the majority of the frame is unusable.

## What surfaces fail and why

Time-of-Flight depth works by emitting a near-infrared (NIR) pulse and measuring the round-trip time. Failure happens when the returned IR signal is too weak, too late, or too distorted for the sensor to extract a phase.

| Surface | Mechanism | Failure rate |
|---|---|---|
| **Polished floors** (linoleum, marble, sealed concrete) | Specular reflection — the IR pulse bounces away from the sensor at the angle of incidence rather than diffusing back | Very high — close to 100% on smooth surfaces |
| **Glass walls and doors** | IR passes through the glass entirely, or reflects off-axis at the wrong angle | Very high — the hardware can't see glass |
| **Glossy furniture** (lacquered desks, polished metal) | Same specular mechanism as polished floors | High — depends on viewing angle |
| **Distant surfaces** (>5 m) | NIR signal attenuates with distance squared; not enough returned energy to phase-lock | High beyond ~5 m |
| **Wet or transparent surfaces** | Combination of absorption and reflection | High |
| **Matte walls, fabric, painted surfaces** | Diffuse reflection — works as designed | Low — these are the surviving 22% |

The pattern is what the literature calls "specular vs diffuse" — the sensor depends on diffuse reflection, and our deployment environment is dominated by surfaces that don't diffuse well.

## Why this is structured failure, not random failure

This matters for the rest of the bootstrap-perception strategy.

If ToF failures were spatially random — a coin flip per pixel — the surviving 22% would be uniformly distributed and easy to use as a sample. They're not. They cluster on the surfaces that actually diffuse IR: matte walls, fabric, the rough patches between polished sections of floor. The dead regions cluster on the surfaces we most need depth for: the glass we'd hit, the polished floor we'd misjudge clearance over.

Two implications:

1. **Median-scale calibration still works** despite the structured failure, because the surviving pixels are still distributed across the depth range — they're just not uniformly distributed across the *image*.
2. **The dead regions are the operationally important regions.** If the sensor failed on random pixels in the matte wall, we wouldn't care. It fails on glass and floor — exactly the surfaces Nav2 needs to know about.

## How the sensor signals what it's failing on

The Femto Bolt publishes a per-pixel confidence map alongside the depth image (`/camera/depth/confidence`). High confidence means the IR phase lock was strong; low confidence means the sensor itself doesn't trust the reading.

This confidence signal is what makes [Confidence-Gated Fusion](confidence-gated-fusion) work — the system doesn't have to *infer* which depth readings are bad, the sensor *tells* it. Threshold of 0.5 (the value used in the runtime fusion node) keeps about 22% of pixels per frame and rejects the rest.

Without that confidence signal, fusion would have to fall back to heuristics (depth = 0 means dead? depth jumps frame-to-frame mean dead?) and the failure detection would be noisier.

## Why this motivates the whole project

Three options were on the table when this failure mode became clear:

1. **Replace the sensor.** Stereo, LiDAR-3D, structured-light alternatives. Either too expensive, too power-hungry, or not commercially available at the form factor we needed.
2. **Live with the blind spots.** Accept that Nav2 would miss obstacles in 78% of pixels. Untenable for deployment.
3. **Bootstrap.** Use the surviving 22% to anchor a learned monocular depth model, fuse the two outputs, and present a dense fused depth to Nav2.

The third option is the approach taken in this work. It is a pragmatic response to a hardware constraint that cannot be engineered around at the deployment scale. See [Bootstrap Perception](bootstrap-perception) for the full operational specification.
