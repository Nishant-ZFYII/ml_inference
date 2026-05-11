---
layout: default
title: "Bootstrap Perception"
---

# Bootstrap perception

The core idea behind this project. One sentence:

> The 22.21% of ToF pixels that survive on reflective indoor surfaces are enough to calibrate a learned monocular depth model, which then fills the 77.79% the sensor lost.

The hardware sensor "bootstraps" the neural network — providing just enough metric anchor that the learned prediction can fill the dead regions without the model needing to be trustworthy on its own.

## The problem this exists to solve

The deployment camera is the [Orbbec Femto Bolt](../hardware) Time-of-Flight depth sensor. On the reflective indoor surfaces we care about — polished floors, glass walls, glossy furniture — the sensor returns valid depth on roughly **22.2%** of pixels per frame. The other 77.8% are zeroed or flagged unreliable.

Without recovery, Nav2's local costmap inherits those dead regions as blind spots. The 2D LiDAR is still working but scans only one horizontal plane — it doesn't see chairs, tabletops, or torsos. The robot's downward-facing perception collapses exactly where it most needs trustworthy geometry.

This is the failure mode that gave the project its identity. Without it, much of the perception story would be unnecessary.

## Why we don't just throw the sensor away

The fair question: if the ToF fails 78% of the time, why keep it?

Because the alternative isn't "perfect sensor vs broken sensor." It's "no depth camera vs a sensor that's still highly valuable where it works and informative even where it doesn't." The Femto Bolt provides four useful signals:

1. **Metric depth where valid** — exact, hardware-grounded, in meters
2. **Confidence map per pixel** — a signal about *which* pixels to trust
3. **RGB stream** — input to the learned model
4. **IMU stream** — feeds the EKF for odometry

The 22.2% of valid pixels per frame are enough to scale-calibrate the learned depth (see [Scale Calibration](scale-calibration)). The confidence map tells us where to gate (see [Confidence-Gated Fusion](confidence-gated-fusion)). RGB feeds the student model. None of those signals would exist without the sensor.

The deeper point: the failure is *spatially structured*, not random. The sensor doesn't go uniformly bad everywhere — it fails in visually meaningful regions (specular floors, glass) and works in others (matte walls, mid-field obstacles). That structure is what makes the surviving pixels useful as an anchor instead of just noise.

## Why "bootstrap"

The system is self-referential: the degraded hardware sensor provides exactly enough information to fix its own failure by calibrating a learned backup. The trustworthy fragments of the system recover the parts that are currently weak.

That word matters because it captures the project's design philosophy: the system should fail gracefully by using surviving trustworthy pieces of itself to repair what's currently broken — not by replacing failed components with unverified alternatives.

## Why this is stronger than three easier stories

Three competing framings would have been easier and weaker:

**"Just use the model"** — fashionable but technically weak. A learned depth model without a physical anchor is much harder to trust in deployment. Every frame is a leap of faith.

**"Just use the depth camera"** — ignores the failure mode that motivated the project in the first place and produces exactly the blind spots we're trying to fix.

**"Just fuse everything somehow"** — sounds practical but vague. Without a clear trust hierarchy, fusion becomes engineering by accumulation rather than a principled sensing strategy.

Bootstrap perception is stronger because it says four specific things:

1. Keep what's physically trustworthy.
2. Identify where it fails.
3. Repair only the failed regions.
4. Convert that repaired geometry into something navigation can actually consume.

## How it works (operationally)

Three steps:

1. **[Scale Calibration](scale-calibration)** — `s = median(d_tof / d_hat)` over pixels where ToF confidence ≥ 0.5. The median is robust to a few bad readings; ~22% of pixels is more than enough sample size.
2. **[Confidence-Gated Fusion](confidence-gated-fusion)** — per pixel: `if confidence ≥ 0.5 AND 0.05 m < tof < 10.0 m → use ToF`, else `use s · student`.
3. **Costmap Integration** — fused depth is back-projected to a 3D point cloud and consumed by Nav2's local costmap.

The training loss mirrors this fusion structure (see `models/losses.py:HybridDepthLoss`) — the student learns to predict what the *fused output* should look like, not what either sensor alone reports.

## Key results this idea is responsible for

- **+55% costmap occupied cells** in narrow corridors (LiDAR + Depth fusion vs LiDAR alone)
- **Temporal jitter reduced** from 0.61% (LiDAR-only) to 0.43% (L+D) — fusion is smoother than raw ToF because the learned depth provides temporally consistent fill
- **9 / 10 Gazebo navigation success** with the [V9](../models/v9-corridor-specialist) student — matches ground-truth depth on the same seeds

## In one breath

Bootstrap perception means the robot doesn't choose between hardware depth and learned depth as if one must replace the other. It uses the surviving valid hardware pixels to anchor a learned dense prediction, then turns that repaired depth into obstacle geometry that Nav2 can use. The project is about reliability under sensor failure, not about claiming a new state-of-the-art depth model.
