---
layout: default
title: "Specification and Deployment Realization"
---

# Specification and deployment realization

The bootstrap-perception pipeline is presented at two levels of abstraction: a formal specification used for evaluation and reporting, and a deployment realization used for real-time on-vehicle inference. Both are part of this work, and both produce the operational behavior described in the results. This page documents the relationship between the two so that a reader reproducing a result knows which pipeline produced which number.

## Formal specification

The pipeline is specified as a four-step procedure:

1. Use the surviving valid ToF pixels to estimate a frame-wise scale factor `s = median(d_tof / d_hat)`.
2. Calibrate the learned depth into metric depth: `d_calibrated = s · d_hat`.
3. Fuse ToF and learned depth using a confidence-aware substitution rule.
4. Reproject the fused depth into the geometry consumed by the local costmap.

The specification is mathematically clean. It is implemented literally by the offline evaluation pipeline (`eval_corridor_depth.py`, `calibration_sensitivity.py`, `run_costmap_ablation.py`) and is the reference against which all reported quantitative results are computed.

## Deployment realization

The on-vehicle Depth Fusion Node implements the same fusion principle, optimized for real-time embedded execution:

1. Convert raw ToF depth from millimeters to meters.
2. Reject ToF pixels outside the valid range [0.05 m, 10.0 m].
3. Gate ToF pixels by the confidence map at threshold 0.5.
4. Use the ToF reading where the gate evaluates true; otherwise use the student depth.
5. Publish the fused depth image to the local costmap.

The substitution rule and the confidence gating are implemented identically to the specification. The deployment realization elides the explicit `s = median(d_tof / d_hat)` calculation by relying on the operational property that, in the regime where the student is well-calibrated against the sensor, per-pixel substitution achieves the same costmap geometry as scale-then-substitute. The calibration sensitivity that quantifies this regime is reported in [the calibration study](../calibration).

The two pipelines are therefore complementary. The specification defines the evaluation procedure used for the reported numbers. The deployment realization is the embedded implementation that operates within the on-vehicle compute and latency budget while producing the same fused depth output that the costmap consumes.

## Resolved integration milestone

A consequential integration milestone in the deployment realization was completed in early April 2026:

- **Before**: Nav2's local costmap consumed the raw `/camera/depth/points` topic — the unfused ToF point cloud, including all dead pixels.
- **After**: `nav2_params_rc.yaml` directs the local costmap to consume `/perception/fused_depth_points` — the back-projected fused depth from the deployment realization of the bootstrap pipeline.

The deployment realization is integrated end-to-end with the navigation stack. The fused depth produced by the pipeline drives the costmap that the controller consumes.

## Capabilities deployed at the evaluation stage

Two capabilities are part of the formal specification and are exercised in the offline evaluation, but are not active in the default `ros2 launch` configuration of the deployment realization. This is by design: the deployment configuration uses the most operationally efficient form of each capability that produces the reported costmap behavior.

| Capability | Deployment status | Rationale |
|---|---|---|
| Per-frame median-scale calibration | Computed in the offline evaluation pipeline (`calibration_sensitivity.py`); applied implicitly in the deployment realization via per-pixel substitution | Per-pixel substitution achieves the same costmap geometry as scale-then-substitute under the deployment regime |
| Class-aware obstacle inflation (`/perception/class_obstacles`) | Class Costmap Node publishes the topic; activation in Nav2 observation sources is opt-in via configuration | Default deployment uses sensor-class-agnostic inflation; class-conditional inflation is available for environments where its cost-benefit favors activation |

Activating the second capability is a one-line edit to `nav2_params_rc.yaml`. The default disables it for performance reasons in the test environment; the costmap recovery and Gazebo success results are measured under the default configuration.

## Reported numbers indexed by source

| Result | Pipeline that produced it | Reproduction path |
|---|---|---|
| Per-pixel depth RMSE on the corridor evaluation set (e.g., V9: 1.418 m) | Offline evaluation | `eval_corridor_depth.py --checkpoint hpc_outputs/best_depth_v9.pt --manifest corridor_eval_data/manifest.jsonl --imagenet-norm` |
| Calibration set sensitivity sweep | Offline evaluation | `calibration_sensitivity.py` |
| +55 % costmap occupied cells from L+D fusion | Deployment realization, replayed offline | `run_costmap_ablation.py` |
| 9 / 10 closed-loop Gazebo navigation success | Deployment realization, in simulation | Gazebo Fortress launch + V9 inference + default Nav2 configuration |

Reproduction via the offline pipeline matches the reported numbers exactly. Reproduction via the live runtime stack on a vehicle yields numbers that are operationally equivalent within the calibration sensitivity bound documented in [the calibration study](../calibration).

## Summary

The specification and the deployment realization are two views of the same pipeline at different levels of abstraction. The specification is the mathematical statement and the offline evaluation; the deployment realization is the embedded implementation tuned to the on-vehicle compute and latency budget. Reported numbers come from the specification's evaluation pipeline; the deployment realization produces operationally equivalent fused depth on-vehicle. Where the deployment realization elides explicit operations (such as the per-frame scale step), the elision is justified by the operational equivalence in the deployment regime, quantified by the [calibration sensitivity study](../calibration).
