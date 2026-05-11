---
layout: default
title: "Concepts"
permalink: /concepts/
---

# Concepts

The conceptual foundations referenced throughout this technical report. Each page is a self-contained treatment of one idea. Pages may be read in any order; the model lineage and evaluation pages link back to these when a concept is introduced by name.

| Page | Subject |
|---|---|
| [Bootstrap Perception]({{ '/concepts/bootstrap-perception' | relative_url }}) | The system-level idea that surviving valid sensor pixels calibrate a learned monocular depth model to fill in for the failed regions. |
| [ToF Failure Modes]({{ '/concepts/tof-failure-modes' | relative_url }}) | The mechanism by which the Femto Bolt loses 77.79 % of its pixels in the corridor environment, and the surface classes that produce the failure. |
| [Scale Calibration]({{ '/concepts/scale-calibration' | relative_url }}) | The median-scale alignment step that anchors the student's geometry to metric units using the surviving ToF pixels. |
| [Confidence-Gated Fusion]({{ '/concepts/confidence-gated-fusion' | relative_url }}) | The per-pixel runtime decision rule combining hardware ToF where valid and scaled student depth elsewhere. |
| [Four-Layer Sensing Hierarchy]({{ '/concepts/four-layer-sensing' | relative_url }}) | The graceful-degradation chain spanning LiDAR, hardware depth, learned depth, and semantic segmentation. |
| [Specification and Deployment Realization]({{ '/concepts/specification-and-deployment' | relative_url }}) | The relationship between the formal pipeline specification (used for evaluation and reporting) and the embedded deployment realization (used for on-vehicle inference). |
| [Knowledge Distillation]({{ '/concepts/knowledge-distillation' | relative_url }}) | The teacher–student training paradigm and the rationale for off-board teacher inference combined with on-board student deployment. |
