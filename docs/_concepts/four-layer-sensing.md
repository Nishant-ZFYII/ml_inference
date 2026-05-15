---
layout: default
title: "Four-Layer Sensing Hierarchy"
---

# Four-layer sensing hierarchy

The architectural pattern that constitutes the principal system-level contribution of this work. Each layer compensates for limitations of the layer above it, producing graceful degradation when any single sensor fails.

## The four layers, in priority order

| Layer | Sensor / source | What it provides | What it can't see |
|---|---|---|---|
| **1. LiDAR (RPLiDAR A2M8)** | 2D rotating laser scanner | Reliable obstacle distance on one horizontal plane, always works | Anything above or below the scan plane: chairs, tabletops, torsos |
| **2. Hardware depth (Femto Bolt ToF)** | Time-of-Flight depth camera | Dense 3D depth where it works | Reflective surfaces (glass, polished floors) — fails on 77.8% of pixels |
| **3. Learned depth (V9 student)** | EfficientViT-B1 monocular depth | Fills the regions where ToF failed | Has no class information; raw scale needs anchoring |
| **4. Semantic segmentation (V9 student)** | Same student model, segmentation head | Class-aware costmap inflation (glass gets wider berth than wall) | Only operates on regions visible in the camera FOV |

Each layer's strength compensates for the layer above's specific weakness:

- LiDAR misses vertical extent → depth cameras add 3D
- ToF fails on glass → learned depth fills the gaps
- Learned depth has no class info → segmentation adds semantics
- Semantic segmentation alone has no geometry → the fused depth provides it

## Why the hierarchy matters more than any one layer

The architectural claim is not that any single layer is best in class. The claim is that this specific *composition* of four sensing layers degrades gracefully under realistic failure conditions:

- If the camera fails entirely, LiDAR keeps the robot from hitting walls (degraded but safe)
- If the LiDAR fails, the depth camera plus learned depth still produces a costmap
- If only the ToF channel fails, the learned depth + LiDAR composition still navigates
- If the segmentation head misclassifies, depth still produces correct geometry — just with default inflation everywhere

A single-sensor system can't degrade gracefully because there's nothing to fall back to. A multi-sensor system without a clear hierarchy can degrade *unpredictably* — losing one sensor changes how the others get weighted in ways that depend on the failure mode. The four-layer hierarchy is the design that makes the degradation predictable.

## Where each layer lives operationally

| Layer | ROS 2 node (NCHSB repo) | Output topic |
|---|---|---|
| 1 — LiDAR | `rplidar_node` | `/scan` |
| 2 — ToF | `orbbec_camera` | `/camera/depth/image_raw`, `/camera/depth/confidence` |
| 3 — Learned depth | Student TRT Node (V9 inference) | `/student/depth` |
| 4 — Semantic seg | Same node, second head | `/student/segmentation` |
| Fusion | Depth Fusion Node | `/perception/fused_depth` |
| Class-aware costmap | Class Costmap Node | `/perception/class_obstacles` (optional) |

The fusion (layer 2 + layer 3) is the [Confidence-Gated Fusion](confidence-gated-fusion) per-pixel substitution. The class-aware costmap layer is implemented and available; activation in the Nav2 observation source list is opt-in via configuration (see [Specification and Deployment Realization](specification-and-deployment)).

## Per-class inflation (layer 4)

The class costmap node turns segmentation classes into per-class obstacle inflation radii:

| Class ID | Class | Inflation radius | Why |
|---|---|---|---|
| 0 | Floor | 0.00 m | Drivable surface |
| 1 | Wall | 0.12 m | Standard obstacle (matches base inflation) |
| 2 | Person | 0.30 m | Safety-critical, dynamic |
| 3 | Furniture | 0.15 m | Standard obstacle |
| 4 | **Glass** | **0.20 m** | Invisible to ToF and LiDAR — only detectable by learned segmentation |
| 5 | Other | 0.10 m | Default |

Glass is the operationally critical class. It's invisible to layers 1 and 2 entirely. Layer 4 is the only place the system has any way to know glass is there.

## In one breath

The hierarchy is the contribution. Each individual component is unremarkable on its own; the value is in how they compose so that any single failure leaves a navigable system behind.
