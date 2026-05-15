---
layout: default
title: Hardware
---

# Hardware

The deployment platform. The student model and runtime fusion exist because of these specific sensors and this specific compute budget — knowing what they are and why they were chosen makes the rest of the docs make more sense.

This page covers the on-board hardware. The off-board training hardware (NYU Greene HPC, L40S GPUs) is documented in [training](training).

---

## The robot — Traxxas Maxx 4S

A 1/10 scale Ackermann-steered RC chassis. The body of the robot.

| Property | Value |
|---|---|
| Scale | 1/10 |
| Steering | Ackermann (front-wheel) |
| Wheelbase | 0.187 m |
| Front track | 0.137 m |
| Rear track | 0.145 m |
| Wheel radius | 0.055 m |
| Max steering angle | 0.489 rad (~28°) |
| Footprint | `[[0.20, 0.08], [0.20, -0.08], [-0.10, -0.08], [-0.10, 0.08]]` |

The chassis isn't a background detail. Its geometry shapes the navigation and control stack:

- The Ackermann turning constraint forces the planner to produce drivable paths (no in-place rotation), which is why we use SmacPlannerHybrid + MPPI rather than the typical Nav2 differential-drive defaults
- The footprint defines the inflation distances the costmap needs to respect
- The wheelbase + max steering angle determine the minimum turning radius (~0.65 m), which determines what corridors the robot can actually negotiate
- The drive motor responsiveness (0-60 in ~3 s with the stock 4S setup) is more than enough for indoor use, but it means we run with conservative MPPI velocity limits to keep behavior predictable

The platform was chosen for three reasons: enough space and payload to mount real sensors and compute, fast and sturdy enough to handle the indoor test environment, and a geometry that maps naturally to F1TENTH-style autonomy thinking. Not just "some RC car" — a deliberate physical commitment to treating the project like a small autonomous vehicle.

---

## The depth camera — Orbbec Femto Bolt

The sensor whose failure mode gave the project its identity. RGB-D camera with Time-of-Flight depth and an onboard IMU.

| Stream | Resolution | Rate | Format | Topic |
|---|---|---|---|---|
| RGB | 1280 × 720 | 30 Hz | `Image` | `/camera/color/image_raw` |
| ToF depth | 640 × 576 | 30 Hz | `Image` (16UC1, mm) | `/camera/depth/image_raw` |
| Confidence | 640 × 576 | 30 Hz | `Image` | `/camera/depth/confidence` |
| IMU | — | 200 Hz | `Imu` | `/camera/imu` |

The ToF channel **fails on 77.79% of pixels** in our corridor (polished floors, glass walls, glossy furniture). This is the central failure mode the project was built to address — see [Bootstrap Perception](concepts/bootstrap-perception) and [ToF Failure Modes](concepts/tof-failure-modes) for the full discussion.

We kept this sensor despite the failure rate because the alternative isn't "perfect sensor vs broken sensor." It's "no depth camera at all, vs a sensor that's still highly valuable where it works and informative even where it doesn't." The Femto Bolt provides metric depth where valid, a confidence map that tells us *which* pixels to trust, RGB for the learned fallback, and IMU for the broader stack. That combination is what makes bootstrap perception possible.

The driver is the `orbbec_camera` ROS 2 package. Connection is USB 3.0. Camera intrinsics are available on `/camera/depth/camera_info`.

---

## The LiDAR — RPLiDAR A2M8

2D rotating laser scanner. The reliable backbone of navigation. Always works, but only sees a single horizontal plane.

| Property | Value |
|---|---|
| Range | 12 m (specified); deployment is bounded by the corridor scale, not sensor range |
| Field of view | 360° |
| Scan rate | ~10 Hz |
| Driver | `rplidar_ros` (`ros-humble-rplidar-ros`) |
| Connection | USB serial @ 1,000,000 baud |
| Frame | `lidar_link` |
| Topic | `/scan` (`sensor_msgs/LaserScan`) |

Used by Nav2's local costmap (primary obstacle source), Nav2's global costmap (only source — no depth at the global scale), AMCL (localization), and SLAM Toolbox (mapping).

The LiDAR's limitation isn't quality — it's dimensionality. It sees a thin slice of the world at scanner mount height. That's excellent for wall geometry, planar occupancy, and localization against a 2D map. It's weak for:

- Anything above or below the scan plane (chairs, tabletops, torsos)
- Glass walls (the laser passes through)
- Low obstacles that miss the scan plane entirely

This is exactly why the Femto Bolt depth and learned depth from the student model are needed — they fill the vertical gap the LiDAR can't see. See [Four-Layer Sensing Hierarchy](concepts/four-layer-sensing).

The mental model: the RPLiDAR is the reliable skeleton of the robot's geometry understanding. Bootstrap perception doesn't replace that skeleton. It adds missing body volume where the 2D scan is inherently incomplete.

---

## The compute — Jetson Orin Nano 8GB

The robot's main computer. The board that runs ROS 2, all perception nodes, navigation, and the hardware bridge.

| Property | Value |
|---|---|
| RAM | 8 GB (shared between CPU and GPU) |
| OS | Ubuntu 22.04 (JetPack 6) |
| ROS 2 | Humble |
| GPU | NVIDIA, with TensorRT FP16 inference |

What runs on it:

| Component | What it does |
|---|---|
| Ackermann HW Node | Motor / steering control via Teensy serial bridge |
| Student TRT Node | EfficientViT-B1 inference (V9 student), ~5 ms |
| Depth Fusion Node | Confidence-gated fusion of ToF + student depth |
| YOLO TRT Node | YOLOv8 detection, ~6 ms |
| Class Costmap Node | Class-aware obstacle inflation (when wired) |
| Nav2 | Full navigation stack (planner, controller, costmap) |
| EKF (robot_localization) | Odometry fusion |

USB allocation:
- USB0 → VESC MINI (drive ESC)
- USB1 → Teensy 4.1 (steering, encoders)
- USB2 → RPLiDAR A2M8
- USB3 → Femto Bolt

Runtime perception performance:
- DA3-Small zero-shot: **218 FPS, 4.6 ms, 2.7 GB RAM** (TensorRT FP16, 308 × 308 input)
- EfficientViT-B1 student: **~5 ms** (TensorRT FP16, 240 × 320 input)

The Jetson matters because it's where the project's ambitions hit the deployment budget. It's easy to design a system that works on a desktop GPU. It's much harder to make the whole stack — perception, navigation, localization, and real sensors — fit on an embedded platform that still has to leave room for ROS 2. That budget is why the student model exists in the first place.

### Known issue: TensorRT runtime

TensorRT on the Jetson uses a ctypes CUDA backend instead of pycuda, because pycuda is incompatible with JetPack 6. This is wired up in `trt_utils.py` and works correctly — but it's a non-standard setup that confused us early in deployment. If you're porting the runtime to a different Jetson configuration, this is the pitfall.

---

## What this hardware setup forces

The hardware budget is tight enough that several design choices weren't really choices:

1. **The student must be small.** ~5 MB ONNX, ~5 ms inference. EfficientViT-B1 sits at the upper end of what fits.
2. **Teachers can't run on the robot.** DA3-Metric-Large alone exceeds the 8 GB RAM budget. Distillation on HPC is the only path.
3. **Perception has to fit alongside Nav2.** The student plus YOLO plus the fusion node plus class costmap is the full perception budget. Adding a fourth network would push the system into swap.
4. **2D LiDAR is the navigation backbone.** A 3D LiDAR would solve the vertical-extent problem but would cost more than the rest of the robot combined and wouldn't fit the form factor.
5. **The depth camera matters even when it fails.** ToF gives metric anchor where it works and confidence info everywhere — both are load-bearing for the bootstrap pipeline.

If any of these constraints relaxed (more RAM, faster GPU, 3D LiDAR), the system architecture would look different. Bootstrap perception is a response to the specific shape of *this* hardware budget.
