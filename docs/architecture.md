---
layout: default
title: Architecture
---

# System architecture

Four diagrams. Each captures one slice of the system: the two-repository split, the student model, the runtime fusion pipeline, and the training loss. The full picture is all four composed.

---

## 1. Two-repository split

The off-board training pipeline (this repository) produces a TensorRT engine that the on-vehicle runtime (sibling repository `NCHSB`) consumes. The `.engine` artifact is the only file that crosses the boundary.

<div class="mermaid">
flowchart TB
    subgraph HPC["Off-board pipeline (ml_inference, NYU HPC)"]
        direction LR
        T["Teacher inference<br/>DA3-Metric-Large<br/>+ YOLOv8 + SAM2-Large"]
        M["manifest.jsonl<br/>per-frame join"]
        TR["train.py<br/>EfficientViT-B1 student"]
        EX["export_trt.py<br/>ONNX → TensorRT FP16"]
        T --> M --> TR --> EX
    end

    ARTIFACT[".engine file<br/>(TensorRT FP16)"]
    EX --> ARTIFACT

    subgraph JETSON["On-vehicle runtime (NCHSB, Jetson Orin Nano)"]
        direction LR
        STN["Student TRT Node<br/>RGB → depth + seg"]
        DFN["Depth Fusion Node<br/>ToF + student → fused"]
        PCN["PointCloud XYZ Node<br/>fused → 3D points"]
        N2["Nav2 local costmap"]
        STN --> DFN --> PCN --> N2
    end

    ARTIFACT --> STN

    style HPC fill:#e8f0ff
    style JETSON fill:#d4e7c5
    style ARTIFACT fill:#fff3cd
</div>

The HPC pipeline runs offline on NYU Greene (L40S 48 GB partitions). The Jetson runtime runs on a Traxxas Maxx 4S testbed at ~30 Hz perception. The `.engine` artifact is the only build product that needs to be transferred between the two systems.

---

## 2. Student model architecture

EfficientViT-B1 encoder (5.31 × 10⁶ parameters) feeds a 128-channel neck and two parallel transposed-convolution decoders that share skip connections from the encoder. ImageNet normalization is applied within the forward pass to ensure inference-time preprocessing parity with training.

<div class="mermaid">
flowchart TB
    IN["RGB input<br/>[B, 3, 240, 320]<br/>(ImageNet normalization in forward)"]

    ENC["EfficientViT-B1 encoder<br/>4 stages: 40 → 80 → 160 → 256 ch<br/>5.31 × 10⁶ parameters"]

    NECK["Neck: Conv2d(256 → 128, 1×1)"]

    DEC_D["Depth decoder<br/>3 × DecoderBlock<br/>+ Conv2d(16, 1) + ReLU"]
    DEC_S["Segmentation decoder<br/>3 × DecoderBlock<br/>+ Conv2d(16, 6)"]

    OUT_D["Depth output<br/>[B, 1, 240, 320] in meters"]
    OUT_S["Segmentation logits<br/>[B, 6, 240, 320]"]

    IN --> ENC
    ENC --> NECK
    NECK --> DEC_D
    NECK --> DEC_S
    ENC -.skip features<br/>(stages 1, 2, 3).-> DEC_D
    ENC -.skip features<br/>(stages 1, 2, 3).-> DEC_S
    DEC_D --> OUT_D
    DEC_S --> OUT_S

    style ENC fill:#e8f0ff
    style NECK fill:#fff3cd
    style DEC_D fill:#e8f0ff
    style DEC_S fill:#e8f0ff
    style OUT_D fill:#d4e7c5
    style OUT_S fill:#d4e7c5
</div>

The encoder's four stages produce feature maps at H/2, H/4, H/8, and H/16 spatial resolutions with 40, 80, 160, and 256 channels respectively. Each `DecoderBlock(in, out)` implements `Upsample(2×) → Conv2d(in + skip, out, 3×3) → BatchNorm → ReLU`. The skip connection on each block draws from the encoder stage with matching spatial resolution: decoder block 1 receives the stage-3 features, block 2 receives stage-2 features, and block 3 receives stage-1 features. The dashed arrows in the diagram represent this aggregate skip topology; the per-block wiring is uniform.

---

## 3. Runtime depth fusion

Per frame on the Jetson: the student model produces a depth prediction at 240 × 320, the prediction is upsampled and scale-aligned to the surviving ToF pixels, and a per-pixel decision rule combines the two signals into a fused depth image.

<div class="mermaid">
flowchart TB
    RGB["RGB frame<br/>1280 × 720"]
    TOF["ToF depth + confidence map<br/>1280 × 720 (~22 % valid pixels)"]

    STUDENT["V9 student inference<br/>TensorRT FP16, ~5 ms<br/>240 × 320 → upsampled to 1280 × 720"]

    SCALE["Median-scale calibration<br/>s = median(d_ToF / d_student)<br/>over pixels with conf ≥ 0.5"]

    GATE{"Per-pixel decision<br/>conf ≥ 0.5 AND<br/>0.05 ≤ d_ToF ≤ 10.0 m?"}

    FUSED["Fused depth output<br/>1280 × 720"]
    NAV["Nav2 local costmap<br/>(via 3D point cloud reprojection)"]

    RGB --> STUDENT
    STUDENT --> SCALE
    TOF --> SCALE
    SCALE --> GATE
    TOF --> GATE
    GATE -->|"yes: substitute d_ToF"| FUSED
    GATE -->|"no: substitute s · d_student"| FUSED
    FUSED --> NAV

    style STUDENT fill:#e8f0ff
    style SCALE fill:#fff3cd
    style GATE fill:#fff3cd
    style FUSED fill:#d4e7c5
    style NAV fill:#d4e7c5
</div>

Two operations per frame:

1. **Median-scale calibration** — `s = median(d_ToF / d_student)` over the surviving ToF pixels (those with confidence above 0.5). One scalar per frame. See [Scale Calibration](concepts/scale-calibration).
2. **Confidence-gated substitution** — per pixel, use the raw ToF reading if its confidence exceeds 0.5 and its depth lies in the trusted range; otherwise use `s · d_student`. See [Confidence-Gated Fusion](concepts/confidence-gated-fusion).

The student segmentation output (not shown in this diagram) is consumed separately by the Class Costmap Node to apply per-class inflation radii (glass = 0.20 m, person = 0.30 m, wall = 0.12 m). This layer is implemented and available; activation in the default Nav2 observation source list is opt-in via configuration. See [Specification and Deployment Realization](concepts/specification-and-deployment).

---

## 4. Training loss composition

The multi-task loss combines a depth term (berHu), a segmentation term (cross-entropy), and an edge-aware smoothness regularizer. The three terms are combined via Kendall multi-task weighting with `log σ²` clamped to `[-2, 2]`.

<div class="mermaid">
flowchart TB
    INPUTS["Forward-pass outputs and per-frame targets<br/>(RGB · predicted depth · predicted seg logits<br/>hybrid depth target (DA3 or ToF) · 6-class seg GT)"]

    INPUTS --> BERHU["berHu loss<br/>(depth supervision)"]
    INPUTS --> CE["Cross-entropy loss<br/>(segmentation supervision)"]
    INPUTS --> EDGE["Edge-aware smoothness<br/>(depth regularizer)"]

    BERHU --> KENDALL
    CE --> KENDALL
    EDGE --> KENDALL

    KENDALL["Kendall multi-task weighting + edge regularizer<br/>0.5 · exp(-log σ_d) · L_depth + 0.5 · log σ_d<br/>+ exp(-log σ_s) · L_seg + 0.5 · log σ_s<br/>+ λ_edge · L_edge<br/>(log σ clamped to [-2, 2], λ_edge = 0.1)"]

    KENDALL --> TOTAL["Total scalar loss<br/>(backpropagated via AdamW)"]

    style INPUTS fill:#f5f5f5
    style BERHU fill:#e8f0ff
    style CE fill:#e8f0ff
    style EDGE fill:#e8f0ff
    style KENDALL fill:#fff3cd
    style TOTAL fill:#d4e7c5
</div>

The hybrid depth target lives in `models/losses.py:HybridDepthLoss`. Important nuance: the training target chooses between DA3 and ToF at the *frame* level (does this frame have a DA3 label?), while the deployment fusion chooses between ToF and student at the *pixel* level (does this pixel have valid ToF?). Both implement the same supervision principle — prefer hardware ground truth where available, fall back to the learned signal where not — at the granularity appropriate to each stage. See [decisions](decisions#1-specification-and-deployment-realization) for the full mapping.
