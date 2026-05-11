---
layout: default
title: Architecture
---

# System architecture

Four diagrams. Each captures one slice of the system: the two-repository split, the student model, the runtime fusion pipeline, and the training loss. All four use a left-to-right reading direction; the page CSS allows horizontal scroll for wide diagrams when the viewport is narrow.

---

## 1. Two-repository split

The off-board training pipeline (this repository) produces a TensorRT engine that the on-vehicle runtime (sibling repository `NCHSB`) consumes. The `.engine` artifact is the only file that crosses the boundary.

<div class="mermaid">
flowchart LR
    T["Teacher inference<br/>DA3 + YOLO + SAM2"]:::hpc
    M["manifest.jsonl"]:::hpc
    TR["train.py<br/>EfficientViT-B1<br/>student"]:::hpc
    EX["export_trt.py<br/>ONNX → TensorRT"]:::hpc
    AR[".engine<br/>(boundary)"]:::pivot
    STN["Student TRT<br/>Node"]:::jet
    DFN["Depth Fusion<br/>Node"]:::jet
    PCN["PointCloud<br/>XYZ Node"]:::jet
    N2["Nav2 local<br/>costmap"]:::jet

    T --> M --> TR --> EX --> AR --> STN --> DFN --> PCN --> N2

    classDef hpc fill:#e8f0ff,stroke:#999
    classDef jet fill:#d4e7c5,stroke:#999
    classDef pivot fill:#fff3cd,stroke:#666,stroke-width:2px
</div>

Blue nodes run on NYU Greene HPC (L40S 48 GB partitions). Green nodes run on the Jetson Orin Nano at ~30 Hz perception. The yellow `.engine` artifact is the only build product that needs to be transferred between the two systems.

---

## 2. Student model architecture

EfficientViT-B1 encoder (5.31 × 10⁶ parameters), 128-channel neck, two parallel transposed-convolution decoders. ImageNet normalization is applied within the forward pass for inference-time preprocessing parity.

<div class="mermaid">
flowchart LR
    IN["RGB input<br/>[B, 3, 240, 320]"]:::input
    ENC["EfficientViT-B1<br/>encoder<br/>(5.31 × 10⁶ params)"]:::block
    NECK["Neck<br/>Conv2d(256→128, 1×1)"]:::neck
    DEC_D["Depth decoder<br/>3 × DecoderBlock<br/>+ Conv2d(16,1) + ReLU"]:::block
    DEC_S["Seg decoder<br/>3 × DecoderBlock<br/>+ Conv2d(16,6)"]:::block
    OUT_D["Depth output<br/>[B, 1, 240, 320]<br/>metric (m)"]:::output
    OUT_S["Seg logits<br/>[B, 6, 240, 320]"]:::output

    IN --> ENC --> NECK
    NECK --> DEC_D --> OUT_D
    NECK --> DEC_S --> OUT_S
    ENC -.skip features.-> DEC_D
    ENC -.skip features.-> DEC_S

    classDef input fill:#f5f5f5,stroke:#999
    classDef block fill:#e8f0ff,stroke:#999
    classDef neck fill:#fff3cd,stroke:#999
    classDef output fill:#d4e7c5,stroke:#999
</div>

The encoder's four stages produce feature maps at H/2, H/4, H/8, and H/16 resolutions with 40, 80, 160, and 256 channels. Each `DecoderBlock(in, out)` implements `Upsample(2×) → Conv2d(in + skip, out, 3×3) → BatchNorm → ReLU`. Skip connections route per-block: decoder block 1 receives stage-3 features, block 2 stage-2, block 3 stage-1. The dashed arrows represent this aggregate skip topology; per-block wiring is uniform across the two decoder paths.

---

## 3. Runtime depth fusion

Per frame on the Jetson: student inference, per-frame median-scale calibration against surviving ToF pixels, per-pixel confidence-gated substitution.

<div class="mermaid">
flowchart LR
    RGB["RGB frame<br/>1280 × 720"]:::input
    TOF["ToF depth<br/>+ confidence<br/>(~22 % valid)"]:::input
    STUDENT["V9 student<br/>TensorRT FP16<br/>~5 ms"]:::block
    SCALE["Median-scale<br/>s = median(d_ToF / d_V9)<br/>over conf ≥ 0.5 pixels"]:::calib
    GATE{"Per-pixel:<br/>conf ≥ 0.5 AND<br/>0.05 ≤ d_ToF ≤ 10.0 m?"}:::calib
    FUSED["Fused depth<br/>1280 × 720"]:::output
    NAV["Nav2 local<br/>costmap"]:::output

    RGB --> STUDENT --> SCALE --> GATE
    TOF --> SCALE
    TOF --> GATE
    GATE -->|yes: d_ToF| FUSED
    GATE -->|no: s · d_V9| FUSED
    FUSED --> NAV

    classDef input fill:#f5f5f5,stroke:#999
    classDef block fill:#e8f0ff,stroke:#999
    classDef calib fill:#fff3cd,stroke:#999
    classDef output fill:#d4e7c5,stroke:#999
</div>

Two operations per frame:

1. **Median-scale calibration** — `s = median(d_ToF / d_student)` over pixels with confidence ≥ 0.5. One scalar per frame. See [Scale Calibration](concepts/scale-calibration).
2. **Confidence-gated substitution** — per pixel, use the raw ToF reading where confident and in range; otherwise use `s · d_student`. See [Confidence-Gated Fusion](concepts/confidence-gated-fusion).

The student segmentation output (not shown here) is consumed separately by the Class Costmap Node to apply per-class inflation radii (glass = 0.20 m, person = 0.30 m, wall = 0.12 m). Activation in the default Nav2 observation source list is opt-in via configuration. See [Specification and Deployment Realization](concepts/specification-and-deployment).

---

## 4. Training loss composition

The multi-task loss combines a depth term (berHu), a segmentation term (cross-entropy), and an edge-aware smoothness regularizer, combined via Kendall multi-task weighting.

<div class="mermaid">
flowchart LR
    INPUTS["Forward outputs<br/>+ per-frame targets"]:::input
    BERHU["berHu loss<br/>(depth)"]:::loss
    CE["Cross-entropy<br/>(segmentation)"]:::loss
    EDGE["Edge-aware<br/>smoothness"]:::loss
    KENDALL["Kendall weighting<br/>+ edge regularizer<br/>(log σ² ∈ [-2, 2])"]:::weight
    TOTAL["Total scalar loss<br/>(AdamW backprop)"]:::output

    INPUTS --> BERHU --> KENDALL
    INPUTS --> CE --> KENDALL
    INPUTS --> EDGE --> KENDALL
    KENDALL --> TOTAL

    classDef input fill:#f5f5f5,stroke:#999
    classDef loss fill:#e8f0ff,stroke:#999
    classDef weight fill:#fff3cd,stroke:#999
    classDef output fill:#d4e7c5,stroke:#999
</div>

The hybrid depth target lives in `models/losses.py:HybridDepthLoss`. Important nuance: the training target chooses between DA3 and ToF at the **frame** level (does this frame have a DA3 label?), while the deployment fusion chooses between ToF and student at the **pixel** level (does this pixel have valid ToF?). Both implement the same supervision principle — prefer hardware ground truth where available, fall back to the learned signal where not — at the granularity appropriate to each stage. See [decisions](decisions#1-specification-and-deployment-realization) for the full mapping.
