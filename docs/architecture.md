---
layout: default
title: Architecture
---

# System architecture

Three diagrams. Each captures one slice of the system: training, inference, and deployment. The full picture is all three composed.

---

## 1. Two-repo split

This site documents the left half. The right half lives in [`NCHSB`](https://github.com/Nishant-ZFYII/NCHSB).

<div class="mermaid">
flowchart LR
    subgraph HPC["NYU HPC (this repo: ml_inference)"]
        T1["DA3-Metric-Large<br/>(depth teacher)"]
        T2["YOLOv8-Large<br/>(detection)"]
        T3["SAM2-Large<br/>(segmentation)"]
        M["manifest.jsonl<br/>(per-frame join)"]
        TR["train.py<br/>EfficientViT-B1 student"]
        EX["export_trt.py<br/>ONNX → TensorRT"]
        T1 --> M
        T2 --> M
        T3 --> M
        M --> TR
        TR --> EX
    end

    subgraph Jetson["Jetson Orin Nano (NCHSB)"]
        STN["Student TRT Node<br/>RGB → depth + seg"]
        DFN["Depth Fusion Node<br/>ToF + student → fused"]
        PCN["PointCloud XYZ Node<br/>fused → 3D points"]
        N2["Nav2 local costmap"]
        STN --> DFN
        DFN --> PCN
        PCN --> N2
    end

    EX -.engine file.-> STN
    style HPC fill:#eef
    style Jetson fill:#efe
</div>

The dashed arrow is the only artifact that crosses the repo boundary: a `.engine` TensorRT file produced by ml_inference and consumed by NCHSB's Student TRT Node.

---

## 2. Student model architecture

EfficientViT-B1 encoder, dual decoders, shared neck. ImageNet normalization is baked into the forward pass so the runtime doesn't need to remember to apply it.

<div class="mermaid">
flowchart TB
    IN["Input RGB<br/>[B, 3, 240, 320]"]
    NORM["ImageNet normalization<br/>(in forward())"]
    ENC["EfficientViT-B1 encoder<br/>5.31M params"]

    F1["stage 1 features<br/>[B, 40, 120, 160]"]
    F2["stage 2 features<br/>[B, 80, 60, 80]"]
    F3["stage 3 features<br/>[B, 160, 30, 40]"]
    F4["stage 4 features<br/>[B, 256, 15, 20]"]

    NECK["Neck: Conv2d(256, 128, 1)"]

    subgraph DepthDec["Depth decoder"]
        DD1["DecoderBlock 128→64"]
        DD2["DecoderBlock 64→32"]
        DD3["DecoderBlock 32→16"]
        DH["Conv2d(16, 1) + ReLU"]
        DD1 --> DD2 --> DD3 --> DH
    end

    subgraph SegDec["Seg decoder"]
        SD1["DecoderBlock 128→64"]
        SD2["DecoderBlock 64→32"]
        SD3["DecoderBlock 32→16"]
        SH["Conv2d(16, 6)<br/>(floor, wall, person, furniture, glass, other)"]
        SD1 --> SD2 --> SD3 --> SH
    end

    DEPTH["depth output<br/>[B, 1, 240, 320] meters"]
    SEG["seg logits<br/>[B, 6, 240, 320]"]

    IN --> NORM --> ENC
    ENC --> F1
    ENC --> F2
    ENC --> F3
    ENC --> F4
    F4 --> NECK
    NECK --> DD1
    NECK --> SD1
    F3 -.skip.-> DD1
    F2 -.skip.-> DD2
    F1 -.skip.-> DD3
    F3 -.skip.-> SD1
    F2 -.skip.-> SD2
    F1 -.skip.-> SD3
    DH --> DEPTH
    SH --> SEG
</div>

`DecoderBlock(in, out)` = `Upsample(2×) → Conv2d(in+skip, out, 3) → BN → ReLU`. Skip connections come from the matching encoder stage.

---

## 3. Runtime depth fusion

What happens on the Jetson when a new RGB frame arrives:

<div class="mermaid">
flowchart LR
    RGB["RGB frame<br/>1280×720"]
    TOF["ToF depth<br/>1280×720<br/>(~22% valid pixels)"]
    CONF["ToF confidence<br/>1280×720"]

    DOWN["Downsample to<br/>240×320"]
    STUDENT["V9 student<br/>TensorRT FP16"]
    SDEPTH["student depth<br/>240×320"]
    SSEG["student seg<br/>240×320"]

    UP["Upsample student<br/>to 1280×720"]

    SCALE["s = median(d_tof / d_hat)<br/>over conf ≥ 0.5 pixels<br/>(per-frame scalar)"]
    SCALED["s · student_depth<br/>1280×720"]

    GATE{"per-pixel:<br/>conf ≥ 0.5 AND<br/>0.05 ≤ d_tof ≤ 10.0?"}
    FUSED["fused depth<br/>1280×720"]

    PCL["back-project to<br/>3D point cloud"]
    NAV["Nav2 local<br/>costmap"]

    RGB --> DOWN --> STUDENT
    STUDENT --> SDEPTH
    STUDENT --> SSEG
    SDEPTH --> UP
    UP --> SCALE
    TOF -.valid pixels.-> SCALE
    CONF -.gate.-> SCALE
    SCALE --> SCALED

    TOF --> GATE
    CONF --> GATE
    GATE -- "yes" --> FUSED
    SCALED -- "no, use scaled student" --> FUSED

    FUSED --> PCL --> NAV
    SSEG -.modulates inflation.-> NAV
</div>

Two operations per frame:

1. **Median-scale calibration** — `s = median(d_tof / d_hat)` over the surviving ToF pixels (those with confidence ≥ 0.5). One scalar per frame. This is the "bootstrap" step — see [Scale Calibration](concepts/scale-calibration).
2. **Confidence-gated substitution** — per pixel, use the raw ToF reading if it's confident and in range; otherwise fall back to `s · student_depth`. See [Confidence-Gated Fusion](concepts/confidence-gated-fusion).

The "modulates inflation" arrow is what lets glass walls inflate wider than concrete walls. The seg head produces a 6-class label per pixel; the Class Costmap Node turns those classes into per-class inflation radii (glass = 0.20 m, person = 0.30 m, wall = 0.12 m, etc.). This layer is implemented and available; activation in the default Nav2 observation source list is opt-in via configuration — see [Specification and Deployment Realization](concepts/specification-and-deployment).

---

## 4. Training loss

Single equation, one diagram. The hybrid depth target is the bridge between training and runtime — both encode the same prior.

<div class="mermaid">
flowchart TB
    DA3["DA3-Metric-Large<br/>per-frame depth"]
    TOF_GT["ToF depth + confidence<br/>(per-frame)"]
    SEG_GT["SAM2 + YOLO + heuristics<br/>6-class seg"]

    HYB{"has_da3 flag<br/>set on this frame?"}
    TARGET["hybrid depth target<br/>= DA3 if has_da3<br/>else = ToF"]

    PRED_D["student depth<br/>prediction"]
    PRED_S["student seg<br/>prediction"]

    LD["berHu(pred_d, target)"]
    LS["CrossEntropy(pred_s, seg_gt)"]
    LE["edge-smoothness(pred_d, RGB)"]

    KW["Kendall uncertainty<br/>weighting"]
    LOSS["total loss"]

    DA3 --> HYB
    TOF_GT --> HYB
    HYB --> TARGET
    TARGET --> LD
    PRED_D --> LD
    PRED_D --> LE
    SEG_GT --> LS
    PRED_S --> LS
    LD --> KW
    LS --> KW
    LE --> KW
    KW --> LOSS
</div>

The hybrid depth target lives in `models/losses.py:HybridDepthLoss`. Important nuance: the training target chooses *between* DA3 and ToF at the *frame* level (does this frame have a DA3 label?), while the deployment fusion chooses between ToF and student at the *pixel* level (does this pixel have valid ToF?). Both implement the same supervision principle — prefer hardware ground truth where available, fall back to the learned signal where not — at the granularity appropriate to each stage. See [decisions](decisions#1-specification-and-deployment-realization) for the full mapping.
