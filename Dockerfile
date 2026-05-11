# Dockerfile — ml_inference evaluation and inference pipeline
# Runs student model inference, depth evaluation, video generation,
# and calibration experiments. No CUDA required for basic eval;
# GPU passthrough optional for faster inference.
#
# Build:   docker build -t ml-inference .
# Run:     docker run --rm -v $(pwd)/output:/app/output ml-inference
# GPU:     docker run --rm --gpus all -v $(pwd)/output:/app/output ml-inference

# Pinned to Bookworm (Debian 12) — Trixie renamed libgl1-mesa-glx → libgl1
# and breaks the apt-get install line below.
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# opencv-python-headless does not need libGL; ffmpeg is for video encoding.
# libglib2.0-0 is needed by some torch / opencv transitive deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir opencv-python-headless

COPY config.py .
COPY models/ models/
COPY dataset/ dataset/
COPY train.py .
COPY eval_corridor_depth.py eval_corridor_da3.py eval_distillation.py .
COPY eval_nearrange_safety.py fpr_audit.py temporal_consistency.py .
COPY calibration_sensitivity.py .
COPY run_costmap_ablation.py analyze_ablation.py costmap_builder.py inflation.py .
COPY compute_paper_stats.py .
COPY generate_demo_videos.py generate_grid_video.py generate_corridor_missing.py .
COPY create_full_comparison.py create_paper_fig2.py .
COPY extract_bag_frames.py extract_corridor_frames.py .

RUN mkdir -p /app/output /app/hpc_outputs /app/corridor_eval_data /app/results

COPY results/ results/

VOLUME ["/app/hpc_outputs", "/app/corridor_eval_data", "/app/output"]

ENTRYPOINT ["python"]
CMD ["--help"]
