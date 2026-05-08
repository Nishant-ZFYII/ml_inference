#!/usr/bin/env python3
"""
Export trained student model to ONNX and optionally compile TensorRT engines.

Usage (laptop -- ONNX only):
    python export_trt.py --checkpoint checkpoints/best.pt --skip-trt

Usage (Jetson -- full pipeline):
    python export_trt.py --checkpoint checkpoints/best.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from models.student import build_student
from config import Config


def export_onnx(model, output_path: str, height: int = 240, width: int = 320):
    """Export model to ONNX with dynamic batch size."""
    model.eval()
    dummy = torch.randn(1, 3, height, width)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=17,
        input_names=["rgb"],
        output_names=["depth", "segmentation"],
        dynamic_axes={
            "rgb": {0: "batch"},
            "depth": {0: "batch"},
            "segmentation": {0: "batch"},
        },
    )
    print(f"ONNX exported: {output_path}")

    # Validate
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully.")


def build_trt_engine(onnx_path: str, engine_path: str, precision: str,
                     calib_data=None, height: int = 240, width: int = 320):
    """Compile TensorRT engine from ONNX model."""
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt not installed. Install JetPack on Jetson "
              "or `pip install tensorrt` on x86.")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Set precision
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("WARNING: FP16 not natively supported on this platform.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("WARNING: INT8 not natively supported on this platform.")
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_data is not None:
            config.int8_calibrator = Int8Calibrator(calib_data, height, width)

    # Dynamic batch: min=1, opt=1, max=8
    profile = builder.create_optimization_profile()
    profile.set_shape("rgb",
                      min=(1, 3, height, width),
                      opt=(1, 3, height, width),
                      max=(8, 3, height, width))
    config.add_optimization_profile(profile)

    print(f"Building TensorRT {precision.upper()} engine...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: TensorRT engine build failed.")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"TensorRT engine saved: {engine_path} "
          f"({os.path.getsize(engine_path) / 1e6:.1f} MB)")


class Int8Calibrator:
    """Entropy calibrator for TensorRT INT8 quantization."""

    def __init__(self, images: np.ndarray, height: int, width: int):
        try:
            import tensorrt as trt
            self._base = trt.IInt8EntropyCalibrator2
        except ImportError:
            return

        self.images = images
        self.batch_size = 1
        self.current_idx = 0
        self.height = height
        self.width = width

        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.device_input = cuda.mem_alloc(
            self.batch_size * 3 * height * width * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.images):
            return None
        import pycuda.driver as cuda
        batch = self.images[self.current_idx:self.current_idx + 1]
        batch = batch.astype(np.float32)
        cuda.memcpy_htod(self.device_input, batch)
        self.current_idx += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


def main():
    p = argparse.ArgumentParser(description="Export student model")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--output-dir", type=str, default="exported")
    p.add_argument("--skip-trt", action="store_true",
                   help="Only export ONNX, skip TensorRT compilation")
    p.add_argument("--calib-images", type=int, default=500,
                   help="Number of calibration images for INT8")
    args = p.parse_args()

    cfg = Config()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Export ONNX
    onnx_path = os.path.join(args.output_dir, "student.onnx")
    export_onnx(model, onnx_path, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH)

    if args.skip_trt:
        print("\nSkipping TensorRT compilation (--skip-trt).")
        print("To compile on Jetson, run without --skip-trt.")
        return

    # Prepare calibration data for INT8
    calib_data = None
    if args.calib_images > 0:
        print(f"Generating {args.calib_images} calibration images...")
        calib_data = np.random.rand(
            args.calib_images, 3, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH
        ).astype(np.float32)

    # Build engines
    for precision in ["fp32", "fp16", "int8"]:
        engine_path = os.path.join(args.output_dir,
                                   f"student_{precision}.engine")
        cd = calib_data if precision == "int8" else None
        build_trt_engine(onnx_path, engine_path, precision, cd,
                         cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH)

    print("\nAll engines exported successfully.")


if __name__ == "__main__":
    main()
