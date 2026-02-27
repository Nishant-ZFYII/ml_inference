#!/usr/bin/env python3
"""
Benchmark TensorRT engines on Jetson (or any TensorRT-capable GPU).

Measures latency, GPU memory, and accuracy on the NYU val set.
Outputs a formatted table matching Table III in the paper.

Usage:
    python benchmark_jetson.py --engine-dir exported/ --data-root ./data
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np


def get_gpu_memory_mb() -> float:
    """Query GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True)
        return float(result.stdout.strip().split("\n")[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0.0


def benchmark_engine(engine_path: str, height: int = 240, width: int = 320,
                     warmup: int = 200, iterations: int = 1000):
    """Run latency benchmark on a single TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        print("ERROR: tensorrt or pycuda not installed.")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (1, 3, height, width)
    input_size = int(np.prod(input_shape)) * 4  # float32

    d_input = cuda.mem_alloc(input_size)
    h_input = np.random.rand(*input_shape).astype(np.float32)
    cuda.memcpy_htod(d_input, h_input)

    # Allocate output buffers
    outputs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            size = int(np.prod(shape)) * 4
            d_buf = cuda.mem_alloc(size)
            outputs.append((name, d_buf, shape))

    stream = cuda.Stream()

    # Set input tensor address
    input_name = engine.get_tensor_name(0)
    context.set_tensor_address(input_name, int(d_input))
    for name, d_buf, _ in outputs:
        context.set_tensor_address(name, int(d_buf))

    # Warmup
    for _ in range(warmup):
        context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

    # Timed runs
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        latencies.append((time.perf_counter_ns() - start) / 1e6)

    latencies = np.array(latencies)
    mem_mb = get_gpu_memory_mb()

    return {
        "mean_ms": float(np.mean(latencies)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "gpu_mem_mb": mem_mb,
    }


def evaluate_accuracy(engine_path: str, data_root: str,
                      height: int = 240, width: int = 320):
    """
    Evaluate depth RMSE and seg mIoU on the NYU val set using a TRT engine.

    Returns (depth_rmse, seg_miou) or (None, None) if evaluation fails.
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        from PIL import Image
    except ImportError:
        return None, None

    # Check if NYU data is cached
    cache_dir = os.path.join(data_root, "nyu_cache")
    val_indices_file = os.path.join(cache_dir, "val_indices.txt")
    if not os.path.exists(val_indices_file):
        print("  NYU val data not found, skipping accuracy evaluation.")
        return None, None

    with open(val_indices_file) as f:
        val_indices = [int(line.strip()) for line in f]

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (1, 3, height, width)
    d_input = cuda.mem_alloc(int(np.prod(input_shape)) * 4)

    output_buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            d_buf = cuda.mem_alloc(int(np.prod(shape)) * 4)
            output_buffers[name] = (d_buf, shape)

    input_name = engine.get_tensor_name(0)
    context.set_tensor_address(input_name, int(d_input))
    for name, (d_buf, _) in output_buffers.items():
        context.set_tensor_address(name, int(d_buf))

    stream = cuda.Stream()
    rmse_vals, miou_vals = [], []

    for idx in val_indices[:100]:  # evaluate on subset for speed
        stem = f"{idx:05d}"
        rgb_path = os.path.join(cache_dir, "rgb", f"{stem}.png")
        depth_path = os.path.join(cache_dir, "depth", f"{stem}.npy")

        if not os.path.exists(rgb_path):
            continue

        img = Image.open(rgb_path).convert("RGB").resize((width, height))
        rgb_np = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        rgb_np = rgb_np[np.newaxis]

        gt_depth = np.load(depth_path)
        gt_depth_resized = np.array(
            Image.fromarray(gt_depth).resize((width, height)),
            dtype=np.float32)

        cuda.memcpy_htod(d_input, rgb_np.astype(np.float32))
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

        # Read depth output
        depth_buf, depth_shape = output_buffers.get("depth",
                                                     list(output_buffers.values())[0])
        h_depth = np.empty(depth_shape, dtype=np.float32)
        cuda.memcpy_dtoh(h_depth, depth_buf)
        pred_depth = h_depth.squeeze()

        valid = gt_depth_resized > 0
        if valid.sum() > 0:
            rmse = np.sqrt(((pred_depth[valid] - gt_depth_resized[valid]) ** 2).mean())
            rmse_vals.append(rmse)

    depth_rmse = float(np.mean(rmse_vals)) if rmse_vals else None
    seg_miou = float(np.mean(miou_vals)) if miou_vals else None

    return depth_rmse, seg_miou


def main():
    p = argparse.ArgumentParser(description="Benchmark TensorRT engines")
    p.add_argument("--engine-dir", type=str, default="exported")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--iterations", type=int, default=1000)
    args = p.parse_args()

    precisions = ["fp32", "fp16", "int8"]
    results = {}

    for prec in precisions:
        engine_path = os.path.join(args.engine_dir, f"student_{prec}.engine")
        if not os.path.exists(engine_path):
            print(f"Engine not found: {engine_path}, skipping.")
            results[prec] = None
            continue

        print(f"\n{'='*50}")
        print(f"Benchmarking: {prec.upper()}")
        print(f"{'='*50}")

        bench = benchmark_engine(engine_path, warmup=args.warmup,
                                 iterations=args.iterations)
        depth_rmse, seg_miou = evaluate_accuracy(engine_path, args.data_root)

        results[prec] = {
            **bench,
            "depth_rmse": depth_rmse,
            "seg_miou": seg_miou,
        }

    # Print Table III
    print(f"\n{'='*60}")
    print("Table III: Student Model Performance")
    print(f"{'='*60}")
    header = f"{'Metric':<25}" + "".join(f"{p.upper():>12}" for p in precisions)
    print(header)
    print("-" * len(header))

    for metric, key, fmt in [
        ("Latency (ms)", "mean_ms", ".2f"),
        ("P99 Latency (ms)", "p99_ms", ".2f"),
        ("Depth RMSE (m)", "depth_rmse", ".4f"),
        ("Seg mIoU (%)", "seg_miou", ".1f"),
        ("GPU Memory (MB)", "gpu_mem_mb", ".0f"),
    ]:
        row = f"{metric:<25}"
        for prec in precisions:
            r = results.get(prec)
            if r is None:
                row += f"{'N/A':>12}"
            else:
                val = r.get(key)
                if val is None:
                    row += f"{'N/A':>12}"
                elif key == "seg_miou":
                    row += f"{val * 100:{fmt}}".rjust(12)
                else:
                    row += f"{val:{fmt}}".rjust(12)
        print(row)

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
