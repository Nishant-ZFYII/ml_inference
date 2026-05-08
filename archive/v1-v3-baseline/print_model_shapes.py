#!/usr/bin/env python3
"""
Inspect MobileNetV3-Small feature map shapes at every block.

Run this BEFORE building the decoder to know the exact channel counts
and spatial resolutions at each skip-connection point.

Usage:
    python print_model_shapes.py
"""

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def main():
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.eval()

    # Our target input: 320x240 (W x H) -> tensor shape [1, 3, 240, 320]
    x = torch.randn(1, 3, 240, 320)

    print("=" * 70)
    print("MobileNetV3-Small Feature Map Shapes")
    print(f"Input: {list(x.shape)}  (batch, C, H, W)")
    print("=" * 70)

    features = model.features

    for i, block in enumerate(features):
        x = block(x)
        print(f"  Block {i:2d}: {str(list(x.shape)):30s}  "
              f"channels={x.shape[1]:4d}  "
              f"spatial={x.shape[2]}x{x.shape[3]}")

    print("=" * 70)
    print("\nRecommended skip-connection points for decoder:")
    print("  (Pick blocks at ~1/2, 1/4, and 1/8 of input resolution)")
    print()

    # Re-run to find good skip points
    x = torch.randn(1, 3, 240, 320)
    h_in, w_in = 240, 320
    skip_candidates = []

    for i, block in enumerate(features):
        x = block(x)
        h, w = x.shape[2], x.shape[3]
        scale = h_in / h
        skip_candidates.append((i, x.shape[1], h, w, scale))

    for idx, ch, h, w, scale in skip_candidates:
        marker = ""
        if abs(scale - 2) < 0.5:
            marker = " <-- skip 1 (1/2 res)"
        elif abs(scale - 4) < 0.5:
            marker = " <-- skip 2 (1/4 res)"
        elif abs(scale - 8) < 1.0:
            marker = " <-- skip 3 (1/8 res, bottleneck)"
        if marker:
            print(f"  Block {idx:2d}: {ch:3d} channels, {h}x{w}{marker}")


if __name__ == "__main__":
    main()
