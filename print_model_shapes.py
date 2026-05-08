#!/usr/bin/env python3
"""
Inspect EfficientViT-B1 feature map shapes at every stage.

Run this BEFORE building the decoder to know the exact channel counts
and spatial resolutions at each skip-connection point.

Usage:
    python print_model_shapes.py
"""

import torch
import timm


def main():
    model = timm.create_model(
        "efficientvit_b1.r288_in1k", pretrained=True, features_only=True
    )
    model.eval()

    x = torch.randn(1, 3, 240, 320)

    print("=" * 70)
    print("EfficientViT-B1 Feature Map Shapes (via timm, features_only=True)")
    print(f"Input: {list(x.shape)}  (batch, C, H, W)")
    print("=" * 70)

    with torch.no_grad():
        features = model(x)

    h_in, w_in = 240, 320
    for i, feat in enumerate(features):
        h, w = feat.shape[2], feat.shape[3]
        scale = h_in / h
        print(
            f"  Stage {i}: {str(list(feat.shape)):30s}  "
            f"channels={feat.shape[1]:4d}  "
            f"spatial={h}x{w}  "
            f"scale=1/{scale:.0f}"
        )

    print("=" * 70)
    print("\nFor student.py decoder:")
    print(f"  feature_info: {model.feature_info.channels()}")
    print(f"  reductions:   {model.feature_info.reduction()}")
    print()
    print("Use the last 3 stages as skip connections and the final stage as bottleneck,")
    print("or pick stages at ~1/2, 1/4, 1/8 scale for skip + deepest as bottleneck.")


if __name__ == "__main__":
    main()
