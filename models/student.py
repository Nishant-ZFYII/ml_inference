"""
Multi-task student model: MobileNetV3-Small encoder + dual decoder heads.

Produces both metric depth and 6-class semantic segmentation from a single
RGB image.  The encoder is shared; each head has its own decoder with skip
connections from the encoder feature pyramid.

Verified feature map shapes (input 320x240):
    Block  0:  16ch, 120x160  (1/2)
    Block  1:  16ch,  60x80   (1/4)
    Block  3:  24ch,  30x40   (1/8)
    Block  4:  40ch,  15x20   (1/16)
    Block 12: 576ch,   8x10   (bottleneck)

Decoder uses 3 transposed-conv upsample stages with skip connections
at blocks 4, 3, and 1, then a final bilinear 4x upsample to reach
full 240x320 resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Skip connection configuration (verified by print_model_shapes.py)
SKIP_INDICES = (1, 3, 4)   # block indices in model.features
SKIP_CHANNELS = (16, 24, 40)  # channels at each skip point
BOTTLENECK_CHANNELS = 576


class DecoderBlock(nn.Module):
    """Transpose-conv upsample + skip connection fusion."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                     stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.up(x)), inplace=True)
        # Handle spatial size mismatch from non-power-of-2 dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class MultiTaskStudent(nn.Module):
    """
    MobileNetV3-Small encoder with shared backbone and two decoder heads:
        - Depth head: predicts single-channel metric depth (meters)
        - Segmentation head: predicts C-channel logits

    Args:
        num_classes: number of segmentation classes (default 6)
        pretrained: use ImageNet-pretrained encoder weights
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder ────────────────────────────────────────────────────
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = backbone.features  # nn.Sequential of 13 blocks (0-12)

        # ── Neck: reduce bottleneck channels ───────────────────────────
        neck_ch = 128
        self.neck = nn.Sequential(
            nn.Conv2d(BOTTLENECK_CHANNELS, neck_ch, 1),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(inplace=True),
        )

        # ── Depth decoder ──────────────────────────────────────────────
        self.depth_d1 = DecoderBlock(neck_ch, SKIP_CHANNELS[2], 64)   # +block4 skip
        self.depth_d2 = DecoderBlock(64, SKIP_CHANNELS[1], 32)        # +block3 skip
        self.depth_d3 = DecoderBlock(32, SKIP_CHANNELS[0], 16)        # +block1 skip
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # depth is non-negative
        )

        # ── Segmentation decoder ───────────────────────────────────────
        self.seg_d1 = DecoderBlock(neck_ch, SKIP_CHANNELS[2], 64)
        self.seg_d2 = DecoderBlock(64, SKIP_CHANNELS[1], 32)
        self.seg_d3 = DecoderBlock(32, SKIP_CHANNELS[0], 16)
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: RGB tensor [B, 3, H, W] normalized to [0, 1]

        Returns:
            depth: [B, 1, H, W] metric depth in meters
            seg:   [B, C, H, W] segmentation logits
        """
        input_size = x.shape[2:]  # (H, W)

        # ── Encoder forward with skip extraction ───────────────────────
        skips = {}
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in SKIP_INDICES:
                skips[i] = x

        # x is now the bottleneck: [B, 576, H/30, W/32]
        feat = self.neck(x)

        # ── Depth decoder ──────────────────────────────────────────────
        d = self.depth_d1(feat, skips[SKIP_INDICES[2]])  # skip from block 4
        d = self.depth_d2(d, skips[SKIP_INDICES[1]])     # skip from block 3
        d = self.depth_d3(d, skips[SKIP_INDICES[0]])     # skip from block 1
        d = F.interpolate(d, size=input_size, mode="bilinear",
                          align_corners=False)
        depth = self.depth_head(d)

        # ── Segmentation decoder ───────────────────────────────────────
        s = self.seg_d1(feat, skips[SKIP_INDICES[2]])
        s = self.seg_d2(s, skips[SKIP_INDICES[1]])
        s = self.seg_d3(s, skips[SKIP_INDICES[0]])
        s = F.interpolate(s, size=input_size, mode="bilinear",
                          align_corners=False)
        seg = self.seg_head(s)

        return depth, seg


def build_student(num_classes: int = 6, pretrained: bool = True) -> MultiTaskStudent:
    """Factory function to build the student model."""
    return MultiTaskStudent(num_classes=num_classes, pretrained=pretrained)
