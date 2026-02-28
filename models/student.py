"""
Multi-task student model: EfficientViT-B1 encoder + dual decoder heads.

Produces both metric depth and 6-class semantic segmentation from a single
RGB image.  The encoder is shared; each head has its own decoder with skip
connections from the encoder feature pyramid.

Verified feature map shapes (input 320x240, via timm features_only=True):
    Stage 0:  32ch, 60x80   (1/4)   ← skip 0
    Stage 1:  64ch, 30x40   (1/8)   ← skip 1
    Stage 2: 128ch, 15x20   (1/16)  ← skip 2
    Stage 3: 256ch,  8x10   (1/32)  ← bottleneck

Decoder uses 3 transposed-conv upsample stages with skip connections
at stages 2, 1, and 0, then a final bilinear upsample to reach
full 240x320 resolution.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class MultiTaskStudent(nn.Module):
    """
    EfficientViT-B1 encoder with shared backbone and two decoder heads:
        - Depth head: predicts single-channel metric depth (meters)
        - Segmentation head: predicts C-channel logits

    Args:
        num_classes: number of segmentation classes (default 6)
        pretrained: use ImageNet-pretrained encoder weights
        backbone_name: timm model name for the encoder
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True,
                 backbone_name: str = "efficientvit_b1.r288_in1k"):
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder ────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        channels = self.encoder.feature_info.channels()  # [32, 64, 128, 256]

        skip_chs = channels[:-1]   # [32, 64, 128]
        bottleneck_ch = channels[-1]  # 256

        # ── Neck: reduce bottleneck channels ───────────────────────────
        neck_ch = 128
        self.neck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, neck_ch, 1),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(inplace=True),
        )

        # ── Depth decoder ──────────────────────────────────────────────
        self.depth_d1 = DecoderBlock(neck_ch, skip_chs[2], 64)    # +stage2 skip (128ch)
        self.depth_d2 = DecoderBlock(64, skip_chs[1], 32)         # +stage1 skip (64ch)
        self.depth_d3 = DecoderBlock(32, skip_chs[0], 16)         # +stage0 skip (32ch)
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # ── Segmentation decoder ───────────────────────────────────────
        self.seg_d1 = DecoderBlock(neck_ch, skip_chs[2], 64)
        self.seg_d2 = DecoderBlock(64, skip_chs[1], 32)
        self.seg_d3 = DecoderBlock(32, skip_chs[0], 16)
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: RGB tensor [B, 3, H, W] normalized to [0, 1]

        Returns:
            depth: [B, 1, H, W] metric depth in meters
            seg:   [B, C, H, W] segmentation logits
        """
        input_size = x.shape[2:]

        # ── Encoder forward ───────────────────────────────────────────
        features = self.encoder(x)  # list of 4 feature maps
        skips = features[:-1]       # stages 0, 1, 2
        bottleneck = features[-1]   # stage 3

        feat = self.neck(bottleneck)

        # ── Depth decoder ──────────────────────────────────────────────
        d = self.depth_d1(feat, skips[2])    # skip from stage 2
        d = self.depth_d2(d, skips[1])       # skip from stage 1
        d = self.depth_d3(d, skips[0])       # skip from stage 0
        d = F.interpolate(d, size=input_size, mode="bilinear",
                          align_corners=False)
        depth = self.depth_head(d)

        # ── Segmentation decoder ───────────────────────────────────────
        s = self.seg_d1(feat, skips[2])
        s = self.seg_d2(s, skips[1])
        s = self.seg_d3(s, skips[0])
        s = F.interpolate(s, size=input_size, mode="bilinear",
                          align_corners=False)
        seg = self.seg_head(s)

        return depth, seg


def build_student(num_classes: int = 6, pretrained: bool = True,
                   backbone: str = "efficientvit_b1.r288_in1k") -> MultiTaskStudent:
    """Factory function to build the student model."""
    return MultiTaskStudent(num_classes=num_classes, pretrained=pretrained,
                            backbone_name=backbone)
