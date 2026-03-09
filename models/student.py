"""
Multi-task student model with configurable backbone + dual decoder heads.

Ported from Vivek's V4 architecture (0.774m RMSE on NYU).

Supported backbones:
  - efficientvit_b1 (EfficientViT-B1, ~5.3M total params)
  - efficientvit_b2 (EfficientViT-B2, ~15.8M total params)
  - mobilenet_v3_small (MobileNetV3-Small, ~1.5M total params, legacy)

Feature map shapes for 320x240 input (EfficientViT-B1):
  Stage 0:  32ch, 60x80   (1/4)   <- skip 0
  Stage 1:  64ch, 30x40   (1/8)   <- skip 1
  Stage 2: 128ch, 15x20   (1/16)  <- skip 2
  Stage 3: 256ch,  8x10   (1/32)  <- bottleneck

Decoder uses 3 transposed-conv upsample stages with skip connections,
then a final bilinear 4x upsample to reach full 240x320 resolution.

Key V4 change: ImageNet normalization happens inside forward(), ensuring
consistency between training and inference regardless of preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

_MBN_SKIP_INDICES = (1, 3, 4)
_MBN_SKIP_CHANNELS = (16, 24, 40)
_MBN_BOTTLENECK_CHANNELS = 576


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
    Shared-encoder, dual-decoder multi-task model.

    Args:
        backbone: 'efficientvit_b1', 'efficientvit_b2', or 'mobilenet_v3_small'
        num_classes: number of segmentation classes (default 6)
        pretrained: use ImageNet-pretrained encoder weights
    """

    def __init__(self, backbone: str = "efficientvit_b1",
                 num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if backbone.startswith("efficientvit"):
            self._build_efficientvit_encoder(backbone, pretrained)
        elif backbone == "mobilenet_v3_small":
            self._build_mobilenet_encoder(pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. "
                             f"Use 'efficientvit_b1', 'efficientvit_b2', "
                             f"or 'mobilenet_v3_small'")

        neck_ch = 128
        self.neck = nn.Sequential(
            nn.Conv2d(self._bottleneck_ch, neck_ch, 1),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(inplace=True),
        )

        s = self._skip_channels
        self.depth_d1 = DecoderBlock(neck_ch, s[2], 64)
        self.depth_d2 = DecoderBlock(64, s[1], 32)
        self.depth_d3 = DecoderBlock(32, s[0], 16)
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.seg_d1 = DecoderBlock(neck_ch, s[2], 64)
        self.seg_d2 = DecoderBlock(64, s[1], 32)
        self.seg_d3 = DecoderBlock(32, s[0], 16)
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def _build_efficientvit_encoder(self, backbone: str, pretrained: bool):
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for EfficientViT. "
                              "Install with: pip install timm")

        timm_name = backbone
        if backbone == "efficientvit_b1":
            timm_name = "efficientvit_b1.r224_in1k"
        elif backbone == "efficientvit_b2":
            timm_name = "efficientvit_b2.r224_in1k"

        self.encoder = timm.create_model(
            timm_name, pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )

        channels = self.encoder.feature_info.channels()
        self._skip_channels = channels[:3]
        self._bottleneck_ch = channels[3]
        self._use_timm = True

    def _build_mobilenet_encoder(self, pretrained: bool):
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = backbone.features
        self._skip_channels = list(_MBN_SKIP_CHANNELS)
        self._bottleneck_ch = _MBN_BOTTLENECK_CHANNELS
        self._use_timm = False

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: RGB tensor [B, 3, H, W] normalized to [0, 1]
        Returns:
            depth: [B, 1, H, W] metric depth in meters
            seg:   [B, C, H, W] segmentation logits
        """
        input_size = x.shape[2:]

        x = (x - self.img_mean) / self.img_std

        if self._use_timm:
            feats = self.encoder(x)
            skip1, skip2, skip3 = feats[0], feats[1], feats[2]
            bottleneck = feats[3]
        else:
            skips = {}
            for i, block in enumerate(self.encoder):
                x = block(x)
                if i in _MBN_SKIP_INDICES:
                    skips[i] = x
            bottleneck = x
            skip1 = skips[_MBN_SKIP_INDICES[0]]
            skip2 = skips[_MBN_SKIP_INDICES[1]]
            skip3 = skips[_MBN_SKIP_INDICES[2]]

        feat = self.neck(bottleneck)

        d = self.depth_d1(feat, skip3)
        d = self.depth_d2(d, skip2)
        d = self.depth_d3(d, skip1)
        d = F.interpolate(d, size=input_size, mode="bilinear",
                          align_corners=False)
        depth = self.depth_head(d)

        s = self.seg_d1(feat, skip3)
        s = self.seg_d2(s, skip2)
        s = self.seg_d3(s, skip1)
        s = F.interpolate(s, size=input_size, mode="bilinear",
                          align_corners=False)
        seg = self.seg_head(s)

        return depth, seg


def build_student(backbone: str = "efficientvit_b1",
                  num_classes: int = 6,
                  pretrained: bool = True) -> MultiTaskStudent:
    """Factory function to build the student model."""
    return MultiTaskStudent(backbone=backbone, num_classes=num_classes,
                            pretrained=pretrained)
