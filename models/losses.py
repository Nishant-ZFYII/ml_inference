"""
Training losses for the multi-task student model.

Key design: the hybrid depth supervision target mirrors the runtime
fusion equation in depth_fusion.py.

    Runtime:  d_fused = ToF  if confidence >= tau  else  d_student
    Training: target  = ToF  if confidence >= tau  else  DA3

This means a reviewer sees the training loss and runtime fusion and
recognises they encode the same prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _image_gradients(x: torch.Tensor):
    """Compute spatial gradients (Sobel-like finite differences)."""
    grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
    grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
    return grad_x, grad_y


class HybridDepthLoss(nn.Module):
    """
    L1 depth loss with configurable supervision target.

    Two modes controlled by `distill_depth`:

    distill_depth=True  (pure distillation, use for NYU stand-in):
        target = DA3 depth everywhere DA3 is available.
        Falls back to GT/ToF only when DA3 is missing.

    distill_depth=False (hybrid, use for corridor data with real ToF):
        Where ToF confidence >= tau  →  supervise to ToF (real sensor)
        Where ToF confidence <  tau  →  supervise to DA3 (neural depth)
    """

    def __init__(self, confidence_threshold: float = 0.5,
                 distill_depth: bool = True):
        super().__init__()
        self.tau = confidence_threshold
        self.distill_depth = distill_depth

    def forward(
        self,
        pred_depth: torch.Tensor,    # [B, 1, H, W]
        tof_depth: torch.Tensor,     # [B, 1, H, W]  ground-truth / ToF
        confidence: torch.Tensor,    # [B, 1, H, W]  0-1
        da3_depth: torch.Tensor | None = None,  # [B, 1, H, W]
        has_da3: bool = False,
    ) -> torch.Tensor:

        if has_da3 and da3_depth is not None:
            if self.distill_depth:
                target = da3_depth
            else:
                target = torch.where(confidence >= self.tau, tof_depth, da3_depth)
        else:
            target = tof_depth

        valid = target > 0
        if valid.sum() == 0:
            return pred_depth.sum() * 0.0

        return F.l1_loss(pred_depth[valid], target[valid])


class EdgeAwareSmoothness(nn.Module):
    """
    Edge-aware depth smoothness loss.

    Penalises depth gradients except where RGB gradients are strong
    (i.e., at object boundaries).

        L_edge = mean( |∇d| · exp(-|∇I|) )
    """

    def forward(
        self,
        pred_depth: torch.Tensor,  # [B, 1, H, W]
        rgb: torch.Tensor,         # [B, 3, H, W]
    ) -> torch.Tensor:
        depth_dx, depth_dy = _image_gradients(pred_depth)
        rgb_dx, rgb_dy = _image_gradients(rgb)

        # Mean absolute RGB gradient across channels
        rgb_dx = rgb_dx.abs().mean(dim=1, keepdim=True)
        rgb_dy = rgb_dy.abs().mean(dim=1, keepdim=True)

        smooth_x = depth_dx.abs() * torch.exp(-rgb_dx)
        smooth_y = depth_dy.abs() * torch.exp(-rgb_dy)

        return smooth_x.mean() + smooth_y.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss:
        L = λ_d · L_depth + λ_s · L_seg + λ_e · L_edge
    """

    def __init__(
        self,
        lambda_depth: float = 1.0,
        lambda_seg: float = 0.5,
        lambda_edge: float = 0.1,
        confidence_threshold: float = 0.5,
        num_classes: int = 6,
        distill_depth: bool = True,
    ):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_seg = lambda_seg
        self.lambda_edge = lambda_edge

        self.depth_loss = HybridDepthLoss(confidence_threshold, distill_depth)
        self.edge_loss = EdgeAwareSmoothness()
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(
        self,
        pred_depth: torch.Tensor,   # [B, 1, H, W]
        pred_seg: torch.Tensor,     # [B, C, H, W]
        rgb: torch.Tensor,          # [B, 3, H, W]
        gt_depth: torch.Tensor,     # [B, 1, H, W]
        gt_seg: torch.Tensor,       # [B, H, W] int64
        confidence: torch.Tensor,   # [B, 1, H, W]
        da3_depth: torch.Tensor | None = None,
        has_da3: bool = False,
    ) -> dict:
        """
        Returns dict with 'total', 'depth', 'seg', 'edge' loss values.
        """
        l_depth = self.depth_loss(pred_depth, gt_depth, confidence,
                                  da3_depth, has_da3)
        l_seg = self.seg_loss(pred_seg, gt_seg)
        l_edge = self.edge_loss(pred_depth, rgb)

        total = (self.lambda_depth * l_depth
                 + self.lambda_seg * l_seg
                 + self.lambda_edge * l_edge)

        return {
            "total": total,
            "depth": l_depth.detach(),
            "seg": l_seg.detach(),
            "edge": l_edge.detach(),
        }
