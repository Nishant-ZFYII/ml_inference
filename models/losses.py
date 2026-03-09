"""
Training losses for the multi-task student model.

Ported from Vivek's V4 recipe:
  - berHu (reverse Huber) depth loss replaces L1
  - Kendall uncertainty weighting with log_sigma clamped to [-2, 2]

The pipeline is teacher-only: student trains against DA3 depth and
YOLO+SAM2 segmentation labels. No GT is used in the loss.
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
    berHu depth loss with hybrid supervision target.

    berHu (Laina et al. 2016):
        L = |e|              if |e| <= c
        L = (e^2 + c^2)/2c   if |e| > c
        where c = 0.2 * max(|e|) per batch

    When DA3 teacher depth is available, supervises against DA3.
    Falls back to GT/ToF when DA3 labels are not available.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__()
        self.tau = confidence_threshold

    def forward(
        self,
        pred_depth: torch.Tensor,
        tof_depth: torch.Tensor,
        confidence: torch.Tensor,
        da3_depth: torch.Tensor | None = None,
        has_da3: bool = False,
    ) -> torch.Tensor:

        if has_da3 and da3_depth is not None:
            target = da3_depth
        else:
            target = torch.where(confidence >= self.tau, tof_depth, tof_depth)

        valid = target > 0
        if valid.sum() == 0:
            return pred_depth.sum() * 0.0

        p = pred_depth[valid]
        t = target[valid]
        e = (p - t).abs()

        c = 0.2 * e.max().detach()

        berhu = torch.where(e <= c, e, (e * e + c * c) / (2.0 * c + 1e-8))
        return berhu.mean()


class EdgeAwareSmoothness(nn.Module):
    """
    Edge-aware depth smoothness loss.
        L_edge = mean( |nabla d| * exp(-|nabla I|) )
    """

    def forward(self, pred_depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        depth_dx, depth_dy = _image_gradients(pred_depth)
        rgb_dx, rgb_dy = _image_gradients(rgb)

        rgb_dx = rgb_dx.abs().mean(dim=1, keepdim=True)
        rgb_dy = rgb_dy.abs().mean(dim=1, keepdim=True)

        smooth_x = depth_dx.abs() * torch.exp(-rgb_dx)
        smooth_y = depth_dy.abs() * torch.exp(-rgb_dy)

        return smooth_x.mean() + smooth_y.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with Kendall uncertainty weighting.

    use_kendall=True (default, V4 recipe):
        Learns per-task log sigma^2 clamped to [-2, 2].
        depth (regression):       0.5 * exp(-log_s) * L_d + 0.5 * log_s
        seg (classification):     exp(-log_s) * L_s + 0.5 * log_s
        edge (regulariser):       lambda_e * L_edge

    use_kendall=False:
        Fixed weights: L = lambda_d * L_d + lambda_s * L_s + lambda_e * L_edge
    """

    def __init__(
        self,
        lambda_depth: float = 1.0,
        lambda_seg: float = 0.5,
        lambda_edge: float = 0.1,
        confidence_threshold: float = 0.5,
        num_classes: int = 6,
        use_kendall: bool = True,
    ):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_seg = lambda_seg
        self.lambda_edge = lambda_edge
        self.use_kendall = use_kendall

        self.depth_loss = HybridDepthLoss(confidence_threshold)
        self.edge_loss = EdgeAwareSmoothness()
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)

        if use_kendall:
            self.log_sigma_d = nn.Parameter(torch.zeros(1))
            self.log_sigma_s = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred_depth: torch.Tensor,
        pred_seg: torch.Tensor,
        rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        gt_seg: torch.Tensor,
        confidence: torch.Tensor,
        da3_depth: torch.Tensor | None = None,
        has_da3: bool = False,
    ) -> dict:
        l_depth = self.depth_loss(pred_depth, gt_depth, confidence,
                                  da3_depth, has_da3)
        l_seg = self.seg_loss(pred_seg, gt_seg)
        l_edge = self.edge_loss(pred_depth, rgb)

        if self.use_kendall:
            log_d = self.log_sigma_d.clamp(-2, 2)
            log_s = self.log_sigma_s.clamp(-2, 2)
            depth_term = 0.5 * torch.exp(-log_d) * l_depth + 0.5 * log_d
            seg_term = torch.exp(-log_s) * l_seg + 0.5 * log_s
            total = depth_term + seg_term + self.lambda_edge * l_edge

            return {
                "total": total,
                "depth": l_depth.detach(),
                "seg": l_seg.detach(),
                "edge": l_edge.detach(),
                "log_sigma_d": log_d.detach(),
                "log_sigma_s": log_s.detach(),
            }
        else:
            total = (self.lambda_depth * l_depth
                     + self.lambda_seg * l_seg
                     + self.lambda_edge * l_edge)

            return {
                "total": total,
                "depth": l_depth.detach(),
                "seg": l_seg.detach(),
                "edge": l_edge.detach(),
            }
