#!/usr/bin/env python3
"""
Training script for the multi-task student model.

Usage (local CPU validation):
    python train.py --epochs 2 --batch-size 4 --device cpu --data-limit 50

Usage (HPC with SLURM):
    See train.slurm
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset.nyu_loader import get_dataloaders
from models.student import build_student
from models.losses import MultiTaskLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-task student model")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data-limit", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def apply_args(cfg: Config, args):
    """Override config with CLI arguments where provided."""
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.LR = args.lr
    if args.data_limit is not None:
        cfg.DATA_LIMIT = args.data_limit
    if args.data_root is not None:
        cfg.DATA_ROOT = args.data_root
    if args.checkpoint_dir is not None:
        cfg.CHECKPOINT_DIR = args.checkpoint_dir
    if args.log_dir is not None:
        cfg.LOG_DIR = args.log_dir
    if args.manifest is not None:
        cfg.MANIFEST_PATH = args.manifest
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers


def compute_metrics(pred_depth, gt_depth, pred_seg, gt_seg, num_classes=6):
    """Compute depth RMSE and segmentation mIoU."""
    # Depth RMSE (only valid pixels)
    valid = gt_depth > 0
    if valid.sum() > 0:
        rmse = torch.sqrt(((pred_depth[valid] - gt_depth[valid]) ** 2).mean())
    else:
        rmse = torch.tensor(0.0)

    # Segmentation mIoU
    pred_labels = pred_seg.argmax(dim=1)  # [B, H, W]
    ious = []
    for c in range(num_classes):
        pred_c = pred_labels == c
        gt_c = gt_seg == c
        intersection = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    miou = np.mean(ious) if ious else 0.0

    return rmse.item(), miou


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_losses = {"total": 0, "depth": 0, "seg": 0, "edge": 0}
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        rgb = batch["rgb"].to(device)
        gt_depth = batch["depth"].to(device)
        gt_seg = batch["seg"].to(device)
        confidence = batch["confidence"].to(device)
        da3_depth = batch["da3_depth"].to(device)
        has_da3 = batch["has_da3"]
        batch_has_da3 = any(h.item() if isinstance(h, torch.Tensor) else h
                           for h in has_da3) if hasattr(has_da3, '__iter__') else bool(has_da3)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred_depth, pred_seg = model(rgb)
                losses = criterion(pred_depth, pred_seg, rgb,
                                   gt_depth, gt_seg, confidence,
                                   da3_depth, batch_has_da3)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_depth, pred_seg = model(rgb)
            losses = criterion(pred_depth, pred_seg, rgb,
                               gt_depth, gt_seg, confidence,
                               da3_depth, batch_has_da3)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k]
        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=6):
    model.eval()
    total_losses = {"total": 0, "depth": 0, "seg": 0, "edge": 0}
    all_rmse, all_miou = [], []
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        rgb = batch["rgb"].to(device)
        gt_depth = batch["depth"].to(device)
        gt_seg = batch["seg"].to(device)
        confidence = batch["confidence"].to(device)
        da3_depth = batch["da3_depth"].to(device)
        has_da3 = batch["has_da3"]
        batch_has_da3 = any(h.item() if isinstance(h, torch.Tensor) else h
                           for h in has_da3) if hasattr(has_da3, '__iter__') else bool(has_da3)

        pred_depth, pred_seg = model(rgb)
        losses = criterion(pred_depth, pred_seg, rgb,
                           gt_depth, gt_seg, confidence,
                           da3_depth, batch_has_da3)

        for k in total_losses:
            total_losses[k] += losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k]

        rmse, miou = compute_metrics(pred_depth, gt_depth, pred_seg, gt_seg,
                                     num_classes)
        all_rmse.append(rmse)
        all_miou.append(miou)
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    avg_rmse = np.mean(all_rmse) if all_rmse else 0.0
    avg_miou = np.mean(all_miou) if all_miou else 0.0

    return avg_losses, avg_rmse, avg_miou


def main():
    args = parse_args()
    cfg = Config()
    apply_args(cfg, args)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not available, using CPU. Training will be slow.")
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Directories
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # Data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")

    # Model
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f}M")

    # Loss
    criterion = MultiTaskLoss(
        lambda_depth=cfg.LAMBDA_DEPTH,
        lambda_seg=cfg.LAMBDA_SEG,
        lambda_edge=cfg.LAMBDA_EDGE,
        confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
        num_classes=cfg.NUM_CLASSES,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS)

    # AMP scaler (CUDA only)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # TensorBoard
    writer = SummaryWriter(cfg.LOG_DIR)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ──────────────────────────────────────────────────
    print(f"\nStarting training for {cfg.EPOCHS} epochs...")
    for epoch in range(start_epoch, cfg.EPOCHS):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion,
                                       optimizer, device, scaler)
        val_losses, val_rmse, val_miou = validate(model, val_loader,
                                                   criterion, device,
                                                   cfg.NUM_CLASSES)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} ({elapsed:.1f}s)  "
              f"lr={lr:.6f}  "
              f"train_loss={train_losses['total']:.4f}  "
              f"val_loss={val_losses['total']:.4f}  "
              f"depth_RMSE={val_rmse:.4f}m  "
              f"seg_mIoU={val_miou*100:.1f}%")

        # TensorBoard
        writer.add_scalar("loss/train_total", train_losses["total"], epoch)
        writer.add_scalar("loss/val_total", val_losses["total"], epoch)
        writer.add_scalar("loss/train_depth", train_losses["depth"], epoch)
        writer.add_scalar("loss/train_seg", train_losses["seg"], epoch)
        writer.add_scalar("loss/train_edge", train_losses["edge"], epoch)
        writer.add_scalar("metrics/val_depth_rmse", val_rmse, epoch)
        writer.add_scalar("metrics/val_seg_miou", val_miou, epoch)
        writer.add_scalar("lr", lr, epoch)

        # Checkpointing
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_losses["total"],
            "best_val_loss": best_val_loss,
            "config": vars(cfg),
        }

        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pt")
            torch.save(ckpt_data, path)
            print(f"  Saved checkpoint: {path}")

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            ckpt_data["best_val_loss"] = best_val_loss
            path = os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
            torch.save(ckpt_data, path)
            print(f"  New best val loss: {best_val_loss:.4f} -> {path}")

    # Save final checkpoint
    final_path = os.path.join(cfg.CHECKPOINT_DIR, "final.pt")
    torch.save(ckpt_data, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
