#!/usr/bin/env python3
"""
Training script for the multi-task student model.

Usage (local CPU validation):
    python train.py --epochs 2 --batch-size 4 --device cpu --data-limit 50

Usage (HPC with SLURM):
    See train.slurm

Two-phase fine-tuning for corridor data:
    # Phase 1: freeze encoder, train decoders only
    python train.py --resume best.pt --freeze-encoder --epochs 10 --lr 1e-3
    # Phase 2: unfreeze, end-to-end fine-tuning
    python train.py --resume corridor_phase1_best.pt --epochs 100 --lr 1e-4
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
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze encoder weights (train decoders only)")
    p.add_argument("--uncertainty-weighting", action="store_true",
                   help="Use Kendall et al. learned task weighting")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--dataset", type=str, default="nyu",
                   choices=["nyu", "tum", "corridor"],
                   help="Dataset to train on (default: nyu)")
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

        if "w_depth" in losses:
            total_losses.setdefault("w_depth", 0)
            total_losses.setdefault("w_seg", 0)
            total_losses["w_depth"] += losses["w_depth"].item()
            total_losses["w_seg"] += losses["w_seg"].item()

        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=6):
    model.eval()
    total_losses = {"total": 0, "depth": 0, "seg": 0, "edge": 0}
    all_rmse, all_miou = [], []
    n_batches = 0
    has_weights = False

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

        if "w_depth" in losses:
            has_weights = True
            total_losses.setdefault("w_depth", 0)
            total_losses.setdefault("w_seg", 0)
            total_losses["w_depth"] += losses["w_depth"].item()
            total_losses["w_seg"] += losses["w_seg"].item()

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
    print(f"Loading data (dataset={args.dataset})...")
    if args.dataset == "tum":
        from dataset.tum_loader import get_tum_dataloaders
        train_loader, val_loader = get_tum_dataloaders(cfg)
    elif args.dataset == "corridor":
        from dataset.corridor_loader import get_corridor_dataloaders
        train_loader, val_loader = get_corridor_dataloaders(cfg)
    else:
        from dataset.nyu_loader import get_dataloaders
        train_loader, val_loader = get_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")

    # Model
    model = build_student(num_classes=cfg.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        total = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Encoder frozen. Trainable: {trainable:.2f}M / {total:.2f}M total")
    else:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {param_count:.2f}M")

    # Loss
    use_uw = args.uncertainty_weighting
    criterion = MultiTaskLoss(
        lambda_depth=cfg.LAMBDA_DEPTH,
        lambda_seg=cfg.LAMBDA_SEG,
        lambda_edge=cfg.LAMBDA_EDGE,
        confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
        num_classes=cfg.NUM_CLASSES,
        uncertainty_weighting=use_uw,
    ).to(device)
    if use_uw:
        print("Uncertainty weighting enabled (Kendall et al. 2018)")

    # Optimizer & scheduler
    # Include criterion params (log-variance) when uncertainty weighting is on
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if use_uw:
        params += list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
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
    best_depth_rmse = float("inf")
    best_seg_miou = 0.0

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

        log_line = (f"Epoch {epoch+1}/{cfg.EPOCHS} ({elapsed:.1f}s)  "
                    f"lr={lr:.6f}  "
                    f"train_loss={train_losses['total']:.4f}  "
                    f"val_loss={val_losses['total']:.4f}  "
                    f"depth_RMSE={val_rmse:.4f}m  "
                    f"seg_mIoU={val_miou*100:.1f}%")
        if use_uw and "w_depth" in val_losses:
            log_line += (f"  w_d={val_losses['w_depth']:.3f}"
                         f"  w_s={val_losses['w_seg']:.3f}")
        print(log_line)

        # TensorBoard
        writer.add_scalar("loss/train_total", train_losses["total"], epoch)
        writer.add_scalar("loss/val_total", val_losses["total"], epoch)
        writer.add_scalar("loss/train_depth", train_losses["depth"], epoch)
        writer.add_scalar("loss/train_seg", train_losses["seg"], epoch)
        writer.add_scalar("loss/train_edge", train_losses["edge"], epoch)
        writer.add_scalar("metrics/val_depth_rmse", val_rmse, epoch)
        writer.add_scalar("metrics/val_seg_miou", val_miou, epoch)
        writer.add_scalar("lr", lr, epoch)
        if use_uw and "w_depth" in train_losses:
            writer.add_scalar("weights/depth", train_losses["w_depth"], epoch)
            writer.add_scalar("weights/seg", train_losses["w_seg"], epoch)

        # Checkpointing
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_losses["total"],
            "val_depth_rmse": val_rmse,
            "val_seg_miou": val_miou,
            "best_val_loss": best_val_loss,
            "best_depth_rmse": best_depth_rmse,
            "best_seg_miou": best_seg_miou,
            "config": vars(cfg),
        }
        if use_uw:
            ckpt_data["criterion"] = criterion.state_dict()

        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pt")
            torch.save(ckpt_data, path)
            print(f"  Saved checkpoint: {path}")

        # Best combined val loss
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            ckpt_data["best_val_loss"] = best_val_loss
            path = os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
            torch.save(ckpt_data, path)
            print(f"  New best val loss: {best_val_loss:.4f} -> {path}")

        # Best depth RMSE (per-task checkpoint)
        if val_rmse < best_depth_rmse:
            best_depth_rmse = val_rmse
            ckpt_data["best_depth_rmse"] = best_depth_rmse
            path = os.path.join(cfg.CHECKPOINT_DIR, "best_depth.pt")
            torch.save(ckpt_data, path)
            print(f"  New best depth RMSE: {val_rmse:.4f}m -> {path}")

        # Best seg mIoU (per-task checkpoint)
        if val_miou > best_seg_miou:
            best_seg_miou = val_miou
            ckpt_data["best_seg_miou"] = best_seg_miou
            path = os.path.join(cfg.CHECKPOINT_DIR, "best_seg.pt")
            torch.save(ckpt_data, path)
            print(f"  New best seg mIoU: {val_miou*100:.1f}% -> {path}")

    # Save final checkpoint
    final_path = os.path.join(cfg.CHECKPOINT_DIR, "final.pt")
    torch.save(ckpt_data, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best depth RMSE: {best_depth_rmse:.4f}m")
    print(f"Best seg mIoU: {best_seg_miou*100:.1f}%")

    writer.close()


if __name__ == "__main__":
    main()
