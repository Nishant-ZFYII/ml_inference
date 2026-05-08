#!/usr/bin/env python3
"""
Training script for the multi-task student model.

V5 recipe (based on Vivek's V4):
  - berHu depth loss
  - Encoder/decoder LR split (encoder 0.1x)
  - Encoder freeze warmup (5 epochs)
  - Kendall uncertainty with [-2, 2] clamping
  - ImageNet normalization inside model
  - Supports combined NYU + LILocBench training

Usage (local CPU validation):
    python train.py --epochs 2 --batch-size 4 --device cpu --data-limit 50

Usage (HPC with SLURM):
    See train_v5_lilocbench.slurm
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
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
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-kendall", action="store_true",
                   help="Disable Kendall weighting, use fixed weights")
    p.add_argument("--kendall-lr", type=float, default=None)
    p.add_argument("--backbone", type=str, default=None,
                   help="Backbone: efficientvit_b1, efficientvit_b2, mobilenet_v3_small")
    p.add_argument("--dataset", type=str, default="nyu",
                   choices=["nyu", "tum", "corridor", "lilocbench", "nyu+lilocbench"],
                   help="Dataset to train on")
    p.add_argument("--lilocbench-manifest", type=str, default=None,
                   help="Path to LILocBench manifest.jsonl (for nyu+lilocbench)")
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
    if args.no_kendall:
        cfg.USE_KENDALL = False
    if args.kendall_lr is not None:
        cfg.KENDALL_LR = args.kendall_lr
    if args.backbone is not None:
        cfg.BACKBONE = args.backbone


def compute_metrics(pred_depth, gt_depth, pred_seg, gt_seg, num_classes=6):
    """Compute depth RMSE and segmentation mIoU."""
    valid = gt_depth > 0
    if valid.sum() > 0:
        rmse = torch.sqrt(((pred_depth[valid] - gt_depth[valid]) ** 2).mean())
    else:
        rmse = torch.tensor(0.0)

    pred_labels = pred_seg.argmax(dim=1)
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


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, scaler=None):
    model.train()
    total_losses = {"total": 0, "depth": 0, "seg": 0, "edge": 0}
    n_batches = 0

    clip_params = list(model.parameters()) + list(criterion.parameters())

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
            torch.nn.utils.clip_grad_norm_(clip_params, cfg.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_depth, pred_seg = model(rgb)
            losses = criterion(pred_depth, pred_seg, rgb,
                               gt_depth, gt_seg, confidence,
                               da3_depth, batch_has_da3)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(clip_params, cfg.GRAD_CLIP_NORM)
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


def get_combined_loaders(cfg, args):
    """Build train/val loaders, optionally combining NYU + LILocBench."""
    dataset_name = args.dataset

    if dataset_name == "nyu":
        from dataset.nyu_loader import get_dataloaders
        return get_dataloaders(cfg)
    elif dataset_name == "tum":
        from dataset.tum_loader import get_tum_dataloaders
        return get_tum_dataloaders(cfg)
    elif dataset_name == "corridor":
        from dataset.corridor_loader import get_corridor_dataloaders
        return get_corridor_dataloaders(cfg)
    elif dataset_name == "lilocbench":
        from dataset.lilocbench_loader import get_lilocbench_dataloaders
        return get_lilocbench_dataloaders(cfg, args.lilocbench_manifest)
    elif dataset_name == "nyu+lilocbench":
        from dataset.nyu_loader import get_dataloaders as get_nyu
        from dataset.lilocbench_loader import get_lilocbench_dataloaders

        nyu_train, nyu_val = get_nyu(cfg)
        ll_train, ll_val = get_lilocbench_dataloaders(
            cfg, args.lilocbench_manifest)

        combined_train = ConcatDataset([nyu_train.dataset, ll_train.dataset])
        combined_val = ConcatDataset([nyu_val.dataset, ll_val.dataset])

        print(f"Combined: NYU {len(nyu_train.dataset)} + "
              f"LILocBench {len(ll_train.dataset)} = "
              f"{len(combined_train)} train samples")

        train_loader = DataLoader(
            combined_train, batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            combined_val, batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )
        return train_loader, val_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    args = parse_args()
    cfg = Config()
    apply_args(cfg, args)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not available, using CPU.")
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # Data
    print(f"Loading data (dataset={args.dataset})...")
    train_loader, val_loader = get_combined_loaders(cfg, args)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")

    # Model
    print(f"Backbone: {cfg.BACKBONE}")
    model = build_student(backbone=cfg.BACKBONE, num_classes=cfg.NUM_CLASSES,
                          pretrained=True)
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
        use_kendall=cfg.USE_KENDALL,
    ).to(device)

    # Optimizer with encoder/decoder LR split (V4 recipe)
    encoder_params = list(model.encoder.parameters()) + list(model.neck.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.") and not n.startswith("neck.")]
    encoder_lr = cfg.LR * cfg.ENCODER_LR_SCALE
    param_groups = [
        {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
        {"params": decoder_params, "lr": cfg.LR, "name": "decoder"},
    ]
    if cfg.USE_KENDALL and list(criterion.parameters()):
        param_groups.append({
            "params": list(criterion.parameters()),
            "lr": cfg.KENDALL_LR,
            "name": "kendall",
        })
    print(f"LR: encoder={encoder_lr:.1e}, decoder={cfg.LR:.1e}"
          + (f", kendall={cfg.KENDALL_LR:.1e}" if cfg.USE_KENDALL else ""))

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    writer = SummaryWriter(cfg.LOG_DIR)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    best_depth_rmse = float("inf")
    best_seg_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_depth_rmse = ckpt.get("best_depth_rmse", float("inf"))
        best_seg_miou = ckpt.get("best_seg_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Encoder freeze warmup (V4 recipe)
    def set_encoder_frozen(frozen: bool):
        for p in model.encoder.parameters():
            p.requires_grad = not frozen
        for p in model.neck.parameters():
            p.requires_grad = not frozen

    if cfg.FREEZE_ENCODER_EPOCHS > 0 and start_epoch < cfg.FREEZE_ENCODER_EPOCHS:
        set_encoder_frozen(True)
        print(f"Encoder frozen for first {cfg.FREEZE_ENCODER_EPOCHS} epochs")

    # Training loop
    print(f"\nStarting training for {cfg.EPOCHS} epochs...")
    for epoch in range(start_epoch, cfg.EPOCHS):
        if epoch == cfg.FREEZE_ENCODER_EPOCHS:
            set_encoder_frozen(False)
            print(f"  Encoder unfrozen at epoch {epoch + 1}")

        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion,
                                       optimizer, device, cfg, scaler)
        val_losses, val_rmse, val_miou = validate(model, val_loader,
                                                   criterion, device,
                                                   cfg.NUM_CLASSES)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log_line = (f"Epoch {epoch+1}/{cfg.EPOCHS} ({elapsed:.1f}s) "
                    f"lr={lr:.6f} "
                    f"train_loss={train_losses['total']:.4f} "
                    f"val_loss={val_losses['total']:.4f} "
                    f"depth_RMSE={val_rmse:.4f}m "
                    f"seg_mIoU={val_miou*100:.1f}%")
        print(log_line)

        writer.add_scalar("loss/train_total", train_losses["total"], epoch)
        writer.add_scalar("loss/val_total", val_losses["total"], epoch)
        writer.add_scalar("loss/train_depth", train_losses["depth"], epoch)
        writer.add_scalar("loss/train_seg", train_losses["seg"], epoch)
        writer.add_scalar("loss/train_edge", train_losses["edge"], epoch)
        writer.add_scalar("metrics/val_depth_rmse", val_rmse, epoch)
        writer.add_scalar("metrics/val_seg_miou", val_miou, epoch)
        writer.add_scalar("lr", lr, epoch)
        if cfg.USE_KENDALL:
            writer.add_scalar("kendall/log_sigma_d",
                              criterion.log_sigma_d.item(), epoch)
            writer.add_scalar("kendall/log_sigma_s",
                              criterion.log_sigma_s.item(), epoch)

        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
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

        if val_rmse < best_depth_rmse and val_rmse > 0:
            best_depth_rmse = val_rmse
            ckpt_data["best_depth_rmse"] = best_depth_rmse
            path = os.path.join(cfg.CHECKPOINT_DIR, "best_depth.pt")
            torch.save(ckpt_data, path)
            print(f"  New best depth RMSE: {best_depth_rmse:.4f}m -> {path}")

        if val_miou > best_seg_miou:
            best_seg_miou = val_miou
            ckpt_data["best_seg_miou"] = best_seg_miou
            path = os.path.join(cfg.CHECKPOINT_DIR, "best_seg.pt")
            torch.save(ckpt_data, path)
            print(f"  New best seg mIoU: {val_miou*100:.1f}% -> {path}")

    if start_epoch < cfg.EPOCHS:
        final_path = os.path.join(cfg.CHECKPOINT_DIR, "final.pt")
        torch.save(ckpt_data, final_path)
        print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best depth RMSE: {best_depth_rmse:.4f}m")
    print(f"Best seg mIoU: {best_seg_miou*100:.1f}%")

    writer.close()


if __name__ == "__main__":
    main()
