from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from resqai.pipelines.flood_severity.config import TrainConfig
from resqai.pipelines.flood_severity.dataset import build_dataloaders
from resqai.pipelines.flood_severity.model import build_efficientnet_b0
from resqai.pipelines.flood_severity.utils import (
    AverageMeter,
    accuracy_top1,
    autocast_if_available,
    get_device,
    resolve_from_repo,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger("resqai.flood_severity")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    mixed_precision: bool,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_if_available(device, enabled=mixed_precision):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_acc = accuracy_top1(logits.detach(), targets)
        loss_meter.update(loss.item(), n=targets.size(0))
        acc_meter.update(batch_acc, n=targets.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc="val", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_acc = accuracy_top1(logits, targets)
        loss_meter.update(loss.item(), n=targets.size(0))
        acc_meter.update(batch_acc, n=targets.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


def run_training(cfg: TrainConfig) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(cfg.seed)
    device = get_device()

    # Mac MPS dataloading is often more stable with fewer workers.
    num_workers = cfg.num_workers
    if device.type == "mps" and num_workers > 0:
        num_workers = 0

    data = build_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pretrained=cfg.pretrained,
        device_type=device.type,
    )

    model = build_efficientnet_b0(num_classes=len(data.class_names), pretrained=cfg.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = None
    if cfg.mixed_precision and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_acc = -1.0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting training: device=%s train_size=%s val_size=%s classes=%s output_dir=%s",
        device.type,
        data.train_size,
        data.val_size,
        data.class_names,
        str(cfg.output_dir),
    )

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=data.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            mixed_precision=cfg.mixed_precision,
        )
        val_loss, val_acc = validate(
            model=model, loader=data.val_loader, criterion=criterion, device=device
        )

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(
                str(cfg.best_ckpt_path),
                {
                    "model_state": model.state_dict(),
                    "class_names": data.class_names,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": asdict(cfg),
                    "preprocess": {
                        "image_size": cfg.image_size,
                    },
                },
            )

        logger.info(
            "epoch=%03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f%s",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            " (best)" if is_best else "",
        )

    if not cfg.best_ckpt_path.exists():
        raise FileNotFoundError(
            f"Training finished but checkpoint not found at: {cfg.best_ckpt_path}. "
            "Verify that your dataset has a non-empty 'val/' split."
        )

    return cfg.best_ckpt_path


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 flood severity classifier.")
    p.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    p.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    p.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    return TrainConfig(
        data_dir=resolve_from_repo(args.data_dir),
        output_dir=resolve_from_repo(args.output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        mixed_precision=not args.no_amp,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
