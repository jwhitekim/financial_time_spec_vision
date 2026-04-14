"""
train.py
========
논문 ViT-num-spec 시계열 예측 학습 스크립트.

사용 예시:
  python train.py --dataset synthetic
  python train.py --dataset financial
  python train.py --dataset temperature
  python train.py --dataset synthetic --freeze_backbone   # head 만 학습

논문 하이퍼파라미터:
  optimizer : AdamW  (weight_decay=0.05)
  batch_size: 128
  max_epochs: 200
  early_stop: patience=10
  loss      : MSE
"""
from __future__ import annotations

import argparse
import os
import sys
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 프로젝트 루트 → sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset import (
    make_synthetic_dataset,
    make_temperature_dataset,
    make_financial_dataset,
)
from src.metrics import MetricAccumulator
from src.model import VisionTransformerForecaster

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── 재현성 ───────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 데이터 로더 ───────────────────────────────────────────────────────────────
def build_loaders(args):
    if args.dataset == "synthetic":
        train_ds = make_synthetic_dataset("train", stride=1)
        val_ds   = make_synthetic_dataset("val",   stride=5)
        test_ds  = make_synthetic_dataset("test",  stride=5)
    elif args.dataset == "temperature":
        train_ds = make_temperature_dataset(split="train", stride=1)
        test_ds  = make_temperature_dataset(split="test",  stride=1)
        val_size   = max(1, int(len(train_ds) * 0.2))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    elif args.dataset == "financial":
        train_ds = make_financial_dataset(split="train", stride=1)
        test_ds  = make_financial_dataset(split="test",  stride=1)
        val_size   = max(1, int(len(train_ds) * 0.2))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        raise ValueError(f"알 수 없는 dataset: {args.dataset}")

    kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"  train epoch {epoch}", unit="batch", leave=False)
    for imgs, targets in pbar:
        imgs    = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss = model(imgs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


# ── 검증 루프 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    acc = MetricAccumulator()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs    = imgs.to(device)
        targets = targets.to(device)
        preds   = model(imgs)
        loss    = nn.functional.mse_loss(preds, targets)
        total_loss += loss.item() * imgs.size(0)
        acc.update(preds, targets)

    results = acc.compute()
    results["mse"] = total_loss / len(loader.dataset)
    return results


# ── 모델 저장 ─────────────────────────────────────────────────────────────────
def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"체크포인트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ViT-num-spec 시계열 예측 학습")

    # 데이터
    parser.add_argument("--dataset", choices=["synthetic", "temperature", "financial"],
                        default="synthetic")
    parser.add_argument("--num_workers", type=int, default=0)

    # 모델
    parser.add_argument("--hf_model", type=str,
                        default="google/vit-base-patch16-224",
                        help="HuggingFace 모델 ID, 로컬 디렉토리, 또는 .bin 파일 경로")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="backbone 동결, head 만 학습")

    # 학습
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience",     type=int,   default=10,
                        help="Early stopping patience")
    parser.add_argument("--seed",         type=int,   default=42)

    # 출력
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--name",       default=None,
                        help="실행 이름 (None → dataset 이름 사용)")

    args = parser.parse_args()

    if args.name is None:
        args.name = args.dataset

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"장치: {device}")

    forecast_len = 10 if args.dataset == "temperature" else 20

    # ── 데이터 로더 ──────────────────────────────────────────────────────────
    logger.info(f"데이터셋 준비: {args.dataset}")
    train_loader, val_loader, test_loader = build_loaders(args)
    logger.info(
        f"  train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,}"
    )

    # ── 모델 ─────────────────────────────────────────────────────────────────
    logger.info(f"모델 로드: {args.hf_model}")
    model = VisionTransformerForecaster(
        hf_model_name=args.hf_model,
        forecast_len=forecast_len,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    logger.info(f"학습 파라미터 수: {model.num_parameters() / 1e6:.1f}M")

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ── 학습 ─────────────────────────────────────────────────────────────────
    best_val_mse = float("inf")
    patience_cnt = 0
    ckpt_path    = os.path.join(args.output_dir, f"{args.name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"[{epoch:3d}/{args.epochs}]  "
            f"train_mse={train_loss:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  "
            f"val_smape={val_metrics['smape']:.4f}  "
            f"val_sign_acc={val_metrics['sign_accuracy']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            patience_cnt = 0
            save_checkpoint(model, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── 테스트 평가 ───────────────────────────────────────────────────────────
    logger.info("=== Test 평가 ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    logger.info(
        f"  test_mse={test_metrics['mse']:.4f}  "
        f"test_smape={test_metrics['smape']:.4f}  "
        f"test_sign_acc={test_metrics['sign_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
