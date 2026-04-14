"""
dataset.py
==========
전처리가 완료된 .npz 파일을 로드하는 Dataset 클래스.

전처리(CWT 계산)는 preprocess.py 에서 별도로 실행한다.
  python preprocess.py --dataset synthetic

생성된 파일:
  data/processed/synthetic_train.npz
  data/processed/synthetic_val.npz
  data/processed/synthetic_test.npz
  ...
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── 기본 transform (ViT 입력용) ──────────────────────────────────────────────
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                         # uint8 [0,255] → float [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),     # [0,1] → [-1,1]
])


# ── NpzDataset ────────────────────────────────────────────────────────────────
class NpzDataset(Dataset):
    """
    preprocess.py 가 저장한 .npz 파일에서 (이미지, 타깃) 쌍을 로드.

    .npz 내부 구조:
      images  : uint8  (N, 128, 128, 3)
      targets : float32 (N, forecast_len)
    """

    def __init__(self, npz_path: str, transform=None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"전처리 파일 없음: {npz_path}\n"
                f"먼저 실행하세요: python preprocess.py --dataset <name>"
            )
        data = np.load(npz_path)
        self._images  = data["images"]    # (N, 128, 128, 3) uint8
        self._targets = data["targets"]   # (N, forecast_len) float32
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        from PIL import Image
        img    = self.transform(Image.fromarray(self._images[idx]))
        target = torch.from_numpy(self._targets[idx])
        return img, target


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def make_synthetic_dataset(split: str = "train", transform=None) -> NpzDataset:
    path = os.path.join(PROCESSED_DIR, f"synthetic_{split}.npz")
    return NpzDataset(path, transform)


def make_temperature_dataset(split: str = "train", transform=None) -> NpzDataset:
    path = os.path.join(PROCESSED_DIR, f"temperature_{split}.npz")
    return NpzDataset(path, transform)


def make_financial_dataset(split: str = "train", transform=None) -> NpzDataset:
    path = os.path.join(PROCESSED_DIR, f"financial_{split}.npz")
    return NpzDataset(path, transform)


# ── 윈도우 생성 헬퍼 (preprocess.py 에서 사용) ───────────────────────────────
def forward_fill(series: np.ndarray) -> np.ndarray:
    series = np.array(series, dtype=np.float64)
    mask = np.isnan(series)
    if not mask.any():
        return series
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, out=idx)
    series[mask] = series[idx[mask]]
    return np.where(np.isnan(series), 0.0, series)


def make_windows(
    series_list: list[np.ndarray],
    input_len: int,
    forecast_len: int,
    stride: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """시계열 리스트 → 정규화된 (windows, targets) 리스트."""
    windows, targets = [], []
    needed = input_len + forecast_len
    for raw in series_list:
        series = forward_fill(np.asarray(raw, dtype=np.float64))
        if len(series) < needed:
            continue
        for start in range(0, len(series) - needed + 1, stride):
            w = series[start : start + input_len]
            t = series[start + input_len : start + input_len + forecast_len]
            denom = (w.max() - w.min()) or 1.0
            windows.append(((w - w.min()) / denom).astype(np.float32))
            targets.append(((t - w.min()) / denom).astype(np.float32))
    return windows, targets
