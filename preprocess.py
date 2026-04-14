"""
preprocess.py
=============
시계열 → Morlet CWT 스펙트로그램 이미지 변환 후 .npz 로 저장.
학습 전 한 번만 실행하면 된다.

사용 예시:
  python preprocess.py --dataset synthetic
  python preprocess.py --dataset financial
  python preprocess.py --dataset temperature
  python preprocess.py --dataset all
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.spectrogram import build_spectrogram_image
from src.dataset import make_windows

OUT_DIR = os.path.join(ROOT, "data", "processed")


# ── 저장 ─────────────────────────────────────────────────────────────────────
def save_npz(windows: list, targets: list, path: str, desc: str):
    N = len(windows)
    images = np.empty((N, 128, 128, 3), dtype=np.uint8)

    for i, win in enumerate(tqdm(windows, desc=desc, unit="샘플")):
        images[i] = build_spectrogram_image(win)

    np.savez_compressed(path, images=images, targets=np.array(targets))
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  저장 완료: {path}  ({N:,}샘플, {size_mb:.1f}MB)")


# ── 데이터셋별 전처리 ─────────────────────────────────────────────────────────
def preprocess_synthetic():
    print("\n=== Synthetic ===")
    from data.generate_synthetic import generate_synthetic_data

    all_data = generate_synthetic_data(num_samples=150_000, T=100)

    splits = {
        "train": all_data[0:80_000],
        "val":   all_data[80_000:100_000],
        "test":  all_data[100_000:150_000],
    }
    strides = {"train": 1, "val": 5, "test": 5}

    for split, data in splits.items():
        series_list = [data[i] for i in range(len(data))]
        windows, targets = make_windows(series_list, input_len=80, forecast_len=20,
                                        stride=strides[split])
        path = os.path.join(OUT_DIR, f"synthetic_{split}.npz")
        save_npz(windows, targets, path, desc=f"synthetic/{split}")


def preprocess_temperature():
    print("\n=== Temperature ===")
    from data.fetch_temperature import parse_tsf_temperature

    tsf_path = os.path.join(ROOT, "data", "raw",
                            "temperature_rain_dataset_without_missing_values.tsf")
    series_dict = parse_tsf_temperature(tsf_path)
    all_series  = list(series_dict.values())

    SPLIT_IDX = 730
    splits = {
        "train": [np.array(s[:SPLIT_IDX]) for s in all_series],
        "test":  [np.array(s[SPLIT_IDX:]) for s in all_series],
    }

    for split, series_list in splits.items():
        windows, targets = make_windows(series_list, input_len=50, forecast_len=10,
                                        stride=1)
        path = os.path.join(OUT_DIR, f"temperature_{split}.npz")
        save_npz(windows, targets, path, desc=f"temperature/{split}")


def preprocess_financial():
    print("\n=== Financial ===")
    import pandas as pd

    csv_path = os.path.join(ROOT, "data", "raw", "sp500_close.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    splits = {
        "train": df.loc["2000-01-01":"2014-12-31"],
        "test":  df.loc["2016-01-01":"2019-12-31"],
    }

    for split, sub_df in splits.items():
        series_list = [sub_df[col].values for col in sub_df.columns]
        windows, targets = make_windows(series_list, input_len=80, forecast_len=20,
                                        stride=1)
        path = os.path.join(OUT_DIR, f"financial_{split}.npz")
        save_npz(windows, targets, path, desc=f"financial/{split}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
HANDLERS = {
    "synthetic":   preprocess_synthetic,
    "temperature": preprocess_temperature,
    "financial":   preprocess_financial,
}


def main():
    parser = argparse.ArgumentParser(description="시계열 → 스펙트로그램 전처리")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "temperature", "financial", "all"],
        default="synthetic",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    targets = list(HANDLERS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        HANDLERS[name]()

    print("\n전처리 완료.")


if __name__ == "__main__":
    main()
