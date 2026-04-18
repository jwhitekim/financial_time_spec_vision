"""
preprocess.py
=============
시계열 → 이미지 변환 후 .npz 로 저장.
학습 전 한 번만 실행하면 된다.

사용 예시:
  # 기본 (스펙트로그램)
  python preprocess.py --dataset synthetic
  python preprocess.py --dataset financial
  python preprocess.py --dataset temperature
  python preprocess.py --dataset all

  # 이미지 타입 선택
  python preprocess.py --dataset financial --image_type lineplot
  python preprocess.py --dataset financial --image_type intensity

  # scale 어블레이션 (스펙트로그램 전용)
  python preprocess.py --dataset financial --scale_max_ratio 0.25
  python preprocess.py --dataset financial --scale_max_ratio 1.0
  python preprocess.py --dataset financial --scale_max_ratio 2.0

  # split 선택 (synthetic 전용)
  python preprocess.py --dataset synthetic --split train

저장 파일 명명 규칙:
  image_type=spec,  scale=0.5  → financial_train.npz          (기본)
  image_type=spec,  scale=0.25 → financial_train_s0.25.npz
  image_type=lineplot           → financial_train_lineplot.npz
  image_type=intensity          → financial_train_intensity.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.spectrogram import build_image, IMAGE_TYPES
from src.dataset import make_windows

OUT_DIR = os.path.join(ROOT, "data", "processed")


# ── 출력 경로 결정 ────────────────────────────────────────────────────────────
def _out_path(base_name: str, image_type: str, scale_max_ratio: float) -> str:
    """
    (예) base_name="financial_train"
      spec + 0.5  → financial_train.npz           (기본, 논문 설정)
      spec + 0.25 → financial_train_s0.25.npz
      lineplot    → financial_train_lineplot.npz
      intensity   → financial_train_intensity.npz
    """
    if image_type == "spec":
        suffix = "" if scale_max_ratio == 0.5 else f"_s{scale_max_ratio}"
    else:
        suffix = f"_{image_type}"
    return os.path.join(OUT_DIR, f"{base_name}{suffix}.npz")


# ── 저장 ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 50_000   # 청크당 ~2.3 GB (128×128×3 uint8 × 50K)


def save_npz(
    windows:    list,
    targets:    list,
    last_vals:  list,
    norm_mins:  list,
    norm_ranges: list,
    path:       str,
    desc:       str,
    image_type:      str   = "spec",
    scale_max_ratio: float = 0.5,
):
    """
    샘플 수가 CHUNK_SIZE 를 초과하면 청크 파일로 분할 저장.
      단일 파일: <path>.npz
      분할 파일: <base>_chunk000.npz, <base>_chunk001.npz, ...

    npz 내부 구조
    -------------
    images     : uint8  (N, 128, 128, 3)
    windows    : float32 (N, input_len)  — ViT-num / 온더플라이 lineplot/intensity 용
    targets    : float32 (N, forecast_len)
    last_vals  : float32 (N,)
    norm_min   : float32 (N,)
    norm_range : float32 (N,)
    """
    N        = len(windows)
    n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE
    win_arr  = np.array(windows, dtype=np.float32)   # (N, input_len)

    def _make_images(win_list):
        imgs = np.empty((len(win_list), 128, 128, 3), dtype=np.uint8)
        for i, w in enumerate(win_list):
            imgs[i] = build_image(w, image_type=image_type,
                                  scale_max_ratio=scale_max_ratio)
        return imgs

    if n_chunks == 1:
        images = np.empty((N, 128, 128, 3), dtype=np.uint8)
        for i, win in enumerate(tqdm(windows, desc=desc, unit="샘플")):
            images[i] = build_image(win, image_type=image_type,
                                    scale_max_ratio=scale_max_ratio)
        np.savez_compressed(
            path,
            images=images,
            windows=win_arr,
            targets=np.array(targets),
            last_vals=np.array(last_vals,    dtype=np.float32),
            norm_min=np.array(norm_mins,     dtype=np.float32),
            norm_range=np.array(norm_ranges, dtype=np.float32),
        )
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  저장 완료: {path}  ({N:,}샘플, {size_mb:.1f}MB)")
    else:
        base       = path.replace(".npz", "")
        saved_paths = []
        pbar = tqdm(total=N, desc=desc, unit="샘플")
        for c in range(n_chunks):
            start = c * CHUNK_SIZE
            end   = min(start + CHUNK_SIZE, N)
            m     = end - start
            chunk_images = np.empty((m, 128, 128, 3), dtype=np.uint8)
            for i, win in enumerate(windows[start:end]):
                chunk_images[i] = build_image(win, image_type=image_type,
                                              scale_max_ratio=scale_max_ratio)
                pbar.update(1)
            chunk_path = f"{base}_chunk{c:03d}.npz"
            np.savez_compressed(
                chunk_path,
                images=chunk_images,
                windows=win_arr[start:end],
                targets=np.array(targets[start:end]),
                last_vals=np.array(last_vals[start:end],    dtype=np.float32),
                norm_min=np.array(norm_mins[start:end],     dtype=np.float32),
                norm_range=np.array(norm_ranges[start:end], dtype=np.float32),
            )
            saved_paths.append(chunk_path)
        pbar.close()
        total_mb = sum(os.path.getsize(p) for p in saved_paths) / 1024 / 1024
        print(f"  저장 완료: {n_chunks}개 청크  ({N:,}샘플, {total_mb:.1f}MB)")


# ── 데이터셋별 전처리 ─────────────────────────────────────────────────────────
def preprocess_synthetic(only_split: str | None = None,
                         image_type: str = "spec",
                         scale_max_ratio: float = 0.5):
    print("\n=== Synthetic ===")
    from data.generate_synthetic import generate_synthetic_data

    all_data = generate_synthetic_data(num_samples=150_000, T=100)

    split_ranges = {
        "train": all_data[0:80_000],
        "val":   all_data[80_000:100_000],
        "test":  all_data[100_000:150_000],
    }
    strides = {"train": 1, "val": 5, "test": 5}

    splits = [only_split] if only_split else list(split_ranges.keys())
    for split in splits:
        data        = split_ranges[split]
        series_list = [data[i] for i in range(len(data))]
        windows, tgts, last_vals, norm_mins, norm_ranges = make_windows(
            series_list, input_len=80, forecast_len=20, stride=strides[split])
        path = _out_path(f"synthetic_{split}", image_type, scale_max_ratio)
        save_npz(windows, tgts, last_vals, norm_mins, norm_ranges, path,
                 desc=f"synthetic/{split}",
                 image_type=image_type, scale_max_ratio=scale_max_ratio)


def preprocess_temperature(image_type: str = "spec",
                           scale_max_ratio: float = 0.5):
    print("\n=== Temperature ===")
    from data.fetch_temperature import parse_tsf_temperature

    tsf_path = os.path.join(ROOT, "data", "raw",
                            "temperature_rain_dataset_without_missing_values.tsf")
    series_dict = parse_tsf_temperature(tsf_path)
    all_series  = list(series_dict.values())

    rng     = np.random.default_rng(42)
    indices = rng.choice(len(all_series), size=244, replace=False)
    selected = [np.array(all_series[i]) for i in sorted(indices)]

    windows, targets, last_vals, norm_mins, norm_ranges = make_windows(
        selected, input_len=50, forecast_len=10, stride=22, max_windows_per_series=30
    )
    print(f"  관측소: 244개, 총 샘플: {len(windows):,}개")
    path = _out_path("temperature_train", image_type, scale_max_ratio)
    save_npz(windows, targets, last_vals, norm_mins, norm_ranges, path,
             desc="temperature",
             image_type=image_type, scale_max_ratio=scale_max_ratio)


def preprocess_financial(image_type: str = "spec",
                         scale_max_ratio: float = 0.5):
    print("\n=== Financial ===")
    import pandas as pd

    csv_path = os.path.join(ROOT, "data", "raw", "sp500_close.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    rng          = np.random.default_rng(42)
    all_cols     = list(df.columns)
    selected_cols = list(rng.choice(all_cols, size=500, replace=False))
    df = df[selected_cols]
    print(f"  선택된 종목 수: {len(selected_cols)}개 (전체 {len(all_cols)}개 중)")

    splits = {
        "train": df.loc["2000-01-01":"2014-12-31"],
        "test":  df.loc["2016-01-01":"2019-12-31"],
    }
    split_cfg = {
        "train": {"stride": 39, "max_windows": 93},   # 500×93 = 46,500
        "test":  {"stride": 30, "max_windows": 31},   # 500×31 = 15,500
    }
    for split, sub_df in splits.items():
        cfg         = split_cfg[split]
        series_list = [sub_df[col].values for col in sub_df.columns]
        windows, targets, last_vals, norm_mins, norm_ranges = make_windows(
            series_list, input_len=80, forecast_len=20,
            stride=cfg["stride"], max_windows_per_series=cfg["max_windows"])
        print(f"  financial/{split}: {len(windows):,}개 "
              f"(목표: {'46,875' if split=='train' else '15,625'})")
        path = _out_path(f"financial_{split}", image_type, scale_max_ratio)
        save_npz(windows, targets, last_vals, norm_mins, norm_ranges, path,
                 desc=f"financial/{split}",
                 image_type=image_type, scale_max_ratio=scale_max_ratio)


# ── 메인 ─────────────────────────────────────────────────────────────────────
HANDLERS = {
    "synthetic":   preprocess_synthetic,
    "temperature": preprocess_temperature,
    "financial":   preprocess_financial,
}


def main():
    parser = argparse.ArgumentParser(description="시계열 → 이미지 전처리")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "temperature", "financial", "all"],
        default="synthetic",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="synthetic 전용: 처리할 split 지정. 예) --split train",
    )
    parser.add_argument(
        "--image_type",
        choices=list(IMAGE_TYPES),
        default="spec",
        help=(
            "이미지 변환 방식 (기본: spec)\n"
            "  spec      : Morlet CWT 스펙트로그램 + intensity stripe (논문 제안)\n"
            "  lineplot  : 꺾은선 그래프 (ViT-lineplot 베이스라인)\n"
            "  intensity : intensity stripe 만 확장 (어블레이션)"
        ),
    )
    parser.add_argument(
        "--scale_max_ratio",
        type=float,
        default=0.5,
        help=(
            "CWT 최대 스케일 = input_len × scale_max_ratio (기본: 0.5 = T/2)\n"
            "어블레이션 추천: 0.25, 0.5, 1.0, 2.0\n"
            "image_type=spec 에서만 효과 있음"
        ),
    )
    args = parser.parse_args()

    if args.split and args.dataset not in ("synthetic",):
        parser.error("--split 은 --dataset synthetic 에서만 사용할 수 있습니다.")

    os.makedirs(OUT_DIR, exist_ok=True)

    dataset_names = list(HANDLERS.keys()) if args.dataset == "all" else [args.dataset]
    for name in dataset_names:
        if name == "synthetic":
            preprocess_synthetic(only_split=args.split,
                                 image_type=args.image_type,
                                 scale_max_ratio=args.scale_max_ratio)
        else:
            HANDLERS[name](image_type=args.image_type,
                           scale_max_ratio=args.scale_max_ratio)

    print("\n전처리 완료.")


if __name__ == "__main__":
    main()
