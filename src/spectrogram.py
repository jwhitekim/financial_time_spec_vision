"""
spectrogram.py
==============
논문 "From Pixels to Predictions: Spectrogram and Vision Transformer
for Better Time Series Forecasting" (ICAIF 2023) 의 핵심 이미지 변환 모듈.

128×128 RGB 이미지 구조:
  rows   0-15  : intensity stripe  (원본 시계열 값, 부호 보존, [0-255])
  rows  16-127 : Morlet CWT 스펙트로그램 magnitude (고주파 위, 저주파 아래)
"""

import numpy as np
import pywt


# ── 상수 ────────────────────────────────────────────────────────────────────
IMG_SIZE       = 128
STRIPE_ROWS    = 16
SPEC_ROWS      = IMG_SIZE - STRIPE_ROWS   # 112
# complex Morlet: 'cmorB-C'  B=대역폭, C=중심 주파수 (논문 w=5 에 대응)
MORLET_WAVELET = "cmor1.5-1.0"


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def _minmax_norm(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """배열을 [0, 1]로 정규화. 분모가 거의 0이면 0 반환."""
    a_min, a_max = arr.min(), arr.max()
    denom = a_max - a_min
    return (arr - a_min) / (denom if denom > eps else 1.0)


def _resize_cols(arr2d: np.ndarray, target_width: int) -> np.ndarray:
    """2D 배열의 열(axis=1)을 target_width로 선형 보간 리사이즈."""
    T = arr2d.shape[1]
    if T == target_width:
        return arr2d
    x_old = np.arange(T)
    x_new = np.linspace(0, T - 1, target_width)
    return np.array([np.interp(x_new, x_old, row) for row in arr2d])


# ── 공개 API ─────────────────────────────────────────────────────────────────
def morlet_cwt(series: np.ndarray, n_scales: int = SPEC_ROWS) -> np.ndarray:
    """
    Morlet CWT (연속 웨이블릿 변환) magnitude 계산.

    Parameters
    ----------
    series   : 1D numpy array (입력 시계열, 정규화 권장)
    n_scales : 주파수 스케일 수 (기본 112, spectrogram 행 수)

    Returns
    -------
    magnitude : ndarray, shape (n_scales, len(series))
                각 스케일의 CWT 계수 절댓값.
                스케일은 로그 간격 (1 ~ T/2), 인덱스 0 = 가장 작은 스케일(고주파).
    """
    T = len(series)
    series = np.asarray(series, dtype=np.float64)   # pywt cmor 는 float64 필요
    scales = np.geomspace(1, max(T / 2, 2), n_scales)
    coef, _ = pywt.cwt(series, scales, MORLET_WAVELET)
    return np.abs(coef)   # (n_scales, T)


def build_spectrogram_image(
    series: np.ndarray,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    1D 시계열 → 128×128 RGB 스펙트로그램 이미지 변환.

    Parameters
    ----------
    series   : 1D numpy array, 입력 시계열 (임의 길이 가능)
    img_size : 출력 이미지 크기 (기본 128)

    Returns
    -------
    rgb : uint8 ndarray, shape (img_size, img_size, 3)
          ViT 입력용 RGB 이미지 (값 범위 0~255).
    """
    stripe_rows = STRIPE_ROWS
    spec_rows   = img_size - stripe_rows   # 112

    # ── 1. 시계열 정규화: [-1, 1] ──────────────────────────────────────────
    s_min, s_max = series.min(), series.max()
    denom = (s_max - s_min) if (s_max - s_min) > 1e-8 else 1.0
    series_norm = 2.0 * (series - s_min) / denom - 1.0   # [-1, 1]

    # ── 2. Intensity Stripe (행 0-15) ──────────────────────────────────────
    #    [-1, 1] → [0, 255]  (0 → 128, 부호 정보 보존)
    stripe_1d = ((series_norm + 1.0) / 2.0 * 255.0).clip(0, 255)
    stripe_2d = np.tile(stripe_1d, (stripe_rows, 1))      # (16, T)
    stripe_2d = _resize_cols(stripe_2d, img_size)          # (16, img_size)
    stripe_2d = stripe_2d.clip(0, 255).astype(np.uint8)

    # ── 3. Morlet CWT Spectrogram (행 16-127) ──────────────────────────────
    magnitude = morlet_cwt(series_norm, n_scales=spec_rows)   # (112, T)

    mag_norm = _minmax_norm(magnitude) * 255.0                 # [0, 255]
    mag_norm = _resize_cols(mag_norm, img_size)                # (112, img_size)
    mag_norm = mag_norm.clip(0, 255).astype(np.uint8)

    # 행 뒤집기: 인덱스 0이 가장 작은 스케일(고주파) → 이미지 상단에 위치
    mag_norm = mag_norm[::-1].copy()

    # ── 4. [stripe; spectrogram] 합치기 ────────────────────────────────────
    gray = np.vstack([stripe_2d, mag_norm])                   # (128, 128)
    rgb  = np.stack([gray, gray, gray], axis=-1)              # (128, 128, 3)
    return rgb


# ── 디버그 / 시각화 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.linspace(0, 1, 80)
    test_series = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 20 * t)
        + 0.1 * np.random.randn(80)
    )

    img = build_spectrogram_image(test_series)
    print(f"이미지 shape : {img.shape}")
    print(f"값 범위     : [{img.min()}, {img.max()}]")

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Spectrogram Image (128×128)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("spectrogram_test.png", dpi=100)
    print("저장 완료: spectrogram_test.png")
