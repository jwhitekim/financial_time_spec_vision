"""
s_t = (A1 + B1·t)·sin(2πt/T1 + φ1) + (A2 + B2·t)·sin(2πt/T2 + φ2)

## 파라미터 정리 - 총 개수 150,000개

파라미터	분포	        의미
A1, A2	   N(1, 0.5)	  진폭
B1, B2	   U(-1/T, 1/T)	  선형 트렌드 기울기
T1	       N(T/5, T/10)	  짧은 주기 (빠른 진동)
T2	       N(T, T/2)	  긴 주기 (느린 파동)
φ1, φ2	   U(0, 2π)	      위상
T	       100 (고정)	   시계열 길이

	
"""

import numpy as np

def generate_synthetic_data(num_samples=150000, T=100, seed=42):
  np.random.seed(seed)
  t = np.arange(1, T + 1)  # t = 1 to T

  # 파라미터 샘플링
  A1 = np.random.normal(1, 0.5, size=(num_samples, 1))
  A2 = np.random.normal(1, 0.5, size=(num_samples, 1))
  B1 = np.random.uniform(-1/T, 1/T, size=(num_samples, 1))
  B2 = np.random.uniform(-1/T, 1/T, size=(num_samples, 1))
  T1 = np.random.normal(T/5, T/10, size=(num_samples, 1))
  T2 = np.random.normal(T, T/2, size=(num_samples, 1))
  phi1 = np.random.uniform(0, 2*np.pi, size=(num_samples, 1))
  phi2 = np.random.uniform(0, 2*np.pi, size=(num_samples, 1))

  # 시계열 생성 (벡터화)
  s = (A1 + B1 * t) * np.sin(2 * np.pi * t / T1 + phi1) \
    + (A2 + B2 * t) * np.sin(2 * np.pi * t / T2 + phi2)

  return s  # shape: (150000, 100)

if __name__ == "__main__":
  synthetic_data = generate_synthetic_data()
  print(f"Shape: {synthetic_data.shape}")
  print(f"Sample [0]: min={synthetic_data[0].min():.3f}, max={synthetic_data[0].max():.3f}")