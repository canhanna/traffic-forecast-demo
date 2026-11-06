"""Giao diện chung. Dữ liệu đã chuẩn hoá z (T,N). predict chỉ được dùng z[: o+1] cho mốc o."""
from __future__ import annotations

import numpy as np


class Forecaster:
    name = "base"
    probabilistic = False  # True nếu trả đủ 3 phân vị (0.05, 0.5, 0.95)

    def fit(self, z: np.ndarray, idx, train_end: int, ctx: dict):
        return self

    def predict(self, z: np.ndarray, idx, origins: np.ndarray, hs: list[int], ctx: dict) -> np.ndarray:
        """Trả (n_origins, len(hs), N, 3): phân vị 0.05/0.5/0.95 (mô hình điểm: lặp lại giá trị điểm)."""
        raise NotImplementedError

    @staticmethod
    def _tile(point: np.ndarray) -> np.ndarray:
        return np.repeat(point[..., None], 3, axis=-1)
