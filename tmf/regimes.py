"""Gán nhãn chế độ cho từng (thời điểm, ô): 0 bình thường, 1 cao điểm, 2 bất thường.

Nhãn xác định HẬU NGHIỆM (dùng giá trị thực tại thời điểm mục tiêu) và chỉ phục vụ
phân tầng khi báo cáo; mô hình không được nhìn thấy nhãn. Hồ sơ mùa vụ và độ lệch
chỉ ước lượng từ tập huấn luyện.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NORMAL, PEAK, ANOMALOUS = 0, 1, 2
NAMES = {NORMAL: "normal", PEAK: "peak", ANOMALOUS: "anomalous"}


def seasonal_profile(z: np.ndarray, idx: pd.DatetimeIndex, train_end: int) -> np.ndarray:
    """(168, N): trung vị theo (thứ, giờ) trên tập huấn luyện."""
    key = idx.day_of_week.values * 24 + idx.hour.values
    prof = np.zeros((168, z.shape[1]), dtype=np.float32)
    for k in range(168):
        sel = np.where(key[:train_end] == k)[0]
        prof[k] = np.median(z[sel], axis=0) if len(sel) else np.nan
    return prof


def label_regimes(z, idx, train_end, k: float = 3.0, peak_q: float = 0.75):
    prof = seasonal_profile(z, idx, train_end)
    key = idx.day_of_week.values * 24 + idx.hour.values
    resid = z - prof[key]
    mad = np.median(np.abs(resid[:train_end] - np.median(resid[:train_end], axis=0)), axis=0)
    thr = k * 1.4826 * np.maximum(mad, 1e-6)
    anomalous = np.abs(resid) > thr
    peak_thr = np.quantile(prof, peak_q, axis=0)
    peak = prof[key] >= peak_thr
    lab = np.zeros(z.shape, dtype=np.int8)
    lab[peak] = PEAK
    lab[anomalous] = ANOMALOUS
    return lab
