"""Mô hình nền chuỗi thời gian qua thư viện chronos-forecasting.

- Chronos-2 (`amazon/chronos-2`, 120M): zero-shot, hoặc có biến đồng hành, hoặc tinh chỉnh (LoRA / đầy đủ) CHỈ trên đoạn huấn luyện.
- Chronos-Bolt (`amazon/chronos-bolt-*`): mô hình nền thứ hai (kiến trúc khác), chỉ lấy trung vị (mô hình điểm).

Mỗi (mốc o, ô n) là một chuỗi riêng; ngữ cảnh chỉ gồm z[o-L+1 : o+1] (quá khứ). Biến đồng hành:
  nb   : trung bình 8 ô lân cận (chỉ quá khứ)           -> thông tin không gian
  city : trung bình toàn khối tại từng thời điểm (quá khứ)
  cal  : đặc trưng lịch (giờ, thứ, ngày lễ), biết trước cả tương lai
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Forecaster

QL = [0.05, 0.5, 0.95]
BOLT_QL = [0.1, 0.5, 0.9]


class ChronosZeroShot(Forecaster):
    def __init__(self, model_id="amazon/chronos-2", context=512, batch=256, threads=4, cov=(), finetune=None,
                 device="cpu", seed=0):
        self.model_id, self.context, self.batch, self.threads = model_id, context, batch, threads
        self.cov, self.ft, self.device, self.seed = tuple(cov), finetune, device, seed
        self.bolt = "bolt" in model_id
        self.probabilistic = not self.bolt
        if self.bolt:
            self.name = "Chronos-Bolt-" + model_id.split("-")[-1]
        else:
            self.name = "Chronos-2" + ("-FT" if finetune else "") + (("[" + ",".join(self.cov) + "]") if self.cov else "")
        if finetune and "cal" in self.cov:
            raise ValueError("tinh chỉnh chưa hỗ trợ biến lịch tương lai trong cài đặt này")

    # -------- tạo đầu vào
    def _prep(self, z, ctx):
        nb8 = None if ctx.get("neigh") is None else ctx["neigh"][:, 1:]
        self._nb = z[:, nb8].mean(axis=2) if ("nb" in self.cov and nb8 is not None) else None
        self._city = z.mean(axis=1) if "city" in self.cov else None
        if "nb" in self.cov and nb8 is None:
            raise ValueError("biến 'nb' cần ctx['neigh'] (nguồn milan_block)")

    def _item(self, z, ctx, lo, o, n, H, future):
        tgt = z[lo:o + 1, n].astype(np.float32)
        if not self.cov:
            return torch.from_numpy(tgt)
        past, fut = {}, {}
        if self._nb is not None:
            past["nb"] = self._nb[lo:o + 1, n].astype(np.float32)
        if self._city is not None:
            past["city"] = self._city[lo:o + 1].astype(np.float32)
        if "cal" in self.cov and future:
            cal = ctx["cal"]
            for i in range(cal.shape[1]):
                past[f"cal{i}"] = cal[lo:o + 1, i]
                fut[f"cal{i}"] = cal[o + 1:o + 1 + H, i]
        d = {"target": tgt, "past_covariates": past}
        if fut:
            d["future_covariates"] = fut
        return d

    # -------- huấn luyện / nạp
    def fit(self, z, idx, train_end, ctx):
        torch.set_num_threads(self.threads)
        dev = self.device
        if self.bolt:
            from chronos import BaseChronosPipeline
            self.pipe = BaseChronosPipeline.from_pretrained(self.model_id, device_map=dev)
            return self
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained(self.model_id, device_map=dev)
        if self.ft:
            torch.manual_seed(self.seed)
            self._prep(z, ctx)
            H = max(ctx["hs"])
            L = min(self.context, train_end)
            inputs = [self._item(z, ctx, 0, train_end - 1, n, H, False) for n in range(z.shape[1])]   # chỉ [0, train_end)
            inputs = [x if isinstance(x, dict) else {"target": x.numpy()} for x in inputs]
            self.pipe = self.pipe.fit(
                inputs, prediction_length=H, finetune_mode=self.ft.get("mode", "lora"),
                learning_rate=self.ft.get("lr", 1e-5), num_steps=self.ft.get("steps", 300),
                batch_size=self.ft.get("batch", 64), context_length=L, remove_printer_callback=True)
        return self

    # -------- dự báo
    def predict(self, z, idx, origins, hs, ctx):
        self._prep(z, ctx)
        N, H = z.shape[1], max(hs)
        keys = [(oi, n) for oi in range(len(origins)) for n in range(N)]
        k = 1 if self.bolt else 3
        out = np.empty((len(origins), N, H, k), dtype=np.float32)
        for i in range(0, len(keys), self.batch):
            chunk = keys[i:i + self.batch]
            inputs = []
            for oi, n in chunk:
                o = int(origins[oi])
                inputs.append(self._item(z, ctx, max(0, o + 1 - self.context), o, n, H, True))   # chỉ <= o
            with torch.no_grad():
                if self.bolt:
                    q, _ = self.pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=BOLT_QL)
                    for (oi, n), t in zip(chunk, q):
                        out[oi, n, :, 0] = t[:, 1].numpy()
                else:
                    q, _ = self.pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=QL)
                    for (oi, n), t in zip(chunk, q):
                        out[oi, n] = t[0].numpy()
        P = out.transpose(0, 2, 1, 3)[:, np.array(hs) - 1]
        if self.bolt:
            return np.maximum(self._tile(P[..., 0]), 0.0).astype(np.float32)
        return np.maximum(np.sort(P, axis=-1), 0.0).astype(np.float32)
