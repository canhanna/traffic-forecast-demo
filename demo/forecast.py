"""Chương trình minh họa: dự báo lưu lượng kèm khoảng tin cậy hiệu chuẩn, cảnh báo bất thường và nguy cơ vượt ngưỡng.

Quy trình (khớp với chương 3-4 của đồ án):
  1. Đọc CSV (cột thời gian + một hay nhiều cột lưu lượng), gộp theo giờ, nội suy khe <= 3 giờ.
  2. Chia mỗi chuỗi cho trung bình lịch sử (thang đo chung) rồi dự báo Chronos-2 zero-shot trên CPU (phân vị 5/50/95%).
  3. Hiệu chuẩn khoảng bằng CQR: chạy lại dự báo tại các mốc quá khứ gần nhất (walk-forward, chỉ dùng dữ liệu <= mốc),
     tính điểm lệch max(lo-y, y-hi) theo từng chân trời, mở rộng khoảng thêm phân vị bảo thủ của điểm lệch.
  4. Tự kiểm tra: dành phần mốc cuối làm "backtest" để báo độ phủ thực tế trước/sau hiệu chuẩn.
  5. Cảnh báo: (a) bất thường trong 24 giờ gần nhất so với hồ sơ mùa vụ (median +- 3 MAD, như regimes.py);
     (b) nguy cơ vượt ngưỡng dung lượng nếu người dùng cung cấp --capacity.
  6. Xuất báo cáo HTML tự chứa (biểu đồ nhúng), CSV dự báo và JSON tóm tắt.

Chạy (từ thư mục experiments):
  .venv/bin/python -m demo.forecast --csv demo/sample_data/slovak_lte1800_5sectors.csv --horizon 24 --capacity 4000
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tmf.regimes import label_regimes, seasonal_profile  # noqa: E402

MIN_HOURS = 14 * 24      # cần tối thiểu 2 tuần để có hồ sơ mùa vụ tuần và hiệu chuẩn
STRIDE = 6               # khoảng cách giữa các mốc hiệu chuẩn (giờ), như thực nghiệm


# ---------------------------------------------------------------- dữ liệu
def load_series(path: str, time_col: str | None = None, columns: list[str] | None = None,
                agg: str = "sum", max_cols: int = 20) -> pd.DataFrame:
    raw = pd.read_csv(path)
    time_col = time_col or raw.columns[0]
    t = pd.to_datetime(raw[time_col], utc=False, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    if t.isna().any():
        raise ValueError(f"Cột thời gian '{time_col}' có giá trị không đọc được ({int(t.isna().sum())} dòng).")
    raw = raw.drop(columns=[time_col]).set_index(t)
    cols = columns or [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    if not cols:
        raise ValueError("Không có cột số nào để dự báo.")
    if len(cols) > max_cols:
        print(f"[cảnh báo] có {len(cols)} cột, chỉ lấy {max_cols} cột đầu", file=sys.stderr)
        cols = cols[:max_cols]
    x = raw[cols].apply(pd.to_numeric, errors="coerce").sort_index()
    x = x[~x.index.duplicated(keep="last")]
    r = x.resample("h")
    x = r.sum(min_count=1) if agg == "sum" else r.mean()
    x = x.interpolate(limit=3, limit_area="inside")
    return x


def prepare(df: pd.DataFrame):
    """Trả (z, scale, df_used): chỉ giữ phần cuối liền mạch (không NaN) của mỗi chuỗi; cắt chung theo chuỗi ngắn nhất."""
    ok_cols, starts = [], []
    for c in df.columns:
        s = df[c]
        nn = s.notna().values
        if not nn[-1]:
            print(f"[bỏ qua] {c}: giá trị cuối cùng bị thiếu", file=sys.stderr); continue
        k = len(nn) - 1
        while k >= 0 and nn[k]:
            k -= 1
        run = len(nn) - 1 - k
        if run < MIN_HOURS:
            print(f"[bỏ qua] {c}: đoạn liền mạch gần nhất chỉ {run} giờ (< {MIN_HOURS})", file=sys.stderr); continue
        ok_cols.append(c); starts.append(k + 1)
    if not ok_cols:
        raise ValueError(f"Không chuỗi nào có đủ {MIN_HOURS} giờ liền mạch ở cuối dữ liệu.")
    s0 = max(starts)
    d = df[ok_cols].iloc[s0:].clip(lower=0.0)
    scale = d.mean().clip(lower=1e-6)
    return (d / scale).values.astype(np.float32), scale, d


# ---------------------------------------------------------------- hiệu chuẩn
def conformal_q(scores: np.ndarray, alpha: float, axis=None) -> np.ndarray:
    """Phân vị bảo thủ của điểm lệch: mức ceil((n+1)(1-alpha))/n với n là số điểm gộp."""
    n = scores.size if axis is None else scores.shape[axis]
    lvl = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, lvl, axis=axis, method="higher")


def _pooled_q(E: np.ndarray, alpha: float, width: int = 2) -> np.ndarray:
    """E (O,H,N). Với mỗi chân trời h gộp điểm lệch của mọi mốc, mọi chuỗi và các chân trời h-width..h+width
    (z-space đã chuẩn hóa theo trung bình nên so sánh được giữa chuỗi). Ít dữ liệu theo từng chuỗi nên gộp để
    ước lượng ổn định (cùng cách với ACI gộp theo ô ở thực nghiệm). Trả (H,) rồi giữ không giảm theo h."""
    H = E.shape[1]
    q = np.array([conformal_q(E[:, max(0, h - width):h + width + 1, :], alpha) for h in range(H)])
    return np.maximum.accumulate(q)


def calibrate(P: np.ndarray, Y: np.ndarray, alpha: float, backtest_frac: float = 0.25):
    """P (O,H,N,3) phân vị 5/50/95 tại O mốc; Y (O,H,N) giá trị thực.
    Trả: Q (H,) hiệu chuẩn trên toàn bộ mốc, và thống kê backtest (hiệu chuẩn các mốc đầu, đo các mốc cuối)."""
    lo, hi = P[..., 0], P[..., 2]
    E = np.maximum(lo - Y, Y - hi)
    O = E.shape[0]
    nt = max(1, int(round(O * backtest_frac)))
    bt = None
    if O - nt >= 8:
        Qb = _pooled_q(E[:O - nt], alpha)[None, :, None]
        yt, lot, hit = Y[O - nt:], lo[O - nt:], hi[O - nt:]
        bt = {"raw": float(((yt >= lot) & (yt <= hit)).mean()),
              "calibrated": float(((yt >= lot - Qb) & (yt <= hit + Qb)).mean()),
              "n_origins_cal": O - nt, "n_origins_test": nt}
    return _pooled_q(E, alpha), bt


# ---------------------------------------------------------------- cảnh báo
def anomaly_status(z: np.ndarray, idx: pd.DatetimeIndex, recent: int = 24, run: int = 3, k: float = 3.0):
    """Trả danh sách dict theo ô: số giờ bất thường trong `recent` giờ cuối, chuỗi dài nhất, hướng (tăng/giảm)."""
    T = len(z)
    train_end = T - recent
    lab = label_regimes(z, idx, train_end, k=k)[-recent:]
    prof = seasonal_profile(z, idx, train_end)
    key = idx.day_of_week.values * 24 + idx.hour.values
    sign = np.sign(z - prof[key])[-recent:]
    out = []
    for n in range(z.shape[1]):
        a = lab[:, n] == 2
        best = cur = 0
        for v in a:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        dirn = "tăng đột biến" if a.any() and sign[a, n].mean() > 0 else ("giảm đột ngột" if a.any() else "")
        out.append({"hours": int(a.sum()), "longest_run": int(best), "alert": bool(best >= run), "direction": dirn})
    return out


def capacity_status(med: np.ndarray, hi: np.ndarray, cap: float | None):
    if cap is None:
        return {"level": "không đặt ngưỡng", "hours_median": 0, "hours_upper": 0}
    hm, hu = int((med > cap).sum()), int((hi > cap).sum())
    level = "CAO (dự báo trung vị vượt ngưỡng)" if hm else ("LƯU Ý (cận trên 90% vượt ngưỡng)" if hu else "AN TOÀN")
    return {"level": level, "hours_median": hm, "hours_upper": hu}


# ---------------------------------------------------------------- chạy
@dataclass
class Result:
    name: str
    hist: pd.Series
    fidx: pd.DatetimeIndex
    med: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    raw_lo: np.ndarray
    raw_hi: np.ndarray
    anomaly: dict
    capacity: dict
    backtest: dict | None


def run(df: pd.DataFrame, horizon: int = 24, alpha: float = 0.10, capacity: float | None = None,
        calib_origins: int = 40, model=None, threads: int = 4) -> list[Result]:
    z, scale, d = prepare(df)
    idx = d.index
    T, N = z.shape
    hs = list(range(1, horizon + 1))
    if model is None:
        from tmf.models.tsfm import ChronosZeroShot
        model = ChronosZeroShot("amazon/chronos-2", 512, threads=threads)
        model.fit(None, None, 0, {})
    o_max = T - 1 - horizon
    origins = np.array([o for o in (o_max - STRIDE * k for k in range(calib_origins)) if o >= 168][::-1])
    if len(origins) < 12:
        raise ValueError(f"Lịch sử quá ngắn để hiệu chuẩn: chỉ có {len(origins)} mốc (cần >= 12).")
    Pc = model.predict(z, idx, origins, hs, {})                      # (O,H,N,3), chỉ dùng dữ liệu <= mốc
    Yc = np.stack([z[o + 1:o + 1 + horizon] for o in origins])          # (O,H,N)
    Q, bt = calibrate(Pc, Yc, alpha)
    Pl = model.predict(z, idx, np.array([T - 1]), hs, {})[0]          # (H,N,3)
    fidx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    an = anomaly_status(z, idx)
    res = []
    for n, name in enumerate(d.columns):
        s = float(scale.iloc[n])
        med = Pl[:, n, 1] * s
        rlo, rhi = Pl[:, n, 0] * s, Pl[:, n, 2] * s
        lo = np.maximum(0.0, Pl[:, n, 0] - Q) * s
        hi = (Pl[:, n, 2] + Q) * s
        res.append(Result(name, d[name], fidx, med, lo, hi, rlo, rhi, an[n], capacity_status(med, hi, capacity), bt))
    return res


# ---------------------------------------------------------------- báo cáo
def _plot(r: Result, capacity: float | None, alpha: float, days: int = 7) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    h = r.hist.iloc[-24 * days:]
    ax.plot(h.index, h.values, color="#334155", lw=1.2, label="Thực tế")
    ax.fill_between(r.fidx, r.lo, r.hi, color="#2563eb", alpha=0.18, label=f"Khoảng {int((1 - alpha) * 100)}% (đã hiệu chuẩn)")
    ax.plot(r.fidx, r.raw_lo, color="#2563eb", lw=0.6, ls=":"); ax.plot(r.fidx, r.raw_hi, color="#2563eb", lw=0.6, ls=":", label="Khoảng chưa hiệu chuẩn")
    ax.plot(r.fidx, r.med, color="#2563eb", lw=1.6, label="Dự báo (trung vị)")
    if capacity is not None:
        ax.axhline(capacity, color="#dc2626", lw=1, ls="--", label="Ngưỡng dung lượng")
    ax.axvline(r.hist.index[-1], color="#94a3b8", lw=0.8)
    ax.set_title(r.name, fontsize=10, loc="left"); ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.tick_params(labelsize=7); fig.autofmt_xdate(); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_report(results: list[Result], out_dir: str, alpha: float, capacity: float | None, horizon: int) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rows, cards, csv = [], [], []
    for r in results:
        pk = int(np.argmax(r.med))
        a = r.anomaly
        an_txt = f"{a['direction']} ({a['hours']}/24 giờ, liên tiếp tối đa {a['longest_run']})" if a["hours"] else "không"
        rows.append(f"<tr><td>{html.escape(r.name)}</td><td>{r.med.max():,.0f} ({r.fidx[pk]:%d/%m %H:%M})</td><td>{r.hi.max():,.0f}</td>"
                    f"<td>{html.escape(r.capacity['level'])}</td><td>{'⚠ ' if a['alert'] else ''}{html.escape(an_txt)}</td></tr>")
        cards.append(f"<h3>{html.escape(r.name)}</h3><img style='max-width:100%' src='data:image/png;base64,{_plot(r, capacity, alpha)}'>")
        for i, t in enumerate(r.fidx):
            csv.append({"series": r.name, "time": t, "h": i + 1, "median": r.med[i], "lo_calibrated": r.lo[i], "hi_calibrated": r.hi[i],
                        "lo_raw": r.raw_lo[i], "hi_raw": r.raw_hi[i]})
    pd.DataFrame(csv).to_csv(os.path.join(out_dir, "forecast.csv"), index=False)
    b = results[0].backtest
    bt_line = (f"độ phủ khoảng 90% trên {b['n_origins_test']} mốc cuối: chưa hiệu chuẩn {b['raw']:.0%} → hiệu chuẩn {b['calibrated']:.0%} (hiệu chuẩn bằng {b['n_origins_cal']} mốc đầu)" if b else "không đủ mốc để backtest")
    cap_txt = f"{capacity:,.0f}" if capacity is not None else "không đặt"
    doc = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>Dự báo lưu lượng</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 12px;color:#0f172a}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd5e1;padding:5px 8px;text-align:left;font-size:13px}}th{{background:#f1f5f9}}
.note{{background:#fffbeb;border:1px solid #fcd34d;padding:8px 12px;font-size:13px}}</style>
<h2>Dự báo lưu lượng {horizon} giờ tới</h2>
<p>Dữ liệu đến {results[0].hist.index[-1]:%d/%m/%Y %H:%M}. Mô hình: Chronos-2 zero-shot (CPU). Khoảng {int((1 - alpha) * 100)}% hiệu chuẩn bằng CQR. Ngưỡng dung lượng: {cap_txt}.</p>
<table><tr><th>Chuỗi</th><th>Đỉnh dự báo (trung vị)</th><th>Cận trên cao nhất</th><th>Nguy cơ vượt ngưỡng</th><th>Bất thường 24 giờ qua</th></tr>{''.join(rows)}</table>
{''.join(cards)}
<p><b>Backtest chung (gộp mọi chuỗi):</b> {bt_line}</p>
<div class="note"><b>Giới hạn.</b> Công cụ minh họa cho đồ án, không thay thế quyết định vận hành. Độ phủ backtest tính trên ít mốc liên tiếp nên chỉ mang tính tham khảo;
trong thực nghiệm, khoảng chưa hiệu chuẩn thường thiếu phủ ở kỳ lễ và giai đoạn bất thường (xem Chương 4). Cảnh báo bất thường dựa trên hồ sơ mùa vụ của lịch sử và nhạy với lịch sử ngắn.</div></html>"""
    path = os.path.join(out_dir, "report.html")
    open(path, "w", encoding="utf-8").write(doc)
    summary = {r.name: {"peak_median": float(r.med.max()), "max_upper": float(r.hi.max()), "capacity": r.capacity,
                        "anomaly": r.anomaly, "backtest": r.backtest} for r in results}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), ensure_ascii=False, indent=1)
    return {"report": path, "summary": summary}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True); ap.add_argument("--time-col"); ap.add_argument("--columns", nargs="*")
    ap.add_argument("--agg", choices=["sum", "mean"], default="sum", help="cách gộp theo giờ nếu dữ liệu mịn hơn 1 giờ")
    ap.add_argument("--horizon", type=int, default=24); ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--capacity", type=float, help="ngưỡng dung lượng, cùng đơn vị với dữ liệu")
    ap.add_argument("--calib-origins", type=int, default=40); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default="demo_out")
    a = ap.parse_args(argv)
    df = load_series(a.csv, a.time_col, a.columns, a.agg)
    res = run(df, a.horizon, a.alpha, a.capacity, a.calib_origins, threads=a.threads)
    out = write_report(res, a.out_dir, a.alpha, a.capacity, a.horizon)
    for n, s in out["summary"].items():
        print(f"{n}: đỉnh {s['peak_median']:,.0f}; cận trên {s['max_upper']:,.0f}; ngưỡng: {s['capacity']['level']}; "
              f"bất thường: {'CÓ' if s['anomaly']['alert'] else 'không'}; backtest: {s['backtest']}")
    print("Báo cáo:", out["report"])


if __name__ == "__main__":
    main()
