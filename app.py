"""Ứng dụng Streamlit: dự báo lưu lượng mạng di động kèm khoảng dự báo hiệu chuẩn, cảnh báo bất thường và vượt dung lượng.

Chạy cục bộ:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
import tempfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demo.forecast import load_series, run, write_report, MIN_HOURS

SAMPLE = os.path.join(os.path.dirname(__file__), "demo", "sample_data", "slovak_lte1800_5sectors.csv")

st.set_page_config(page_title="Dự báo lưu lượng mạng di động", page_icon="📶", layout="wide")


@st.cache_resource(show_spinner="Đang tải mô hình Chronos-2 (lần đầu có thể mất vài phút)...")
def get_model():
    from tmf.models.tsfm import ChronosZeroShot
    m = ChronosZeroShot("amazon/chronos-2", 512, threads=2)
    m.fit(None, None, 0, {})
    return m


@st.cache_data(show_spinner=False)
def read_csv(content: bytes, agg: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        return load_series(path, agg=agg)
    finally:
        os.unlink(path)


def chart(r, capacity, days=7):
    h = r.hist.iloc[-24 * days:]
    hist = pd.DataFrame({"Thời gian": h.index, "Giá trị": h.values, "Loại": "Thực tế"})
    fc = pd.DataFrame({"Thời gian": r.fidx, "Trung vị": r.med, "Cận dưới (đã hiệu chuẩn)": r.lo, "Cận trên (đã hiệu chuẩn)": r.hi,
                       "Cận dưới (gốc)": r.raw_lo, "Cận trên (gốc)": r.raw_hi})
    x = alt.X("Thời gian:T", title=None)
    line_h = alt.Chart(hist).mark_line(color="#334155").encode(x=x, y=alt.Y("Giá trị:Q", title="Lưu lượng"))
    band = alt.Chart(fc).mark_area(opacity=0.2, color="#2563eb").encode(x=x, y="Cận dưới (đã hiệu chuẩn):Q", y2="Cận trên (đã hiệu chuẩn):Q")
    raw_lo = alt.Chart(fc).mark_line(color="#2563eb", strokeDash=[3, 3], opacity=0.6).encode(x=x, y="Cận dưới (gốc):Q")
    raw_hi = alt.Chart(fc).mark_line(color="#2563eb", strokeDash=[3, 3], opacity=0.6).encode(x=x, y="Cận trên (gốc):Q")
    med = alt.Chart(fc).mark_line(color="#2563eb", strokeWidth=2).encode(x=x, y="Trung vị:Q")
    layers = [band, raw_lo, raw_hi, med, line_h]
    if capacity:
        layers.append(alt.Chart(pd.DataFrame({"y": [capacity]})).mark_rule(color="#dc2626", strokeDash=[6, 4]).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=300).interactive()


st.title("📶 Dự báo lưu lượng mạng di động")
st.caption("Chương trình minh họa của đồ án: Chronos-2 zero-shot, khoảng dự báo 90% hiệu chuẩn bằng CQR, cảnh báo bất thường và nguy cơ vượt dung lượng.")

with st.sidebar:
    st.header("Dữ liệu")
    src = st.radio("Nguồn dữ liệu", ["Dữ liệu mẫu (5 sector)", "Tải lên tệp CSV"])
    upload, agg = None, "sum"
    if src.startswith("Tải"):
        upload = st.file_uploader("Tệp CSV (cột đầu là thời gian, các cột sau là lưu lượng)", type="csv")
        agg = st.radio("Cách gộp theo giờ (nếu dữ liệu mịn hơn 1 giờ)", ["sum", "mean"], format_func=lambda v: "Tổng" if v == "sum" else "Trung bình", horizontal=True)
    st.header("Tham số")
    horizon = st.slider("Số giờ dự báo (bước dự báo)", 1, 48, 24)
    cap = st.number_input("Ngưỡng dung lượng (0 = không đặt)", min_value=0.0, value=4000.0 if src.startswith("Dữ liệu mẫu") else 0.0, step=100.0)
    calib = st.slider("Số mốc hiệu chuẩn", 12, 40, 40, help="Mỗi mốc cách nhau 6 giờ; nhiều mốc hơn thì hiệu chuẩn ổn định hơn nhưng chạy lâu hơn.")
    go = st.button("Chạy dự báo", type="primary", width='stretch')
    st.caption(f"Mỗi chuỗi cần ít nhất {MIN_HOURS // 24} ngày dữ liệu liên tục theo giờ ở cuối.")

if go:
    try:
        if src.startswith("Dữ liệu mẫu"):
            df = load_series(SAMPLE)
        elif upload is None:
            st.warning("Hãy chọn tệp CSV.")
            st.stop()
        else:
            df = read_csv(upload.getvalue(), agg)
        with st.spinner("Đang dự báo và hiệu chuẩn..."):
            res = run(df, horizon=horizon, capacity=cap or None, calib_origins=calib, model=get_model())
        st.session_state["res"] = (res, cap or None, horizon)
    except ValueError as e:
        st.error(str(e))
        st.stop()

if "res" not in st.session_state:
    st.info("Chọn dữ liệu và tham số ở thanh bên rồi bấm **Chạy dự báo**.")
    st.stop()

res, capacity, horizon = st.session_state["res"]
st.subheader(f"Kết quả dự báo {horizon} giờ tới")
rows = []
for r in res:
    a = r.anomaly
    rows.append({"Chuỗi": r.name, "Đỉnh dự báo (trung vị)": round(float(r.med.max())), "Cận trên 90% cao nhất": round(float(r.hi.max())),
                 "Nguy cơ vượt ngưỡng": r.capacity["level"],
                 "Bất thường 24 giờ qua": (f"⚠ {a['direction']} ({a['hours']}/24 giờ)" if a["alert"] else (f"{a['direction']} ({a['hours']}/24 giờ)" if a["hours"] else "Không"))})
st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

b = res[0].backtest
if b:
    c1, c2, c3 = st.columns(3)
    c1.metric("Độ phủ khoảng 90% trước hiệu chuẩn", f"{b['raw']:.0%}")
    c2.metric("Độ phủ sau hiệu chuẩn", f"{b['calibrated']:.0%}")
    c3.metric("Số mốc tự kiểm tra", b["n_origins_test"])
    st.caption("Tự kiểm tra trên các mốc cuối của lịch sử (hiệu chuẩn bằng các mốc đầu). Chỉ có ít mốc liên tiếp nên chỉ mang tính tham khảo.")

tabs = st.tabs([r.name for r in res])
for t, r in zip(tabs, res):
    with t:
        st.altair_chart(chart(r, capacity), width='stretch')
        st.caption("Đường xám: thực tế 7 ngày gần nhất; đường xanh: dự báo trung vị; vùng xanh: khoảng 90% đã hiệu chuẩn; nét đứt: khoảng chưa hiệu chuẩn; nét đỏ: ngưỡng dung lượng.")

with tempfile.TemporaryDirectory() as d:
    out = write_report(res, d, 0.10, capacity, horizon)
    html_bytes = open(out["report"], "rb").read()
    csv_bytes = open(os.path.join(d, "forecast.csv"), "rb").read()
c1, c2 = st.columns(2)
c1.download_button("Tải dự báo (CSV)", csv_bytes, "forecast.csv", "text/csv", width='stretch')
c2.download_button("Tải báo cáo (HTML)", html_bytes, "report.html", "text/html", width='stretch')

st.warning("Công cụ minh họa cho đồ án, không thay thế quyết định vận hành. Ở kỳ lễ hoặc giai đoạn bất thường, độ phủ của khoảng dự báo có thể thấp hơn mục tiêu 90%.")
