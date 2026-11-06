# Ứng dụng minh họa dự báo lưu lượng mạng di động

Ứng dụng Streamlit của đồ án: dự báo lưu lượng theo giờ bằng Chronos-2 (zero-shot), khoảng dự báo 90% hiệu chuẩn bằng CQR,
cảnh báo bất thường 24 giờ gần nhất và cảnh báo vượt ngưỡng dung lượng.

## Chạy cục bộ
```
pip install -r requirements.txt
streamlit run app.py
```
Lần chạy đầu tiên tải trọng số `amazon/chronos-2` (khoảng 0,5 GB) từ Hugging Face.

## Dữ liệu vào
Tệp CSV: cột đầu là thời gian, các cột sau là lưu lượng (tối đa 20 cột). Mỗi chuỗi cần ít nhất 14 ngày dữ liệu liên tục theo giờ ở cuối.
Dữ liệu mẫu: `demo/sample_data/slovak_lte1800_5sectors.csv` (5 sector từ bộ dữ liệu Slovakia, CC-BY 4.0).

## Triển khai trên Streamlit Community Cloud
1. Đẩy thư mục này lên một kho GitHub (tệp chính: `app.py`).
2. Vào https://share.streamlit.io, chọn *Create app*, trỏ tới kho đó và tệp `app.py`.
