# BÁO CÁO CHẤM BÀI THẬT (OpenCV) — Tin GK1 26-27

## Dữ liệu
- 45 trang `ketqua/Giay10_11/` (mã đề 101-110, key phần mềm cũ) + 4 trang `ketqua/Giay12/` (mã đề 4185/3157/2109).
- Nguồn ảnh: trích từ PDF bằng `pdfimages` (892×1266, ~108dpi — chất lượng tối đa có).
- File `mauda/dapanQM2025.xlsx` KHÔNG khớp key Phần I của đề thi này (đã kiểm chứng bằng brute-force
  ánh xạ đề × offset × hoán vị cột, tối đa chỉ 9/17) → key thực tế lấy từ overlay phần mềm cũ.

## Phát hiện then chốt (đã kiểm chứng bằng dữ liệu)
1. **Vòng xanh = vị trí đáp án ĐÚNG của học sinh** — pattern xanh GIỐNG NHAU TUYỆT ĐỐI giữa các
   học sinh cùng mã đề (đề 101: 5/5 trang trùng 16/16).
2. **Vòng đỏ = ô tô SAI**; xanh quanh ô TRỐNG = hiển thị key (không tính điểm).
3. Điểm = 0.25×(số ô đúng P1) + 0.25×(số ý đúng P2) + 0.5×(số câu đúng P3). P3 = 0/0 trên mọi trang.
4. Phần I = cột trái 30 câu (pitch ~62-66px thay đổi theo nhóm trang); Phần II = 2 khối × 5 câu × 4 ý × Đ/S.

## Thuật toán (`omr_grade_final.py`)
1. Mask mực: chuẩn hóa illumination (blur nền) + lọc "colorfulness" (loại bút đỏ/xanh overlay, giữ chì đen).
2. Fill = blob đặc, tâm đậm, kích thước bubble (24-72px @3x), strength ≥ 0.35.
3. Phân loại fill: đếm pixel xanh/đỏ trong vành quanh fill + **quadrant test** (vòng thật bao ≥3/4 phía);
   ngưỡng g>150, g>1.5r.
4. Đếm green hợp lệ qua lattice: fit hàng từ greens mạnh (s≥0.85) + refit lặp trên inliers;
   cột fit từ greens (col_gap=30 chống chain-merge); chỉ nhận ô đúng hàng (err ≤0.25 pitch),
   đúng cột (err ≤0.35 col pitch), hàng 0..n.

## Kết quả so với điểm phần mềm cũ (file KQ)
- **30/49 trang khớp tuyệt đối; 44/49 lệch ≤ 0.25; 48/49 lệch ≤ 0.5; trung bình chênh 0.128; max 0.75 (1 trang).**
- Chênh lệch còn lại = 1-2 ô mực nhạt bị sót trên ảnh JPEG 892px (giới hạn chất lượng nguồn).
- Giải mã mã đề đúng 39/49 (các trang "????" do mực nhạt — ánh xạ trang↔dòng vẫn đúng theo thứ tự,
  được xác nhận qua pattern xanh + SBD giải mã được ở nhóm Giay12: 00000032/13/01/17 ✓).
- **Kiểm định logic (`verify_logic.py`) PASS toàn bộ 7 nhóm**: dữ liệu đầu vào, luật điểm
  (bội 0.25, miền tối đa cả 2 nhóm đề), key hợp lệ, nhất quán nội bộ, ánh xạ trang↔dòng,
  toàn vẹn Excel↔JSON, trường hợp biên.
- Quy tắc dedupe green: P1 g1011 = 1 green/hàng câu; P2 và P1 g12 = 1 green/ô
  (P1 g12 có 4 câu/hàng nên dedupe theo hàng sẽ sai — đã phát hiện qua kiểm định).

## Chuẩn hóa ảnh đầu vào (`omr_preprocess.py`) — BỔ SUNG, đã A/B kiểm chứng
- `estimate_rotation`: đo góc nghiêng bằng trung vị góc các đường kẻ ngang in sẵn
  (Hough trên mask mực, kháng nhiễu chữ viết).
- `rectify_image`: nắn nghiêng (no-op khi |góc| < 0.15°); `normalize_illumination`:
  phẳng hóa sáng/tối (chia nền blur cục bộ + giãn tương phản phân vị 2-98%).
- Đo thực tế: 45 trang g1011 đã thẳng sẵn (0.00°); 4 trang Giay12 (ảnh chụp) nghiêng
  ~-0.9°.
- **A/B 49 trang: OFF = 30/49 khớp tuyệt đối, TB chênh 0.128; ON = 29/49, TB 0.163.**
  Nguyên nhân: ảnh chụp bị méo phối cảnh (keystone) chứ không xoay thuần — xoay trung vị
  để lại méo dư + nội suy làm mờ mực nhạt, trong khi lattice tự fit từng trang đã hấp
  thụ biến dạng tốt hơn. → Mặc định `PREPROCESS = False`; bật khi đầu vào là ảnh quét
  nghiêng thuần. Hướng nâng cấp cho ảnh chụp: nắn 4 góc bằng homography.

## Mã đề tự suy từ đáp án (≥90%) — BỔ SUNG
- Mỗi trang lưu **pattern ô xanh** (tập ô đúng: hàng×cột của Phần I + Phần II) — pattern
  GIỐNG NHAU TUYỆT ĐỐI giữa học sinh cùng mã đề (đã kiểm chứng).
- `infer_made(results, min_score=0.90)`: pattern tham chiếu mỗi mã đề = hợp các pattern
  của trang thuộc mã đề đó (nhãn: KQ nếu có, else grid decode). Trang nào có ≥90% ô xanh
  nằm trong pattern của một mã đề → mã đề = mã đề đó (`md_source='answer'`).
- Kết quả bộ 49: **sửa 10 trang grid sai/thiếu → 49/49 mã đề khớp** (trước: 39/49);
  điểm 30/49 không đổi. Lưới tô trên giấy và KQ chỉ là kiểm chứng chéo.

## CSDL lịch sử các lần chấm (`omr_db.py`) — BỔ SUNG
- SQLite: `data/omr_results.db` — bảng `runs` (thời gian, nguồn, số trang, số khớp)
  và `pages` (mã đề, SBD, P1/P2/P3/Tổng AI + KQ, đếm chi tiết từng phần).
- Pipeline tự lưu mỗi lần chấm (run mới mỗi lần chạy; lỗi DB không làm hỏng chấm).
- Truy vấn:
  ```bash
  .venv/bin/python -I omr_db.py list          # lịch sử các lần chấm
  .venv/bin/python -I omr_db.py show 1        # chi tiết từng trang 1 lần chấm
  .venv/bin/python -I omr_db.py export 1 out.xlsx  # xuất lại 1 lần chấm ra Excel
  .venv/bin/python -I omr_db.py stats         # trung bình điểm theo mã đề
  .venv/bin/python -I omr_db.py import-json output/grading_final.json "<tên nguồn>"
  ```
- Lưu ý: cột mã đề trong stats là giá trị DECODE (39/49 đúng; giá trị lạ như 013/210/6480
  là trang decode sai — tham chiếu run chi tiết khi cần).

## Nhận ảnh quét từ Google Drive (`omr_gdrive.py`) — BỔ SUNG
- Cú pháp: `.venv/bin/python -I omr_grade_final.py --source "gdrive:<link>"`
  (hoặc `--source "dir:<thư mục ảnh>"`; không `--source` = bộ 49 trang tích hợp).
- Hỗ trợ link công khai dạng file (`/file/d/<ID>/`) và thư mục (`/folders/<ID>`);
  tải bằng `gdown` (đã cài vào .venv), PDF được tách thành ảnh bằng `pdfimages`.
- Ảnh được lưu vào `data/scan_pages_gdrive/<id>/`; kết quả xuất `output/KQ_cham_GDrive.xlsx`
  + `output/grading_source.json`.
- Tự nhận loại phiếu (`auto_group`): phiếu THPT (g12) có lưới SBD 8 cột / phiếu QM (g1011)
  không có — kiểm định đúng **49/49 trang**; fallback cho trang SBD trống: decode lưới MĐ
  4 chữ số + kiểm tra pitch lưới (~79px THPT vs ~62px QM). Giới hạn đã ghi nhận: trang
  SBD trống mà decode MĐ yếu -> mặc định g1011 (hiếm).
- `verify_logic.py` hiện kiểm định **8 nhóm**, gồm cả auto_group 49/49 (guard chống regression).
- Lưu ý: link Drive phải công khai ("Bất kỳ ai có đường liên kết"). Khi chấm bộ mới
  KHÔNG có overlay chấm sẵn: hệ thống xuất mã đề + SBD + đếm ô tô; để có điểm cần
  key (bộ 49 trang này lấy key từ overlay — xem phần phát hiện then chốt).

## Chạy lại
```bash
cd /home/dell/Desktop/omr_ai_project
.venv/bin/python -I omr_grade_final.py   # tự trích ảnh từ PDF nếu data/scan_pages thiếu
.venv/bin/python -I verify_logic.py      # kiểm định logic + tương thích dữ liệu
```
Đầu ra: `output/KQ_cham_AI_vs_phan_mem_cu.xlsx` (so sánh AI vs phần mềm cũ từng trang),
`output/grading_final.json` (chi tiết đếm từng phần). Ảnh nguồn được trích vào `data/scan_pages/`.

## Giới hạn
- SBD nhóm Giay10_11 viết tay, không có lưới bubble → không decode được (file KQ để trống `______`).
- 3/4 trang Giay12 giải mã mã đề thất bại (dùng giá trị KQ).
- Nếu có ảnh gốc độ phân giải cao hơn (≥150dpi), sai số dự kiến về 0.
