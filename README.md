# Hệ Thống Chấm Trắc Nghiệm Bằng AI (OpenCV) — GÓI CHẠY LOCAL

Chấm phiếu trả lời trắc nghiệm từ ảnh quét: chuẩn hóa ảnh → tự nhận loại phiếu
→ phát hiện ô tô → chấm điểm → đối chiếu KQ phần mềm cũ → xuất Excel + lưu CSDL lịch sử.

**Đã kiểm chứng trên 49 bài thật (Tin GK1 26-27): 30/49 khớp tuyệt đối với phần mềm cũ,
48/49 lệch ≤ 0.5 điểm, trung bình chênh 0.128.**

## Web UI (giao diện trình duyệt)

```bash
bash run_web.sh           # mở http://127.0.0.1:8501 (hoặc: bash run_web.sh 8502)
```

4 tab:
- **Bộ 49 trang mẫu** — bảng kết quả so KQ + tải Excel + nút chấm lại
- **Tải ảnh lên** — chọn nhiều ảnh jpg/png, chấm ngay, loại phiếu tự nhận, tải Excel
- **Google Drive** — dán link công khai (ảnh/PDF/thư mục), tải về + chấm
- **Lịch sử CSDL** — mọi lần chấm đã lưu, xem chi tiết từng trang

## Mã đề tự suy từ đáp án (≥90%)

Mã đề được suy từ **pattern ô xanh** (vị trí các ô tô đúng — giống hệt nhau giữa
các học sinh cùng mã đề, đã kiểm chứng): nếu ≥90% ô xanh của trang nằm trong pattern
tham chiếu của một mã đề → nhận mã đề đó (cột "Nguồn mã đề" = `answer`). Lưới mã đề
tô trên giấy và mã đề trong KQ chỉ là kiểm chứng chéo. Trên bộ 49 trang: **49/49 mã đề
đúng** (grid decode chỉ 39/49 — 10 trang được suy/sửa từ đáp án).

## Chạy nhanh

```bash
cd omr_ai_chambai
bash setup.sh          # 1 lần: cài Python 3.12 + .venv + thư viện
bash run_cham_bai.sh   # chấm bộ 49 trang tích hợp + kiểm định + lịch sử
```

Kết quả nằm trong `output/`:
- `KQ_cham_AI_vs_phan_mem_cu.xlsx` — so sánh AI vs phần mềm cũ từng trang
- `KQ_cham_GDrive.xlsx`, `grading_source.json` — khi chấm nguồn bên ngoài
- `data/omr_results.db` — CSDL SQLite mọi lần chấm

## Chấm từ nguồn khác

```bash
# Link Google Drive (công khai): 1 ảnh / 1 PDF / cả thư mục
bash run_cham_bai.sh "gdrive:https://drive.google.com/drive/folders/<ID>"

# Thư mục ảnh local
bash run_cham_bai.sh "dir:/duong/dan/anh"

# Chấm lại bộ tích hợp
bash run_cham_bai.sh
```

Ảnh/PDF tải về nằm ở `data/scan_pages_gdrive/<id>/`; PDF vector được render bằng
pdftoppm; loại phiếu được tự nhận (lưới SBD 8 cột = THPT / không có = QM,
fallback: decode lưới mã đề + pitch).

## Truy vấn CSDL

```bash
.venv/bin/python -I omr_db.py list              # các lần chấm
.venv/bin/python -I omr_db.py show 1            # chi tiết từng trang
.venv/bin/python -I omr_db.py export 1 out.xlsx # xuất 1 lần chấm
.venv/bin/python -I omr_db.py stats             # trung bình điểm theo mã đề
```

## Kiểm định logic (chạy tự động trong run_cham_bai.sh)

```bash
.venv/bin/python -I verify_logic.py   # 8 nhóm kiểm định: luật điểm, key, nhất quán,
                                      # ánh xạ trang↔dòng, toàn vẹn Excel, auto_group
```

## Chấm theo MẪU GIẤY + LƯỚI ĐÁP ÁN (fiducial)

Quy trình 6 bước, đúng thứ tự yêu cầu: chọn mẫu giấy → nạp đáp án → dựng lưới →
ảnh khớp mẫu bằng ô đen định vị → đối chiếu bubble → tô màu + điểm.

| Mẫu giấy | File đáp án | Cấu trúc |
|---|---|---|
| `maugiay/12.png` (THPT) | `dapan/k12dapan.xlsx` | PHẦN I 40 câu (4 khối × 10) · PHẦN II 8 câu Đ/S (2 hàng × 4) · PHẦN III 6 câu số |
| `maugiay/10_11.png` (QM) | `dapan/k11dapan.xlsx` | PHẦN I 30 câu cột trái (3 khối × 10) · PHẦN II 10 câu Đ/S (2 hàng × 5) · PHẦN III 10 câu số |

Key Smarttest ghi theo câu gốc; hệ thống tự nối: key câu 1..N → PHẦN I, key N+1.. → PHẦN II
(ví dụ k12: 24 câu ABCD + câu 25-28 Đ/S → PHẦN II câu 1-4 trên phiếu).

**Ảnh kết quả (thiết kế theo mẫu phần mềm cũ):**
- Khối chữ **ĐỎ cỡ lớn** ở góc trái trên:
  ```
  P1: 24/24 = 6.00
  P2: 4/4; 4/4; 4/4; 4/4 = 4.00      (điểm từng câu Đ/S)
  P3: 0/0 = 0.00
  Tong: 10.00
  ```
- Vòng **XANH ĐẬM** cỡ lớn quanh bubble tô **ĐÚNG**; **ĐỎ** quanh bubble tô **SAI**;
  **VÀNG** cho câu **trống** hoặc **tô nhiều đáp án**.
- Vòng xanh cả **chữ số SBD / Mã đề** đã chọn trên lưới ID (khi đọc được).

**Căn khớp ảnh với mẫu (3 tầng, tự động):**
1. Phát hiện ô đen vuông in trên phiếu (mask mực + `MORPH_OPEN` bỏ nét mảnh + lọc vuông đặc).
2. Lấy 4 ô đen ở 4 góc → homography; kiểm tra hợp lệ (không suy biến, đúng tỉ lệ khung).
3. Tinh chỉnh **ICP** (ghép tâm bubble mẫu ↔ blob ảnh) + fallback **SIFT** khi thiếu mốc.
Chất lượng khớp = tỉ lệ ô đáp án chiếu đúng vào bubble trong ảnh (≥0.70 mới chấm).

**Chạy CLI:**
```bash
# chấm 1 cặp mẫu + key, tự chọn mã đề
.venv/bin/python cham_theo_mau.py --template 12 \
    --key /home/dell/Documents/tulieuchambai/dapan/k12dapan.xlsx \
    --images /home/dell/Documents/tulieuchambai/Test --out output/theo_mau

# tự nhận mẫu cho từng ảnh (12.png hoặc 10_11.png)
.venv/bin/python cham_theo_mau.py --auto --out output/theo_mau --json output/theo_mau/ket_qua.json
```

## ⓪ Tiền xử lý: đưa ảnh về ĐÚNG CHUẨN MẪU (`omr_prep.py`)

Mọi ảnh upload / Google Drive / chụp điện thoại được chuẩn hoá TRƯỚC khi chấm:

| Bước | Việc làm | Vì sao cần |
|---|---|---|
| 1 | **Xoay theo EXIF** (PIL `exif_transpose`) | ảnh điện thoại hay bị lưu "nằm ngang" |
| 2 | **Thử 4 hướng xoay 0/90/180/270**, chọn hướng khớp mẫu nhất | nhiều máy không ghi EXIF; đo trên bộ Test11: **8/10 ảnh chỉ khớp sau khi xoay 270°** (quality 0,5–0,7 → 1,00) |
| 3 | **Căn mẫu** bằng 4 ô đen định vị + ICP (SIFT dự phòng) | khử nghiêng + phối cảnh |
| 4 | **Warp về khung mẫu × scale 2** (`S·H⁻¹`) | đưa ảnh về cùng hệ toạ độ với 12.png / 10_11.png |
| 5 | **Kiểm chứng bắt buộc**: 4/4 ô định vị phải dò thấy trên ảnh warp + mật độ mực 2–50% | chống homography suy biến (ảnh warp trắng trơn vẫn có thể báo quality 1,0) |
| 6 | **Chuẩn hoá sáng** (chia nền cục bộ) | bóng đổ, chụp thiếu sáng |

Ảnh không qua được bước 5 bị **báo lỗi rõ ràng**, không bao giờ chuẩn hoá bừa.
Trong web: tab 🟩 có mục *“⓪ Xem trước bước chuẩn hoá ảnh theo mẫu”* (xem ảnh gốc /
ảnh chuẩn hoá cạnh nhau + tải ZIP), và ô tick **“⓪ Chuẩn hoá ảnh về đúng mẫu trước khi chấm”**
(mặc định BẬT). SBD/Mã đề vẫn được đọc trên ảnh gốc (sau khi xoay) để giữ độ phân giải.

### Nhận dạng AI + TÔ ĐEN bubble đã tô (trong bước ⓪)

`omr_prep.mark_filled()` chạy sau khi warp, gồm 2 phần:

1. **Nhận dạng (không giám sát)** — mỗi ô lấy điểm mực bằng `read_cells`; nền của
   trang = **trung vị** toàn bộ ô (đa số là ô trống) + biên 0,05; trong **cùng một câu**,
   ô được coi là đã tô nếu điểm ≥ 75% ô đậm nhất của câu đó và ≥ nền. Nhờ vậy bắt được
   cả ô tô đậm (0,6) lẫn ô bút chì mờ (0,16) trên **cùng một trang** — phân 2 cụm
   k-means toàn trang sẽ nuốt mất nhóm mờ.
2. **Tô đen** — vẽ đĩa đen đúng tâm/bán kính bubble đã nhận dạng lên bản sao ảnh,
   xuất `<tên>_todam.jpg` để kiểm tra bằng mắt (không dùng để chấm).

Trên trang PXL_...120214113 (bút chì rất mờ): nhận đúng **27/28** ô đã tô, nhờ đó
P2 tăng từ 5/12 → **11/12**, tổng 5,00 → **6,75**.

### Ảnh riêng cho phần MÃ ĐỀ / SBD

`omr_prep.id_crop()` cắt đúng vùng SBD + Mã đề từ ảnh **đã chuẩn hoá**, phóng to 2× và
tăng tương phản (CLAHE) — hiển thị ngay trong bảng ⓪ để người chấm nhìn và đối chiếu
mã đề. Bubble SBD/Mã đề đã đọc được cũng được **tô đen** cùng các ô đáp án.

### Chấm điểm TRÊN ảnh đã chuẩn hoá

Chấm điểm (nút **▶️ Chấm theo mẫu**) chạy cả chuỗi trên chính ảnh vừa chuẩn hoá:
ô tô đọc bằng `H = scale·I` trên ảnh warp (sai số < 1 px), SBD/Mã đề đọc trên ảnh gốc
(giữ độ phân giải), rồi xuất SBD · mã đề · nguồn · P1/P2 · tổng · bảng từng câu ·
ảnh kết quả 3 màu · Excel.

## ② Đối chiếu ảnh đã chuẩn hoá với LƯỚI ĐÁP ÁN (`omr_match.py`)

Sau bước ⓪, ảnh nằm đúng hệ toạ độ mẫu nên **không cần homography** khi chấm:

| Bước | Việc làm |
|---|---|
| 1 | Dựng lưới đáp án từ mẫu + đề: mỗi ô = (phần, câu, lựa chọn, toạ độ mẫu, đáp án) |
| 2 | Chiếu lưới lên ảnh: `p_ảnh = p_mẫu × scale` (sai số < 1 px) |
| 3 | Đọc mực từng ô bằng `omr_sheetgrid.read_cells` với `H = scale·I` — giữ nguyên thang mực đã kiểm chứng |
| 4 | Quyết định trống / một đáp án / tô nhiều bằng **ngưỡng mực tự thích ứng** |
| 5 | So với key → đúng/sai từng câu, P1/P2/tổng, **bảng đối chiếu chi tiết** + ảnh 3 màu |

Đo **“bám lưới”**: tỉ lệ ô mà vị trí đọc được còn nằm trong 0,6 bán kính bubble so với
toạ độ mẫu. Dưới 60% → báo *“lưới không bám được vòng tròn”* thay vì chấm bừa.
Khi chưa biết mã đề, `match_all()` chấm thử **tất cả đề trong key** và chọn đề có nhiều
câu đúng nhất (đo trên Test11: đề đúng đạt 7,00 điểm, các đề khác 2,25–3,75 → tách rất rõ).

**Thành phần đi kèm:**
- **Đọc SBD / Mã đề** (`omr_idscan.py`) — 5 bước, chạy trên ảnh **đã warp về hệ toạ độ mẫu**
  (khử phối cảnh bằng 4 mốc định vị, phóng 4×):
  1. **Matched filter vành khuyên** → bản đồ đáp ứng tại tâm mọi vòng tròn in trên giấy.
  2. **Tinh chỉnh lưới** bằng mô hình affine 4 tham số (dx, dy, sx, sy) tối đa hoá tổng đáp ứng
     trên **toàn bộ** 80 ô SBD / 40 ô Mã đề (gộp nhiều ô nên bền với nhiễu cục bộ).
  3. **Bám vòng tròn thật:** dò contour tròn trong vùng ID, cụm hoá 1D rồi **neo hàng 0 vào cụm
     trên cùng** và dùng **bước thật** (median gap) — nhờ đó bản photocopy in lệch tỉ lệ
     (pitch 30 → 31,9 px) không bị trôi dẫn tới lệch cả lưới 1 hàng. Cột: chọn cửa sổ cụm x
     đồng đều nhất theo thứ tự trái→phải.
  4. **Chuẩn hoá nền mực in theo hàng** (phân vị 25 theo cột): hàng in chữ số nhiều nét (8, 6)
     đậm hơn hẳn hàng 1 nét (1, 7) — nếu không trừ nền thì "in đậm" bị nhầm thành "học sinh tô".
     Quyết định: vượt nền ≥ 0,05 + mực tuyệt đối ≥ 0,34 + điểm/nhì ≥ 1,03.
  5. **Ràng buộc mã đề theo key:** Mã đề chỉ có 4-13 giá trị hợp lệ, nên khi cách đọc tự do
     không nằm trong key, hệ thống thử **mọi đề trong key** trên ma trận điểm và chỉ đổi khi
     biên đủ rõ, có mực thật, **và** không tệ hơn hẳn cách đọc tự do (chống "ép" sang đề sai).
  Lưới: 12.png = SBD 8 cột + Mã đề 4 cột; 10_11.png = Mã đề 3 cột (SBD viết tay).
  Chỉ ghi SBD/Mã đề khi đọc đủ **và** chắc chắn; nếu không → `NA` (không đoán bừa).
  Mã đề chỉ được dùng để chấm nếu điểm của nó sát đề khớp nhất (kiểm tra chéo).
- **Đặt tên ảnh kết quả:** `<SBD>_<Mã đề>_<YYYYMMDD>_<HHMMSS>.jpg` (ngày/giờ từ EXIF,
  fallback mtime; trùng tên tự thêm `_2`, `_3`).
- **Excel 2 sheet:** `Tổng hợp` (SBD, mã đề, nguồn mã đề, khớp mẫu, P1/P2, tổng, số sai,
  trống/nhiều) và `Chi tiết` (từng câu: trạng thái đúng/sai/trống/tô nhiều, đáp án, bài làm).
- **CSDL SQLite:** mỗi lần chấm tự lưu thành một `run` (xem `omr_db.py list/show`).
- **Ngưỡng mực TỰ THÍCH ỨNG** (`omr_sheetgrid.adapt_thresh`): bút chì mờ / bản photocopy
  cho ô đã tô chỉ ~0,25-0,45 mật độ mực, ngưỡng cứng cũ 0,35 biến chúng thành "trống"
  (đo trên ảnh thật: **14/40 câu bị bỏ oan**, điểm 6,50 thay vì 9,50). Ngưỡng giờ lấy
  nền nhiễu = phân vị 65 của toàn bộ ô trong ảnh + biên 0,10, kẹp [0,10 … 0,40].
- **Kiểm thử tự động:** `test_sheetgrid.py` — 61 kiểm tra (lưới mẫu, lưới SBD/MĐ, ghép key,
  quy tắc trống/một/nhiều, công thức điểm, 3 màu, chặn homography suy biến, chịu hàng sót,
  chấm đầu-cuối 2 ảnh chuẩn, từ chối ảnh lạ). Chạy: `.venv/bin/python test_sheetgrid.py`.

**Trên web:** tab **🟩 Chấm theo mẫu (fiducial)** — chọn mẫu, trỏ file đáp án, tải ảnh,
xem bảng điểm (kèm SBD, mã đề, nguồn), chi tiết từng câu + tải ZIP ảnh 3 màu.

**Đã kiểm chứng (2026-11-02):**
- `Test/IMG_20261030_042800.jpg` + `12.png` + đề 3157 → 24/24 P1, 16/16 P2, **10.0 điểm**;
  đọc đúng **SBD 50120074** và **Mã đề 3157** từ lưới, vẽ vòng xanh lên đúng 8+4 chữ số
  (ảnh đối chiếu: mọi vòng xanh trùng khít bubble tô).
- `Test/IMG_20261030_042144.jpg` + `10_11.png` + đề 132 → 16/16 P1, 12/12 P2, **7.0 điểm**.
- Kiểm thử nhánh sai: chấm ảnh 3157 bằng key 4185 → 19 câu đỏ đúng vị trí.
- Kiểm thử nhánh vàng: tô thêm 1 đáp án vào câu 1 → câu 1 vàng, điểm 10.0 → 9.75.

## Cấu trúc

```
omr_ai_chambai/
├── setup.sh, run_cham_bai.sh      # cài môi trường / chạy chấm
├── omr_grade_final.py             # pipeline chính (CLI --source)
├── omr_fiducial.py                # phát hiện mốc đen + bubble + homography
├── omr_sheetgrid.py               # lưới mẫu → lưới đáp án → đọc → 3 màu
├── cham_theo_mau.py               # CLI chấm theo mẫu giấy + key
├── webapp.py                      # web Streamlit (7 tab)
├── omr_grade_v3.py, omr_grade_v5.py   # helpers (mask, fills, decode, lattice)
├── omr_preprocess.py              # chuẩn hóa ảnh: đo nghiêng (Hough), nắn, phẳng sáng
├── omr_gdrive.py                  # nhận ảnh từ Google Drive (gdown + pdfimages/pdftoppm)
├── omr_db.py                      # CSDL SQLite lịch sử các lần chấm
├── verify_logic.py                # kiểm định logic/tương thích dữ liệu
├── data/
│   ├── scan_pages/                # 49 ảnh nguồn (g1011-000..044, g12-000..003)
│   ├── tulieu/                    # đáp án + KQ phần mềm cũ (tự chứa)
│   └── omr_results.db
└── output/                        # kết quả Excel/JSON/ảnh overlay
```

## Thuật toán (tóm tắt)

1. **Chuẩn hóa**: đo góc nghiêng bằng trung vị các đường kẻ ngang (Hough trên mask mực);
   phẳng hóa sáng/tối (chia nền blur cục bộ). Mặc định nắn nghiêng OFF (A/B trên 49 trang:
   ảnh chụp méo phối cảnh — nắn thuần làm xấu kết quả; bật khi quét nghiêng thuần).
2. **Mask mực**: tối hơn nền cục bộ + lọc "colorfulness" (loại bút đỏ/xanh overlay).
3. **Fill**: blob tròn đặc, tâm đậm, kích thước bubble; phân loại xanh/đỏ bằng quadrant
   test (vòng thật bao ≥3/4 phía).
4. **Đếm điểm**: lattice tự fit từng trang (refit lặp) + dedupe theo hàng câu/ô;
   điểm = 0.25×(P1 đúng) + 0.25×(ý P2 đúng) + 0.5×(câu P3 đúng).
5. **Key**: đề 101-110 của bài này KHÔNG khớp `dapanQM2025` (đã kiểm chứng) — key lấy
   từ overlay phần mềm cũ (vòng xanh = ô tô đúng, giống nhau tuyệt đối giữa HS cùng mã đề).

## Giới hạn đã ghi nhận

- SBD phiếu QM viết tay → không decode được (cột SBD để trống).
- 1/49 trang lệch 0.75 (mực nhạt ở ảnh 892px — giới hạn chất lượng nguồn).
- Chấm theo mẫu: ảnh phải thấy rõ **4 ô đen định vị** (hoặc đủ đặc trưng để SIFT khớp).
  Ảnh chụp nghiêng mạnh, mất góc, thiếu sáng → hệ thống báo "không khớp mẫu", không chấm bừa.
- Lô `Test/IMG_20261102_*` chụp nghiêng/photocopy mờ: 7/10 ảnh khớp mẫu tốt; các ảnh còn
  lại cần chụp thẳng, đủ 4 góc. Lô `Test/IMG_20261030_*` (ảnh chuẩn) khớp 8/10.
- Phiếu biến thể (PHẦN I 20 câu, PHẦN II 4 câu) vẫn chấm đúng các câu có trong key vì
  lưới đáp án chỉ dùng các ô keyed; các phần in thêm trên phiếu bị bỏ qua.
- Chấm bộ MỚI không có overlay cần key: hiện xuất mã đề/SBD/đếm ô; chấm điểm cần
  key đáp án đúng định dạng (câu: A-D; ý: Đ/S) hoặc bộ overlay chuẩn hóa.
