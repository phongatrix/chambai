# LỊCH SỬ PHÁT TRIỂN — PROMPT & THUẬT TOÁN
## Hệ thống chấm trắc nghiệm bằng AI (dự án omr_ai_chambai)

Tài liệu này ghi lại **toàn bộ các yêu cầu (prompt)** đã dùng để xây dựng hệ thống,
theo thứ tự thời gian, và **thuật toán** được xây dựng/cải tiến ở từng bước — từ đầu
đến lúc hoàn chỉnh.

---

## PHẦN A — DÒNG THỜI GIAN CÁC PROMPT

### Giai đoạn 1 — Dựng nền tảng (bản omr_ai_project)

| # | Prompt (yêu cầu) | Kết quả xây dựng |
|---|---|---|
| 1 | Chạy dự án | Khởi động web Streamlit 8502, kiểm tra toàn bộ pipeline |
| 2 | "bỏ mục 'Tải ảnh lên'" | Bỏ tab upload cũ trong bản web đầu |
| 3 | "vẫn còn tình trạng xử lý chưa đúng… phân tích và sử dụng các kỹ thuật… xử lý triệt để" | Sửa lỗi đọc bubble, đồng bộ mẫu |
| 4 | "fix lỗi hình 1 — bỏ phần 'Bộ 49 trang mẫu'" | Bỏ tab demo, dọn giao diện |
| 5 | "vẫn còn tình trạng xử lý chưa đúng…" | Sửa tiếp lỗi đọc & bố cục |
| 6 | "bố trí lại các thành phần trong ảnh kết quả (mẫu THPT 12.png)…" | Bố cục P1/P2/P3 theo tiêu đề PHẦN |
| 7 | "bỏ chữ '(suy)', 'OPENCV'" | Gỡ nhãn khỏi tiêu đề/tab |
| 8 | "bỏ nội dung trong hình" (caption 49 bài) | Bỏ caption thống kê |
| 9 | "bố trí lại… vị trí ở đầu mũi tên đỏ" | Dời khối chữ sang phải theo mũi tên |
| 10 | "undo lại task trước đó" | Hoàn tác dịch phải |
| 11 | "bố trí lại… vị trí các phần nằm giống như trong hình" | Chốt bố cục theo ảnh mẫu |
| 12 | "vẫn còn đè lên nội dung ảnh — điều chỉnh lại" | Sửa chồng chữ |
| 13 | "P2 dời xuống vùng trống ngay bên dưới" | Dời P2 vào khoảng trắng |
| 14 | "vị trí P2 phải nằm ở vị trí trống tương tự như P1 và P3" | Thống nhất cả 3 dòng |
| 15 | "dời phần P2 xuống vị trí như hình" | Chốt vị trí cuối |
| 16 | "copy dự án hoàn chỉnh vào /home/dell/Documents/chambaiV3" | Copy + sync |

### Giai đoạn 2 — Chuẩn hoá & nhận dạng (bản omr_ai_chambai)

| # | Prompt | Thuật toán/kết quả |
|---|---|---|
| 17 | "chuẩn hóa hình ảnh phần SBD và MĐ" | Warp 4 mốc góc + ICP lưới + đọc bubble |
| 18 | "sau chuẩn hóa dùng hình này để chấm" | `omr_match` đối chiếu lưới đáp án |
| 19 | "bổ sung AI nhận dạng + tô đen bubble đã tô" | `mark_filled`: k-means 2 cụm + tô đen |
| 20 | "copy dự án vào chambaiV3" | Sync bản copy |
| 21 | "xử lý nhận dạng SBD trong NA_3157…" | Chấp nhận warp bằng ICP lưới đáp án |
| 22 | "xử lý lại bố trí P2/P3 trong 50120319" | Dò hàng tiêu đề trên ảnh thật |
| 23 | "nhận dạng bubble SBD kết hợp OCR chữ viết tay" | `omr_ocr` (Tesseract, thử nghiệm) |
| 24 | "copy dự án hoàn chỉnh vào chambaiV3" | Re-sync |

### Giai đoạn 3 — Hoàn thiện (bản omr_ai_chambai hiện tại)

| # | Prompt | Thuật toán/kết quả |
|---|---|---|
| 25 | "sửa đường dẫn 12.png/10_11.png thành tương đối /data/maugiay" | Project tự chứa mẫu + đáp án |
| 26 | "tích hợp GOOGLE DRIVE vào phần upload, bỏ tab GDrive" | `scan_links` nhập link thẳng vào luồng chấm |
| 27 | "đã dán link GDrive nhưng không chuẩn hóa" | Sửa gdown 6.x (bỏ `remaining_ok`) + báo lỗi chia sẻ |
| 28 | "xử lý nhận dạng SBD trong NA_3157/200621+200623" | Chấp nhận warp bằng ICP lưới đáp án, 20/20 ok |
| 29 | "xử lý lại bố trí P2/P3 trong 50120319" | Dò hàng tiêu đề trên ảnh thật |
| 30 | "copy dự án hoàn chỉnh vào chambaiV3" | Re-sync bản mới nhất |

---

## PHẦN B — THUẬT TOÁN CHI TIẾT TỪNG MODULE

### 1. `omr_prep.py` — Chuẩn hoá ảnh về đúng mẫu giấy

**Đầu vào:** ảnh chụp điện thoại/scan (bất kỳ góc, độ nghiêng, ánh sáng nào)
**Đầu ra:** ảnh warp về đúng khung mẫu giấy (mẫu × 2), đã chuẩn hoá sáng

Thuật toán 6 bước:
1. **EXIF transpose** — xoay theo metadata ảnh
2. **Tự xoay 4 hướng** (0/90/180/270) — chọn hướng cho quality cao nhất
   (8/10 ảnh điện thoại KHÔNG có EXIF orientation nên phải thử)
3. **Căn mốc**: 4 ô đen góc trang → homography → ICP tinh chỉnh (SIFT dự phòng)
4. **Warp về khung mẫu**: `M = S·H⁻¹` (scale ×2)
5. **Tinh chỉnh lưới ô đáp án bằng ICP**: ghép tâm ô mẫu ↔ vòng tròn in thật
   (RANSAC, 4 vòng lặp) — sửa sai số phối cảnh còn dư 10–20 px
6. **Chuẩn hoá sáng** (chia nền cục bộ 51×51 + kéo giãn)

**Kiểm chứng chấp nhận ảnh:** 4/4 mốc góc đúng chỗ + mật độ mực 2–50%
(nếu < 4/4 mà ICP hit ≥ 50 % tâm ô khớp lưới đáp án thì vẫn nhận).

### 2. `omr_idscan.py` — Đọc Số báo danh + Mã đề từ lưới bubble

**Nguyên lý:** hàng của lưới SBD/MĐ in sẵn số 0–9, nên "đọc digit" = tìm hàng
có mực đậm nhất trong cột.

Thuật toán:
1. Warp ảnh về hệ mẫu (dùng H từ bước chuẩn hoá)
2. **Matched filter vành khuyên** (annulus r = 0,34·pitch) → bản đồ đáp ứng
   tại tâm vòng tròn in
3. **Tinh chỉnh affine 4 tham số** (dx, dy, sx, sy) — tối đa hoá tổng đáp ứng
   trên toàn bộ 80 ô SBD / 40 ô MĐ, biên ±0,55 pitch (chống trôi nửa hàng)
4. **Bám vòng tròn thật**: cụm tâm vòng → neo hàng 0 + hồi quy lại bước
   (chống trôi 1–2 hàng trên bản photocopy)
5. **Chuẩn hoá nền mực in theo hàng**: trừ phân vị 25 theo cột — hàng in số
   nhiều nét (8, 6) đậm hơn hàng ít nét (1, 7)
6. **Quyết định**: vượt nền ≥ 0,05 + mực tuyệt đối ≥ 0,34 + điểm/nhì ≥ 1,03

Lưu ý các lần sai và cách sửa:
- Sai 1 hàng: do neo nhầm cụm nhiễu → neo vào cụm GẦN hàng danh nghĩa nhất
- Sai vì "in đậm" nhầm "đã tô": trừ nền theo hàng (phân vị 25)
- Đọc nhầm khi lưới lệch: bám vòng tròn thật thay vì lưới suy từ mẫu

### 3. `omr_sheetgrid.py` — Lưới ô đáp án + chấm đúng/sai

- `build_answer_grid(paper, template, key)` — dựng lưới (phần, câu, lựa chọn,
  toạ độ mẫu, đáp án) từ mẫu + key
- `read_cells` — đọc mực từng ô bằng **độ đậm trung bình** `(1−norm)/0,30`
  (đo "tỉ lệ điểm tối" sai với ảnh photocopy: ô tô thật chỉ 0,16–0,19)
- `decide` — trống/một đáp án/tô nhiều với **ngưỡng tự thích ứng theo ảnh**
  (trừ nền = trung vị toàn ảnh + biên)
- `grade` — so đáp án, tính P1/P2/P3
- `_find_section_rows` — dò 3 hàng tiêu đề "PHẦN…" trên ảnh (mực chỉ ở mép
  trái 1,5–9,5 %W, phần phải 12–95 %W trống) → để đặt chữ đỏ đúng chỗ

### 4. `omr_match.py` — Đối chiếu ảnh chuẩn hoá với lưới đáp án

1. Dựng lưới đáp án từ mẫu + đề
2. Chiếu lưới lên ảnh: `p_ảnh = p_mẫu × scale` (không cần homography — ảnh
   đã nằm trong hệ mẫu)
3. Đọc mực bằng `read_cells` với `H = scale·I`
4. Quyết định trống/một/nhiều (ngưỡng tự thích ứng)
5. So key → đúng/sai từng câu, P1/P2/tổng
6. `match_all` — chấm thử TẤT CẢ đề trong key, chọn đề nhiều câu đúng nhất
   (ràng buộc mạnh: đề đúng đạt ~100 %, đề khác chỉ ~30–60 %)

### 5. `omr_prep.mark_filled` — Nhận dạng AI + TÔ ĐEN bubble

- Mỗi ô chấm điểm "đã tô" (vành khuyên + tâm)
- **k-means 2 cụm** toàn trang phân TRỐNG / ĐÃ TÔ (AI không giám sát)
- Tô đen bán kính 0,62 × bán kính bubble (nằm gọn trong viền)
- Xuất `_todam.jpg` để kiểm tra bằng mắt

### 6. `omr_idscan.fallback_positions` — Khoanh SBD/MĐ khi không giải được

- Mỗi cột lấy ô đậm nhất, chấp nhận khi vượt 1,8 × trung vị cột
- Chỉ dùng để HIỂN THỊ (khoanh xanh), không dùng để chấm

### 7. `omr_ocr.py` — OCR chữ số viết tay (THỬ NGHIỆM, MẶC ĐỊNH TẮT)

- Dò 12 ô in sẵn (8 SBD + 4 MĐ) phía trên lưới ID
- Cắt ô → chuẩn hoá sáng → phóng ×8 → nhị phân thích ứng → xoá viền ô
  (flood-fill từ 4 góc) → largest contour → chữ số vuông → Tesseract psm 10
- **Kết quả:** chưa ổn định trên bản photocopy (ô trống bị đoán "7") →
  `ENABLED = False`, không nối vào chấm điểm. Cần huấn luyện CNN nhỏ
  (~30–50 mẫu cắt từ phiếu thật) để thay thế.

### 8. `omr_idalign.py` — Định vị riêng khối SBD/Mã đề bằng mốc chữ nhật

- Lấy khối SBD/MĐ trên ảnh mẫu làm khuôn, dò lại trong ảnh chuẩn hoá bằng
  **template matching trên ẢNH BIÊN** (Canny + `TM_CCORR_NORMED`, quét tỉ lệ
  0,92–1,08 và góc ±4°)
- Khớp xám trực tiếp chỉ đạt 0,35 — khớp biên đạt **0,894**
- Trả độ dời (dx, dy) để dịch ảnh → lưới ID bám đúng hàng/cột

### 9. `omr_gdrive.py` — Lấy ảnh từ Google Drive

- `parse_gdrive_link` — nhận link thư mục / file / `dir:<thư mục>`
- `fetch_gdrive` — `gdown.download_folder` / `gdown.download` (đã sửa cho
  gdown 6.x: bỏ kwarg `remaining_ok` bị loại bỏ)
- `prepare_pages` — copy ảnh, PDF → tách trang (`pdfimages`/`pdftoppm`)
- `scan_links` — nhiều link một lần, báo cáo từng link

### 10. `omr_idscan`/`omr_gdrive` — ràng buộc mã đề theo key

- Mã đề đọc từ lưới phải có trong key; nếu không, thử TẤT CẢ đề trong key
  trên ma trận điểm và chỉ đổi khi: biên đủ rõ (≥ 0,12), có mực thật
  (≥ 0,30), và không tệ hơn hẳn cách đọc tự do (≥ 0,80×) — chống "ép" đề sai

---

## PHẦN C — CÁC LỖI ĐÃ TÌM VÀ SỬA (bài học)

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| Ô tô bị đọc "trống" trên ảnh photocopy | thang đo "tỉ lệ điểm < 0,78" không hợp với bút chì mờ (0,85–0,95) | đổi sang độ đậm trung bình `(1−norm)/0,30` |
| Lưới ô lệch 10–20 px sau warp | phối cảnh/méo ống kính còn dư | ICP lưới ô đáp án (ghép tâm ô ↔ vòng thật, RANSAC) |
| SBD/MĐ đọc sai 1 hàng | neo nhầm cụm nhiễu | neo cụm GẦN hàng danh nghĩa nhất, chấm điểm theo số cụm khớp |
| "In đậm" nhầm "đã tô" | hàng in số nhiều nét đậm hơn | trừ nền theo hàng (phân vị 25) |
| homography suy biến (ảnh trắng) vẫn qua kiểm tra | corner check 4/4 không đủ | thêm kiểm chứng ICP lưới đáp án (hit ≥ 50 %) |
| gdown 6.x lỗi `remaining_ok` | API thay đổi | gọi không kwarg, có dự phòng |
| link Drive 404 | thư mục chưa chia sẻ công khai | báo lỗi rõ trong UI + hướng dẫn chia sẻ |
| P2 đè bubble | vị trí cố định đo từ ảnh khác | dò hàng tiêu đề trên ảnh thật |
| 2 cột SBD trùng toạ độ | cụm bị sót | làm đều cột theo bước trung vị |
| `cham_theo_mau.py` bị ghi đè nhầm | patch script viết sai biến | khôi phục từ bản copy, rút kinh nghiệm patch |

---

## PHẦN D — KIỂM THỬ & ĐỒNG BỘ

- **`test_sheetgrid.py`**: 61 kiểm tra (lưới mẫu, lưới ID, ghép key, quy tắc
  trống/một/nhiều, công thức điểm, 3 màu, chặn homography suy biến, chịu
  hàng sót, chấm đầu-cuối, từ chối ảnh lạ) — **61/61 PASS**
- **`verify_logic.py`**: 8 nhóm giả định trên bộ 49 trang — **PASS toàn bộ**
- **AppTest web**: 0 exception
- **Đồng bộ**: `sync_all.sh` copy code + key + CSDL sang 3 bản copy
  (`omr_ai_project`, `omr_ai_project (2)`, `/home/dell/Documents/chambaiV3`)

---

## PHẦN E — VỊ TRÍ DỰ ÁN

| Bản | Đường dẫn | Vai trò |
|---|---|---|
| Gốc (đang chạy web 8502) | `/home/dell/Desktop/omr_ai_chambai` | Nguồn đồng bộ |
| Copy | `/home/dell/Documents/chambaiV3` | Copy đầy đủ (kèm .venv) |
| Copy | `/home/dell/Desktop/omr_ai_project` | Copy |
| Copy | `/home/dell/Desktop/omr_ai_project (2)` | Copy |
