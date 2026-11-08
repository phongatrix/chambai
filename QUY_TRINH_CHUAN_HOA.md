# QUY TRÌNH CHUẨN HÓA ẢNH PHIẾU TRẮC NGHIỆM — TỔNG QUÁT

> Phiên bản: 2026-11-07 · Áp dụng cho **mọi mẫu giấy** có: (1) mốc định vị
> hình chữ nhật/vuông đen, (2) lưới ô bubble (cells), (3) tùy chọn: dải màu
> in sẵn. Cài đặt: `omr_prep.py` (+ `omr_enhance`, `omr_deblur`, `omr_sheetgrid`,
> `omr_coloralign`, `omr_colorgrid`, `omr_aiassist`).

---

## 0. MỤC TIÊU & ĐẦU RA

| | |
|---|---|
| **Đầu vào** | 1 ảnh chụp/JPG bất kỳ (EXIF, zoom, nghiêng, lề lớn, mờ, giấy cong) |
| **Đầu ra** | Ảnh warp đúng hệ tọa độ MẪU × `scale` (+ ảnh tô đậm bubble, IDs) |
| **Nguyên tắc** | "AI khởi đầu, HÌNH HỌC quyết định" — mọi kết quả phải qua cổng ring trung thực |
| **Định danh** | CV2 seed cố định mỗi ảnh (`setRNGSeed(20261104)`) → kết quả không trôi giữa các run |

---

## 1. SƠ ĐỒ TỔNG QUÁT (bậc thang 5 tầng + cổng)

```
┌─ TẦNG 1 ─ ảnh ENHANCE ────────────────────────────────────────────┐
│ decode → crop-lề*(tắt) → enhance → best_orientation(4 hướng)      │
│   └ mỗi hướng: align_photo (dải mốc → RANSAC cặp điểm → ICP)      │
│ warp(cỡ mẫu×scale) → refine lưới ×2 → verify(corner/lưới)         │
│ → [rescue: ring re-selection → SIFT → fine-rotation → band-rescue]│
│ → band-affine refine (cong) → RING GATE → CỔNG MÀU → OK?          │
├─ TẦNG 2 ─ RAW-RETRY ─ ảnh GỐC (không enhance) ─ chạy lại tầng 1 ──┤
├─ TẦNG 3 ─ grid_first_align (căn cloud ring, min_rate 0.45) ───────┤
├─ TẦNG 4 ─ _fallback_align: SIFT full-frame → grid_first(0.30)     │
│           → color-grid → chạy lại tầng 1 với init_H ──────────────┤
├─ TẦNG 5 ─ DocShadow DEWARP (làm phẳng giấy cong) → chạy lại T1 ───┤
└─ vẫn fail → trả FAIL + lý do thật (→ AI đọc trực tiếp nếu bật) ───┘
   (*crop-lề: mặc định TẮT — xem mục 6)
```

---

## 2. CHI TIẾT TỪNG BƯỚC

### B0. Decode — `load_bgr`
- EXIF transpose (ảnh điện thoại nằm ngang) → RGB→BGR
- Thu nhỏ cạnh dài về ≤ 2600 px (LANCZOS) — giảm chi phí, giữ đủ chi tiết

### B1. Tăng nét tự động — `omr_enhance.enhance_if_blurry`
| Điều kiện | Hành động |
|---|---|
| Laplacian ≥ 150 (moi: 300) | Nguyên vẹn (`enhanced=False`) |
| Laplacian < 150 | CLAHE(L) + unsharp mask |
| Sau enhance vẫn Lap < **40** (cực đoan) | **Bộ sàng deblur**: Wiener (Gaussian σ + đĩa) / Richardson-Lucy / DocShadow / combo — chọn theo **detectability** (số mốc+ring dò được) |

### B2. Căn mẫu — `best_orientation` + `align_photo`
1. Thử 4 hướng xoay (0/90/180/270; ảnh ngang ưu tiên 90/270 trước)
2. Mỗi hướng `align_photo`:
   - Dò ô đen vuông trên ảnh theo **12 dải kích thước** (18–110px, thử dải
     gần cỡ kỳ vọng `9 × approx` trước)
   - `_match_fiducials`: RANSAC trên CẶP ĐIỂM (bất biến xoay), seed cố định
     `20261106`, cap 20000 giả thuyết/lần gọi
   - Fallback khi không khớp: mở rộng anchor (min_area 10) → SIFT (`align_sift`)
   - ICP 4 vòng lặp: kéo H về lưới ô mẫu thật
3. Xác thực warp thử: 4 mốc góc phải đúng chỗ + mực 2–50%
4. Thu **≥ 2 ứng viên hướng** (để re-selection bằng ring khi cần)

### B3. Warp + Refine lưới
- `M = scale·I × H⁻¹` → warp về `mẫu × scale`
- `refine_grid_by_cells`: ICP lưới ô ↔ blob thật; `hit ≥ 0.5` → warp lại từ
  ảnh gốc (không warp chồng)

### B4. Verify + Rescue ladder (chỉ attempt 0)
Nhận ảnh khi **4/4 mốc đúng** HOẶC **grid_hit ≥ 0.50**. Nếu không:
1. **Ring re-selection**: chọn hướng có ring cao nhất trong các ứng viên
   (≥ 0.55) — chống nhầm hướng do phiếu tự-similar
2. **SIFT candidate**: `_sift_candidate` (6000 features, RANSAC 4.0)
3. **Fine rotation**: quay ±6/±12/±18/±24° căn lại từng góc
4. **BAND-RESCUE (mới)**: giấy cong không đều — lặp band-refine với số dải/
   ngưỡng nới dần; nếu `ring_part` sau refine **đồng đều** (`min ≥ 0.42`,
   `spread ≤ 0.22`) → warp đúng mẫu dù corner kém (mốc bị che/bong) → nhận

### B5. Hiệu chỉnh phiếu cong — `band_affine_refine` (chạy mỗi attempt)
- Dò ring trên warp (accept_filled — bắt cả bubble tô)
- Ghép ô mẫu ↔ ring (ngưỡng rộng 1.6×pitch để GHÉP; đo hit bằng ngưỡng
  **HẸP** 0.55×pitch để ĐÁNH GIÁ — bug đã sửa: đo bằng tol rộng làm
  min_gain vô hiệu)
- Chia `nbands` dải ngang × `nbands_x` cột (mặc định 2: trái/phải) →
  dịch trung vị (dx, dy) mỗi ô lưới → nội suy → `cv2.remap`
- Lặp tối đa 4 lượt (nbands/tol nới dần), dừng khi không cải thiện
- **Chỉ nhận khi hit tăng ≥ min_gain** — ảnh phẳng đi qua nguyên vẹn

### B6. RING GATE (cổng bám mẫu — quan trọng nhất)
| Mẫu | Chế độ | Ngưỡng cứng |
|---|---|---|
| 12 | ALL (224 ô mẫu) | ring ≥ **0.60** |
| moi | ALL (142 ô) | ring ≥ **0.50** |
| 10_11 | theo phần P1/P2/P3 | 0.40/0.15/0.15 + sàn tổng thể |
| Sàn tổng thể (part mode) | lưới KEYED ≤150 ô | ≥ **0.55**; lưới FULL | ≥ 0.32 |

- **RING MỀM (soft accept)**: `corner ≥ 0.75` VÀ `grid_hit ≥ 0.50` VÀ
  `rate ≥ 0.45` → nhận với cờ `ring_soft=True` (biến thể in/photocopy nhạt).
  Part-mode thêm điều kiện **đồng đều phần** `max−min ≤ 0.20` (chặn phiếu
  khác mẫu cùng khung mốc — moi-sheet chênh 0.33 bị chặn).
- **CỨU ring thấp**: thử đếm lại với `accept_filled=True` (bubble tô đậm
  vẫn là bằng chứng ring) — ghi cả 2 giá trị vào info.
- Rescue trước đó vẫn chạy trước khi quyết định.

### B7. Cổng màu (TÙY CHỌN — chỉ mẫu có dải màu in sẵn)
- `omr_coloralign.color_agreement`: chiếu band màu ẢNH (đo trên ảnh gốc)
  qua H ngược về hệ mẫu, so IoU với band MẪU
- `agreement < 0.40` → từ chối (chỉ khi CẢ mẫu và ảnh có màu)
- Ảnh không màu (moi, scan B/W) → tự bỏ qua

### B8. Kết thúc
- Ink gate 2–50% → `normalize_light` (cân sáng cục bộ) → tô đậm bubble
- Đọc SBD/Mã đề trên ảnh gốc (`omr_idscan.scan_ids`) → phục vụ chấm

---

## 3. BẬC THANG FALLBACK — `normalize_to_template`

| Tầng | Nội dung | Điều kiện |
|---|---|---|
| 1 | `_normalize_core` trên ảnh ENHANCE | luôn |
| 2 | `_normalize_core` trên ẢNH GỐC (`_no_enhance`) | tầng 1 fail VÀ đã enhance VÀ không có init_H |
| 3 | `grid_first_align` (căn cloud ring toàn trang, quét θ±14° × scale, min_rate 0.45) | tầng 2 fail |
| 4 | `_fallback_align`: **SIFT full-frame** (q≥0.30, không cần mốc góc) → **grid_first min_rate 0.30** → **color-grid**; H vẫn qua đủ gates | tầng 3 fail |
| 5 | **DocShadow dewarp** (`data/models/docshadow_sd7k.onnx`, 1600px, ~20s CPU) làm phẳng giấy cong → chạy lại tầng 1 trên ảnh đã phẳng | tầng 4 fail |
| — | vẫn fail → trả info tốt nhất + lý do thật | |

Mỗi tầng nhận kết quả chỉ khi **đủ ICP + verify + ring gate** → không bao giờ
nhận warp sai chỉ vì "đã cố cứu".

---

## 4. BẢNG THAM SỐ & NGƯỠNG TỔNG HỢP

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Decode | max_side | 2600 px |
| Enhance | BLUR_THR / moi / EXTREME | 150 / 300 / 40 |
| Align | MIN_ALIGN | 0.45 |
| Align | dải mốc | 18–110px, bước 8, break q≥0.90 |
| Align | _match_fiducials | dt≥0.40·max, tol_tight≥8px, cap 20000 hyp |
| Verify | corner tol | max(10, 1.2·9·scale) |
| Verify | mực | 2%–50% |
| Ring | kích thước ring | 6–60px; filled 6–20px |
| Ring | RING_MIN_PART | 12:ALL .60 · moi:ALL .50 · 10_11:.40/.15/.15 |
| Ring mềm | điều kiện | corner≥.75 & grid≥.50 & rate≥.45 & spread≤.20 (part) |
| Band | dải | nbands 10→18, tol 1.6→2.8, lặp ≤4, min_gain 0.008 |
| Dewarp | DocShadow | 1600px, float16 input của UVDoc ≠ dùng |
| Scale chuẩn hóa | moi / còn lại | 3.0 / 2.0 |

---

## 5. KILL-SWITCH MÔI TRƯỜNG

| Biến | Tác dụng |
|---|---|
| `OMR_DEBLUR=classic` | bỏ bộ sàng deblur, dùng RL cũ |
| `OMR_DEWARP=0` | tắt DocShadow dewarp (tầng 5) |
| `OMR_COLOR=0` | tắt cổng màu |
| `OMR_CROP=1` | BẬT crop-lề (mặc định TẮT — xem mục 6) |
| `OMR_PREP_PROFILE=1` | in ms từng stage ra stderr |

---

## 6. CROP-LỀ THEO KHUNG MẪU — THỬ NGHIỆM, ĐÃ TẮT

Ý tưởng: ảnh chụp có lề lớn + zoom khác nhau → cắt lề về đúng tỉ lệ mẫu
(lề mẫu × zoom từ bbox mốc). **Đã cài `crop_to_template_framing` nhưng MẶC
ĐỊNH TẮT** (`OMR_CROP=1` để bật).

⚠ **Bài học hồi quy 2026-11-07**: bật mặc định làm test11 11/12→**0/12**,
test12_20 20/20→**12/20**, audit lỗi. Nguyên nhân: bbox mốc percentile trên
ảnh chụp nhiễu → zoom SAI → cắt VÀO PHIẾU. Muốn bật lại an toàn cần dò
viền giấy chắc chắn hơn (ngưỡng sáng thích nghi) hoặc suy zoom từ H sau
align thành công.

---

## 7. KIỂM CHỨNG & HỒI QUY

| Lô | Mẫu | Kết quả |
|---|---|---|
| test11 (12) | 10_11 | 11/12 (121112 mờ cực độ — chụp lại) |
| test12 (5) | 12 | 5/5 |
| test12_20 (20) | 12 | 20/20 |
| Thu/11a8 (38) | moi | 38/38 |
| Thu/12a7 (43) | 12 | 43/43 |
| Music/temp/goc (7) | moi (tự nhận qua alt) | 7/7 |

Lệnh hồi quy chuẩn (phải XANH trước mọi commit):
```bash
.venv/bin/python validate_testdata.py   # test11 11/12 + test12_20 20/20 + SBD 17/18
.venv/bin/python test_sheetgrid.py      # 61/61 PASS
.venv/bin/python audit_logic.py         # A1–A7 ĐỨNG
```

---

## 8. CẠM BẪY ĐÃ GẶP (đừng lặp lại)

1. **Seed RNG**: mọi findHomography RANSAC phụ thuộc RNG toàn cục — seed
   mỗi ảnh + mỗi hàm matching, nếu không kết quả "trôi" giữa các run.
2. **Đo hit bằng tol RỘNG**: ngưỡng ghép cặp ≠ ngưỡng đánh giá — dùng tol
   rộng để đo hit làm min_gain vô hiệu (refine không bao giờ áp dụng).
3. **Crop theo bbox mốc heuristic**: mốc dò được trên ảnh nhiễu không đủ
   tin để suy zoom → cắt vào phiếu. Phải hồi quy ngay khi đổi ảnh đầu vào.
4. **Mốc ảnh ≠ bộ mốc mẫu**: ảnh mờ mất mốc đầu/cuối → bbox không tương
   ứng → mọi suy luận từ bbox đều sai.
5. **Cổng mềm phải có điều kiện phân biệt**: chỉ nới ngưỡng khi có bằng
   chứng ĐÚNG MẪU (ring đồng đều phần + color agreement nếu có màu) —
   moi-sheet cùng khung mốc 4 góc không được lách qua.
6. **Model dtype**: UVDoc input float16; DocShadow input float32 0-1
   (không phải -1..1). Kiểm tra IO bằng onnx.load trước khi viết glue.
7. **align_sift trả 2 giá trị** `(Hs, qs)` — unpack 3 sẽ raise bị try/except
   nuốt → phương án SIFT "chạy ngầm không bao giờ được dùng".
8. **Cập nhật ảnh đầu vào → hồi quy NGAY**: mọi thay đổi preprocessing
   (crop/resize/dewarp) bắt buộc chạy validate_testdata trước khi tin.
