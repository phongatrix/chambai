# -*- coding: utf-8 -*-
"""
OMR PREP — Chuẩn hoá ảnh đầu vào về ĐÚNG CHUẨN MẪU GIẤY
========================================================
Mọi ảnh upload / lấy từ Google Drive / chụp điện thoại đều được đưa về cùng
một hệ toạ độ với ảnh mẫu (12.png hoặc 10_11.png) TRƯỚC khi chấm.

Quy trình 6 bước:
  1. Xoay theo EXIF (PIL exif_transpose) — ảnh điện thoại hay bị "nằm ngang".
  2. Thu nhỏ về cạnh dài ~1600 px để dò mốc nhanh và ổn định.
  3. Dò các Ô ĐEN VUÔNG in trên phiếu (mốc định vị) bằng mask mực chuẩn hoá
     sáng: g / blur(g,51) < 0.80, lọc theo độ đặc (fill) và tỉ lệ cạnh.
  4. Khớp mốc ảnh ↔ mốc mẫu theo VỊ TRÍ TƯƠNG ĐỐI trong khung mốc (không phụ
     thuộc tỉ lệ/tọa độ tuyệt đối), rồi RANSAC homography.
  5. Kiểm chứng: chiếu lại TOÀN BỘ mốc mẫu lên ảnh đã warp, đếm số mốc có ô
     đen thật tại đúng vị trí (dung sai 1.5 lần cạnh mốc). Chấp nhận khi
     ≥ 70% mốc khớp — nhờ vậy không bao giờ "chuẩn hoá bừa".
  6. Chuẩn hoá sáng (chia nền cục bộ + kéo giãn tương phản) rồi xuất ảnh
     kích thước = mẫu × scale.

Nếu bước 3-5 thất bại, hàm trả ok=False kèm lý do; nơi gọi quyết định bỏ qua
hoặc quay về căn chỉnh trực tiếp (align_photo).
"""
from __future__ import annotations

import cv2
import numpy as np
import sys
import os
import time

# ---- PROFILE TUỲ CHỌN: OMR_PREP_PROFILE=1 -> in ms từng stage ra stderr ----
# Mặc định TẮT (không tốn gì ngoài 1 phép so sánh bool). Chỉ đo thời gian,
# không đụng logic/RNG nên không đổi kết quả.
_PROFILE = os.environ.get("OMR_PREP_PROFILE") == "1"


def _prof_tick(t0):
    """Trả (ms, now) từ mốc t0; (None, None) khi profile tắt."""
    if not _PROFILE:
        return None, None
    now = time.perf_counter()
    return 1000.0 * (now - t0), now


def _prof_log(stage, ms):
    if ms is not None:
        print(f"[profile] {stage:<14} {ms:7.0f} ms", file=sys.stderr)

MARKER_THR = 0.80        # ngưỡng mực (chuẩn hoá theo nền) cho ô đen in
MARKER_MAX_FRAC = 0.004  # diện tích tối đa của 1 mốc (tỉ lệ ảnh)
MIN_MARKERS = 4
MAX_MATCH_DIST = 0.13    # dung sai khớp theo toạ độ chuẩn hoá
MIN_VERIFY = 0.70        # tỉ lệ mốc phải khớp sau khi warp
MIN_ALIGN = 0.45         # sàn quality để ĐƯA ảnh tới bước kiểm chứng corner/lưới.
                         # Ảnh chụp gần/mốc mờ (bộ Test11) chỉ align được quality
                         # 0,5-0,7 — Gate 0,70 loại chúng TRƯỚC khi tới kiểm chứng
                         # 4 mốc góc / lưới ô, trong khi cách chuẩn hoá trước vẫn
                         # chấp nhận (corner-check trong best_orientation đã chạy).
                         # Dưới 0,45: homography quá yếu, bỏ luôn.

# SCALE CHUẨN HOÁ theo mẫu: ảnh warp = mẫu × scale. Bubble phiếu moi NHỎ hơn
# mẫu cũ (~40% — cùng khổ A4 nhưng chứa nhiều phần hơn) nên ở scale 2.0 đường
# kính vòng in chỉ ~6.6 px — XUYÊN SÀN ngưỡng 6 px của bộ dò ring
# (omr_sheetgrid._ring_centers), ring rate sụp từ 0.7-0.85 xuống 0.04-0.39
# (đo thật 2026-11-05). Scale 3.0 -> vòng ~9.9 px, dò chắc.
NORM_SCALE = {"moi": 3.0}


def norm_scale_for(paper, base=2.0):
    """Scale chuẩn hoá đề xuất cho 1 loại mẫu (mặc định 2.0 như cũ)."""
    return NORM_SCALE.get(paper, base)


# ---------- 1. đọc ảnh ----------

def load_bgr(path, max_side=2600):
    """Đọc ảnh + xoay theo EXIF + thu nhỏ (giữ tỉ lệ)."""
    from PIL import Image, ImageOps
    try:
        with Image.open(str(path)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            s = min(1.0, max_side / float(max(w, h)))
            if s < 1.0:
                im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
            arr = np.asarray(im)[:, :, ::-1].copy()
    except Exception:
        arr = cv2.imread(str(path))
        if arr is None:
            raise FileNotFoundError(path)
        h, w = arr.shape[:2]
        s = min(1.0, max_side / float(max(w, h)))
        if s < 1.0:
            arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return arr


# ---------- 1b. CẮT LỀ THEO KHUNG CHUẨN MẪU (2026-11-07) ----------

_FRAMING_CACHE = {}


def _framing_md5(img):
    import hashlib
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def crop_to_template_framing(img, tpl_path):
    """CẮT BỚT LỀ ẢNH CHỤP theo khung chuẩn của MẪU (yêu cầu user 2026-11-07).

    Ảnh chụp có LỀ LỚN quanh phiếu (nền bàn/giấy khác) làm warp toàn khung
    lệch tỉ lệ → chuẩn hóa cắt mất mép phiếu (bug thật: mất header "Lớp" +
    đáy Phần III → chấm sai). Ảnh chụp còn có ZOOM khác nhau (chụp gần/xa).

    Thuật toán (xử lý zoom ra/vào — user chỉ định):
      1. Lề CHUẨN của mẫu = khoảng cách từ mốc hình chữ nhật đen ngoài cùng
         đến mép ảnh mẫu (L/T/R/B qua template_markers).
      2. Dò mốc đen trên ẢNH GỐC → bbox mốc (percentile 2-98 chống nhiễu).
      3. HỆ SỤ ZOOM = (bbox mốc ảnh) / (bbox mốc mẫu) — riêng theo trục.
      4. Lề MỤC TIÊU mỗi cạnh = lề mẫu × zoom của trục đó; cắt từng cạnh
         ĐỘC LẬP về mức mục tiêu. GIỚI HẠN chống cắt vào phiếu: lề sau cắt
         ≤ 0.33 × kích thước phiếu (bbox mốc) của trục tương ứng.
    Ảnh đã khung chuẩn (lề ≤ mục tiêu mọi cạnh) → không cắt (no-op)."""

    if img is None or not getattr(img, "size", 0):
        return img
    if os.environ.get("OMR_CROP") == "0":
        return img
    ck = (_framing_md5(img), str(tpl_path))
    if ck in _FRAMING_CACHE:
        return _FRAMING_CACHE[ck]
    out = img
    try:
        h_img, w_img = img.shape[:2]
        tpl_im = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
        if tpl_im is None:
            _FRAMING_CACHE[ck] = img
            return img
        if tpl_im.ndim == 3 and tpl_im.shape[2] == 4:
            tpl_im = tpl_im[:, :, :3]
        Ht, Wt = tpl_im.shape[:2]
        # hướng ảnh phải khớp hướng mẫu (portrait↔portrait) mới đối chiếu được
        if (w_img > h_img) == (Wt > Ht):
            ms_t = template_markers(tpl_path)
            if len(ms_t) >= 4:
                xs_t = [m[0] for m in ms_t]
                ys_t = [m[1] for m in ms_t]
                mt_l, mt_r = min(xs_t), Wt - max(xs_t)
                mt_t, mt_b = min(ys_t), Ht - max(ys_t)
                bw_t = max(xs_t) - min(xs_t)
                bh_t = max(ys_t) - min(ys_t)
                if bw_t >= 10 and bh_t >= 10:
                    mk = detect_markers(img)
                    if len(mk) >= 6:
                        pxs = np.percentile([m[0] for m in mk], [2, 98])
                        pys = np.percentile([m[1] for m in mk], [2, 98])
                        bw_p = float(pxs[1] - pxs[0])
                        bh_p = float(pys[1] - pys[0])
                        if bw_p >= 20 and bh_p >= 20:
                            zx = bw_p / bw_t          # zoom riêng từng trục
                            zy = bh_p / bh_t
                            # lề mục tiêu = lề mẫu × zoom (xử lý zoom ra/vào)
                            tgt_l = mt_l * zx
                            tgt_r = mt_r * zx
                            tgt_t = mt_t * zy
                            tgt_b = mt_b * zy
                            # giới hạn: lề sau cắt ≤ 1/3 phiếu (chống cắt vào ô)
                            cap_x = 0.33 * bw_p
                            cap_y = 0.33 * bh_p
                            # lề sau cắt = MỤC TIÊU (lề mẫu × zoom), giới hạn:
                            # không vượt cap (chống cắt vào ô) và không vượt
                            # lề hiện tại (không thể "mở rộng" lề)
                            new_l = min(max(tgt_l, 0.0), cap_x, pxs[0])
                            new_r = min(max(tgt_r, 0.0), cap_x,
                                        w_img - pxs[1])
                            new_t = min(max(tgt_t, 0.0), cap_y, pys[0])
                            new_b = min(max(tgt_b, 0.0), cap_y,
                                        h_img - pys[1])
                            l = int(round(max(0.0, pxs[0] - new_l)))
                            t = int(round(max(0.0, pys[0] - new_t)))
                            r = int(round(min(w_img, pxs[1] + new_r)))
                            b = int(round(min(h_img, pys[1] + new_b)))
                            if (r - l) >= 200 and (b - t) >= 200:
                                removed = (w_img * h_img) - \
                                    ((r - l) * (b - t))
                                if removed >= 0.02 * w_img * h_img:
                                    out = img[t:b, l:r].copy()
                                    print(f"[crop] cắt lề theo khung mẫu: "
                                          f"{w_img}x{h_img} -> {r-l}x{b-t} "
                                          f"(L{l} T{t} R{w_img-r} B{h_img-b} "
                                          f"zoom={zx:.2f}/{zy:.2f})",
                                          file=sys.stderr)
    except Exception:                                        # noqa
        out = img
    if len(_FRAMING_CACHE) > 8:
        _FRAMING_CACHE.pop(next(iter(_FRAMING_CACHE)))
    _FRAMING_CACHE[ck] = out
    return out


# ---------- 1b. CĂN TRƯỚC BẰNG BIẾN ĐỔI TƯƠNG TỰ (2026-11-07) ----------

_PREALIGN_CACHE = {}


def prealign_similarity(img, tpl_path, scale=2.0):
    """Nắn ảnh gốc về KHUNG MẪU bằng similarity TRƯỚC khi căn mẫu chi tiết
    (yêu cầu user 2026-11-07: "zoom tỉ lệ, xoay nhẹ, cắt xén bớt ảnh gốc
    trước khi co giãn các mốc định vị"). TRẢ (canvas, A3) — A3 = similarity
    mẫu->ảnh gốc để hợp H về đúng hệ toạ độ ảnh; hoặc None.

    Vấn đề thật: homography 8 bậc tự do — khi mốc khớp hơi lệch (nhiễu
    threshold, cong cục bộ) H tự do KÉO DÃN nội dung. Đo trên 090151 (lô
    11a8): H cho scale_x 6.70 vs scale_y 7.23 (dị dạng ~9 %) và residual
    lưới ô sau warp median 9 px (template scale) dù corner-check 4/4.

    Giải pháp: fit BIẾN ĐỔI TƯƠNG TỰ (scale ĐỀU + xoay + tịnh tiến — 4 tham
    số, KHÔNG THỂ kéo dãn/khớp nghiêng) từ 4 ô đen GÓC lớn nhất ảnh ↔ mẫu,
    warp ảnh lên canvas đúng cỡ mẫu × scale (lề thừa bị CẮT XÉN tự nhiên).
    Sau đó ảnh đã "đúng khung" — homography phía sau chỉ phải sửa sai số
    nhỏ (vài %) nên không còn dư địa kéo dãn.

    Bền vững: pairing 4 góc theo CỰC TRỊ (x+y, x−y) giữ thứ tự vòng — chỉ
    lệch bội 90° (được best_orientation hấp thụ), không bao giờ LÂNG GƯƠNG.
    Trả None (bỏ qua, luồng gốc nguyên vẹn) khi: thiếu mốc, residual khớp
    quá lớn, thang phi lý. KILL-SWITCH: env OMR_PREALIGN=0."""
    if img is None or not getattr(img, "size", 0):
        return None
    tpl_im = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
    if tpl_im is None:
        return None
    Ht, Wt = tpl_im.shape[:2]
    ck = (_framing_md5(img), str(tpl_path), round(float(scale), 3))
    if ck in _PREALIGN_CACHE:
        return _PREALIGN_CACHE[ck]
    out = None
    try:
        # ---- 4 ĐỈNH KHUNG của MẪU: template_fiducials (hình chữ nhật LỚN
        # NHẤT có ô đen thật 4 góc — (12,12)-(450,646) với mẫu 12). KHÔNG dùng
        # "marker lớn nhất": mẫu 12 bubble SBD/MĐ 13.7px LỚN HƠN mốc góc →
        # dst4 thành cụm SBD nhỏ → similarity phóng 15× → canvas rác
        # (bug thật 2026-11-08). ----
        quad_t, _ms_all = template_fiducials(tpl_path)
        if len(quad_t) != 4:
            return None
        dst4_fixed = np.float32(quad_t)
        # ---- 4 ô đen GÓC trên ẢNH: lọc theo KÍCH THƯỚC MỐC KỲ VỌNG (2026-11-08)
        # (lọc "top lớn nhất" cũ dính BÓNG TỐI lớn trên ảnh chụp — blob 160-178px
        # tại đầu trang lấn át 4 mốc góc thật ~36px → prealign bị từ chối 8/10
        # ảnh lô 11a8). Kỳ vọng = mốc góc mẫu ~9.5px × zoom ảnh/mẫu; dải 0.35-2.8×
        # đủ rộng cho lề ảnh lớn, loại bóng/blobber siêu lớn và nhiễu nhỏ.
        # 4 GÓC CỰC TRỊ trong dải vẫn tự chọn đúng mốc (bubble nằm BÊN TRONG). ----
        mk = detect_markers(img)
        if len(mk) < 4:
            return None
        approx = img.shape[1] / max(1, Wt)          # zoom ảnh/mẫu gần đúng
        exp_sz = 9.5 * approx                       # cỡ mốc góc kỳ vọng (px)
        big_p = [m for m in mk
                 if 0.35 * exp_sz <= max(m[2], m[3]) <= 2.8 * exp_sz]
        if len(big_p) < 4:
            return None
        # pairing CỰC TRỊ (giữ chiều vòng — xem docstring): TL,TR,BR,BL
        def _quad4_extreme(lst):
            p = np.float32([[m[0], m[1]] for m in lst])
            s_, d_ = p[:, 0] + p[:, 1], p[:, 0] - p[:, 1]
            return p[[int(np.argmin(s_)), int(np.argmax(d_)),
                      int(np.argmax(s_)), int(np.argmin(d_))]]
        dst4 = dst4_fixed                     # khung mẫu (TL,TR,BR,BL)
        src4 = _quad4_extreme(big_p)          # toạ độ ẢNH
        # ---- fit SIMILARITY (mẫu -> ảnh): 4 tham số, không stretch ----
        A, _inl = cv2.estimateAffinePartial2D(dst4.reshape(-1, 1, 2),
                                              src4.reshape(-1, 1, 2))
        if A is None or not np.all(np.isfinite(A)):
            return None
        proj = cv2.transform(dst4.reshape(-1, 1, 2), A).reshape(-1, 2)
        res = np.linalg.norm(proj - src4, axis=1)
        msz = float(np.median([max(m[2], m[3]) for m in big_p]))
        # ---- CỔNG CHẤP NHẬN: phân biệt PHỐI CẢNH THẬT vs GHÉP SAI ----
        # Ảnh chụp nghiêng mạnh cho quad HÌNH THANG (mép trên/dưới chênh ~13%)
        # — similarity KHÔNG thể khớp hết, mốc xa lệch 10-20% đường chéo PHIẾU
        # (090116: 227-258px / diag 2680). Ghép SAI (lẩng gương/xoay 90°) cho
        # residual 30-60% diag. Ngưỡng 22% max / 12% median của diag QUAD + tỉ
        # lệ khung quad ảnh phải ≈ quad mẫu (±28%) — homography phía sau sửa
        # phần phối cảnh còn lại, prealign chỉ cần "đúng khung, không kéo dãn".
        qdiag = float(np.hypot(*(src4.max(0) - src4.min(0))))
        tdiag = float(np.hypot(*(dst4.max(0) - dst4.min(0))))
        qw = float(src4[:, 0].max() - src4[:, 0].min())
        qh = float(src4[:, 1].max() - src4[:, 1].min())
        tw = float(dst4[:, 0].max() - dst4[:, 0].min())
        th = float(dst4[:, 1].max() - dst4[:, 1].min())
        if qdiag <= 0 or qh <= 0:
            return None
        _ar_img, _ar_tpl = qw / qh, tw / th
        if not (0.72 * _ar_tpl <= _ar_img <= 1.28 * _ar_tpl):
            return None
        if float(res.max()) > max(40.0, 0.25 * qdiag, 1.6 * msz) \
                or float(np.median(res)) > max(28.0, 0.14 * qdiag):
            return None
        # ---- QUAD CHIẾU PHẢI NẰM TRONG ẢNH (2026-11-08, bug 12a7): ghép nhầm
        # 4 ô to/bóng → similarity phóng phiếu 15× (6780px > ảnh 1950px) mà
        # residual vẫn lọt cổng → canvas rác, pipeline q=0. Cản chặn tuyệt
        # đối: 4 đỉnh quad mẫu chiếu qua A phải nằm trong ảnh (±5 % lề). ----
        _pj = cv2.transform(dst4.reshape(-1, 1, 2), A).reshape(-1, 2)
        _ih, _iw = img.shape[:2]
        if (_pj[:, 0].min() < -0.05 * _iw or _pj[:, 0].max() > 1.05 * _iw
                or _pj[:, 1].min() < -0.05 * _ih
                or _pj[:, 1].max() > 1.05 * _ih):
            return None
        a, b = float(A[0, 0]), float(A[0, 1])
        sc = float(np.hypot(a, float(A[1, 0])))
        if not (0.05 <= sc <= 60.0):
            return None
        # ---- image -> canvas: nghịch đảo A × scale (canvas = mẫu × scale) ----
        inv = cv2.invertAffineTransform(A.astype(np.float32))
        C = inv.copy()
        C[:, :2] *= float(scale)
        C[:, 2] *= float(scale)
        Wc, Hc = int(round(Wt * scale)), int(round(Ht * scale))
        if Wc < 100 or Hc < 100:
            return None
        try:
            out = cv2.warpAffine(img, C, (Wc, Hc), flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        except cv2.error:                    # INTER_AREA không có ở bản cũ
            out = cv2.warpAffine(img, C, (Wc, Hc), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        print(f"[prealign] similarity mẫu->{Wc}x{Hc}: scale ảnh {sc:.2f}x, "
              f"xoay {np.degrees(np.arctan2(float(A[1, 0]), a)):.1f}°, "
              f"residual max {res.max():.0f}px",
              file=sys.stderr)
        # trả kèm A3 (mẫu -> ảnh) để wrapper HỢP H về hệ ẢNH GỐC: H фото =
        # A3 · D(1/scale) · H_core (canvas = khung mẫu × scale)
        A3 = np.eye(3, dtype=np.float64)
        A3[:2, :] = A
        out = (out, A3)
    except Exception:                                        # noqa
        out = None
    if len(_PREALIGN_CACHE) > 8:
        _PREALIGN_CACHE.pop(next(iter(_PREALIGN_CACHE)))
    _PREALIGN_CACHE[ck] = out
    return out


# ---------- 2. dò mốc ----------

def detect_markers(img):
    """Danh sách ô đen vuông đặc: [(cx, cy, w, h)]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bg = cv2.blur(g.astype(np.float32), (51, 51))
    norm = g.astype(np.float32) / (bg + 1e-6)
    dark = (norm < MARKER_THR).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    H, W = gray.shape
    amax = max(64, int(MARKER_MAX_FRAC * W * H))
    cs, hz = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    has_child = set()
    if hz is not None:
        for row in hz[0]:
            if row[3] != -1:
                has_child.add(row[3])
    out = []
    for i, c in enumerate(cs):
        if i in has_child:            # có lỗ bên trong -> không phải ô đặc
            continue
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if a < 25 or a > amax or w < 4 or h < 4:
            continue
        if a / (w * h) <= 0.72 or not (0.55 <= w / h <= 1.80):
            continue
        out.append((x + w / 2.0, y + h / 2.0, float(w), float(h)))
    out.sort(key=lambda z: -z[2] * z[3])
    dedup = []
    for m in out:
        if all(abs(m[0] - o[0]) > 2.5 or abs(m[1] - o[1]) > 2.5 for o in dedup):
            dedup.append(m)
    return dedup


_TPL_CACHE = {}

# cloud ring mẫu cho grid_first_align (xem docstring trong hàm)
_GF_TPL = {}


def template_markers(tpl_path):
    """Mốc trên ẢNH MẪU. Mốc mẫu nhỏ (~9 px) nên phóng 3x trước khi dò."""
    ck = ("mk", str(tpl_path))
    if ck in _TPL_CACHE:
        return _TPL_CACHE[ck]
    img = cv2.imread(str(tpl_path))
    if img is None:
        raise FileNotFoundError(tpl_path)
    h, w = img.shape[:2]
    for sc in (3.0, 4.0, 2.0, 1.0):
        big = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)
        ms = detect_markers(big)
        ms = [m for m in ms if (m[2] / sc) <= 40]
        if len(ms) >= MIN_MARKERS:
            out = [(m[0] / sc, m[1] / sc, m[2] / sc, m[3] / sc) for m in ms]
            _TPL_CACHE[ck] = out
            return out
    _TPL_CACHE[ck] = []
    return []


def _cluster_vals(vals, tol):
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


_TF_CACHE = {}


def template_fiducials(tpl_path):
    """BỐN ô định vị ở 4 góc của mẫu (cùng với danh sách đầy đủ để kiểm chứng).

    Tìm HÌNH CHỮ NHẬT LỚN NHẤT mà 4 đỉnh đều có ô đen thật:
      12.png  -> (12,12) (450,12) (450,646) (12,646)   [khung mốc mép trang]
      10_11   -> (46,103) (360,103) (46,546) (360,546) [khung trong]
    Cách này không phụ thuộc kích thước mốc (mẫu 12.png có ô 9 px lẫn ô 13 px).

    Dung sai cụm ADAPTIVE theo kích thước ảnh (mặc định 2.0 px chỉ đúng cho
    mẫu in sạch 426-600 px): mẫu moi.png dựng từ ẢNH CHỤP còn dư méo phối
    cảnh ±15 px (đã scale về 600 px vẫn còn ~5-6 px) — cụm y của mốc trái/
    phải không gộp vào nhau nếu giữ 2.0 px → chỉ tìm được khung TRONG.
    0.012×cạnh dài ≈ 7 px ở 600 px. Đã kiểm chứng: quad 12.png KHÔNG ĐỔI;
    10_11.png lệch 3 px trên cạnh trái (dưới ngưỡng corner-check 10-21 px) —
    validate_testdata là lưới an toàn cuối cùng.

    CACHE theo (path, mtime): hàm chỉ phụ thuộc file mẫu, nhưng best_orientation
    gọi theo TỪNG hướng xoay, _normalize_core gọi theo từng lượt verify,
    _fine_rotation_search gọi theo TỪNG góc — mỗi ảnh chuẩn hoá phải tính lại
    vô ích (imread + cụm hoá + quét hình chữ nhật) dù mẫu không đổi."""
    try:
        ck = (str(tpl_path), round(os.path.getmtime(str(tpl_path)), 3))
    except OSError:
        ck = (str(tpl_path), None)
    if ck in _TF_CACHE:
        return _TF_CACHE[ck]
    ms = template_markers(tpl_path)
    if len(ms) < MIN_MARKERS:
        out = ([], ms)
        _TF_CACHE[ck] = out
        return out
    img = cv2.imread(str(tpl_path))
    h, w = img.shape[:2]
    tol = max(2.0, 0.012 * float(max(h, w)))
    xs = _cluster_vals([m[0] for m in ms], tol=tol)
    ys = _cluster_vals([m[1] for m in ms], tol=tol)
    have = np.zeros((len(ys), len(xs)), bool)
    for m in ms:
        ix = int(np.argmin(np.abs(np.array(xs) - m[0])))
        iy = int(np.argmin(np.abs(np.array(ys) - m[1])))
        have[iy, ix] = True
    best = None
    for i1 in range(len(xs)):
        for i2 in range(i1 + 1, len(xs)):
            for j1 in range(len(ys)):
                for j2 in range(j1 + 1, len(ys)):
                    if not (have[j1, i1] and have[j1, i2]
                            and have[j2, i1] and have[j2, i2]):
                        continue
                    area = (xs[i2] - xs[i1]) * (ys[j2] - ys[j1])
                    if best is None or area > best[0]:
                        best = (area, i1, i2, j1, j2)
    if best is None:
        out = ([], ms)
        _TF_CACHE[ck] = out
        return out
    _, i1, i2, j1, j2 = best
    quad = [(xs[i1], ys[j1]), (xs[i2], ys[j1]), (xs[i2], ys[j2]), (xs[i1], ys[j2])]
    out = (quad, ms)
    _TF_CACHE[ck] = out
    return out


def _marker_corners(ms, shape):
    """4 ô đen ở 4 góc trên ẢNH CHỤP (theo 4 hướng cực trị)."""
    if len(ms) < MIN_MARKERS:
        return None
    h, w = shape[:2]
    # lọc mốc quá nhỏ (nhiễu) và quá lớn (mảng bóng đổ)
    sizes = np.array([max(m[2], m[3]) for m in ms], np.float32)
    med = float(np.median(sizes))
    keep = [m for m, sz in zip(ms, sizes) if 0.55 * med <= sz <= 2.6 * med]
    if len(keep) < MIN_MARKERS:
        keep = ms
    pts = np.array([[m[0], m[1]] for m in keep], np.float64)
    ssum, sdif = pts[:, 0] + pts[:, 1], pts[:, 0] - pts[:, 1]
    idx = [int(np.argmin(ssum)), int(np.argmax(sdif)),
           int(np.argmax(ssum)), int(np.argmin(sdif))]      # TL, TR, BR, BL
    if len(set(idx)) < 4:
        return None
    q = np.float32([[keep[i][0], keep[i][1]] for i in idx])
    # kiểm tra hình dạng hợp lý: không quá dẹt, chiếm phần lớn khung
    d01 = np.linalg.norm(q[1] - q[0])
    d12 = np.linalg.norm(q[2] - q[1])
    if min(d01, d12) < 0.25 * min(w, h) or max(d01, d12) > 1.6 * max(w, h):
        return None
    return q


def paper_quad(img):
    """4 góc TỜ GIẤY (quad lớn nhất, tỉ lệ ~A4 dọc). None nếu không thấy."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    min_area = 0.15 * w * h
    for c in cs:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        q = approx.reshape(4, 2).astype(np.float32)
        q = _order_quad(q)
        wa = np.linalg.norm(q[1] - q[0])
        ha = np.linalg.norm(q[3] - q[0])
        if ha <= 1e-6:
            continue
        ar = wa / ha
        if not (0.45 <= ar <= 1.05):     # phiếu dọc (A4 ~0,71)
            continue
        if best is None or a > best[0]:
            best = (a, q)
    return None if best is None else best[1]


def _order_quad(q):
    """Sắp 4 điểm: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    q = np.asarray(q, np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    idx = np.argsort(ang)
    q = q[idx]
    # bắt đầu từ điểm trên-trái (góc phần tư thứ nhất âm)
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    q = np.roll(q, -start, axis=0)
    if np.linalg.norm(q[0] - q[1]) > np.linalg.norm(q[3] - q[0]):
        pass
    return q.astype(np.float32)


def _refine_by_markers(img, H, scale, tpl_ms, Wt, Ht):
    """Tinh chỉnh H bằng cách ghép mốc mẫu ↔ ô đen thật trên ảnh đã warp."""
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    w1 = cv2.warpPerspective(img, S @ H, (int(Wt * scale), int(Ht * scale)),
                             flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    wms = detect_markers(w1)
    if len(wms) < 6:
        return H, w1, 0
    wx = np.array([m[0] for m in wms], np.float64)
    wy = np.array([m[1] for m in wms], np.float64)
    src, dst = [], []
    tol = 0.06 * Wt * scale
    for m in tpl_ms:
        px, py = m[0] * scale, m[1] * scale
        d = np.hypot(wx - px, wy - py)
        if d.size:
            j = int(np.argmin(d))
            if d[j] <= tol:
                src.append([wx[j], wy[j]])
                dst.append([px, py])
    if len(src) < 6:
        return H, w1, len(src)
    S2 = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]], np.float64)
    Hf, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 8.0)
    if Hf is None:
        return H, w1, len(src)
    return S2 @ Hf @ S @ H, w1, len(src)


def _verify_warp(warped, tpl_ms, scale):
    """Tỉ lệ mốc của MẪU có ô đen đúng vị trí trên ảnh đã warp."""
    if not tpl_ms:
        return 0.0
    wms = detect_markers(warped)
    if not wms:
        return 0.0
    wx = np.array([m[0] for m in wms], np.float64)
    wy = np.array([m[1] for m in wms], np.float64)
    hit = 0
    for m in tpl_ms:
        px, py = m[0] * scale, m[1] * scale
        tol = max(8.0, 1.2 * max(m[2], m[3]) * scale)
        d = np.hypot(wx - px, wy - py)
        if d.size and d.min() <= tol:
            hit += 1
    return hit / float(len(tpl_ms))


# ---------- 4. chuẩn hoá sáng ----------

def _ink_map(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.blur(gray, (51, 51))
    return gray / (bg + 1e-6)


def _corner_check(warped, quad, scale):
    """Tỉ lệ 4 ô định vị của mẫu thực sự có Ô ĐEN được dò thấy trên ảnh warp."""
    ms = detect_markers(warped)
    if len(ms) < 4:
        return 0.0
    mx = np.array([m[0] for m in ms], np.float64)
    my = np.array([m[1] for m in ms], np.float64)
    hit = 0
    for (tx, ty) in quad:
        px, py = tx * scale, ty * scale
        tol = max(10.0, 1.2 * 9.0 * scale)   # mốc mẫu ~9 px -> 1,2 lần cạnh
        d = np.hypot(mx - px, my - py)
        if d.size and float(d.min()) <= tol:
            hit += 1
    return hit / 4.0


def _ink_fraction(bgr):
    nrm = _ink_map(bgr)
    return float((nrm < 0.80).mean())


def detect_blob_centers(img):
    """Tâm các vòng tròn in trên phiếu (dùng cho ICP lưới ô)."""
    import omr_sheetgrid as S
    try:
        return S._blob_centers(img)
    except Exception:                                        # noqa
        return np.zeros((0, 2), np.float32)


def refine_grid_by_cells(warped, cells, scale, iters=4, tol_frac=0.55):
    """TINH CHỈNH lưới ô bằng ICP trên VÒNG TRÒN IN THẬT.

    Sau khi warp, các ô mẫu có thể lệch 10-20 px (phối cảnh còn dư, méo ống
    kính). Hàm này ghép từng tâm ô mẫu với tâm vòng tròn gần nhất rồi fit lại
    homography (RANSAC) — lặp vài lần là bám đúng lưới.

    Trả (H_correction, so_khop) với H áp lên ảnh ĐÃ warp (toạ độ warp).
    """
    if not cells:
        return None, 0.0
    pts = np.float32([[c[3] * scale, c[4] * scale] for c in cells]).reshape(-1, 1, 2)
    # bước lưới gần đúng để đặt ngưỡng
    ys = sorted({round(c[4], 2) for c in cells})
    dy = np.median(np.diff(ys)) * scale if len(ys) > 2 else 20.0
    tol = max(6.0, tol_frac * float(dy))
    blobs = detect_blob_centers(warped)
    if len(blobs) < 20:
        return None, 0.0
    Hc = np.eye(3, dtype=np.float64)
    matched = 0
    for _ in range(iters):
        proj = cv2.perspectiveTransform(pts, Hc).reshape(-1, 2)
        dd = np.linalg.norm(proj[:, None, :] - blobs[None, :, :], axis=2)
        j = dd.argmin(axis=1)
        near = dd[np.arange(len(proj)), j]
        keep = near <= tol
        matched = int(keep.sum())
        if matched < 30:
            return None, 0.0
        Hn, _ = cv2.findHomography(pts[keep], blobs[j][keep], cv2.RANSAC, 3.0)
        if Hn is None:
            break
        Hc = Hn
    return Hc, matched / max(1, len(pts))


RING_MIN_PART = {
    # cổng BÁM MẪU — xác thực ảnh có bám sát MẪU GIẤY ĐÃ CHỌN hay không.
    # "ALL" = tỉ lệ khớp trên TOÀN BỘ ô của template (build_template_cells):
    #   mẫu 12:  đúng mẫu nét 90-97%, đúng mẫu photocopy nhạt 67-82%,
    #            phiếu in BIẾN THỂ (P1 nửa chiều dài) 50-51%,
    #            chấm nhầm SAI mẫu 16-17%  -> ngưỡng 0.60 tách sạch
    # "P1"/"P2"/... = theo phần trên keyed cells (mẫu 10_11 photocopy mờ:
    #   ring detector bị dấu chấm của HS + mực nhạt hạn chế — đo đúng mẫu
    #   P1 0.65-0.71, P2 0.56-0.81 -> ngưỡng thấp tương ứng)
    "12": {"ALL": 0.60},
    "10_11": {"P1": 0.40, "P2": 0.15, "P3": 0.15},
    # moi: cells = P1 (80) + P2 (32) từ moi_cells.json; cùng bản in chụp nét
    # ring 0.65-0.85, ảnh mờ (Laplacian 56-235) 0.46-0.57 — dị dạng ống kính
    # còn dư sau homography, ICP không lấy hết. Phiếu SAI mẫu/biến thể
    # (đo trên mẫu cũ: 0.03-0.54 theo phần) < 0.35 -> ngưỡng 0.50 tách sạch
    "moi": {"ALL": 0.50},
}
RING_MIN_DEFAULT = 0.50


def grid_quality(warped, cells, scale, part="P1"):
    """Chất lượng hình học của warp theo PHẦN user quan trọng nhất (P1 — mã đề
    có thể suy từ key, P1 thì không — chỉ đạo user 2026-11-08).

    Trả (tỉ_lệ ô có ring đúng vị trí trong phần, residual khớp affine trung vị
    ở template scale). Residual ĐO ĐỘ BẮM LƯỚI — thấp = ảnh không bị kéo dãn;
    ring_rate tổng (cổng bám mẫu) có thể cao dù warp lệch trục (đo thật
    090151: pass1 ring 0.80/res 0.63 vs prealign ring 0.78/res 0.27)."""
    import omr_sheetgrid as S
    if not cells or warped is None:
        return 0.0, 99.0
    pc = [c for c in cells if c[0] == part] if part else list(cells)
    if len(pc) < 8:
        pc = list(cells)
    pts = np.float32([[c[3] * scale, c[4] * scale] for c in pc])
    ys = sorted({round(c[4], 2) for c in pc})
    dy = float(np.median(np.diff(ys))) * scale if len(ys) > 2 else 22.0 * scale
    tol = 0.45 * dy
    rings = S._ring_centers(S._ring_ready(warped), accept_filled=True)
    if len(rings) < 10 or not len(pts):
        return 0.0, 99.0
    dd = np.linalg.norm(pts[:, None, :] - rings[None, :, :], axis=2)
    near = dd.min(axis=1)
    j = dd.argmin(axis=1)
    ok = near <= tol
    rate = float(ok.sum()) / float(len(pts))
    if ok.sum() < 8:
        return rate, 99.0
    src = pts[ok] / scale
    dst = rings[j][ok] / scale
    A = np.hstack([src, np.ones((len(src), 1))])
    coef, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    r = np.linalg.norm(dst - A @ coef, axis=1)
    return rate, float(np.median(r))


def ring_part_rates(warped, cells, scale, enhance_rings=False,
                    accept_filled=False):
    """Tỉ lệ ô có VÒNG TRÒN IN đúng vị trí, TÍNH RIÊNG THEO PHẦN (P1/P2/P3).

    Đếm ring thật (_ring_centers — loại chữ số/khung); mỗi phần phải bám mẫu
    độc lập nên phiếu thiếu cả nửa template (in biến thể) không thể lọt.
    enhance_rings: tăng nét bản warp TRƯỚC khi dò — chỉ bật cho mẫu moi (ảnh
    chụp mờ); bật cho mẫu cũ làm TỤT rate trên bản scan (đã đo:
    test12_20 20/20 -> 18/20)."""
    import omr_sheetgrid as S
    rings = S._ring_centers(S._ring_ready(warped) if enhance_rings else warped,
                            accept_filled=accept_filled)
    if len(rings) < 10:
        return {}
    ys = sorted({round(c[4], 2) for c in cells})
    dy = float(np.median(np.diff(ys))) * scale if len(ys) > 2 else 22.0
    tol = 0.45 * dy
    acc = {}
    for c in cells:
        d = np.linalg.norm(rings - np.float32([c[3] * scale, c[4] * scale]),
                           axis=1)
        a = acc.setdefault(c[0], [0, 0])
        a[1] += 1
        if len(d) and float(d.min()) <= tol:
            a[0] += 1
    return {k: v[0] / max(1, v[1]) for k, v in acc.items()}


def ring_grid_stats(warped, cells, scale, enhance_rings=False,
                    accept_filled=False):
    """Thống kê bám mẫu: tỉ lệ ô mẫu có VÒNG TRÒN IN đúng vị trí (sau warp) +
    residual trung vị (px).

    Khác refine_grid_by_cells (hit đếm MỌI blob gần — chữ số/khung cũng tính,
    phiếu khác layout vẫn đạt 1.0), hàm này chỉ đếm RING thật qua
    omr_sheetgrid._ring_centers — đo CHÍNH XÁC mức bám sát mẫu giấy."""
    import omr_sheetgrid as S
    pts = np.float32([[c[3] * scale, c[4] * scale] for c in cells])
    if not len(pts):
        return 0.0, 99.0
    ys = sorted({round(c[4], 2) for c in cells})
    dy = float(np.median(np.diff(ys))) * scale if len(ys) > 2 else 22.0
    tol = 0.45 * dy
    rings = S._ring_centers(S._ring_ready(warped) if enhance_rings else warped,
                            accept_filled=accept_filled)
    if len(rings) < 10:
        return 0.0, 99.0
    dd = np.linalg.norm(pts[:, None, :] - rings[None, :, :], axis=2)
    j = dd.argmin(axis=1)
    near = dd[np.arange(len(pts)), j]
    ok = near <= tol
    rate = float(ok.sum()) / float(len(pts))
    res = float(np.median(near[ok])) if ok.any() else 99.0
    return rate, res


def normalize_light(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.blur(gray, (51, 51))
    gain = 235.0 / np.maximum(bg, 1.0)
    out = np.clip(bgr.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    return out


# ---------- 4b. NHẬN DẠNG BUBBLE ĐÃ TÔ (AI không giám sát) + TÔ ĐEN ----------

def _cell_scores(warped, cells, scale):
    """Điểm mực của từng ô + vị trí thật (dùng read_cells đã kiểm chứng)."""
    import omr_sheetgrid as S
    H = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    reads = S.read_cells(warped, cells, H, exp_w=7.0 * scale)
    keys, scores, pos = [], [], []
    for c in cells:
        v = reads.get((c[0], c[1], c[2]))
        if v is None:
            keys.append((c[0], c[1], c[2]))
            scores.append(0.0)
            pos.append((c[3] * scale, c[4] * scale, 0.5 * 7.0 * scale))
            continue
        keys.append((c[0], c[1], c[2]))
        scores.append(float(v[0]))
        # v[3] ĐÃ là bán kính bubble (read_cells trả bán kính vòng)
        pos.append((float(v[1]), float(v[2]), float(v[3])))
    return keys, np.asarray(scores, np.float32), pos


def classify_filled(keys, scores, min_mask=0.12, rel=0.75):
    """Nhận dạng ô ĐÃ TÔ: ngưỡng nền tự thích ứng + so sánh TRONG CÙNG CÂU.

    Một ảnh có thể chứa cả ô tô đậm (0,6) lẫn ô tô bút chì mờ (0,16) nên phân
    2 cụm toàn trang (k-means) sẽ nuốt mất nhóm mờ. Vì vậy:
      - nền = trung vị toàn ảnh (đa số ô trống) + biên;
      - trong mỗi câu, ô được coi là đã tô nếu điểm ≥ 75% ô đậm nhất của câu đó
        (bắt được cả trường hợp tô 2 ô), và ≥ ngưỡng nền.
    Trả (mask bool, ngưỡng nền, mức đậm nhất).
    """
    import numpy as _np
    n = len(scores)
    if n == 0:
        return _np.zeros(0, bool), 0.0, 0.0
    sc = _np.asarray(scores, _np.float32)
    base = float(_np.percentile(sc, 50))
    floor = max(min_mask, base + 0.05)
    groups = {}
    for k, v in zip(keys, sc):
        groups.setdefault((k[0], k[1]), []).append(float(v))
    mask = _np.zeros(n, bool)
    peak = 0.0
    for idx, k in enumerate(keys):
        sc_i = float(sc[idx])
        mx = max(groups[(k[0], k[1])])
        peak = max(peak, mx)
        if mx >= floor and sc_i >= max(floor, rel * mx):
            mask[idx] = True
    return mask, floor, peak


def ink_bubbles(warped, pos, filled, radius_scale=0.62):
    """TÔ ĐEN các bubble đã tô trên ảnh (bản sao) — để người xem kiểm tra.

    Bán kính 0,62 × bán kính bubble: chấm đen nằm GỌN TRONG viền vòng tròn in
    sẵn (khi để 0,9 thì chấm phủ kín cả viền, nhìn khó phân biệt ô nào là ô in,
    ô nào là ô đã tô).
    """
    out = warped.copy()
    for (x, y, r), is_fill in zip(pos, filled):
        if not is_fill:
            continue
        cv2.circle(out, (int(round(x)), int(round(y))),
                   max(3, int(round(r * radius_scale))), (0, 0, 0), -1, cv2.LINE_AA)
    return out


def mark_filled(warped, cells, scale=2.0, min_mask=0.10):
    """Nhận dạng + tô đen. Trả (ảnh_đã_tô, mask, thống_kê)."""
    keys, scores, pos = _cell_scores(warped, cells, scale)
    filled, floor, peak = classify_filled(keys, scores, min_mask=min_mask)
    inked = ink_bubbles(warped, pos, filled)
    return inked, {k: bool(v) for k, v in zip(keys, filled)}, {
        "n_cells": int(scores.size), "n_filled": int(filled.sum()),
        "muc_tô": round(peak, 3), "muc_nền": round(floor, 3)}


def id_crop(warped, paper, tpl_path, scale=2.0, up=2.0, pad=6.0):
    """Ảnh RIÊNG của vùng SBD + Mã đề, phóng to + tăng tương phản.

    Dùng để kiểm tra/đối chiếu nhanh phần mã đề sau khi đã chuẩn hoá.
    """
    import omr_sheetgrid as S
    cells = S.build_id_cells(paper, tpl_path)
    if not cells:
        return None
    xs = [c[3] for c in cells]
    ys = [c[4] for c in cells]
    x0 = int(max(0, (min(xs) - pad) * scale))
    x1 = int(min(warped.shape[1], (max(xs) + pad) * scale))
    y0 = int(max(0, (min(ys) - pad) * scale))
    y1 = int(min(warped.shape[0], (max(ys) + pad) * scale))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = warped[y0:y1, x0:x1]
    crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)


# ---------- 5. API chính ----------

def paper_of_template(tpl_path):
    """Suy khoá mẫu ('12' | '10_11' | 'moi') từ tên file mẫu."""
    name = str(tpl_path).lower()
    if "moi" in name:
        return "moi"
    return "10_11" if "10_11" in name or "10-11" in name else "12"


def best_orientation(img, tpl_path, paper, rotations=(0, 90, 180, 270), cells=None,
                     allow_sift=False):
    """Thử 4 hướng xoay, trả (H, rotated_img, quality, rot, cands).

    Ảnh điện thoại thường KHÔNG có EXIF orientation (máy cầm ngang) nên phải
    thử xoay theo nội dung. Đo trên bộ Test11: 8/10 ảnh chỉ khớp sau khi xoay
    270°, quality 0,5-0,7 -> 1,000.

    cands = [(rot, H, im, q)] của MỌI hướng có q ≥ 0.3 (sắp theo q giảm) —
    normalize_to_template dùng để CHỐT HƯỚNG bằng ring rate khi kiểm chứng
    corner/lưới của hướng tốt nhất thất bại (phiếu méo ống kính: corner gần
    như không bao giờ 4/4, còn quality giữa các hướng chênh nhau không đáng
    kể vì phiếu tự-similar; ring rate mới là thước "ĐÚNG BẢN IN").
    """
    import omr_sheetgrid as S
    ref = S.build_template_cells(paper, tpl_path)
    use_cells = cells if cells else ref
    tpl_img = S._imread_cached(tpl_path)
    tpl_h, tpl_w = tpl_img.shape[:2]
    # Ảnh NGANG mà mẫu DỌC -> gần như chắc chắn đã bị xoay 90/270; thử trước
    # để khỏi tốn 2-3 lần căn mẫu vô ích (mỗi lần ~1,5 s ở ảnh 2600 px)
    if len(rotations) > 2 and img.shape[1] > img.shape[0] * 1.15:
        rotations = (90, 270, 0, 180) + tuple(r for r in rotations
                                             if r not in (0, 90, 180, 270))
    best = (None, img, 0.0, 0)
    best_ok = None
    cands = []
    for rot in rotations:
        if rot == 90:
            im = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            im = cv2.rotate(img, cv2.ROTATE_180)
        elif rot == 270:
            im = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            im = img
        _t0 = time.perf_counter() if _PROFILE else 0.0
        try:
            H, ph, q = S.align_photo("", tpl_path, use_cells,
                                     photo=im, ref_cells=ref,
                                     allow_sift=allow_sift)
        except Exception:
            if _PROFILE:
                print(f"[profile]   align rot={rot:<3} LỖI          "
                      f"{1000 * (time.perf_counter() - _t0):7.0f} ms",
                      file=sys.stderr)
            continue
        if _PROFILE:
            print(f"[profile]   align rot={rot:<3} q={q:.2f} "
                  f"{1000 * (time.perf_counter() - _t0):7.0f} ms",
                  file=sys.stderr)
        if q is None or q < 0.3:
            continue
        cands.append((rot, H, im, float(q)))
        # xác thực bằng warp thử: 4 ô định vị phải nằm đúng chỗ
        ok = False
        try:
            Mq = np.linalg.inv(np.asarray(H, np.float64))
            wq = cv2.warpPerspective(im, Mq, (int(tpl_w), int(tpl_h)),
                                     flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            quad, _ = template_fiducials(tpl_path)
            cf = _corner_check(wq, quad, 1.0) if len(quad) == 4 else 0.0
            ink = _ink_fraction(wq)
            ok = cf >= 1.0 and 0.02 <= ink <= 0.50
        except Exception:
            ok = False
        if ok and (best_ok is None or q > best_ok[2]):
            best_ok = (H, im, float(q), rot)
        if q > best[2]:
            best = (H, im, float(q), rot)
        # chất lượng đã tuyệt đối -> không cần thử hướng xoay còn lại
        # (mỗi lần thử tốn ~0,7-1,5 s: quét mốc + ICP cả ảnh).
        # NHƯNG vẫn thu tối thiểu 2 ứng viên: normalize_to_template chốt lại
        # hướng bằng ring rate khi ring gate fail — cần ≥2 giả thuyết H để so
        # (align bất biến xoay nên ảnh sạch các hướng hội tụ về cùng warp,
        # thêm 1 lần align chỉ tốn ~0,5-2 s).
        if q >= 0.97 and ok and len(cands) >= 2:
            break
    # Thử lại có SIFT nếu quét nhanh không tìm được gì tốt (ảnh khó) — BỎ QUA
    # với ảnh gần như TRẮNG (mực < 1.5 %): không có mốc/vòng nào để khớp,
    # SIFT tốn 4-5 s mà chắc chắn vô ích (ảnh chụp sại/trắng giấy).
    if (best_ok is None and best[2] < 0.70) and not allow_sift \
            and _ink_fraction(img) >= 0.015:
        return best_orientation(img, tpl_path, paper, rotations=rotations,
                                cells=cells, allow_sift=True)
    # ---- ỨNG VIÊN MÀU (2026-11-06): căn theo vùng CÙNG MÀU mẫu↔ảnh khi cả
    # 2 bên có màu (chế độ 2D / dải). rot=-1 để _best_rotation_by_ring không
    # bỏ sót; H là mẫu->ẢNH GỐC (ảnh đã exif_transpose từ load_bgr). ----
    try:
        import os as _os
        import omr_coloralign as CA
        if _os.environ.get("OMR_COLOR") != "0" \
                and CA.needs_color_check(tpl_path, img):
            _hc = CA.align_by_color(img, tpl_path)
            if _hc is not None:
                cands.append((-1, np.asarray(_hc, np.float64), img, 1.0))
    except Exception:                                        # noqa
        pass
    out = best_ok if best_ok is not None else best
    cands.sort(key=lambda z: -z[3])
    return out + (cands,)


_SIFT_H_CACHE = {}


def _sift_candidate(img, tpl_path):
    """Ứng viên homography (mẫu -> ảnh) từ SIFT + RANSAC — "cứu" ảnh MỜ mà
    matching ô đen bị lệch (~10-20 px) làm ICP kẹt vào blob lân cận.

    Đã chứng minh ở build_moi_template: SIFT đăng ký đúng 12/12 ảnh (62-753
    inliers) trong khi marker matching của align_photo lệch trên 4/12 ảnh
    mờ nhất (Laplacian 56-235). Ảnh mẫu tính feature MỘT LẦN (cache).
    Trả (rot=-1, H, img, q) hoặc None. rot=-1: không trùng hướng nào để
    _best_rotation_by_ring không bỏ sót."""
    key = str(tpl_path)
    if key not in _SIFT_H_CACHE:
        tg = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if tg is None:
            return None
        sift = cv2.SIFT_create(nfeatures=6000)
        tkp, tdes = sift.detectAndCompute(tg, None)
        _SIFT_H_CACHE[key] = (tkp, tdes)
    tkp, tdes = _SIFT_H_CACHE[key]
    if tdes is None or len(tkp) < 30:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sift = cv2.SIFT_create(nfeatures=6000)
    kp, des = sift.detectAndCompute(g, None)
    if des is None or len(kp) < 30:
        return None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    try:
        mt = flann.knnMatch(tdes, des, k=2)
    except Exception:                                        # noqa
        return None
    good = [m for m, n in mt if m.distance < 0.7 * n.distance]
    if len(good) < 30:
        return None
    src = np.float32([tkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if Hs is None or mask is None or int(mask.sum()) < 30:
        return None
    q = min(0.99, float(mask.sum()) / max(1, len(good)))
    return (-1, Hs, img, q)


def _best_rotation_by_ring(cands, cells, W, H, cur_rot, scale=1.0,
                           enhance_rings=False, accept_filled=False):
    """Chọn ứng viên hướng xoay có RING RATE cao nhất trên warp thô (không ICP).

    Trả (rate, H, im, q, rot) nếu tốt hơn rot hiện tại và rate ≥ 0.55, hoặc
    None. Ring rate là thước ĐÚNG BẢN IN — hướng đúng tách biệt rõ (≥0.6 so
    với <0.3 của hướng sai). W×H = cỡ warp đích (template × scale)."""
    best = None
    for r2, H2, im22, q2 in cands:
        if r2 == cur_rot:
            continue
        try:
            Mw = np.linalg.inv(np.asarray(H2, np.float64))
            w2 = cv2.warpPerspective(im22, Mw, (int(W), int(H)),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255))
            rr2, _res = ring_grid_stats(w2, cells, scale,
                                        enhance_rings=enhance_rings,
                                        accept_filled=accept_filled)
        except Exception:                                    # noqa
            continue
        if best is None or rr2 > best[0]:
            best = (rr2, H2, im22, q2, r2)
    if best is None or best[0] < 0.55:
        return None
    return best


def _fine_rotation_search(tpl_path, ref, cells, H0, im0, rot0):
    """XOAY TỈNH (mọi góc): phiếu nghiêng 5-30° làm VIỆC ĐỐI THỨ TỰ 4 ô đen
    sai (q vẫn ~1.0 vì phiếu tự-similar) -> corner chỉ 2/4 dù H hình học hợp lệ.

    Thử quay ảnh quanh hướng tốt nhất, góc TO -> NHỎ (-6..-24/+6..+24°); với
    mỗi góc căn lại mẫu và xác thực như normalize: 4/4 mốc đúng HOẶC lưới ô
    đáp án khớp >= 50 % (bubble TRÙNG lưới mẫu). Trả (H, im, q, dphi, rot)
    của ứng viên đạt ĐẦU TIÊN, hoặc None."""
    import omr_sheetgrid as S
    tpl = S._imread_cached(tpl_path)
    if tpl is None:
        return None
    use_cells = cells if cells else ref
    for dphi in (-6, 6, -12, 12, -18, 18, -24, 24):
        Mr = cv2.getRotationMatrix2D((im0.shape[1] / 2, im0.shape[0] / 2),
                                     dphi, 1.0)
        imr = cv2.warpAffine(im0, Mr, (im0.shape[1], im0.shape[0]),
                             flags=cv2.INTER_LINEAR, borderValue=(255,) * 3)
        try:
            Hr, _p, qr = S.align_photo("", tpl_path, use_cells, photo=imr,
                                       ref_cells=ref, allow_sift=False)
        except Exception:                                # noqa
            continue
        if qr is None or qr < 0.3:
            continue
        try:
            wq = cv2.warpPerspective(
                imr, np.linalg.inv(np.asarray(Hr, np.float64)),
                (tpl.shape[1], tpl.shape[0]), flags=cv2.INTER_LINEAR,
                borderValue=(255,) * 3)
            quad, _ = template_fiducials(tpl_path)
            cf = _corner_check(wq, quad, 1.0) if len(quad) == 4 else 0.0
            ink = _ink_fraction(wq)
            if not (0.02 <= ink <= 0.50):
                continue
            if cf >= 1.0:
                return Hr, imr, float(qr), dphi, rot0
            if cells:
                Hc, hit = refine_grid_by_cells(wq, cells, 1.0)
                if Hc is not None and hit >= 0.50:
                    return Hr, imr, float(qr), dphi, rot0
        except Exception:                                # noqa
            continue
    return None


def grid_first_align(img, tpl_path, cells, scale=2.0, min_rate=0.45):
    """CĂN LƯỚI TRỰC TIẾP cho ảnh MỜ/CỤT (2026-11-06, yêu cầu người dùng:
    "tạo lưới theo mẫu giấy, đối chiếu ô đen định vị để lấy bubble tô lên
    lưới"). BỎ QUA ô đen định vị 4 góc — quét (góc × thang) khớp TRỰC TIẾP
    cloud RING của ảnh mẫu (phủ TOÀN trang) với cloud ring dò được trên ảnh
    (kể cả bubble đã tô đậm), canh tâm khối từng bước và tự loại điểm ngoài
    khung (phiếu cụt). TỰ KIỂM CHỨNG: tỉ lệ ring mẫu có ring ảnh đúng vị trí
    phải đạt min_rate. Trả H (mẫu -> ảnh) hoặc None."""
    import omr_sheetgrid as S
    tpl_img = S._imread_cached(tpl_path)
    if tpl_img is None:
        return None
    Ht_, Wt_ = tpl_img.shape[:2]
    if not cells:
        return None
    # ---- điểm template = ring của chính ẢNH MẪU (phủ toàn trang). Dùng ô
    # đáp án thì MẤT ĐỐI XỨNG: cloud ảnh phủ cả trang, cloud ô chỉ phủ
    # trái+giữa -> canh tâm khối lệch -> mọi (θ,s) đều sai (bug thật:
    # score trần 0.14 ở mọi góc/thang). ----
    # mẫu in sạch có bubble 4px < sàn 6px của bộ dò — PHÓNG 3× rồi dò
    # (10_11.png: 17 ring ở 1x -> 484 ring ở 3x, phủ toàn trang)
    # CACHE theo mẫu (path, mtime): upscale 3× + dò ring + ma trận pitch chỉ
    # phụ thuộc file mẫu — trước đây tính lại cho MỖI ẢNH fail (~1-2 s).
    try:
        _ck = (str(tpl_path), round(os.path.getmtime(str(tpl_path)), 3))
    except OSError:
        _ck = (str(tpl_path), None)
    if _ck in _GF_TPL:
        T, pitch_t = _GF_TPL[_ck]
    else:
        _SCL = 3.0
        up = cv2.resize(tpl_img, None, fx=_SCL, fy=_SCL,
                        interpolation=cv2.INTER_CUBIC)
        tp = S._ring_centers(S._ring_ready(up), accept_filled=True) / _SCL
        if len(tp) < 40:
            return None
        T = np.float32(tp)
        dnn = np.linalg.norm(T[None, :, :] - T[:, None, :], axis=2)
        np.fill_diagonal(dnn, 1e9)
        pitch_t = float(np.median(dnn.min(axis=1)))
        _GF_TPL[_ck] = (T, pitch_t)
    approx = img.shape[1] / float(Wt_)
    # ---- ring ảnh: dò ở NHIỀU THANG thu nhỏ rồi gộp (ảnh mờ: noise giảm
    # mạnh khi thu — 121112: 1344 ring nhiễu ở 1.0x, 177 ring sạch 0.35x) ----
    rings = []
    for ds in (0.35, 0.5, 1.0):
        im2 = cv2.resize(img, None, fx=ds, fy=ds,
                         interpolation=cv2.INTER_AREA)
        r2 = S._ring_centers(S._ring_ready(im2), accept_filled=True) / ds
        rings.extend(r2.tolist())
    if len(rings) < 30:
        return None
    _minsep = 0.012 * max(img.shape[:2])
    # dedup giữ nguyên NGHĨA greedy-first (điểm sau bị loại nếu < minsep từ
    # MỘT điểm đã giữ bất kỳ) nhưng mảng hoá: vòng Python thuần trên >1300 ring
    # × ~300 điểm giữ mất ~0,2-0,5 s mỗi ảnh fail. float64 = đúng số học của
    # vòng cũ (Python float) — kết quả kept GIỐNG HỆT từng điểm.
    _minsep2 = float(_minsep) ** 2
    _kept = np.empty((len(rings), 2), np.float64)
    _nk = 0
    for x, y in rings:
        if _nk:
            _K = _kept[:_nk]
            if float(np.min((_K[:, 0] - x) ** 2 + (_K[:, 1] - y) ** 2)) <= _minsep2:
                continue
        _kept[_nk] = (x, y)
        _nk += 1
    kept = _kept[:_nk].tolist()
    if len(kept) > 300:
        step = len(kept) / 300.0
        kept = [kept[int(i * step)] for i in range(300)]
    R = np.float32(kept)
    n_cells = len(T)
    best = None          # (score, theta, s, tv, inside)

    def _eval(theta_deg, sf_list):
        nonlocal best
        for theta_deg in np.arange(theta_deg[0], theta_deg[1] + 1e-9,
                                   theta_deg[2]):
            th = np.deg2rad(theta_deg)
            Rm = np.float32([[np.cos(th), -np.sin(th)],
                             [np.sin(th), np.cos(th)]])
            for sf in sf_list:
                s = approx * sf
                M2 = s * Rm
                tvis = T
                inside = None
                for _it in range(2):      # lặp loại điểm ngoài khung (cụt)
                    c_vis = (tvis - tvis.mean(axis=0)) @ M2.T + R.mean(axis=0)
                    inside = ((c_vis[:, 0] > 0.02 * img.shape[1]) &
                              (c_vis[:, 0] < 0.98 * img.shape[1]) &
                              (c_vis[:, 1] > 0.02 * img.shape[0]) &
                              (c_vis[:, 1] < 0.98 * img.shape[0]))
                    if inside.sum() < max(30, 0.6 * n_cells):
                        break
                    tvis = tvis[inside]
                if inside is None or inside.sum() < max(30, 0.6 * n_cells):
                    continue
                tv = R.mean(axis=0) - tvis.mean(axis=0) @ M2.T
                proj = tvis @ M2.T + tv
                d = np.linalg.norm(R[None, :, :] - proj[:, None, :], axis=2)
                tol = 0.6 * pitch_t * s
                # chia theo SỐ ĐIỂM NHÌN THẤY (phiếu cụt chỉ thấy một phần)
                score = float((d.min(axis=1) <= tol).sum()) / max(1, len(tvis))
                if best is None or score > best[0]:
                    best = (score, float(theta_deg), float(s), tv.copy(),
                            inside.copy())

    # BƯỚC QUÉT phải đủ mịn: sai số θ di chuyển điểm rìa (bán kính ~1600 px
    # ảnh) theo 1600·Δθ_rad, sai thang theo Δs/s·1600 — cả hai phải < tol
    # (~21 px). Ảnh vào gần như thẳng đứng (best_orientation đã xử 90°) nên
    # quét hẹp ±14° là đủ.
    # thang quét 0.65..1.15 × approx: phiếu có thể KHÔNG chiếm hết khung
    # (chụp xa / có lề) — 041626/121112: best chạm cận 1.15 cho thấy s thật
    # nằm dưới approx (bug thật 2026-11-06)
    _eval((-14.0, 14.0, 0.7), np.linspace(0.65, 1.15, 21))
    if best is None:
        return None
    print(f"[grid_first] quét thô: score={best[0]:.3f} θ={best[1]} "
          f"s={best[2]:.2f}", file=sys.stderr)
    if best[0] < 0.45:
        return None
    # tinh chỉnh mịn quanh best (±1°, thang ±5%)
    th0, sc0 = best[1], best[2]
    _eval((th0 - 1.0, th0 + 1.0, 0.2),
          np.linspace(max(0.65, (sc0 / approx) - 0.05),
                      min(1.15, (sc0 / approx) + 0.05), 9))
    print(f"[grid_first] tinh chỉnh: score={best[0]:.3f} θ={best[1]} "
          f"s={best[2]:.2f}", file=sys.stderr)
    if best[0] < 0.45:
        return None
    th = np.deg2rad(best[1])
    s = best[2]
    Rm = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th), np.cos(th)]], np.float64)
    H = np.eye(3, dtype=np.float64)
    H[:2, :2] = s * Rm
    H[:2, 2] = np.asarray(best[3], np.float64)
    # TỰ KIỂM CHỨNG lần cuối trên warp thật (cùng đường ống cổng ring):
    Mw = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64) @ \
        np.linalg.inv(H)
    w = cv2.warpPerspective(img, Mw, (int(Wt_ * scale), int(Ht_ * scale)),
                            flags=cv2.INTER_LINEAR, borderValue=(255,) * 3)
    rate, _res = ring_grid_stats(w, cells, scale, enhance_rings=True,
                                 accept_filled=True)
    if rate < min_rate:
        print(f"[grid_first] tự kiểm chứng FAIL: rate={rate:.3f} < "
              f"{min_rate}", file=sys.stderr)
        return None
    return H


def _fallback_align(img, tpl_path, cells, paper, scale=2.0):
    """PHƯƠNG ÁN PHỤ khi căn góc + grid_first đều thất bại (2026-11-06).

    Dành cho 2 dạng ảnh còn fail:
      • mờ cực độ thiếu mốc góc (test11 121112: SIFT 7/16, grid_first 0.146);
      • nét, đủ mốc nhưng ring vừa phải bị cổng cứng từ chối oan (test12
        qhashpix: ring 0.51 < 0.60 dù corner/grid khít).
    Trả H (mẫu→ảnh) hoặc None. Thứ tự theo điểm mạnh của từng phương án:
      (1) SIFT full-frame — không cần mốc góc, khớp đặc trưng toàn trang;
      (2) grid_first_align min_rate=0.30 — quét (θ,s) cloud ring toàn trang,
          nới sàn cho ảnh cụt/khít một phần;
      (3) color-grid — căn theo vùng CÙNG MÀU mẫu↔ảnh, độc lập mốc góc.
    H trả về vẫn phải qua ĐỦ ICP + corner/ring gate của luồng thường nên
    chỉ ảnh thực sự đúng mẫu mới được nhận."""
    import omr_sheetgrid as S
    if cells is None:
        return None
    # (1) SIFT full-frame
    try:
        Hs, qs = S.align_sift(img, tpl_path, cells, ref_cells=cells)
        if Hs is not None and qs is not None and qs >= 0.30:
            print("[fallback] SIFT full-frame OK", file=sys.stderr)
            return Hs
    except Exception:                                        # noqa
        pass
    # (2) grid_first_align min_rate thấp (0.30 thay 0.45)
    try:
        Hg = grid_first_align(img, tpl_path, cells, scale=scale, min_rate=0.30)
        if Hg is not None:
            print("[fallback] grid_first (soft 0.30) OK", file=sys.stderr)
            return Hg
    except Exception:                                        # noqa
        pass
    # (3) color-grid (vùng cùng màu)
    try:
        from omr_colorgrid import align_color_grid
        res = align_color_grid(img, tpl_path)
        if res is not None and res[1] >= 0.25:
            print("[fallback] color-grid OK", file=sys.stderr)
            return res[0]
    except Exception:                                        # noqa
        pass
    return None


def band_affine_refine(warped, cells, scale, nbands=10, tol_frac=1.6,
                       min_pts=20, min_gain=0.02, nbands_x=2):
    """HIỆU CHỈNH PHIẾU CONG/MÉO (2026-11-06): homography TOÀN CỤC không mô
    tả được giấy cong — mốc 4 góc khớp (q≈1.0) nhưng ring ở giữa trang lệch
    hàng chục px (lô 12a7: 8/43 ảnh ring 0.02-0.54 dù corner 1.0, SIFT 100+
    inliers). Giải thuật (xử lý ảnh kinh điển, KHÔNG cần model):
      1. Dò ring trên ảnh ĐÃ WARP (accept_filled=True — bắt cả bubble tô).
      2. Ghép mỗi tâm ô mẫu ↔ ring gần nhất với ngưỡng RỘNG (1.6 × pitch
         hàng, sàn 45 px) — đủ rộng cho độ cong thực tế.
      3. Chia các cặp theo y thành `nbands` DẢI NGANG; mỗi dải lấy dịch
         chuyển TRUNG VỊ (dx, dy) — bền với cặp sai.
      4. Nội suy tuyến tính trường dịch (dx(y), dy(y)) giữa tâm các dải rồi
         `cv2.remap` hiệu chỉnh cả ảnh (dịch cục bộ theo hàng = mô hình
         "row-affine", đúng dạng giấy cong).
      5. nbands_x=2 (2026-11-07, lô goc/12a7): phiếu cong làm DỊCH X KHÁC
         NHAU giữa nửa TRÁI (Phần I sai) và nửa PHẢI (Phần II/III đúng) —
         dx(y) đồng nhất không sửa được. Mỗi dải y tính dx RIÊNG cho nửa
         trái/phải, nội suy tuyến tính theo x.
      6. Trả (ảnh_mới, tỉ_lệ_khớp_mới) CHỈ KHI tỉ lệ khớp tăng ≥ min_gain;
         ngược lại (None, hit_old) — ảnh phẳng (đa số) đi qua nguyên vẹn."""
    import omr_sheetgrid as S
    if not cells or warped is None:
        return None, 0.0
    pts = np.float32([[c[3] * scale, c[4] * scale] for c in cells])
    ys = sorted({round(c[4], 2) for c in cells})
    dy_row = float(np.median(np.diff(ys))) * scale if len(ys) > 2 else 22.0 * scale
    rings = S._ring_centers(S._ring_ready(warped), accept_filled=True)
    if len(rings) < 20:
        return None, 0.0
    dd = np.linalg.norm(pts[:, None, :] - rings[None, :, :], axis=2)
    j = dd.argmin(axis=1)
    near = dd[np.arange(len(pts)), j]
    tol = max(45.0, tol_frac * dy_row)
    # ĐO "hit" bằng ngưỡng HẸP (khớp THẬT) — đo bằng tol RỘNG (chỉ để GHÉP
    # cặp) thì hit_old đã cao sẵn và min_gain không bao giờ đạt → refine
    # không áp dụng dù warp đúng (bug thật 2026-11-06: 090940 hit=0.73 nhưng
    # improved=False, ring kẹt 0.02 vĩnh viễn).
    tol_meas = max(12.0, 0.55 * dy_row)
    hit_old = float((near <= tol_meas).sum()) / float(len(pts))
    keep = near <= tol
    if keep.sum() < min_pts:
        return None, hit_old
    src = pts[keep]
    dst = rings[j][keep].astype(np.float32)
    y_lo, y_hi = float(src[:, 1].min()), float(src[:, 1].max())
    span = max(1e-9, y_hi - y_lo)
    bidx = np.clip(((src[:, 1] - y_lo) / span * nbands).astype(int),
                   0, nbands - 1)
    Hh, Ww = warped.shape[:2]
    half_w = Ww * 0.5
    bidx_x = np.clip((src[:, 0] // half_w).astype(int), 0, nbands_x - 1)
    d_ys, d_dx, d_dy = [], [], []
    for b in range(nbands):
        for bx in range(nbands_x):
            m = (bidx == b) & (bidx_x == bx)
            if int(m.sum()) < max(4, min_pts // (nbands * nbands_x)):
                continue
            d = dst[m] - src[m]
            d_ys.append(float(src[m][:, 1].mean()))
            d_dx.append(float(np.median(d[:, 0])))
            d_dy.append(float(np.median(d[:, 1])))
    if len(d_ys) < 2:
        return None, hit_old
    order = np.argsort(d_ys)
    d_ys = np.asarray(d_ys)[order]
    d_dx = np.asarray(d_dx)[order]
    d_dy = np.asarray(d_dy)[order]
    yy = np.arange(Hh, dtype=np.float32)
    dy_map = np.interp(yy, d_ys, d_dy).astype(np.float32)
    xx_g = np.arange(Ww, dtype=np.float32)[None, :]          # (1, Ww)
    if nbands_x <= 1:
        dx_map = np.interp(yy, d_ys, d_dx).astype(np.float32)[:, None]
    else:
        # dx riêng nửa trái/phải: nội suy theo y cho từng nửa rồi trộn theo x
        dxL_ys, dxL, dxR_ys, dxR = [], [], [], []
        for b in range(nbands):
            for bx in range(nbands_x):
                m = (bidx == b) & (bidx_x == bx)
                if int(m.sum()) < max(4, min_pts // (nbands * nbands_x)):
                    continue
                d = dst[m] - src[m]
                if bx == 0:
                    dxL_ys.append(float(src[m][:, 1].mean()))
                    dxL.append(float(np.median(d[:, 0])))
                else:
                    dxR_ys.append(float(src[m][:, 1].mean()))
                    dxR.append(float(np.median(d[:, 0])))
        if len(dxL_ys) < 2 or len(dxR_ys) < 2:
            dx_map = np.interp(yy, d_ys, d_dx).astype(np.float32)[:, None]
        else:
            dxL_map = np.interp(yy, np.asarray(dxL_ys), np.asarray(dxL))
            dxR_map = np.interp(yy, np.asarray(dxR_ys), np.asarray(dxR))
            fx = np.clip(xx_g / half_w, 0.0, 1.0)       # 0=trái 1=phải
            # (Hh,1)*(1,Ww) -> (Hh,Ww) — bản cũ `xx[0]` NameError (crash bị
            # try/except nuốt → band-refine CHẾT âm thầm từ 2026-11-07)
            dx_map = (dxL_map[:, None] * (1 - fx)
                      + dxR_map[:, None] * fx).astype(np.float32)
    # DẤU ĐÚNG (kiểm chứng tổng hợp + thật 2026-11-08): d = ring − ô mẫu;
    # out(x) = warped(x + d) đưa ring về ĐÚNG vị trí ô — bản cũ `x − d` dịch
    # NGƯỢC chiều nên gần như không bao giờ cải thiện hit (luôn bị từ chối).
    map_x = xx_g + dx_map
    map_y = yy[:, None] + dy_map[:, None] + np.zeros((1, Ww), np.float32)
    out = cv2.remap(warped, map_x, map_y, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rings2 = S._ring_centers(S._ring_ready(out), accept_filled=True)
    if len(rings2) < 20:
        return None, hit_old
    dd2 = np.linalg.norm(pts[:, None, :] - rings2[None, :, :], axis=2)
    near2 = dd2.min(axis=1)
    hit_new = float((near2 <= tol_meas).sum()) / float(len(pts))
    if hit_new >= hit_old + min_gain:
        return out, hit_new
    return None, hit_old



def normalize_to_template(photo, tpl_path, scale=2.0, verify=True, paper=None,
                          cells=None, ink=False, ink_only=False, init_H=None):
    """Wrapper: chạy _normalize_core trên bản ENHANCE; nếu vẫn fail mà ảnh
    ĐÃ được enhance thì thử lại trên ẢNH GỐC và giữ kết quả tốt hơn.

    Lý do thật (2026-11-06, lô /home/dell/Videos): enhance (unsharp) đổiartifact
    nét làm marker matching ra H YẾU trên ảnh vốn khá nét — 041626/041604
    enhance FAIL (q=1.0 SIFT suy biến, corner 0/4) còn ẢNH GỐC OK (ring
    0.51-0.54). Ngược lại ảnh MỜ thật cần enhance (test11 121112). Chỉ ảnh
    fail mới tốn lần chạy thứ hai."""
    # DECODE MỘT LẦN: trước đây pass raw-retry và pass grid_first gọi load_bgr
    # lại từ đĩa (EXIF + LANCZOS ~0,5-1 s/lần). Pipeline không đột biến mảng
    # đầu vào (enhance/rotate/warp đều tạo mảng mới) nên dùng lại mảng gốc là
    # an toàn — pixel GIỐNG HỆT từng pass như cũ.
    _raw = load_bgr(photo) if (isinstance(photo, (str, bytes))
                               or hasattr(photo, "__fspath__")) else photo
    # ---- PREALIGN-FIRST (2026-11-08, chỉ đạo user lần 2: "ảnh gốc xoay nhẹ,
    # zoom, lấy đinh vị, cắt dư theo đinh vị" + "mã đề có thể suy, P1 mới
    # quan trọng"). Đo per-part trên 090151: prealign cho P1 residual 0.27
    # (pass1 0.63), P2 0.18 (pass1 1.8), MD 0.70 (pass1 0.43) — warp pass1
    # KÉO DÃN lệch trục làm chấm P1 sai dù ring tổng cao hơn. Nên: prealign là
    # ĐƯỜNG CHÍNH cho mẫu moi (zoom đều + xoay nhẹ + cắt lề theo 4 đinh vị,
    # KHÔNG thể kéo dãn); luồng cũ là DỰ PHÒNG khi prealign không dựng được
    # (thiếu mốc góc / quad phi lý) hoặc không qua cổng kiểm chứng.
    # KILL-SWITCH OMR_PREALIGN=0; 'all' áp cho mọi mẫu.
    _pa_mode = os.environ.get("OMR_PREALIGN", "")
    _pa_paper = None
    if init_H is None and _pa_mode != "0":
        # MẶC ĐỊNH MỌI MẪU (2026-11-08): prealign similarity có chốt "quad
        # chiếu phải nằm trong ảnh" + residual + tỉ lệ khung — sai thì tự
        # bỏ qua về luồng cũ. 12a7 (090802/090940) cần nó để đỡ nghiêng.
        _pa_paper = paper or paper_of_template(tpl_path)
    # ---- HELPER RETRY ALIGN-FULL (2026-11-08): ICP trên keyed cells (dải hẹp
    # ô có đáp án) EXTRAPOLATE tệ trên ảnh chụp phối cảnh (090136: ring
    # 0.80→0.48, bị loại oan); nhưng full-cells ICP cũng làm 2 ảnh test11 tụt
    # (ô SBD/MĐ viết tay kéo RANSAC nhầm). Nên: mặc định KEYED (hành vi cũ
    # nguyên vẹn); CHỈ khi keyed FAIL mới retry 1 lần với đủ ô template.
    def _run_core(img_, init_H_, no_enh_):
        """2 lượt: keyed trước (hành vi cũ) — fail thì 1 lượt đủ ô mẫu
        (align_full, 090802). KHÔNG đa-rút thăm seed: đo thật 12a7 cả lô
        40 phút mà 090802/090940 vẫn fail — bỏ."""
        try:
            i_ = _normalize_core(img_, tpl_path, scale, verify, paper, cells,
                                 ink, ink_only, init_H_, no_enh_)
        except Exception:                                    # noqa
            i_ = {"ok": False, "quality": 0.0, "reason": "lỗi xử lý"}
        if not i_.get("ok") and cells:
            try:
                i2_ = _normalize_core(img_, tpl_path, scale, verify, paper,
                                      cells, ink, ink_only, init_H_,
                                      no_enh_, _align_full=True)
            except Exception:                                # noqa
                i2_ = {"ok": False, "quality": 0.0}
            if i2_.get("ok"):
                i2_["align_full"] = True
                return i2_
        return i_

    if _pa_paper is not None:
        try:
            _pa = prealign_similarity(_raw, tpl_path, scale)
        except Exception:                                        # noqa
            _pa = None
        if _pa is not None:
            _pa, _pa_A3 = _pa
            info = _run_core(_pa, None, False)
            info["prealigned"] = True
            if info.get("ok") and info.get("H") is not None:
                # H về đúng hệ ẢNH GỐC (semantics: H maps mẫu -> ảnh nạp):
                # điểm mẫu p -> canvas H_core(p) -> ảnh A3(H_core(p)/scale)
                try:
                    _D = np.diag([1.0 / scale, 1.0 / scale, 1.0])
                    info["H"] = _pa_A3 @ _D @ np.asarray(info["H"], np.float64)
                except Exception:                            # noqa
                    pass
            if info.get("ok") and cells:
                # CHẤP NHẬN NHANH khi P1 đạt chất lượng cao (rate ≥ 0.95 +
                # residual ≤ 0.55 px/tpl). Trung bình → chạy thêm pass 1 và
                # CHỌN theo chất lượng P1 (090206: prealign res 1.74 thua
                # pass1 0.35 → chọn pass1; 090151: prealign 0.27 thắng luôn).
                try:
                    _rt, _rs = grid_quality(info["warped"], cells, scale)
                except Exception:                                # noqa
                    _rt, _rs = 0.0, 99.0
                if _rt >= 0.95 and _rs <= 0.55:
                    return info
                info1 = _run_core(_raw, None, False)
                _rt1, _rs1 = ((grid_quality(info1["warped"], cells, scale)
                               if info1.get("ok") and info1.get("warped")
                               is not None else (0.0, 99.0)))
                _better_pa = (_rt, -_rs)
                _better_p1 = (_rt1, -_rs1)
                if info.get("ok") != info1.get("ok"):
                    if info.get("ok"):
                        return info
                    info = info1
                elif _better_pa >= _better_p1:
                    return info
                else:
                    info1["prealigned"] = False
                    info = info1
                if info.get("ok"):
                    return info
            elif info.get("ok"):
                return info
            info["prealigned"] = False
    # ---- CẮT LỀ THEO KHUNG CHUẨN MẪU (2026-11-07) — MẶC ĐỊNH TẮT ----
    # Ảnh chụp có lề lớn quanh phiếu làm warp toàn khung lệch tỉ lệ → chuẩn
    # hóa cắt mất mép phiếu. Xử lý zoom ra/vào: hệ số zoom suy từ bbox mốc
    # đen ảnh/mẫu, lề cắt = lề mẫu × zoom.
    # ⚠ HỒI QUY 2026-11-07: bật mặc định làm test11 0/12 + test12_20 12/20
    # (bbox mốc percentile trên ảnh nhiễu → zoom sai → cắt vào phiếu). Chỉ
    # BẬT thủ công khi cần: env OMR_CROP=1.
    if init_H is None and os.environ.get("OMR_CROP") == "1":
        try:
            _raw = crop_to_template_framing(_raw, tpl_path)
        except Exception:                                    # noqa
            pass
    info = _run_core(_raw, init_H, False)
    if info.get("ok") or not info.get("enhanced") or init_H is not None:
        return info
    info2 = _run_core(_raw, None, True)
    if info2.get("ok"):
        info2["raw_better"] = True
        info2["prealigned"] = False
        return info2
    # ---- CỨU CUỐI: CĂN LƯỚI TRỰC TIẾP (bỏ qua ô định vị — ảnh mờ/cụt) ----
    # H từ lưới phải qua TỰ KIỂM CHỨNG ring-rate chặt trong grid_first_align,
    # sau đó vẫn chạy đủ ICP + corner/ring gate của luồng thường.
    if cells and init_H is None:
        try:
            Hg = grid_first_align(_raw, tpl_path, cells, scale=scale)
        except Exception:                                    # noqa
            Hg = None
        if Hg is not None:
            info3 = _run_core(_raw, Hg, False)
            if info3.get("ok"):
                info3["grid_first"] = True
                return info3
        # ---- FALLBACK ALIGN (2026-11-06): 3 phương án phụ khi căn góc +
        # grid_first đều thất bại (ảnh mờ thiếu mốc / biến thể in). H trả về
        # vẫn phải qua ĐỦ ICP + corner/ring gate nên chỉ ảnh đúng mẫu mới
        # được nhận. ----
        try:
            Hf = _fallback_align(_raw, tpl_path, cells, paper, scale=scale)
        except Exception:                                    # noqa
            Hf = None
        if Hf is not None:
            info4 = _run_core(_raw, Hf, False)
            if info4.get("ok"):
                info4["fallback_align"] = True
                return info4
        # ---- DEWARP GIẤY CONG bằng DocShadow ONNX (2026-11-06) ----
        # Lô 12a7: homography toàn cục không fit được giấy cong — SIFT 100+
        # inliers, corner 1.0 nhưng ring 0.02-0.54. DocShadow (document
        # unwarping) làm phẳng giấy rồi chạy lại luồng thường; đã chứng
        # minh cứu 4/8 ảnh fail của lô (090812/090901/090928/091055).
        # ~20 s/ảnh CPU — chỉ chạy khi các bậc trên đều thất bại.
        import os as _os
        if _os.environ.get("OMR_DEWARP") != "0":
            try:
                import omr_deblur as _DB
                _ds = _DB._docshadow_candidate(_raw)
            except Exception:                                    # noqa
                _ds = None
            if _ds is not None:
                info5 = _run_core(_ds, None, False)
                if info5.get("ok"):
                    info5["dewarp"] = True
                    return info5
    return info if (info.get("quality") or 0) >= (info2.get("quality") or 0) \
        else info2


def _normalize_core(photo, tpl_path, scale=2.0, verify=True, paper=None,
                    cells=None, ink=False, ink_only=False, init_H=None,
                    _no_enhance=False, _align_full=False, _rng_offset=0):
    """Đưa 1 ảnh (đường dẫn hoặc BGR) về ĐÚNG CHUẨN MẪU.

    Quy trình: EXIF → thử 4 hướng xoay → căn mẫu (fiducial + ICP + SIFT dự phòng)
    → warp về kích thước mẫu × scale → chuẩn hoá sáng.
    Trả {"ok", "warped", "H", "quality", "rot", "reason"}.

    CV2 SEED: cv2.findHomography(RANSAC) dùng RNG TOÀN CỤC của OpenCV — nếu
    không reset, kết quả 1 ảnh phụ thuộc các ảnh đã xử lý trước đó trong cùng
    tiến trình (server chạy lâu → kết quả "trôi" giữa các lần chấm). Seed cố
    định trước mỗi ảnh => cùng ảnh luôn cho cùng kết quả.
    """
    cv2.setRNGSeed(20261104 + _rng_offset)
    import omr_sheetgrid as S
    _t = time.perf_counter() if _PROFILE else None
    img = load_bgr(photo) if (isinstance(photo, (str, bytes))
                              or hasattr(photo, "__fspath__")) else photo
    _ms, _t = _prof_tick(_t)
    _prof_log("load", _ms)
    # ---- TĂNG NÉT TỰ ĐỘNG cho ảnh MỜ (CLAHE + unsharp — omr_enhance) ----
    # chỉ kích hoạt khi var Laplacian < 150 (ảnh nét 500-700 đi qua nguyên vẹn)
    enh_used = False
    try:
        import omr_enhance as _ENH
        # thr theo MẪU: ảnh chụp moi mờ (Laplacian 175-235) phải được làm nét
        # TRƯỚC khi căn — marker centroid nhòe làm H lệch ~15 px, ICP không
        # lấy lại được. Mẫu cũ giữ 150 (hành vi nguyên vẹn — validate_testdata
        # test11 11/12 | test12_20 20/20 đã kiểm chứng cả hai ngưỡng).
        # _no_enhance: lượtretry ẢNH GỐC từ wrapper (enhance làm marker
        # matching yếu trên ảnh vốn nét — xem docstring normalize_to_template)
        if _no_enhance:
            img, enh_used = img, False
        else:
            img, enh_used = _ENH.enhance_if_blurry(
                img, thr=300.0 if paper == "moi" else _ENH.BLUR_THR,
                tpl_path=str(tpl_path))
    except Exception:                                    # noqa
        enh_used = False
    _ms, _t = _prof_tick(_t)
    _prof_log("enhance", _ms)
    tpl = S._imread_cached(tpl_path)
    if tpl is None:
        raise FileNotFoundError(tpl_path)
    paper = paper or paper_of_template(tpl_path)
    Ht, Wt = tpl.shape[:2]
    # ---- Ô CĂN CHỈNH (2026-11-08, bug thật 090136 lô 11a8): ICP/homography
    # fit trên KEYED cells (chỉ ô CÓ đáp án — dải hẹp) EXTRAPOLATE TỆ sang phần
    # còn lại: ring sụp 0.80 → 0.48 → ảnh ĐÚNG MẪU bị loại oan. NHƯNG full-cells
    # ICP cũng làm test11 tụt (2 ảnh scan khớp keyed-ICP tốt hơn — ô SBD/MĐ viết
    # tay trong full cells kéo RANSAC nhầm). Nên: mặc định keyed (hành vi cũ
    # nguyên vẹn); wrapper RETRY _align_full=True khi keyed FAIL.
    align_cells = cells
    if cells and _align_full:
        try:
            _ref_full = S.build_template_cells(paper, tpl_path)
            if _ref_full and len(cells) < len(_ref_full):
                align_cells = _ref_full
        except Exception:                                    # noqa
            pass
    info = {"ok": False, "warped": None, "H": None, "quality": 0.0, "rot": 0,
            "reason": "", "enhanced": enh_used}
    try:
        if init_H is not None:
            # ---- H CỨU (vd từ Gemini rescue) — bỏ best_orientation, vào thẳng
            #      vòng kiểm chứng: warp + ICP + corner/ring gate vẫn quyết
            #      định như ảnh thường; quality 1.0 chỉ để qua cổng MIN_ALIGN.
            H = np.asarray(init_H, np.float64)
            im2, q, rot = img, 1.0, 0
            cands = [(0, H, im2, 1.0)]
            info["rescue_H"] = True
        else:
            rots = (0,) if ink_only else (0, 90, 180, 270)
            H, im2, q, rot, cands = best_orientation(img, tpl_path, paper,
                                                     rotations=rots,
                                                     cells=align_cells)
    except Exception as e:                                   # noqa
        info["reason"] = f"lỗi căn mẫu: {e}"
        return info
    _ms, _t = _prof_tick(_t)
    _prof_log("orient", _ms)
    ref = None              # lưới template (cho xoay tỉnh), dựng khi cần
    # tăng nét khi DÒ RING chỉ cho mẫu moi (ảnh chụp mờ) — mẫu cũ TẮT
    er = (paper == "moi")
    # nhận BUBBLE TÔ ĐẬM làm bằng chứng ring — chỉ mẫu moi (HS tô bút đậm
    # làm ring mất lỗ, ring rate trần bị chặn dưới 0.5 dù warp đúng:
    # 041621 P1 44/80 ô tô đậm -> ring rate trần 0.55)
    fa = (paper == "moi")
    # ---- 2 LƯỢT: (0) căn 4 hướng xoay; (1) XOAY TỈNH khi xác thực thất bại ----
    for attempt in (0, 1):
        _t = time.perf_counter() if _PROFILE else None
        info.update(quality=round(q, 3), rot=rot)
        if H is None or q < MIN_ALIGN:
            if _ink_fraction(img) < 0.015:
                info["reason"] = ("ảnh gần như trắng/mờ (mực < 1.5 %) — không đủ "
                                  "nét để căn mẫu, chụp lại rõ hơn")
            else:
                info["reason"] = f"không khớp mẫu (quality {q:.2f} < {MIN_ALIGN})"
            return info
        # H: mẫu -> ảnh, nên để warp ẢNH VỀ HỆ MẪU phải dùng S · H⁻¹
        Ss = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
        M = Ss @ np.linalg.inv(np.asarray(H, np.float64))
        warped = cv2.warpPerspective(im2, M, (int(Wt * scale), int(Ht * scale)),
                                     flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

        # ---- TINH CHỈNH LƯỚI (ICP trên vòng tròn in thật) ----
        # Warp từ 4 mốc còn dư sai số 10-20 px (phối cảnh/méo ống kính) -> ô đọc
        # lệch khỏi vòng, điểm tụt về 0. Ghép lại lưới ô với blob thật rồi warp lại
        # từ ẢNH GỐC (không warp chồng lên ảnh đã warp để khỏi mất nét).
        if cells:
            Hc, hit = refine_grid_by_cells(warped, align_cells, scale)
            if Hc is not None and hit >= 0.5:
                M2 = Hc @ M          # áp hiệu chỉnh lưới lên ma trận warp ban đầu
                cand = cv2.warpPerspective(im2, M2, (int(Wt * scale), int(Ht * scale)),
                                           flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                info["grid_hit"] = round(float(hit), 3)
                if cand is not None and cand.size:
                    warped = cand
        _ms, _t = _prof_tick(_t)
        _prof_log("warp+refine", _ms)

        # ---- KIỂM CHỨNG BẮT BUỘC: chống homography suy biến (ảnh warp trắng trơn
        #      vẫn có thể cho quality 1.0 vì mọi ô cùng rơi vào một vùng sáng) ----
        if verify:
            quad, _ = template_fiducials(tpl_path)
            cf = _corner_check(warped, quad, scale) if len(quad) == 4 else 0.0
            # ---- KIỂM CHỨNG BẰNG LƯỚI Ô ĐÁP ÁN (ICP) ----
            # 4 mốc góc có thể bị che/mất (ảnh chụp gần, mốc mờ) -> cf chỉ 2-3/4
            # dù warp vẫn ĐÚNG. Lưới ô đáp án (hàng trăm vòng tròn) là bằng chứng
            # mạnh hơn nhiều: nếu >= 50 % tâm ô khớp vòng in thật thì nhận ảnh.
            _w_changed = False
            if cf < 1.0 and cells:
                Hc, hit = refine_grid_by_cells(warped, align_cells, scale)
                info["grid_hit"] = round(float(hit), 3)
                if Hc is not None and hit >= 0.50:
                    M2 = Hc @ M
                    warped = cv2.warpPerspective(
                        im2, M2, (int(Wt * scale), int(Ht * scale)),
                        flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                    M = M2
                    info["M_final"] = M2
                    info["refined"] = True
                    _w_changed = True
                    cf = max(cf, float(hit))
            # dò lại mốc CHỈ khi warp đã được refine thay thế — warp giữ nguyên
            # thì detect_markers cho cùng kết quả (định tính), chạy lại chỉ tốn
            # ~0,3-0,5 s mỗi ảnh
            if _w_changed:
                cf = _corner_check(warped, quad, scale) if len(quad) == 4 else 0.0
            ink = _ink_fraction(warped)
            info["corner"] = round(cf, 2)
            info["ink"] = round(ink, 3)
            _ms, _t = _prof_tick(_t)
            _prof_log("verify", _ms)
            # Nhận ảnh nếu 4/4 mốc đúng HOẶC lưới ô đáp án khớp >= 50 %
            if cf < 1.0 and float(info.get("grid_hit") or 0.0) < 0.50:
                if attempt == 0 and not ink_only:
                    # ---- CHỐT HƯỚNG XOAY bằng RING RATE: phiếu méo ống kính
                    #      làm corner gần như không bao giờ 4/4, còn quality
                    #      giữa các hướng xoay chênh nhau không đáng kể (phiếu
                    #      tự-similar) -> hướng 90° có thể thắng nhầm (đã gặp:
                    #      q=0.887@rot90 thắng q thấp hơn @rot180 đúng).
                    #      Đo ở SCALE THẬT của mẫu (norm_scale_for — moi cần
                    #      3.0; scale 1 làm ring moi 3.3px < sàn 6px của bộ dò
                    #      -> luôn 0.0, re-selection không bao giờ nhả ứng
                    #      viên — bug thật 2026-11-06). ----
                    if cells and len(cands) > 1:
                        _ns = NORM_SCALE.get(paper, 2.0)
                        got = _best_rotation_by_ring(cands, align_cells,
                                                     int(Wt * _ns),
                                                     int(Ht * _ns), rot,
                                                     scale=_ns,
                                                     enhance_rings=er,
                                                     accept_filled=fa)
                        if got is not None:
                            _rr, H, im2, q, rot = got
                            info["rot_by_ring"] = rot
                            continue            # xác thực lại với H mới
                    # ---- ỨNG VIÊN SIFT: ảnh mờ mà marker matching lệch thì
                    #      mọi ứng viên hướng đều sai — SIFT (đã chứng minh
                    #      đăng ký đúng 12/12 ảnh ở build_moi_template) ----
                    if cells:
                        try:
                            _sc = _sift_candidate(im2, tpl_path)
                        except Exception:                    # noqa
                            _sc = None
                        if _sc is not None:
                            _ns2 = NORM_SCALE.get(paper, 2.0)
                            got = _best_rotation_by_ring([_sc], align_cells,
                                                         int(Wt * _ns2),
                                                         int(Ht * _ns2),
                                                         cur_rot=None,
                                                         scale=_ns2,
                                                         enhance_rings=er,
                                                         accept_filled=fa)
                            if got is not None:
                                _rr, H, im2, q, _r2 = got
                                info["sift_rescue"] = True
                                continue            # xác thực lại với H mới
                    # ---- XOAY TỈNH: phiếu nghiêng 5-30° đối thứ tự 4 ô đen
                    #      SAI (q ~1.0, corner 2/4) -> thử quay ảnh to->nhỏ
                    #      tới khi bubble TRÙNG lưới mẫu rồi xác thực lại ----
                    if ref is None:
                        import omr_sheetgrid as _SG
                        ref = _SG.build_template_cells(paper, tpl_path)
                    found = _fine_rotation_search(tpl_path, ref, cells,
                                                  H, im2, rot)
                    if found is not None:
                        H, im2, q, _dphi, rot = found
                        info["fine_rot"] = _dphi
                        continue            # xác thực lại với H mới
                    # ---- CỨU CUỐI TRƯỚC KHI TỪ CHỐI (2026-11-06): giấy cong
                    # KHÔNG ĐỀU (090940: mốc góc 2/4, grid 0.30) — band-refine
                    # lặp kéo ring từ 0.02 lên 0.49. Nếu ring_part sau refine
                    # ĐỒNG ĐỀU và ≥0.42 thì warp đúng mẫu (mốc góc bị che/bong
                    # không đáng kể) → nhận với band_rescue. ----
                    if cells:
                        try:
                            _wb = warped
                            for _nb, _tf in ((10, 1.6), (8, 2.0),
                                             (14, 2.4), (18, 2.8)):
                                _o, _h = band_affine_refine(
                                    _wb, align_cells, scale, nbands=_nb,
                                    tol_frac=_tf, min_gain=0.008)
                                if _o is None:
                                    break
                                _wb = _o
                            if _wb is not warped:
                                _pr = ring_part_rates(
                                    _wb, align_cells, scale,
                                    enhance_rings=er, accept_filled=True)
                                _sp = ((max(_pr.values()) - min(_pr.values()))
                                       if _pr else 1.0)
                                _mr = min(_pr.values()) if _pr else 0.0
                                # CỨU THEO VÙNG MÀU (2026-11-08): thay tiêu chí
                                # ring_part (keyed, dải hẹp) bằng ring TỪNG
                                # VÙNG — 090940 corner 2/4 do giấy cong nhưng
                                # mọi vùng ≥ 0.50 = đúng mẫu chắc chắn.
                                _zr_ok = False
                                try:
                                    import omr_zones as _OZ2
                                    _zz2 = _OZ2.zones_for(paper, tpl_path,
                                                          cells=None)
                                    if _zz2:
                                        _zr2 = _OZ2.zone_ring_rates(
                                            _wb, align_cells, scale, _zz2)
                                        _zr_ok = (len(_zr2) >= 2
                                                  and min(_zr2.values()) >= 0.50)
                                        if _zr_ok:
                                            info["ring_zone"] = {
                                                k: round(v, 2) for k, v in
                                                _zr2.items()}
                                except Exception:            # noqa
                                    _zr_ok = False
                                if (_pr and _mr >= 0.42 and _sp <= 0.22) \
                                        or _zr_ok:
                                    # band_rescue ĐẠT: ring_part đồng đều ≥0.42
                                    # = warp đúng MẪU (mọi phần khớp như nhau).
                                    # Mốc góc 2/4 ở đây do bị che/bong/ngoài
                                    # khung, KHÔNG phải sai mẫu → nhận warp,
                                    # xác thực lại bằng ring gate bên dưới.
                                    warped = _wb
                                    info["band_rescue"] = round(_mr, 2)
                                    info["band_spread"] = round(_sp, 2)
                                    info["corner_rescued"] = True
                                    cf = 1.0        # bỏ chặn corner (ring đã xác nhận)
                                    ink = _ink_fraction(warped)
                                    info["corner"] = 0.75
                                    info["ink"] = round(ink, 3)
                                    break           # ra khỏi nhánh corner, đi tiếp ring gate
                        except Exception:                    # noqa
                            pass
                info["reason"] = f"chỉ {int(cf * 4)}/4 ô định vị nằm đúng chỗ"
                return info
            if cf < 1.0:
                info["reason"] = ""            # warp đã được xác nhận bằng lưới ô
        if not (0.02 <= ink <= 0.50):
            info["reason"] = f"ảnh warp không giống phiếu (mực {ink:.1%})"
            return info
        # ---- HIỆU CHỈNH PHIẾU CONG (band-affine, 2026-11-06) ----
        # Homography toàn cục không mô tả được giấy cong: mốc 4 góc khớp
        # nhưng ring giữa trang lệch (lô 12a7: 8 ảnh ring 0.02-0.54 dù
        # corner 1.0). Chia lưới thành dải ngang, căn từng dải theo ring
        # thật rồi remap — CHỈ dùng khi độ khớp tăng (ảnh phẳng không đổi).
        # LẶP (2026-11-06): cong KHÔNG ĐỀU cần nhiều lượt — mỗi lượt nới
        # ngưỡng ghép + tăng số dải (090940: ring 0.02 → 0.49 sau 5 lượt).
        if cells:
            try:
                for _nb, _tf in ((10, 1.6), (8, 2.0), (14, 2.4), (18, 2.8)):
                    _out_b, _hit_b = band_affine_refine(
                        warped, align_cells, scale, nbands=_nb, tol_frac=_tf,
                        min_gain=0.008)
                    if _out_b is not None:
                        warped = _out_b
                        info["band_refine"] = round(_hit_b, 2)
                        info["band_refine_iters"] = \
                            int(info.get("band_refine_iters") or 0) + 1
                    else:
                        break          # không còn cải thiện → dừng sớm
            except Exception:                                # noqa
                pass
        # ---- HIỆU CHỈNH THEO VÙNG MÀU (omr_zones, 2026-11-08) ----
        # Giấy cong làm các PHẦN dịch KHÁC NHAU; dải ngang không mô tả được khi
        # hai phần nằm CẠNH nhau cùng hàng (P1 trái / P2 phải của mẫu 10_11).
        # Vùng màu trên mẫu (tc10_11/tc12 — cùng màu = cùng phần, user chỉ
        # định) là đơn vị vật lý đúng: căn dịch chuyển RIÊNG từng vùng màu rồi
        # nội suy mượt 1/d² giữa các vùng. CHỈ nhận khi tỉ lệ khớp tăng
        # (nguyên tắc như band_affine_refine — ảnh phẳng đi qua nguyên vẹn).
        # KILL-SWITCH: env OMR_ZONE=0.
        if cells and os.environ.get("OMR_ZONE") != "0":
            try:
                import omr_zones as OZ
                _zz = OZ.zones_for(paper, tpl_path, cells=None)
                if _zz:
                    # HOÀN NGUYÊN cố định 1.6 × 2 lượt (2026-11-08): leo thang
                    # 2.2/2.8 ghép nhầm hàng kề trên ảnh phẳng → warp trôi
                    # (091032 tụt 0.6 → 0.09). 090802 vẫn pass ở 1.6.
                    for _ in range(2):
                        _oz, _hn, _ho = OZ.zone_refine(warped, align_cells,
                                                       scale, _zz)
                        if _oz is None:
                            break
                        warped = _oz
                        info["zone_refine"] = round(_hn, 3)
                        info["zone_refine_iters"] = \
                            int(info.get("zone_refine_iters") or 0) + 1
            except Exception:                                # noqa
                pass
        # ---- CỔNG BÁM MẪU (xác thực bám sát mẫu giấy đã chọn) ----
        # hit của refine_grid_by_cells đếm MỌI blob gần vị trí ô (chữ số,
        # khung, nét) nên phiếu in KHÁC layout vẫn đạt grid=1.0. Cổng này đếm
        # VÒNG TRÒN IN đúng vị trí ô — theo TOÀN BỘ ô template ("ALL") hoặc
        # theo từng phần — đo CHÍNH XÁC mức bám mẫu, phiếu in biến thể/sai
        # mẫu bị TỪ CHỐI rõ ràng thay vì warp sai rồi chấm nhầm.
        if cells:
            mins = RING_MIN_PART.get(paper, {})
            if "ALL" in mins:
                all_cells = S.build_template_cells(paper, tpl_path)
                rate, res = ring_grid_stats(warped, all_cells, scale,
                                            enhance_rings=er,
                                            accept_filled=fa)
                # ---- THỬ LẠI với accept_filled (2026-11-06) ----
                # bubble ĐÃ TÔ ĐẬM làm mất lỗ ring trắng nên bộ dò mặc định bỏ
                # sót; bật accept_filled đếm thêm vòng đặc 6-20px vẫn là bằng
                # chứng ring in đúng vị trí (090940: 0.23 → 0.49; ring_part
                # P1 0.49/P2 0.48 đồng đều = đúng mẫu). Chỉ dùng để CỨU khi
                # mặc định chưa đạt; giữ nguyên hành vi cũ khi đã đạt.
                if rate < mins["ALL"] and not fa:
                    _rate_f, _ = ring_grid_stats(warped, all_cells, scale,
                                                 enhance_rings=er,
                                                 accept_filled=True)
                    if _rate_f > rate:
                        info["ring_rate_default"] = round(rate, 2)
                        info["ring_rate_filled"] = round(_rate_f, 2)
                        rate = _rate_f
                info["ring_rate"] = round(rate, 2)
                if rate < mins["ALL"]:
                    # ---- CHẤP NHẬN THEO VÙNG MÀU (2026-11-08, bug thật
                    # 090802 lô 12a7): phiếu cong nặng có ring TOÀN CỤC dao
                    # động 0.46-0.60 QUANH sàn 0.60 (chạy 2 lần ra 2 kết
                    # quả!) trong khi TỪNG VÙNG MÀU cực vững (đỏ 0.65 /
                    # hồng 0.83 / lá 0.81 — xa ngưỡng). Ring toàn cục là
                    # con số biên giới; ring theo vùng mới ổn định. Quy
                    # tắc: MỌI vùng ≥ 0.50 + toàn cục ≥ 0.50 → ĐÚNG MẪU
                    # (phiếu SAI mẫu cho từng vùng 0.03-0.54, đa phần
                    # < 0.45 — ngưỡng 0.50 vẫn tách sạch; webapp vẫn giữ
                    # margin-alt +0.08 riêng cho fallback mẫu dự phòng). ----
                    try:
                        import omr_zones as _OZ
                        _zz = _OZ.zones_for(paper, tpl_path, cells=None)
                        _zr = (_OZ.zone_ring_rates(warped, all_cells, scale,
                                                   _zz) if _zz else {})
                    except Exception:                            # noqa
                        _zr = {}
                    if _zr and len(_zr) >= 2 and rate >= 0.50 \
                            and min(_zr.values()) >= 0.50:
                        info["ring_zone"] = {k: round(v, 2)
                                             for k, v in _zr.items()}
                        info["ring_zone_ok"] = True
                    if not info.get("ring_zone_ok"):
                        if attempt == 0 and not ink_only:
                            # verification có thể ĐẠU qua (corner khít / grid_hit
                            # ≥ 0.5 — blob đếm MỌI nét nên dễ) trong khi warp thực
                            # ra lệch ~15-20 px -> ring tụt (4 ảnh mờ nhất của lô
                            # 2026-11-05: ring 0.2-0.46). Cứu bằng: (1) các ứng
                            # viên hướng xoay còn lại, (2) SIFT (đăng ký đúng
                            # 12/12 ảnh ở build_moi_template).
                            pool = list(cands)
                            try:
                                _sc = _sift_candidate(im2, tpl_path)
                            except Exception:                # noqa
                                _sc = None
                            if _sc is not None:
                                pool.append(_sc)
                            if len(pool) > 1:
                                got = _best_rotation_by_ring(
                                    pool, all_cells, int(Wt * scale),
                                    int(Ht * scale), rot, scale=scale,
                                    enhance_rings=er, accept_filled=fa)
                                if got is not None:
                                    _rr, H, im2, q, _r2 = got
                                    info["rot_by_ring"] = _r2
                                    continue        # ring lại với H mới
                        # ---- RING MỀM (2026-11-06): phiếu NÉT, khung căn KHÍT
                        # (corner ≥ 3/4 + grid_hit ≥ 0.50) nhưng ring hơi thấp —
                        # thường là biến thể in / photocopy nhạt / 1 mốc bị che,
                        # KHÔNG phải phiếu khác mẫu → chấp nhận với cờ ring_soft.
                        # KHÔNG dùng màu (yêu cầu user: ảnh không màu chỉ dùng mốc
                        # hình chữ nhật đen). Moi-sheet vào 12 có ring cực thấp
                        # (< 0.3) nên không qua floor, không lách được cổng mềm. ----
                        _soft = ((info.get("corner") or 0) >= 0.75
                                 and float(info.get("grid_hit") or 0.0) >= 0.50
                                 and rate >= 0.45)
                        if _soft:
                            info["ring_soft"] = True
                        else:
                            info["reason"] = (
                                f"phiếu KHÔNG bám sát mẫu đã chọn — chỉ "
                                f"{rate:.0%} ô trùng vòng in của mẫu "
                                f"(cần ≥ {mins['ALL']:.0%}). Kiểm tra "
                                "phiếu có đúng mẫu giấy này không.")
                            return info
            else:
                prate = ring_part_rates(warped, cells, scale,
                                        enhance_rings=er, accept_filled=fa)
                info["ring_part"] = {k: round(v, 2) for k, v in prate.items()}
                # ---- SÀN TỔNG THỂ (2026-11-06, lô 11a8): phiếu thuộc HỌ MẪU
                # KHÁC (cùng khung 4 mốc góc, layout bên trong khác) có thể
                # LÁCH qua cổng theo-phần nhờ vùng trùng — moi-sheet warp vào
                # 10_11 đạt P1 0.42 (vừa quá 0.40) / P2 0.75 / tổng 0.50 VỪA
                # KHÍT mốc. Tổng thể trên chính các ô keyed tách sạch: phiếu
                # ĐÚNG mẫu 0.62-0.96 (test11 ≥ 0.62, test12_20 ≥ 0.82), phiếu
                # khác mẫu 0.38-0.50 -> sàn 0.55 (dư địa 0.07 so với thật).
                rate_all, _res_all = ring_grid_stats(warped, cells, scale,
                                                     enhance_rings=er,
                                                     accept_filled=fa)
                info["ring_rate"] = round(rate_all, 2)
                # Sàn 0.55 CHỈ với lưới KEYED (webapp — đo 2026-11-06: test11
                # thật 0.62-0.88, test12_20 0.82-0.96, phiếu khác mẫu 0.38-0.50);
                # lưới FULL (build_template_cells — mọi ô mẫu, gồm ô không có
                # trong key) cho phiếu thật chỉ 0.38-0.49 (photocopy nhạt) nên
                # sàn là RING_MIN_DEFAULT 0.50 (trước khi thêm sàn tổng thể).
                _ring_floor = 0.55 if len(cells) <= 150 else 0.32
                # ---- RING MỀM (2026-11-06): như ALL-mode — khung căn KHÍT
                # (corner ≥ 3/4 + grid_hit ≥ 0.50) và ring tổng thể ≥ 0.45
                # → chấp nhận với cờ ring_soft. KHÔNG dùng màu (user: ảnh
                # không màu chỉ dùng mốc hình chữ nhật đen). CHỐNG phiếu
                # khác mẫu: moi-sheet vào 10_11 có ring KHÔNG ĐỒNG ĐỀU
                # theo phần (P1 0.42 vs P2 0.75 — chênh 0.33) trong khi
                # phiếu đúng mẫu đều nhau → nếu chênh lệch max-min của
                # ring_part > 0.20 → KHÔNG soft accept. ----
                _prate_spread = (max(prate.values()) - min(prate.values())
                                 if prate else 0.0)
                _soft = ((info.get("corner") or 0) >= 0.75
                         and float(info.get("grid_hit") or 0.0) >= 0.50
                         and rate_all >= 0.45
                         and _prate_spread <= 0.20)
                if _soft:
                    info["ring_soft"] = True
                if prate and rate_all < _ring_floor and not _soft:
                    info["reason"] = (
                        f"phiếu KHÔNG bám sát mẫu đã chọn — tổng thể chỉ "
                        f"{rate_all:.0%} ô trùng vòng in (cần ≥ "
                        f"{_ring_floor:.0%}). Phiếu có thể thuộc mẫu in "
                        "khác/biến thể.")
                    return info
                bad = {k: v for k, v in prate.items()
                       if v < mins.get(k, RING_MIN_DEFAULT)}
                if prate and bad and not _soft:
                    info["reason"] = ("phiếu KHÔNG bám sát mẫu đã chọn — ô "
                                      "trùng vòng in theo phần: "
                                      + ", ".join(f"{k} {v:.0%}" for k, v
                                                  in sorted(prate.items()))
                                      + f"; không đạt: {sorted(bad)}")
                    return info
        _ms, _t = _prof_tick(_t)
        _prof_log("ring-gate", _ms)
        # ---- CỔNG MÀU (2026-11-06): các vùng cùng màu giữa mẫu và ảnh phải
        # khớp vị trí — phát hiện chéo mẫu lách qua ring-gate (moi-sheet warp
        # vào 10_11 đạt ring 0.50). Chỉ bật khi cả 2 bên có màu. H4 cuối cùng
        # (sample-template -> ảnh) đã có sẵn. ----
        try:
            import omr_coloralign as CA
            if CA.needs_color_check(tpl_path, im2):
                _ca = CA.color_agreement(im2, tpl_path, H)
                info["color_agreement"] = round(_ca, 2) if _ca is not None else None
                if _ca is not None and _ca < 0.40:
                    info["reason"] = (
                        "phiếu KHÔNG bám sát mẫu đã chọn — vùng màu không khớp "
                        f"(khớp {_ca:.0%}). Phiếu có thể thuộc mẫu in khác.")
                    info["ok"] = False
                    return info
        except Exception:                                    # noqa
            pass
        break
    warped = normalize_light(warped)
    # ---- CỔNG CHIỀU 180° (2026-11-08, bug thật 090901 lô 12a7): warp khớp
    # MỐC nhưng NỘI DUNG lộn ngược — khung mốc của mẫu 12 ĐỐI XỨNG 180° nên
    # corner-check không phát hiện, và các đường cứu (DocShadow/SIFT/fallback)
    # có thể nhả warp ngược chiều. So sánh ring rate ảnh hiện tại vs bản xoay
    # 180°: nếu bản xoay KHỚP HƠN rõ (+0.10) → LẤY BẢN XOAY (nội dung về đúng
    # khung mẫu, H=scale·I vẫn hợp lệ). Chặn triệt để kết quả lộn ngược mà
    # ring-gate không thể thấy (ring đo trên warp ngược = thấp → tưởng là do
    # giấy cong rồi qua band-rescue). ----
    if cells:
        try:
            _r0, _ = ring_grid_stats(warped, align_cells or cells, scale,
                                     enhance_rings=er, accept_filled=fa)
            if _r0 < 0.95:
                _r180, _ = ring_grid_stats(
                    cv2.rotate(warped, cv2.ROTATE_180),
                    align_cells or cells, scale, enhance_rings=er,
                    accept_filled=fa)
                if _r180 > _r0 + 0.10:
                    warped = cv2.rotate(warped, cv2.ROTATE_180)
                    info["flip180"] = True
        except Exception:                                    # noqa
            pass
    info.update(ok=True, warped=warped, H=H, reason="", src_img=im2)
    # ---- tuỳ chọn: NHẬN DẠNG + TÔ ĐEN các bubble đã tô (k-means 2 cụm) ----
    if ink and cells:
        try:
            inked, marks, st = mark_filled(warped, cells, scale=scale)
            # ---- phần MÃ ĐỀ / SBD: tô đen đúng các chữ số ĐỌC ĐƯỢC ----
            try:
                import omr_idscan as _ID
                # đọc SBD/Mã đề trên ẢNH GỐC (độ phân giải cao) rồi chiếu vị trí
                # sang hệ ảnh chuẩn hoá để tô đen
                ids = _ID.scan_ids(im2, H, paper, tpl_path)
                Ss = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
                M = Ss @ np.linalg.inv(np.asarray(H, np.float64))
                sc_r = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
                n_id = 0
                for part in ("SBD", "MD"):
                    pts = ids.get("positions", {}).get(part) or []
                    if not pts:
                        continue
                    xy = np.float32([[x, y] for (_c, _d, x, y, _r) in pts])
                    back = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), M).reshape(-1, 2)
                    for (col, dig, _x, _y, r), b in zip(pts, back):
                        cv2.circle(inked, (int(round(b[0])), int(round(b[1]))),
                                   max(3, int(round(r * sc_r * 0.62))), (0, 0, 0), -1,
                                   cv2.LINE_AA)
                        marks[(part, col, dig)] = True
                        n_id += 1
                # ---- BỔ SUNG: nếu KHÔNG đọc được chuỗi số (SBD/Mã đề) thì vẫn
                # TÔ ĐẬM ô có mực đậm nhất trong TỪNG CỘT của lưới ID, để phần
                # "Số báo danh + Mã đề" ở góc phải trên luôn được đánh dấu.
                for part in ("SBD", "MD"):
                    if ids.get("positions", {}).get(part):
                        continue
                    try:
                        import omr_sheetgrid as _S2
                        _idc = _S2.build_id_cells(paper, tpl_path)
                        rp = _ID.read_part(warped, _idc, part, scale=scale)
                    except Exception:                            # noqa
                        rp = None
                    if not rp:
                        continue
                    norm = rp.get("norm")
                    row_digits = rp.get("row_digits") or []
                    if norm is None or not row_digits:
                        continue
                    for ci_idx, ci in enumerate(rp["cols"]):
                        col_vals = norm[ci_idx]
                        j = int(np.argmax(col_vals))
                        if float(col_vals[j]) < 0.10:
                            continue
                        ry = rp["row_of"].get(int(row_digits[j]))
                        x = rp["col_x"].get(ci)
                        y = rp["row_y"].get(ry) if ry is not None else None
                        if x is None or y is None:
                            continue
                        cv2.circle(inked, (int(round(x)), int(round(y))),
                                   max(3, int(round(rp["b_r"] * 0.62))), (0, 0, 0), -1,
                                   cv2.LINE_AA)
                        marks[(part, ci, str(row_digits[j]))] = True
                        n_id += 1
                info["ids"] = {"SBD": ids.get("SBD"), "MD": ids.get("MD")}
                info["ids_result"] = ids        # để bước chấm dùng lại, khỏi đọc 2 lần
                st["n_id"] = n_id
            except Exception as e:                               # noqa
                info["ink_error"] = str(e)
            info.update(inked=inked, marks=marks, ink_stats=st)
        except Exception as e:                                   # noqa
            info["ink_error"] = str(e)
    return info


def prepare_file(path, tpl_path, out_dir, scale=2.0):
    """Chuẩn hoá 1 file và ghi ra đĩa. Trả (đường_dẫn_mới|None, info)."""
    from pathlib import Path
    info = normalize_to_template(path, tpl_path, scale=scale)
    if not info["ok"]:
        return None, info
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(path).stem}_chuan.jpg"
    cv2.imwrite(str(out), info["warped"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out, info
