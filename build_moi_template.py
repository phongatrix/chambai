# -*- coding: utf-8 -*-
"""DỰNG MẪU GIẤY MỚI (moi.png) TỪ BỘ ẢNH CHỤP PHIẾU CÙNG LOẠI
=========================================================
Mẫu giấy mới (đề 2025: PHẦN I 20 câu ABCD + PHẦN II 4 câu Đ/S +
PHẦN III 8 câu điền số + lưới Mã đề 3 chữ số) KHÔNG có bản in sạch —
chỉ có ảnh chụp điện thoại (mỗi tờ có chữ viết/tô của học sinh khác nhau,
một nửa số ảnh bị lộn ngược 180°).

Thuật toán (chạy 1 lần, không tốn API):
  1. Đọc mọi ảnh trong thư mục, thu về cạnh dài ~2300 px.
  2. Chọn ảnh NÉT NHẤT (var Laplacian) làm khung tham chiếu.
  3. SIFT + RANSAC đăng ký MỌI ảnh về khung tham chiếu — SIFT bất biến
     xoay nên ảnh lộn ngược 180° tự động khớp đúng chỗ, không cần xoay trước.
  4. Chuẩn hoá sáng từng ảnh (ink-map: chia nền cục bộ) rồi lấy PHÂN VỊ 40
     theo pixel: nét in (đậm trên MỌI tờ) giữ nguyên; chữ viết/tô của từng
     học sinh (chỉ xuất hiện trên ≤ 4/12 tờ) biến mất.
  5. Kéo tương phản, cắt đúng khung giấy, xuất data/maugiay/moi.png.

Kèm `--gemini`: gửi ảnh mẫu sạch cho Gemini đọc CẤU TRÚC (tên phần, dải y,
số câu, số ý, lưới Mã đề) dưới dạng JSON — dùng sinh config PAPERS['moi'],
luôn được kiểm chứng chéo bằng clustering hình học trước khi tin.

Kết quả in ra kèm báo cáo tự kiểm tra: số mốc định vị, 4 ô cực trị,
số vòng tròn in, cấu trúc từng phần (P1 20×4, P2, P3, Mã đề).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

WORK = 2304          # cạnh dài khi đăng ký
OUT_SIDE = 1600      # cạnh dài bản render đầy đủ (đo lattice trên bản này)
FINAL_SIDE = 600     # cạnh cao mẫu XUẤT RA — cỡ mẫu cũ để chấm nhanh
PCT_INK = 20         # phân vị ink-map: 20 = nét in chắc (chỉ cần đậm trên ≥3/12 tờ)
GAIN = 3.4           # hệ số kéo tương phản mực
SIFT_MIN_INLIERS = 30


# ---------- 1. đọc + đăng ký ----------

def load_gray(path, side=WORK):
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = im.shape[:2]
    s = side / float(max(w, h))
    if s < 1.0:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return im


def ink_map(g8):
    g = g8.astype(np.float32)
    bg = cv2.blur(g, (61, 61))
    return np.clip(g / (bg + 1e-6), 0.0, 1.2)


def register_stack(paths):
    """Đăng ký mọi ảnh về khung của ảnh nét nhất. Trả (mảng ink đã đăng ký, báo cáo)."""
    cv2.setRNGSeed(20261105)
    imgs, sharp = [], []
    for p in paths:
        im = load_gray(p)
        imgs.append((Path(p).name, im))
        sharp.append((float(cv2.Laplacian(im, cv2.CV_64F).var()), Path(p).name))
    sharp.sort(reverse=True)
    ref_name = sharp[0][1]
    ref = next(im for n, im in imgs if n == ref_name)
    print(f"[1] ảnh tham chiếu (nét nhất): {ref_name} "
          f"(Laplacian {sharp[0][0]:.0f}; thấp nhất {sharp[-1][0]:.0f})")

    sift = cv2.SIFT_create(nfeatures=8000)
    kp2, de2 = sift.detectAndCompute(ref, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    h, w = ref.shape[:2]
    stack, rep = [], []
    for name, im in imgs:
        kp1, de1 = sift.detectAndCompute(im, None)
        mt = flann.knnMatch(de1, de2, k=2)
        good = [m for m, n in mt if m.distance < 0.7 * n.distance]
        if len(good) < SIFT_MIN_INLIERS:
            rep.append({"file": name, "ok": False, "reason": f"ít match ({len(good)})"})
            print(f"    - {name}: BỎ (chỉ {len(good)} match SIFT)")
            continue
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hm is None or mask is None or int(mask.sum()) < SIFT_MIN_INLIERS:
            rep.append({"file": name, "ok": False, "reason": "RANSAC yếu"})
            print(f"    - {name}: BỎ (RANSAC < {SIFT_MIN_INLIERS} inliers)")
            continue
        ink = cv2.warpPerspective(ink_map(im), Hm, (w, h),
                                  flags=cv2.INTER_LINEAR, borderValue=1.0)
        stack.append(ink)
        rep.append({"file": name, "ok": True, "inliers": int(mask.sum()),
                    "matches": len(good)})
        print(f"    + {name}: {int(mask.sum())}/{len(good)} inliers")
    if len(stack) < 3:
        raise SystemExit("Quá ít ảnh đăng ký được — không dựng được mẫu.")
    return np.stack(stack), rep, ref_name


# ---------- 2.Render mẫu ----------

SHIP_PCT = 50         # phân vị bản XUẤT: ≥50 = chữ viết/tô của ≤ nửa số tờ biến
                      # mất hoàn toàn. Bản này chỉ dùng cho MỐC ĐỊNH VỊ (ô đen
                      # in đậm trên 12/12 tờ) — KHÔNG dùng dò vòng: lưới ô lấy
                      # từ bản PCT_INK (vòng đậm hơn) qua lattice fit. Ghost
                      # fill còn lại ở bản PCT thấp bị detect_markers nuốt thành
                      # hàng trăm marker GIẢ -> _match_fiducials O(n²) nổ 60 s/ảnh.


def _render_from_med(med, crop=None):
    """Ink-map trung vị (float) -> ảnh gray mực đen/nền trắng, đã cắt khung.

    crop=None: tự tính rect cắt từ bản này và TRẢ VỀ (ảnh, rect). Khi truyền
    crop có sẵn: áp đúng rect đó và trả (ảnh, crop) — dùng để crop bản ship
    CÙNG RECT với bản rings: nếu mỗi bản tự crop, hai khung lệch nhau 24-29 px
    (đo thật 2026-11-05) làm lưới ô (đo trên bản rings) KHÔNG khớp template
    (bản ship) — 4/12 ảnh mờ bị ICP kéo không nổi khoảng lệch này."""
    out = 255 - np.clip((1.0 - med) * GAIN, 0.0, 1.0) * 255
    out = out.astype(np.uint8)
    out = cv2.medianBlur(out, 3)
    h, w = out.shape
    if crop is None:
        # cắt sát khung giấy: hàng/cột có mực thật (loại dải nền rìa)
        dark = (out < 150)
        rows = np.where(dark.mean(axis=1) > 0.004)[0]
        cols = np.where(dark.mean(axis=0) > 0.004)[0]
        if len(rows) and len(cols):
            y0, y1 = max(0, rows[0] - int(0.01 * h)), min(h, rows[-1] + int(0.01 * h))
            x0, x1 = max(0, cols[0] - int(0.01 * w)), min(w, cols[-1] + int(0.01 * w))
        else:
            y0, y1, x0, x1 = 0, h, 0, w
        crop = (y0, y1, x0, x1)
    y0, y1, x0, x1 = crop
    out = out[y0:y1, x0:x1]
    h, w = out.shape[:2]
    s = OUT_SIDE / float(max(w, h))
    if s < 1.0:
        out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    # chuẩn hoá nền về trắng thật (nét in giữ nguyên tỉ lệ)
    bg = cv2.blur(out.astype(np.float32), (51, 51))
    out = np.clip(out.astype(np.float32) * (240.0 / np.maximum(bg, 1.0)),
                  0, 255).astype(np.uint8)
    return out, crop


def render_template(stack):
    """Trả (bản_lattice_PCT_INK, bản_xuất_SHIP_PCT) CÙNG KHUNG, cùng chiều.

    Hai bản phải crop bằng CÙNG rect (xem _render_from_med) — nếu không, lưới
    ô và template lệch nhau vài chục px."""
    med_rings = np.percentile(stack, PCT_INK, axis=0)
    med_ship = np.percentile(stack, SHIP_PCT, axis=0)
    tpl_rings, crop = _render_from_med(med_rings)
    tpl_ship, _ = _render_from_med(med_ship, crop=crop)
    # quyết định xoay 1 LẦN trên bản rings (vòng đậm -> đếm ring chắc), áp cho
    # cả bản ship (ring mờ ở SHIP_PCT, tự đo không đủ tin)
    tpl_rings, rotated = fix_orientation(tpl_rings)
    if rotated:
        tpl_ship = cv2.rotate(tpl_ship, cv2.ROTATE_180)
    return tpl_rings, tpl_ship


def fix_orientation(out):
    """Xoay 180° nếu phiếu đang LỘN, đo bằng SỐ VÒNG TRÒN IN hai nửa trang.

    Ảnh tham chiếu có thể là ảnh chụp lộn ngược — SIFT bất biến xoay nên
    đăng ký vẫn khớp, nhưng mẫu xuất ra phải đúng chiều để nhãn P1/P2/P3
    (đánh số câu từ trên xuống) không bị ngược. Đếm mực theo mật độ là bị
    lừa (khối PHẦN III dày mực hơn header) — đếm VÒNG TRÒN IN thì chắc:
    phiếu xuôi chiều: PHẦN III (đáy) ≫ Mã đề (đỉnh, ~30 vòng).
    Trả (ảnh_đã_đúng_chiều, có_xoay_không)."""
    h, w = out.shape[:2]
    sys.path.insert(0, str(Path(__file__).parent))
    import omr_sheetgrid as S
    rings = S._ring_centers(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
    if len(rings) < 20:
        return out, False
    top = int((rings[:, 1] < 0.35 * h).sum())
    bot = int((rings[:, 1] > 0.60 * h).sum())
    if top > bot * 1.5:
        print(f"    (mẫu lộn ngược: {top} vòng ở đỉnh vs {bot} ở đáy — xoay 180°)")
        return cv2.rotate(out, cv2.ROTATE_180), True
    return out, False


# ---------- 3. lưới ô (lattice fit) + tự kiểm tra cấu trúc ----------

# Cửa sổ tìm theo TỈ LỆ ảnh (x0, x1, y0, y1) — bản in này cố định layout:
#   P1: cột trái dưới   | P2: dải giữa trên | MD: cụm phải trên
MOI_WINDOWS = {
    "P1": (0.09, 0.26, 0.42, 0.80),
    "P2": (0.26, 0.80, 0.32, 0.50),
    "MD": (0.74, 1.00, 0.12, 0.34),
}
MOI_SHAPE = {"P1": (20, 4), "P2": (4, 8)}   # (số hàng, số cột cụm)


def _cluster_1d(vals, tol):
    vals = sorted(vals)
    g = [[vals[0]]]
    for v in vals[1:]:
        if v - g[-1][-1] <= tol:
            g[-1].append(v)
        else:
            g.append([v])
    return [(float(np.mean(c)), len(c)) for c in g]


def _fit_lattice_1d(coords, expect_n, tol_keep):
    """Khớp lưới đều `expect_n` điểm từ toạ độ 1D (thiếu điểm vẫn nội suy).

    Phiếu thật có NHIỄU giữa các hàng: header "A B C D" lặp lại giữa khối
    P1 (sau câu 10), cụm rác 1-2 thành viên… nên cách "nội suy từ 2 biên"
    bị cụm rác đánh lừa. Ở đây thử MỌI cặp (cụm_đầu, cụm_cuối) mà bước
    span/(expect_n−1) ≈ pitch (trung vị khoảng cách cụm kế), chọn cặp cho
    nhiều điểm lưới được bám nhất. Trả (grid, số điểm bám, số cụm thô)
    hoặc None."""
    if not coords:
        return None
    cl = _cluster_1d(coords, tol_keep)
    if len(cl) < 3:
        return None
    med_n = float(np.median([n for _c, n in cl]))
    cl = [c for c in cl if c[1] >= max(2.0, 0.5 * med_n)]     # bỏ cụm rác
    if len(cl) < expect_n - 2:
        return None
    centers = [c for c, _n in cl]
    diffs = [b - a for a, b in zip(centers, centers[1:])]
    pitch = float(np.median(diffs))
    if pitch <= 0:
        return None
    best = None
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            span = centers[j] - centers[i]
            k = span / (expect_n - 1)
            if not (0.72 * pitch <= k <= 1.38 * pitch):
                continue
            grid = [centers[i] + t * k for t in range(expect_n)]
            hit = sum(1 for g in grid
                      if any(abs(c - g) <= 0.5 * k for c in centers))
            if best is None or hit > best[0]:
                best = (hit, grid, k)
    if best is None or best[0] < expect_n * 0.6:
        return None
    return best[1], best[0], len(cl)


def extract_cells(tpl):
    """Lưới ô ngữ nghĩa của mẫu moi bằng LATTICE FIT (bền vững hơn đếm-cụm
    của build_template_cells: ô sót detection được nội suy từ 2 biên).

    `tpl` là đường dẫn ảnh HOẶC ảnh gray/BGR đã render (đo trên BẢN ĐẦY ĐỦ
    1600 px — gọi trước khi thu mẫu về 600 px).

    Trả (cells, report). ĐẢM BẢO đủ: P1 20×4 = 80 ô, P2 4 câu × 4 ý × Đ/S
    = 32 ô, Mã đề 3 cột × 10 hàng. Lỗi cấu trúc → report có khoá *_error
    và cells rỗng (caller không được dùng)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import omr_sheetgrid as S
    img = tpl if isinstance(tpl, np.ndarray) else cv2.imread(str(tpl))
    h, w = img.shape[:2]
    bs = S._upscaled_bubbles_img(img)
    cells, rep = [], {}

    # ---- P1: 4 cột × 20 hàng ----
    x0, x1, y0, y1 = MOI_WINDOWS["P1"]
    p1 = [b for b in bs if x0 * w <= b[0] <= x1 * w and y0 * h <= b[1] <= y1 * h]
    cols = [c for c, n in _cluster_1d([b[0] for b in p1], 6) if n >= 8]
    if len(cols) != 4:
        rep["P1_error"] = f"cột {len(cols)} thay vì 4: {[round(c) for c in cols]}"
        return cells, rep
    lat = _fit_lattice_1d([b[1] for b in p1], 20, 5.0)
    if lat is None:
        rep["P1_error"] = "không khớp lưới 20 hàng"
        return cells, rep
    rys, hit, _ = lat
    rep["P1"] = {"rows_hit": f"{hit}/20", "cols": [round(c) for c in cols]}
    for ri, ry in enumerate(rys):
        for ci, cx in enumerate(cols):
            cells.append(("P1", ri + 1, "ABCD"[ci], round(cx, 2), round(ry, 2)))

    # ---- P2: 8 cột (4 câu × Đ/S) × 4 hàng (a-d) ----
    x0, x1, y0, y1 = MOI_WINDOWS["P2"]
    p2 = [b for b in bs if x0 * w <= b[0] <= x1 * w and y0 * h <= b[1] <= y1 * h]
    cols2 = [c for c, n in _cluster_1d([b[0] for b in p2], 6) if n >= 4]
    if len(cols2) != 8:
        rep["P2_error"] = f"cột {len(cols2)} thay vì 8: {[round(c) for c in cols2]}"
        return cells, rep
    lat2 = _fit_lattice_1d([b[1] for b in p2], 4, 5.0)
    if lat2 is None:
        rep["P2_error"] = "không khớp lưới 4 hàng (a-d)"
        return cells, rep
    rys2, hit2, _ = lat2
    rep["P2"] = {"rows_hit": f"{hit2}/4", "cols": [round(c) for c in cols2]}
    for ri, ry in enumerate(rys2):
        for ci, cx in enumerate(cols2):
            cau = ci // 2 + 1
            side = "D" if ci % 2 == 0 else "S"
            cells.append(("P2", (cau, chr(97 + ri)), side, round(cx, 2),
                          round(ry, 2)))

    # ---- Mã đề: 3 cột × 10 hàng (0-9) ----
    x0, x1, y0, y1 = MOI_WINDOWS["MD"]
    md = [b for b in bs if x0 * w <= b[0] <= x1 * w and y0 * h <= b[1] <= y1 * h]
    cols3 = [c for c, n in _cluster_1d([b[0] for b in md], 6) if n >= 3]
    if not cols3:
        rep["MD_error"] = "không thấy cột Mã đề"
        return cells, rep
    lat3 = _fit_lattice_1d([b[1] for b in md], 10, 4.0)
    if lat3 is None:
        rep["MD_error"] = "không khớp lưới 10 hàng Mã đề"
        return cells, rep
    rys3, hit3, _ = lat3
    rep["MD"] = {"rows_hit": f"{hit3}/10", "cols": len(cols3)}
    for ci, cx in enumerate(cols3):
        for ri, ry in enumerate(rys3):
            cells.append(("MD", ci + 1, str(ri), round(cx, 2), round(ry, 2)))
    return cells, rep


def structure_report(tpl_path):
    """Đếm mốc/vòng tròn và kiểm tra cấu trúc từng phần trên ảnh mẫu."""
    sys.path.insert(0, str(Path(__file__).parent))
    import omr_prep as P
    import omr_sheetgrid as S

    img = cv2.imread(str(tpl_path))
    h, w = img.shape[:2]
    rep = {"size": [w, h]}
    quad, ms = P.template_fiducials(tpl_path)
    rep["n_markers"] = len(ms)
    rep["fiducial_quad"] = [[round(x, 1), round(y, 1)] for x, y in quad]
    if ms:
        tl = min(ms, key=lambda m: m[0] + m[1])
        tr = max(ms, key=lambda m: m[0] - m[1])
        br = max(ms, key=lambda m: m[0] + m[1])
        bl = min(ms, key=lambda m: m[0] - m[1])
        rep["extremes"] = {"TL": [round(tl[0]), round(tl[1])],
                           "TR": [round(tr[0]), round(tr[1])],
                           "BR": [round(br[0]), round(br[1])],
                           "BL": [round(bl[0]), round(bl[1])]}
    bs = S._upscaled_bubbles(tpl_path)
    rep["n_bubbles"] = len(bs)

    def rowbands(sel, tol=3.5):
        out = []
        for b in sorted(sel, key=lambda z: z[1]):
            if out and b[1] - out[-1][0] <= tol:
                out[-1][1].append(b)
            else:
                out.append([b[1], [b]])
        return out

    def ncol(mem, gap=8):
        cx = sorted(m[0] for m in mem)
        c = 1
        for a, b in zip(cx, cx[1:]):
            if b - a > gap:
                c += 1
        return c

    # P1: cột trái
    p1 = [b for b in bs if b[0] < 0.28 * w]
    rows4 = [r for r in rowbands(p1) if ncol(r[1]) == 4]
    rep["p1_rows_4cols"] = len(rows4)
    if rows4:
        rep["p1_y"] = [round(rows4[0][0], 1), round(rows4[-1][0], 1)]
    # Mã đề: phải trên
    md = [b for b in bs if b[0] > 0.70 * w and b[1] < 0.40 * h]
    rep["md_rows"] = len(rowbands(md))
    # P2 / P3: giữa phải
    mid = [b for b in bs if b[0] > 0.28 * w]
    rep["mid_rows"] = len(rowbands(mid))
    return rep


# ---------- 4. Gemini đọc cấu trúc (tuỳ chọn) ----------

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACKS = ["gemini-3.7-flash", "gemini-2.5-flash"]
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")
STRUCT_PROMPT = """Bạn là chuyên gia phân tích phiếu trắc nghiệm OMR của Việt Nam.
Ảnh này là MẪU GIẤY ĐÃ SẠCH (đã trừ chữ viết học sinh). Hãy trả về CHÍNH XÁC
một khối JSON (không thêm chữ nào khác) mô tả cấu trúc:

{"phan_i": {"tu_cau": 1, "so_cau": 20, "lua_chon": "ABCD", "vi_tri": "cot trai",
            "y_top_frac": 0.55, "y_bot_frac": 0.82},
 "phan_ii": {"so_cau": 4, "y_kieu": ["a","b","c","d"], "dang": "Dung/Sai",
             "y_top_frac": 0.36, "y_bot_frac": 0.52},
 "phan_iii": {"so_cau": 8, "dang": "dien_so", "chu_so": "0-9", "dau": ["-",","],
              "y_top_frac": 0.52, "y_bot_frac": 0.97},
 "ma_de": {"so_chu_so": 3, "x_left_frac": 0.75, "y_top_frac": 0.20,
           "y_bot_frac": 0.36},
 "sbd": {"cach_ghi": " viet tay", "co_luoi_bubble": false}}

Trong đó *_frac là TOẠ ĐỘ PHẦN TRĂM chiều cao/chiều ngang ảnh (0.0 = mép trên).
Chỉ dựa vào những gì THẤY được trên ảnh; nếu không chắc về một mục hãy đo
khoảng cách giữa các khối trên ảnh cẩn thận."""


def gemini_structure(jpg_bytes, api_key, timeout=60):
    import base64
    import time as _t
    body = {
        "contents": [{"parts": [
            {"text": STRUCT_PROMPT},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(jpg_bytes).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2000},
    }
    last = None
    for model in [GEMINI_MODEL] + GEMINI_FALLBACKS:
        url = GEMINI_URL.format(model=model, key=api_key)
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            txt = ""
            for cand in (data.get("candidates") or []):
                for part in ((cand.get("content") or {}).get("parts") or []):
                    txt += part.get("text") or ""
            return _parse_json(txt)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (503, 429, 404):
                _t.sleep(3)
                continue
            raise
    raise last


def _parse_json(txt):
    import re
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("Gemini không trả JSON: " + txt[:200])
    return json.loads(m.group(0))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("indir", help="thư mục chứa ảnh chụp phiếu cùng loại")
    ap.add_argument("-o", "--out", default=None,
                    help="đường dẫn mẫu xuất ra (mặc định data/maugiay/moi.png)")
    ap.add_argument("--gemini", action="store_true",
                    help="gửi ảnh mẫu sạch cho Gemini đọc cấu trúc JSON")
    ap.add_argument("--key-file", default=None,
                    help="file chứa Gemini API key (mặc định data/gemini_key.txt)")
    args = ap.parse_args()
    root = Path(__file__).parent
    out = Path(args.out) if args.out else root / "data" / "maugiay" / "moi.png"

    paths = sorted([p for p in Path(args.indir).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not paths:
        raise SystemExit(f"Không có ảnh trong {args.indir}")
    print(f"[0] {len(paths)} ảnh trong {args.indir}")

    stack, reg_rep, ref_name = register_stack(paths)
    print(f"[2] median ink-map {len(stack)} ảnh — bản rings P{PCT_INK} (đo lattice) "
          f"+ bản xuất P{SHIP_PCT} (sạch ghost) → render")
    tpl_rings, tpl = render_template(stack)

    # ---- LƯỚI Ô đo trên bản RINGS ĐẦY ĐỦ (1600 px — vòng đậm, lattice chắc) ----
    print("[4b] lưới ô (lattice fit, trên bản rings đầy đủ):")
    cells, crep = extract_cells(tpl_rings)
    print(json.dumps(crep, ensure_ascii=False, indent=2))
    n_p1 = sum(1 for c in cells if c[0] == "P1")
    n_p2 = sum(1 for c in cells if c[0] == "P2")
    n_md = sum(1 for c in cells if c[0] == "MD")
    if not cells or n_p1 != 80 or n_p2 != 32 or n_md != 30:
        raise SystemExit(f"!! lưới ô sai: P1={n_p1} (cần 80), P2={n_p2} (cần 32), "
                         f"MD={n_md} (cần 30) — KHÔNG ghi cells.")

    # ---- THU MẪU (bản xuất) về cỡ mẫu cũ (~600 px cao) để chấm NHANH:
    # mọi bước trong pipeline (dò mốc, _match_fiducials, _ring_centers) tốn
    # theo diện tích ảnh warp — mẫu 1083×1600 làm normalize chậm ~100 s/ảnh.
    h0, w0 = tpl.shape[:2]
    s = FINAL_SIDE / float(h0)
    tpl = cv2.resize(tpl, (int(round(w0 * s)), FINAL_SIDE),
                     interpolation=cv2.INTER_AREA)
    cells = [(p, q, o, round(x * s, 2), round(y * s, 2))
             for p, q, o, x, y in cells]
    for k in ("P1", "P2", "MD"):
        crep.pop(k, None)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tpl, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"[3] đã lưu mẫu: {out} ({tpl.shape[1]}×{tpl.shape[0]}) — cells đã "
          f"scale ×{s:.3f}")

    cj = out.parent / "moi_cells.json"
    cj.write_text(json.dumps({
        "template": out.name,
        "size": [tpl.shape[1], tpl.shape[0]],
        "cells": cells,
        "report": crep,
    }, ensure_ascii=False, indent=1))
    print(f"    OK: P1={n_p1} P2={n_p2} MD={n_md} → {cj}")

    # vẽ preview: mốc đỏ + bubble xanh + ô lattice xanh lá
    import omr_prep as P
    import omr_sheetgrid as S
    vis = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR) if tpl.ndim == 2 else tpl.copy()
    for x, y, _w, _h in P.template_markers(out):
        cv2.rectangle(vis, (int(x - _w / 2), int(y - _h / 2)),
                      (int(x + _w / 2), int(y + _h / 2)), (0, 0, 255), 1)
    for _part, _cq, _opt, cx, cy in cells:
        cv2.circle(vis, (int(cx), int(cy)), 7, (0, 180, 0), 1)
    prev = out.parent / "moi_preview.jpg"
    cv2.imwrite(str(prev), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[5] preview (mốc đỏ, ô lattice xanh lá): {prev}")

    if args.gemini:
        kf = Path(args.key_file) if args.key_file else root / "data" / "gemini_key.txt"
        api_key = kf.read_text().strip() if kf.is_file() else ""
        if not api_key:
            print("!! --gemini nhưng không có key — bỏ qua")
            return
        print("[6] Gemini đọc cấu trúc mẫu…")
        ok, buf = cv2.imencode(".jpg", tpl, [cv2.IMWRITE_JPEG_QUALITY, 90])
        st = gemini_structure(buf.tobytes(), api_key)
        sj = out.parent / "moi_structure.json"
        sj.write_text(json.dumps(st, ensure_ascii=False, indent=2))
        print(f"    đã lưu {sj}")
        print(json.dumps(st, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
