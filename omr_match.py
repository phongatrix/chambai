# -*- coding: utf-8 -*-
"""
OMR MATCH — Đối chiếu ẢNH ĐÃ CHUẨN HOÁ với LƯỚI ĐÁP ÁN
======================================================
Đầu vào: ảnh đã qua `omr_prep.normalize_to_template` — tức ảnh nằm ĐÚNG hệ toạ
độ mẫu giấy (kích thước = mẫu × scale, không còn nghiêng/phối cảnh).

Quy trình 5 bước:
  1. Dựng LƯỚI ĐÁP ÁN từ mẫu + đề đã chọn: mỗi ô = (phần, câu, lựa chọn,
     toạ độ trên mẫu, đáp án đúng).          -> omr_sheetgrid.build_answer_grid
  2. Chiếu lưới lên ảnh: p_ảnh = p_mẫu × scale  (ảnh đã chuẩn hoá nên KHÔNG
     cần homography nữa — sai số chỉ còn dưới 1 px).
  3. Tinh chỉnh từng ô ±0,22 bước lưới theo ĐÁP ỨNG VÒNG TRÒN (bám tâm vòng
     in thật), rồi đo mực trong lòng vòng (vành khuyên + tâm) -> điểm "đã tô".
  4. Quyết định mỗi câu: trống / một đáp án / tô nhiều — ngưỡng mực TỰ THÍCH
     ỨNG theo ảnh (xem omr_sheetgrid.adapt_thresh).
  5. So với đáp án -> đúng/sai từng câu, điểm P1/P2/P3, tổng; kèm bảng đối
     chiếu chi tiết và ảnh kết quả 3 màu (xanh/đỏ/vàng).

Nếu bước 3 không đủ ô bám được vòng tròn (< 60%) thì báo "lệch lưới" thay vì
chấm bừa.
"""
from __future__ import annotations

import cv2
import numpy as np

import omr_idscan as ID
import omr_sheetgrid as S

MIN_REG = 0.60          # tỉ lệ ô bám được vòng tròn tối thiểu để tin kết quả


# ---------- 1-3. đọc mực trên lưới ----------

BUBBLE_TEMPLATE_PX = 7.0     # đường kính bubble trên ảnh mẫu (như read_cells)


def bubble_radius(scale=1.0):
    """Bán kính bubble trên ảnh đã chuẩn hoá (mẫu × scale)."""
    return 0.5 * BUBBLE_TEMPLATE_PX * scale


def _ring_resp(gray, r):
    """Đáp ứng vành khuyên với bán kính r (px) — không phụ thuộc bước lưới."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32) / 255.0
    dark = 1.0 - g
    K = int(2 * r + 3)
    if K % 2 == 0:
        K += 1
    yy, xx = np.mgrid[0:K, 0:K]
    c = (K - 1) / 2.0
    d = np.hypot(xx - c, yy - c)
    ker = ((d >= 0.60 * r) & (d <= 1.18 * r)).astype(np.float32)
    ker /= max(1.0, ker.sum())
    return cv2.filter2D(dark, -1, ker)


def read_grid(warped, cells, scale=2.0):
    """Đọc mực tại từng ô của lưới trên ảnh ĐÃ CHUẨN HOÁ.

    Vì ảnh đã nằm đúng hệ mẫu (mẫu × scale), phép biến đổi chỉ là phép scale —
    dùng lại `omr_sheetgrid.read_cells` (đã kiểm chứng trên bộ 49 trang và
    test_sheetgrid) để giữ NGUYÊN thang mực/norm<0,78 như luồng chấm cũ.

    Trả (reads, reg) với reg = tỉ lệ ô mà vị trí đọc được còn nằm trong 0,6 bán
    kính bubble so với toạ độ mẫu (đo độ "bám lưới").
    """
    H = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    reads = S.read_cells(warped, cells, H, exp_w=BUBBLE_TEMPLATE_PX * scale)
    r_bub = bubble_radius(scale)
    good = 0
    for (part, cau, opt, cx, cy) in cells:
        v = reads.get((part, cau, opt))
        if v is None:
            continue
        if np.hypot(v[1] - cx * scale, v[2] - cy * scale) <= 0.6 * r_bub:
            good += 1
    return reads, good / max(1, len(cells))


# ---------- 4-5. quyết định + so đáp án ----------

def adaptive_thresh(reads, lo=0.08, hi=0.40, margin=0.05):
    """Ngưỡng mực cho ảnh ĐÃ CHUẨN HOÁ.

    Dùng TRUNG VỊ (p50) làm mức nền: trên ảnh đã căn mẫu, tuyệt đại đa số ô là
    trống nên p50 ≈ mực in nền. Cộng biên 0,05 để tách ô có tô. Đo thực tế: ảnh
    bút chì mờ PXL_...120214113 có p50=0,06, ô tô chỉ 0,16 -> ngưỡng cũ
    (p65+0,10 = 0,19) bỏ sót 8 câu; ngưỡng mới 0,11 chỉ còn 1 câu trống thật.
    """
    if not reads:
        return 0.20
    sc = np.asarray([v[0] for v in reads.values()], np.float32)
    return float(np.clip(float(np.percentile(sc, 50)) + margin, lo, hi))


def decide_reads(reads, thresh=None, rel=0.72):
    if thresh is None:
        thresh = adaptive_thresh(reads)
    return S.decide(reads, None, thresh=thresh, rel=rel)


def compare_rows(detail, grid, paper):
    """Bảng đối chiếu từng câu để hiển thị/xuất Excel."""
    rows = []
    for (part, cau), (st, exp, got) in sorted(detail.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if isinstance(cau, tuple):
            cau_txt = f"{cau[0]}{cau[1]}"
        else:
            cau_txt = cau
        rows.append({
            "phần": part,
            "câu": cau_txt,
            "đáp án": exp if exp is not None else "",
            "bài làm": got if got is not None else "",
            "trạng thái": {"ok": "đúng", "wrong": "sai",
                           "blank": "trống", "multi": "tô nhiều"}.get(st, st),
            "điểm": 0.25 if st == "ok" else 0.0,
        })
    return rows


def match_normalized(warped, paper, tpl_path, key, de, scale=2.0, thresh=None):
    """ĐỐI CHIẾU ảnh đã chuẩn hoá với lưới đáp án của một đề.

    Trả dict: {"ok", "reg", "rows", "detail", "stats", "p1", "p2", "tong",
               "decided", "grid"}.
    """
    grid = S.build_answer_grid(paper, tpl_path, key["các đề"][de])
    # chỉ đọc các ô CÓ đáp án trong key (P1 đủ 4 lựa chọn, P2 đủ Đ/S, bỏ P3)
    cells = [c for c in grid["cells"]
             if grid["answers"].get((c[0], c[1], c[2]))]
    reads, reg = read_grid(warped, cells, scale=scale)
    out = {"ok": False, "reg": round(reg, 3), "rows": [], "detail": {},
           "stats": {}, "p1": 0.0, "p2": 0.0, "tong": 0.0,
           "decided": {}, "grid": grid, "reason": ""}
    if reg < MIN_REG:
        out["reason"] = (f"lưới không bám được vòng tròn ({reg:.0%} < {MIN_REG:.0%}) "
                         f"— ảnh chưa chuẩn hoá đúng mẫu")
        return out
    dec = decide_reads(reads, thresh=thresh)
    s1, s2, detail, stats = S.grade(dec, grid)
    out.update(ok=True, decided=dec, detail=detail, stats=stats,
               p1=s1, p2=s2, tong=round(s1 + s2, 2),
               rows=compare_rows(detail, grid, paper))
    return out


def match_all(key, warped, paper, tpl_path, scale=2.0, thresh=None):
    """Đối chiếu ảnh với TẤT CẢ đề trong key, trả về đề khớp nhất.

    Dùng khi chưa biết mã đề: đề nào cho nhiều câu "đúng" nhất là đề của bài.
    """
    best = None
    for de in key["các đề"]:
        r = match_normalized(warped, paper, tpl_path, key, de, scale=scale, thresh=thresh)
        if not r["ok"]:
            continue
        n_ok = r["stats"]["P1"]["ok"] + r["stats"]["P2"]["ok"]
        if best is None or n_ok > best[0]:
            best = (n_ok, de, r)
    if best is None:
        return None, None
    return best[1], best[2]
