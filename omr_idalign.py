# -*- coding: utf-8 -*-
"""
OMR ID ALIGN — Định vị RIÊNG khối "Số báo danh + Mã đề" bằng mốc chữ nhật
==========================================================================
Khối SBD/Mã đề nằm ở góc phải trên và được bao bởi các Ô ĐEN CHỮ NHẬT in sẵn
(2 ô vuông lớn ở góc + dãy thanh đen dọc mép phải). Warp toàn trang theo 4 mốc
không đủ chính xác cho vùng nhỏ này (sai số 1-2 px cũng làm lệch cả ô).

Cách làm: dùng chính khối đó trên ẢNH MẪU làm "khuôn", dò lại trong ảnh đã
chuẩn hoá bằng TEMPLATE MATCHING đa tỉ lệ + đa góc (chuẩn hoá tương quan
TM_CCOEFF_NORMED), rồi suy ra một phép biến đổi CỤC BỘ cho vùng này.

Kết quả: ảnh warp riêng cho khối ID — đọc/ tô đen trên ảnh đó rồi chiếu ngược
vị trí về ảnh chuẩn hoá để vẽ.
"""
from __future__ import annotations

import cv2
import numpy as np

ID_PAD = 10.0          # nới rộng vùng ID trên mẫu (px mẫu)


def id_template_box(paper, tpl_path, pad=ID_PAD):
    """Bounding box (x0, y0, x1, y1) của khối SBD/Mã đề trên ẢNH MẪU."""
    import omr_sheetgrid as S
    cells = S.build_id_cells(paper, tpl_path)
    if not cells:
        return None
    xs = [c[3] for c in cells]
    ys = [c[4] for c in cells]
    tpl = cv2.imread(str(tpl_path))
    h, w = tpl.shape[:2]
    x0 = max(0, int(min(xs) - pad))
    y0 = max(0, int(min(ys) - pad))
    x1 = min(w, int(max(xs) + pad))
    y1 = min(h, int(max(ys) + pad))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return x0, y0, x1, y1


def _edges(gray):
    """Ảnh BIÊN (Canny + giãn) — so khớp bền với khác biệt độ đậm/độ tương phản."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    e = cv2.Canny(g, 50, 150)
    return cv2.dilate(e, np.ones((3, 3), np.uint8))


def _patch(tpl_path, box, scale=1.0, angle=0.0):
    img = cv2.imread(str(tpl_path))
    x0, y0, x1, y1 = box
    p = img[y0:y1, x0:x1]
    if scale != 1.0:
        p = cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if abs(angle) > 0.1:
        h, w = p.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        p = cv2.warpAffine(p, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
    return _edges(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY))


def locate_id_block(warped, tpl_path, paper, scale,
                    scales=(0.92, 0.96, 1.0, 1.04, 1.08),
                    angles=(-4, -2, 0, 2, 4), min_score=0.55):
    """Dò khối SBD/Mã đề trong ảnh ĐÃ warp; trả phép biến đổi cục bộ.

    Trả {"score", "M" (3x3, ảnh warp -> ảnh warp đã căn lại), "box"} hoặc None.
    """
    box = id_template_box(paper, tpl_path)
    if box is None:
        return None
    g_img = _edges(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    H, W = g_img.shape
    bx0, by0, bx1, by1 = [int(round(v * scale)) for v in box]
    # vùng tìm: quanh vị trí danh nghĩa ±15% kích thước khối
    pad = max(30, int(0.18 * max(bx1 - bx0, by1 - by0)))
    sx0, sy0 = max(0, bx0 - pad), max(0, by0 - pad)
    sx1, sy1 = min(W, bx1 + pad), min(H, by1 + pad)
    if sx1 - sx0 < 30 or sy1 - sy0 < 30:
        return None
    search = g_img[sy0:sy1, sx0:sx1]

    best = None
    for sc in scales:
        for ang in angles:
            p = _patch(tpl_path, box, scale=scale * sc, angle=ang)
            ph, pw = p.shape[:2]
            if ph >= search.shape[0] or pw >= search.shape[1]:
                continue
            res = cv2.matchTemplate(search, p, cv2.TM_CCORR_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (float(mx), sc, ang, loc, (pw, ph))
    if best is None or best[0] < min_score:
        return None
    score, sc, ang, loc, (pw, ph) = best
    # Vị trí khối ID theo mẫu (hệ warp) và vị trí khối tìm được
    cx_w = (box[0] + box[2]) / 2.0 * scale
    cy_w = (box[1] + box[3]) / 2.0 * scale
    cx_m = sx0 + loc[0] + pw / 2.0
    cy_m = sy0 + loc[1] + ph / 2.0
    # DỊCH ẢNH đi (cx_w-cx_m, cy_w-cy_m) để nội dung khối ID về đúng chỗ mẫu
    dx, dy = cx_w - cx_m, cy_w - cy_m
    return {"score": score, "dx": float(dx), "dy": float(dy),
            "box": box, "scale": sc, "angle": ang}





def shift_image(img, dx, dy):
    """Dịch ảnh (dx, dy) — dùng để đưa khối ID về đúng vị trí mẫu."""
    H, W = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderValue=(255, 255, 255))


def scan_ids_aligned(photo, H, paper, tpl_path, scale=2.0, md_candidates=None,
                     debug=None):
    """Đọc SBD/Mã đề SAU KHI căn cục bộ khối ID bằng mốc chữ nhật.

    Ảnh gốc -> warp về hệ mẫu (H) -> dò lại khối SBD/Mã đề bằng template
    matching trên BIÊN -> warp cục bộ -> đọc -> chiếu vị trí ngược về ảnh gốc.
    Trả dict như omr_idscan.scan_ids (positions ở toạ độ ẢNH GỐC).
    """
    import omr_idscan as ID
    tpl = cv2.imread(str(tpl_path))
    if tpl is None or photo is None:
        raise FileNotFoundError(tpl_path)
    Ht, Wt = tpl.shape[:2]
    Ss = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    M = Ss @ np.linalg.inv(np.asarray(H, np.float64))
    warp = cv2.warpPerspective(photo, M, (int(Wt * scale), int(Ht * scale)),
                               flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    base = ID.scan_ids(warp, Ss, paper, tpl_path, md_candidates=md_candidates,
                       debug=debug)
    loc = locate_id_block(warp, tpl_path, paper, scale)
    if not loc:
        return _to_photo(base, None, H, scale, "lưới chuẩn", 0.0)
    warp2 = shift_image(warp, loc["dx"], loc["dy"])
    res2 = _scan_ids_geo(warp2, paper, tpl_path, scale, md_candidates)
    if not (res2.get("MD") or res2.get("SBD")):
        res2 = ID.scan_ids(warp2, Ss, paper, tpl_path, md_candidates=md_candidates)
    if not (res2.get("MD") or res2.get("SBD")):
        res2, loc = base, None
    return _to_photo(res2, loc, H, scale, "mốc chữ nhật" if loc else "lưới chuẩn",
                     float(loc["score"]) if loc else 0.0)


def _to_photo(res, loc, H, scale, method, score):
    """Chiếu positions từ hệ ảnh-warp2 về hệ ẢNH GỐC."""
    out = dict(res or {})
    back = np.linalg.inv(np.asarray(H, np.float64))   # ảnh gốc -> hệ mẫu
    dx = loc["dx"] if loc else 0.0
    dy = loc["dy"] if loc else 0.0
    pos = {}
    for part, lst in (res.get("positions") or {}).items():
        if not lst:
            continue
        xy = np.float32([[(x - dx) / scale, (y - dy) / scale]
                         for (_c, _d, x, y, _r) in lst])
        p2 = cv2.perspectiveTransform(xy.reshape(-1, 1, 2).astype(np.float64),
                                      back).reshape(-1, 2)
        rr = float(np.median([r for (_c, _d, _x, _y, r) in lst])) / scale
        pos[part] = [(c, d, float(b[0]), float(b[1]), rr)
                     for (c, d, _x, _y, _r), b in zip(lst, p2)]
    out["positions"] = pos
    out["id_method"] = method
    out["id_score"] = round(float(score), 3)
    return out


def _scan_ids_geo(warped, paper, tpl_path, scale, md_candidates=None):
    """scan_ids nhưng đọc từng phần bằng read_part_geo (lưới dò được)."""
    import omr_idscan as ID
    import omr_sheetgrid as S
    id_cells = S.build_id_cells(paper, tpl_path)
    if not id_cells:
        return {"SBD": None, "MD": None, "positions": {}, "raw": {}, "score": {}}
    out = {"SBD": None, "MD": None, "positions": {}, "raw": {}, "score": {}}
    for part in ("SBD", "MD"):
        rp = ID.read_part_geo(warped, id_cells, part, scale=scale)
        if not rp:
            continue
        val = rp["value"]
        if part == "MD" and md_candidates:
            cand, csc, cmargin = ID.resolve_candidates(rp, md_candidates)
            if cand and cand != val:
                free = ID._free_score(rp)
                if cmargin >= ID.MD_SWITCH_MARGIN and csc >= ID.MD_SWITCH_SCORE \
                        and csc >= 0.80 * free:
                    val = cand
            elif cand:
                val = cand
        out[part] = val
        out["raw"][part] = rp["raw"]
        out["score"][part] = rp["detail"]
        out["positions"][part] = ID.positions_for(rp, val) if val else rp["positions"]
    return out
