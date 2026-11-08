# -*- coding: utf-8 -*-
"""VÙNG MÀU TRÊN MẪU GIẤY (2026-11-08, yêu cầu user):

Mẫu tc10_11.png / tc12.png (bản RGB GIỐNG HỆT 10_11.png / 12.png + VẼ THÊM
KHUNG MÀU quanh từng phần — px khác biệt chỉ nằm trên viền màu) dùng để ĐỊNH
VỊ các phần của phiếu:

    tc10_11 (399×561):  ĐỎ(H≈0)   = Mã đề (ID, 30 ô)
                        HỒNG(H≈167) = PHẦN I (72 bubble, 18 câu × A-D)
                        XANH DƯƠNG(H≈116) = PHẦN II (60 bubble, 10 câu Đ/S)
                        XANH LÁ(H≈64) = PHẦN III (2 khung, lưới chữ số)
    tc12 (461×661):     ĐỎ   = SBD+MĐ (ID, y<205) VÀ PHẦN I (160 bubble, 40 câu)
                        HỒNG = PHẦN II Câu 1-6 (48 bubble)
                        XANH LÁ = PHẦN II Câu 7-8 (16) + PHẦN III (lưới chữ số)

Quy tắc user: "các vùng có cùng màu sẽ cùng 1 phần, mỗi mẫu giấy có 3 phần".
Màu = NHÓM VẬT LÝ (giấy cong làm các vùng cùng màu dịch chuyển cùng nhau) —
nhãn "phần" của vùng ghi theo màu; nhãn chấm điểm vẫn theo ô (c[0]=P1/P2/P3).

Dùng cho:
  1. PHÂN TÍCH CẤU TRÚC: đếm bubble/vùng, vị trí — lưu {paper}_zones.json.
  2. CHUẨN HÓA THEO VÙNG: `zone_refine` — hiệu chỉnh dịch chuyển (dx,dy) RIÊNG
     từng vùng màu từ ring thật (giống band_affine_refine nhưng khớp PHẦN GIẤY
     thay vì dải ngang), nội suy mượt giữa các vùng rồi remap.
"""
import cv2
import numpy as np
import os

# màu (H OpenCV) -> nhãn vùng; riêng 'do' của tc12 có 2 rect (ID trên + P1 dưới)
_COLOR_PART = {
    "10_11": {"do": "MD", "hong": "P1", "duong": "P2", "la": "P3"},
    "12": {"do": "P1", "hong": "P2", "la": "P3"},   # rect đỏ y<205 -> SBD/MĐ
}
# nhãn phần -> màu ảo khi mẫu KHÔNG có sidecar khung màu (moi — sinh vùng từ
# chính toạ độ ô trong cells_file, "vùng màu" chỉ là NHÓM VẬT LÝ để căn riêng)
_PART_COLOR = {"P1": "hong", "P2": "duong", "P3": "la", "MD": "do",
               "SBD": "do"}
# mẫu gốc -> mẫu có viền màu (sidecar)
_TC_SIDECAR = {"10_11": "tc10_11.png", "12": "tc12.png"}

_ZONES_CACHE = {}


def _hue_mask(hsv, name):
    H, Sa, V = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    if name == "do":
        m = ((H <= 10) | (H >= 175))
    elif name == "hong":
        m = (H >= 150) & (H < 175)
    elif name == "vang":
        m = (H >= 20) & (H < 40)
    elif name == "la":
        m = (H >= 55) & (H < 75)
    elif name == "duong":
        m = (H >= 110) & (H < 130)
    else:
        return None
    return ((m & (Sa > 60) & (V > 60)).astype(np.uint8)) * 255


def detect_color_rects(tpl_color_path):
    """Dò các khung màu trên mẫu: [(tên_màu, (x1,y1,x2,y2), area)]."""
    im = cv2.imread(str(tpl_color_path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return []
    rgb = im[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    out = []
    for name in ("do", "hong", "vang", "la", "duong"):
        mask = _hue_mask(hsv, name)
        if mask is None or mask.sum() < 50:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if w < 15 or h < 15 or a < 40:
                continue
            out.append((name, (int(x), int(y), int(x + w), int(y + h)), int(a)))
    out.sort(key=lambda z: -z[2])
    return out


def _digit_grid_count(img, rect):
    """Đếm bubble LƯỚI CHỮ SỐ trong rect (MD/SBD/P3) — logic như build_id_cells:
    cụm cột → mỗi cột phải có ≥4 hàng khớp lưới (loại nhiễu) → tổng."""
    import omr_sheetgrid as S
    x1, y1, x2, y2 = rect
    sub = img[max(0, y1 + 3):max(0, y2 - 3), max(0, x1 + 3):max(0, x2 - 3)]
    if sub.size == 0:
        return 0
    ms = S._small_circles(sub)
    if len(ms) < 6:
        return 0
    xs = sorted(m[0] for m in ms)
    col_x = [c[0] for c in S._col_groups([(x, 0, 0) for x in xs], gap=6)]
    if not col_x:
        return 0
    pitch_x = (float(np.median([b - a for a, b in zip(col_x, col_x[1:])]))
               if len(col_x) > 1 else 9.0)
    groups = [[col_x[0]]]
    for x in col_x[1:]:
        if x - groups[-1][-1] <= 1.8 * pitch_x:
            groups[-1].append(x)
        else:
            groups.append([x])
    total = 0
    for gcols in groups:
        ys_all = sorted(m[1] for m in ms
                        if any(abs(m[0] - gx) <= 3 for gx in gcols))
        if len(ys_all) < 4:
            continue
        # lưới chữ số: MD/SBD 10 hàng (0-9), P3 12 hàng (- , 0-9) — chọn fit
        # khớp NHIỀU HÀNG CỘT NHẤT (hàng = số DIFF khớp, không đếm trùng noise)
        best = 0
        for exp in (12, 10):
            fit = S._fit_lattice(ys_all, 5.0, 20.0, expect=exp)
            if not fit:
                continue
            y0, pitch, _e = fit
            ks = set()
            for y in ys_all:
                t = (y0 + (exp - 1) * pitch - y) / pitch
                k = int(round(t))
                if abs(t - k) <= 0.30 and 0 <= k < exp:
                    ks.add(k)
            best = max(best, len(ks) * len(gcols))
        total += best
    return total


def zones_for(paper, tpl_path, cells=None):
    """Cấu trúc vùng màu của mẫu: trả dict hoặc None nếu mẫu không có sidecar màu.

    Trả {"paper", "tpl": (W,H), "sidecar", "zones": [
            {"color", "part", "rects", "n_bubble", "cells": [...]}...],
          "cell_zone": {(part,câu,opt): color}...}  (cell_zone key dạng tuple)."""
    ck = (paper, str(tpl_path))
    if ck in _ZONES_CACHE:
        return _ZONES_CACHE[ck]
    sidecar = _TC_SIDECAR.get(paper)
    out = None
    if sidecar:
        sc_path = os.path.join(os.path.dirname(str(tpl_path)), sidecar)
        if not os.path.exists(sc_path):
            sc_path = os.path.join("data", "maugiay", sidecar)
        if os.path.exists(sc_path):
            out = _build(paper, tpl_path, sc_path, cells)
    if out is None:
        # MẪU KHÔNG CÓ KHUNG MÀU (moi): sinh vùng từ TỌA ĐỘ Ô thật trong
        # cells_file — mỗi PHẦN = 1 vùng (bbox ô + lề 6px). "Vùng màu" chỉ là
        # nhãn nhóm vật lý (P1/P2/MD) — đúng tinh thần "cùng phần căn chung".
        out = _build_from_cells(paper, tpl_path, cells)
    if len(_ZONES_CACHE) > 8:
        _ZONES_CACHE.pop(next(iter(_ZONES_CACHE)))
    _ZONES_CACHE[ck] = out
    return out


def _build_from_cells(paper, tpl_path, cells):
    """Vùng màu suy từ toạ độ ô (mẫu không vẽ khung màu — moi)."""
    import omr_sheetgrid as S
    tpl = S._imread_cached(tpl_path)
    if tpl is None:
        return None
    Ht, Wt = tpl.shape[:2]
    try:
        cs = list(cells) if cells else S.build_template_cells(paper, tpl_path)
    except Exception:                                        # noqa
        return None
    if len(cs) < 8:
        return None
    from collections import Counter
    if Counter(str(c[0]) for c in cs).get("P1", 0) < 8:
        return None
    zones = []
    cell_zone = {}
    for part in sorted({str(c[0]) for c in cs}):
        pc = [c for c in cs if str(c[0]) == part]
        xs = [c[3] for c in pc]
        ys = [c[4] for c in pc]
        m = 6.0
        rect = (max(0, int(min(xs) - m)), max(0, int(min(ys) - m)),
                min(Wt, int(max(xs) + m)), min(Ht, int(max(ys) + m)))
        color = _PART_COLOR.get(part, part)
        zones.append({"color": color, "part": part, "rect": list(rect),
                      "n_bubble": len(pc), "n_cells": len(pc)})
        for c in pc:
            cell_zone[(c[0], str(c[1]), str(c[2]))] = color
    merged = {}
    for z in zones:
        mkey = merged.setdefault(z["color"], {"color": z["color"],
                                              "parts": set(), "rects": [],
                                              "n_bubble": 0, "n_cells": 0,
                                              "rect_detail": []})
        mkey["parts"].add(z["part"])
        mkey["rects"].append(z["rect"])
        mkey["n_bubble"] += z["n_bubble"]
        mkey["n_cells"] += z["n_cells"]
        mkey["rect_detail"].append({"rect": z["rect"], "part": z["part"],
                                    "n_bubble": z["n_bubble"],
                                    "n_cells": z["n_cells"]})
    out = {"paper": paper, "tpl": [Wt, Ht], "sidecar": None,
           "zones": [{"color": c, "parts": sorted(m["parts"]),
                      "rects": m["rects"], "n_bubble": m["n_bubble"],
                      "n_cells": m["n_cells"],
                      "rect_detail": m["rect_detail"]}
                     for c, m in merged.items()],
           "cell_zone": cell_zone}
    try:
        import json
        j = dict(out)
        j["cell_zone"] = {"|".join(k): v for k, v in cell_zone.items()}
        with open(os.path.join(os.path.dirname(str(tpl_path)),
                               f"{paper}_zones.json"), "w",
                  encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=1)
    except Exception:                                        # noqa
        pass
    return out


def _build(paper, tpl_path, sc_path, cells):
    import omr_sheetgrid as S
    tpl = S._imread_cached(tpl_path)
    if tpl is None:
        return None
    Ht, Wt = tpl.shape[:2]
    sc = cv2.imread(str(sc_path), cv2.IMREAD_UNCHANGED)
    if sc is None or sc.shape[:2] != (Ht, Wt):
        return None
    rects = detect_color_rects(sc_path)
    if not rects:
        return None
    if cells is None:
        try:
            cells = S.build_template_cells(paper, tpl_path)
        except Exception:                                    # noqa
            cells = []
    img_sc = cv2.imread(str(sc_path))
    # ---- GÁN Ô = RECT NHỎ NHẤT CHỨA (2026-11-08): rect hồng tc12 bị DÍNH
    # (span P1+P2 do viền chạm nhau) — nếu gán theo rect to trước thì ô P1 bị
    # ghi đè lung tung. Rect NHỎ (cụ thể) luôn CHIẾM ô trước; rect to chỉ
    # nhận ô còn lại. ----
    order = sorted(range(len(rects)), key=lambda i: rects[i][2])
    cell_owner = {}               # idx cell -> (color, rect)
    zone_cells = {i: [] for i in range(len(rects))}
    for i in order:
        color, rect, _a = rects[i]
        x1, y1, x2, y2 = rect
        for ci, c in enumerate(cells):
            if ci in cell_owner:
                continue
            if x1 - 3 <= c[3] <= x2 + 3 and y1 - 3 <= c[4] <= y2 + 3:
                cell_owner[ci] = i
                zone_cells[i].append(c)
    zones = []
    cell_zone = {}
    for i, (color, rect, _a) in enumerate(rects):
        x1, y1, x2, y2 = rect
        inside = zone_cells[i]
        part = _COLOR_PART.get(paper, {}).get(color, color)
        # rect ĐỎ của tc12: rect TRÊN (y<0.35H, không ô đáp án) = SBD/MĐ (ID)
        if paper == "12" and color == "do" and not inside and y2 < 0.35 * Ht:
            part = "ID"
        if inside:
            n_bubble = len(inside)
            for c in inside:
                cell_zone[(c[0], str(c[1]), str(c[2]))] = color
        else:
            # vùng không có ô đáp án: MD/SBD đếm CHÍNH XÁC qua build_id_cells
            # (lưới chiến đấu đã kiểm chứng); P3 đếm bằng lưới chữ số ước lượng
            n_bubble = None
            try:
                idc = S.build_id_cells(paper, tpl_path)
                n_id = sum(1 for c in idc
                           if x1 - 3 <= c[3] <= x2 + 3 and y1 - 3 <= c[4] <= y2 + 3)
                if n_id:
                    n_bubble = n_id
                for c in idc:
                    if x1 - 3 <= c[3] <= x2 + 3 and y1 - 3 <= c[4] <= y2 + 3:
                        cell_zone[(c[0], str(c[1]), str(c[2]))] = color
            except Exception:                                # noqa
                pass
            if n_bubble is None:
                n_bubble = _digit_grid_count(img_sc, rect)
        zones.append({"color": color, "part": part, "rect": list(rect),
                      "n_bubble": int(n_bubble), "n_cells": len(inside)})
    # gộp các rect CÙNG MÃU thành 1 vùng vật lý (cùng màu = cùng phần)
    merged = {}
    for z in zones:
        m = merged.setdefault(z["color"], {"color": z["color"],
                                           "parts": set(), "rects": [],
                                           "n_bubble": 0, "n_cells": 0})
        m["parts"].add(z["part"])
        m["rects"].append(z["rect"])
        m["n_bubble"] += z["n_bubble"]
        m["n_cells"] += z["n_cells"]
    out = {"paper": paper, "tpl": [Wt, Ht], "sidecar": str(sc_path),
           "zones": [{"color": c, "parts": sorted(m["parts"]),
                      "rects": m["rects"], "n_bubble": m["n_bubble"],
                      "n_cells": m["n_cells"],
                      # chi tiết TỪNG KHUNG — mỗi khung màu là 1 vùng riêng biệt
                      "rect_detail": [{"rect": list(z["rect"]),
                                       "part": z["part"],
                                       "n_bubble": z["n_bubble"],
                                       "n_cells": z["n_cells"]}
                                      for z in zones if z["color"] == c]}
                     for c, m in merged.items()],
           "cell_zone": cell_zone}
    # lưu JSON minh bạch (ghi đè mỗi lần cấu trúc đổi — mẫu tĩnh nên hiếm)
    try:
        import json
        j = dict(out)
        j["cell_zone"] = {"|".join(k): v for k, v in cell_zone.items()}
        with open(os.path.join(os.path.dirname(str(tpl_path)),
                               f"{paper}_zones.json"), "w",
                  encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=1)
    except Exception:                                        # noqa
        pass
    return out


def zone_of(cells_key, zones):
    """Màu vùng của 1 ô (key (part, câu, opt)); None nếu ngoài mọi vùng."""
    if not zones:
        return None
    return zones.get("cell_zone", {}).get((cells_key[0], str(cells_key[1]),
                                           str(cells_key[2])))


def zone_refine(warped, cells, scale, zones, tol_frac=1.6, min_pts=6,
                min_gain=0.008):
    """HIỆU CHỈNH THEO VÙNG MÀU (mở rộng band_affine_refine theo PHẦN GIẤY).

    Giấy cong làm các PHẦN dịch khác nhau (trái/phải, trên/dưới) — dải ngang
    không mô tả được khi P1/P2 nằm CẠNH nhau cùng hàng. Vùng màu (cùng màu =
    cùng phần, user chỉ định) là đơn vị vật lý đúng:

      1. Dò ring trên ảnh warp, ghép tâm ô ↔ ring gần (ngưỡng rộng 1.6 pitch).
      2. Mỗi VÙNG MÀU lấy dịch chuyển TRUNG VỊ (dx, dy) của các cặp trong vùng.
      3. Trường dịch mọi điểm ảnh = trung bình có TRỌNG SỐ 1/d² tới tâm các
         vùng (+ vùng nền = median toàn cục) — mượt, không có đường hở.
      4. remap; NHẬN chỉ khi tỉ lệ khớp (ngưỡng khít 0.55 pitch) tăng ≥ min_gain.

    Trả (ảnh_mới|None, hit_mới, hit_cũ)."""
    import omr_sheetgrid as S
    if not zones or not zones.get("zones") or warped is None or not cells:
        return None, 0.0, 0.0
    zmap = zones["cell_zone"]
    pts_all = np.float32([[c[3] * scale, c[4] * scale] for c in cells])
    keys = [(c[0], str(c[1]), str(c[2])) for c in cells]
    ys = sorted({round(c[4], 2) for c in cells})
    dy_row = float(np.median(np.diff(ys))) * scale if len(ys) > 2 else 22.0 * scale
    rings = S._ring_centers(S._ring_ready(warped), accept_filled=True)
    if len(rings) < 20:
        return None, 0.0, 0.0
    dd = np.linalg.norm(pts_all[:, None, :] - rings[None, :, :], axis=2)
    j = dd.argmin(axis=1)
    near = dd[np.arange(len(pts_all)), j]
    tol = max(45.0, tol_frac * dy_row)
    tol_meas = max(12.0, 0.55 * dy_row)
    hit_old = float((near <= tol_meas).sum()) / float(len(pts_all))
    keep = near <= tol
    if int(keep.sum()) < min_pts * len(zones["zones"]):
        return None, hit_old, hit_old
    # median dịch từng VÙNG — với vùng nhiều rect (vd xanh lá 2 khung) lấy
    # riêng TỪNG rect để phản ứng với cong cục bộ giữa các khung
    zoffs = []                       # (cx, cy, dx, dy, trọng_số)
    Hh, Ww = warped.shape[:2]
    for z in zones["zones"]:
        idx_color = [i for i, k in enumerate(keys)
                     if zmap.get(k) == z["color"] and keep[i]]
        if not idx_color:
            continue
        for r in z["rects"]:
            ridx = [i for i in idx_color
                    if r[0] * scale - 3 <= pts_all[i][0] <= r[2] * scale + 3
                    and r[1] * scale - 3 <= pts_all[i][1] <= r[3] * scale + 3]
            # ô ID không nằm trong cell_zone (MD/SBD/P3) — ghép theo VỊ TRÍ
            if not ridx and z["n_cells"] == 0:
                ridx = [i for i in range(len(keys))
                        if keep[i] and zmap.get(keys[i]) is None
                        and r[0] * scale - 3 <= pts_all[i][0] <= r[2] * scale + 3
                        and r[1] * scale - 3 <= pts_all[i][1] <= r[3] * scale + 3]
            if len(ridx) < min_pts:
                continue
            d = rings[j][ridx] - pts_all[ridx]
            cx = float((r[0] + r[2]) * 0.5 * scale)
            cy = float((r[1] + r[3]) * 0.5 * scale)
            zoffs.append((cx, cy, float(np.median(d[:, 0])),
                          float(np.median(d[:, 1])), float(len(ridx))))
    if len(zoffs) < 2:
        return None, hit_old, hit_old
    # trường dịch mượt: trọng số 1/d² tới tâm vùng; nền = median toàn cục
    d_all = rings[j][keep] - pts_all[keep]
    bg_dx, bg_dy = float(np.median(d_all[:, 0])), float(np.median(d_all[:, 1]))
    gy, gx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    dx_map = np.full((Hh, Ww), bg_dx, np.float32)
    dy_map = np.full((Hh, Ww), bg_dy, np.float32)
    wsum = np.full((Hh, Ww), 1e-6, np.float32)
    accx = np.zeros((Hh, Ww), np.float32)
    accy = np.zeros((Hh, Ww), np.float32)
    for (cx, cy, dx, dy, w) in zoffs:
        dist2 = (gx - cx) ** 2 + (gy - cy) ** 2 + 1e4
        wi = (w / dist2).astype(np.float32)
        accx += wi * dx
        accy += wi * dy
        wsum += wi
    dx_map = accx / wsum
    dy_map = accy / wsum
    # DẤU ĐÚNG (kiểm chứng tổng hợp + thật 2026-11-08): d = ring − ô mẫu;
    # out(x) = warped(x + d) đưa nội dung ring về ĐÚNG vị trí ô — map x−d dịch
    # NGƯỢC chiều (ring xa thêm, hit không đổi → bị từ chối; đo 090817:
    # x−d 0.750→0.750, x+d 0.750→0.795).
    map_x = (np.arange(Ww, dtype=np.float32)[None, :] + dx_map)
    map_y = (gy + dy_map)
    out = cv2.remap(warped, map_x, map_y, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rings2 = S._ring_centers(S._ring_ready(out), accept_filled=True)
    if len(rings2) < 20:
        return None, hit_old, hit_old
    dd2 = np.linalg.norm(pts_all[:, None, :] - rings2[None, :, :], axis=2)
    near2 = dd2.min(axis=1)
    hit_new = float((near2 <= tol_meas).sum()) / float(len(pts_all))
    if hit_new >= hit_old + min_gain:
        return out, hit_new, hit_old
    return None, hit_old, hit_old


_P3_CELLS_CACHE = {}


def _p3_circles(sub):
    """Vòng tròn nhỏ trong LƯỚI P3 — nới circularity 0.55 (bản _small_circles
    0.70 mất 3/4 bubble 6-7px của mẫu moi)."""
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (8 <= a <= 150 and 3 <= w <= 14 and 3 <= h <= 14
                and 0.60 <= w / h <= 1.60):
            continue
        p = cv2.arcLength(c, True)
        if 4 * np.pi * a / (p * p + 1e-9) < 0.55:
            continue
        out.append((x + w / 2.0, y + h / 2.0, float(w)))
    res = []
    for m in out:
        if all(abs(m[0] - o[0]) > 1.5 or abs(m[1] - o[1]) > 1.5 for o in res):
            res.append(m)
    return res


def build_p3_cells(paper, tpl_path, zones=None):
    """Ô PHẦN III (LƯỚI CHỮ SỐ 12 hàng '- , 9..0') — DÒNG TÂM CỤM TRỰC TIẾP:
    cụm hàng (tol 2.5px, giữ 12 hàng nhiều circle nhất) + cụm cột (tol 1.5px,
    ≥3 circle) TRONG TỪNG KHỐI (tách khối theo khoảng hở hàng > 1.6×pitch —
    moi có 2 khối Câu 1-5 / Câu 6-8 xếp chồng). Gán circle vào cặp (hàng, cột)
    GẦN NHẤT → ô. Nhãn hàng từ trên: ['-','=', '9'..'0']; câu = cột//4+1
    (mỗi câu 4 cột), đánh số LIÊN TỤC qua khối/khung.

    Trả [('P3', câu, opt, x, y)] với opt = f'c{cột_trong_câu}_{nhãn}'."""
    import omr_sheetgrid as S
    ck = (paper, str(tpl_path))
    if ck in _P3_CELLS_CACHE:
        return _P3_CELLS_CACHE[ck]
    tpl = S._imread_cached(tpl_path)
    if tpl is None:
        return []
    Ht, Wt = tpl.shape[:2]
    if zones is None:
        zones = zones_for(paper, tpl_path, cells=None)
    rects = []
    if zones:
        for z in zones["zones"]:
            if "P3" in z["parts"]:
                rects.extend(z["rects"])
    if not rects:
        # moi (không khung màu): P3 = vùng phải-dưới P2, dưới MD
        try:
            cs = S.build_template_cells(paper, tpl_path)
            p1x = max(c[3] for c in cs if c[0] == "P1")
            p2y = max(c[4] for c in cs if c[0] == "P2")
            rects = [(int(p1x) + 8, int(p2y) + 8, Wt - 2, Ht - 2)]
        except Exception:                                    # noqa
            return []

    def _cluster(vals, tol):
        vals = sorted(vals)
        out = []
        for v in vals:
            if out and v - out[-1][-1] <= tol:
                out[-1].append(v)
            else:
                out.append([v])
        return [float(np.mean(g)) for g in out]

    labels = ["-", ",", "9", "8", "7", "6", "5", "4", "3", "2", "1", "0"]
    cells = []
    cau_off = 0
    for r in rects:
        x1, y1, x2, y2 = r
        sub = tpl[max(0, y1 + 2):min(Ht, y2 - 2),
                  max(0, x1 + 2):min(Wt, x2 - 2)]
        if sub.size == 0:
            continue
        ms = _p3_circles(sub)
        # bubble quá nhỏ/nhạt (moi 6-7px mất 2/3) — CHẠY CẢ bộ dò PHÓNG 3×
        # (_upscaled_bubbles_img, đã chứng minh ở build_moi_template) và giữ
        # bộ phát hiện nhiều hơn
        try:
            ms2 = S._upscaled_bubbles_img(sub)
            if len(ms2) > len(ms):
                ms = ms2
        except Exception:                                    # noqa
            pass
        if len(ms) < 24:
            continue
        rows = _cluster([m[1] for m in ms], 2.5)
        if len(rows) < 10:
            continue
        # tách KHỐI theo khoảng hở hàng > 1.6×pitch
        pitches = [b - a for a, b in zip(rows, rows[1:])]
        pitch = float(np.median(pitches))
        blocks = [[rows[0]]]
        for a, b in zip(rows, rows[1:]):
            if b - a > 1.6 * pitch:
                blocks.append([])
            blocks[-1].append(b)
        for blk in blocks:
            # 12 hàng nhiều circle nhất của khối
            ylo, yhi = blk[0] - pitch * 0.6, blk[-1] + pitch * 0.6
            sub_ms = [m for m in ms if ylo <= m[1] <= yhi]
            if len(sub_ms) < 24:
                continue
            rws = _cluster([m[1] for m in sub_ms], 2.5)
            rws = sorted(rws, key=lambda yy: sum(
                1 for m in sub_ms if abs(m[1] - yy) <= pitch * 0.45))[-12:]
            rws = sorted(rws)[:12]
            if len(rws) < 10:
                continue
            cols = _cluster([m[0] for m in sub_ms], 1.5)
            cols = [c for c in cols
                    if sum(1 for m in sub_ms
                           if abs(m[0] - c) <= pitch * 0.45) >= 3]
            if len(cols) < 4:
                continue
            for ci, cx in enumerate(cols):
                cau = cau_off + ci // 4 + 1
                cinc = ci % 4 + 1
                for ri, ry in enumerate(rws):
                    hit = any(abs(m[1] - ry) <= pitch * 0.45
                              and abs(m[0] - cx) <= pitch * 0.45
                              for m in sub_ms)
                    if hit:
                        cells.append(("P3", cau, f"c{cinc}_{labels[ri]}",
                                      float(x1 + 2 + cx), float(y1 + 2 + ry)))
            cau_off += (len(cols) + 3) // 4
    _P3_CELLS_CACHE[ck] = cells
    return cells


def zone_ring_rates(warped, cells, scale, zones, tol_frac=0.45):
    """Ring rate RIÊNG TỪNG VÙNG MÀU (dùng cell_zone của ô). Trả {màu: rate} —
    'la' gộp mọi rect cùng màu. Ô không thuộc vùng nào bỏ qua."""
    import omr_sheetgrid as S
    if not zones or not cells or warped is None:
        return {}
    zmap = zones.get("cell_zone", {})
    byc = {}
    for c in cells:
        col = zmap.get((c[0], str(c[1]), str(c[2])))
        if col:
            byc.setdefault(col, []).append(c)
    if not byc:
        return {}
    rings = S._ring_centers(S._ring_ready(warped), accept_filled=True)
    if len(rings) < 10:
        return {}
    ys = sorted({round(c[4], 2) for c in cells})
    dy = float(np.median(np.diff(ys))) * scale if len(ys) > 2 else 22.0 * scale
    tol = tol_frac * dy
    out = {}
    for col, cs in byc.items():
        pts = np.float32([[c[3] * scale, c[4] * scale] for c in cs])
        d = np.linalg.norm(pts[:, None, :] - rings[None, :, :], axis=2)
        out[col] = float((d.min(axis=1) <= tol).mean())
    return out
