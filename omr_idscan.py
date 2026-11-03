# -*- coding: utf-8 -*-
"""
OMR ID SCAN — đọc "7. Số báo danh" & "8. Mã đề thi"
===================================================
Quy trình (chạy hoàn toàn trên ảnh đã căn mẫu bằng fiducial):

  1. Warp ảnh về hệ toạ độ MẪU (khử phối cảnh) với hệ số phóng `scale`
     -> lưới SBD/Mã đề của mẫu nằm đúng chỗ, sạch nhiễu hình học.
  2. Dựng "bản đồ đáp ứng vòng tròn" (matched filter vành khuyên) để bắt
     tâm các vòng in trên giấy.
  3. Tinh chỉnh lưới bằng mô hình affine 4 tham số (dx, dy, sx, sy):
     tối đa hoá tổng đáp ứng trên TOÀN BỘ ô (8×10 SBD / 4×10 MĐ) — nhờ
     gộp 80-120 ô nên bền với nhiễu cục bộ, và tự bù bản in khác tỉ lệ.
  4. Snap từng tâm ô về cực đại đáp ứng gần nhất (<= 0.30 pitch).
  5. Đọc mực trong lòng vòng: hàng 0..9 của lưới chính là chữ số 0..9,
     nên giải mã = chọn hàng có mật độ mực cao nhất + kiểm tra độ tin cậy
     (điểm tuyệt đối và biên so với hàng nhì).

Không cần model học sâu: "AI" ở đây là matched-filter + tối ưu tham số
toàn cục + luật tin cậy, đúng loại bài toán hình học lặp lại này.
"""
from __future__ import annotations

import cv2
import numpy as np

import omr_sheetgrid as S

# ngưỡng tin cậy (hiệu chỉnh trên bộ ảnh Test/)
MIN_SCORE = 0.34      # mực tuyệt đối tối thiểu của ô được chọn
MIN_RATIO = 1.03      # điểm (đã trừ nền) ô nhất / ô nhì
MIN_DELTA = 0.03      # mức vượt nền in tối thiểu
MD_SWITCH_MARGIN = 0.12   # biên tối thiểu để đổi sang đề hợp lệ trong key
MD_SWITCH_SCORE = 0.30    # tổng điểm tối thiểu của đề được chọn


# ---------- 1. warp về hệ toạ độ mẫu ----------

def warp_to_template(photo, H, tpl_shape, scale=4.0):
    """Ảnh trong hệ toạ độ mẫu (khử phối cảnh), phóng `scale`."""
    h, w = tpl_shape[:2]
    Ss = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    M = Ss @ np.linalg.inv(np.asarray(H, np.float64))
    return cv2.warpPerspective(photo, M, (int(round(w * scale)), int(round(h * scale))),
                               flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def _ring_response(gray_u8, pitch, ksize=None):
    """Bản đồ đáp ứng vành khuyên: cao tại tâm vòng tròn in trên giấy."""
    g = cv2.GaussianBlur(gray_u8, (3, 3), 0).astype(np.float32) / 255.0
    dark = 1.0 - g
    r = max(2.0, 0.34 * pitch)
    K = int(2 * r + 3)
    if K % 2 == 0:
        K += 1
    yy, xx = np.mgrid[0:K, 0:K]
    c = (K - 1) / 2.0
    d = np.hypot(xx - c, yy - c)
    ker = ((d >= 0.60 * r) & (d <= 1.18 * r)).astype(np.float32)
    s = ker.sum()
    if s > 0:
        ker /= s
    return cv2.filter2D(dark, -1, ker), r


# ---------- 2. lưới danh nghĩa theo mẫu ----------

def _group_lattice(cells, part):
    """Lưới danh nghĩa của 1 phần: cột = x, hàng = [(chữ_số, y)].

    Hàng 0..9 của lưới IN chính là chữ số 0..9, nên giải mã = chọn hàng
    có mực đậm nhất; ở đây giữ cặp (digit, y) để không lẫn y với digit.
    """
    sub = [c for c in cells if c[0] == part]
    if not sub:
        return None
    xs = sorted({round(c[3], 3) for c in sub})
    ys = sorted({round(c[4], 3) for c in sub})
    row_of = {}
    for c in sub:
        row_of.setdefault(int(c[2]), round(c[4], 3))
    rows = [(d, row_of[d]) for d in sorted(row_of)]
    return {"cols": xs, "rows": [y for _, y in rows], "row_digits": [d for d, _ in rows]}


# ---------- 3. tinh chỉnh affine toàn cục ----------

def _sample(resp, x, y):
    h, w = resp.shape
    xi, yi = int(round(x)), int(round(y))
    if xi < 1 or yi < 1 or xi >= w - 1 or yi >= h - 1:
        return 0.0
    return float(resp[yi, xi])


def _lattice_score(resp, lat, sx, sy, dx, dy, scale):
    """Tổng đáp ứng trên toàn bộ ô của lưới với 4 tham số affine."""
    cx = np.array(lat["cols"], np.float64) * sx
    cy = np.array(lat["rows"], np.float64) * sy
    xs = cx * scale + dx
    ys = cy * scale + dy
    return sum(_sample(resp, x, y) for x in xs for y in ys)


def _refine_affine(resp, lat, scale, pitch_px):
    """Hill-climb (dx, dy, sx, sy) quanh nghiệm đồng nhất."""
    best = dict(dx=0.0, dy=0.0, sx=1.0, sy=1.0)
    best_s = _lattice_score(resp, lat, 1.0, 1.0, 0.0, 0.0, scale)
    # Biên an toàn: lưới ID không được trôi quá nửa pitch / quá +-12% tỉ lệ,
    # nếu không sẽ "khớp nhầm" sang hàng bên cạnh trên bản photocopy.
    lim_d = 0.55 * pitch_px
    steps = [(pitch_px * 0.18, 0.020)] * 4 + [(pitch_px * 0.08, 0.008)] * 4 + \
            [(pitch_px * 0.03, 0.004)] * 3
    for step, sstep in steps:
        improved = True
        while improved:
            improved = False
            for key, delta in (("dx", step), ("dy", step),
                               ("sx", sstep), ("sy", sstep)):
                for sgn in (+1, -1):
                    cand = dict(best)
                    cand[key] = cand[key] + sgn * delta
                    if key in ("sx", "sy") and not (0.88 <= cand[key] <= 1.12):
                        continue
                    if key in ("dx", "dy") and abs(cand[key]) > lim_d:
                        continue
                    sc = _lattice_score(resp, lat, cand["sx"], cand["sy"],
                                        cand["dx"], cand["dy"], scale)
                    if sc > best_s + 1e-9:
                        best, best_s = cand, sc
                        improved = True
    return best, best_s


# ---------- 4. đọc mực từng ô ----------

def _bubble_radius(gray, box, pitch):
    """Ước lượng bán kính vòng tròn in trong vùng ID (px trên ảnh warp)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return 0.34 * pitch
    sub = gray[y0:y1, x0:x1]
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(sub, (3, 3), 0), 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 10)
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rs = []
    lo, hi = 0.16 * pitch, 0.46 * pitch
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if w < 8 or h < 8 or not (0.75 <= w / h <= 1.35):
            continue
        rr = max(w, h) / 2.0
        if not (lo <= rr <= hi):
            continue
        per = cv2.arcLength(c, True)
        if 4 * np.pi * a / (per * per + 1e-9) < 0.78:
            continue
        rs.append(rr)
    if len(rs) < 6:
        return 0.34 * pitch
    return float(np.median(rs))


def _fill_features(dark, x, y, r):
    """Đặc trưng "ô đã tô" của một bubble.

    Vòng in sẵn: lòng trắng, chỉ có nét chữ số ở giữa -> vành khuyên sáng.
    Học sinh tô: vành khuyên (vùng giữa nét số và vòng tròn) bị phủ mực.
    Vì vậy đo riêng 3 vùng: vành khuyên (ann), độ phủ đậm (frac), tâm (cen).
    """
    R = max(3, int(round(r * 0.95)))
    xi, yi = int(round(x)), int(round(y))
    if xi - R < 0 or yi - R < 0 or xi + R + 1 > dark.shape[1] or yi + R + 1 > dark.shape[0]:
        return 0.0, 0.0, 0.0
    roi = dark[yi - R:yi + R + 1, xi - R:xi + R + 1]
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    dn = np.hypot(xx - R, yy - R) / max(1e-6, 0.95 * R)
    ann = (dn >= 0.30) & (dn <= 0.80)
    cen = dn <= 0.28
    a = float(roi[ann].mean()) if ann.any() else 0.0
    f = float((roi[ann] > 0.55).mean()) if ann.any() else 0.0
    c = float(roi[cen].mean()) if cen.any() else 0.0
    return a, f, c


def _fill_score(a, f, c):
    """Điểm "đã tô" tổng hợp — vành khuyên là tín hiệu chính.

    Chuẩn hoá /0,30 để cùng thang với ô đáp án (read_cells): ô tô ~0,3-0,5,
    ô trống ~0,0-0,1.
    """
    return float(np.clip((0.45 * a + 0.35 * f + 0.20 * c) / 0.30, 0.0, 1.0))


def _detect_circles(gray, box, pitch):
    """Dò tâm các vòng tròn in trong vùng ID (ảnh đã warp)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    sub = cv2.GaussianBlur(gray[y0:y1, x0:x1], (3, 3), 0)
    bw = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 10)
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = 0.15 * pitch, 0.66 * pitch
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if w < 7 or h < 7 or not (0.66 <= w / h <= 1.52):
            continue
        per = cv2.arcLength(c, True)
        if 4 * np.pi * a / (per * per + 1e-9) < 0.52:
            continue
        rr = max(w, h) / 2.0
        if not (lo <= rr <= hi):
            continue
        out.append((x0 + x + w / 2.0, y0 + y + h / 2.0, rr))
    return out


def _cluster_1d(vals, gap):
    """Cụm 1D: các giá trị cách nhau <= gap thuộc cùng cụm. Trả tâm cụm."""
    vals = sorted(vals)
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.median(g)) for g in groups]


def _fit_uniform(centers):
    """Khớp lưới đều y = a + k*p (k = 0..n-1) bằng bình phương tối thiểu."""
    n = len(centers)
    if n < 3:
        return None
    k = np.arange(n, dtype=np.float64)
    y = np.asarray(centers, np.float64)
    p, a = np.polyfit(k, y, 1)      # y = p*k + a
    return float(a), float(p)


def _refine_with_circles(lat, circles, aff, scale, pitch):
    """Hiệu chuẩn lại HÀNG/CỘT theo vòng tròn in thật.

    Hàng: giữ các cụm nằm trong dải hợp lý quanh lưới danh nghĩa (loại mốc đen
    / chữ ở lề), NEO hàng 0 vào cụm đầu rồi khớp lại bước bằng hồi quy trên các
    cụm khớp — nhờ vậy bản photocopy in lệch tỉ lệ (bước thật 32,7 px so với
    mẫu 30 px) không bị trôi dẫn tới lệch 1-2 hàng ở cuối lưới.
    Cột: chọn cửa sổ ncols cụm x đồng đều nhất theo thứ tự trái→phải.
    """
    row_y, col_x = {}, {}
    ys = [c[1] for c in circles]
    xs = [c[0] for c in circles]
    nom = [ry * aff["sy"] * scale + aff["dy"] for ry in lat["rows"]]
    up, dn = nom[0], nom[-1]

    cl = _cluster_1d(ys, gap=0.55 * pitch)
    nrow = len(lat["rows"])

    # Thử MỌI cụm làm hàng 0 (miễn nằm trong +-1.6 pitch của hàng danh nghĩa),
    # chấm điểm bằng số cụm khớp lưới đều -> chọn nghiệm nhiều khớp nhất.
    # Cần thiết vì có bản in khối ID thấp/cao hơn mẫu hơn 1 pitch.
    best = None
    gaps_all = [b - a for a, b in zip(cl, cl[1:])]
    cand_g = [g for g in gaps_all if 0.60 * pitch <= g <= 1.60 * pitch]
    p0 = float(np.median(cand_g)) if cand_g else pitch
    if len(cl) >= 6 and 0.82 * pitch <= p0 <= 1.25 * pitch:
        for t, anchor in enumerate(cl):
            if abs(anchor - up) > 1.60 * pitch:
                continue
            ks, ys_ok, resid = [], [], []
            for c in cl:
                k = int(round((c - anchor) / p0))
                if 0 <= k < nrow and abs(c - (anchor + k * p0)) <= 0.45 * p0:
                    ks.append(k)
                    ys_ok.append(c)
                    resid.append(abs(c - (anchor + k * p0)))
            if len(ks) < 6 or len(set(ks)) != len(ks):
                continue
            score = (len(ks), -float(np.mean(resid)),
                     -abs(anchor - up) / max(1e-6, pitch))
            if best is None or score > best[0]:
                best = (score, ks, ys_ok)

    angled = False
    if best is not None:
        _, ks, ys_ok = best
        # np.polyfit(bậc 1) trả [slope, intercept] — đừng đảo thứ tự
        p_fit, a = np.polyfit(np.asarray(ks, np.float64),
                              np.asarray(ys_ok, np.float64), 1)
        for idx, ry in enumerate(lat["rows"]):
            row_y[ry] = float(a + idx * p_fit)
        angled = True
    if not angled:
        for idx, ry in enumerate(lat["rows"]):
            near = [v for v in ys if abs(v - nom[idx]) <= 0.45 * pitch]
            row_y[ry] = float(np.median(near)) if len(near) >= 3 else nom[idx]

    # ---- CỘT: chọn cửa sổ ncols cụm x đồng đều nhất (bám thứ tự trái->phải) ----
    cxs = _cluster_1d(xs, gap=0.55 * pitch)
    ncols = len(lat["cols"])
    nom_cols = [ci * aff["sx"] * scale + aff["dx"] for ci in lat["cols"]]
    picked = None
    if len(cxs) >= ncols:
        nom_px = ((nom_cols[-1] - nom_cols[0]) / max(1, ncols - 1)) if ncols > 1 else pitch
        best = None
        for st in range(len(cxs) - ncols + 1):
            win = cxs[st:st + ncols]
            gaps = np.diff(win)
            if len(gaps) and (gaps.max() > 1.9 * max(1e-6, gaps.min())):
                continue
            spread = float(np.std(gaps)) if len(gaps) else 0.0
            dev = abs(float(np.mean(gaps)) - nom_px) if len(gaps) else 0.0
            sc = spread + 0.35 * dev
            if best is None or sc < best[0]:
                best = (sc, win)
        if best is not None and best[0] <= 0.45 * pitch:
            picked = {ci: float(v) for ci, v in zip(lat["cols"], best[1])}
    if picked:
        col_x = picked
    else:
        for ci, nx in zip(lat["cols"], nom_cols):
            near = [v for v in cxs if abs(v - nx) <= 0.40 * pitch]
            col_x[ci] = float(np.median(near)) if near else float(nx)

    # ---- LÀM ĐỀU CỘT: chống 2 cột bị gán CÙNG một x (khi cụm bị sót) ----
    seq = [col_x[ci] for ci in lat["cols"]]
    if len(seq) > 1:
        gap_min = min(b - a for a, b in zip(seq, seq[1:]))
        if gap_min < 0.55 * pitch:
            good = [b - a for a, b in zip(seq, seq[1:]) if b - a >= 0.55 * pitch]
            px = float(np.median(good)) if good else pitch
            x0c = seq[0]
            for i_, ci in enumerate(lat["cols"]):
                col_x[ci] = x0c + i_ * px
    return col_x, row_y


def read_part(photo_w, cells, part, scale=4.0, debug=None):
    """Đọc 1 phần ('SBD'|'MD') trên ảnh đã warp về hệ toạ độ mẫu.

    Lưới bám theo VÒNG TRÒN IN THẬT: cụm y -> neo hàng 0 + hồi quy lại bước
    (chống trôi nửa/một hàng trên bản photocopy); cụm x -> cửa sổ cột đều nhất.
    Chữ số = hàng đậm nhất SAU KHI TRỪ NỀN MỰC IN theo hàng (hàng in số 8/6
    đậm hơn hẳn hàng 1/7 nên không trừ nền sẽ nhầm "in đậm" thành "đã tô").
    """
    lat = _group_lattice(cells, part)
    if not lat or len(lat["cols"]) < 1 or len(lat["rows"]) < 2:
        return None
    gray = cv2.cvtColor(photo_w, cv2.COLOR_BGR2GRAY)
    pitch_t = float(np.median(np.diff(lat["rows"])))
    pitch = pitch_t * scale
    resp, _ = _ring_response(gray, pitch)
    # ẢNH MỰC CHUẨN HOÁ NỀN: dark = 1 - gray/nền_cục_bộ (0 = giấy trắng).
    # Đo bằng (255-gray)/255 như trước bị bão hoà trên giấy sáng -> ô tô và ô
    # trống gần như bằng nhau (0,15 vs 0,11) và không đọc được SBD/Mã đề.
    gs = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    bg = cv2.blur(gs, (51, 51))
    dark = np.clip(1.0 - gs / (bg + 1e-6), 0.0, 1.0)
    aff, grid_score = _refine_affine(resp, lat, scale, pitch)

    # hộp dò rộng 1.8 pitch: có bản in đặt khối ID lệch tới ~1.4 pitch so với mẫu
    bx0 = min(lat["cols"]) * scale - 1.8 * pitch
    bx1 = max(lat["cols"]) * scale + 1.8 * pitch
    by0 = min(lat["rows"]) * scale - 1.8 * pitch
    by1 = max(lat["rows"]) * scale + 1.8 * pitch
    # Bán kính bubble suy TRỰC TIẾP từ bước lưới: đo trên ảnh thật cho
    # r = (0,29-0,34) x pitch ở mọi bản in -> dùng 0,30 x pitch là ổn định,
    # không phụ thuộc contour (từng cho 12,5-27 px trên cùng một bộ ảnh).
    b_r = 0.30 * pitch
    circles = _detect_circles(gray, (bx0, by0, bx1, by1), pitch)
    col_x, row_y = _refine_with_circles(lat, circles, aff, scale, pitch)

    # Căn lại CỘT bằng chính điểm "đã tô": quét ngang +-0,55 bước cột và chọn x
    # cho ô đậm nhất của cột là lớn nhất. Lưới lệch nửa cột (do cụm nhiễu) sẽ
    # chia đôi hai bubble -> max tụt, nên tự loại.
    snap_r = min(0.18 * pitch, 0.50 * b_r)

    ncols, nrows = len(lat["cols"]), len(lat["rows"])
    raw = np.zeros((ncols, nrows), np.float32)
    pos = {}
    hits = 0
    for i_, ci in enumerate(lat["cols"]):
        cx0 = col_x[ci]
        for j_, ry in enumerate(lat["rows"]):
            cy0 = row_y[ry]
            y0, y1 = int(max(0, cy0 - snap_r)), int(min(gray.shape[0], cy0 + snap_r + 1))
            x0, x1 = int(max(0, cx0 - snap_r)), int(min(gray.shape[1], cx0 + snap_r + 1))
            px, py = cx0, cy0
            if y1 - y0 >= 2 and x1 - x0 >= 2:
                win = resp[y0:y1, x0:x1]
                iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
                px, py = x0 + ix, y0 + iy
                if float(win[iy, ix]) >= 0.25:
                    hits += 1
            a, f, c = _fill_features(dark, px, py, b_r)
            raw[i_, j_] = _fill_score(a, f, c)
            pos[(i_, j_)] = (px, py)

    # ---- CHỌN ĐỘ DỜI HÀNG: lưới có thể lệch đúng 1 hàng (lỗi hay gặp) ----
    # Thử lệch -1/0/+1 bước và chọn lưới cho mỗi cột có một ô VƯỢT TRỘI
    # (max - trung vị lớn nhất). Lưới đúng: ô tô nổi hẳn; lưới lệch: các ô đều.
    def _mat_at(dy_off):
        m = np.zeros((ncols, nrows), np.float32)
        pp = {}
        for i_, ci in enumerate(lat["cols"]):
            cx0 = col_x[ci]
            for j_, ry in enumerate(lat["rows"]):
                cy0 = row_y[ry] + dy_off
                y0, y1 = int(max(0, cy0 - snap_r)), int(min(gray.shape[0], cy0 + snap_r + 1))
                x0, x1 = int(max(0, cx0 - snap_r)), int(min(gray.shape[1], cx0 + snap_r + 1))
                px_, py_ = cx0, cy0
                if y1 - y0 >= 2 and x1 - x0 >= 2:
                    win = resp[y0:y1, x0:x1]
                    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
                    px_, py_ = x0 + ix, y0 + iy
                a, f, c = _fill_features(dark, px_, py_, b_r)
                m[i_, j_] = _fill_score(a, f, c)
                pp[(i_, j_)] = (px_, py_)
        return m, pp

    pitch_y = float(np.median(np.diff([row_y[r] for r in lat["rows"]])))
    v0 = float((raw.max(axis=1) - np.median(raw, axis=1)).sum())
    best_v, best_dy = v0, 0.0
    for k in (-1, 1):
        m, pp = _mat_at(k * pitch_y)
        v = float((m.max(axis=1) - np.median(m, axis=1)).sum())
        if v > best_v * 1.15:
            best_v, best_dy, raw, pos = v, k * pitch_y, m, pp
    if best_dy:
        row_y = {r: y + best_dy for r, y in row_y.items()}

    # nền theo hàng (mực in sẵn) rồi chuẩn hoá
    base = np.percentile(raw, 25, axis=0) if ncols >= 3 else raw.min(axis=0)
    norm = raw - base

    out, detail, positions = [], [], []
    for i_ in range(ncols):
        order = np.argsort(-norm[i_])
        b1 = int(order[0])
        b2 = int(order[1]) if nrows > 1 else int(order[0])
        s1, s2 = float(norm[i_, b1]), float(norm[i_, b2])
        raw1 = float(raw[i_, b1])
        ratio = s1 / max(0.02, s2)
        ok = (s1 >= MIN_DELTA and raw1 >= MIN_SCORE and ratio >= MIN_RATIO)
        digit = int(lat["row_digits"][b1])
        if not ok:
            out.append("?")
            detail.append((lat["cols"][i_], -1, round(raw1, 2), round(s1, 2), round(ratio, 2)))
            continue
        out.append(str(digit))
        detail.append((lat["cols"][i_], digit, round(raw1, 2), round(s1, 2), round(ratio, 2)))
        px, py = pos.get((i_, b1), (0, 0))
        positions.append((lat["cols"][i_], str(digit), float(px), float(py), float(b_r)))

    hit_rate = hits / max(1, ncols * nrows)
    value = "".join(out) if out and "?" not in out else None
    if debug is not None:
        debug[part] = {"affine": aff, "grid_score": round(grid_score, 3),
                       "hit_rate": round(hit_rate, 3), "detail": detail,
                       "raw": list(out), "bubble_r": round(b_r, 1),
                       "n_circles": len(circles),
                       "score_matrix": np.round(norm, 2).tolist()}
    return {"value": value, "raw": list(out), "detail": detail,
            "positions": positions, "hit_rate": hit_rate, "ncols": ncols,
            "norm": norm, "row_digits": list(lat["row_digits"]),
            "cols": list(lat["cols"]), "col_x": dict(col_x), "row_y": dict(row_y),
            "row_of": {int(d): ry for d, ry in zip(lat["row_digits"], lat["rows"])},
            "b_r": float(b_r)}


def read_part_geo(warped, cells, part, scale=2.0):
    """Đọc phần ID bằng hình học DÒ ĐƯỢC (không dựa vào vị trí mẫu).

    Dùng sau khi khối SBD/Mã đề đã được căn cục bộ: vùng này sạch nên có thể
    dò đúng 10 hàng × n cột vòng tròn. Nếu số cụm khớp chính xác thì lưới lấy
    theo cụm (tránh lỗi lệch 1 hàng của lưới suy từ mẫu); nếu không, quay về
    read_part thường.
    """
    lat = _group_lattice(cells, part)
    if not lat:
        return None
    ncols, nrows = len(lat["cols"]), len(lat["rows"])
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    pitch = float(np.median(np.diff(lat["rows"]))) * scale
    bx0 = min(lat["cols"]) * scale - 1.4 * pitch
    bx1 = max(lat["cols"]) * scale + 1.4 * pitch
    by0 = min(lat["rows"]) * scale - 1.4 * pitch
    by1 = max(lat["rows"]) * scale + 1.4 * pitch
    cs = _detect_circles(gray, (bx0, by0, bx1, by1), pitch)
    if len(cs) < 10:
        return read_part(warped, cells, part, scale=scale)
    ys = _cluster_1d([c[1] for c in cs], gap=0.55 * pitch)
    xs = _cluster_1d([c[0] for c in cs], gap=0.55 * pitch)
    if len(ys) != nrows or len(xs) != ncols:
        return read_part(warped, cells, part, scale=scale)
    # lưới dò được: hàng trên cùng = chữ số 0 (đúng thứ tự in trên phiếu)
    b_r = 0.30 * pitch
    gs = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    bg = cv2.blur(gs, (51, 51))
    dark = np.clip(1.0 - gs / (bg + 1e-6), 0.0, 1.0)
    raw = np.zeros((ncols, nrows), np.float32)
    pos = {}
    for i_, cx in enumerate(xs):
        for j_, cy in enumerate(ys):
            a, f, c = _fill_features(dark, cx, cy, b_r)
            raw[i_, j_] = _fill_score(a, f, c)
            pos[(i_, j_)] = (cx, cy)
    base = np.percentile(raw, 25, axis=0) if ncols >= 3 else raw.min(axis=0)
    norm = raw - base
    out, detail, positions = [], [], []
    for i_ in range(ncols):
        order = np.argsort(-norm[i_])
        b1 = int(order[0])
        b2 = int(order[1]) if nrows > 1 else int(order[0])
        s1, s2 = float(norm[i_, b1]), float(norm[i_, b2])
        raw1 = float(raw[i_, b1])
        ratio = s1 / max(0.02, s2)
        ok = (s1 >= MIN_DELTA and raw1 >= MIN_SCORE and ratio >= MIN_RATIO)
        digit = int(lat["row_digits"][b1])
        if not ok:
            out.append("?")
            detail.append((lat["cols"][i_], -1, round(raw1, 2), round(s1, 2), round(ratio, 2)))
            continue
        out.append(str(digit))
        detail.append((lat["cols"][i_], digit, round(raw1, 2), round(s1, 2), round(ratio, 2)))
        px_, py_ = pos[(i_, b1)]
        positions.append((lat["cols"][i_], str(digit), float(px_), float(py_), float(b_r)))
    value = "".join(out) if out and "?" not in out else None
    return {"value": value, "raw": list(out), "detail": detail,
            "positions": positions, "hit_rate": 1.0, "ncols": ncols,
            "norm": norm, "row_digits": list(lat["row_digits"]),
            "cols": list(lat["cols"]), "col_x": {c: x for c, x in zip(lat["cols"], xs)},
            "row_y": {r: y for r, y in zip(lat["rows"], ys)},
            "row_of": {int(d): r for d, r in zip(lat["row_digits"], lat["rows"])},
            "b_r": float(b_r)}


def fallback_positions(warped, paper, tpl_path, part, scale=2.0,
                       rel=1.8, floor=0.05):
    """Vị trí ô ĐẬM NHẤT của từng cột (để VẼ VÒNG khi không giải được chuỗi số).

    Dùng cho lưới SBD: nhiều phiếu chỉ tô vài chữ số hoặc mực quá mờ nên không
    ghép được thành số hoàn chỉnh; hàm này vẫn trả về các ô có mực nổi bật so
    với chính cột đó để ảnh kết quả khoanh được chỗ học sinh đã tô.
    """
    import omr_sheetgrid as _S
    try:
        idc = _S.build_id_cells(paper, tpl_path)
        rp = read_part(warped, idc, part, scale=scale)
    except Exception:                                            # noqa
        return []
    if not rp:
        return []
    norm, digs = rp.get("norm"), rp.get("row_digits") or []
    if norm is None or not digs:
        return []
    out = []
    for i_, ci in enumerate(rp["cols"]):
        col = np.asarray(norm[i_], np.float32)
        j = int(np.argmax(col))
        med = float(np.median(col))
        if float(col[j]) < max(floor, rel * max(med, 1e-6)):
            continue
        ry = rp["row_of"].get(int(digs[j]))
        x = rp["col_x"].get(ci)
        y = rp["row_y"].get(ry) if ry is not None else None
        if x is None or y is None:
            continue
        out.append((ci, str(digs[j]), float(x), float(y), float(rp["b_r"])))
    return out


def positions_for(res_part, value):
    """Vị trí (warp) của từng chữ số trong `value` — để vẽ vòng khớp số đã báo."""
    if not res_part or not value:
        return []
    row_of, col_x, row_y = res_part["row_of"], res_part["col_x"], res_part["row_y"]
    out = []
    for col_i, ci in enumerate(res_part["cols"]):
        if col_i >= len(value):
            break
        d = value[col_i]
        if not d.isdigit() or int(d) not in row_of:
            continue
        ry = row_of[int(d)]
        out.append((ci, str(d), float(col_x.get(ci, 0.0)),
                    float(row_y.get(ry, 0.0)), float(res_part["b_r"])))
    return out


def _free_score(res):
    """Tổng điểm của các chữ số đang đọc được (bỏ qua ô '?')."""
    if not res:
        return 0.0
    norm, digs = res["norm"], res["row_digits"]
    idx = {d: i for i, d in enumerate(digs)}
    tot = 0.0
    for col, ch in enumerate(res["raw"]):
        if ch == "?" or not str(ch).isdigit():
            continue
        tot += float(norm[col, idx[int(ch)]])
    return tot


def resolve_candidates(res, candidates):
    """Chọn mã đề TRONG danh sách hợp lệ khớp lưới điểm tốt nhất.

    Mỗi đề chỉ có 3-4 chữ số, mà tập đề hợp lệ rất nhỏ (4-13) -> ràng buộc
    theo key mạnh hơn nhiều so với chọn từng chữ số độc lập (nhất là khi
    một ô bị mờ/photocopy). Trả (đề, điểm, biên so với đề nhì).
    """
    if not res or not candidates:
        return None, 0.0, 0.0
    norm = res["norm"]
    digs = res["row_digits"]
    idx = {d: i for i, d in enumerate(digs)}
    ncol = norm.shape[0]
    scored = []
    for cand in candidates:
        cs = str(cand)
        if len(cs) != ncol or any(not ch.isdigit() or int(ch) not in idx for ch in cs):
            continue
        sc = 0.0
        for col, ch in enumerate(cs):
            sc += float(norm[col, idx[int(ch)]])
        scored.append((sc, cs))
    if not scored:
        return None, 0.0, 0.0
    scored.sort(reverse=True)
    best, second = scored[0], (scored[1] if len(scored) > 1 else (0.0, ""))
    return best[1], best[0], best[0] - second[0]


# ---------- 5. API chính ----------

def scan_ids(photo, H, paper, tpl_path, scale=4.0, parts=("SBD", "MD"), debug=None,
             md_candidates=None):
    """Đọc SBD + Mã đề trên 1 ảnh (đã có H căn mẫu).

    Trả {"SBD": "50120074"|None, "MD": "3157"|None, "score": {...},
         "raw": {...}, "positions": {...}, "hit": {...}}
    """
    tpl = cv2.imread(str(tpl_path))
    if tpl is None:
        raise FileNotFoundError(tpl_path)
    id_cells = S.build_id_cells(paper, tpl_path)
    if not id_cells:
        return {"SBD": None, "MD": None, "score": {}, "raw": {},
                "positions": {}, "hit": {}}
    w = warp_to_template(photo, H, tpl.shape, scale=scale)
    res = {"SBD": None, "MD": None, "score": {}, "raw": {}, "positions": {},
           "hit": {}}
    parts_res = {}
    for part in parts:
        r = read_part(w, id_cells, part, scale=scale,
                      debug=(debug if debug is not None else None))
        if not r:
            continue
        parts_res[part] = r
        res[part] = r["value"]
        res["raw"][part] = r["raw"]
        res["score"][part] = r["detail"]
        res["positions"][part] = r["positions"]
        res["hit"][part] = {"hit_rate": round(r["hit_rate"], 3), "ncols": r["ncols"]}
        if part == "MD" and md_candidates:
            cand, csc, cmargin = resolve_candidates(r, md_candidates)
            res["md_constrained"] = None
            if cand:
                # điểm của cách đọc TỰ DO — dùng làm mốc so sánh
                free = _free_score(r)
                if cand != r["value"]:
                    # chỉ đổi khi: biên rõ, có mực thật, và đề trong key không
                    # tệ hơn hẳn cách đọc tự do (tránh "ép" sang đề sai)
                    if cmargin >= MD_SWITCH_MARGIN and csc >= MD_SWITCH_SCORE \
                            and csc >= 0.80 * free:
                        res["md_constrained"] = {"từ": r["value"], "thành": cand,
                                                 "điểm": round(csc, 2),
                                                 "biên": round(cmargin, 2)}
                        res[part] = cand
                else:
                    res[part] = cand
    # vị trí vẽ lại theo giá trị CHỐT (sau ràng buộc key) — trước đây chỉ có
    # vị trí của cách đọc tự do nên vòng vẽ lệch với số đã báo
    for part, rp in parts_res.items():
        res["positions"][part] = positions_for(rp, res.get(part))

    # toạ độ trong `positions` đang ở hệ WARP -> đổi về hệ ẢNH GỐC để vẽ overlay
    Hm = np.asarray(H, np.float64)
    for part in ("SBD", "MD"):
        pts = res["positions"].get(part) or []
        if not pts:
            continue
        xy = np.float32([[px / scale, py / scale] for (_, _, px, py, _) in pts])
        back = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        r_photo = max(2.0, float(np.median([r_ for (_, _, _, _, r_) in pts])) / scale
                      * float(np.mean(np.linalg.norm(Hm[:2, :2], axis=0))))
        res["positions"][part] = [(c, d, float(b[0]), float(b[1]), r_photo)
                                  for (c, d, _, _, _), b in zip(pts, back)]
    res["warp"] = w
    return res
