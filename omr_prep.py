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

MARKER_THR = 0.80        # ngưỡng mực (chuẩn hoá theo nền) cho ô đen in
MARKER_MAX_FRAC = 0.004  # diện tích tối đa của 1 mốc (tỉ lệ ảnh)
MIN_MARKERS = 4
MAX_MATCH_DIST = 0.13    # dung sai khớp theo toạ độ chuẩn hoá
MIN_VERIFY = 0.70        # tỉ lệ mốc phải khớp sau khi warp


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


def template_fiducials(tpl_path):
    """BỐN ô định vị ở 4 góc của mẫu (cùng với danh sách đầy đủ để kiểm chứng).

    Tìm HÌNH CHỮ NHẬT LỚN NHẤT mà 4 đỉnh đều có ô đen thật:
      12.png  -> (12,12) (450,12) (450,646) (12,646)   [khung mốc mép trang]
      10_11   -> (46,103) (360,103) (46,546) (360,546) [khung trong]
    Cách này không phụ thuộc kích thước mốc (mẫu 12.png có ô 9 px lẫn ô 13 px).
    """
    ms = template_markers(tpl_path)
    if len(ms) < MIN_MARKERS:
        return [], ms
    img = cv2.imread(str(tpl_path))
    h, w = img.shape[:2]
    xs = _cluster_vals([m[0] for m in ms], tol=2.0)
    ys = _cluster_vals([m[1] for m in ms], tol=2.0)
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
        return [], ms
    _, i1, i2, j1, j2 = best
    quad = [(xs[i1], ys[j1]), (xs[i2], ys[j1]), (xs[i2], ys[j2]), (xs[i1], ys[j2])]
    return quad, ms


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
    """Suy khoá mẫu ('12' | '10_11') từ tên file mẫu."""
    name = str(tpl_path).lower()
    return "10_11" if "10_11" in name or "10-11" in name else "12"


def best_orientation(img, tpl_path, paper, rotations=(0, 90, 180, 270), cells=None,
                     allow_sift=False):
    """Thử 4 hướng xoay, trả (H, rotated_img, quality, rot) tốt nhất.

    Ảnh điện thoại thường KHÔNG có EXIF orientation (máy cầm ngang) nên phải
    thử xoay theo nội dung. Đo trên bộ Test11: 8/10 ảnh chỉ khớp sau khi xoay
    270°, quality 0,5-0,7 -> 1,000.
    """
    import omr_sheetgrid as S
    ref = S.build_template_cells(paper, tpl_path)
    use_cells = cells if cells else ref
    tpl_img = cv2.imread(str(tpl_path))
    tpl_h, tpl_w = tpl_img.shape[:2]
    # Ảnh NGANG mà mẫu DỌC -> gần như chắc chắn đã bị xoay 90/270; thử trước
    # để khỏi tốn 2-3 lần căn mẫu vô ích (mỗi lần ~1,5 s ở ảnh 2600 px)
    if len(rotations) > 2 and img.shape[1] > img.shape[0] * 1.15:
        rotations = (90, 270, 0, 180) + tuple(r for r in rotations
                                             if r not in (0, 90, 180, 270))
    best = (None, img, 0.0, 0)
    best_ok = None
    for rot in rotations:
        if rot == 90:
            im = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            im = cv2.rotate(img, cv2.ROTATE_180)
        elif rot == 270:
            im = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            im = img
        try:
            H, ph, q = S.align_photo("", tpl_path, use_cells,
                                     photo=im, ref_cells=ref,
                                     allow_sift=allow_sift)
        except Exception:
            continue
        if q is None or q < 0.3:
            continue
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
        # normalize_to_template còn kiểm chứng 4/4 mốc góc ở bước cuối nên
        # không cần _corner_check quyết định việc dừng.
        if q >= 0.97:
            break
    # Thử lại có SIFT nếu quét nhanh không tìm được gì tốt (ảnh khó)
    if (best_ok is None and best[2] < 0.70) and not allow_sift:
        return best_orientation(img, tpl_path, paper, rotations=rotations,
                                cells=cells, allow_sift=True)
    return best_ok if best_ok is not None else best


def normalize_to_template(photo, tpl_path, scale=2.0, verify=True, paper=None,
                          cells=None, ink=False, ink_only=False):
    """Đưa 1 ảnh (đường dẫn hoặc BGR) về ĐÚNG CHUẨN MẪU.

    Quy trình: EXIF → thử 4 hướng xoay → căn mẫu (fiducial + ICP + SIFT dự phòng)
    → warp về kích thước mẫu × scale → chuẩn hoá sáng.
    Trả {"ok", "warped", "H", "quality", "rot", "reason"}.
    """
    import omr_sheetgrid as S
    img = load_bgr(photo) if (isinstance(photo, (str, bytes))
                              or hasattr(photo, "__fspath__")) else photo
    tpl = cv2.imread(str(tpl_path))
    if tpl is None:
        raise FileNotFoundError(tpl_path)
    paper = paper or paper_of_template(tpl_path)
    Ht, Wt = tpl.shape[:2]
    info = {"ok": False, "warped": None, "H": None, "quality": 0.0, "rot": 0,
            "reason": ""}
    try:
        rots = (0,) if ink_only else (0, 90, 180, 270)
        H, im2, q, rot = best_orientation(img, tpl_path, paper, rotations=rots,
                                          cells=cells)
    except Exception as e:                                   # noqa
        info["reason"] = f"lỗi căn mẫu: {e}"
        return info
    info.update(quality=round(q, 3), rot=rot)
    if H is None or q < MIN_VERIFY:
        info["reason"] = f"không khớp mẫu (quality {q:.2f} < {MIN_VERIFY})"
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
        Hc, hit = refine_grid_by_cells(warped, cells, scale)
        if Hc is not None and hit >= 0.5:
            M2 = Hc @ M          # áp hiệu chỉnh lưới lên ma trận warp ban đầu
            cand = cv2.warpPerspective(im2, M2, (int(Wt * scale), int(Ht * scale)),
                                       flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            info["grid_hit"] = round(float(hit), 3)
            if cand is not None and cand.size:
                warped = cand

    # ---- KIỂM CHỨNG BẮT BUỘC: chống homography suy biến (ảnh warp trắng trơn
    #      vẫn có thể cho quality 1.0 vì mọi ô cùng rơi vào một vùng sáng) ----
    if verify:
        quad, _ = template_fiducials(tpl_path)
        cf = _corner_check(warped, quad, scale) if len(quad) == 4 else 0.0
        # ---- KIỂM CHỨNG BẰNG LƯỚI Ô ĐÁP ÁN (ICP) ----
        # 4 mốc góc có thể bị che/mất (ảnh chụp gần, mốc mờ) -> cf chỉ 2-3/4
        # dù warp vẫn ĐÚNG. Lưới ô đáp án (hàng trăm vòng tròn) là bằng chứng
        # mạnh hơn nhiều: nếu >= 50 % tâm ô khớp vòng in thật thì nhận ảnh.
        if cf < 1.0 and cells:
            Hc, hit = refine_grid_by_cells(warped, cells, scale)
            info["grid_hit"] = round(float(hit), 3)
            if Hc is not None and hit >= 0.50:
                M2 = Hc @ M
                warped = cv2.warpPerspective(
                    im2, M2, (int(Wt * scale), int(Ht * scale)),
                    flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                M = M2
                info["M_final"] = M2
                info["refined"] = True
                cf = max(cf, float(hit))
        cf = _corner_check(warped, quad, scale) if len(quad) == 4 else 0.0
        ink = _ink_fraction(warped)
        info["corner"] = round(cf, 2)
        info["ink"] = round(ink, 3)
        # Nhận ảnh nếu 4/4 mốc đúng HOẶC lưới ô đáp án khớp >= 50 %
        if cf < 1.0 and float(info.get("grid_hit") or 0.0) < 0.50:
            info["reason"] = f"chỉ {int(cf * 4)}/4 ô định vị nằm đúng chỗ"
            return info
        if cf < 1.0:
            info["reason"] = ""            # warp đã được xác nhận bằng lưới ô
        if not (0.02 <= ink <= 0.50):
            info["reason"] = f"ảnh warp không giống phiếu (mực {ink:.1%})"
            return info
    warped = normalize_light(warped)
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
