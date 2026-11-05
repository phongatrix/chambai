# -*- coding: utf-8 -*-
"""
SHEET GRID OMR (omr_sheetgrid.py)
=================================
Quy trình đúng theo yêu cầu:
  1. Chọn MẪU GIẤY (ảnh phiếu trắng): 12.png (THPT) hoặc 10_11.png (QM)
  2. Nạp ĐÁP ÁN: k12dapan.xlsx ↔ 12.png ; k11dapan.xlsx ↔ 10_11.png
  3. Dựng LƯỚI ĐÁP ÁN: mỗi ô = (phần, câu, lựa chọn) + toạ độ trên mẫu + đáp án key
  4. ẢNH BÀI THI được căn khớp mẫu bằng 4 ô đen định vị (fiducial) + tinh chỉnh ICP
  5. Đối chiếu bubble tô đen với lưới -> ĐÚNG (xanh) / SAI (đỏ) / TRỐNG-NHIỀU (vàng)
  6. Tính điểm + xuất ảnh kết quả

Màu:  GREEN = đúng, RED = sai, YELLOW = trống hoặc tô nhiều đáp án.
"""
import cv2
import numpy as np
from pathlib import Path

GREEN = (0, 170, 0)
RED = (0, 0, 235)
YELLOW = (0, 200, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------- cấu hình mẫu ----------

PAPERS = {
    "12": {          # THPT: 40 câu ABCD (4 khối × 10), 8 câu Đ/S, 6 câu số
        "tên": "Mẫu THPT 12.png",
        "key_goi_y": "k12dapan.xlsx",
        "p1_bands": (230, 360),     # dải y của PHẦN I (ảnh mẫu)
        "p1_x": None,
        "p2_bands": (395, 448),     # dải y của PHẦN II
        "p2_x": None,
        "p1_blocks": 4,             # số khối câu theo hàng ngang
        "p1_rows_per_block": 10,
        "p2_cau": 8,
        "p2_per_row": 8,
        "p2_y": 4,                  # số ý a-d
    },
    "10_11": {       # QM: 30 câu ABCD cột trái (3 khối × 10), 10 câu Đ/S, 10 câu số
        "tên": "Mẫu QM 10_11.png",
        "key_goi_y": "k11dapan.xlsx",
        "p1_bands": (285, 540),
        "p1_x": (50, 106),          # cột PHẦN I bên trái
        "p2_bands": (215, 305),
        "p2_x": (110, 380),
        "p1_blocks": 1,
        "p1_rows_per_block": 30,
        "p2_cau": 10,
        "p2_per_row": 5,
        "p2_y": 4,
    },
}


# ---------- 1. PHÂN TÍCH MẪU ----------

def _upscaled_bubbles(path, scale=None):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = scale if scale else (3 if max(h, w) < 800 else 1)
    up = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s != 1 else img
    import omr_fiducial as F
    minr = max(3, int(round(3 * s)))
    bs = F.detect_bubbles(up, min_r=minr, max_r=40 * s)
    return [(x / s, y / s, bw / s) for x, y, bw in bs]


_SQ_CACHE = {}


def _square_candidates(path, thr=0.80):
    """Ứng viên ô đen vuông (mọi kích thước) — mask tính 1 lần, cache theo ảnh.

    Cache key với MẢNG (ndarray) phải là HASH NỘI DUNG: str(ndarray) bị numpy
    rút gọn (chỉ còn vài giá trị góc + kích thước) -> hai ảnh khác nhau cùng
    kích thước (vd các biến thể xoay có viền trắng giống nhau) va CHÙM key và
    dùng NHẦM ứng viên của ảnh trước — bug thật, ảnh hưởng best_orientation
    (mỗi hướng xoay sinh 1 mảng cùng cỡ)."""
    if isinstance(path, (str, bytes)) or hasattr(path, "__fspath__"):
        img = cv2.imread(str(path))
        key = ("f", str(path), thr)
    else:
        img = path
        import hashlib
        key = ("a", hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest(),
               thr)
    if key in _SQ_CACHE:
        return _SQ_CACHE[key]
    if img is None:
        raise FileNotFoundError(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.blur(g.astype(np.float32), (51, 51))
    norm = g.astype(np.float32) / (bg + 1e-6)
    dark = (norm < thr).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < 20:
            continue
        # minAreaRect: KHÍT với ô xoay BẤT KỲ góc nào (bounding box thẳng trục
        # làm fill = 1/(cosθ+sinθ)² — ô xoay >~9° rơi dưới 0.75 và biến mất,
        # đúng nguyên nhân phiếu nghiêng không còn mốc)
        (rcx, rcy), (rw, rh), _ang = cv2.minAreaRect(c)
        if rw < 2 or rh < 2:
            continue
        fill = a / (rw * rh + 1e-9)
        lo_s, hi_s = min(rw, rh), max(rw, rh)
        # 0.85: ô VUÔNG xoay bất kỳ fill ≈ 0.95+; hình TRÒN fill = π/4 ≈ 0.785
        # (bubble in đậm phải bị LOẠI — nếu không hàng trăm bubble thành ứng
        # viên và RANSAC nổ thời gian)
        if fill > 0.85 and 0.70 <= lo_s / hi_s <= 1.40:
            out.append((rcx, rcy, float(lo_s), float(hi_s)))
    _SQ_CACHE[key] = out
    return out


def _fiducial_squares(path, side=(6, 16), thr=0.80):
    """Lọc ứng viên theo dải kích thước cạnh."""
    return [m for m in _square_candidates(path, thr)
            if side[0] <= m[2] <= side[1] and side[0] <= m[3] <= side[1]]


def _row_bands(bubbles, tol=5.0):
    if not bubbles:
        return []
    ys = sorted(b[1] for b in bubbles)
    groups = [[ys[0]]]
    for y in ys[1:]:
        if y - groups[-1][-1] <= tol:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [(float(np.mean(g)), [b for b in bubbles if abs(b[1] - float(np.mean(g))) <= tol])
            for g in groups]


def _col_groups(bubbles, gap=12.0):
    xs = sorted(b[0] for b in bubbles)
    if not xs:
        return []
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    out = []
    for g in groups:
        xm = float(np.mean(g))
        out.append((xm, [b for b in bubbles if abs(b[0] - xm) <= gap]))
    return out


_CELLS_CACHE = {}


def build_template_cells(paper, template_png):
    """Lưới ô ngữ nghĩa của mẫu: [(phần, câu, lựa_chọn, x, y)]."""
    ck = (paper, str(template_png))
    if ck in _CELLS_CACHE:
        return _CELLS_CACHE[ck]
    cfg = PAPERS[paper]
    bs = _upscaled_bubbles(template_png)
    W = cv2.imread(str(template_png)).shape[1]
    cells = []
    # ---- PHẦN I ----
    lo, hi = cfg["p1_bands"]
    p1 = [b for b in bs if lo <= b[1] <= hi]
    px = cfg.get("p1_x")
    if px:
        p1 = [b for b in p1 if px[0] <= b[0] <= px[1]]
    rows = sorted([r for r in _row_bands(p1, tol=2.5)
                   if len(_col_groups(r[1], gap=8)) == cfg["p1_blocks"] * 4],
                  key=lambda r: r[0])
    for ri, (ym, mem) in enumerate(rows):
        cg = _col_groups(mem, gap=8)
        # giữ đúng 4×blocks cụm
        if len(cg) != cfg["p1_blocks"] * 4:
            continue
        for ci, (xm, mm) in enumerate(cg):
            block, opt = ci // 4, ci % 4
            cq = block * cfg["p1_rows_per_block"] + ri + 1
            for b in mm:
                cells.append(("P1", cq, "ABCD"[opt], b[0], b[1]))
    # ---- PHẦN II ----
    lo2, hi2 = cfg["p2_bands"]
    p2 = [b for b in bs if lo2 <= b[1] <= hi2]
    px2 = cfg.get("p2_x")
    if px2:
        p2 = [b for b in p2 if px2[0] <= b[0] <= px2[1]]
    per_row = cfg["p2_per_row"]
    rows2 = sorted([r for r in _row_bands(p2, tol=2.5)
                    if len(_col_groups(r[1], gap=8)) == per_row * 2],
                   key=lambda r: r[0])
    rows2 = rows2[: cfg["p2_y"] * (cfg["p2_cau"] // per_row)]
    for ri, (ym, mem) in enumerate(rows2):
        cg = _col_groups(mem, gap=8)
        if len(cg) != per_row * 2:
            continue
        block = ri // cfg["p2_y"]
        yk = chr(97 + ri % cfg["p2_y"])
        for ci, (xm, mm) in enumerate(cg):
            cq = block * per_row + ci // 2 + 1
            side = "D" if ci % 2 == 0 else "S"      # Đ đứng trước S
            for b in mm:
                cells.append(("P2", (cq, yk), side, b[0], b[1]))
    _CELLS_CACHE[ck] = cells
    return cells


# ---------- 2. LƯỚI ĐÁP ÁN + KEY ----------

def key_p2_offset(key):
    p1 = key.get("p1", {})
    return max([int(c) for c in p1], default=0)


def build_answer_grid(paper, template_png, key):
    """Gắn đáp án key vào từng ô mẫu. Trả {"cells": [...], "answers": {...}}."""
    cells = build_template_cells(paper, template_png)
    off = key_p2_offset(key)
    answers = {}
    for part, q, opt, x, y in cells:
        if part == "P1":
            exp = key.get("p1", {}).get(q) or key.get("p1", {}).get(str(q))
            answers[(part, q, opt)] = exp
        else:
            cq, yk = q
            exp = key.get("p2", {}).get((cq + off, yk))
            if exp is None:
                exp = key.get("p2", {}).get((str(cq + off), yk))
            answers[(part, q, opt)] = exp
    return {"cells": cells, "answers": answers, "p2_offset": off}


def grid_summary(grid):
    from collections import Counter
    cnt = Counter()
    for (part, q, opt), exp in grid["answers"].items():
        if exp:
            cnt[part] += 1
    return dict(cnt)


# ---------- 3. CĂN KHỚP ẢNH VỚI MẪU ----------

def _match_fiducials(tsq, psq, approx):
    """Ghép ô mẫu <-> ô ảnh bằng RANSAC trên CẶP ĐIỂM — BẤT BIẾN XOAY.

    _quad4 chọn 4 đỉnh cực trị theo quy ước "ảnh gần thẳng": phiếu nghiêng
    5-30° làm cực trị rơi vào phần tử SAI (hộp mã đề, ô cạnh mép) -> H khớp
    nhầm 2/4 góc (q vẫn ~1.0 vì phiếu tự-similar). Hàm này thay vậy:

    với mỗi cặp ô mẫu (i,j) × cặp ô ảnh (k,l) có khoảng cách tương thích
    (thang trong [0.4, 2.5] × approx), biến đổi similarity i→k, j→l rồi dự
    đoán vị trí toàn bộ ô mẫu; ô ảnh gần nhất trong bán chịu tol tính 1 inlier
    (mỗi ô ảnh dùng 1 lần). Giả thuyết nhiều inlier nhất (≥4) thắng; trả
    (src 4 điểm mẫu, dst 4 điểm ảnh) trải rộng nhất để fit homography."""
    T = np.float32([[m[0], m[1]] for m in tsq])
    Pp = np.float32([[m[0], m[1]] for m in psq])
    n, m_ = len(T), len(Pp)
    if n < 4 or m_ < 4:
        return None
    DT = np.linalg.norm(T[:, None, :] - T[None, :, :], axis=-1)
    DP = np.linalg.norm(Pp[:, None, :] - Pp[None, :, :], axis=-1)
    # chỉ dùng cặp mốc GỐC DÀI (dt ≥ 40% cặp xa nhất): baseline ngắn cho nhiều
    # giả thuyết sai mà ít thông tin — chặt chẽ hơn và nhanh hơn nhiều lần
    dt_min = 0.40 * float(DT.max())
    med_px = float(np.median([p_[2] for p_ in psq]))   # cỡ mốc trên ảnh
    tol_tight = max(8.0, 0.40 * med_px)                # ngưỡng KHÍT theo mốc
    hyps = []                     # (số inlier lỏng, keep, nearest)
    for i in range(n):
        for j in range(i + 1, n):
            dt = float(DT[i, j])
            if dt < dt_min:
                continue
            lo_d, hi_d = 0.4 * approx * dt, 2.5 * approx * dt
            kp, lp = np.where((DP >= lo_d) & (DP <= hi_d))
            kp, lp = kp[kp < lp], lp[kp < lp]
            for k, l in zip(kp.tolist(), lp.tolist()):
                vT, vP = T[j] - T[i], Pp[l] - Pp[k]
                nT = float(np.linalg.norm(vT))
                sp = float(np.linalg.norm(vP)) / max(1e-9, nT)
                cos = float(vT @ vP) / max(1e-9, nT * np.linalg.norm(vP))
                sin = float(vT[0] * vP[1] - vT[1] * vP[0]) / max(1e-9, nT * np.linalg.norm(vP))
                Rm = np.array([[cos, -sin], [sin, cos]], np.float32)
                pred = (T - T[i]) @ Rm.T * sp + Pp[k]
                # tol LỎNG theo baseline (phối cảnh lệch xa góc tới % kích thước)
                tol = max(14.0, 0.025 * dt * sp)
                # dd dạng (số ô ảnh, số ô mẫu): khoảng cách mỗi ô ảnh -> mỗi ô mẫu
                dd = np.linalg.norm(Pp[:, None, :] - pred[None, :, :], axis=-1)
                nearest = dd.argmin(axis=0)          # (n,) ô ảnh gần nhất mỗi ô mẫu
                okm = dd[nearest, np.arange(n)] < tol
                if okm.sum() < 4:
                    continue
                seen, keep = set(), []
                for idx in range(n):
                    if okm[idx] and int(nearest[idx]) not in seen:
                        seen.add(int(nearest[idx]))
                        keep.append(idx)
                hyps.append((len(keep), keep, nearest))
                if len(keep) >= n:
                    break
            if any(h[0] >= n for h in hyps):
                break
        if any(h[0] >= n for h in hyps):
            break
    if not hyps:
        return None
    # ---- STAGE 2: chấm lại bằng HOMOGRAPHY + ngưỡng KHÍT ----
    # similarity chỉ xấp xỉ phối cảnh (lệch tới vài % cỡ ảnh ở góc xa) nên
    # tol lỏng bắt thêm giả thuyết SAI; fit H thật từ inlier rồi đếm lại inlier
    # với ngưỡng khít theo cỡ mốc — giả thuyết đúng sẽ bứt lên tuyệt đối.
    hyps.sort(key=lambda h: -h[0])
    best_tight, best_pick = 3, None
    for _lo_n, keep, nearest in hyps[:6]:
        if len(keep) < 4:
            continue
        src4 = np.float32([T[p] for p in keep])
        dst4 = np.float32([Pp[int(nearest[p])] for p in keep])
        Hm, _ = cv2.findHomography(src4, dst4, 0)
        if Hm is None:
            continue
        proj = cv2.perspectiveTransform(T.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        dd = np.linalg.norm(Pp[:, None, :] - proj[None, :, :], axis=-1)
        near2 = dd.argmin(axis=0)
        okm = dd[near2, np.arange(n)] < tol_tight
        seen, keep2 = set(), []
        for idx in range(n):
            if okm[idx] and int(near2[idx]) not in seen:
                seen.add(int(near2[idx]))
                keep2.append(idx)
        if len(keep2) > best_tight:
            best_tight, best_pick = len(keep2), (keep2, near2)
    if best_pick is None:
        return None
    keep, nearest = best_pick
    if len(keep) > 4:                   # lấy 4 inlier trải rộng nhất (cực trị)
        sub = T[keep]
        s = sub[:, 0] + sub[:, 1]
        d = sub[:, 0] - sub[:, 1]
        pick = [keep[int(np.argmin(s))], keep[int(np.argmax(d))],
                keep[int(np.argmax(s))], keep[int(np.argmin(d))]]
    else:
        pick = keep
    src = np.float32([T[p] for p in pick])
    dst = np.float32([Pp[int(nearest[p])] for p in pick])
    return src, dst


def _quad4(ms):
    """Chọn 4 đỉnh cực trị rồi sắp theo cùng một quy ước (bắt đầu TL, thuận chiều).

    Thứ tự theo góc quanh tâm + điểm bắt đầu min(x+y) ổn định với ảnh xoay < 45°;
    cả mẫu và ảnh dùng cùng quy ước nên tương ứng 1-1.
    """
    a = np.float32([[m[0], m[1]] for m in ms])
    s = a[:, 0] + a[:, 1]
    d = a[:, 0] - a[:, 1]
    quad = np.float32([a[np.argmin(s)], a[np.argmax(d)], a[np.argmax(s)], a[np.argmin(d)]])
    return _order_quad(quad)


def _order_quad(q):
    q = np.asarray(q, np.float32)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    k = int(np.argmin(q[:, 0] + q[:, 1]))
    q = np.roll(q, -k, axis=0)
    area = 0.5 * sum(q[i, 0] * q[(i + 1) % 4, 1] - q[(i + 1) % 4, 0] * q[i, 1] for i in range(4))
    if area < 0:
        q = q[::-1]
        k = int(np.argmin(q[:, 0] + q[:, 1]))
        q = np.roll(q, -k, axis=0)
    return q.astype(np.float32)


def _blob_centers(photo):
    """Tâm các cấu trúc tròn/nét trong ảnh (dùng cho kiểm tra khớp lưới + ICP).

    Adaptive threshold chịu được ảnh tối/ánh sáng không đều tốt hơn ngưỡng toàn cục.
    """
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 10)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if 40 <= a <= 8000 and 6 <= w <= 120 and 6 <= h <= 120 and 0.5 <= w / h <= 2.0:
            out.append((x + w / 2.0, y + h / 2.0))
    return np.float32(out) if out else np.zeros((0, 2), np.float32)


def _ring_centers(photo):
    """Tâm các VÒNG TRÒN IN (ring có đúng 1 lỗ con) — CHẶT hơn _blob_centers.

    _blob_centers nhận MỌI blob 6-120px (chữ số, khung, nét chữ) nên đếm
    'khớp lưới' quá dễ — phiếu in KHÁC layout vẫn đạt hit 1.0. Ring phải là
    contour ngoài có đúng 1 lỗ con, kích thước/hình dạng hợp lý."""
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    # KHÔNG GaussianBlur ở đây: ring photocopy NHẠT (nét 1-2 px, xám 100-200)
    # bị blur làm tan trước adaptiveThreshold — mất ~90 % ring (đo: 545 → 36);
    # adaptiveThreshold (blockSize 41) vốn đã chống nhiễu đủ tốt
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 10)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, hz = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hz is None:
        return np.zeros((0, 2), np.float32)
    for i, c in enumerate(cs):
        if hz[0][i][3] != -1:            # chỉ contour NGOÀI
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        lo_s, hi_s = min(rw, rh), max(rw, rh)
        if not (6 <= hi_s <= 60 and 0.55 <= lo_s / hi_s <= 1.45):
            continue
        kids = sum(1 for q in hz[0] if q[3] == i)
        if kids in (1, 2):
            # ring chuẩn (1 lỗ) hoặc ring CÓ DẤU CHẤM bên trong (2 con) —
            # bubble đã tô vẫn là mốc bám mẫu hợp lệ
            out.append((cx, cy))
            continue
        if kids == 0 and hi_s <= 22:
            # ring MẤT LỖ/ĐỨT NÉT do photocopy nhạt (contour thành vòng cung hở,
            # erode làm rỗng lõi nên không dùng erode): quyết bằng ĐỘ SÁNG TÂM —
            # ring: tâm = GIẤY SÁNG > vành mực; chấm đặc/ký tự: tâm TỐI
            ix, iy = int(round(cx)), int(round(cy))
            if not (0 <= ix < g.shape[1] and 0 <= iy < g.shape[0]):
                continue
            core_val = float(g[iy, ix])
            band_val = float(g[np.clip(c[:, 0, 1], 0, g.shape[0] - 1),
                               np.clip(c[:, 0, 0], 0, g.shape[1] - 1)].mean())
            if core_val - band_val > 25:
                out.append((cx, cy))
    return np.float32(out) if out else np.zeros((0, 2), np.float32)


def _scale_of(H):
    return float(np.sqrt(abs(np.linalg.det(np.asarray(H, np.float64)[:2, :2]))))


def _quality(H, cells, blobs):
    """Tỉ lệ ô chiếu vào đúng bubble ảnh (dung sai theo thang ảnh)."""
    if H is None or not len(blobs):
        return 0.0
    pts = np.float32([[c[3], c[4]] for c in cells]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    d = np.linalg.norm(proj[:, None, :] - blobs[None, :, :], axis=2).min(axis=1)
    return float((d < max(18.0, 0.62 * 7.0 * _scale_of(H))).mean())


def _h_valid(H, cells, img_shape, ref_cells=None):
    """Loại homography suy biến: các ô phải trải rộng đúng tỉ lệ trên ảnh."""
    cells = ref_cells if ref_cells else cells
    if H is None or not np.all(np.isfinite(H)):
        return False
    if abs(np.linalg.det(H)) < 1e-12:
        return False
    pts = np.float32([[c[3], c[4]] for c in cells]).reshape(-1, 1, 2)
    try:
        pr = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(pr)):
        return False
    Hh, Ww = img_shape[:2]
    bw = pr[:, 0].max() - pr[:, 0].min()
    bh = pr[:, 1].max() - pr[:, 1].min()
    tw = max(c[3] for c in cells) - min(c[3] for c in cells)
    th = max(c[4] for c in cells) - min(c[4] for c in cells)
    if tw < 1 or th < 1:
        return False
    r_img, r_tpl = bw / max(1e-6, bh), tw / max(1e-6, th)
    if not (0.35 * r_tpl <= r_img <= 2.8 * r_tpl):
        return False
    if bw * bh < 0.12 * Ww * Hh:
        return False
    if bw > 1.35 * Ww or bh > 1.35 * Hh:
        return False
    return True


_SIFT_CACHE = {}


def _sift_features(gray):
    sift = cv2.SIFT_create(nfeatures=4000)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def align_sift(photo, template_png, cells, ref_cells=None, max_side=1200):
    """Fallback: khớp đặc trưng ảnh (SIFT) khi thiếu ô đen định vị.

    Ảnh được hạ mẫu về ~max_side để tỉ lệ với mẫu (giảm sai khớp do chênh độ phân giải).
    """
    tkey = (str(template_png), id(cells))
    if tkey not in _SIFT_CACHE:
        tg = cv2.imread(str(template_png), cv2.IMREAD_GRAYSCALE)
        _SIFT_CACHE[tkey] = (tg.shape, _sift_features(tg))
    tshape, (tkp, tdes) = _SIFT_CACHE[tkey]
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    k = min(1.0, float(max_side) / max(1, max(g.shape)))
    if k < 0.999:
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    pkp, pdes = _sift_features(g)
    if tdes is None or pdes is None or len(pkp) < 10 or len(tkp) < 10:
        return None, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(tdes, pdes, k=2)
    good = []
    for m in knn:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < 12:
        return None, 0.0
    src = np.float32([tkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([pkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None or mask is None or mask.sum() < 10:
        return None, 0.0
    # quy H từ ảnh-hạ-mẫu về ảnh gốc: x_small = k*x_full
    if k < 0.999:
        K = np.array([[k, 0, 0], [0, k, 0], [0, 0, 1]], np.float64)
        H = np.linalg.inv(K) @ H
    H = H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0)
    if not _h_valid(H, cells, photo.shape, ref_cells):
        return None, 0.0
    return H, _quality(H, cells, _blob_centers(photo))


def align_photo(photo_path, template_png, cells, photo=None, ref_cells=None,
                allow_sift=True):
    """Căn ảnh bài thi với mẫu: 4 ô đen góc -> homography -> ICP tinh chỉnh.

    Trả (H, photo_bgr, quality 0..1) với H: toạ độ mẫu -> toạ độ ảnh.
    allow_sift=False: bỏ dự phòng SIFT (rẻ hơn ~1,5 s/lần) — dùng khi quét
    nhiều hướng xoay, sau đó chạy lại một lần có SIFT nếu cần.
    """
    ph = photo if photo is not None else cv2.imread(str(photo_path))
    if ph is None:
        raise FileNotFoundError(photo_path)
    tsq = _fiducial_squares(template_png, side=(6, 16))
    if len(tsq) < 4:
        if not allow_sift:
            return None, ph, 0.0
        Hs, qs = align_sift(ph, template_png, cells, ref_cells)
        return Hs, ph, qs
    src = _quad4(tsq)
    blobs = _blob_centers(ph)
    if len(blobs) < 20:
        return None, ph, 0.0
    # tìm dải kích thước mốc trong ảnh cho khớp mẫu (thang ảnh/mẫu ~ W_ảnh/W_mẫu)
    # Dải bắt đầu 18 px (trước đây 30): hỗ trợ ảnh chụp XA hơn — template 461 px
    # có mốc 8-9 px nên ảnh ~2-3.3× (mốc 18-30 px) trước đây không thể căn.
    # Các ô lớn (hộp mã đề 250 px+) vẫn bị chặn bởi hi = lo*2.2 và _quad4.
    tw = cv2.imread(str(template_png)).shape[1]
    approx = ph.shape[1] / max(1, tw)          # ~7-8x
    best = None
    # ---- GHÉP MỐC RANSAC CẶP ĐIỂM (bất biến xoay/nghiêng) ----
    # quét dải kích thước theo THỨ TỰ GẦN CỠ MỐC KỲ VỌNG nhất (mốc mẫu ~8-9 px
    # × thang approx): dải đúng thường thành công ngay, khỏi quét hết 12 dải.
    exp_px = 9.0 * approx
    los = sorted(range(18, 110, 8),
                 key=lambda lo_: abs((lo_ + lo_ * 2.2) / 2.0 - exp_px))
    for lo in los:
        hi = int(lo * 2.2)
        psq = _fiducial_squares(ph, side=(lo, hi))
        if len(psq) < 4:
            continue
        mt = _match_fiducials(tsq, psq, approx)
        if mt is None:
            continue
        src_m, dst_m = mt
        H = cv2.getPerspectiveTransform(src_m, dst_m)
        if not _h_valid(H, cells, ph.shape, ref_cells):
            continue
        q = _quality(H, cells, blobs)
        if best is None or q > best[0]:
            best = (q, H)
        if q >= 0.90:                    # đủ tốt — không quét thêm dải
            break
    if best is None:
        # dự phòng: quy ước cực trị cũ (ảnh gần thẳng) theo từng dải size
        for lo in range(18, 110, 8):
            hi = int(lo * 2.2)
            psq = _fiducial_squares(ph, side=(lo, hi))
            if len(psq) < 4:
                continue
            dst = _quad4(psq)
            # lọc theo tỉ lệ thang hợp lý
            scale = np.linalg.norm(dst[1] - dst[0]) / max(1e-6, np.linalg.norm(src[1] - src[0]))
            if not (0.35 * approx <= scale <= 3.0 * approx):
                continue
            H = cv2.getPerspectiveTransform(src, dst)
            if not _h_valid(H, cells, ph.shape, ref_cells):
                continue
            q = _quality(H, cells, blobs)
            if best is None or q > best[0]:
                best = (q, H)
    if best is None:
        if not allow_sift:
            return None, ph, 0.0
        # fallback SIFT
        Hs, qs = align_sift(ph, template_png, cells, ref_cells)
        return Hs, ph, qs
    q, H = best
    # ICP: ghép TẤT CẢ ô mẫu (phủ toàn phiếu, kể cả SBD/Mã đề) với blob gần nhất
    # -> hiệu chỉnh cả vùng rìa; chất lượng q vẫn đo trên các ô có đáp án.
    icp_cells = ref_cells if ref_cells else cells
    tpl_pts = np.float32([[c[3], c[4]] for c in icp_cells])
    for _ in range(4):
        proj = cv2.perspectiveTransform(tpl_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        dd = np.linalg.norm(proj[:, None, :] - blobs[None, :, :], axis=2)
        j = dd.argmin(axis=1)
        dm = dd[np.arange(len(proj)), j]
        keep = dm < max(30.0, 0.9 * 7.0 * _scale_of(H))
        if keep.sum() < 30:
            break
        Hn, mask = cv2.findHomography(tpl_pts[keep], blobs[j][keep], cv2.RANSAC, 10.0)
        if Hn is None or not _h_valid(Hn, cells, ph.shape, ref_cells):
            break
        H = Hn
        q = _quality(H, cells, blobs)
    if q < 0.5 and allow_sift:
        Hs, qs = align_sift(ph, template_png, cells, ref_cells)
        if Hs is not None and qs > q:
            return Hs, ph, qs
    return H, ph, q


# ---------- 4. ĐỌC ĐÁP ÁN TRÊN ẢNH ----------

def _ring_index(photo):
    """Tâm + bán kính các vòng tròn/bubble trong ảnh."""
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    bg = cv2.blur(g.astype(np.float32), (51, 51))
    norm = g.astype(np.float32) / (bg + 1e-6)
    dark = (norm < 0.80).astype(np.uint8) * 255
    cs, hz = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (30 <= a <= 5000) or w < 8 or h < 8:
            continue
        circ = 4 * np.pi * a / (cv2.arcLength(c, True) ** 2 + 1e-9)
        if circ > 0.55 and 0.60 <= w / h <= 1.65:
            out.append((x + w / 2.0, y + h / 2.0, max(w, h)))
    return np.float32(out) if out else np.zeros((0, 3), np.float32)


def read_cells(photo, cells, H, thresh=0.42, snap_frac=0.85, exp_w=None):
    """Đọc mực từng ô. Trả {(part,câu,opt): (score, cx, cy, r)}.

    Homography đã chính xác (ICP) nên KHÔNG snap xa: chỉ nhận snap khi lệch
    <= 0.85 bán kính bubble (tránh nhảy sang hàng/câu khác).
    """
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    bg = cv2.blur(g.astype(np.float32), (51, 51))
    norm = g.astype(np.float32) / (bg + 1e-6)
    rings = _ring_index(photo)
    # kích thước bubble kỳ vọng từ thang homography (bubble mẫu ~7px)
    if exp_w is None:
        exp_w = 7.0 * _scale_of(H)
    med_w = exp_w
    if len(rings):
        cand = rings[(rings[:, 2] >= 0.55 * exp_w) & (rings[:, 2] <= 1.75 * exp_w), 2]
        if len(cand) >= 5:
            med_w = float(np.median(cand))
    med_r = med_w * 0.5
    pts = np.float32([[c[3], c[4]] for c in cells]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    out = {}
    for cell, (px, py) in zip(cells, proj):
        cx, cy, rr = px, py, med_r
        if len(rings):
            d = np.linalg.norm(rings[:, :2] - np.float32([px, py]), axis=1)
            j = int(np.argmin(d))
            if d[j] <= med_r * snap_frac and abs(rings[j][2] - med_w) <= med_w * 0.55:
                cx, cy = rings[j][:2]
                rr = rings[j][2] * 0.5
        r = max(5, int(rr * 0.62))
        x0, y0 = int(cx - r), int(cy - r)
        x1, y1 = int(cx + r + 1), int(cy + r + 1)
        if x0 < 0 or y0 < 0 or x1 > norm.shape[1] or y1 > norm.shape[0]:
            out[(cell[0], cell[1], cell[2])] = (0.0, cx, cy, rr)
            continue
        roi = norm[y0:y1, x0:x1]
        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        disk = ((xx - roi.shape[1] / 2 + 0.5) ** 2 + (yy - roi.shape[0] / 2 + 0.5) ** 2) <= (r * 0.95) ** 2
        vals = roi[disk]
        if not vals.size:
            out[(cell[0], cell[1], cell[2])] = (0.0, cx, cy, rr)
            continue
        # ĐIỂM MỰC = độ đậm TRUNG BÌNH trong lòng vòng (1 - norm), kẹp 0..1.
        # Đo "tỉ lệ điểm tối < 0,78" chỉ đúng với ảnh in đậm; bản photocopy /
        # bút chì mờ cho pixel 0,85-0,95 nên bị tính là TRỐNG (đo trên trang
        # 12.png thật: ô đã tô chỉ đạt 0,16-0,19 trong khi ô trống 0,0 -> lệch
        # cột/ô). Trung bình độ đậm tách rõ: ô tô ~0,12-0,15, ô trống ~0,01-0,04.
        score = float(np.clip((1.0 - float(vals.mean())) / 0.30, 0.0, 1.0))
        out[(cell[0], cell[1], cell[2])] = (score, cx, cy, rr)
    return out


def adapt_thresh(reads, lo=0.08, hi=0.40, margin=0.05):
    """Ngưỡng mực TỰ THÍCH ỨNG theo từng ảnh.

    Bút chì mờ / bản photocopy cho ô đã tô chỉ ~0,25-0,45 mật độ mực, trong khi
    ngưỡng cứng 0,35-0,42 biến chúng thành "trống" (đo trên ảnh thật: 14/40 câu
    bị bỏ oan). Ngược lại giấy in đậm có nền mực ~0,15 nên ngưỡng thấp sẽ bắt
    nhầm. Vì vậy lấy nền nhiễu là phân vị 65 của TOÀN BỘ ô trong ảnh (đa số ô
    là trống) rồi cộng biên, kẹp trong [lo, hi].
    """
    sc = np.array([v[0] for v in reads.values()], np.float32)
    if sc.size < 8:
        return 0.25
    # TRUNG VỊ (p50) là mức nền: trên phiếu, tuyệt đại đa số ô trống nên p50
    # xấp xỉ mực in nền. Dùng p65+0,10 (bản cũ) đẩy ngưỡng lên 0,19 trên ảnh
    # bút chì mờ -> 8 câu bị bỏ oan (đo trên PXL_20261102_120214113).
    noise = float(np.percentile(sc, 50))
    return float(np.clip(noise + margin, lo, hi))


def decide(reads, grid=None, thresh=None, rel=0.72):
    """Gom theo câu -> trạng thái: one / blank / multi.

    Quyết định TƯƠNG ĐỐI để chống nhiễu bản photocopy:
      - s1 < thresh                -> trống  (thresh=None: tự thích ứng theo ảnh)
      - s2 >= max(thresh, rel*s1)  -> tô nhiều đáp án
      - còn lại                    -> chọn ô điểm cao nhất
    """
    from collections import defaultdict
    if thresh is None:
        thresh = adapt_thresh(reads)
    by_q = defaultdict(list)
    for (part, q, opt), (s, cx, cy, r) in reads.items():
        by_q[(part, q)].append((opt, s, cx, cy, r))
    decided = {}
    for key_q, lst in by_q.items():
        lst = sorted(lst, key=lambda z: -z[1])
        s1 = lst[0][1] if lst else 0.0
        s2 = lst[1][1] if len(lst) > 1 else 0.0
        if s1 < thresh:
            state, sel = "blank", []
        elif s2 >= max(thresh, rel * s1):
            state = "multi"
            sel = [o for o, s, _, _, _ in lst if s >= max(thresh, rel * s1)]
        else:
            state, sel = "one", [lst[0][0]]
        decided[key_q] = {"state": state, "sel": sel,
                          "cells": {o: (s, cx, cy, r) for o, s, cx, cy, r in lst}}
    return decided


def grade(decided, grid, part1_score=0.25, part2_score=0.25):
    """Tính đúng/sai từng câu + điểm. Trả (điểm, chi tiết, thống kê)."""
    detail, stats = {}, {"P1": {"ok": 0, "n": 0}, "P2": {"ok": 0, "n": 0}}
    s1 = s2 = 0.0
    for (part, q), info in decided.items():
        exp = None
        for opt, (s, cx, cy, r) in info["cells"].items():
            e = grid["answers"].get((part, q, opt))
            if e:
                exp = e
                break
        if exp is None:
            continue  # ô không thuộc key
        per = part1_score if part == "P1" else part2_score
        if info["state"] == "one":
            got = info["sel"][0]
            ok = got == exp
            if ok:
                detail[(part, q)] = ("ok", exp, got)
                stats[part]["ok"] += 1
                if part == "P1":
                    s1 += per
                else:
                    s2 += per
            else:
                detail[(part, q)] = ("wrong", exp, got)
        elif info["state"] == "multi":
            detail[(part, q)] = ("multi", exp, ",".join(info["sel"]))
        else:
            detail[(part, q)] = ("blank", exp, "")
        stats[part]["n"] += 1
    return round(s1, 2), round(s2, 2), detail, stats


# ---------- 4b. SỐ BÁO DANH & MÃ ĐỀ TRÊN LƯỚI MẪU ----------

ID_REGION = {"12": (0.60, 0.36), "10_11": (0.72, 0.36)}


def _small_circles(img):
    """Vòng tròn nhỏ của lưới SBD/Mã đề (4-7px): nhận cả lỗ trong của contour."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (8 <= a <= 150 and 3 <= w <= 14 and 3 <= h <= 14 and 0.60 <= w / h <= 1.60):
            continue
        p = cv2.arcLength(c, True)
        if 4 * np.pi * a / (p * p + 1e-9) < 0.70:
            continue
        out.append((x + w / 2.0, y + h / 2.0, float(w)))
    res = []
    for m in out:
        if all(abs(m[0] - o[0]) > 1.5 or abs(m[1] - o[1]) > 1.5 for o in res):
            res.append(m)
    return res


def _fit_lattice(vals, lo_gap, hi_gap, expect=10, tol=0.30):
    """Tìm lưới đều (start, pitch) khớp nhiều điểm nhất — chịu được hàng bị sót.

    Ứng viên pitch lấy từ các hiệu gần nhau (và nửa hiệu, phòng khi cách 2 hàng);
    với mỗi (start, pitch) đếm số hàng phân biệt khớp. Trả (start, pitch, expect).
    """
    vals = sorted(vals)
    if len(vals) < 4:
        return None
    diffs = [b - a for a, b in zip(vals, vals[1:]) if lo_gap <= b - a <= hi_gap]
    if not diffs:
        return None
    cand_p = []
    for d in diffs:
        for pp in (d, d / 2.0, d / 3.0):
            if lo_gap <= pp <= hi_gap and all(abs(pp - q) > 0.15 for q in cand_p):
                cand_p.append(pp)
    # Neo vào HÀNG CUỐI (thường in rõ, ít bị che) rồi suy ngược: y0 = y_bottom - 9*pitch.
    # Nhờ vậy không bị lệch 1 hàng khi hàng đầu bị sót.
    best = None
    for pitch in cand_p:
        for yb in vals:
            ks = {}
            for v in vals:
                t = (yb - v) / pitch
                k = int(round(t))
                if abs(t - k) <= tol and 0 <= k < expect:
                    ks[k] = min(ks.get(k, 9e9), abs(t - k))
            if 0 not in ks:          # y_bottom phải đúng là hàng cuối
                continue
            score = (len(ks), -sum(ks.values()) / max(1, len(ks)))
            if len(ks) >= 6 and (best is None or score > best[0]):
                best = (score, yb, pitch)
    if best is None:
        return None
    yb, pitch = float(best[1]), float(best[2])
    return yb - (expect - 1) * pitch, pitch, expect


def build_id_cells(paper, template_png):
    """Ô SBD/Mã đề trên mẫu: [(phần 'SBD'|'MD', cột, chữ_số, x, y)]."""
    ck = ("id", paper, str(template_png))
    if ck in _CELLS_CACHE:
        return _CELLS_CACHE[ck]
    img = cv2.imread(str(template_png))
    if img is None:
        raise FileNotFoundError(template_png)
    Hh, Ww = img.shape[:2]
    xf, yf = ID_REGION[paper]
    ms = [m for m in _small_circles(img) if m[1] < yf * Hh and m[0] > xf * Ww]
    # cụm cột (pitch dọc ~7-14px) — tách SBD và MĐ bằng khoảng trống > 1.8 pitch
    xs = sorted(m[0] for m in ms)
    cols = _col_groups([(x, 0, 0) for x in xs], gap=6)
    col_x = [c[0] for c in cols]
    if not col_x:
        _CELLS_CACHE[ck] = []
        return []
    pitch_x = float(np.median([b - a for a, b in zip(col_x, col_x[1:])])) if len(col_x) > 1 else 9.0
    groups = [[col_x[0]]]
    for x in col_x[1:]:
        if x - groups[-1][-1] <= 1.8 * pitch_x:
            groups[-1].append(x)
        else:
            groups.append([x])
    cells = []
    fitted = []
    for g in groups:
        ys = [m[1] for m in ms if any(abs(m[0] - gx) <= 3 for gx in g)]
        fit = _fit_lattice(ys, 5.0, 20.0, expect=10)
        if not fit:
            continue
        y0, pitch, n = fit
        # chỉ giữ CỘT có đủ số vòng khớp lưới (loại cột rác/mốc đen)
        good_cols = []
        for gx in g:
            near = [(m[0], m[1]) for m in ms if abs(m[0] - gx) <= 3]
            hit = 0
            for (mx, my) in near:
                t = (my - y0) / pitch
                k = int(round(t))
                if abs(t - k) <= 0.35 and 0 <= k < n:
                    hit += 1
            if hit >= 4:
                good_cols.append((float(np.mean([mx for mx, _ in near])), hit))
        if good_cols:
            fitted.append((min(c[0] for c in good_cols), good_cols, y0, pitch, n))
    # đặt tên theo HÌNH HỌC: nhóm bên trái = SBD, nhóm bên phải = Mã đề
    fitted.sort(key=lambda z: z[0])
    for gi, (xleft, good_cols, y0, pitch, n) in enumerate(fitted):
        if len(fitted) >= 2:
            name = "SBD" if gi == 0 else "MD"
        else:
            name = "MD"
        # dựng lại lưới CỘT đều từ 2 biên (không để mất cột giữa do sót detection)
        gx = sorted(c[0] for c in good_cols)
        if len(gx) >= 2:
            px = float(np.median([b - a for a, b in zip(gx, gx[1:])]))
            ncols = int(round((gx[-1] - gx[0]) / max(1e-6, px))) + 1
            col_x = [gx[0] + i * px for i in range(ncols)]
        else:
            col_x = gx
        for ci, cx in enumerate(col_x):
            for r in range(n):
                cells.append((name, ci + 1, str(r), cx, y0 + r * pitch))
    _CELLS_CACHE[ck] = cells
    return cells


def _blobs_with_size(img):
    """Blob tròn/nét kèm kích thước (dùng cho lưới ID cục bộ)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 10)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if 40 <= a <= 12000 and 8 <= w <= 140 and 8 <= h <= 140 and 0.5 <= w / h <= 2.0:
            out.append((x + w / 2.0, y + h / 2.0, float(max(w, h))))
    return out


def _proj_lattice(proj, n, lo, hi):
    """Tìm (offset, pitch) của n đường kẻ từ projection 1D bằng autocorrelation."""
    p = np.asarray(proj, np.float32)
    if p.size < 8 or p.max() <= 0:
        return None
    p = p - p.mean()
    ac = np.correlate(p, p, mode="full")[p.size - 1:]
    lo_i = max(2, int(lo))
    hi_i = min(len(ac) - 2, int(hi))
    if hi_i <= lo_i:
        return None
    lag = lo_i + int(np.argmax(ac[lo_i:hi_i + 1]))
    best = None
    for pitch in (lag - 1, lag, lag + 1):
        if pitch < lo:
            continue
        for off in range(0, int(pitch)):
            idx = off + np.arange(n) * pitch
            if idx[-1] >= proj.size:
                continue
            vals = [float(np.max(proj[max(0, int(i) - 1):int(i) + 2])) for i in idx]
            sc = float(np.sum(vals))
            if best is None or sc > best[0]:
                best = (sc, off, float(pitch))
    if best is None:
        return None
    return best[1], best[2]


def local_id_cells(photo, H, id_cells, exp_w=None):
    """Dò lại lưới SBD/Mã đề NGAY TRÊN ẢNH trong vùng đã chiếu từ mẫu.

    Cột: cụm tâm vòng tròn; Hàng: projection + autocorrelation (10 hàng).
    Cần thiết vì có bản in có lưới ID khác tỉ lệ so với ảnh mẫu.
    """
    if not id_cells:
        return []
    if exp_w is None:
        exp_w = 7.0 * _scale_of(H)
    pts = np.float32([[c[3], c[4]] for c in id_cells]).reshape(-1, 1, 2)
    pr = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    mx = 1.8 * exp_w
    x0 = max(0, int(pr[:, 0].min() - mx)); x1 = min(photo.shape[1], int(pr[:, 0].max() + mx))
    y0 = max(0, int(pr[:, 1].min() - mx)); y1 = min(photo.shape[0], int(pr[:, 1].max() + mx))
    if x1 - x0 < 4 * exp_w or y1 - y0 < 4 * exp_w:
        return []
    sub = photo[y0:y1, x0:x1]
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 41, 10)
    # --- hàng: 10 hàng, pitch suy từ autocorrelation ---
    rowproj = (mask > 0).sum(axis=1).astype(np.float32)
    rfit = _proj_lattice(rowproj, 10, lo=0.95 * exp_w, hi=2.6 * exp_w)
    if rfit is None:
        return []
    y_off, pitch_y = rfit
    row_y = [y_off + k * pitch_y for k in range(10)]
    # --- cột: cụm x của vòng tròn ---
    cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cxs = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (60 <= a <= 12000 and 10 <= w <= 140 and 10 <= h <= 140 and 0.6 <= w / h <= 1.6):
            continue
        p = cv2.arcLength(c, True)
        if 4 * np.pi * a / (p * p + 1e-9) < 0.78:
            continue
        # chỉ lấy vòng nằm gần 1 trong 10 hàng
        if min(abs((y + h / 2) - ry) for ry in row_y) <= 0.45 * pitch_y:
            cxs.append(x + w / 2.0)
    if len(cxs) < 8:
        return []
    cxs = sorted(cxs)
    pitch_x = float(np.median([b - a for a, b in zip(cxs, cxs[1:]) if 0.5 * exp_w <= b - a <= 1.8 * exp_w] or [exp_w]))
    cols = [[cxs[0]]]
    for x in cxs[1:]:
        if x - cols[-1][-1] <= 0.55 * pitch_x:
            cols[-1].append(x)
        else:
            cols.append([x])
    col_x = [float(np.mean(c)) for c in cols]
    col_x = [x for x in col_x if sum(1 for v in cxs if abs(v - x) <= 0.5 * pitch_x) >= 2]
    if len(col_x) < 3:
        return []
    # Tách nhóm SBD/MĐ: nếu đủ 12 cột (8 SBD + 4 MĐ) thì cắt tại KHE LỚN NHẤT.
    dx = [b - a for a, b in zip(col_x, col_x[1:])]
    med_dx = float(np.median(dx)) if dx else pitch_x
    groups = []
    if len(col_x) >= 10:
        gi = int(np.argmax(dx))
        left, right = col_x[:gi + 1], col_x[gi + 1:]
        if 6 <= len(left) <= 9 and 3 <= len(right) <= 5:
            groups = [("SBD", left), ("MD", right)]
        else:
            groups = [("MD", col_x)]
    elif len(col_x) in (3, 4):
        groups = [("MD", col_x)]
    elif len(col_x) > 4:
        gi = int(np.argmax(dx))
        left, right = col_x[:gi + 1], col_x[gi + 1:]
        groups = [("MD", left)] if 3 <= len(left) <= 4 else []
        if 3 <= len(right) <= 4:
            groups.append(("MD", right))
    out = []
    for name, gr in groups:
        if len(gr) < 3:
            continue
        px = float(np.median([b - a for a, b in zip(gr, gr[1:])])) if len(gr) > 1 else med_dx
        ncols = int(round((gr[-1] - gr[0]) / max(1e-6, px))) + 1
        want = 8 if name == "SBD" else (3 if len(gr) == 3 else 4)
        if abs(ncols - want) > 1:
            continue
        ncols = want
        for ci in range(ncols):
            gx = gr[0] + ci * px
            for r, ry in enumerate(row_y):
                out.append((name, ci + 1, str(r), gx + x0, ry + y0))
    return out


def read_ids(photo, id_cells, H, thresh=0.35, exp_w=None):
    """Giải SBD / Mã đề: mỗi cột lấy chữ số có điểm cao nhất.

    Trả {"SBD": "50120074" | None, "MD": "3157" | None,
         "score": {"SBD": [...], "MD": [...]}, "raw": {...}}
    """
    if not id_cells:
        return {"SBD": None, "MD": None, "score": {}, "raw": {}, "positions": {}}
    # id_cells đã ở toạ độ ẢNH (local_id_cells) -> H=None nghĩa là không biến đổi
    Hm = np.eye(3, dtype=np.float64) if H is None else H
    h_, w_ = photo.shape[:2]
    if H is None:
        # tránh đọc ngoài biên: read_cells cần H suy ra thang; dùng H ước lượng theo bubble
        Hm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    reads = read_cells(photo, id_cells, Hm, exp_w=exp_w)
    from collections import defaultdict
    by_col = defaultdict(list)
    for (part, col, dig), (s, cx, cy, r) in reads.items():
        by_col[(part, col)].append((str(dig), s))
    out, score, raw, positions = {}, {}, {}, {}
    for part in ("SBD", "MD"):
        cols = sorted(c for (p, c) in by_col if p == part)
        digits, detail, pos = [], [], []
        for c in cols:
            lst = sorted(by_col[(part, c)], key=lambda z: -z[1])
            d, s = lst[0]
            detail.append((c, int(d), round(s, 2)))
            digits.append(d if s >= thresh else "?")
            sc, cx, cy, rr = reads[(part, c, d)]
            if s >= thresh:
                pos.append((c, d, float(cx), float(cy), float(rr)))
        positions[part] = pos
        if cols and "?" not in digits:
            out[part] = "".join(digits)
        else:
            out[part] = None      # thiếu/không chắc -> coi như không đọc được
        score[part] = detail
        raw[part] = digits
    return {"SBD": out.get("SBD"), "MD": out.get("MD"), "score": score, "raw": raw,
            "positions": positions}


# ---------- 5. VẼ KẾT QUẢ ----------

def _p2_line(detail):
    """Chuỗi P2 theo từng câu: '4/4; 4/4; 4/4; 4/4'."""
    by_q = {}
    for (part, q), (st, exp, got) in detail.items():
        if part != "P2":
            continue
        cq = q[0] if isinstance(q, tuple) else q
        ok, n = by_q.get(cq, (0, 0))
        by_q[cq] = (ok + (1 if st == "ok" else 0), n + 1)
    if not by_q:
        return "0/0"
    return "; ".join(f"{by_q[c][0]}/{by_q[c][1]}" for c in sorted(by_q))


def _p1_line(stats):
    return f"{stats['P1']['ok']}/{stats['P1']['n']}"


_RULE_FRAC = {}


def template_rule_frac(tpl_path, lo=0.18, hi=0.42):
    """Tỉ lệ y/H của DÒNG KẺ ĐỨT trên ẢNH MẪU.

    Dòng kẻ đứt có RẤT NHIỀU đoạn mực ngắn (mỗi gạch là một đoạn) — khác với
    đường kẻ bảng (1-2 đoạn dài) và dòng chữ (30-40 đoạn nhưng mực thấp).
    Điều kiện: số đoạn mực >= 40 và mật độ mực >= 0.22; chọn hàng đậm nhất.
    """
    ck = str(tpl_path)
    if ck in _RULE_FRAC:
        return _RULE_FRAC[ck]
    img = cv2.imread(str(tpl_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = (g < 128)
    x0, x1 = int(w * 0.03), int(w * 0.97)
    best = None
    for y in range(max(3, int(h * lo)), min(h - 3, int(h * hi))):
        row = d[y, x0:x1].astype(np.int8)
        r = float(row.mean())
        if not (0.10 <= r <= 0.60):
            continue
        runs = int(((row[1:] == 1) & (row[:-1] == 0)).sum())
        if runs < 30:
            continue
        if best is None or r > best[1]:
            best = (y, r)
    val = None if best is None else best[0] / float(h)
    _RULE_FRAC[ck] = val
    return val


# Vị trí tương đối (theo chiều cao ảnh) của 3 dòng điểm — đo trên ảnh kết quả
# chuẩn của mẫu 12.png; mỗi dòng nằm ngay TRÊN tiêu đề PHẦN tương ứng.
def _find_section_rows(img, lo=0.25, hi=0.90):
    """Dò 3 HÀNG TIÊU ĐỀ "PHẦN I/II/III" trên ảnh đã chấm.

    Đặc trưng: chữ in hoa nằm SÁT MÉP TRÁI (2-13 % chiều rộng) còn phần lớn
    bên phải của hàng đó TRỐNG — nhờ vậy không nhầm với hàng bubble.
    Ngưỡng mực hạ dần (0,06 → 0,03) vì bản photocopy in rất nhạt; dừng khi đủ
    3 cụm. Trả [y1, y2, y3] hoặc [].
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    d = (g < 128)
    # Dải TRÁI hẹp (1,5-9,5 %) = chỗ chữ "PHẦN …"; dải PHẢI bắt đầu từ 12 %
    # (đã vào vùng bubble) — nhờ vậy các hàng BỊ TÔ ĐEN ở cột A (x≈13-18 %)
    # không bị nhận nhầm là tiêu đề.
    left = d[:, int(w * 0.015):int(w * 0.095)].mean(axis=1)
    right = d[:, int(w * 0.12):int(w * 0.95)].mean(axis=1)
    for lt in (0.06, 0.05, 0.04, 0.03):
        rows = [y for y in range(int(h * lo), int(h * hi))
                if left[y] > lt and right[y] < 0.02]
        if not rows:
            continue
        groups = [[rows[0]]]
        for y in rows[1:]:
            if y - groups[-1][-1] <= 8:
                groups[-1].append(y)
            else:
                groups.append([y])
        ys = [float(np.median(gg)) for gg in groups if len(gg) >= 3]
        if ys:
            return ys
    return []


_SEC_FIXED_12 = {"P1": 0.324, "P2": 0.518, "P3": 0.681}


_SEC_Y = {}


def _section_headings(tpl_path, paper="12"):
    """y (toạ độ MẪU) của 3 tiêu đề PHẦN I / II / III trên ảnh mẫu 12.png.

    Dò trực tiếp DÒNG CHỮ tiêu đề trong dải ngay TRÊN khối ô của từng phần
    (dòng in hoa "PHẦN I"… có mực vừa, nhiều đoạn nét). Không suy từ vị trí ô
    vì khoảng cách tiêu đề → hàng đầu khác nhau giữa các phần.
    """
    ck = str(tpl_path)
    if ck in _SEC_Y:
        return _SEC_Y[ck]
    img = cv2.imread(str(tpl_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = (g < 128)
    x0, x1 = int(w * 0.02), int(w * 0.55)

    def _find_in(y_lo, y_hi):
        best = None
        for y in range(max(2, int(y_lo)), min(h - 2, int(y_hi))):
            row = d[y, x0:x1].astype(np.int8)
            r = float(row.mean())
            if not (0.03 <= r <= 0.18):
                continue
            runs = int(((row[1:] == 1) & (row[:-1] == 0)).sum())
            if runs < 12:
                continue
            if best is None or r > best[1]:
                best = (y, r)
        return None if best is None else float(best[0])

    try:
        cells = build_template_cells(paper, tpl_path)
    except Exception:                                            # noqa
        cells = []
    out = {}
    for part, key_ in (("P1", "P1"), ("P2", "P2")):
        ys = [c[4] for c in cells if c[0] == part]
        if not ys:
            continue
        y_top = min(ys)
        yy = _find_in(y_top - 55, y_top - 6)
        out[key_] = yy if yy is not None else y_top - 26.0
    if "P3" not in out:
        yy = _find_in(h * 0.64, h * 0.82)
        if yy is not None:
            out["P3"] = yy
    _SEC_Y[ck] = out or None
    return _SEC_Y[ck]


def _find_dashed_rule(img, lo=0.18, hi=0.29, min_ratio=0.04):
    """Tìm DÒNG KẺ ĐỨT dưới dòng "Chú ý: Thí sinh…" — trả y hoặc None.

    Dòng đứt nằm ngay dưới dòng chữ đó và TRÊN khung PHẦN I; trong dải
    18–34 % chiều cao, nó là hàng có mật độ mực lớn nhất (các gạch đứt nối
    liền). Cách này ổn định hơn việc dò "hàng sạch" vì bản photocopy có nhiều
    hàng nhiễu.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    x0, x1 = int(w * 0.05), int(w * 0.95)
    ratio = (g[:, x0:x1] < 128).mean(axis=1)
    y0, y1 = int(h * lo), int(h * hi)
    if y1 <= y0 + 4:
        return None
    band = ratio[y0:y1]
    k = int(np.argmax(band))
    return int(y0 + k) if float(band[k]) >= min_ratio else None


def render_result(photo, decided, detail, stats, out_path, paper, made=None,
                  s1=0.0, s2=0.0, radius_scale=1.0, sbd=None, ids=None,
                  p3_text="0/0", show_sbd=False, md_inferred=False,
                  md_note="", tpl_path=None):
    """Vẽ kết quả theo mẫu tham chiếu:

    - Khối chữ ĐỎ lớn góc trái trên:
        P1: 24/24 = 6.00
        P2: 4/4; 4/4; 4/4; 4/4 = 4.00
        P3: 0/0 = 0.00
        Tong: 10.00
    - Vòng XANH ĐẬM quanh bubble tô đúng, ĐỎ quanh bubble tô sai,
      VÀNG cho câu trống/tô nhiều; vòng xanh cả chữ số SBD/Mã đề đã chọn.
    """
    img = photo.copy()
    Hh, Ww = img.shape[:2]
    GREEN_B = (0, 235, 0)
    RED_B = (0, 0, 240)
    YELLOW_B = (0, 215, 255)

    # ---- vòng bubble ----
    for (part, q), info in decided.items():
        st, exp, got = detail.get((part, q), (None, None, None))
        if st is None:
            continue
        if st in ("ok", "wrong"):
            col = GREEN_B if st == "ok" else RED_B
            s, cx, cy, r = info["cells"][got]
            lw = max(2, int(r * 0.42))
            cv2.circle(img, (int(cx), int(cy)), int(max(9, r * 1.28)), col, lw, cv2.LINE_AA)
        else:  # blank / multi -> vàng toàn bộ ô của câu
            for opt, (s, cx, cy, r) in info["cells"].items():
                lw = max(2, int(r * 0.36))
                cv2.circle(img, (int(cx), int(cy)), int(max(9, r * 1.28)), YELLOW_B, lw, cv2.LINE_AA)

    # ---- vòng ĐÁP ÁN ĐÚNG cho các câu làm sai (màu CAM, mảnh, bán kính lớn hơn)
    # để người chấm thấy ngay "bài làm ở đâu – đáp án ở đâu".
    ORANGE_B = (0, 140, 255)
    for (part, qq), info_ in decided.items():
        st_, exp_, got_ = detail.get((part, qq), (None, None, None))
        if st_ not in ("wrong", "blank", "multi") or not exp_:
            continue
        cell = info_["cells"].get(exp_)
        if cell is None:
            continue
        _s, kcx, kcy, kr = cell
        cv2.circle(img, (int(kcx), int(kcy)), int(max(10, kr * 1.55)), ORANGE_B,
                   max(1, int(kr * 0.28)), cv2.LINE_AA)

    # ---- vòng chữ số SBD / Mã đề ----
    if ids and ids.get("positions"):
        BLUE_B = (255, 120, 0)
        for part, lst in ids["positions"].items():
            # Mã đề suy từ đáp án (lưới không đọc được) -> vẽ XANH DƯƠNG để
            # phân biệt với ô học sinh tô thật (vòng xanh lá)
            col_id = BLUE_B if (part.upper() == "MD" and md_inferred) else GREEN_B
            for (col, dig, cx, cy, r) in lst:
                lw = max(2, int(r * 0.40))
                cv2.circle(img, (int(cx), int(cy)), int(max(8, r * 1.25)), col_id,
                           lw, cv2.LINE_AA)

    # ---- khối điểm đỏ ----
    total = s1 + s2
    lines = []
    if show_sbd and sbd:
        lines.append(f"SBD: {sbd}")
    if made and show_sbd:
        lines.append(f"Ma de: {made}{(' ' + md_note) if md_note else ''}")
    lines += [
        f"P1: {_p1_line(stats)} = {s1:.2f}",
        f"P2: {_p2_line(detail)} = {s2:.2f}",
        f"P3: {p3_text} = 0.00",
        f"Tong: {total:.2f}",
    ]
    # Chữ đỏ đặt NGAY DƯỚI đường kẻ đứt (dải "Chú ý: Thí sinh cần đọc kỹ…"),
    # cỡ chữ = 1/2 bản trước để không đè lên phần đầu phiếu.
    fs = max(0.40, 0.71 * Ww / 943.0)
    th = max(1, int(round(fs * 1.35)))
    gap = int(round(fs * 34))
    x = int(Ww * 0.010)
    rule_y = None
    if tpl_path:
        frac = template_rule_frac(tpl_path)
        if frac:
            rule_y = int(round(frac * Hh))
    if rule_y is None:
        rule_y = _find_dashed_rule(img)
    if rule_y is not None:
        y = int(min(Hh * 0.60, rule_y + gap * 0.9))
    else:
        y = int(Hh * 0.045)
    # ---- RIÊNG MẪU 12.png: đặt từng khối đỏ ĐÚNG CHỖ trên bài quét, căn theo
    # vị trí các tiêu đề PHẦN I / PHẦN II / PHẦN III (đo trên ảnh kết quả chuẩn
    # của mẫu này, tính theo tỉ lệ chiều rộng/cao ảnh):
    #   hàng PHẦN I  : "P1: …" (16,3 %W) · "Tong: …" (39,7 %W) · "Ma de: …" (60,3 %W)
    #   hàng PHẦN II : "P2: …" (19,6 %W)
    #   hàng PHẦN III: "P3: …" (19,6 %W)
    if paper == "12":
        p1_line = next((ln for ln in lines if ln.startswith("P1:")), None)
        p2_line = next((ln for ln in lines if ln.startswith("P2:")), None)
        p3_line = next((ln for ln in lines if ln.startswith("P3:")), None)
        tong_line = next((ln for ln in lines if ln.startswith("Tong:")), None)
        md_line = next((ln for ln in lines if ln.startswith("Ma de:")), None)
        sbd_line = next((ln for ln in lines if ln.startswith("SBD:")), None)
        # Ưu tiên DÒ HÀNG TIÊU ĐỀ trên chính ảnh (chính xác, không đè bubble);
        # nếu không dò được thì dùng hệ số đo sẵn.
        rows_sec = _find_section_rows(img)
        _dy = int(round(fs * 8))
        y_sec = {}
        for key_, frac_ in _SEC_FIXED_12.items():
            want = frac_ * Hh
            near = [r for r in rows_sec if abs(r - want) <= 70]
            # lấy cụm GẦN vị trí đo sẵn nhất (tránh cụm nhiễu sát bên dưới,
            # vd hàng 449 ngay dưới tiêu đề PHẦN I)
            y_sec[key_] = int(round(min(near, key=lambda r: abs(r - want)))) \
                if near else int(round(want)) + _dy
        y_sec1, y_sec2, y_sec3 = y_sec["P1"], y_sec["P2"], y_sec["P3"]
        for ln, xf, yf in (
            (p1_line, 0.163, y_sec1),
            (tong_line, 0.397, y_sec1),
            (md_line, 0.603, y_sec1),
            (sbd_line, 0.800, y_sec1),
            (p2_line, 0.196, y_sec2),
            (p3_line, 0.196, y_sec3),
        ):
            if not ln:
                continue
            cv2.putText(img, ln, (int(Ww * xf), yf), cv2.FONT_HERSHEY_DUPLEX,
                        fs, RED_B, th, cv2.LINE_AA)
    else:
        for ln in lines:
            cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_DUPLEX, fs, RED_B, th,
                        cv2.LINE_AA)
            y += gap
    # chú thích màu (nhỏ, góc dưới-trái) để ảnh tự giải thích khi gửi cho người khác
    legend = "Xanh = dung | Do = sai | Cam = dap an dung | Vang = trong/to nhieu"
    fl = max(0.45, 0.52 * Ww / 943.0)
    cv2.putText(img, legend, (int(Ww * 0.010), int(Hh * 0.988)),
                cv2.FONT_HERSHEY_SIMPLEX, fl, (30, 30, 200),
                max(1, int(fl * 1.4)), cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path
