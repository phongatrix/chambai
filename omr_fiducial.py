# -*- coding: utf-8 -*-
"""
FIDUCIAL-BASED OMR (omr_fiducial.py)

Kiến trúc theo yêu cầu: chấm đen vuông/chữ nhật in trên phiếu là mốc định vị.
1. template_from_blank(img): từ ẢNH MẪU (phiếu trắng) -> phát hiện mốc đen + bubble
   -> mô hình mẫu: {markers: [(x,y)...], bubbles: [(x,y,w)...], khung}
2. align_page(photo, template): phát hiện mốc trong ảnh chụp -> khớp với mẫu
   (RANSAC homography) -> ma trận biến đổi mẫu -> ảnh.
3. read_fills(photo, template, H): lấy mẫu mực tại từng bubble (vị trí mẫu
   chiếu sang ảnh) -> ô nào tô đậm.
4. read_answers(template, filled_mask): gom bubble thành lưới hàng/cột từng phần
   -> đáp án A/B/C/D, Đ/S, số.
"""
import cv2
import numpy as np


def detect_markers(img, min_area=25, max_area=None):
    """Mốc đen vuông/chữ nhật trên MASK MỰC chuẩn hóa sáng (kháng shadow).

    Dùng RETR_LIST (KHÔNG phải EXTERNAL) vì phiếu có thể có viền trang bao ngoài
    — RETR_EXTERNAL sẽ gộp toàn bộ ô đen bên trong vào 1 contour viền.
    Chỉ nhận contour ĐẶC (không có contour con = không có lỗ).
    """
    if isinstance(img, (str, bytes)) or hasattr(img, "__fspath__"):
        img = cv2.imread(str(img))
        if img is None:
            raise FileNotFoundError(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bg = cv2.blur(gray.astype(np.float32), (51, 51))
    norm = gray.astype(np.float32) / (bg + 1e-6)
    dark = (norm < 0.80).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, hz = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    amax = max_area if max_area else int(0.004 * W * H)
    # tập contour có con (có lỗ) -> loại
    has_child = set()
    if hz is not None:
        for row in hz[0]:
            if row[3] != -1:
                has_child.add(row[3])
    ms = []
    for i, c in enumerate(cs):
        if i in has_child:
            continue
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (min_area <= a <= amax):
            continue
        fill = a / (w * h)
        asp = w / h
        if fill > 0.72 and 0.55 <= asp <= 1.8 and w >= 4 and h >= 4:
            ms.append((x + w / 2, y + h / 2, w, h))
    # khử trùng lặp (contour lồng nhau của cùng 1 ô đen)
    out = []
    for m in sorted(ms, key=lambda z: -z[2] * z[3]):
        if all(abs(m[0] - o[0]) > 2 or abs(m[1] - o[1]) > 2 for o in out):
            out.append(m)
    return out


def detect_bubbles(img, min_r=3, max_r=40):
    """Bubble tròn (vòng): contour có lỗ, hình tròn."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs, hz = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hz = hz[0] if hz is not None else []
    out = []
    for i, c in enumerate(cs):
        if i < len(hz) and hz[i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if w < 2 * min_r or w > 2 * max_r or h > 2 * max_r:
            continue
        circ = 4 * np.pi * a / (cv2.arcLength(c, True) ** 2 + 1e-9)
        if circ > 0.6 and 0.7 <= w / h <= 1.4:
            out.append((x + w / 2, y + h / 2, w))
    return out


def template_from_blank(path):
    """Dựng mô hình mẫu từ ảnh phiếu trắng."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    markers = detect_markers(img)
    bubbles = detect_bubbles(img)
    return {"path": str(path), "size": [w, h],
            "markers": markers, "bubbles": bubbles}


def _marker_matrix(ms):
    return np.array([[m[0], m[1]] for m in ms], np.float32)


def _sheet_bbox(img):
    """BBox tờ giấy: vùng sáng lớn nhất (fallback: toàn khung)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, br = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    br = cv2.morphologyEx(br, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cs, _ = cv2.findContours(br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    if cs:
        c = max(cs, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.2 * W * H:
            return x, y, w, h
    return 0, 0, W, H


def _margin_markers(ms, x0, x1, frac=0.10):
    """Mốc biên trái/phải THEO BBOX TỜ GIẤY, sắp theo y."""
    w = x1 - x0
    left = sorted([m for m in ms if m[0] <= x0 + w * frac], key=lambda m: m[1])
    right = sorted([m for m in ms if m[0] >= x1 - w * frac], key=lambda m: m[1])
    return left, right


def align_page(photo_path, template, min_inl=6):
    """Khớp ảnh chụp với mẫu bằng MỐC BIÊN: 5 trái + 5 phải (theo dải 8% bbox),
    ghép theo thứ tự dọc -> homography RANSAC. Trả (H, photo, markers_photo, markers_tpl)."""
    ph = cv2.imread(str(photo_path))
    if ph is None:
        raise FileNotFoundError(photo_path)
    pm = detect_markers(ph, min_area=200)
    tm = template["markers"]
    tw, th = template["size"]
    if len(pm) < 6 or len(tm) < 6:
        return None, ph, pm, tm

    def margins(ms, x0, x1, n=5, frac=0.09):
        w = x1 - x0
        left = sorted([m for m in ms if m[0] <= x0 + w * frac], key=lambda m: m[1])
        right = sorted([m for m in ms if m[0] >= x1 - w * frac], key=lambda m: m[1])
        # giữ n mốc phân bố đều nhất theo y (tránh mốc phụ)
        def pick(lst):
            if len(lst) <= n:
                return lst
            idx = np.linspace(0, len(lst) - 1, n).round().astype(int)
            return [lst[i] for i in idx]
        return pick(left), pick(right)

    tl, tr = margins(tm, 0, tw)
    xs = [m[0] for m in pm]
    pl, pr = margins(pm, min(xs), max(xs))
    if min(len(tl), len(pl)) < 2 or min(len(tr), len(pr)) < 2:
        return None, ph, pm, tm
    src_pts, dst_pts = [], []
    for i in range(min(len(tl), len(pl))):
        src_pts.append(tl[i][:2])
        dst_pts.append(pl[i][:2])
    for i in range(min(len(tr), len(pr))):
        src_pts.append(tr[i][:2])
        dst_pts.append(pr[i][:2])
    if len(src_pts) < 4:
        return None, ph, pm, tm
    H, mask = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts),
                                 cv2.RANSAC, ransacReprojThreshold=40.0)
    if H is None:
        return None, ph, pm, tm
    inl = int(mask.sum()) if mask is not None else 0
    if inl < min_inl:
        return None, ph, pm, tm
    return H, ph, pm, tm


def read_fills(photo, template, H, snap=True):
    """Đọc mực tại từng bubble mẫu (chiếu qua homography) + SNAP cục bộ về tâm
    vòng tròn thật gần nhất trong ảnh (khớp lưới mẫu -> lưới ảnh)."""
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    bg = cv2.blur(gray.astype(np.float32), (41, 41))
    norm = gray.astype(np.float32) / (bg + 1e-6)
    dark = (norm < 0.80).astype(np.uint8) * 255
    cs, hz = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hz = hz[0] if hz is not None else []
    ring_centers = []
    for i, c in enumerate(cs):
        if i < len(hz) and hz[i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (30 <= a <= 4000) or w < 7 or h < 7:
            continue
        circ = 4 * np.pi * a / (cv2.arcLength(c, True) ** 2 + 1e-9)
        if circ > 0.6 and 0.65 <= w / h <= 1.5:
            ring_centers.append((x + w / 2, y + h / 2, max(w, h)))
    ring_centers = np.array(ring_centers, np.float32) if ring_centers else np.zeros((0, 3), np.float32)
    bs = np.float32([[b[0], b[1]] for b in template["bubbles"]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(bs, H).reshape(-1, 2)
    res = []
    for i, ((bx, by, bw), (px, py)) in enumerate(zip(template["bubbles"], proj)):
        cx, cy = px, py
        if snap and len(ring_centers):
            d = np.linalg.norm(ring_centers[:, :2] - np.float32([px, py]), axis=1)
            j = int(np.argmin(d))
            if d[j] <= max(30.0, ring_centers[j][2] * 1.6):
                cx, cy = ring_centers[j][:2]
        r = 8
        x0, x1 = int(cx - r), int(cx + r + 1)
        y0, y1 = int(cy - r), int(cy + r + 1)
        if x0 < 0 or y0 < 0 or x1 > norm.shape[1] or y1 > norm.shape[0]:
            res.append((i, (bx, by), cx, cy, 0.0))
            continue
        roi = norm[y0:y1, x0:x1]
        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        disk = ((xx - r) ** 2 + (yy - r) ** 2) <= (r * 0.8) ** 2
        vals = roi[disk]
        score = float((vals < 0.62).mean()) if vals.size else 0.0
        res.append((i, (bx, by), cx, cy, score))
    return res


def main():
    import sys
    blank = sys.argv[1] if len(sys.argv) > 1 else "/home/dell/Pictures/Screenshots/Screenshot From 2026-11-02 11-54-00.png"
    photo = sys.argv[2] if len(sys.argv) > 2 else "/home/dell/Documents/tulieuchambai/Test/IMG_20261030_042800.jpg"
    tpl = template_from_blank(blank)
    print(f"Mẫu: {len(tpl['markers'])} mốc đen, {len(tpl['bubbles'])} bubble, khung {tpl['size']}")
    H, ph, pm, tm = align_page(photo, tpl)
    print(f"Ảnh chụp: {len(pm)} mốc phát hiện; mẫu {len(tm)} mốc")
    if H is None:
        print("KHÔNG khớp được")
        return
    fills = read_fills(ph, tpl, H)
    strong = sorted([f for f in fills if f[4] > 0.45], key=lambda z: -z[4])
    print(f"Homography OK; bubble score>0.45: {len(strong)}/{len(fills)}")
    for f in strong[:15]:
        print(f"  bubble#{f[0]} mẫu({f[1][0]:.0f},{f[1][1]:.0f}) -> ảnh({f[2]:.0f},{f[3]:.0f}) score={f[4]:.2f}")


if __name__ == "__main__":
    main()
