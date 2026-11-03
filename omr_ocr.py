# -*- coding: utf-8 -*-
"""
OMR OCR — Đọc CHỮ SỐ VIẾT TAY trong các ô "Số báo danh" / "Mã đề"
==================================================================
Phiếu 12.png có 12 ô in sẵn ở góc phải trên: 8 ô Số báo danh + 4 ô Mã đề.
Học sinh có thể TÔ bubble (đọc bằng omr_idscan) hoặc VIẾT TAY số vào ô.
Module này đọc phần viết tay để bổ trợ khi lưới bubble trống/không đọc được.

Quy trình:
  1. Dò 12 ô trên ẢNH MẪU (đường kẻ dọc trong dải ngay trên lưới ID).
  2. Trên ảnh đã chuẩn hoá: cắt từng ô (bỏ viền), phóng to, chuẩn hoá sáng,
     nhị phân Otsu.
  3. Tesseract (psm 10 = một ký tự, whitelist 0-9) -> chữ số + độ tin cậy.
  4. Ô trống (mực quá ít hoặc conf thấp) -> bỏ.
"""
from __future__ import annotations

# ⚠️ MẶC ĐỊNH TẮT: Tesseract trên ô số rất nhỏ của bản photocopy đang cho kết
# quả sai (đọc ô trống thành "7"). Bật lại chỉ sau khi hiệu chuẩn/huấn luyện.
ENABLED = False

import cv2
import numpy as np

_BOX_CACHE = {}


def _vertical_lines(gray_bin, min_ratio=0.55):
    """Vị trí các đường kẻ dọc (cột có nhiều mực)."""
    h, w = gray_bin.shape
    col = gray_bin.sum(axis=0) / 255.0
    thr = min_ratio * h
    xs = [i for i, v in enumerate(col) if v >= thr]
    # gộp các cột liền nhau
    out = []
    for x in xs:
        if out and x - out[-1][-1] <= 1:
            out[-1].append(x)
        else:
            out.append([x])
    return [float(np.mean(g)) for g in out]


def find_digit_boxes(tpl_path, paper, scale=1.0, up=1.0):
    """12 ô chữ số trên ẢNH MẪU (toạ độ mẫu): [{part, idx, x0..y1}, ...].

    `up` chỉ dùng khi mẫu quá nhỏ (12.png 461 px) — nội bộ phóng để dò đường kẻ.
    """
    ck = (str(tpl_path), paper, up)
    if ck in _BOX_CACHE:
        return _BOX_CACHE[ck]
    import omr_sheetgrid as S
    img = cv2.imread(str(tpl_path))
    if img is None:
        return []
    cells = S.build_id_cells(paper, tpl_path)
    if not cells:
        return []
    ys = [c[4] for c in cells]
    grid_top = min(ys)
    if up != 1.0:
        img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dải ngay TRÊN lưới bubble: cách đỉnh lưới 4..22 px (mẫu)
    xs_cells = [c[3] for c in cells]
    x_lo = max(0, int((min(xs_cells) - 8) * up))
    x_hi = min(g.shape[1], int((max(xs_cells) + 8) * up))
    # ô chữ số nằm trong khoảng 30..7 px phía trên đỉnh lưới (mẫu 12.png:
    # lưới bắt đầu y=77, ô ở y≈48..69)
    box_y0 = max(0.0, grid_top - 30.0)
    box_y1 = max(box_y0 + 6.0, grid_top - 7.0)
    y0 = int(box_y0 * up)
    y1 = int(box_y1 * up)
    # CHỈ lấy dải phía trên khối ID (nếu lấy cả trang sẽ bắt nhầm đường kẻ
    # của bảng bên trái -> sinh ra ô rác ở x=16..117)
    band = g[y0:y1, x_lo:x_hi]
    _, bw = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    xs = _vertical_lines(bw)
    if len(xs) < 6:
        _BOX_CACHE[ck] = []
        return []
    # tách SBD / MĐ theo TOẠ ĐỘ ĐÃ BIẾT của lưới (không dùng khe lớn nhất:
    # khe lớn nhất thường nằm ở mép trái trang -> gộp sai toàn bộ)
    md_start = min([c[3] for c in cells if c[0].upper() == "MD"], default=None)
    sb_end = max([c[3] for c in cells if c[0].upper() == "SBD"], default=None)
    if md_start is not None and sb_end is not None:
        cut_x = (sb_end + md_start) / 2.0 * up
        # xs là toạ độ TRONG dải -> cộng x_lo để so với toạ độ mẫu
        left = [v for v in xs if (v + x_lo) < cut_x]
        right = [v for v in xs if (v + x_lo) >= cut_x]
    else:
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        cut = int(np.argmax(gaps))
        left, right = xs[:cut + 1], xs[cut + 1:]
    boxes = []

    def _emit(part, lines):
        for i in range(len(lines) - 1):
            x0, x1 = (lines[i] + x_lo) / up, (lines[i + 1] + x_lo) / up
            if x1 - x0 < 4:
                continue
            boxes.append({"part": part, "idx": len([b for b in boxes if b["part"] == part]),
                          "x0": x0, "y0": box_y0, "x1": x1, "y1": box_y1})

    _emit("SBD", left)
    _emit("MD", right)
    _BOX_CACHE[ck] = boxes
    return boxes


def _ocr_one(warped, box, scale, up=8.0, margin=0.30):
    """OCR 1 ô; trả (digit|None, conf 0..1, ink 0..1).

    Bỏ viền ô (margin 30%) để không đọc nhầm đường kẻ; tách lấy ĐÚNG vùng nét
    chữ (bounding box của mực) rồi mới đưa vào Tesseract — psm 10 rất dễ "đoán
    bừa" khi ảnh có viền.
    """
    x0 = int(round(box["x0"] * scale))
    x1 = int(round(box["x1"] * scale))
    y0 = int(round(box["y0"] * scale))
    y1 = int(round(box["y1"] * scale))
    H, W = warped.shape[:2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None, 0.0, 0.0
    mx = int((x1 - x0) * margin)
    my = int((y1 - y0) * margin)
    roi = warped[y0 + my:y1 - my, x0 + mx:x1 - mx]
    if roi.size == 0:
        return None, 0.0, 0.0
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip(g / (float(np.percentile(g, 90)) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    ink = float((g < 150).mean())
    if ink < 0.02:                       # ô trống (chỉ còn nhiễu)
        return None, 0.0, ink
    big = cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (5, 5), 0)
    bw = cv2.adaptiveThreshold(big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 12)
    # XOÁ ĐƯỜNG VIỀN Ô: xả nước (flood fill) từ mép ảnh — mọi nét nối với mép
    # là đường kẻ khung, không phải chữ số
    ff = bw.copy()
    hh_, ww_ = ff.shape
    mask = np.zeros((hh_ + 2, ww_ + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 0)
    cv2.floodFill(ff, mask, (ww_ - 1, 0), 0)
    cv2.floodFill(ff, mask, (0, hh_ - 1), 0)
    cv2.floodFill(ff, mask, (ww_ - 1, hh_ - 1), 0)
    if int((ff > 0).sum()) < 0.02 * ff.size:
        return None, 0.0, ink
    cs, _ = cv2.findContours(ff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, 0.0, ink
    # giữ nét lớn nhất (chữ số), bỏ chấm nhiễu
    c = max(cs, key=cv2.contourArea)
    bx, by, bwid, bhei = cv2.boundingRect(c)
    # chữ số viết tay phải CAO và CÂN đối; nét khung/nhiễu thì dẹt hoặc quá nhỏ
    if bhei < 0.45 * ff.shape[0] or bwid < 0.20 * ff.shape[1]:
        return None, 0.0, ink
    if not (0.18 <= bwid / float(bhei) <= 1.05):
        return None, 0.0, ink
    digit_img = 255 - ff[by:by + bhei, bx:bx + bwid]
    hh, ww = digit_img.shape
    side = max(hh, ww)
    canvas = np.full((side, side), 255, np.uint8)
    canvas[(side - hh) // 2:(side - hh) // 2 + hh,
           (side - ww) // 2:(side - ww) // 2 + ww] = digit_img
    canvas = cv2.resize(canvas, (96, 96), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(canvas, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
    try:
        import pytesseract
        from pytesseract import Output
        cfg = "--psm 10 -c tessedit_char_whitelist=0123456789"
        d = pytesseract.image_to_data(canvas, config=cfg, output_type=Output.DICT)
        best, bestc = None, -1.0
        for txt, conf in zip(d.get("text", []), d.get("conf", [])):
            t = (txt or "").strip()
            if t.isdigit() and float(conf) > bestc:
                best, bestc = t, float(conf)
        if best is None:
            alt = pytesseract.image_to_string(canvas, config=cfg).strip()
            if alt.isdigit():
                best, bestc = alt, 45.0
        if best and bestc >= 65:
            return best, bestc / 100.0, ink
        return None, max(0.0, bestc / 100.0), ink
    except Exception:                                            # noqa
        return None, 0.0, ink


def read_written_ids(warped, tpl_path, paper, scale=2.0, force=False):
    """Đọc SBD / Mã đề VIẾT TAY trong các ô (THỬ NGHIỆM).

    Trả {"SBD": "06"|None, "MD": "3157"|None, "boxes": [{...}], "conf": {...}}
    Chỉ chạy khi ENABLED=True hoặc force=True — xem ghi chú đầu file.
    """
    if not (ENABLED or force):
        return {"SBD": None, "MD": None, "boxes": [], "conf": {}, "disabled": True}
    boxes = find_digit_boxes(tpl_path, paper, scale=scale)
    if not boxes:
        return {"SBD": None, "MD": None, "boxes": [], "conf": {}}
    # Ô in sẵn cũng lệch theo khối ID -> dùng ĐỘ DỜI của khối ID (mốc chữ nhật)
    # để dịch toạ độ ô cho khớp ảnh thật.
    try:
        import omr_idalign as _IA
        loc = _IA.locate_id_block(warped, tpl_path, paper, scale)
        if loc:
            dx = loc["dx"] / float(scale)
            dy = loc["dy"] / float(scale)
            boxes = [{**b, "x0": b["x0"] + dx, "x1": b["x1"] + dx,
                      "y0": b["y0"] + dy, "y1": b["y1"] + dy} for b in boxes]
    except Exception:                                            # noqa
        pass
    out = {"SBD": [], "MD": []}
    conf = {"SBD": [], "MD": []}
    kept = []
    for b in boxes:
        d, c, ink = _ocr_one(warped, b, scale)
        out[b["part"]].append(d)
        conf[b["part"]].append(round(c, 2))
        kept.append({**b, "digit": d, "conf": round(c, 2), "ink": round(ink, 3)})
    res = {"boxes": kept, "conf": conf}
    for part in ("SBD", "MD"):
        ds = out[part]
        res[part] = "".join(d for d in ds if d) if any(ds) else None
    return res
