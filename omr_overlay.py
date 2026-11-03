# -*- coding: utf-8 -*-
"""
RENDER OVERLAY DAQUET (omr_overlay.py)

Vẽ kết quả chấm lên ảnh bài thi theo format daquet/:
- vòng XANH quanh ô học sinh tô ĐÚNG; vòng ĐỎ quanh ô tô SAI
- khối chữ đỏ góc trái trên: P1: đúng/trả lời = điểm; P2: ...; P3: ...; Tổng: ...
Đầu ra: ảnh JPG overlay (giống kết quả phần mềm chấm).
"""
import cv2
import numpy as np

GREEN = (0, 170, 0)
RED = (0, 0, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_overlay(orig_bgr, result, fills_detail=None, scale_factor=3.0):
    """Vẽ overlay lên ảnh gốc.

    orig_bgr: ảnh gốc (đã đúng hướng EXIF)
    result: dict kết quả grade (my, counts, fills_detail tuỳ chọn)
    fills_detail: {zone: [[cx, cy, w, label]]} tọa độ ở ảnh 3x (nếu có)
    """
    img = orig_bgr.copy()
    H, W = img.shape[:2]
    fd = fills_detail if fills_detail is not None else result.get("fills_detail")
    if fd:
        for zone, lst in fd.items():
            for (cx, cy, w, label) in lst:
                sx, sy = cx / scale_factor, cy / scale_factor
                r = max(6, int(w * 0.62 / scale_factor))
                color = GREEN if label == "green" else RED if label == "red" else None
                if color is None:
                    continue
                cv2.circle(img, (int(sx), int(sy)), r, color, 3)
    c1 = result.get("counts", {})
    ng1, nr1 = c1.get("P1", {}).get("green", 0), c1.get("P1", {}).get("red", 0)
    ng2, nr2 = c1.get("P2", {}).get("green", 0), c1.get("P2", {}).get("red", 0)
    ng3 = c1.get("P3", {}).get("green", 0)
    my = result.get("my", {})
    lines = [
        f"P1: {ng1}/{ng1 + nr1} = {my.get('P1', 0):.2f}",
        f"P2: {ng2}/{ng2 + nr2} = {my.get('P2', 0):.2f}",
        f"P3: {ng3}/{ng3} = {my.get('P3', 0):.2f}",
        f"Tong: {my.get('tong', 0):.2f}",
    ]
    md = result.get("md") or (result.get("kq") or {}).get("made")
    if md:
        lines.insert(0, f"Ma de: {md}")
    fs = max(0.45, W / 940 * 0.55)
    th = int(fs * 26)
    y = int(H * 0.035)
    for ln in lines:
        cv2.putText(img, ln, (int(W * 0.02), y), FONT, fs, RED, max(1, int(fs * 1.8)), cv2.LINE_AA)
        y += th
    return img


def _scan_datetime(orig_path):
    """Ngày/giờ quét: EXIF DateTimeOriginal -> EXIF DateTime -> mtime file."""
    from PIL import Image
    from PIL.ExifTags import TAGS
    p = str(orig_path)
    try:
        with Image.open(p) as pim:
            ex = pim.getexif()
            ifd = ex.get_ifd(0x8769) if ex else {}
            for tag_id in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
                v = ifd.get(tag_id) or ex.get(tag_id)
                if v:
                    s = str(v).strip()
                    if len(s) >= 19 and s[4] == ":":
                        return s[0:4] + s[5:7] + s[8:10], s[11:13] + s[14:16] + s[17:19]
    except Exception:
        pass
    import os
    import datetime as _dt
    st = _dt.datetime.fromtimestamp(os.path.getmtime(p))
    return st.strftime("%Y%m%d"), st.strftime("%H%M%S")


def _safe(s, default="NA"):
    s = str(s or "").strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
    return s if s else default


def render_batch(items, out_dir):
    """items: [(orig_path, result)]; xuất JPG overlay với tên:
    <SBD>_<MaDe>_<NgayQuet>_<GioQuet>.jpg (đoạn thiếu thay 'NA').
    Trả list đường dẫn."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    used = set()
    for orig_path, result in items:
        img = cv2.imread(str(orig_path))
        if img is None:
            continue
        try:
            from PIL import Image as PILImage
            with PILImage.open(str(orig_path)) as pim:
                tag = pim.getexif().get(274)
            import omr_grade_final as _g
            if tag in _g.EXIF_ROTATE:
                img = cv2.rotate(img, _g.EXIF_ROTATE[tag])
        except Exception:
            pass
        rendered = render_overlay(img, result)
        ngay, gio = _scan_datetime(orig_path)
        k = result.get("kq") or {}
        sbd = _safe(result.get("sbd") or k.get("sbd"))
        made = _safe(result.get("md") or k.get("made"))
        name = f"{sbd}_{made}_{ngay}_{gio}.jpg"
        p = Path(out) / name
        n = 1
        while str(p) in used or p.exists():
            n += 1
            p = Path(out) / f"{sbd}_{made}_{ngay}_{gio}_{n}.jpg"
        used.add(str(p))
        cv2.imwrite(str(p), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths
