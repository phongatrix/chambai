# -*- coding: utf-8 -*-
"""
ĐÁP ÁN + MÃ QR (omr_key.py)

1. Phân tích file Excel đáp án -> nhận dạng MẪU:
   - "QM": hàng đầu = 'Đề \\ Câu' + số câu 1..N + '1a'..'Xd' + cột số -> mỗi DÒNG = 1 đề
   - "list": cột Question / Answer / Score -> mỗi DÒNG = 1 câu (đề duy nhất)
2. Sinh file mẫu đáp án (mẫu QM, mẫu list) để người dùng tải về điền theo.
3. Mã QR cho từng mẫu key: nội dung "OMRKEY:<key_id>" — web dùng để chọn key.
   Đọc QR từ ảnh bằng QRCodeDetector của OpenCV (không cần thư viện ngoài).
"""
import io
from pathlib import Path

import cv2
import numpy as np
import openpyxl

QRCODE_PREFIX = "OMRKEY:"


def detect_key_format(path):
    """Trả (định dạng, key, mô_tả) hoặc (None, None, lỗi)."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return None, None, "File trống"
    hdr = rows[0]
    hdr_str = [str(h).strip().upper() if h is not None else "" for h in hdr]

    def _is_num(h):
        return isinstance(h, (int, float)) and not isinstance(h, bool)

    # mẫu list: Question/Answer/Score
    if any("QUESTION" in h for h in hdr_str) and any("ANSWER" in h for h in hdr_str):
        i_q = next(i for i, h in enumerate(hdr_str) if "QUESTION" in h)
        i_a = next(i for i, h in enumerate(hdr_str) if "ANSWER" in h)
        i_s = next((i for i, h in enumerate(hdr_str) if "SCORE" in h), None)
        key = {"đề": "list", "câu": {}}
        n = 0
        for r in rows[1:]:
            if r and r[i_q] is not None and r[i_a] is not None:
                q = str(r[i_q]).strip()
                a = str(r[i_a]).strip().upper()
                s = float(r[i_s]) if (i_s is not None and r[i_s] is not None) else 0.25
                key["câu"][q] = {"answer": a, "score": s}
                n += 1
        if n == 0:
            return None, None, "Mẫu list không có dòng câu nào"
        return "list", key, f"Mẫu list: {n} câu, đề duy nhất"

    # mẫu Smarttest (QM.transpose): hàng đầu = 'Câu\Mã đề' + mã đề; mỗi DÒNG = 1 câu
    import re as _re
    codes_hdr = [h for h in hdr[1:] if _is_num(h)]
    first_col = [r[0] for r in rows[1:] if r and r[0] is not None]
    looks_transposed = (codes_hdr and all(100 <= int(float(c)) <= 9999 for c in codes_hdr)
                        and first_col and all(_is_num(v) and float(v) <= 99
                                              for v in first_col[:6]))
    if looks_transposed:
        des = {}
        n_cau = 0
        for r in rows[1:]:
            if not r or r[0] is None:
                continue
            cq = int(float(r[0]))
            n_cau += 1
            for j, h in enumerate(hdr[1:], 1):
                if j < len(r) and r[j] is not None:
                    de = str(int(float(h)))
                    des.setdefault(de, {"p1": {}, "p2": {}, "p3": {}})
                    v = str(r[j]).strip().upper().replace("Đ", "D")
                    if _re.fullmatch(r"[DS]{4}", v):
                        for k, ch in enumerate(v):
                            des[de]["p2"][(cq, chr(97 + k))] = ch
                    elif _re.fullmatch(r"[-]?\d+([.,]\d+)?", v):
                        des[de]["p3"][cq] = float(v.replace(",", "."))
                    else:
                        des[de]["p1"][cq] = v
        if not des or n_cau == 0:
            return None, None, "Mẫu transpose không có dữ liệu"
        de0 = next(iter(des.values()))
        mô_tả = (f"Mẫu Smarttest: {len(des)} đề × {n_cau} câu ABCD "
                 f"(mã đề: {', '.join(list(des)[:6])}...)")
        return "Smarttest", {"đề": "QM", "các đề": des}, mô_tả

    # mẫu QM: hàng đầu có 'ĐỀ' + cột câu; mỗi dòng = 1 đề

    def _is_ds(h):
        import re
        return isinstance(h, str) and bool(re.match(r"^\d+[a-dA-D]$", h.strip()))

    ds_positions = [i for i, h in enumerate(hdr) if _is_ds(str(h).strip().upper() if h is not None else "")]
    if ds_positions:
        ds_start, ds_end = min(ds_positions), max(ds_positions)
        abcd_cols = [i for i in range(1, ds_start) if _is_num(hdr[i])]
        num_cols = [i for i in range(ds_end + 1, len(hdr)) if _is_num(hdr[i])]
    else:
        abcd_cols = [i for i in range(1, len(hdr)) if _is_num(hdr[i])]
        ds_cols, num_cols = [], []
    de_col = 0
    if not abcd_cols:
        return None, None, "Không tìm thấy cột câu số trong hàng đầu"
    key = {"đề": "QM", "n_abcd": len(abcd_cols), "abcd_cols": abcd_cols,
           "ds_cols": ds_positions, "num_cols": num_cols, "các đề": {}}
    for r in rows[1:]:
        if r and r[de_col] is not None:
            v = r[de_col]
            de = str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v).strip()
            p1 = {abcd_cols.index(i) + 1: str(r[i]).strip().upper()
                  for i in abcd_cols if i < len(r) and r[i] is not None}
            p2 = {}
            for j, ci in enumerate(ds_positions):
                if ci < len(r) and r[ci] is not None:
                    cq = j // 4 + 1
                    yk = chr(97 + j % 4)
                    p2[(cq, yk)] = str(r[ci]).strip().upper()
            p3 = {k + 1: float(r[ci]) for k, ci in enumerate(num_cols)
                  if ci < len(r) and r[ci] is not None}
            key["các đề"][de] = {"p1": p1, "p2": p2, "p3": p3}
    if not key["các đề"]:
        return None, None, "Mẫu QM không có dòng đề nào"
    de0 = next(iter(key["các đề"].values()))
    mô_tả = (f"Mẫu QM: {len(key['các đề'])} đề × {len(de0['p1'])} câu ABCD"
             f" + {len(de0['p2'])} ô Đ/S + {len(de0['p3'])} câu số")
    return "QM", key, mô_tả


def make_sample_qm(n_de=2, n_abcd=18, n_ds_cau=4, n_so=6, path=None):
    """Sinh file mẫu đáp án QM (hàng đầu = khung, mỗi dòng = 1 đề)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    hdr = ["Đề \\ Câu"] + [c for c in range(1, n_abcd + 1)] + \
          [f"{c}{y}" for c in range(1, n_ds_cau + 1) for y in "abcd"] + \
          [c for c in range(1, n_so + 1)]
    ws.append(hdr)
    for d in range(n_de):
        de = 101 + d
        row = [de] + ["A"] * n_abcd + ["D"] * (n_ds_cau * 4) + [0] * n_so
        ws.append(row)
    buf = io.BytesIO() if path is None else None
    if path:
        wb.save(path)
        return Path(path)
    wb.save(buf)
    buf.seek(0)
    return buf


def make_sample_list(n=18, path=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Question", "Answer", "Score"])
    for c in range(1, n + 1):
        ws.append([c, "A", 0.25])
    if path:
        wb.save(path)
        return Path(path)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def make_qr_png(payload):
    """Sinh ảnh QR (PNG bytes) cho payload (vd 'OMRKEY:abc123')."""
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def decode_qr_image(img_bgr_or_bytes):
    """Đọc mọi mã QR trong ảnh; trả list payload (chỉ nhận tiền tố OMRKEY:)."""
    if isinstance(img_bgr_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(img_bgr_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = img_bgr_or_bytes
    if img is None:
        return []
    det = cv2.QRCodeDetector()
    ok, texts, _, _ = det.detectAndDecodeMulti(img)
    found = []
    if ok:
        found = [t for t in texts if t]
    if not found:
        t, _, _ = det.detectAndDecode(img)
        if t:
            found = [t]
    return [t for t in found if t.startswith(QRCODE_PREFIX)]


if __name__ == "__main__":
    import sys
    p = make_sample_qm(path="/tmp/sample_qm.xlsx")
    fmt, key, desc = detect_key_format(p)
    print(f"Mẫu QM tự sinh: {fmt} | {desc}")
    p2 = make_sample_list(path="/tmp/sample_list.xlsx")
    fmt2, key2, desc2 = detect_key_format(p2)
    print(f"Mẫu list tự sinh: {fmt2} | {desc2}")
    png = make_qr_png(QRCODE_PREFIX + "test123")
    payload = decode_qr_image(png)
    print(f"QR sinh/đọc: {payload}")
