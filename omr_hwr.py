# -*- coding: utf-8 -*-
"""NHẬN DẠNG CHỮ VIẾT TAY — Số báo danh (SBD) + STT trên phiếu chấm.

Kiến trúc 2 tầng (engine tuỳ chọn):
  1. GEMINI VISION (chính) — cắt dải ĐẦU TRANG phiếu (gồm ô SBD và STT viết
     tay), gửi ảnh cho Gemini (REST, API key của giáo viên) với yêu cầu trả
     JSON CHỈ GỒM CHỮ SỐ. Không cần template — model tự tìm các ô số.
  2. CNN NỘI BỘ (dự phòng) — omr_ocr.read_written_ids (model digit_cnn.keras
     tự huấn luyện, chỉ đọc được SBD/MĐ theo ô in trên template).

Cả hai đều qua BỘ KIỂM SOÁT: chuỗi chỉ gồm chữ số, độ dài hợp lý
(SBD 4-10 số, STT 1-4 số) — sai thì trả None, KHÔNG BAO GIỜ đoán bừa.
"""
import base64
import json
import re
import urllib.request

GEMINI_MODEL = "gemini-3.8-flash"   # 2.0-flash đã bị Google ngừng (2026-11)
GEMINI_FALLBACKS = ["gemini-3.7-flash", "gemini-2.5-flash"]  # dự phòng khi quá tải
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")

PROMPT = (
    "Đây là ảnh ĐẦU PHIẾU thi trắc nghiệm (photocopy, có thể mờ/nghiêng). "
    "Bên trong có các Ô VUÔNG in sẵn để thí sinh VIẾT TAY chữ số:\n"
    "- Ô 'SỐ BÁO DANH' (SBD): chuỗi 4-10 chữ số.\n"
    "- Ô 'STT' (số thứ tự, nếu có): 1-4 chữ số.\n"
    "NHIỆM VỤ: đọc CHỮ VIẾT TAY trong từng ô (KHÔNG đọc số in sẵn, KHÔNG đọc "
    "lưới bubble).\n"
    "Chỉ trả về JSON đúng format, không thêm chữ:\n"
    '{"SBD": "<các chữ số hoặc null>", "STT": "<các chữ số hoặc null>"}\n'
    "Nếu ô trống/không đọc được → null. TUYỆT ĐỐI không suy đoán."
)


# ---------- 1. CẮT DẢI ĐẦU TRANG ----------

def header_crop(warped, frac=0.16, upscale=2.0):
    """Dải ĐẦU phiếu (trên cùng, đủ rộng chứa SBD + STT), phóng to cho rõ."""
    h, w = warped.shape[:2]
    crop = warped[0:max(40, int(h * frac)), :]
    if upscale != 1.0:
        crop = cv2_resize(crop, upscale)
    return encode_jpeg(crop)


def cv2_resize(img, scale):
    import cv2
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)


def encode_jpeg(img, quality=92):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ---------- 2. ENGINE GEMINI ----------

def gemini_read(api_key, jpg_bytes, model=GEMINI_MODEL, timeout=30):
    """Gửi ảnh cho Gemini, trả {"SBD": str|None, "STT": str|None}.

    Lỗi mạng/API → raise Exception (caller tự quyết định fallback)."""
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(jpg_bytes).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.0,
                             "maxOutputTokens": 200},
    }
    import time as _time
    last_err = None
    for model_ in [model] + list(GEMINI_FALLBACKS):
        url = GEMINI_URL.format(model=model_, key=api_key)
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            txt = ""
            for cand in (data.get("candidates") or []):
                for part in ((cand.get("content") or {}).get("parts") or []):
                    txt += part.get("text") or ""
            return parse_gemini_answer(txt)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (503, 429, 404):
                _time.sleep(3)       # quá tải/quota/model ngừng — thử kế tiếp
                continue
            raise
    raise last_err


def parse_gemini_answer(txt):
    """Rút JSON khỏi câu trả lời (chống markdown ```json ...```)."""
    m = re.search(r"\{[^{}]*\}", txt, re.S)
    out = {"SBD": None, "STT": None}
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except Exception:                                        # noqa
        return out
    for k, lo, hi in (("SBD", 4, 10), ("STT", 1, 4)):
        v = str(d.get(k) or "").strip()
        v = re.sub(r"\D", "", v)             # chỉ giữ chữ số
        if lo <= len(v) <= hi:
            out[k] = v
    return out


# ---------- 3. ENGINE CNN NỘI BỘ (dự phòng) ----------

def cnn_read(warped, tpl_path, paper):
    """Đọc SBD viết tay bằng CNN nội bộ (chỉ có SBD/MĐ theo ô template)."""
    try:
        import omr_ocr as O
        r = O.read_written_ids(warped, tpl_path, paper, force=True)
        sbd = (r or {}).get("SBD")
        if sbd and 4 <= len(str(sbd)) <= 10:
            return {"SBD": str(sbd), "STT": None, "engine": "cnn"}
    except Exception:                                        # noqa
        pass
    return {"SBD": None, "STT": None, "engine": "none"}


# ---------- 4. GỌI CHUNG ----------

def read_fields(warped, tpl_path, paper, api_key=None, read_stt=True,
                model=None):
    """Đọc SBD + STT viết tay. Gemini ưu tiên (đủ SBD + STT), CNN dự phòng.

    model: model người dùng chọn ở tab Đáp án & QR (None = GEMINI_MODEL mặc
    định, vẫn có chuỗi fallback).
    Trả {"SBD": str|None, "STT": str|None, "engine": "gemini"|"cnn"|"none"}."""
    if api_key:
        try:
            jpg = header_crop(warped)
            if jpg:
                r = gemini_read(api_key, jpg, model=model)
                r["engine"] = "gemini"
                # SBD Gemini đọc được nhưng CNN không — tin Gemini khi đủ dài
                return r
        except Exception:                                    # noqa
            pass
    return cnn_read(warped, tpl_path, paper)
