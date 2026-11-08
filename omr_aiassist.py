# -*- coding: utf-8 -*-
"""AI ASSIST — cầu nối GEMINI hỗ trợ thuật toán OMR (2026-11-06)
================================================================
KẾT HỢP THUẬT TOÁN + AI theo nguyên tắc "AI khởi đầu, HÌNH HỌC quyết định":
  1. RESCUE ĐỊNH VỊ (`rescue_homography`) — ảnh mờ/lệch mà marker matching +
     SIFT đều yếu: Gemini trả toạ độ CÁC Ô ĐEN ĐỊNH VỊ nhìn thấy trên ảnh
     (bbox chuẩn hoá 0-1000) -> đổi về px -> `_match_fiducials` (RANSAC cặp
     điểm của omr_sheetgrid — chống outlier, không tin mù 1 box nào) -> H
     khởi đầu -> normalize_to_template(init_H=H) chạy TIẾP warp + ICP +
     RING GATE như ảnh thường. Ring gate không đạt vẫn bị loại — AI không
     thể "ép" một phiếu sai mẫu vào kết quả.
  2. CHẤM TRỰC TIẾP (`read_answers` + `grade_ai`) — ảnh KHÔNG thể chuẩn hoá
     (mờ nặng/mép cụt): Gemini đọc SBD + Mã đề + đáp án PHẦN I (ABCD) +
     PHẦN II (Đúng/Sai) ngay trên ảnh -> so với key -> điểm, ĐÁNH DẤU RÕ
     "AI trực tiếp" để giáo viên rà soát.
  3. SBD/STT viết tay đã có riêng ở omr_hwr.py (không lặp ở đây).

Key: env GEMINI_API_KEY -> data/gemini_key.txt. Model chain giảm dần; các
lỗi 429/503/404 tự chuyển model kế (như omr_hwr). KHÔNG gọi API khi không
cần: mọi kết quả AI được cache theo (tên file, kích thước) trong session.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash",
                 "gemini-3.6-flash", "gemini-3.5-flash"]
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")


# ---------- 1. hạ tầng ----------

def load_key(root=None):
    """Gemini API key: env GEMINI_API_KEY trước, rồi data/gemini_key.txt."""
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    p = Path(root or Path(__file__).parent) / "data" / "gemini_key.txt"
    try:
        return p.read_text().strip()
    except Exception:                                            # noqa
        return ""


def _parse_json(txt):
    """Rút JSON khỏi câu trả lời (markdown fence, dấu phẩy thừa…)."""
    t = re.sub(r"```[a-z]*", "", str(txt or "")).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("không thấy khối JSON trong câu trả lời")
    j = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    return json.loads(j)


def gemini_text(prompt, jpg_bytes, api_key, timeout=60, max_tokens=3000,
                model_first=None):
    """Gửi 1 ảnh + prompt, trả TEXT thô (model chain + retry như gemini_json).

    model_first: model người dùng CHỌN ở tab Đáp án & QR — đặt ĐẦU chuỗi,
    các model GEMINI_MODELS còn lại làm dự phòng."""
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(jpg_bytes).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
    }
    last = None
    chain = ([model_first] if model_first else []) + \
        [m for m in GEMINI_MODELS if m != model_first]
    for model in chain:
        req = urllib.request.Request(
            GEMINI_URL.format(model=model, key=api_key),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            txt = ""
            for cand in (data.get("candidates") or []):
                for part in ((cand.get("content") or {}).get("parts") or []):
                    txt += part.get("text") or ""
            if not txt.strip():
                last = RuntimeError(f"{model}: trả lời rỗng")
                continue
            return txt
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 503, 404):
                time.sleep(2)        # quá tải/quota/model ngừng — thử kế tiếp
                continue
            raise
        except Exception as e:                                   # noqa
            last = e
    raise last


def gemini_json(prompt, jpg_bytes, api_key, timeout=60, max_tokens=3000,
                model_first=None):
    """Gửi 1 ảnh + prompt, trả JSON đã parse. Lỗi hết model -> raise.

    model_first: model người dùng chọn — đặt ĐẦU chuỗi dự phòng (bug thật
    2026-11-06: read_answers truyền model_first nhưng hàm không nhận ->
    TypeError, AI-direct chết âm thầm trong webapp)."""
    return _parse_json(gemini_text(prompt, jpg_bytes, api_key, timeout,
                                   max_tokens, model_first=model_first))


def list_models(api_key, timeout=15):
    """Danh sách model Gemini DÙNG ĐƯỢC hiện tại của key (GET /v1beta/models
    — rẻ, không tốn quota generate). Lọc: chỉ model chat/vision gemini-flash|
    pro, loại TTS/embedding/image/robotics/computer-use/nano/live."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models"
           f"?key={api_key}&pageSize=100")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    out = []
    for m in data.get("models", []):
        name = str(m.get("name", "")).replace("models/", "")
        if "gemini" not in name:
            continue
        if any(t in name for t in ("tts", "embedding", "image", "robotics",
                                   "computer-use", "nano", "live", "omni",
                                   "transcribe", "preview-customtools")):
            continue
        out.append(name)
    return sorted(set(out), reverse=True)


def validate_key(api_key, timeout=15):
    """Kiểm tra key bằng GET /v1beta/models — RẺ (không tốn quota generate),
    trả (True, "OK") hoặc (False, lý do). Dùng cho modal nhập key khi khởi
    động để dán sai không được lưu."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models"
           f"?key={api_key}&pageSize=1")
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            json.loads(resp.read().decode())
        return True, "OK"
    except urllib.error.HTTPError as e:
        if e.code in (400, 403):
            return False, "key không hợp lệ / bị từ chối"
        return False, f"HTTP {e.code}"
    except Exception as e:                                        # noqa
        return False, str(e)[:100]


def encode_jpeg(img, quality=88):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ---------- 2. RESCUE ĐỊNH VỊ ----------

PROMPT_CORNERS = """Ảnh PHIẾU TRẮC NGHIỆM đã VẼ THÊM LƯỚI ĐỎ tham chiếu
(hàng 0-9 từ trên xuống, cột A-J từ trái sang, mỗi ô = 10% ảnh) — lưới chỉ
ĐỂ ĐỌC TOẠ ĐỘ.

Trên phiếu có MỘT KHUNG NỘI DUNG LỚN NHẤT (chứa PHẦN I, PHẦN II, PHẦN III)
và ở 4 GÓC của khung đó có in 4 Ô VUÔNG ĐEN ĐẶC nhỏ (ô định vị mực in, cạnh
chỉ ~1-2% chiều ngang phiếu). KHÔNG phải vòng tròn, KHÔNG phải chữ.

Nhiệm vụ: tìm ĐÚNG 4 ô vuông đen ở 4 GÓC khung ngoài đó (góc trên-trái,
trên-phải, dưới-phải, dưới-trái). Dùng lưới đỏ để đọc toạ độ chính xác
(thang 0-1000 cả hai chiều, góc trên trái ảnh = 0,0).

Trả về ĐÚNG một khối JSON, không thêm chữ nào khác:
{"goc": [{"cx": 41, "cy": 106}, {"cx": 370, "cy": 106},
         {"cx": 370, "cy": 587}, {"cx": 41, "cy": 587}]}

(ví dụ — thay bằng toạ độ THẬT đọc từ ảnh; thứ tự bất kỳ)"""


def _draw_reference_grid(img):
    """Vẽ lưới đỏ 10% + nhãn hàng/cột lên BẢN SAO — Gemini đọc toạ độ qua
    lưới chính xác hơn hẳn nhìn ảnh trần (kinh nghiệm phổ biến với VLM)."""
    import cv2
    import numpy as np
    vis = img.copy()
    h, w = vis.shape[:2]
    for i in range(1, 10):
        y = int(h * i / 10)
        x = int(w * i / 10)
        cv2.line(vis, (0, y), (w, y), (0, 0, 255), 2)
        cv2.line(vis, (x, 0), (x, h), (0, 0, 255), 2)
    for i in range(10):
        yc = int(h * (i + 0.5) / 10)
        xc = int(w * (i + 0.5) / 10)
        cv2.putText(vis, str(i), (8, yc + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.8, h / 900), (0, 0, 255), 2)
        cv2.putText(vis, chr(65 + i), (xc - 14, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.8, h / 900), (0, 0, 255), 2)
    return vis


def detect_corners(img, api_key, model=None):
    """Gemini trả toạ độ 4 ô góc khung ngoài (hệ 0-1000).

    Parse REGEX từng object (model hay trả JSON cắt cụt giữa mảng)."""
    txt = gemini_text(PROMPT_CORNERS, encode_jpeg(_draw_reference_grid(img)),
                      api_key, timeout=45, max_tokens=1200, model_first=model)
    pat = re.compile(r'\{\s*["\']cx["\']\s*:\s*([\d.]+)\s*,\s*["\']cy["\']'
                     r'\s*:\s*([\d.]+)\s*\}')
    pts = []
    for m in pat.finditer(txt):
        cx, cy = float(m.group(1)), float(m.group(2))
        if 0 <= cx <= 1000 and 0 <= cy <= 1000:
            pts.append((cx, cy))
    return pts


def _order_quad(pts):
    """Sắp 4 điểm thành TL, TR, BR, BL theo tổng/hiệu toạ độ."""
    import numpy as np
    if len(pts) != 4:
        return None
    P = np.float32(pts)
    idx = [int(np.argmin(P[:, 0] + P[:, 1])), int(np.argmax(P[:, 0] - P[:, 1])),
           int(np.argmax(P[:, 0] + P[:, 1])), int(np.argmin(P[:, 0] - P[:, 1]))]
    if len(set(idx)) < 4:
        return None
    return P[idx]


def rescue_homography(img, tpl_path, api_key, model=None):
    """H (mẫu -> ảnh) từ 4 ô góc do GEMINI chỉ ra, có KIỂM CHỨNG HÌNH HỌC.

    Trả (H, n_pts) hoặc (None, n_pts). H chỉ là ĐIỂM KHỞI ĐẦU —
    normalize_to_template sẽ ICP + corner/ring gate lại như ảnh thường;
    Gemini KHÔNG THỂ ép phiếu sai mẫu qua cổng ring."""
    import cv2
    import numpy as np
    from omr_prep import template_fiducials
    quad_t, _ms = template_fiducials(tpl_path)
    if len(quad_t) != 4:
        return None, 0
    pts = detect_corners(img, api_key, model=model)
    if len(pts) != 4:
        return None, len(pts)
    q = _order_quad(pts)
    if q is None:
        return None, len(pts)
    H_img, W_img = img.shape[:2]
    qp = np.float32([[x / 1000.0 * W_img, y / 1000.0 * H_img] for x, y in q])
    T = np.float32(quad_t)
    # ---- kiểm chứng hình học TRƯỚC khi tin: tỉ lệ cạnh + diện tích ----
    dt = np.linalg.norm(T[1] - T[0])          # cạnh trên mẫu
    dp = np.linalg.norm(qp[1] - qp[0])
    ht = np.linalg.norm(T[3] - T[0])
    hp = np.linalg.norm(qp[3] - qp[0])
    ar_t, ar_p = dt / max(1e-6, ht), dp / max(1e-6, hp)
    sc_w, sc_h = dp / max(1e-6, dt), hp / max(1e-6, ht)
    if not (0.55 <= ar_p / max(1e-6, ar_t) <= 1.8):
        return None, len(pts)                  # méo tỉ lệ — Gemini sai
    if not (0.25 <= min(sc_w, sc_h) and max(sc_w, sc_h) <= 12.0):
        return None, len(pts)                  # thang vô lý
    if abs(sc_w - sc_h) > 0.35 * max(sc_w, sc_h):
        return None, len(pts)                  # co giãn 2 chiều lệch xa
    area = 0.5 * abs(np.cross(qp[1] - qp[0], qp[3] - qp[0])) * 2
    if area < 0.10 * W_img * H_img:
        return None, len(pts)
    H = cv2.getPerspectiveTransform(T, qp)
    if H is None:
        return None, len(pts)
    return H, len(pts)


# ---------- 3. CHẤM TRỰC TIẾP ----------

def _prompt_answers(so_cau_p1, so_cau_p2):
    return f"""Ảnh là PHIẾU TRẮC NGHIỆM đã tô bằng bút/chì. Hãy đọc CHÍNH XÁC:
- "sbd": số báo danh VIẾT TAY ở dòng "3. Số báo danh" (chỉ chữ số; trống thì null)
- "ma_de": mã đề trong ô "Mã đề" (chỉ chữ số; trống thì null)
- "p1": PHẦN I — {so_cau_p1} câu trắc nghiệm A/B/C/D, mỗi câu ô đã tô đậm.
  Trả theo dạng {{"1": "A", "2": "C", ...}}; câu trống hoặc tô nhiều -> "?".
- "p2": PHẦN II — {so_cau_p2} câu, mỗi câu 4 ý a) b) c) d), mỗi ý chọn
  Đúng hoặc Sai. Trả theo dạng {{"1a": "Đ", "1b": "S", ...}}; ý trống/tô
  nhiều -> "?".
Chỉ dựa vào VỎ TÔ THẬT nhìn thấy được, không suy đoán. Trả về ĐÚNG một
khối JSON:"""


def read_answers(img, api_key, so_cau_p1=20, so_cau_p2=4, model=None):
    """Gemini đọc SBD/mã đề/đáp án trực tiếp trên ảnh. Trả dict chuẩn hoá.

    max_tokens 4000 (2026-11-06): model 3.x tốn thinking token — 2500 bị
    ăn hết trước khi ra JSON ("không thấy khối JSON" trên ảnh 9693)."""
    d = gemini_json(_prompt_answers(so_cau_p1, so_cau_p2),
                    encode_jpeg(img), api_key, max_tokens=4000,
                    model_first=model)
    p1 = {}
    for k, v in (d.get("p1") or {}).items():
        m = re.fullmatch(r"\d+", str(k).strip())
        if not m:
            continue
        ch = str(v).strip().upper()
        p1[int(k)] = ch if ch in ("A", "B", "C", "D") else "?"
    p2 = {}
    for k, v in (d.get("p2") or {}).items():
        m = re.fullmatch(r"(\d+)\s*([a-dA-D])", str(k).strip())
        if not m:
            continue
        ch = str(v).strip().upper()
        if ch in ("Đ", "D", "T", "TRUE", "DUNG"):
            ch = "Đ"
        elif ch in ("S", "F", "FALSE", "SAI"):
            ch = "S"
        else:
            ch = "?"
        p2[(int(m.group(1)), m.group(2).lower())] = ch
    sbd = re.sub(r"\D", "", str(d.get("sbd") or "")) or None
    made = re.sub(r"\D", "", str(d.get("ma_de") or "")) or None
    return {"sbd": sbd, "ma_de": made, "p1": p1, "p2": p2,
            "raw": d}


def grade_ai(ans, key, part1_score=0.25, part2_score=0.25):
    """Chấm đáp án đọc bằng AI so với key (cùng tỉ lệ điểm với S.grade:
    P1 0.25/câu, P2 0.25/ý mặc định). Trả dict hoặc None nếu không khớp đề."""
    des = key.get("các đề") or {}
    if not des:
        return None
    made = ans.get("ma_de")
    de = des.get(made) if made else None
    note_de = "AI đọc"
    if de is None:
        # không đọc được mã đề: thử TẤT CẢ đề, chọn đề khớp nhiều nhất
        best = None
        for de_name, de_key in des.items():
            ok = 0
            for cq, ch in ans.get("p1", {}).items():
                if ch != "?" and de_key.get("p1", {}).get(cq) == ch:
                    ok += 1
            if best is None or ok > best[0]:
                best = (ok, de_name, de_key)
        if best is None or best[0] < 3:
            return None
        de, note_de = best[1], "AI suy mã đề"
        made = de
        de = best[2]
    p1_key = de.get("p1") or {}
    p2_key = de.get("p2") or {}
    p1_ok = sum(1 for cq, ch in ans["p1"].items()
                if ch != "?" and p1_key.get(cq) == ch)
    p1_n = sum(1 for ch in p1_key.values())
    p2_ok = 0
    for kk, ch in ans["p2"].items():
        kv = p2_key.get(kk)
        if ch != "?" and kv and ch == str(kv).strip().upper():
            p2_ok += 1
    p2_n = len(p2_key)
    diem = p1_ok * part1_score + p2_ok * part2_score
    return {"made": str(made), "note_de": note_de,
            "p1_ok": p1_ok, "p1_n": p1_n,
            "p2_ok": p2_ok, "p2_n": p2_n,
            "diem": round(diem, 2)}
