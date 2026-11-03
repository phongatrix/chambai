# -*- coding: utf-8 -*-
"""
OMR CNN — Mạng nơ-ron nhận dạng CHỮ SỐ VIẾT TAY trong ô SBD/Mã đề
==================================================================
Thay thế Tesseract (không ổn định trên bản photocopy 18×46 px).

Pipeline 3 bước:
  1. THU THẬP DỮ LIỆU TỰ ĐỘNG (không gán nhãn tay):
     bubble SBD/MĐ đọc ĐÚNG và ĐỦ (không có '?') cho biết học sinh viết số
     nào trong các ô phía trên -> cắt ô viết tay, NHÃN = chữ số đó.
  2. TRAIN CNN nhỏ (Keras):
     input 48×48×1 (nét chữ, nền trắng) -> Conv/MaxPool ×2 -> Dense -> 10 lớp.
     Tăng cường dữ liệu: xoay ±10°, dời ±10 %, tỉ lệ 0,9–1,1, độ sáng/ng prototypes.
  3. SUY LUẬN: `DigitCNN.predict(crop)` -> (digit, confidence).

Mô hình lưu ở data/models/digit_cnn.keras; nếu có file này, omr_ocr sẽ dùng
CNN thay Tesseract tự động.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMG = 48          # kích thước input chuẩn của CNN
MODEL_DIR_NAME = "data/models"
MODEL_NAME = "digit_cnn.keras"
META_NAME = "digit_cnn_meta.json"


# ---------- 1. cắt nét chữ số từ ô viết tay ----------

def crop_writing(warped, box, scale=2.0, margin=0.10):
    """Cắt vùng viết tay trong 1 ô (đã bỏ viền ô bằng margin)."""
    x0 = int(round(box["x0"] * scale))
    x1 = int(round(box["x1"] * scale))
    y0 = int(round(box["y0"] * scale))
    y1 = int(round(box["y1"] * scale))
    H, W = warped.shape[:2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    mx = int((x1 - x0) * margin)
    my = int((y1 - y0) * margin)
    roi = warped[y0 + my:y1 - my, x0 + mx:x1 - mx]
    if roi.size == 0:
        return None
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # chuẩn hoá sáng theo phân vị 90 (khử khác biệt giữa các trang)
    g = np.clip(g / (float(np.percentile(g, 90)) + 1e-6) * 255.0, 0, 255)
    return g.astype(np.uint8)


def extract_stroke(roi_gray, min_ink=0.008):
    """Tách NÉT CHỮ: nhị phân thích ứng -> xoá viền (flood-fill 4 góc) ->
    thành phần lớn nhất -> bbox có đệm. Trả (ảnh nét đen/trắng, ink_ratio)."""
    big = cv2.resize(roi_gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (5, 5), 0)
    bw = cv2.adaptiveThreshold(big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 12)
    ink = float((bw > 0).mean())
    if ink < min_ink:
        return None, ink
    # xoá viền ô: mọi nét nối với mép là đường kẻ khung
    ff = bw.copy()
    hh, ww = ff.shape
    mask = np.zeros((hh + 2, ww + 2), np.uint8)
    for pt in ((0, 0), (ww - 1, 0), (0, hh - 1), (ww - 1, hh - 1)):
        cv2.floodFill(ff, mask, pt, 0)
    cs, _ = cv2.findContours(ff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, ink
    c = max(cs, key=cv2.contourArea)
    bx, by, bw_, bh_ = cv2.boundingRect(c)
    if bw_ < 0.15 * ww or bh_ < 0.15 * hh:
        return None, ink
    # ghép cả các thành phần chồng lên bbox của nét chính (vd dấu phẩy chữ số)
    sel = np.zeros_like(ff)
    for cc in cs:
        x, y, w2, h2 = cv2.boundingRect(cc)
        if (x < bx + bw_ and x + w2 > bx and y < by + bh_ and y + h2 > by):
            sel[y:y + h2, x:x + w2] = np.maximum(sel[y:y + h2, x:x + w2],
                                                 ff[y:y + h2, x:x + w2])
    if not sel.any():
        return None, ink
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    digit = 255 - sel[y0:y1 + 1, x0:x1 + 1]
    return digit, ink


def to_input(digit_img, img=IMG):
    """Chuyển nét chữ về kích thước input CNN (giữ tỉ lệ, căn giữa)."""
    hh, ww = digit_img.shape[:2]
    side = max(hh, ww)
    canvas = np.full((side, side), 255, np.uint8)
    y0 = (side - hh) // 2
    x0 = (side - ww) // 2
    canvas[y0:y0 + hh, x0:x0 + ww] = digit_img
    out = cv2.resize(canvas, (img, img), interpolation=cv2.INTER_AREA)
    x = out.astype(np.float32) / 255.0
    return x[..., None]          # (IMG, IMG, 1)


def augment(img48, rng=None):
    """Tăng cường 1 mẫu: xoay/dời/tỉ lệ/độ sáng (trả list mẫu mới)."""
    rng = rng or np.random.default_rng(0)
    outs = []
    h = img48.shape[0]
    for ang in (-8, -4, 0, 4, 8):
        M = cv2.getRotationMatrix2D((h / 2, h / 2), ang, 1.0)
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                M2 = M.copy()
                M2[0, 2] += dx
                M2[1, 2] += dy
                a = cv2.warpAffine(img48, M2, (h, h), flags=cv2.INTER_LINEAR,
                                   borderValue=1.0)
                gain = rng.uniform(0.85, 1.15)
                a = np.clip((a - 0.5) * gain + 0.5, 0, 1)
                outs.append(a)
    return outs


# ---------- 2. thu thập dữ liệu có nhãn ----------

def collect_from_pages(pages, tpl_path, paper, scale=2.0):
    """Duyệt các trang ĐÃ CHUẨN HOÁ, cắt ô viết tay + nhãn từ bubble.

    pages: list đường dẫn ảnh đã chuẩn hoá (bản _chuan hoặc _todam).
    Chỉ lấy nhãn từ trang bubble đọc ĐỦ và ĐỦ (không có '?').
    Trả (samples, labels, skipped_pages) — sample = ảnh 48×48 float.
    """
    import omr_ocr as O
    import omr_idscan as ID
    samples, labels, skipped = [], [], []
    H = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    for p in pages:
        img = cv2.imread(str(p))
        if img is None:
            skipped.append(str(p))
            continue
        r = ID.scan_ids(img, H, paper, str(tpl_path), debug={})
        raw = r.get("raw") or {}
        # nhãn đáng tin: chuỗi ĐỦ chữ số, không có '?'
        sbd_ok = raw.get("SBD") and "?" not in raw["SBD"]
        md_ok = raw.get("MD") and "?" not in raw["MD"]
        if not (sbd_ok or md_ok):
            skipped.append(str(p))
            continue
        boxes = O.find_digit_boxes(tpl_path, paper, scale=scale)
        n_sbd = len([b for b in boxes if b["part"] == "SBD"])
        for b in boxes:
            part = b["part"]
            if part == "SBD":
                if not sbd_ok or b["idx"] >= len(raw["SBD"]):
                    continue
                label = raw["SBD"][b["idx"]]
            else:
                if not md_ok or b["idx"] >= len(raw["MD"]):
                    continue
                label = raw["MD"][b["idx"]]
            roi = crop_writing(img, b, scale=scale)
            if roi is None:
                continue
            dig, ink = extract_stroke(roi)
            if dig is None:
                continue
            samples.append(to_input(dig))
            labels.append(int(label))
    return samples, labels, skipped


# ---------- 3. CNN ----------

def build_model(img=IMG, nclass=10):
    from tensorflow import keras
    m = keras.Sequential([
        keras.layers.Input((img, img, 1)),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(nclass, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def train_model(samples, labels, out_dir, epochs=40, val_frac=0.15):
    """Huấn luyện CNN; trả (model, lịch sử, số mẫu train/val)."""
    from tensorflow import keras
    X = np.asarray(samples, np.float32)
    y = np.asarray(labels, np.int32)
    n = len(y)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    vi, ti = idx[:n_val], idx[n_val:]
    Xv, yv = X[vi], y[vi]
    Xt, yt = X[ti], y[ti]
    # cân bằng lớp bằng tăng cường trên tập train
    gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.12, brightness_range=[0.8, 1.2], fill_mode="constant",
        cval=1.0)
    m = build_model(X.shape[1])
    hist = m.fit(gen.flow(Xt[..., None] if Xt.ndim == 3 else Xt[..., None],
                          yt, batch_size=32),
                 validation_data=(Xv[..., None], yv),
                 epochs=epochs, verbose=0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / MODEL_NAME
    m.save(mpath)
    meta = {"img": IMG, "nclass": 10, "n_train": int(len(yt)),
            "n_val": int(len(yv)), "val_acc": float(hist.history["val_accuracy"][-1])}
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return m, hist, meta


# ---------- 4. suy luận ----------

class DigitCNN:
    """Bộ nhận dạng chữ số viết tay bằng CNN đã huấn luyện."""

    def __init__(self, model_path):
        from tensorflow import keras
        self.model = keras.models.load_model(str(model_path))
        meta_path = Path(model_path).parent / META_NAME
        self.img = IMG
        if meta_path.is_file():
            try:
                self.img = int(json.loads(meta_path.read_text())["img"])
            except Exception:                                       # noqa
                pass

    def predict(self, digit_img):
        """digit_img: ảnh nét đen/trắng (bất kỳ cỡ) -> (digit|None, conf)."""
        if digit_img is None:
            return None, 0.0
        x = to_input(digit_img, self.img)[None, ...]
        p = self.model.predict(x, verbose=0)[0]
        d = int(np.argmax(p))
        return d, float(p[d])
