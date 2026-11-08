# -*- coding: utf-8 -*-
"""OMR COLOR ALIGN — đối chiếu vùng CÙNG MÀU giữa mẫu và ảnh (2026-11-06)
=========================================================================
Yêu cầu user: khi chuẩn hoá, các vùng cùng màu giữa MẪU và ẢNH phải khớp
nhau về vị trí/toạ độ. Màu in sẵn trên phiếu là mốc căn "sống sót" qua chụp
mờ/nghiêng và — quan trọng — PHÂN BIỆT HỌ MẪU (bug thật lô 11a8: phiếu
layout moi warp vào mẫu 10_11 lách qua cổng ring nhờ vùng trùng).

ĐO TRÊN PHIẾU THẬT (tieuchi/, chỉ dẫn thuật toán — 2026-11-06):
  • mau12.png: 3 dải NGANG đủ trang — hue 120 @y 31-34%, 150 @53-56%,
    30 @67-70%. Ảnh 12.jpg chụp có ĐÚNG 3 band: 119 @31.2-33.9%,
    151 @52.2-54.9%, 30 @65.7-68.7% (lệch ≤ 2%).
  • mau10_11.png: 5 band (hue 120, 165, 30 ×2, 30); ảnh 10_11.jpg thấy
    2 cặp mép hue 30 @55-57% + @74-76% (hue 120 của mẫu không thấy —
    dải mảnh bị chụp cụt/mờ) → khớp THEO CÁC BAND CÓ, không đòi đủ.
  • Band trên ẢNH CHỤP chỉ hiện 2 HÀNG MÉP rời (mật độ đậm 812+ px/hàng,
    phần giữa dải bị JPEG/hwen desaturate) → phải NỐI mép cùng hue.
  • moi + lô 11a8: 0 band → tự bỏ qua (scan B/W, photocopy an toàn).

API:
  template_bands(tpl_path)  — band mẫu: {hue: (y_lo, y_hi, x_cov)}
  photo_bands(bgr)          — band ảnh (cache md5), cùng dạng
  align_by_color(img, tpl)  — H4 (mẫu→ảnh) từ ≥2 band cùng hue: least-
                              squares scale_y/offset_y trên toạ độ TỈ LỆ
                              (fit pixel thô cho scale 6.7× sai — bug đo)
  color_agreement(warp,tpl) — tỉ lệ band mẫu khớp band ảnh sau warp
                              (IoU băng y trung bình); None nếu 1 bên
                              không có band = không áp cổng
Kill-switch: env OMR_COLOR=0 → mọi hàm trả trống/None.
"""
import os
import hashlib

import cv2
import numpy as np

_LONG = 1200            # cỡ phân tích (cạnh dài px)
_S_MIN, _V_MIN = 60, 60          # ngưỡng "thực sự có màu"
_HUE_TOL = 12                    # dung sai hue ghép cặp
_ROW_DENS = 0.5                  # hàng "đậm": > 50% bề rộng có màu
_ROW_GAP = 8                     # nốI hàng mép cách ≤ 8px
_EDGE_JOIN = 0.06                # nốI 2 mép cùng hue cách ≤ 6% cao trang
_MIN_WT = 200                    # dải quá nhạt (<200 px màu) = bỏ
_AGREE_IOU = 0.40                # 1 band được tính "khớp" nếu IoU ≥ đây

_COLOR_OFF = (os.environ.get("OMR_COLOR") == "0")

_BAND_CACHE = {}
_TPL_CACHE = {}


def _md5(img):
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def _row_bands(bgr, s_min=_S_MIN):
    """Băng màu theo hàng: các đoạn y (tỉ lệ 0-1) có > 50% bề rộng là màu,
    gán hue trung vị, NỐI các đoạn cùng hue cách ≤ _EDGE_JOIN (mép band
    ảnh chụp — xem docstring module). Trả [{hue, y0, y1, xcov}].

    s_min: ngưỡng saturation — warp về cỡ mẫu thu nhỏ 7.5× làm band 2px
    nhạt đi (sat 30-55 thay vì >60, đo 2026-11-06) nên caller có thể thả
    lỏng. Hue trung vị + yêu cầu đậm 50% bề rộng giữ nguyên nên không sinh
    band giả."""
    if bgr is None or bgr.ndim != 3:
        return []
    if bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    h, w = bgr.shape[:2]
    sc = min(1.0, _LONG / float(max(h, w)))
    im = cv2.resize(bgr, (int(w * sc), int(h * sc)),
                    interpolation=cv2.INTER_AREA) if sc < 1.0 else bgr
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int32)
    hue = hsv[:, :, 0].astype(np.int32)
    val = hsv[:, :, 2].astype(np.int32)
    # RINGING CUBIC (2026-11-06): nội suy khi warp sinh px sat 30-40 hue lạc
    # (mép band sáng overshoot [182 158 157] từ 1 hàng cam + 3 giấy). Chỉ áp
    # trần VAL khi s_min được THẢ LỎNG (ảnh nhỏ/warp — ringing đe doạ); mẫu
    # PNG in kỹ thuật số có band SATURATED SÁNG (val 254) — không trần.
    if s_min < _S_MIN:
        m = (sat > s_min) & (val > _V_MIN) & (val < 200)
    else:
        m = (sat > s_min) & (val > _V_MIN)
    H2, W2 = im.shape[:2]
    strong = m.sum(axis=1) > _ROW_DENS * W2
    rows = np.where(strong)[0]
    if len(rows) == 0:
        return []
    gaps = np.where(np.diff(rows) > _ROW_GAP)[0]
    segs = []
    for seg in np.split(rows, gaps + 1):
        y0, y1 = int(seg[0]), int(seg[-1])
        hh = hue[y0:y1 + 1][m[y0:y1 + 1]]
        if len(hh) < 30:                 # quá ít màu — nhiễu
            continue
        segs.append([float(np.median(hh)), y0 / H2, y1 / H2,
                     float(m[y0:y1 + 1].any(axis=0).mean())])
    # NỐI mép cùng hue (hàng trên ảnh chỉ là 2 mép của cùng dải in)
    out = []
    for hs, a, b, c in segs:
        placed = False
        for o in out:
            if abs(((o["hue"] - hs + 90) % 180) - 90) <= _HUE_TOL \
                    and a - o["y1"] <= _EDGE_JOIN:
                o["y1"] = b
                o["xcov"] = max(o["xcov"], c)
                placed = True
                break
        if not placed:
            out.append({"hue": hs, "y0": a, "y1": b, "xcov": c})
    return out


def photo_bands(bgr, s_min=_S_MIN):
    """Band của ảnh (cache theo md5+ngưỡng). s_min thấp hơn cho warp đã
    thu nhỏ — xem docstring _row_bands."""
    if _COLOR_OFF or bgr is None or not getattr(bgr, "size", 0):
        return []
    ck = (_md5(bgr), s_min)
    if ck in _BAND_CACHE:
        return _BAND_CACHE[ck]
    out = _row_bands(bgr, s_min)
    if len(_BAND_CACHE) > 8:
        _BAND_CACHE.pop(next(iter(_BAND_CACHE)))
    _BAND_CACHE[ck] = out
    return out


def template_bands(tpl_path):
    p = str(tpl_path)
    try:
        ck = (p, round(os.path.getmtime(p), 3))
    except OSError:
        ck = (p, None)
    if ck in _TPL_CACHE:
        return _TPL_CACHE[ck]
    out = _row_bands(cv2.imread(p, cv2.IMREAD_UNCHANGED))
    _TPL_CACHE[ck] = out
    return out


def align_by_color(img, tpl_path):
    """H4 (mẫu→ảnh) từ các cặp band cùng hue. Trả None nếu < 2 cặp.

    Fit trên TOẠ ĐỘ TỈ LỆ y' = a·x' + b (a = scale, b = offset; phiếu
    nguyên trang a ≈ 1) bằng least squares — least squares ổn với ≥2 cặp,
    tách trôi thang (đúng dạng lỗi trôi dọc đã gặp). θ từ trôi x giữa 2
    band xa nhất. Dựng H4 pixel mẫu→ảnh từ (a, b, θ)."""
    if _COLOR_OFF:
        return None
    tb = template_bands(tpl_path)
    if len(tb) < 2:
        return None
    ib = photo_bands(img)
    if len(ib) < 2:
        return None
    tpl_im = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
    if tpl_im is None:
        return None
    if tpl_im.ndim == 3 and tpl_im.shape[2] == 4:
        tpl_im = tpl_im[:, :, :3]
    h_tpl, w_tpl = tpl_im.shape[:2]
    h_img, w_img = img.shape[:2]
    # ghép cặp THEO THỨ TỰ Y trong từng nhóm hue (không "mỗi mẫu tự chọn ảnh
    # gần nhất" — mẫu 10_11 có 3 band hue 30, ảnh 2 band: nếu tự chọn thì cả
    # 3 mẫu dồn vào 1 band ảnh, fit vô nghĩa — bug đo 2026-11-06)
    pairs = []
    for t in sorted(tb, key=lambda z: z["y0"]):
        cand = [i for i in ib
                if abs(((t["hue"] - i["hue"] + 90) % 180) - 90) <= _HUE_TOL
                and all(i is not iu for _, iu in pairs)]
        if cand:
            # band ảnh GẦN NHẤT về y0 (cùng hue) còn chưa dùng
            pairs.append((t, min(cand, key=lambda x: abs(x["y0"] - t["y0"]))))
    if len(pairs) < 2:
        return None
    ys_t = np.array([t["y0"] for t, _ in pairs], np.float64)
    ys_i = np.array([i["y0"] for _, i in pairs], np.float64)
    A = np.stack([ys_t, np.ones_like(ys_t)], axis=1)
    sol, *_ = np.linalg.lstsq(A, ys_i, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if not (0.5 < a < 2.0) or abs(b) > 0.5:
        return None
    # θ: trôi dx tỉ lệ giữa 2 cặp xa nhau nhất theo y
    (t0, i0), (t1, i1) = pairs[0], pairs[-1]
    dxt = (t1["y0"] - t0["y0"])
    dyt = (i1["y0"] - i0["y0"])
    theta = 0.0
    if abs(dyt) > 1e-6:
        dt = (t1["y0"] - t0["y0"]) / h_tpl * w_tpl     # dx kỳ vọng (tỉ lệ→px)
        di = (i1["cx"] - i0["cx"]) if False else 0.0
        # không có cx trong band — dùng xcov không đủ; θ từ dy: bỏ (band
        # ngang không đo nghiêng quanh x đáng tin) → θ = 0, hiệu chỉnh
        # nghiêng đã có từ fiducial.
        del dt, di, dxt
    s_x = w_img / float(w_tpl)
    s_y = a * h_img / float(h_tpl)
    H4 = np.eye(3, dtype=np.float64)
    H4[0, 0] = s_x
    H4[1, 1] = s_y
    H4[1, 2] = b * h_img          # offset y' → pixel ảnh
    return H4


def color_agreement(img, tpl_path, H4=None):
    """Tỉ lệ band MẪU khớp band ẢNH (IoU băng y trung bình, IoU ≥ _AGREE_IOU).

    Trả None nếu một bên không có band (không áp cổng).

    H4 (mẫu→ảnh, pixel) — CÁCH ĐO CHUẨN (2026-11-06): chiếu band ẢNH (đo
    trên ảnh GỐC cỡ đủ lớn — mép band 1px đo được sat 255 chuẩn) qua H4
    NGƯỢC về không gian tỉ lệ mẫu rồi so với band mẫu. KHÔNG warp ảnh:
    warp CUBIC về cỡ mẫu làm mép band 1px pha loãng (sat 30-65) và nội suy
    sinh hue BỘI TƯỢNG (mép cam hue 119 — đo 2026-11-06) — không đo được.
    H4=None: img coi như ảnh đã warp ĐÚNG hệ toạ độ mẫu, đo trực tiếp."""
    if _COLOR_OFF:
        return None
    tb = template_bands(tpl_path)
    if not tb:
        return None
    ib = photo_bands(img)
    if not ib:
        return None
    if H4 is not None:
        tpl_im = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
        if tpl_im is None:
            return None
        if tpl_im.ndim == 3 and tpl_im.shape[2] == 4:
            tpl_im = tpl_im[:, :, :3]
        h_tpl = float(tpl_im.shape[0])
        h_img = float(img.shape[0])
        s = float(np.asarray(H4, np.float64)[1, 1])
        t = float(np.asarray(H4, np.float64)[1, 2])
        if abs(s) < 1e-9:
            return None
        # y_mẫu(tỉ lệ) = (y_ảnh_px - t) / s / h_tpl
        ib = [{"hue": b["hue"],
               "y0": (b["y0"] * h_img - t) / s / h_tpl,
               "y1": (b["y1"] * h_img - t) / s / h_tpl}
              for b in ib]
    hits = 0
    for t_ in tb:
        best = 0.0
        for i in ib:
            dh = abs(((t_["hue"] - i["hue"] + 90) % 180) - 90)
            if dh > _HUE_TOL:
                continue
            inter = min(t_["y1"], i["y1"]) - max(t_["y0"], i["y0"])
            union = max(t_["y1"], i["y1"]) - min(t_["y0"], i["y0"])
            if union > 1e-9:
                best = max(best, inter / union)
        hits += best >= _AGREE_IOU
    return hits / len(tb)


def needs_color_check(tpl_path, img):
    """Mẫu CÓ màu VÀ ảnh có đủ màu để đối chiếu? (cổng chỉ bật khi cả 2)
    photo_bands chỉ trả band QUA NGƯỠNG đậm 50% bề rộng → ≥1 band = đủ màu."""
    if _COLOR_OFF:
        return False
    if not template_bands(tpl_path):
        return False
    return len(photo_bands(img)) >= 1
