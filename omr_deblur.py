# -*- coding: utf-8 -*-
"""OMR DEBLUR — bộ sàng khử mờ CÓ ĐO LƯỜNG cho ảnh mờ cực đoan (2026-11-06)
==========================================================================
Dành cho ảnh chụp mờ out-of-focus (Laplacian var < EXTREME_THR=40 SAU
CLAHE+unsharp — vd ảnh 9693_tinhoc.png: raw 6.0, enhance cũ chỉ 302.9 nhưng
normalize vẫn fail ring 3% vì marker matching lock nhầm).

NGUYÊN TẮC "HÌNH HỌC QUYẾT ĐỊNH": mỗi ứng viên khử mờ được chấm bằng
DETECTABILITY — số đặc điểm pipeline THỰC SỰ dò được trên chính ảnh đó:
  • ô đen đặc (omr_prep.detect_markers — fill > 0.72, không lỗ)  ×3
  • ứng viên vuông dải mốc 18-110 px (omr_sheetgrid._fiducial_squares) ×2
  • ring vòng tròn in (omr_sheetgrid._ring_centers, accept_filled) /10
Marker/vuông khó làm giả bằng nhiễu (ràng buộc fill + tỉ lệ cạnh) nên
không thể "gian lận điểm" bằng cách khử mờ quá gắt sinh nhiễu; ring có trần.

ỨNG VIÊN (đều là kỹ thuật kinh điển CPU, không model SR nặng 19-84 s/tile):
  • Wiener deconvolution (FFT): PSF Gaussian σ ∈ {0.8,1.2,1.6,2.2} và PSF
    ĐĨA r ∈ {2,3,4} (out-of-focus thật sự là đĩa, không phải Gaussian) —
    khử mờ trên ẢNH GỐC (CLAHE phi tuyến làm sai mô hình blur).
  • Richardson-Lucy: tham số hiện tại (σ1.2 i15) + σ1.6 i30.
  • DocShadow ONNX (xoá bóng/khôi phục nền giấy — data/models, chuẩn 0-1,
    768 px, output hợp lệ [0.05..0.82] đã đo 2026-11-06).
Mỗi ứng viên qua CLAHE+unsharp như đường ống cũ rồi mới chấm.

KILL-SWITCH: env OMR_DEBLUR=classic → omr_enhance dùng đúng hành vi cũ.
"""
import cv2
import numpy as np
import os
import sys
import hashlib

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "models")
_DS_INPUT = "image"
_DS_LONG_SIDE = 768          # chạy model ở cỡ này rồi phóng về gốc
_DS_TIMEOUT_OK = True        # tắt cứng DocShadow nếu từng lỗi nghiêm trọng

_DS_SESS = None
_DS_KEY = None


def _psf_gaussian(sigma):
    """Kernel Gaussian 2D chuẩn hoá tổng = 1 (mô hình mờ ống kính)."""
    k = int(max(3, round(sigma * 3))) | 1          # lẻ, đủ rộng ~3σ
    g = cv2.getGaussianKernel(k, sigma)
    psf = (g @ g.T).astype(np.float32)
    return psf / psf.sum()


def _psf_disk(radius):
    """Kernel ĐĨA đặc (mô hình out-of-focus thực sự)."""
    r = int(radius)
    k = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.float32((xx ** 2 + yy ** 2) <= r * r + 1e-9)
    return d / d.sum()


def wiener_deconv(bgr, psf, K=0.01):
    """Khử mờ Wiener theo từng kênh trong miền tần số.

    F = H* / (|H|² + K) · G  — K = tỉ số nhiễu/tín hiệu. Pad REFLECT quanh mép
    (2 × bán kính PSF) để hạn chế ringing biên, cộng padding đủ để psf không
    wrap. Trả uint8 BGR."""
    kh, kw = psf.shape
    H, W = bgr.shape[:2]
    ph, pw = kh // 2 + 2, kw // 2 + 2
    padded = cv2.copyMakeBorder(bgr, ph, ph, pw, pw, cv2.BORDER_REFLECT)
    Hp, Wp = padded.shape[:2]
    P = np.zeros((Hp, Wp), np.float32)
    P[:kh, :kw] = psf
    P = np.roll(np.roll(P, -(kh // 2), axis=0), -(kw // 2), axis=1)
    HH = np.fft.rfft2(P)
    HHc = np.conj(HH)
    denom = (HH * HHc).real + K
    out = np.empty((H, W, 3), np.float32)
    for c in range(3):
        ch = padded[:, :, c].astype(np.float32) / 255.0
        G = np.fft.rfft2(ch)
        F = (HHc / denom) * G
        rec = np.fft.irfft2(F, s=(Hp, Wp))[ph:ph + H, pw:pw + W]
        out[:, :, c] = rec
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _rl_candidate(bgr, sigma, iters):
    """Richardson-Lucy qua omr_enhance.deblur_rl (xám → BGR)."""
    import omr_enhance as E
    return E.deblur_rl(bgr, sigma=sigma, iters=iters)


# ---------- DocShadow (xoá bóng / khôi phục nền giấy) ----------

def _docshadow_session():
    global _DS_SESS, _DS_KEY
    path = os.path.join(MODELS_DIR, "docshadow_sd7k.onnx")
    if not os.path.isfile(path):
        return None
    key = (path, round(os.path.getmtime(path), 3))
    if _DS_SESS is None or _DS_KEY != key:
        import onnxruntime as ort
        _DS_SESS = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        _DS_KEY = key
    return _DS_SESS


_DS_CACHE = {}             # md5 ảnh -> kết quả DocShadow (9 s/lần chạy CPU)


def _docshadow_candidate(bgr):
    """Chạy DocShadow ở 768 px rồi phóng về cỡ gốc. Ảnh lớn hơn _DS_LONG_SIDE
    chỉ bị thu khi NẠP model; kết quả nội suy INTER_CUBIC về (H, W) gốc.
    Cache theo md5 nội dung — bộ sàng gọi nhiều lần (combo)."""
    global _DS_TIMEOUT_OK
    if not _DS_TIMEOUT_OK:
        return None
    try:
        ck = hashlib.md5(np.ascontiguousarray(bgr).tobytes()).hexdigest()
    except Exception:                                        # noqa
        ck = None
    if ck and ck in _DS_CACHE:
        return _DS_CACHE[ck]
    try:
        sess = _docshadow_session()
        if sess is None:
            return None
        h, w = bgr.shape[:2]
        sc = min(1.0, _DS_LONG_SIDE / float(max(h, w)))
        im = cv2.resize(bgr, (int(w * sc), int(h * sc)),
                        interpolation=cv2.INTER_AREA) if sc < 1.0 else bgr
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        y = sess.run(None, {_DS_INPUT: rgb.transpose(2, 0, 1)[None]})[0][0]
        out = np.clip(y.transpose(1, 2, 0), 0.0, 1.0)
        out = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        if ck:
            if len(_DS_CACHE) > 4:
                _DS_CACHE.pop(next(iter(_DS_CACHE)))
            _DS_CACHE[ck] = out
        return out
    except Exception as e:                                   # noqa
        print(f"[deblur] DocShadow bỏ qua: {e!r:.80}", file=sys.stderr)
        _DS_TIMEOUT_OK = False
        return None


# ---------- ĐIỂM ĐO ----------

def _align_probe(bgr, tpl_path):
    """Thử GHÉP MỐC thật với mẫu: trả số ô mẫu có ô vuông ảnh tại vị trí
    chiếu (sau homography 4 cặp tốt nhất) — đo trực tiếp KHẢ NĂNG CĂN của
    ảnh, thứ mà số đếm đặc điểm không phân biệt được (ảnh DS đếm ngang
    baseline nhưng rot lock ĐÚNG, corner 4/4). Dùng 2 dải kích thước quanh
    cỡ mốc kỳ vọng (9 × approx) cho rẻ."""
    import omr_prep as P
    import omr_sheetgrid as S
    try:
        tpl = S._imread_cached(tpl_path)
        if tpl is None:
            return 0.0
        tsq = S._fiducial_squares(tpl_path, side=(6, 16))
        if len(tsq) < 4:
            return 0.0
        approx = bgr.shape[1] / float(tpl.shape[1])
        exp = 9.0 * approx
        best = 0.0
        T = np.float32([[m[0], m[1]] for m in tsq])
        for lo in (int(exp) - 8, int(exp), int(exp) + 8):
            hi = int(lo * 2.2)
            psq = S._fiducial_squares(bgr, side=(max(6, lo), hi))
            if len(psq) < 4:
                continue
            mt = S._match_fiducials(tsq, psq, approx)
            if mt is None:
                continue
            src_m, dst_m = mt
            H4 = cv2.getPerspectiveTransform(src_m, dst_m)
            proj = cv2.perspectiveTransform(T.reshape(-1, 1, 2), H4).reshape(-1, 2)
            Pp = np.float32([[m[0], m[1]] for m in psq])
            med = float(np.median([p[2] for p in psq]))
            tol = max(8.0, 0.40 * med)
            dd = np.linalg.norm(Pp[:, None, :] - proj[None, :, :], axis=-1)
            best = max(best, float((dd.min(axis=0) <= tol).sum()))
        return best
    except Exception:                                        # noqa
        return 0.0


def detectability(bgr, tpl_path=None):
    """Số đặc điểm pipeline dò được: markers ×3 + vuông dải mốc ×2 +
    ring (trần 800) /10 + KHẢ NĂNG CĂN ×4 (nếu có mẫu) — xem docstring
    module. Thành phần align là đứa phân biệt ứng viên "đếm ngang nhau"
    nhưng chỉ 1 đứa lock đúng hướng (đo trên ảnh 9693: DS rot=0 corner 4/4
    trong khi clahe+unsharp rot=90 corner 1/4)."""
    import omr_prep as P
    import omr_sheetgrid as S
    n_mk = n_sq = n_rg = 0
    try:
        n_mk = len(P.detect_markers(bgr))
    except Exception:                                        # noqa
        pass
    try:
        n_sq = len(S._fiducial_squares(bgr, side=(18, 110)))
    except Exception:                                        # noqa
        pass
    try:
        n_rg = len(S._ring_centers(bgr, accept_filled=True))
    except Exception:                                        # noqa
        pass
    n_al = _align_probe(bgr, tpl_path) if tpl_path else 0.0
    return 3.0 * n_mk + 2.0 * n_sq + min(n_rg, 800) / 10.0 + 4.0 * n_al


# ---------- BỘ SÀNG ----------

_SUITE_CACHE = {}          # md5 nội dung ảnh -> (ảnh_chọn, tên, điểm)


def deblur_suite(bgr, tpl_path=None, log=True):
    """Chạy các ứng viên khử mờ, trả (ảnh_tốt_nhất, tên, điểm).

    Đầu vào là ẢNH GỐC (trước CLAHE — deconvolution cần mô hình blur tuyến
    tính; CLAHE phi tuyến làm sai PSF). Baseline 'clahe+unsharp' là ứng viên
    đầu tiên nên kết quả KHÔNG BAO GIỜ kém hơn đường cũ. tpl_path (tuỳ chọn)
    bật thành phần ALIGNABILITY của điểm đo — đứa phân biệt ứng viên "đếm
    ngang nhau" nhưng chỉ một đứa lock đúng hướng. Dừng sớm khi một ứng viên
    đạt ≥ 2× baseline (tiết kiệm cho ảnh chạm mốc nhanh).
    Cache theo md5 nội dung: wrapper normalize gọi enhance lại ở pass sau
    (init_H) — tái dùng kết quả, khỏi tính lại ~20 s."""
    import omr_enhance as E
    try:
        ck = hashlib.md5(np.ascontiguousarray(bgr).tobytes()).hexdigest()
    except Exception:                                        # noqa
        ck = None
    if ck and ck in _SUITE_CACHE:
        return _SUITE_CACHE[ck]

    base = E.enhance_blurry(bgr)
    base_sc = detectability(base, tpl_path)   # baseline đã enhance, chấm trực tiếp
    best_sc, best_name, best_img = base_sc, "clahe+unsharp", base
    stop = max(300.0, 2.0 * base_sc)
    cands = []
    # Wiener Gaussian trước (rẻ nhất, đúng mô hình mờ ống kính)
    for sigma in (0.8, 1.2, 1.6, 2.2):
        try:
            w = wiener_deconv(bgr, _psf_gaussian(sigma))
        except Exception:                                    # noqa
            continue
        cands.append((f"Wiener-G σ{sigma}", w))
    # RL: tham số cũ + bản mạnh hơn
    for name, args in (("RL σ1.2 i15 (cũ)", (1.2, 15)),
                       ("RL σ1.6 i30", (1.6, 30))):
        try:
            r = _rl_candidate(bgr, *args)
        except Exception:                                    # noqa
            continue
        cands.append((name, r))
    # Wiener đĩa (out-of-focus thật)
    for r in (2, 3, 4):
        try:
            w = wiener_deconv(bgr, _psf_disk(r))
        except Exception:                                    # noqa
            continue
        cands.append((f"Wiener-Đĩa r{r}", w))
    # DocShadow (xoá bóng) + COMBO DS→deconv (nền sạch -> deconv hiệu quả hơn;
    # đo trên ảnh 9693: DS giúp rot lock đúng, DS→RL cho corner 4/4)
    ds = _docshadow_candidate(bgr)
    if ds is not None:
        cands.append(("DocShadow", ds))
        for name, args in (("DS→RL σ1.2 i15", (1.2, 15)),
                           ("DS→RL σ1.6 i30", (1.6, 30))):
            try:
                r = _rl_candidate(ds, *args)
            except Exception:                                # noqa
                continue
            cands.append((name, r))
        try:
            cands.append(("DS→Wiener-G σ1.2",
                          wiener_deconv(ds, _psf_gaussian(1.2))))
        except Exception:                                    # noqa
            pass

    for name, cand in cands:
        try:
            # hậu xử lý thống nhất như đường ống cũ (CLAHE+unsharp) — RL/Wiener
            # sinh ảnh "phẳng" cần unsharp cho ring nổi, DocShadow cần ngưỡng
            # tương phản về đúng quy chuẩn pipeline
            cand_e = E.enhance_blurry(cand)
            sc = detectability(cand_e, tpl_path)
        except Exception:                                    # noqa
            continue
        if log:
            print(f"[deblur]   {name:<22} điểm={sc:.0f}", file=sys.stderr)
        if sc > best_sc:
            best_sc, best_name, best_img = sc, name, cand_e
        if best_sc >= stop:
            break
    if log:
        print(f"[deblur] chọn {best_name} (điểm {best_sc:.0f}, "
              f"baseline {base_sc:.0f})", file=sys.stderr)
    out = (best_img, best_name, best_sc)
    if ck and len(_SUITE_CACHE) > 8:
        _SUITE_CACHE.pop(next(iter(_SUITE_CACHE)))
    if ck:
        _SUITE_CACHE[ck] = out
    return out
