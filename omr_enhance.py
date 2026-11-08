# -*- coding: utf-8 -*-
"""Tăng nét + tương phản cho ảnh chấm BỊ MỜ (out-focus/chụp nhòe/tương phản thấp).

Dùng thuật toán xử lý ảnh kinh điển, KHÔNG cần model:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L của
     LAB — néli tương phản CỤC BỘ:Bubble mờ/bút chì nhạt trên ảnh xám nổi lên
     mà không phóng đại chênh sáng toàn trang.
  2. Unsharp mask (ảnh − Gaussian(ảnh) × amount) — khôi phục cạnh bubble/mốc
     bị mềm do mất nét.

AUTO-TRIGGER: chỉ áp dụng khi ảnh THẬT SỰ mờ (var Laplacian < BLUR_THR).
Ảnh nét (batch chuẩn var 500-700) đi qua KHÔNG ĐỘNG — kết quả chấm giữ nguyên.
"""
import cv2
import numpy as np

BLUR_THR = 150.0        # ngưỡng var Laplacian: dưới ngưỡng này coi là mờ
                        # (ảnh nét của dự án: 500-700; ảnh mờ của người dùng: 6)


def blur_score(bgr):
    """Điểm nét = phương sai Laplacian (thấp = mờ)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def enhance_blurry(bgr, clahe_clip=2.5, unsharp_amount=1.4, radius=5):
    """CLAHE(L) + unsharp mask — trả ảnh đã tăng nét/tương phản."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), radius)
    return cv2.addWeighted(out, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)


def deblur_rl(bgr, ksize=5, sigma=1.2, iters=15):
    """KHỬ MỜ Richardson-Lucy (deconvolution) — kỹ thuật kinh điển
    (skimage.restoration.richardson_lucy, xem khảo sát trên mạng 2026-11-06:
    các model deep SR Swin2SR/Real-ESRGAN ONNX trên CPU mất 19-84 s/TILE —
    ảnh 2600 px cần ~600 tile = bất khả thi; RL mất ~4 s TOÀN ẢNH và tăng
    Laplacian 16 -> 1152 trên ảnh mờ nhất dự án).

    PSF: Gaussian ksize×ksize sigma (xấp xỉ blur chụp xa/mất nét). Trả BGR.
    Dùng cho ảnh MỜ CỰC ĐOAN (blur_score < EXTREME_THR) — ảnh thường đã đủ
    bởi CLAHE+unsharp ở enhance_blurry."""
    import cv2 as _cv
    g = _cv.cvtColor(bgr, _cv.COLOR_BGR2GRAY)
    gk = _cv.getGaussianKernel(ksize, sigma)
    psf = (gk @ gk.T).astype(np.float32)
    try:
        from skimage.restoration import richardson_lucy as _rl
        out = _rl(g.astype(np.float32) / 255.0, psf, num_iter=iters,
                  clip=False)
    except Exception:                                    # noqa
        return bgr
    out8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return _cv.cvtColor(out8, _cv.COLOR_GRAY2BGR)


EXTREME_THR = 40.0      # dưới ngưỡng này = mờ CỰC ĐOAN -> RL sau enhance


def enhance_if_blurry(bgr, thr=BLUR_THR, tpl_path=None):
    """Tự phát hiện ảnh mờ rồi tăng nét; ảnh nét trả NGUYÊN BỘ (không chạm).

    2 TẦNG (2026-11-06): blur < thr -> CLAHE+unsharp; sau đó nếu vẫn
    blur < EXTREME_THR (mờ cực đoan, CLAHE không cứu nổi) -> BỘ SÀNG khử mờ
    (omr_deblur: Wiener/RL/DocShadow chấm bằng detectability) — tpl_path nếu
    có sẽ bật thành phần ALIGNABILITY (ghép mốc thật với mẫu) để chọn được
    ứng viên CĂN ĐƯỢC, không chỉ đếm nhiều."""
    try:
        if blur_score(bgr) >= thr:
            return bgr, False
    except Exception:                                    # noqa
        return bgr, False
    out = enhance_blurry(bgr)
    try:
        # ---- ĐIỀU KIỆN TẦNG CỰC ĐOAN đo trên ẢNH GỐC (2026-11-06) ----
        # trước đây đo SAU CLAHE+unsharp: CLAHE+unsharp PHÓNG ĐẠI Laplacian
        # bằng nhiễu (9693: raw 6.0 -> sau enhance 302.9) nên tầng extreme
        # không bao giờ chạy dù đặc điểm vẫn dò không được. Raw < 40 mới là
        # "mờ cực đoan" thật sự (121112: 15.3, 9693: 6.0).
        if blur_score(bgr) < EXTREME_THR:
            # ---- BỘ SÀNG KHỬ MỜ CÓ ĐO LƯỜNG (2026-11-06, omr_deblur) ----
            # thay RL cố định bằng bộ sàng ứng viên (Wiener/RL/DocShadow)
            # chấm bằng DETECTABILITY + ALIGNABILITY — số mốc/ring pipeline
            # THỰC SỰ dò được và GHÉP ĐƯỢC với mẫu; đầu vào là ẢNH GỐC
            # (deconvolution cần mô hình blur tuyến tính, CLAHE phi tuyến
            # làm sai PSF). Baseline 'clahe+unsharp' là 1 ứng viên nên kết
            # quả không bao giờ kém đường cũ. OMR_DEBLUR=classic → cũ.
            import os as _os
            if _os.environ.get("OMR_DEBLUR") != "classic":
                try:
                    import omr_deblur as _DB
                    _img2, _name, _sc = _DB.deblur_suite(bgr, tpl_path=tpl_path)
                    return _img2, True
                except Exception as _e:                      # noqa
                    print(f"[deblur] suite lỗi, quay về RL cũ: {_e!r:.80}")
            out2 = deblur_rl(out)
            out2 = enhance_blurry(out2)      # unsharp sau RL cho ring nổi
            if blur_score(out2) > blur_score(out):
                return out2, True
    except Exception:                                    # noqa
        pass
    return out, True
