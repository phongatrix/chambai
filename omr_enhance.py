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


def enhance_if_blurry(bgr, thr=BLUR_THR):
    """Tự phát hiện ảnh mờ rồi tăng nét; ảnh nét trả NGUYÊN BỘ (không chạm)."""
    try:
        if blur_score(bgr) >= thr:
            return bgr, False
    except Exception:                                    # noqa
        return bgr, False
    return enhance_blurry(bgr), True
