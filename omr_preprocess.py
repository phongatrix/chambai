# -*- coding: utf-8 -*-
"""
CHUẨN HÓA ẢNH ĐẦU VÀO (omr_preprocess.py)

Xử lý ảnh bài thi bị lệch / thiếu chuẩn trước khi chấm:
1. estimate_rotation: ước lượng góc nghiêng trang từ các đường kẻ ngang in sẵn
   (biên khung Phần I/II/III, bảng Điểm, lưới SBD/MĐ) bằng Hough trên mask mực.
   Trung vị góc -> kháng nhiễu chữ viết tay/bút chì.
2. rectify_image: xoay ảnh quanh tâm để đường kẻ ngang nằm ngang (nắn nghiêng).
3. normalize_illumination: phẳng hóa độ sáng (chia nền blur cục bộ) + giãn tương phản
   theo phân vị 2-98% -> chống vùng tối/soi góc của ảnh chụp.

Thuật toán là CV cổ điển (không cần mô hình DL): ổn định, không cần huấn luyện,
đo được từng bước trên chính dữ liệu. Dùng trước pipeline chấm (lattice tự fit
đã chịu được biến dạng nhỏ; bước này giúp mực nhạt đạt ngưỡng phát hiện).
"""
import cv2
import numpy as np


def estimate_rotation(img, min_line_frac=0.22, max_angle_deg=12.0):
    """Góc nghiêng (độ) của trang: trung vị góc các đường ngang dài trên mask mực."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.blur(gray, (61, 61))
    ink = ((gray.astype(np.float32) / (bg + 1e-6)) < 0.82).astype(np.uint8) * 255
    h, w = ink.shape
    lines = cv2.HoughLinesP(ink, 1, np.pi / 180, threshold=140,
                            minLineLength=int(min_line_frac * w), maxLineGap=14)
    if lines is None:
        return 0.0, 0
    angles = []
    for l in np.asarray(lines).reshape(-1, 4):
        dx, dy = l[2] - l[0], l[3] - l[1]
        if abs(dx) < 1e-6:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        # chỉ giữ đường NGANG (góc nhỏ); đường dọc bị loại
        if abs(ang) <= max_angle_deg:
            angles.append(ang)
    if len(angles) < 3:
        return 0.0, len(angles)
    return float(np.median(angles)), len(angles)


def rectify_image(img, max_angle_deg=12.0):
    """Nắn nghiêng: xoay -góc(trung vị) quanh tâm, nền trắng."""
    ang, n = estimate_rotation(img, max_angle_deg=max_angle_deg)
    if abs(ang) < 0.15 or n < 3:
        return img, 0.0
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -ang, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated, ang


def normalize_illumination(gray, blur_k=61, pct_lo=2, pct_hi=98):
    """Gray phẳng sáng: chia nền cục bộ + giãn tương phản theo phân vị."""
    g = gray.astype(np.float32)
    bg = cv2.blur(g, (blur_k, blur_k))
    flat = np.clip(g / (bg + 1e-6) * 255.0, 0, 255)
    lo, hi = np.percentile(flat, [pct_lo, pct_hi])
    if hi - lo < 1e-3:
        return flat.astype(np.uint8)
    return np.clip((flat - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def prepare_document(img):
    """Chuẩn hóa hoàn chỉnh: nắn nghiêng + trả ảnh; in thống kê để giám sát."""
    img_r, ang = rectify_image(img)
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    # thống kê sáng/tối trước-sau (để báo cáo, không bắt buộc dùng)
    flat = normalize_illumination(gray)
    return img_r, ang, flat


if __name__ == "__main__":
    import sys
    from pathlib import Path
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/omr_orig")
    for f in sorted(base.glob("*.jpg"))[:12]:
        img = cv2.imread(str(f))
        ang, n = estimate_rotation(img)
        print(f"{f.stem}: nghiêng {ang:+.2f}° ({n} đường ngang)")
