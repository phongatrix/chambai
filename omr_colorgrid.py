# -*- coding: utf-8 -*-
"""OMR COLOR GRID — lưới mẫu chuẩn từ vùng MÀU + chuẩn hóa ảnh theo lưới (2026-11-06)
=========================================================================
Ý tưởng: màu in sẵn trên mẫu (dải ngang, block màu) là MỐC CĂN "sống sót" qua
chụp mờ/nghiêng và PHÂN BIỆT HỌ MẪU. Từ các vùng màu của mẫu ta XÂY DỰNG LŨI
MẪU CHUẨN: mỗi ô bubble của mẫu được gán vào dải màu chứa tâm y của nó (khung
màu). Khi chuẩn hóa ảnh chụp, ta:
  1. Tìm các vùng màu tương ứng trên ảnh (cùng hue, cùng thứ tự y).
  2. Fit phép biến đổi mẫu→ảnh (H4) từ các cặp vùng màu (align_by_color).
  3. Đưa lưới ô mẫu qua H4 -> vị trí dự kiến từng ô trên ảnh.
  4. Xác thực: mỗi ô dự kiến phải có vòng tròn/bubble in tại vị trí đó
     (color_grid_agreement). Nếu khớp >= ngưỡng -> chấp nhận H4 làm init_H
     cho pipeline chuẩn hoá (align_photo + ICP + ring gate).

API:
  color_grid(tpl_path)          -> list ô [{x, y, w, hue_group, hue, ...}] hoặc []
  align_color_grid(img, tpl)    -> (H4, score, info) hoặc None
  color_grid_agreement(warp, grid, tpl_path, H4) -> tỉ lệ ô có vòng tại vị trí
                                                      chiếu (None nếu không màu)
  apply_color_grid(...)         -> tiện ích áp H4 lên ảnh (warp) để xem/trả ảnh

Kill-switch: env OMR_COLOR=0 -> mọi hàm trả trống/None.
"""
import os
import hashlib

import cv2
import numpy as np

import omr_coloralign as CA
import omr_sheetgrid as S

_COLOR_OFF = (os.environ.get("OMR_COLOR") == "0")
_GRID_CACHE = {}


def color_grid(tpl_path):
    """Lưới mẫu chuẩn từ vùng MÀU.

    Mỗi ô bubble của mẫu (từ _upscaled_bubbles, cỡ mẫu) được gán vào DÃI MÀU
    chứa tâm y của nó (từ template_bands). Trả list dict:
      {x, y, w, hue_group, hue, part, question, option}
    hoặc [] nếu mẫu không có màu (moi, scan đen trắng, photocopy -> bỏ qua).
    """
    ck = str(tpl_path)
    if ck in _GRID_CACHE:
        return _GRID_CACHE[ck]
    out = []
    try:
        tpl = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
        if tpl is None:
            _GRID_CACHE[ck] = []
            return []
        if tpl.ndim == 3 and tpl.shape[2] == 4:
            tpl = tpl[:, :, :3]
        bands = CA.template_bands(tpl_path)
        if not bands:
            _GRID_CACHE[ck] = []
            return []
        bs = S._upscaled_bubbles(tpl_path)   # [(x, y, w)] ở cỡ mẫu
        h_tpl = float(tpl.shape[0])
        for i, (x, y, w) in enumerate(bs):
            yrel = y / h_tpl
            # gán vào dải màu chứa tâm y; nếu không rơi vào dải nào -> dải gần nhất
            group = None
            best_d = 1e9
            for bi, b in enumerate(bands):
                if b["y0"] <= yrel <= b["y1"]:
                    group = bi
                    break
                d = abs((b["y0"] + b["y1"]) / 2.0 - yrel)
                if d < best_d:
                    best_d = d
                    group = bi
            out.append({
                "index": i,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "hue_group": group,
                "hue": float(bands[group]["hue"]),
            })
    except Exception:  # noqa
        out = []
    _GRID_CACHE[ck] = out
    return out


def align_color_grid(img, tpl_path, grid=None):
    """CẨN CHỈNH ẢNH DỰA TRÊN LŨI MÀU (lưới mẫu chuẩn).

    Trả (H4, score, info) hoặc None.

    Bước 1: H4 thô từ vùng màu qua CA.align_by_color (least squares scale_y/
    offset_y trên tỉ lệ — tách trôi thang; KHÔNG có offset_x vì dải ngang
    collinear).
    Bước 2: REFINE bằng lưới ô bubble: chiếu tâm từng ô MẪU qua H4 thô, tìm
    vòng tròn ảnh trong ngưỡng, fit SIMILARITY (scale/rotation/translation —
    cv2.estimateAffinePartial2D RANSAC) trên các cặp (tâm ô mẫu, tâm vòng).
    Đây là "chuẩn hóa dựa vào lưới mẫu chuẩn": lưới chuẩn khớp vòng ảnh cho
    phép biến đổi HOÀN CHỈNH (có offset_x, scale_x, góc nghiêng nhẹ).
    """
    if _COLOR_OFF:
        return None
    if grid is None:
        grid = color_grid(tpl_path)
    if not grid:
        return None
    H4 = CA.align_by_color(img, tpl_path)
    if H4 is None:
        return None
    H4 = np.asarray(H4, np.float64)
    tpl_im = cv2.imread(str(tpl_path))
    if tpl_im is None:
        return None
    h_tpl, w_tpl = tpl_im.shape[:2]
    h_img, w_img = img.shape[:2]
    base = _grid_score(img, grid, H4, h_tpl, w_tpl, h_img, w_img)
    H4r, score = _refine_with_grid(img, grid, H4, h_tpl, w_tpl, h_img, w_img,
                                   base)
    info = {"bands": len(grid), "score": float(score),
            "refined": bool(score > base + 0.02)}
    return H4r, score, info


def _refine_with_grid(img, grid, H4, h_tpl, w_tpl, h_img, w_img, base_score):
    """Refine H4 bằng khớp vòng tròn bubble (lưới mẫu chuẩn) — similarity.

    Chiếu tâm ô mẫu qua H4 thô, tìm vòng ảnh gần nhất trong ngưỡng; với đủ
    cặp (>6) fit similarity RANSAC. Giữ kết quả nếu score cải thiện."""
    try:
        rings = S._ring_centers(img, accept_filled=True)
        if len(rings) < 8:
            return H4, base_score
        H = np.asarray(H4, np.float64)
        sx_img = w_img / float(w_tpl)
        src, dst = [], []
        for g in grid:
            px = H[0, 0] * g["x"] + H[0, 2]
            py = H[1, 1] * g["y"] + H[1, 2]
            dd = np.linalg.norm(rings[:, :2] - np.float32([px, py]), axis=1)
            j = int(np.argmin(dd))
            tol = 0.35 * g["w"] * sx_img
            if float(dd[j]) <= tol:
                src.append((g["x"], g["y"]))
                dst.append((float(rings[j][0]), float(rings[j][1])))
        if len(src) < 8:
            return H4, base_score
        M, _ = cv2.estimateAffinePartial2D(
            np.float32(src), np.float32(dst), method=cv2.RANSAC,
            ransacReprojThreshold=8.0)
        if M is None:
            return H4, base_score
        H4n = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
        sc = _grid_score(img, grid, H4n, h_tpl, w_tpl, h_img, w_img)
        if sc > base_score:
            return H4n, sc
        return H4, base_score
    except Exception:  # noqa
        return H4, base_score


def _grid_score(img, grid, H4, h_tpl, w_tpl, h_img, w_img):
    """Tỷ lệ ô mẫu có VÒNG TRÒN tại vị trí dự kiến trên ảnh (dự án qua H4)."""
    try:
        rings = S._ring_centers(img, accept_filled=True)
        if len(rings) < 10:
            return 0.0
        hit = 0
        for g in grid:
            # y_pred = s_y * y_tpl + t_y  (H4 chỉ có scale_y + offset_y theo y;
            # x_pred = s_x * x_tpl      do H4[0,0] = w_img/w_tpl, H4[0,2] = 0)
            H = np.asarray(H4, np.float64)
            px = H[0, 0] * g["x"] + H[0, 2]
            py = H[1, 1] * g["y"] + H[1, 2]
            # vòng gần nhất trong ngưỡng 0.25 * cỡ ô (px ảnh)
            dd = np.linalg.norm(rings[:, :2] - np.float32([px, py]), axis=1)
            j = int(np.argmin(dd))
            if float(dd[j]) <= 0.25 * g["w"] * (w_img / float(w_tpl)):
                hit += 1
        return hit / float(len(grid))
    except Exception:  # noqa
        return 0.0


def color_grid_agreement(warped, grid, tpl_path, H4):
    """Tỉ lệ ô mẫu có vòng tròn tại vị trí chiếu qua H4 trên ảnh ĐÃ WARP.

    Tương tự _grid_score nhưng nhận warped đã có sẵn (tránh warp lại). Dùng
    để xác thực sau khi normalize. Trả None nếu không có màu/grid.
    """
    if _COLOR_OFF or not grid:
        return None
    try:
        tpl_im = cv2.imread(str(tpl_path))
        if tpl_im is None:
            return None
        h_tpl, w_tpl = tpl_im.shape[:2]
        h_img, w_img = warped.shape[:2]
        return _grid_score(warped, grid, np.asarray(H4, np.float64),
                           h_tpl, w_tpl, h_img, w_img)
    except Exception:  # noqa
        return None


def apply_color_grid(img, tpl_path, grid=None):
    """Tiện ích: trả (warped, H4) — ảnh chụp đưa về hệ tỉ lệ mẫu theo lưới màu,
    hoặc (None, None) nếu không đủ màu. Dùng để xem/trả ảnh chuẩn hóa."""
    if _COLOR_OFF:
        return None, None
    res = align_color_grid(img, tpl_path, grid)
    if res is None:
        return None, None
    H4, score, _info = res
    tpl_im = cv2.imread(str(tpl_path))
    if tpl_im is None:
        return None, None
    H4 = np.asarray(H4, np.float64)
    warped = cv2.warpPerspective(img, H4,
                                 (int(tpl_im.shape[1]), int(tpl_im.shape[0])),
                                 flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                 borderValue=(255, 255, 255))
    return warped, H4
