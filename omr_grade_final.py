# -*- coding: utf-8 -*-
"""
CHẤM ĐIỂM CUỐI CÙNG (fill + nhãn màu overlay).

Đã xác minh bằng dữ liệu:
- Vòng xanh quanh fill = ô tô ĐÚNG (giống nhau tuyệt đối giữa HS cùng mã đề = key)
- Vòng đỏ = ô tô SAI; xanh quanh ô trống = hiển thị key (bỏ qua)
- Điểm = 0.25 x (#fill xanh ở P1) + 0.25 x (#fill xanh ở P2); P3 trống hết (0/0)
- dapanQM2025 KHÔNG khớp key P1 của đề thi này -> key lấy từ overlay (duy nhất có)

Đầu ra: bảng so my vs KQ 49 trang + Excel output/ + JSON debug.
"""
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omr_grade_v3 import (ORIG, OUT, TULIEU, cluster, color_pixels, ink_mask,  # noqa
                          decode_top, find_fills, load_kq)
from omr_grade_v5 import fit_lattice_1d  # noqa
from omr_preprocess import rectify_image  # noqa

# Chuẩn hóa đầu vào: nắn nghiêng trang theo đường kẻ ngang in sẵn.
# A/B trên 49 trang (2026-11-01): OFF = 30/49 khớp tuyệt đối, TB chênh 0.128;
# ON = 29/49, TB 0.163 — ảnh chụp bị méo PHỐI CẢNH (keystone), xoay trung vị
# để lại méo dư + nội suy làm mờ mực nhạt; lattice tự fit từng trang hấp thụ
# biến dạng tốt hơn. Mặc định OFF; bật khi đầu vào là ảnh quét nghiêng thuần.
PREPROCESS = False
RECTIFY_PERSP = False  # nắn phối cảnh: cần hiệu chuẩn khung trước khi bật


EXIF_ROTATE = {6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE,
               3: cv2.ROTATE_180}


def rectify_perspective(img, target_w=930):
    """Nắn phối cảnh: tìm quad 4 góc tờ giấy (vùng sáng lớn nhất) -> homography
    về khung chuẩn. Trả (ảnh đã nắn, quad hoặc None)."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return img, None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * W * H:
        return img, None
    peri = cv2.arcLength(c, True)
    quad = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(quad) != 4:
        # xấp xỉ cứng: hull -> 4 điểm xa nhất
        hull = cv2.convexHull(c)
        quad = cv2.approxPolyDP(hull, 0.03 * peri, True)
    if len(quad) != 4:
        return img, None
    pts = quad.reshape(4, 2).astype(np.float32)
    # sắp: tl, tr, br, bl theo tổng/hiệu tọa độ
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    # kích thước chuẩn: rộng/thiệu theo cạnh thật
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    tw, th = int(max(wA, wB)), int(max(hA, hB))
    if tw < 300 or th < 300:
        return img, None
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]),
                                    np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]))
    out = cv2.warpPerspective(img, M, (tw, th), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    scale = target_w / tw
    if scale < 1.0:
        out = cv2.resize(out, (target_w, int(th * scale)), interpolation=cv2.INTER_AREA)
    return out, [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]


def process_page(path, preprocess=None):
    """Đọc trang: tự xoay theo EXIF Orientation + (tuỳ chọn) nắn nghiêng."""
    img = cv2.imread(str(path))
    try:
        from PIL import Image as PILImage
        with PILImage.open(str(path)) as pim:
            tag = pim.getexif().get(274)
        if tag in EXIF_ROTATE:
            img = cv2.rotate(img, EXIF_ROTATE[tag])
    except Exception:
        pass
    ang = 0.0
    do = PREPROCESS if preprocess is None else preprocess
    if do:
        img, ang = rectify_image(img)
    # NẮN PHỐI CẢNH: TẮT theo mặc định — cần hiệu chuẩn riêng cho khung đã nắn
    # (quad hiện tại nhầm nền bàn; bật thủ công khi đã hiệu chuẩn)
    try:
        if RECTIFY_PERSP and max(img.shape[:2]) > 1150:
            img, _q = rectify_perspective(img)
    except Exception:
        pass
    img3 = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    mask = ink_mask(img3)
    return {"img3": img3, "mask": mask, "W": img3.shape[1], "H": img3.shape[0],
            "angle": ang}

# Ảnh gốc trích từ PDF + đầu ra: tương đối theo thư mục chứa script
_ROOT = Path(__file__).resolve().parent
OUTDIR = _ROOT / "output"
OUTDIR.mkdir(exist_ok=True)
ORIG = _ROOT / "data/scan_pages"
PDFS = [
    ("g1011", TULIEU / "ketqua/Giay10_11/Tin - GK1 - 26-27_7605001358703799744.pdf", 45),
    ("g12", TULIEU / "ketqua/Giay12/Tin 12-GK1-26-27_3112491266583288225.pdf", 4),
]


def ensure_sources():
    """Trích ảnh gốc từ 2 PDF kết quả nếu data/scan_pages chưa đủ."""
    ORIG.mkdir(parents=True, exist_ok=True)
    import subprocess
    for grp, pdf, expect in PDFS:
        have = len(list(ORIG.glob(f"{grp}-*.jpg")))
        if have >= expect:
            continue
        if not pdf.is_file():
            raise FileNotFoundError(f"Không thấy PDF nguồn: {pdf}")
        print(f"Trích ảnh từ {pdf.name} ...")
        subprocess.run(["pdfimages", "-j", str(pdf), str(ORIG / grp)], check=True)
        assert len(list(ORIG.glob(f"{grp}-*.jpg"))) >= expect


def valid_green_cells(fills, pitch_guess, n_rows, col_gap=30, dedupe="row", return_fit=False,
                      rightmost=None):
    """Tập ô green hợp lệ (k, ci) sau lattice + dedupe; return_fit=True -> (cells, (y0, pitch)).

    dedupe="row": mỗi hàng câu chỉ 1 key (P1) -> giữ green mạnh nhất cùng hàng.
    dedupe="cell": mỗi ô (hàng ý, cột Đ/S) chỉ 1 key (P2) -> giữ mạnh nhất cùng ô.
    """
    greens = [f for f in fills if f.get("label") == "green" and f["strength"] >= 0.5]
    if len(greens) < 3:
        return ({(i, -1) for i in range(len(greens))} if greens else set(), None) \
            if return_fit else ({(i, -1) for i in range(len(greens))} if greens else set())
    strong = [f for f in greens if f["strength"] >= 0.85]
    base = strong if len(strong) >= 4 else greens
    fit = fit_lattice_1d([f["cy"] for f in base], pitch_guess)
    if not fit:
        return ({(i, -1) for i in range(len(greens))}, None) if return_fit \
            else {(i, -1) for i in range(len(greens))}
    y0, pitch = fit
    xs = cluster([f["cx"] for f in greens], col_gap)
    if len(xs) < 2:
        xs = cluster([f["cx"] for f in base], col_gap)
    if rightmost and len(xs) > rightmost:
        # cột đáp án = 4 cụm NHIỀU GREEN NHẤT (ABCD luôn có green nhiều hơn nhiễu)
        support = {xc: sum(1 for f in greens if abs(f["cx"] - xc) <= col_gap * 0.8)
                   for xc in xs}
        xs = sorted(sorted(support, key=lambda c: (-support[c], c))[:rightmost])
    col_pitch = float(np.median(np.diff(xs))) if len(xs) > 1 else pitch
    # thu thập greens hợp lệ kèm (hàng, cột)
    valid_list = []
    for _ in range(2):  # refit lặp: inliers -> fit lại
        valid_list = []
        for f in greens:
            k = int(round((f["cy"] - y0) / pitch))
            if k < 0 or k >= n_rows:
                continue
            if abs(f["cy"] - (y0 + pitch * k)) > 0.25 * pitch:
                continue
            ci = int(np.argmin([abs(f["cx"] - c) for c in xs])) if xs else -1
            if ci >= 0 and abs(f["cx"] - xs[ci]) > 0.35 * col_pitch:
                continue
            valid_list.append((f, k, ci))
        if len(valid_list) < 4 or len(valid_list) >= 0.8 * len(greens):
            break
        fit2 = fit_lattice_1d([f["cy"] for f, _, _ in valid_list], pitch)
        if fit2:
            y0, pitch = fit2
    # dedupe: key chỉ 1 đáp án mỗi hàng (P1) / mỗi ô (P2); giữ green mạnh nhất
    groups = {}
    for f, k, ci in valid_list:
        key = (k, ci) if dedupe == "cell" else k
        if key not in groups or f["strength"] > groups[key][0]["strength"]:
            groups[key] = (f, k, ci)
    cells = {(k, ci) for (f, k, ci) in groups.values()}
    if return_fit:
        fit = fit_lattice_1d([f["cy"] for f, _, _ in groups.values()], pitch) \
            if groups else None
        return cells, fit
    return cells


def count_valid_greens(fills, pitch_guess, n_rows, col_gap=30, dedupe="row"):
    """Đếm greens hợp lệ (số ô sau lattice + dedupe)."""
    return len(valid_green_cells(fills, pitch_guess, n_rows, col_gap, dedupe))


P3_SCORE_PER_CAU = 0.5  # không ảnh hưởng: P3 0/0 trên mọi trang


def classify_fill(gmask, rmask, f):
    """Nhãn fill: xanh/đỏ chỉ khi vòng màu BAO QUANH fill (>=3/4 góc phần tư)."""
    cx, cy, w = f["cx"], f["cy"], f["w"]
    r_in, r_out = w * 0.5, w * 0.85
    x0, x1 = int(cx - r_out), int(cx + r_out + 1)
    y0, y1 = int(cy - r_out), int(cy + r_out + 1)
    if x0 < 0 or y0 < 0 or x1 > gmask.shape[1] or y1 > gmask.shape[0]:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    d2 = dx * dx + dy * dy
    ring = (d2 >= r_in * r_in) & (d2 <= r_out * r_out)
    ang = np.arctan2(dy, dx)
    qi = np.floor((ang + np.pi) / (np.pi / 2)).astype(int) % 4
    gq, rq = [], []
    for sec in range(4):
        qsel = ring & (qi == sec)
        gq.append(int(gmask[y0:y1, x0:x1][qsel].sum()))
        rq.append(int(rmask[y0:y1, x0:x1][qsel].sum()))
    g, r = sum(gq), sum(rq)
    nqg = sum(1 for v in gq if v >= 40)
    nqr = sum(1 for v in rq if v >= 40)
    if g > 150 and nqg >= 3 and g > 1.5 * r:
        return "green"
    if r > 150 and nqr >= 3 and r > 1.5 * g:
        return "red"
    return None


def auto_group(page):
    """Tự nhận loại phiếu: phiếu THPT (g12) có lưới SBD 8 CỘT riêng — phiếu QM (g1011)
    không có (SBD viết tay). Fallback: đếm cột lưới MÃ ĐỀ (3 cột = QM, 4 cột = THPT)
    dùng cho trang SBD trống (học sinh không tô)."""
    mask, W, H = page["mask"], page["W"], page["H"]
    sbd_fills = [f for f in find_fills(mask, W * 0.65, W * 0.90, H * 0.09, H * 0.27,
                                       size=(30, 75)) if f["strength"] > 0.7]
    if len(cluster([f["cx"] for f in sbd_fills], 30)) >= 5:
        return "g12"
    # fallback 1: decode lưới MĐ đủ 4 chữ số + pitch lưới đúng của THPT (~79px,
    # QM ~62px) — chặn giả decode từ nhiễu cột trên phiếu QM
    # (dùng cho trang SBD trống; giới hạn đã ghi nhận: SBD trống + decode fail -> g1011)
    top4, tfills = decode_top(page, "g12")
    if top4 and top4.get("md") and len(top4["md"]) == 4 and all(d is not None for d in top4["md"]):
        xs = cluster([f["cx"] for f in tfills], 30)
        if len(xs) >= 4:
            mcols = xs[-4:]
            ys = sorted(f["cy"] for f in tfills
                        if any(abs(f["cx"] - x) < 30 for x in mcols))
            difs = [b - a for a, b in zip(ys, ys[1:]) if 50 < b - a < 130]
            pitch = float(np.median(difs)) if difs else 0.0
            if pitch >= 71:
                return "g12"
    return "g1011"


def collect_pages(source=None):
    """Nguồn trang: None = bộ 49 trang tích hợp; 'gdrive:<url>' = Drive; khác = thư mục."""
    if source is None:
        ensure_sources()
        kq = load_kq()
        files = sorted(ORIG.glob("g1011-*.jpg")) + sorted(ORIG.glob("g12-*.jpg"))
        base = [(f, ("g1011" if f.stem.startswith("g1011") else "g12")) for f in files]
        cnt = {"g1011": 0, "g12": 0}
        pages = []
        for f, g in base:
            i = cnt[g]
            cnt[g] += 1
            pages.append((f, g, kq[g][i] if i < len(kq[g]) else None))
        return pages, kq
    if source.startswith("gdrive:"):
        from omr_gdrive import fetch_and_prepare
        pages_dir, _ = fetch_and_prepare(source[len("gdrive:"):])
    else:
        pages_dir = Path(source[4:] if source.startswith("dir:") else source)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(p for p in Path(pages_dir).iterdir() if p.suffix.lower() in exts)
    if not files:
        raise RuntimeError(f"Không có ảnh trang nào trong {pages_dir}")
    return [(f, None, None) for f in files], None


def grade_page_full(f, grp=None, krow=None):
    """Chấm 1 trang — dùng chung cho CLI và webapp. grp=None -> tự nhận loại phiếu."""
    if grp is None:
        page0 = process_page(f)
        grp = auto_group(page0)
    page = process_page(f)
    W, H = page["W"], page["H"]
    gmask = color_pixels(page["img3"], "green")
    rmask = color_pixels(page["img3"], "red")
    if grp == "g1011":
        zones = {"top": (0.60, 1.0, 0.028, 0.22), "P1": (0.0, 0.24, 0.30, 0.99),
                 "P2": (0.24, 1.0, 0.21, 0.52), "P3": (0.24, 1.0, 0.52, 1.0)}
    else:
        zones = {"top": (0.50, 1.0, 0.0, 0.32), "P1": (0.0, 1.0, 0.32, 0.585),
                 "P2": (0.0, 1.0, 0.585, 0.72), "P3": (0.0, 1.0, 0.72, 1.0)}
    zone_fills = {}
    for zn, (x0, x1, y0, y1) in zones.items():
        zone_fills[zn] = find_fills(page["mask"], W * x0, W * x1, H * y0, H * y1,
                                    size=(24, 72))
    for zn in ("P1", "P2", "P3"):
        for fl in zone_fills[zn]:
            fl["label"] = classify_fill(gmask, rmask, fl)
            fl["g"] = 0
            fl["r"] = 0
    cells1 = valid_green_cells(zone_fills["P1"], 66 if grp == "g1011" else 67,
                               30 if grp == "g1011" else 10,
                               dedupe="row" if grp == "g1011" else "cell",
                               rightmost=4 if grp == "g1011" else None)
    cells2 = valid_green_cells(zone_fills["P2"], 65, 8 if grp == "g1011" else 4,
                               dedupe="cell")
    pattern = sorted([["P1", k, c] for k, c in cells1] + [["P2", k, c] for k, c in cells2])
    ng = {"P1": len(cells1),
          "P2": len(cells2),
          "P3": sum(1 for fl in zone_fills["P3"] if fl["label"] == "green")}
    nr = {zn: sum(1 for fl in zone_fills[zn] if fl["label"] == "red")
          for zn in ("P1", "P2", "P3")}
    nf = {zn: len(zone_fills[zn]) for zn in ("P1", "P2", "P3")}
    my = {"P1": round(0.25 * ng["P1"], 2),
          "P2": round(0.25 * ng["P2"], 2),
          "P3": round(P3_SCORE_PER_CAU * ng["P3"], 2)}
    my["tong"] = round(my["P1"] + my["P2"] + my["P3"], 2)
    top, _ = decode_top(page, grp)
    res = {"file": Path(f).stem, "grp": grp,
           "md": top.get("md_str") if top else None,
           "sbd": top.get("sbd_str") if top else None,
           "pattern": pattern,
           "counts": {"P1": {"green": ng["P1"], "red": nr["P1"], "fills": nf["P1"]},
                      "P2": {"green": ng["P2"], "red": nr["P2"], "fills": nf["P2"]},
                      "P3": {"green": ng["P3"], "red": nr["P3"], "fills": nf["P3"]}},
           "my": my, "kq": krow,
           "fills_detail": {zn: [[round(fl["cx"]), round(fl["cy"]), round(fl["w"]),
                                  fl.get("label") or ""] for fl in zone_fills[zn]]
                            for zn in ("P1", "P2", "P3")}}
    if krow:
        res["diff"] = {"P1": round(my["P1"] - (krow["p1"] or 0), 2),
                       "P2": round(my["P2"] - (krow["p2"] or 0), 2),
                       "tong": round(my["tong"] - (krow["tong"] or 0), 2)}
        res["match"] = abs(res["diff"]["tong"]) < 0.05
        res["md_match"] = (res["md"] == krow["made"]) if res["md"] else None
    return res


def infer_made(results, min_score=0.90, canonical_refs=None):
    """Suy mã đề từ pattern ô xanh.

    canonical_refs: {đề: pattern} CHUẨN (nạp từ keys_builtin.json — nhãn đã kiểm
    chứng, KHÔNG tin nhãn grid decode của phiên). Thiếu trang nào không khớp
    canonical -> dùng tham chiếu nội phiên (nhóm theo pattern giống nhau)."""
    def label_of(r):
        k = r.get("kq") or {}
        if k.get("made"):
            return str(k["made"]).strip()
        md = r.get("md")
        return md if (md and "?" not in str(md)) else None

    # CHỈ dùng ref chuẩn (từ keys_builtin.json, nhãn đã kiểm chứng) —
    # nhãn grid decode của phiên là rác khi lưới decode lỗi, tự-trùng luôn thắng.
    refs = {}
    if canonical_refs:
        for de, dk in canonical_refs.items():
            pat = dk.get("pattern") if isinstance(dk, dict) else None
            if pat:
                refs[str(de)] = {tuple(p) for p in pat}
    n_fixed = 0
    n_fixed = 0
    for r in results:
        pat = {tuple(p) for p in (r.get("pattern") or [])}
        lab = label_of(r)
        grid_md = r.get("md")
        if not pat:
            r["md_source"] = "none"
            continue
        def recall(ref, max_shift=3):
            best = 0.0
            for d in range(-max_shift, max_shift + 1):
                shifted = {(s, k + d, c) for (s, k, c) in ref}
                best = max(best, len(pat & shifted) / len(ref)) if ref else 0.0
            return best

        scored = sorted(((recall(ref), lab2) for lab2, ref in refs.items()), reverse=True)
        best_score, best_lab = scored[0] if scored else (0.0, None)
        runner = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= min_score and (best_score >= 1.6 * runner or len(refs) == 1):
            changed = str(grid_md or "") != str(best_lab)
            r["md"], r["md_source"], r["md_score"] = best_lab, "answer", round(best_score, 2)
            if changed:
                n_fixed += 1
        else:
            r["md_source"] = ("kq" if (r.get("kq") or {}).get("made")
                              else ("grid" if grid_md else "none"))
            r["md_score"] = round(best_score, 2)
    return n_fixed, refs


LET = "ABCD"
P1_OFFSET_PITCHES = 23.15  # lưới MĐ -> câu 1 P1 (đo trên bộ 49 trang)


def robust_grid_y0(ys, pitch):
    """Chọn y0 (hàng 0) của lưới = ứng viên có NHIỀU điểm nhất khớp lattice ±0.3 pitch."""
    ys = sorted(set(round(float(y)) for y in ys))
    best, best_n = ys[0], -1
    for y0 in ys:
        n = sum(1 for y in ys
                if abs(y - (y0 + pitch * round((y - y0) / pitch))) <= 0.3 * pitch)
        if n > best_n:
            best, best_n = y0, n
    return float(best)


def extract_answers_raw(page, grp):
    """Trích đáp án thô (không cần overlay): trả dict {P1: {câu: A-D},
    P2: {(câu, ý): 'D'/'S'}, p1_skip, p2_skip, geometry}.

    Neo câu 1 của P1 vào lưới MÃ ĐỀ (y0_md + P1_OFFSET_PITCHES*pitch_md).
    Trả về cả câu bị tô nhiều lựa chọn (invalid) để suy đề/chấm công bằng.
    """
    mask, W, H = page["mask"], page["W"], page["H"]
    out = {"grp": grp, "P1": {}, "P2": {}, "P1_invalid": [], "P2_invalid": [],
           "P3_fills": 0}
    # --- lưới MĐ: y0 + pitch ---
    top, md_fills = decode_top(page, grp)
    if not top or len(md_fills) < 3:
        out["error"] = "md_fails"
        return out
    ys = sorted(f["cy"] for f in md_fills)
    difs = [b - a for a, b in zip(ys, ys[1:]) if 40 < b - a < 110]
    pitch_md = float(np.median(difs)) if difs else (62.0 if grp == "g1011" else 79.0)
    y0_md = robust_grid_y0(ys, pitch_md)
    out["md_geo"] = (round(y0_md), round(pitch_md, 1))

    if grp == "g1011":
        fills = find_fills(mask, 0, W * 0.24, H * 0.30, H * 0.99, size=(28, 70))
        if len(fills) < 4:
            out["error"] = "p1_fills_thieu"
            return out
        xs = cluster([f["cx"] for f in fills if f["w"] >= 36], 40)
        sel = [x for x in xs if x > W * 0.055][-4:]
        if len(sel) < 3:
            out["error"] = "p1_cols_thieu"
            return out
        lat = [f for f in fills if f["strength"] >= 0.85
               and min(abs(f["cx"] - c) for c in sel) < 63 * 0.45]
        fit = fit_lattice_1d([f["cy"] for f in lat], 63)
        if not fit:
            out["error"] = "p1_rows_fail"
            return out
        y0, pitch = fit
        cau1_y = y0_md + P1_OFFSET_PITCHES * pitch_md
        A = int(round((cau1_y - y0) / pitch))  # hàng lattice ứng với câu 1
        out["p1_anchor_A"] = A
        cells1 = []
        for f in fills:
            k = int(round((f["cy"] - y0) / pitch))
            if abs(f["cy"] - (y0 + pitch * k)) > 0.4 * pitch:
                continue
            ci = int(np.argmin([abs(f["cx"] - c) for c in sel]))
            if abs(f["cx"] - sel[ci]) > 0.5 * pitch:
                continue
            cells1.append((k, ci))
        out["cells1"] = cells1
        out["p1_geo"] = (round(y0), round(pitch, 1))
        # P2: 4 ý-hàng khối 1 + cặp Đ/S
        fills2 = find_fills(mask, W * 0.24, W, H * 0.21, H * 0.52, size=(28, 70))
        lat2 = [f for f in fills2 if f["strength"] >= 0.85]
        fit2 = fit_lattice_1d([f["cy"] for f in lat2], 65)
        if fit2:
            y02, pitch2 = fit2
            xs2 = cluster([f["cx"] for f in fills2 if f["w"] >= 34], 45)
            pairs = []
            i = 0
            while i < len(xs2) - 1:
                gap = xs2[i + 1] - xs2[i]
                if 80 <= gap <= 200:
                    pairs.append((xs2[i], xs2[i + 1]))
                    i += 2
                else:
                    i += 1
            rcs = cluster([f["cy"] for f in lat2], 36)
            out["p2_rows"] = len(rcs)
            for f in fills2:
                # gán ý-hàng: cluster tâm gần nhất
                if not rcs:
                    break
                ri = min(range(len(rcs)), key=lambda j: abs(f["cy"] - rcs[j]))
                if ri >= 4:  # chỉ khối 1 (câu 1-5, key câu 1-4)
                    continue
                yk = chr(97 + ri)
                for pi, (xd, xs_) in enumerate(pairs):
                    cq = pi + 1
                    if abs(f["cx"] - xd) < 0.5 * pitch2:
                        cell = (cq, yk, "D")
                    elif abs(f["cx"] - xs_) < 0.5 * pitch2:
                        cell = (cq, yk, "S")
                    else:
                        continue
                    out["P2"].setdefault((cq, yk), []).append(cell)
            for kk, lst in list(out["P2"].items()):
                cols = {c[2] for c in lst}
                if len(cols) == 1:
                    out["P2"][kk] = next(iter(cols))
                else:
                    out["P2_invalid"].append(kk)
                    del out["P2"][kk]
        fills3 = find_fills(mask, W * 0.24, W, H * 0.52, H, size=(28, 70))
        out["P3_fills"] = len(fills3)
        return out
    # g12 (THPT): P1 = 10 hàng x 4 khối; hai giả thuyết câu; P2 = 4 ý hàng x 8 câu
    fills = find_fills(mask, 0, W, H * 0.32, H * 0.585, size=(28, 70))
    fit = fit_lattice_1d([f["cy"] for f in fills if f["strength"] >= 0.85], 67)
    if not fit:
        out["error"] = "g12_rows_fail"
        return out
    y0, pitch = fit
    # lưới cột 3 tham số + slope phối cảnh: x' = x - slope*(k - k_mid)
    fx = [(f["cx"], int(round((f["cy"] - y0) / pitch))) for f in fills]
    k_mid = float(np.median([k for _, k in fx]))
    best_lat, best_inl = None, 0
    for slope10 in range(-15, 16):  # slope = slope10/1 px/hàng (skew phối cảnh)
        slope = slope10 * 1.0
        fx2 = np.array([x - slope * (k - k_mid) for (x, k) in fx])
        for p1i in range(70, 132, 2):
            p1 = p1i / 100.0
            for pg in range(500, 751, 5):
                rel = fx2 - np.arange(0, 280, 4)[:, None]  # x0 grid vector hóa
                for xi, x0 in enumerate(range(0, 280, 4)):
                    r = rel[xi]
                    blk = np.floor(r / pg).astype(int)
                    pos = (r - blk * pg) / p1
                    pi = np.round(pos)
                    n = int(((np.abs(pos - pi) * p1 < 24) & (blk >= 0) & (blk <= 3)
                             & (pi >= 0) & (pi <= 3)).sum())
                    if n > best_inl:
                        best_inl, best_lat = n, (x0, p1, pg, slope)
                if best_inl >= 40:
                    break
            if best_inl >= 40:
                break
        if best_inl >= 40:
            break
    if not best_lat or best_inl < 30:
        out["error"] = "g12_cols_fail"
        return out
    x0, p1, pg, slope = best_lat
    out["p1_cols"] = (x0, round(p1, 2), round(pg, 1), round(slope, 2), best_inl)
    for f in fills:
        k = int(round((f["cy"] - y0) / pitch))
        if abs(f["cy"] - (y0 + pitch * k)) > 0.4 * pitch:
            continue
        rel = f["cx"] - x0 - slope * (k - k_mid)
        blk = int(rel // pg)
        if not (0 <= blk <= 3):
            continue
        pos = (rel - blk * pg) / p1
        pi = round(pos)
        if abs(pos - pi) * p1 > 24 or not (0 <= pi <= 3):
            continue
        ci = blk * 4 + pi
        out.setdefault("cells_g12", []).append((k, ci))
    fills2 = find_fills(mask, 0, W, H * 0.585, H * 0.72, size=(28, 70))
    fit2 = fit_lattice_1d([f["cy"] for f in fills2 if f["strength"] >= 0.85], 65)
    if fit2:
        y02, pitch2 = fit2
        xs2 = cluster([f["cx"] for f in fills2 if f["w"] >= 34], 45)
        pairs = []
        i = 0
        while i < len(xs2) - 1:
            gap = xs2[i + 1] - xs2[i]
            if 80 <= gap <= 200:
                pairs.append((xs2[i], xs2[i + 1]))
                i += 2
            else:
                i += 1
        rcs = cluster([f["cy"] for f in fills2 if f["strength"] >= 0.85], 36)
        for f in fills2:
            if not rcs:
                break
            ri = min(range(len(rcs)), key=lambda j: abs(f["cy"] - rcs[j]))
            if ri >= 4:
                continue
            yk = chr(97 + ri)
            for pi, (xd, xs_) in enumerate(pairs):
                cq = pi + 1
                if abs(f["cx"] - xd) < 0.5 * pitch2:
                    out["P2"].setdefault((cq, yk), []).append("D")
                elif abs(f["cx"] - xs_) < 0.5 * pitch2:
                    out["P2"].setdefault((cq, yk), []).append("S")
        for kk, lst in list(out["P2"].items()):
            if len(set(lst)) == 1:
                out["P2"][kk] = lst[0]
            else:
                out["P2_invalid"].append(kk)
                del out["P2"][kk]
    return out


def grade_with_key(page, grp, key, min_score=0.90):
    """Chấm 1 trang bằng key upload: trích đáp án -> suy đề (>=min_score) -> điểm."""
    ext = extract_answers_raw(page, grp)
    if "error" in ext:
        return {"file": Path(page.get("path", "?")).stem if page.get("path") else "?",
                "grp": grp, "error": ext["error"], "my": {"P1": 0, "P2": 0, "P3": 0, "tong": 0}}
    des = key["các đề"] if key.get("đề") == "QM" else {"list": key["câu"]}
    cells1 = ext.get("cells1") or []
    p1_hyp = {}
    if not cells1:
        for (hyp, cq), ans in (ext.get("P1") or {}).items():
            p1_hyp.setdefault(hyp, []).append((cq, ans))
    A0 = ext.get("p1_anchor_A", 0)
    n_cells = len(cells1) + len(ext.get("P2") or {})
    min_answered = max(8, int(0.5 * n_cells)) if n_cells else 8
    per_de = {}
    for de, dk in des.items():
        p1k = dk.get("p1", {})
        p2k = dk.get("p2", {})
        p1_max = max(p1k) if p1k else 0
        best_de = (0.0, 0, A0)  # (score, matched, anchor)
        # (a) g1011: cells1 + A-search
        for dA in (range(-8, 9) if cells1 else (0,)):
            A = A0 + dA
            answered = matched = 0
            for (row, ci) in cells1:
                cq = row - A + 1
                if cq in p1k:
                    answered += 1
                    matched += (LET[ci] == p1k[cq])
            for (cq, yk), ans in (ext.get("P2") or {}).items():
                for kcq in (cq, cq + p1_max):
                    if (kcq, yk) in p2k:
                        answered += 1
                        matched += (ans == p2k[(kcq, yk)])
                        break
            if answered >= min_answered:
                score = matched / answered
                if (matched, score) > (best_de[1], best_de[0]):
                    best_de = (score, matched, A)
        # (b) g12/THPT: quét (giả thuyết H1/H2 × hàng câu 1) theo key
        cg = ext.get("cells_g12") or []
        for hyp in ("H1", "H2"):
            for row0 in range(0, 21):
                answered = matched = 0
                for (k, ci) in cg:
                    rr = k - row0
                    if not (0 <= rr < 10):
                        continue
                    cq = (4 * rr + ci // 4 + 1) if hyp == "H1" else (10 * (ci // 4) + rr + 1)
                    if cq in p1k:
                        answered += 1
                        matched += (LET[ci % 4] == p1k[cq])
                for (cq, yk), ans in (ext.get("P2") or {}).items():
                    for kcq in (cq, cq + p1_max):
                        if (kcq, yk) in p2k:
                            answered += 1
                            matched += (ans == p2k[(kcq, yk)])
                            break
                if answered >= min_answered:
                    score = matched / answered
                    if (matched, score) > (best_de[1], best_de[0]):
                        best_de = (score, matched, f"{hyp}@{row0}")
        per_de[de] = best_de
    ranked = sorted(per_de.items(), key=lambda kv: (kv[1][1], kv[1][0]), reverse=True)
    (best_de_code, (best_score, best_matched, best_A)) = ranked[0]
    runner_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
    best = des[best_de_code]
    # xác nhận đề: >=90% (quy tắc người dùng) HOẶC phân biệt rõ với đề nhì
    confirmed = best_score >= min_score or (best_score >= 0.50 and
                                            best_score >= 1.6 * runner_score)
    md_rule = ("90%" if best_score >= min_score else
               "margin" if confirmed else None)
    if not confirmed:
        return {"file": "?", "grp": grp,
                "error": f"đề không xác định (khớp tối đa {best_score:.0%})",
                "my": {"P1": 0, "P2": 0, "P3": 0, "tong": 0}}
    dk = best
    p1k, p2k = dk.get("p1", {}), dk.get("p2", {})
    p1_max = max(p1k) if p1k else 0
    s1 = s2 = 0.0
    p1_ans = {}
    if cells1:
        for (row, ci) in cells1:
            cq = row - best_A + 1
            p1_ans[cq] = LET[ci]
            if cq in p1k and LET[ci] == p1k[cq]:
                s1 += 0.25
    elif ext.get("cells_g12") and "@" in str(best_A):
        hyp, row0 = str(best_A).split("@")
        row0 = int(row0)
        for (k, ci) in ext["cells_g12"]:
            rr = k - row0
            if not (0 <= rr < 10):
                continue
            cq = (4 * rr + ci // 4 + 1) if hyp == "H1" else (10 * (ci // 4) + rr + 1)
            p1_ans[cq] = LET[ci % 4]
            if cq in p1k and LET[ci % 4] == p1k[cq]:
                s1 += 0.25
    p2_ans = {}
    for (cq, yk), ans in (ext.get("P2") or {}).items():
        p2_ans[f"{cq}{yk}"] = ans
        for kcq in (cq, cq + p1_max):
            if (kcq, yk) in p2k and ans == p2k[(kcq, yk)]:
                s2 += 0.25
                break
    return {"file": "?", "grp": grp, "de_inferred": best_de_code, "key_score": round(best_score, 2),
            "md_rule": md_rule, "runner_score": round(runner_score, 2),
            "p1_anchor_used": best_A,
            "P1_answers": p1_ans, "P2_answers": p2_ans,
            "P1_invalid": ext["P1_invalid"], "P2_invalid": ext["P2_invalid"],
            "my": {"P1": round(s1, 2), "P2": round(s2, 2), "P3": 0.0,
                   "tong": round(s1 + s2, 2)}}


_collect_pages_src = collect_pages


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Chấm điểm phiếu trắc nghiệm từ ảnh quét")
    ap.add_argument("--source", default=None,
                    help="'gdrive:<link Drive>' hoặc 'dir:<thư mục ảnh>'; bỏ trống = bộ 49 trang tích hợp")
    ap.add_argument("--out", default=None, help="Tên file Excel đầu ra")
    args = ap.parse_args()

    pages, kq = _collect_pages_src(args.source)
    print(f"Nguồn: {'GDrive/thư mục' if args.source else 'tích hợp'} — {len(pages)} trang; "
          f"KQ đối chiếu: {'có' if kq else 'không (chấm độc lập)'}")
    results = []
    for f, grp, krow in pages:
        res = grade_page_full(f, grp, krow)
        results.append(res)
        print(f"{res['file']:12s} md={res['md'] or '????'} "
              f"({'✓' if res.get('md_match') else '?':s}) "
              f"P1 g{res['counts']['P1']['green']}/r{res['counts']['P1']['red']}/{res['counts']['P1']['fills']} "
              f"P2 g{res['counts']['P2']['green']}/r{res['counts']['P2']['red']}/{res['counts']['P2']['fills']} | "
              f"my {res['my']['tong']:5.2f} vs KQ {krow['tong'] if krow else '?':>5} "
              f"{'✓' if res.get('match') else ('d' + format(res['diff']['tong'], '+.2f') if krow else '')}")
    canon = None
    kb = _ROOT / "data/keys_builtin.json"
    if kb.is_file():
        canon = json.loads(kb.read_text(encoding="utf-8")).get("các đề")
    n_fixed, _ = infer_made(results, canonical_refs=canon)
    print(f"\n--- Mã đề suy từ đáp án (pattern ô xanh >= 90%): sửa {n_fixed} trang ---")
    if kq:
        md_ok = sum(1 for r in results
                    if str(r.get("md") or "") == str((r.get("kq") or {}).get("made") or ""))
        ok = sum(1 for r in results if r.get("match"))
        print(f"=== TỔNG: khớp điểm {ok}/{len(results)}; khớp mã đề (suy từ đáp án) {md_ok}/{len(results)}")
    json_name = "grading_final.json" if kq else "grading_source.json"
    (OUTDIR / json_name).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    if kq:
        ok = sum(1 for r in results if r.get("match"))
        # (thống kê mã đề đã in ở phần suy luận phía trên)
    # Excel
    out_name = args.out or ("KQ_cham_AI_vs_phan_mem_cu.xlsx" if kq else "KQ_cham_GDrive.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "KQ chấm AI"
    ws.append(["STT", "File", "SBD", "Mã đề", "P1 (AI)", "P1 (KQ)", "P2 (AI)", "P2 (KQ)",
               "P3 (AI)", "P3 (KQ)", "Tổng (AI)", "Tổng (KQ)", "Chênh"])
    for idx, r in enumerate(results, 1):
        k = r["kq"] or {}
        ws.append([idx, r["file"], r["sbd"] or k.get("sbd", ""), r["md"] or k.get("made", ""),
                   r["my"]["P1"], k.get("p1"), r["my"]["P2"], k.get("p2"),
                   r["my"]["P3"], k.get("p3"), r["my"]["tong"], k.get("tong"),
                   r["diff"]["tong"] if "diff" in r else None])
    for col, wdt in zip("ABCDEFGHIJKLM", (5, 12, 12, 8, 8, 8, 8, 8, 8, 8, 10, 10, 8)):
        ws.column_dimensions[col].width = wdt
    wb.save(OUTDIR / out_name)
    print(f"Đã lưu: {OUTDIR / out_name}")
    # Lưu lần chấm vào CSDL lịch sử (SQLite); lỗi DB không ảnh hưởng kết quả chấm
    try:
        from omr_db import save_run
        src_label = args.source if args.source else "built-in"
        rid = save_run(src_label, results)
        print(f"Đã lưu lần chấm vào CSDL: run {rid} ({len(results)} trang) — data/omr_results.db")
    except Exception as db_err:
        print(f"(Lưu CSDL bỏ qua: {db_err})")
    # Render overlay daquet-style
    try:
        from omr_overlay import render_batch
        pages_all, _ = _collect_pages_src(args.source)
        items = [(f, r) for (f, g, k), r in zip(pages_all, results)]
        out_dir = OUTDIR / "da_quet" / (str(rid) if "rid" in dir() else "latest")
        paths = render_batch(items, out_dir)
        print(f"Đã render {len(paths)} ảnh overlay daquet-style: {out_dir}")
    except Exception as ov_err:
        print(f"(Render overlay bỏ qua: {ov_err})")


if __name__ == "__main__":
    main()
