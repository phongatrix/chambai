# -*- coding: utf-8 -*-
"""
Chấm điểm thật phiếu QM bằng OpenCV: fill-based + nhãn overlay.

Semantics đã xác minh: vòng xanh quanh fill = ô tô ĐÚNG; đỏ = ô tô SAI;
xanh quanh ô TRỐNG = hiển thị đáp án (không tính điểm).

Đầu ra:
- Điểm độc lập (fill + key) so với điểm phần mềm cũ (file KQ)
- SBD + mã đề decode từ fill
- Báo cáo chênh lệch
"""
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import openpyxl

warnings.filterwarnings("ignore")
ORIG = Path("/tmp/omr_orig")
OUT = Path("/tmp/omr_out")
OUT.mkdir(exist_ok=True)
_R = Path(__file__).resolve().parent
_tulieu_local = _R / "data/tulieu"
TULIEU = _tulieu_local if _tulieu_local.is_dir() else Path("/home/dell/Documents/tulieuchambai")

# ---------- ảnh ----------
def ink_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.blur(gray, (61, 61))
    rel = gray / (bg + 1e-6)
    b, g, r = img[..., 0].astype(np.int32), img[..., 1].astype(np.int32), img[..., 2].astype(np.int32)
    colorful = np.maximum(np.maximum(abs(r - g), abs(r - b)), abs(g - b))
    ink = (rel < 0.82) & (colorful < 55)
    return (ink.astype(np.uint8)) * 255


def process_page(path):
    img = cv2.imread(str(path))
    img3 = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    mask = ink_mask(img3)
    return {"img3": img3, "mask": mask, "W": img3.shape[1], "H": img3.shape[0]}


def color_pixels(img3, kind):
    hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if kind == "green":
        m = (h >= 40) & (h <= 85) & (s > 90) & (v > 40)
    else:
        m = (((h <= 10) | (h >= 168)) & (s > 90) & (v > 40))
    return m.astype(np.uint8)


def sample_fill(mask, cx, cy, r):
    x0, x1 = int(cx - r), int(cx + r + 1)
    y0, y1 = int(cy - r), int(cy + r + 1)
    if x0 < 0 or y0 < 0 or x1 > mask.shape[1] or y1 > mask.shape[0]:
        return 0.0
    roi = mask[y0:y1, x0:x1].astype(np.float32) / 255.0
    h_, w_ = roi.shape
    yy, xx = np.mgrid[0:h_, 0:w_]
    disk = ((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2) <= r * r
    return float(roi[disk].mean()) if disk.sum() else 0.0


def annulus_color(gmask, rmask, cx, cy, r_in, r_out):
    """Đếm px xanh/đỏ trong vành quanh (cx,cy)."""
    x0, x1 = int(cx - r_out), int(cx + r_out + 1)
    y0, y1 = int(cy - r_out), int(cy + r_out + 1)
    if x0 < 0 or y0 < 0 or x1 > gmask.shape[1] or y1 > gmask.shape[0]:
        return 0, 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (d2 >= r_in * r_in) & (d2 <= r_out * r_out)
    g = int(gmask[y0:y1, x0:x1][ring].sum())
    rd = int(rmask[y0:y1, x0:x1][ring].sum())
    return g, rd


# ---------- blob ----------
def find_fills(mask, x_min, x_max, y_min, y_max, size=(26, 82)):
    """Fill đặc = blob tròn, tâm đậm; trả list (cx, cy, w, h, strength)."""
    sub = mask[int(y_min):int(y_max), int(x_min):int(x_max)]
    contours, hierarchy = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    fills = []
    for i, c in enumerate(contours):
        if i < len(hierarchy) and hierarchy[i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 220 or not (size[0] <= w <= size[1] and size[0] <= h <= size[1]):
            continue
        if not (0.55 <= w / h <= 1.8):
            continue
        cx, cy = x + w / 2 + x_min, y + h / 2 + y_min
        strength = sample_fill(mask, cx, cy, min(w, h) * 0.30)
        if strength < 0.35:
            continue
        fills.append({"cx": cx, "cy": cy, "w": w, "h": h, "strength": strength})
    return fills


def cluster(vals, gap):
    vals = sorted(vals)
    if not vals:
        return []
    grps = [[vals[0]]]
    for v in vals[1:]:
        if v - grps[-1][-1] <= gap:
            grps[-1].append(v)
        else:
            grps.append([v])
    return [float(np.mean(g)) for g in grps]


def fit_lattice_1d(vals, pitch_guess, gap=None):
    v = np.array(sorted(vals), float)
    if len(v) < 3:
        return None
    gap = gap or pitch_guess * 0.55
    centers = cluster(v.tolist(), gap)
    if len(centers) < 3:
        return None
    difs = np.diff(centers)
    cand = difs[difs < 1.6 * pitch_guess]
    pitch = float(np.median(cand)) if len(cand) >= 2 else float(pitch_guess)
    y0 = centers[0]
    for _ in range(4):
        k = np.round((v - y0) / pitch)
        A = np.vstack([np.ones_like(k), k]).T
        sol, *_ = np.linalg.lstsq(A, v, rcond=None)
        y0, pitch = float(sol[0]), float(sol[1])
        if pitch <= 0 or pitch > pitch_guess * 1.8:
            return None
    return y0, pitch


def nearest_idx(v, centers, pitch, y0):
    return int(round((v - y0) / pitch))


# ---------- key ----------
def load_qm_key():
    wb = openpyxl.load_workbook(TULIEU / "mauda/dapanQM2025.xlsx", data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    key = {}
    for r in rows[1:]:
        if r and r[0] is not None:
            de = int(r[0])
            p1 = {i: str(r[i]).strip().upper() for i in range(1, 41) if r[i] is not None}
            p2 = {}
            for j in range(41, 57):
                if r[j] is not None:
                    # cột 41..44 = câu 1 ý a-d; 45..48 = câu 2 ...
                    cq = (j - 41) // 4 + 1
                    yk = chr(97 + (j - 41) % 4)
                    p2[(cq, yk)] = str(r[j]).strip().upper()
            p3 = {}
            for j in range(73, 79):
                if r[j] is not None:
                    p3[j - 72] = float(r[j])
            key[de] = {"p1": p1, "p2": p2, "p3": p3}
    return key


def load_kq():
    out = {}
    for grp, rel in [("g1011", "ketqua/Giay10_11/KQ Diem Tin - GK1 - 26-27.xlsx"),
                     ("g12", "ketqua/Giay12/KQ Diem Tin 12-GK1-26-27.xlsx")]:
        wb = openpyxl.load_workbook(TULIEU / rel, data_only=True)
        ws = wb.active
        rows = []
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r and r[0] is not None:
                rows.append({"sbd": str(r[0]).strip().strip("'"),
                             "made": str(r[1]).strip().strip("'") if r[1] is not None else None,
                             "p1": float(r[2]) if r[2] is not None else None,
                             "p2": float(r[3]) if r[3] is not None else None,
                             "p3": float(r[4]) if r[4] is not None else None,
                             "tong": float(r[5]) if r[5] is not None else None})
        out[grp] = rows
    return out


# ---------- decode top zone ----------
def decode_top(page, grp):
    """Decode MĐ (và SBD cho g12) từ fill vùng trên phải."""
    mask, W, H = page["mask"], page["W"], page["H"]
    pitch_md = 62.0 if grp == "g1011" else 79.0
    y_min, y_max = H * 0.028, H * (0.22 if grp == "g1011" else 0.30)
    x_min = W * 0.60
    fills = [f for f in find_fills(mask, x_min, W, y_min, y_max, size=(32, 58))
             if f["strength"] > 0.85]
    if len(fills) < 3:
        return None, fills
    xs = cluster([f["cx"] for f in fills], 38)
    grids = [[xs[0]]]
    for x in xs[1:]:
        if x - grids[-1][-1] <= 100:
            grids[-1].append(x)
        else:
            grids.append([x])
    res = {"md": None, "sbd": None, "fills": fills}

    def digits_for(cols, anchor="min"):
        if anchor == "min":
            y0 = min(f["cy"] for f in fills if any(abs(f["cx"] - x) < 40 for x in cols))
        out = []
        for cx in cols:
            fc = min((f for f in fills if abs(f["cx"] - cx) < 40),
                     key=lambda f: f["cy"], default=None)
            out.append(int(round((fc["cy"] - y0) / pitch_md)) if fc else None)
        return out

    if grp == "g1011":
        g = grids[-1]
        if len(g) != 3:
            return res, fills
        md = digits_for(g)
        res["md"] = md
        res["md_str"] = "".join(str(d) if d is not None else "?" for d in md)
    else:
        md_cols = grids[-1][-4:] if len(grids[-1]) >= 4 else None
        sbd_cols = []
        for gg in grids[:-1] if len(grids) > 1 else []:
            sbd_cols.extend(gg)
        if md_cols:
            res["md"] = digits_for(md_cols)
            res["md_str"] = "".join(str(d) if d is not None else "?" for d in res["md"])
        if sbd_cols and len(sbd_cols) >= 6:
            res["sbd"] = digits_for(sbd_cols[-8:] if len(sbd_cols) > 8 else sbd_cols)
            res["sbd_str"] = "".join(str(d) if d is not None else "?" for d in res["sbd"])
    return res, fills


# ---------- zones ----------
def zones(grp, W, H):
    if grp == "g1011":
        return {"top": (0.60, 1.0, 0.0, 0.25),
                "P1": (0.0, 0.24, 0.30, 1.0),
                "P2": (0.24, 1.0, 0.22, 0.50),
                "P3": (0.24, 1.0, 0.50, 1.0)}
    return {"top": (0.50, 1.0, 0.0, 0.32),
            "P1": (0.0, 1.0, 0.32, 0.585),
            "P2": (0.0, 1.0, 0.585, 0.72),
            "P3": (0.0, 1.0, 0.72, 1.0)}


def main():
    key = load_qm_key()
    kq = load_kq()
    print(f"Key QM: {len(key)} đề; P2 câu: {sorted({c for de in key.values() for (c, _) in de['p2']})}")
    files = sorted(ORIG.glob("g1011-*.jpg")) + sorted(ORIG.glob("g12-*.jpg"))
    results = []
    for i, f in enumerate(files):
        grp = "g1011" if f.stem.startswith("g1011") else "g12"
        row_idx = i if grp == "g1011" else i - 45
        krow = kq[grp][row_idx] if row_idx < len(kq[grp]) else None
        page = process_page(f)
        W, H = page["W"], page["H"]
        gmask = color_pixels(page["img3"], "green")
        rmask = color_pixels(page["img3"], "red")
        zs = zones(grp, W, H)
        top, top_fills = decode_top(page, grp)
        # label fill theo màu annulus
        def label(f):
            g, r = annulus_color(gmask, rmask, f["cx"], f["cy"], f["w"] * 0.5, f["w"] * 0.82)
            if g > 40 and g > 1.5 * r:
                return "green"
            if r > 40 and r > 1.5 * g:
                return "red"
            return None
        res = {"file": f.stem, "grp": grp, "md": top["md_str"] if top else None,
               "sbd": top["sbd_str"] if top and "sbd_str" in top else None}
        # đếm theo vùng
        zone_fills = {}
        for zn, (x0, x1, y0, y1) in zs.items():
            zf = [fl for fl in find_fills(page["mask"], W * x0, W * x1, H * y0, H * y1)
                  if not (zn == "top" and False)]
            zone_fills[zn] = zf
        for zn in ("P1", "P2", "P3"):
            ng = sum(1 for fl in zone_fills[zn] if label(fl) == "green")
            nr = sum(1 for fl in zone_fills[zn] if label(fl) == "red")
            res[f"{zn}_green"] = ng
            res[f"{zn}_red"] = nr
        res["my_p1"] = round(0.25 * res["P1_green"], 2)
        res["my_p2"] = round(0.25 * res["P2_green"], 2)
        res["my_tong"] = round(res["my_p1"] + res["my_p2"], 2)
        if krow:
            res["kq"] = krow
            res["match"] = abs(res["my_tong"] - (krow["tong"] or 0)) < 0.05
        results.append(res)
        print(f"{f.stem:12s} md={res['md']} sbd={res.get('sbd')} | "
              f"P1 g/r={res['P1_green']}/{res['P1_red']} P2 g/r={res['P2_green']}/{res['P2_red']} | "
              f"my={res['my_tong']:.2f} kq={krow['tong'] if krow else '?'} "
              f"{'✓' if res.get('match') else '✗'}")
    (OUT / "stage_fill.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for r in results if r.get("match"))
    print(f"\nKhớp tổng điểm: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
