# -*- coding: utf-8 -*-
"""
Chấm điểm thật lattice-based: fill -> đáp án -> key -> điểm, đối chiếu KQ.

- P1 g1011: cột trái, 30 hàng x 4 cột ABCD; key câu 1-18. Anchor offset A kiểm định bằng
  key (greens phải trùng key đáp án).
- P2 g1011: 2 khối x 5 câu, mỗi câu 4 ý hàng x 2 cột Đ/S; key câu 1-4.
- P3: đếm fill (kỳ vọng 0).
- g12 (THPT): P1 10 hàng x 4 khối x 4 cột (H1: câu=4r+b+1 | H2: câu=10b+r+1) - kiểm định
  bằng nhất quán key suy ra giữa 2 HS mã đề 4185; P2 4 ý hàng x 8 câu x Đ/S.
"""
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omr_grade_v3 import (ORIG, OUT, TULIEU, annulus_color, cluster, color_pixels,  # noqa
                          find_fills, ink_mask, load_kq, load_qm_key, process_page,
                          sample_fill)

LET = "ABCD"


def label_fill(gmask, rmask, f):
    g, r = annulus_color(gmask, rmask, f["cx"], f["cy"], f["w"] * 0.5, f["w"] * 0.82)
    if g > 40 and g > 1.5 * r:
        return "green"
    if r > 40 and r > 1.5 * g:
        return "red"
    return None


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


def cell_of(point, y0, pitch, n_max):
    k = int(round((point - y0) / pitch))
    return k if 0 <= k <= n_max else None


def grade_page(path, grp, key, kq_row, verbose=False):
    page = process_page(path)
    W, H = page["W"], page["H"]
    gmask = color_pixels(page["img3"], "green")
    rmask = color_pixels(page["img3"], "red")
    pitch_r = 63.0 if grp == "g1011" else 67.0
    out = {"file": path.stem, "grp": grp, "kq": kq_row}

    if grp == "g1011":
        fills = find_fills(page["mask"], 0, W * 0.24, H * 0.30, H * 0.99, size=(28, 70))
        for f in fills:
            f["label"] = label_fill(gmask, rmask, f)
        # cột: từ fill w>=38 (loại chữ số câu nhỏ hơn)
        xs = cluster([f["cx"] for f in fills if f["w"] >= 38], 40)
        cols = [x for x in xs if x < W * 0.25]
        if len(cols) < 3 or len(cols) > 5:
            out["error"] = f"P1 cols={len(cols)}"
            return out
        cols = cols[:4] if len(cols) >= 4 else cols
        # hàng: từ fill (tất cả, gồm cả nhãn)
        fit = fit_lattice_1d([f["cy"] for f in fills], pitch_r)
        if not fit:
            out["error"] = "P1 rows fit fail"
            return out
        y0, pitch = fit
        # gán ô: hàng k=0..29, cột
        cells = {}
        for f in fills:
            k = cell_of(f["cy"], y0, pitch, 29)
            if k is None:
                continue
            ci = int(np.argmin([abs(f["cx"] - c) for c in cols]))
            if abs(f["cx"] - cols[ci]) > pitch * 0.8:
                continue
            cells.setdefault((k, ci), []).append(f)
        # anchor A: câu(row) = row - A + 1 (row 0 = câu 1 khi A=0)
        best = None
        for A in range(0, 4):
            agree, tot = 0, 0
            for (row, ci), fs in cells.items():
                cq = row - A + 1
                if cq < 1 or cq > 18:
                    continue
                for f in fs:
                    if f["label"] in ("green", "red"):
                        tot += 1
                        de = int(kq_row["made"])
                        if de in key and key[de]["p1"].get(cq) == LET[ci]:
                            agree += 1
            if tot and (best is None or agree / tot > best[1]):
                best = (A, agree / tot)
        out["P1_anchor"] = best
        A = best[0] if best else 0
        # đáp án P1: mỗi câu; fill chọn = fill có nhãn nếu có, else fill mạnh nhất
        answers, score = {}, 0.0
        de = int(kq_row["made"])
        k1 = key.get(de, {}).get("p1", {})
        for (row, ci), fs in cells.items():
            cq = row - A + 1
            if cq not in k1:
                continue
            lab = [f for f in fs if f["label"]]
            pick = lab[0] if len(lab) == 1 else (max(fs, key=lambda z: z["strength"]) if len(fs) == 1 else None)
            if pick is not None:
                ans = LET[ci]
                answers[cq] = ans
                if ans == k1[cq]:
                    score += 0.25
        out["P1_score"] = round(score, 2)
        out["P1_answers"] = answers
        # P2
        fills2 = find_fills(page["mask"], W * 0.24, W, H * 0.21, H * 0.52, size=(28, 70))
        for f in fills2:
            f["label"] = label_fill(gmask, rmask, f)
        fit2 = fit_lattice_1d([f["cy"] for f in fills2], 65)
        if not fit2:
            out["error2"] = "P2 rows fit fail"
            return out
        y02, pitch2 = fit2
        xs2 = cluster([f["cx"] for f in fills2 if f["w"] >= 38], 45)
        cols2 = [x for x in xs2 if x > W * 0.22]
        # ghép cặp (Đ,S): khoảng cách ~130
        pairs = []
        i = 0
        while i < len(cols2) - 1:
            if cols2[i + 1] - cols2[i] < 0.55 * pitch2 * 2:  # ~<190
                pairs.append((cols2[i], cols2[i + 1]))
                i += 2
            else:
                i += 1
        score2, answers2 = 0.0, {}
        k2 = key.get(de, {}).get("p2", {})
        cells2 = {}
        for f in fills2:
            k = cell_of(f["cy"], y02, pitch2, 7)
            if k is None:
                continue
            for pi, (xd, xs_) in enumerate(pairs):
                if abs(f["cx"] - xd) < pitch2 * 0.55:
                    ci = 0  # Đ
                elif abs(f["cx"] - xs_) < pitch2 * 0.55:
                    ci = 1  # S
                else:
                    continue
                cells2.setdefault((k, pi, ci), []).append(f)
        for (row, pi, ci), fs in cells2.items():
            cq = pi + 1
            yk = chr(97 + row % 4)
            if (cq, yk) not in k2:
                continue
            lab = [f for f in fs if f["label"]]
            pick = lab[0] if len(lab) == 1 else (fs[0] if len(fs) == 1 else None)
            if pick is None:
                continue
            ans = "D" if ci == 0 else "S"
            answers2[(cq, yk)] = ans
            if ans == k2[(cq, yk)]:
                score2 += 0.25
        out["P2_score"] = round(score2, 2)
        out["P2_answers"] = {f"{c}{y}": v for (c, y), v in answers2.items()}
        # P3: đếm fill
        fills3 = find_fills(page["mask"], W * 0.24, W, H * 0.52, H, size=(28, 70))
        out["P3_fills"] = len(fills3)
        out["tong"] = round(out["P1_score"] + out["P2_score"], 2)
        return out
    # ---- g12 ----
    fills = find_fills(page["mask"], 0, W, H * 0.32, H * 0.585, size=(28, 70))
    for f in fills:
        f["label"] = label_fill(gmask, rmask, f)
    fit = fit_lattice_1d([f["cy"] for f in fills], pitch_r)
    if not fit:
        out["error"] = "P1 rows fit fail"
        return out
    y0, pitch = fit
    xs = cluster([f["cx"] for f in fills if f["w"] >= 38], 45)
    cols = [x for x in xs if x < W * 0.98]
    if len(cols) < 12 or len(cols) > 20:
        out["error"] = f"P1 g12 cols={len(cols)}"
        return out
    cols = merge16(cols)
    cells = {}
    for f in fills:
        k = cell_of(f["cy"], y0, pitch, 9)
        if k is None:
            continue
        ci = int(np.argmin([abs(f["cx"] - c) for c in cols]))
        if abs(f["cx"] - cols[ci]) > pitch * 0.8:
            continue
        cells.setdefault((k, ci), []).append(f)
    # giả thuyết câu
    for hyp, fn in (("H1", lambda row, ci: 4 * row + ci // 4 + 1),
                    ("H2", lambda row, ci: 10 * (ci // 4) + row + 1)):
        answers, score = {}, 0.0
        for (row, ci), fs in cells.items():
            cq = fn(row, ci)
            lab = [f for f in fs if f["label"]]
            pick = lab[0] if len(lab) == 1 else (max(fs, key=lambda z: z["strength"]) if len(fs) == 1 else None)
            if pick is not None:
                answers[cq] = LET[ci % 4]
                score += 0.0
        out[f"ans_{hyp}"] = answers
    out["P1_cols"] = cols
    out["P1_y0"] = round(y0, 1)
    out["P1_pitch"] = round(pitch, 2)
    return out


def merge16(cols, want=16):
    while len(cols) > want:
        difs = [(cols[i + 1] - cols[i], i) for i in range(len(cols) - 1)]
        difs.sort()
        _, i = difs[0]
        cols[i] = (cols[i] + cols[i + 1]) / 2
        del cols[i + 1]
    return cols


def main():
    key = load_qm_key()
    kq = load_kq()
    files = sorted(ORIG.glob("g1011-*.jpg")) + sorted(ORIG.glob("g12-*.jpg"))
    results = []
    n_ok = 0
    for i, f in enumerate(files):
        grp = "g1011" if f.stem.startswith("g1011") else "g12"
        row_idx = i if grp == "g1011" else i - 45
        krow = kq[grp][row_idx] if row_idx < len(kq[grp]) else None
        if grp != "g1011":
            results.append({"file": f.stem, "grp": grp, "kq": krow})
            continue
        r = grade_page(f, grp, key, krow)
        if krow:
            r["match"] = abs(r.get("tong", 99) - (krow["tong"] or 0)) < 0.05
            n_ok += r["match"]
        results.append(r)
        print(f"{f.stem:12s} A={r.get('P1_anchor')} P1={r.get('P1_score')} P2={r.get('P2_score')} "
              f"T={r.get('tong')} KQ={krow['tong'] if krow else '?'} "
              f"{'✓' if r.get('match') else '✗'} err={r.get('error') or r.get('error2') or ''}")
    (OUT / "stage_lattice.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")
    print(f"\nKhớp tổng: {n_ok}/45")


if __name__ == "__main__":
    main()
