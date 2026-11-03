# -*- coding: utf-8 -*-
"""
KIỂM THỬ LOGIC (test_sheetgrid.py) — chạy: .venv/bin/python test_sheetgrid.py

Các nhóm kiểm tra:
 1. Lưới mẫu: số ô/câu, toạ độ hợp lệ, khoá duy nhất
 2. Lưới SBD/Mã đề
 3. Ghép key: offset P2, phủ đủ đáp án, cả 4 đề
 4. decide(): trống / 1 đáp án / tô nhiều (quy tắc tương đối)
 5. grade(): điểm và trạng thái
 6. render_result(): đúng màu XANH/ĐỎ/VÀNG
 7. _h_valid: chặn homography suy biến
 8. _fit_lattice: chịu được hàng bị sót
 9. Đầu-cuối trên 2 ảnh chuẩn (điểm, SBD, mã đề)
10. Từ chối ảnh không khớp mẫu
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import omr_sheetgrid as S
from omr_key import detect_key_format
import cham_theo_mau as CTM

ROOT = Path(__file__).resolve().parent
TPL = {
    "12": "/home/dell/Documents/tulieuchambai/maugiay/12.png",
    "10_11": "/home/dell/Documents/tulieuchambai/maugiay/10_11.png",
}
KEY = {
    "12": "/home/dell/Documents/tulieuchambai/dapan/k12dapan.xlsx",
    "10_11": "/home/dell/Documents/tulieuchambai/dapan/k11dapan.xlsx",
}
PHOTO = {
    "12": "/home/dell/Documents/tulieuchambai/Test/IMG_20261030_042800.jpg",
    "10_11": "/home/dell/Documents/tulieuchambai/Test/IMG_20261030_042144.jpg",
}
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def group(title):
    print("\n" + title)


def main():
    keys = {p: detect_key_format(KEY[p])[1] for p in TPL}

    # ---------- 1. lưới mẫu ----------
    group("1. Lưới mẫu")
    expect = {"12": {"P1": (160, 40), "P2": (64, 8)}, "10_11": {"P1": (120, 30), "P2": (80, 10)}}
    cell_cache = {}
    for p, tpl in TPL.items():
        cells = S.build_template_cells(p, tpl)
        cell_cache[p] = cells
        w, h = cv2.imread(tpl).shape[1], cv2.imread(tpl).shape[0]
        for part, (n_cells, n_q) in expect[p].items():
            sub = [c for c in cells if c[0] == part]
            qs = {c[1] if not isinstance(c[1], tuple) else c[1][0] for c in sub}
            check(f"{p} {part}: {n_cells} ô / {n_q} câu",
                  len(sub) == n_cells and len(qs) == n_q, f"{len(sub)} ô, {len(qs)} câu")
        check(f"{p}: toạ độ nằm trong khung mẫu",
              all(0 <= c[3] <= w and 0 <= c[4] <= h for c in cells))
        keyset = {(c[0], c[1], c[2]) for c in cells}
        check(f"{p}: khoá ô duy nhất", len(keyset) == len(cells),
              f"{len(keyset)}/{len(cells)}")
        # mỗi câu P1 đủ 4 lựa chọn
        p1 = {}
        for (part, q, opt, x, y) in cells:
            if part == "P1":
                p1.setdefault(q, set()).add(opt)
        check(f"{p}: mỗi câu P1 đủ ABCD", all(v == set("ABCD") for v in p1.values()))

    # ---------- 2. lưới SBD/Mã đề ----------
    group("2. Lưới SBD / Mã đề")
    ids12 = S.build_id_cells("12", TPL["12"])
    ids10 = S.build_id_cells("10_11", TPL["10_11"])
    check("12: SBD 8 cột × 10 hàng", sum(1 for c in ids12 if c[0] == "SBD") == 80)
    check("12: Mã đề 4 cột × 10 hàng", sum(1 for c in ids12 if c[0] == "MD") == 40)
    check("12: mỗi cột đủ chữ số 0-9", all(
        sorted(c[2] for c in ids12 if c[0] == p and c[1] == col) == [str(i) for i in range(10)]
        for p in ("SBD", "MD") for col in sorted({c[1] for c in ids12 if c[0] == p})))
    check("10_11: Mã đề 3 cột × 10 hàng", len(ids10) == 30 and len({c[1] for c in ids10}) == 3)

    # ---------- 3. ghép key ----------
    group("3. Ghép key vào lưới")
    for p, tpl in TPL.items():
        cells = cell_cache[p]
        for de, kd in keys[p]["các đề"].items():
            g = S.build_answer_grid(p, tpl, kd)
            check(f"{p} đề {de}: offset P2 = {g['p2_offset']}",
                  g["p2_offset"] == max(kd["p1"]))
            miss1 = [c for c in kd.get("p1", {}) if not any(
                x[0] == "P1" and x[1] == c and g["answers"].get((x[0], x[1], x[2]))
                for x in cells)]
            miss2 = [k for k in kd.get("p2", {}) if not any(
                x[0] == "P2" and (x[1][0] + g["p2_offset"], x[1][1]) == k
                and g["answers"].get((x[0], x[1], x[2])) for x in cells)]
            check(f"{p} đề {de}: phủ đủ đáp án P1/P2", not miss1 and not miss2,
                  f"thiếu {miss1[:3]} {miss2[:3]}")
        # đáp án mỗi ô trong 1 câu phải nhất quán
        g = S.build_answer_grid(p, tpl, list(keys[p]["các đề"].values())[0])
        by_q = {}
        for (part, q, opt), exp in g["answers"].items():
            if exp:
                by_q.setdefault((part, q), set()).add(exp)
        check(f"{p}: đáp án nhất quán trong từng câu", all(len(v) == 1 for v in by_q.values()))

    # ---------- 4. decide ----------
    group("4. decide(): trống / một / nhiều")
    def mk(scores):
        return {( "P1", 1, o): (s, 0, 0, 20) for o, s in scores.items()}
    d = S.decide(mk({"A": 0.05, "B": 0.08, "C": 0.02, "D": 0.0}))
    check("trống khi mọi ô < ngưỡng", d[("P1", 1)]["state"] == "blank")
    d = S.decide(mk({"A": 0.05, "B": 0.95, "C": 0.1, "D": 0.03}))
    check("một đáp án khi trội", d[("P1", 1)]["state"] == "one" and d[("P1", 1)]["sel"] == ["B"])
    d = S.decide(mk({"A": 0.02, "B": 0.9, "C": 0.85, "D": 0.05}))
    check("tô nhiều khi 2 ô gần bằng", d[("P1", 1)]["state"] == "multi"
          and set(d[("P1", 1)]["sel"]) == {"B", "C"})
    d = S.decide(mk({"A": 0.02, "B": 0.9, "C": 0.4, "D": 0.05}))
    check("không coi nhiễu nhẹ là tô nhiều", d[("P1", 1)]["state"] == "one")

    # ---------- 5. grade ----------
    group("5. grade(): điểm & trạng thái")
    deckey = {"p1": {1: "C", 2: "A", 3: "D", 4: "B"}, "p2": {(5, "a"): "D"}, "p3": {}}
    g = S.build_answer_grid("12", TPL["12"], deckey)
    # giả lập: câu đúng, câu sai, câu trống, câu tô nhiều (câu 5a P2)
    reads = {}
    for opt in "ABCD":
        reads[("P1", 1, opt)] = (0.95 if opt == "C" else 0.02, 0, 0, 20)
        reads[("P1", 2, opt)] = (0.95 if opt == "B" else 0.02, 0, 0, 20)   # sai (key A)
        reads[("P1", 3, opt)] = (0.02, 0, 0, 20)                          # trống
    for side in "DS":
        reads[("P2", (1, "a"), side)] = (0.8 if side in "DS" else 0, 0, 0, 20)  # tô cả 2
    dec = S.decide(reads)
    s1, s2, detail, stats = S.grade(dec, g)
    check("câu đúng -> ok", detail[("P1", 1)][0] == "ok")
    check("câu sai -> wrong", detail[("P1", 2)][0] == "wrong")
    check("câu trống -> blank", detail[("P1", 3)][0] == "blank")
    check("P2 tô 2 -> multi", detail[("P2", (1, "a"))][0] == "multi")
    check("điểm P1 = 0.25 (1 câu đúng)", abs(s1 - 0.25) < 1e-9, str(s1))
    check("điểm P2 = 0", abs(s2) < 1e-9, str(s2))
    check("thống kê n đếm số câu có key", stats["P1"]["n"] >= 3 and stats["P1"]["ok"] == 1)

    # ---------- 6. màu ----------
    group("6. render_result(): XANH / ĐỎ / VÀNG")
    img = np.full((400, 600, 3), 255, np.uint8)
    cells = [("P1", 1, o, 100 + 40 * i, 100) for i, o in enumerate("ABCD")]
    H = np.eye(3)
    reads = {("P1", 1, "A"): (0.9, 100, 100, 12), ("P1", 1, "B"): (0.02, 140, 100, 12),
             ("P1", 1, "C"): (0.02, 180, 100, 12), ("P1", 1, "D"): (0.02, 220, 100, 12)}
    dec = S.decide(reads)
    grid = {"cells": cells, "answers": {("P1", 1, o): "A" for o in "ABCD"}, "p2_offset": 0}
    s1, s2, detail, stats = S.grade(dec, grid)
    S.render_result(img.copy(), dec, detail, stats, "/tmp/_t_color.jpg", "12", made="X", s1=s1, s2=s2)
    def has_color(img, box, pred):
        y0, y1, x0, x1 = box
        hsv = img[y0:y1, x0:x1].reshape(-1, 3)
        return any(pred(int(b), int(g), int(r)) for b, g, r in hsv)

    out = cv2.imread("/tmp/_t_color.jpg")
    ok_green = has_color(out, (80, 122, 80, 122), lambda b, g, r: g > 120 and r < 90 and b < 90)
    check("màu XANH khi đúng", ok_green)
    # sai: đổi key
    grid2 = {"cells": cells, "answers": {("P1", 1, o): "D" for o in "ABCD"}, "p2_offset": 0}
    s1, s2, detail, stats = S.grade(dec, grid2)
    S.render_result(img.copy(), dec, detail, stats, "/tmp/_t_color2.jpg", "12", made="X")
    out2 = cv2.imread("/tmp/_t_color2.jpg")
    ok_red = has_color(out2, (80, 122, 80, 122), lambda b, g, r: r > 150 and g < 90 and b < 90)
    check("màu ĐỎ khi sai", ok_red)
    # trống -> vàng
    dec3 = S.decide({("P1", 1, o): (0.01, 100 + 40 * i, 100, 12) for i, o in enumerate("ABCD")})
    s1, s2, detail, stats = S.grade(dec3, grid)
    S.render_result(img.copy(), dec3, detail, stats, "/tmp/_t_color3.jpg", "12", made="X")
    out3 = cv2.imread("/tmp/_t_color3.jpg")
    ok_yellow = has_color(out3, (80, 122, 80, 122), lambda b, g, r: r > 170 and g > 150 and b < 100)
    check("màu VÀNG khi trống/nhiều", ok_yellow)

    # ---------- 7. _h_valid ----------
    group("7. Chặn homography suy biến")
    ref = S.build_template_cells("12", TPL["12"])
    degenerate = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1]])
    check("H suy biến bị loại", not S._h_valid(degenerate, ref, (4608, 3456)))
    good = np.array([[6.0, 0, 100], [0, 6.0, 100], [0, 0, 1]])
    check("H hợp lệ được nhận", S._h_valid(good, ref, (4608, 3456)))
    check("H=None bị loại", not S._h_valid(None, ref, (4608, 3456)))

    # ---------- 8. _fit_lattice ----------
    group("8. _fit_lattice chịu hàng bị sót")
    true_y0, true_p = 120.0, 13.0
    vals = [true_y0 + k * true_p for k in range(10)]
    vals_missing = [v for i, v in enumerate(vals) if i not in (0, 4)]  # sót hàng 0 và 4
    fit = S._fit_lattice(vals_missing, 5.0, 20.0, expect=10)
    check("khôi phục đúng lưới khi sót 2 hàng",
          fit is not None and abs(fit[0] - true_y0) < 0.6 and abs(fit[1] - true_p) < 0.3,
          str(fit))
    fit2 = S._fit_lattice(vals, 5.0, 20.0, expect=10)
    check("lưới đầy đủ khớp chính xác",
          abs(fit2[0] - true_y0) < 0.1 and abs(fit2[1] - true_p) < 0.1, str(fit2))

    # ---------- 9. đầu-cuối ----------
    group("9. Chấm đầu-cuối 2 ảnh chuẩn")
    r12 = CTM.grade_one(PHOTO["12"], "12", TPL["12"], keys["12"])
    check("12: khớp mẫu >= 0.9", r12 is not None and r12["quality"] >= 0.90,
          f"{r12 and r12['quality']}")
    if r12:
        check("12: P1 24/24 và P2 16/16",
              r12["stats"]["P1"] == {"ok": 24, "n": 24} and r12["stats"]["P2"] == {"ok": 16, "n": 16},
              str(r12["stats"]))
        check("12: tổng 10.0", abs(r12["score"] - 10.0) < 1e-9, str(r12["score"]))
        check("12: SBD đọc = 50120074", r12["sbd"] == "50120074", str(r12["sbd"]))
        check("12: mã đề = 3157 (nguồn lưới)", r12["made"] == "3157" and r12["made_source"] == "lưới",
              f"{r12['made']}/{r12['made_source']}")
    rq = CTM.grade_one(PHOTO["10_11"], "10_11", TPL["10_11"], keys["10_11"])
    check("10_11: khớp mẫu >= 0.9", rq is not None and rq["quality"] >= 0.90,
          f"{rq and rq['quality']}")
    if rq:
        check("10_11: P1 16/16 và P2 12/12",
              rq["stats"]["P1"] == {"ok": 16, "n": 16} and rq["stats"]["P2"] == {"ok": 12, "n": 12},
              str(rq["stats"]))
        check("10_11: tổng 7.0", abs(rq["score"] - 7.0) < 1e-9, str(rq["score"]))
        check("10_11: mã đề = 132", rq["made"] == "132", f"{rq['made']} ({rq['made_source']})")

    # ---------- 10. từ chối ảnh lạ ----------
    group("10. Từ chối ảnh không khớp mẫu")
    blank = np.full((700, 500, 3), 245, np.uint8)
    cv2.putText(blank, "KHONG PHAI PHIEU", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite("/tmp/_t_blank.jpg", blank)
    rbad = CTM.grade_one("/tmp/_t_blank.jpg", "12", TPL["12"], keys["12"])
    check("ảnh không phải phiếu -> None", rbad is None)

    # ---------- tổng kết ----------
    print("\n" + "=" * 60)
    print(f"KẾT QUẢ: {len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
