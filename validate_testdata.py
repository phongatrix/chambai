# -*- coding: utf-8 -*-
"""KIỂM TRA thuật toán chấm trên dữ liệu thật: test11 (12 ảnh, giấy 10_11) và
test12_20 (20 ảnh, giấy 12 — có kết quả đối chiếu từ đợt NA_3157 2026-11-03).

Chạy ĐÚNG luồng webapp: normalize_batch (song song) -> grade_prepared từng trang
(đúng tham số: todam + ids từ bước chuẩn hoá). In bảng kết quả + thống kê.
"""
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dell/Documents/chambaiV3")
import importlib.util

spec = importlib.util.spec_from_file_location("w", "/home/dell/Documents/chambaiV3/webapp.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
import cham_theo_mau as CTM

ROOT = Path("/home/dell/Documents/chambaiV3")
reg = w.load_registry()
KEY = w.merge_keys_grading(list(reg.keys()), reg)   # như khi user chọn cả 3 key

# Kết quả đối chiếu (ghi nhận 2026-11-05, regrade 20/20 trang norm_20261103_200436)
REF_SBD_12 = {"50120207", "50120359", "50120319", "50120262", "50120207",
              "50120219", "50120233", "50120294", "50120440", "50120268",
              "50120268", "50120348", "50120421", "50120239", "50120389",
              "50120222", "50120271", "50120376", "50120365", "50120410"}


def run_batch(folder, paper, tpl, label):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    print(f"\n{'=' * 72}\n{label}: {len(paths)} ảnh | paper={paper} | "
          f"mẫu={Path(tpl).name} | {len(KEY['các đề'])} đề trong key gộp: "
          f"{sorted(KEY['các đề'])}")
    t0 = time.time()
    okp, rows, inkp, extras = w.normalize_batch(paths, tpl, paper, KEY,
                                                f"/tmp/_val_{paper}", None,
                                                ink=True)
    t_prep = time.time() - t0
    n_ok = sum(1 for x in okp if x)
    print(f"⓪ Chuẩn hoá: {n_ok}/{len(paths)} OK trong {t_prep:.1f}s "
          f"({t_prep / max(1, len(paths)):.1f}s/ảnh)")
    sbds, mades, made_src, scores, fails = [], [], {}, [], []
    t0 = time.time()
    for k_, op in enumerate(okp):
        name = Path(paths[k_]).name
        if not op:
            print(f"  ✗ {name}: KHÔNG chuẩn hoá — {rows[k_]['ghi chú'][:60]}")
            fails.append(name)
            continue
        use = (inkp[k_] if (inkp[k_] and Path(inkp[k_]).is_file())
               else str(op))
        ex = extras[k_] or {}
        t1 = time.time()
        r = CTM.grade_prepared(use, paper, tpl, KEY, ids=ex.get("ids"),
                               made=None, quality=ex.get("quality", 1.0))
        dt = time.time() - t1
        if r is None:
            print(f"  ✗ {name}: grade_prepared trả None")
            fails.append(name)
            continue
        sbd = r.get("sbd")
        made = str(r["made"])
        s1 = r["stats"]["P1"]["ok"]; n1 = r["stats"]["P1"]["n"]
        s2 = r["stats"]["P2"]["ok"]; n2 = r["stats"]["P2"]["n"]
        print(f"  ✓ {name}: SBD={sbd or '—':<10} đề={made:<5} "
              f"({r['made_source']}) P1 {s1}/{n1} P2 {s2}/{n2} "
              f"tổng={r['score']:.2f} | {dt:.1f}s")
        if sbd:
            sbds.append(sbd)
        mades.append(made)
        made_src[r["made_source"]] = made_src.get(r["made_source"], 0) + 1
        scores.append(r["score"])
    t_grade = time.time() - t0
    print(f"── Chấm {len(mades)} trang trong {t_grade:.1f}s "
          f"({t_grade / max(1, len(mades)):.2f}s/trang)")
    print(f"   Nguồn mã đề: {made_src}")
    if sbds:
        dup = {s for s in sbds if sbds.count(s) > 1}
        print(f"   SBD: {len(set(sbds))} số khác nhau, trùng: "
              f"{sorted(dup) if dup else 'không'}")
    if paper == "12" and REF_SBD_12:
        got = set(sbds)
        match = len(got & {s for s in REF_SBD_12})
        print(f"   ĐỐI CHIẾU SBD với kết quả chuẩn 2026-11-03: "
              f"{match}/{len(got)} số trùng khớp")
    if mades:
        bad = [m for m in mades if m not in KEY["các đề"]]
        print(f"   Mã đề: {sorted(set(mades))} — "
              f"{'đều nằm trong key gộp' if not bad else 'LẠ (ngoài key): ' + str(bad)}")
    return n_ok, len(paths), mades, scores


n11 = run_batch("/home/dell/Documents/tulieuchambai/test11", "10_11",
                str(ROOT / "data/maugiay/10_11.png"), "BỘ TEST11")
n12 = run_batch("/home/dell/Documents/tulieuchambai/test12_20", "12",
                str(ROOT / "data/maugiay/12.png"), "BỘ TEST12_20")
print(f"\n{'=' * 72}\nTỔNG KẾT: test11 {n11[0]}/{n11[1]} | test12_20 {n12[0]}/{n12[1]}")
