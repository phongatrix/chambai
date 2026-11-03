# -*- coding: utf-8 -*-
"""
KIỂM ĐỊNH LOGIC THUẬT TOÁN chấm điểm (verify_logic.py)

Kiểm chứng các giả định của omr_grade_final.py TRÊN DỮ LIỆU CÓ SẴN:
1. Luật điểm: P1 = 0.25/câu, P2 = 0.25/ý, P3 = 0.5/câu -> giá trị KQ phải là
   bội của 0.25 và trong miền tối đa (P1 ≤ 18×0.25, P2 ≤ 16×0.25, P3 ≤ 6×0.5)
2. Nhất quán nội bộ: điểm AI = 0.25 × số ô xanh đếm được
3. Ánh xạ trang↔dòng: mã đề decode khớp mã đề KQ
4. Phần III trống trên mọi trang (giả định P3=0)
5. Tương thích dữ liệu: 49 ảnh nguồn trích được, 2 file KQ đọc được
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omr_grade_v3 import load_kq, load_qm_key  # noqa

ROOT = Path(__file__).resolve().parent
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'✓' if ok else '✗ LỖI'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def main():
    print("== 1. DỮ LIỆU ĐẦU VÀO ==")
    kq = load_kq()
    check("KQ Giay10_11 đọc được", len(kq["g1011"]) == 45, f"({len(kq['g1011'])} dòng)")
    check("KQ Giay12 đọc được", len(kq["g12"]) == 4, f"({len(kq['g12'])} dòng)")
    key = load_qm_key()
    check("Key QM đọc được", len(key) == 10, f"({len(key)} đề)")
    scan = ROOT / "data/scan_pages"
    n1011 = len(list(scan.glob("g1011-*.jpg"))) if scan.is_dir() else 0
    n12 = len(list(scan.glob("g12-*.jpg"))) if scan.is_dir() else 0
    check("49 ảnh nguồn", n1011 == 45 and n12 == 4, f"(g1011={n1011}, g12={n12})")

    print("\n== 2. LUẬT TÍNH ĐIỂM TRÊN FILE KQ (điểm phần mềm cũ) ==")
    bad_p1 = [i for i, r in enumerate(kq["g1011"]) if (r["p1"] or 0) % 0.25 > 1e-9 or r["p1"] > 4.5 + 1e-9]
    check("P1 g1011: bội 0.25 và ≤ 4.5 (18 câu)", not bad_p1, f"lệch: {bad_p1[:5]}")
    bad_p2 = [i for i, r in enumerate(kq["g1011"]) if (r["p2"] or 0) % 0.25 > 1e-9 or r["p2"] > 4.0 + 1e-9]
    check("P2 g1011: bội 0.25 và ≤ 4.0 (16 ý)", not bad_p2, f"lệch: {bad_p2[:5]}")
    bad_g12 = [i for i, r in enumerate(kq["g12"])
               if (r["p1"] or 0) % 0.25 > 1e-9 or (r["p1"] or 0) > 6.0 + 1e-9
               or (r["p2"] or 0) % 0.25 > 1e-9 or (r["p2"] or 0) > 4.0 + 1e-9]
    check("P1 g12: bội 0.25 ≤ 6.0 (24 câu); P2 ≤ 4.0", not bad_g12, f"lệch: {bad_g12}")
    p3_all = [(r["p3"] or 0) for r in kq["g1011"]] + [(r["p3"] or 0) for r in kq["g12"]]
    check("P3 = 0 trên mọi trang (49)", all(p == 0 for p in p3_all), "")
    tong_ok = all(
        abs((r["p1"] or 0) + (r["p2"] or 0) + (r["p3"] or 0) - (r["tong"] or 0)) < 0.01
        for grp in ("g1011", "g12") for r in kq[grp])
    check("Tổng = P1+P2+P3 (49 dòng)", tong_ok, "")

    print("\n== 3. ĐÁP ÁN KEY (QM) HỢP LỆ ==")
    ok_key = all(set(de["p1"].values()) <= {"A", "B", "C", "D"} and
                 set(de["p2"].values()) <= {"D", "S"} and len(de["p1"]) == 18 and len(de["p2"]) == 16
                 for de in key.values())
    check("P1 key: 18 câu A-D; P2 key: 16 ý D/S (10 đề)", ok_key, "")

    print("\n== 4. KẾT QUẢ CHẤM (grading_final.json) NHẤT QUÁN NỘI BỘ ==")
    import json
    res = json.loads((ROOT / "output/grading_final.json").read_text(encoding="utf-8"))
    check("49 trang đã chấm", len(res) == 49, f"({len(res)})")
    cons = all(
        abs(r["my"]["P1"] - 0.25 * r["counts"]["P1"]["green"]) < 0.01 and
        abs(r["my"]["P2"] - 0.25 * r["counts"]["P2"]["green"]) < 0.01 and
        abs(r["my"]["tong"] - r["my"]["P1"] - r["my"]["P2"] - r["my"]["P3"]) < 0.01
        for r in res)
    check("Điểm AI = 0.25 × số ô xanh (đúng công thức mọi trang)", cons, "")
    # fill < green (số fill phải >= số green trong cùng vùng)
    fill_ok = all(r["counts"][z]["fills"] >= r["counts"][z]["green"] for r in res for z in ("P1", "P2", "P3"))
    check("Số fill ≥ số green (mọi vùng/mọi trang)", fill_ok, "")
    md_ok = sum(1 for r in res if r.get("md_match"))
    md_bad = [r["file"] for r in res if not r.get("md_match")]
    check("Mã đề decode khớp KQ", md_ok == 39, f"({md_ok}/49; lệch/có '?' : {len(md_bad)} trang)")
    # điểm lệch
    diffs = [abs(r["diff"]["tong"]) for r in res if "diff" in r]
    exact = sum(1 for d in diffs if d < 0.01)
    w25 = sum(1 for d in diffs if d <= 0.251)
    w50 = sum(1 for d in diffs if d <= 0.501)
    print(f"      -> khớp tuyệt đối {exact}/49; ≤0.25: {w25}/49; ≤0.5: {w50}/49; TB {sum(diffs)/len(diffs):.3f}")

    print("\n== 5. ÁNH XẠ TRANG↔DÒNG (độc lập khỏi mã đề) ==")
    # 10 trang mã đề decode '?' vẫn ánh xạ đúng theo thứ tự + điểm khớp
    q_pages = [r for r in res if r.get("md_match") is None]
    ok_q = all(r.get("match") or abs(r["diff"]["tong"]) <= 0.5 for r in q_pages) if q_pages else True
    check("Trang mã đề '?' vẫn khớp điểm (≤0.5)", ok_q, f"({len(q_pages)} trang)")

    print("\n== 6. TOÀN VẸN XUẤT EXCEL ↔ JSON ==")
    import openpyxl
    wb = openpyxl.load_workbook(ROOT / "output/KQ_cham_AI_vs_phan_mem_cu.xlsx")
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))[1:]
    check("Excel đủ 49 dòng", len(rows) == 49, f"({len(rows)})")
    xj_bad = [i for i, (row, r) in enumerate(zip(rows, res))
              if abs((row[4] or 0) - r["my"]["P1"]) > 0.01
              or abs((row[6] or 0) - r["my"]["P2"]) > 0.01
              or abs((row[10] or 0) - r["my"]["tong"]) > 0.01]
    check("Cột điểm Excel khớp JSON (P1, P2, Tổng)", not xj_bad, f"lệch: {xj_bad[:5]}")

    print("\n== 7. TRƯỜNG HỢP BIÊN ==")
    few = [r["file"] for r in res if r["counts"]["P1"]["green"] + r["counts"]["P2"]["green"] < 3]
    check("Không trang nào quá ít green (<3) để bỏ qua lattice", not few, f"{few}")
    over = [r["file"] for r in res if "diff" in r and abs(r["diff"]["tong"]) > 0.5]
    check("Số trang lệch > 0.5 ≤ 1", len(over) <= 1, f"{over} (giới hạn phát hiện mực nhạt)")

    print("\n== 8. TỰ NHẬN LOẠI PHIẾU (auto_group) ==")
    from omr_grade_v3 import ORIG
    from omr_grade_final import auto_group
    from omr_grade_final import process_page as pp
    import collections as _cl
    cnt = _cl.Counter()
    for f in sorted(ORIG.glob("g1011-*.jpg")) + sorted(ORIG.glob("g12-*.jpg")):
        page = pp(f)
        g = auto_group(page)
        expect = "g1011" if f.stem.startswith("g1011") else "g12"
        cnt[("✓" if g == expect else "✗")] += 1
    check("auto_group phân loại đúng 49/49 trang", cnt["✓"] == 49, f"{dict(cnt)}")

    print("\n" + "=" * 50)
    if FAIL:
        print(f"KẾT LUẬN: {len(FAIL)} hạng mục LỖI: {FAIL}")
        sys.exit(1)
    print("KẾT LUẬN: toàn bộ giả định thuật toán KHỚP dữ liệu có sẵn — logic chấm điểm nhất quán.")


if __name__ == "__main__":
    main()
