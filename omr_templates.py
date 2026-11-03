# -*- coding: utf-8 -*-
"""
REGISTRY MẪU GIẤY + LƯỚI ĐÁP ÁN (omr_templates.py)

Kiến trúc: chọn mẫu giấy -> nạp đáp án -> dựng LƯỚI ĐÁP ÁN (mỗi ô = câu × lựa chọn,
gắn đáp án key và vị trí lattice) -> ảnh khớp lưới -> đúng/sai -> điểm.

Mỗi mẫu khai báo hình học tường minh (vùng, cấu trúc lattice, cách đánh số câu)
để pipeline xử lý theo mẫu đã chọn thay vì phỏng đoán.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATES_FILE = ROOT / "data/templates_registry.json"

# ---------- Định nghĩa các mẫu đã hiệu chuẩn trên dữ liệu thật ----------

TEMPLATE_GK1 = {
    "name": "QM_GK1",
    "tên_hiển_thị": "Phiếu QM — GK1 (30 câu ABCD cột trái + Đ/S 2 khối + số)",
    "grp": "g1011",
    "structure": {
        "P1": {"mô_tả": "30 câu, cột trái, 4 cột ABCD, pitch hàng ~63-66px@3x",
               "câu": 30, "mỗi_hàng": 1, "tùy_chọn": "ABCD"},
        "P2": {"mô_tả": "10 câu Đ/S, 2 khối × 5 câu, mỗi câu 4 ý (a-d) × Đúng/Sai",
               "câu": 10, "ý_mỗi_câu": 4, "cột_ý": "Đ,S"},
        "P3": {"mô_tả": "10 câu số, 2 khối × 5 câu, 4 cột chữ số", "câu": 10},
    },
    "vùng": {"P1": [0.0, 0.24, 0.30, 0.99], "P2": [0.24, 1.0, 0.21, 0.52],
             "P3": [0.24, 1.0, 0.52, 1.0]},
    "anchor": {"loại": "lưới_mã_đề", "offset_câu1_pitch": 23.15},
}

TEMPLATE_THPT = {
    "name": "THPT_40",
    "tên_hiển_thị": "Phiếu THPT (40 câu ABCD 4 khối + 8 câu Đ/S + 6 câu số)",
    "grp": "g12",
    "structure": {
        "P1": {"mô_tả": "40 câu, 4 khối × 10 hàng, mỗi ô = 4 cột ABCD",
               "câu": 40, "mapping": ["H1: câu=4×hàng+khối+1", "H2: câu=10×khối+hàng+1"],
               "tùy_chọn": "ABCD"},
        "P2": {"mô_tả": "8 câu Đ/S, 4 hàng ý × 16 cột (8 câu × Đ/S)", "câu": 8,
               "ý_mỗi_câu": 4, "cột_ý": "Đ,S"},
        "P3": {"mô_tả": "6 câu số", "câu": 6},
    },
    "vùng": {"P1": [0.0, 1.0, 0.32, 0.585], "P2": [0.0, 1.0, 0.585, 0.72],
             "P3": [0.0, 1.0, 0.72, 1.0]},
    "anchor": {"loại": "quét_H1H2_hàng_câu1_0_20", "ghi_chú": "key tự giải anchor"},
}

TEMPLATE_SMARTTEST_LANDSCAPE = {
    "name": "SMARTTEST_LS",
    "tên_hiển_thị": "Phiếu Smarttest ngang (ảnh điện thoại) — CẦN HIỆU CHUẨN",
    "grp": "g12",
    "structure": {"P1": {"mô_tả": "lưới trả lời ngang, pitch bubble ~15-20px@crop",
                         "câu": 28, "tùy_chọn": "ABCD"}},
    "vùng": None,
    "anchor": {"loại": "chưa_hiệu_chuẩn", "ghi_chú": "cần đo geometry trên 1 ảnh mẫu"},
}

TEMPLATES = {t["name"]: t for t in (TEMPLATE_GK1, TEMPLATE_THPT, TEMPLATE_SMARTTEST_LANDSCAPE)}


def list_templates():
    return {name: t["tên_hiển_thị"] for name, t in TEMPLATES.items()}


def get_template(name):
    return TEMPLATES.get(name)


# ---------- DỰNG LƯỚI ĐÁP ÁN ----------

def build_answer_grid(template, key):
    """Lưới đáp án: mỗi ô = (câu, lựa chọn, đáp án key, phần, chỉ số lattice).

    Trả {"mẫu": name, "số_ô": n, "câu": {câu: {"tùy_chọn": ..., "đáp án": ...}},
    "ô": [...]} — JSON-serializable.
    """
    name = template["name"]
    grp = template["grp"]
    if key.get("đề") != "QM":
        return {"mẫu": name, "lỗi": "key phải định dạng QM/Smarttest (mỗi đề 1 dòng)"}
    ô = []
    câu_map = {}
    for de, dk in key["các đề"].items():
        p1k, p2k, p3k = dk.get("p1", {}), dk.get("p2", {}), dk.get("p3", {})
        # P1
        for cq in sorted(p1k):
            opts = "ABCD" if grp in ("g1011", "g12") else "ABCD"
            row = {
                "câu": str(cq), "phần": "P1", "loại": "ABCD",
                "đáp án": p1k[cq],
                "vị trí": (f"P1 hàng {cq} cột trái" if grp == "g1011"
                           else f"P1 (mapping H1/H2) câu {cq}"),
                "đề": de,
            }
            ô.append(row)
            câu_map.setdefault(str(cq), row)
        # P2
        for (cq, yk), val in sorted(p2k.items()):
            ô.append({
                "câu": f"{cq}{yk}", "phần": "P2", "loại": "Đ/S",
                "đáp án": val,
                "vị trí": (f"P2 khối {0 if cq <= 5 else 1}, ý {yk}, cặp cột {cq}"
                           if grp == "g1011" else f"P2 hàng ý {yk}, cột Đ/S câu {cq}"),
                "đề": de,
            })
            câu_map.setdefault(f"{cq}{yk}", ô[-1])
        # P3
        for cq in sorted(p3k):
            ô.append({"câu": str(cq), "phần": "P3", "loại": "SỐ",
                      "đáp án": str(p3k[cq]), "vị trí": f"P3 câu {cq}", "đề": de})
            câu_map.setdefault(str(cq), ô[-1])
    n_câu = len({r["câu"] for r in ô})
    return {"mẫu": name, "grp": grp, "số_đề": len(key["các đề"]),
            "số_câu": n_câu, "số_ô": len(ô), "câu": câu_map, "ô": ô}


def save_grid(grid, path):
    Path(path).write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    return Path(path)


def load_templates_registry():
    """Registry mẫu do người dùng thêm (lưu data/templates_registry.json)."""
    if TEMPLATES_FILE.is_file():
        try:
            user = json.loads(TEMPLATES_FILE.read_text(encoding="utf-8"))
            for name, t in user.items():
                TEMPLATES.setdefault(name, t)
        except Exception:
            pass
    return TEMPLATES


if __name__ == "__main__":
    import sys
    from omr_key import detect_key_format
    tname = sys.argv[1] if len(sys.argv) > 1 else "QM_GK1"
    kpath = sys.argv[2] if len(sys.argv) > 2 else "data/keys_builtin.json"
    t = get_template(tname)
    if not t:
        print(f"Không có mẫu {tname}; các mẫu: {list(TEMPLATES)}")
        sys.exit(2)
    if kpath.endswith(".json"):
        key = json.loads(Path(kpath).read_text(encoding="utf-8"))
    else:
        _, key, _ = detect_key_format(kpath)
    grid = build_answer_grid(t, key)
    print(f"Template {tname}: {grid.get('số_đề', '-')} đề × {grid.get('số_câu', '-')} câu "
          f"= {grid.get('số_ô', '-')} ô đáp án")
    for r in grid["ô"][:6]:
        print("  ", r["câu"], r["phần"], r["đáp án"], "|", r["vị trí"])
    out = save_grid(grid, f"/tmp/grid_{tname}.json")
    print(f"Đã lưu {out}")
