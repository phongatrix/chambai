#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHẤM THEO MẪU GIẤY (cham_theo_mau.py)
====================================
chọn mẫu giấy -> nạp đáp án -> dựng lưới -> ảnh khớp mẫu bằng ô đen định vị
-> đối chiếu bubble -> ĐÚNG(xanh)/SAI(đỏ)/TRỐNG-NHIỀU(vàng) -> điểm.

Ví dụ:
  .venv/bin/python cham_theo_mau.py \
      --template 12 --key /home/dell/Documents/tulieuchambai/dapan/k12dapan.xlsx \
      --images /home/dell/Documents/tulieuchambai/Test --out output/theo_mau

  # tự chọn mẫu (12.png hoặc 10_11.png) cho từng ảnh:
  .venv/bin/python cham_theo_mau.py --auto --out output/theo_mau

Đầu ra:
  <out>/<SBD>_<Mã đề>_<YYYYMMDD>_<HHMMSS>.jpg   ảnh kết quả 3 màu
  <out>/ket_qua.xlsx                             tổng hợp + chi tiết từng câu
  <out>/ket_qua.json                             dữ liệu thô (nếu --json)
  + tự lưu vào CSDL SQLite (trừ khi --no-db)
"""
import argparse
import json
import sys
import re
from pathlib import Path

import cv2
import numpy as np

import omr_sheetgrid as S
import omr_idscan as ID
import omr_prep as PREP
from omr_key import detect_key_format

ROOT = Path(__file__).resolve().parent
# Mẫu giấy + đáp án nằm TRONG thư mục project (data/maugiay, data/dapan)
# -> project mang đi đâu cũng chạy, không phụ thuộc đường dẫn tuyệt đối.
ROOT_TPL = Path(__file__).resolve().parent / "data"
DEFAULT_TPLS = {
    "12": ROOT_TPL / "maugiay" / "12.png",
    "10_11": ROOT_TPL / "maugiay" / "10_11.png",
}
DEFAULT_KEYS = {
    "12": ROOT_TPL / "dapan" / "k12dapan.xlsx",
    "10_11": ROOT_TPL / "dapan" / "k11dapan.xlsx",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STATE_VI = {"ok": "đúng", "wrong": "sai", "blank": "trống", "multi": "tô nhiều"}


# ---------- key ----------

def load_key(path):
    fmt, key, desc = detect_key_format(str(path))
    return key, fmt, desc


def keyed_cells(paper, tpl_path, key, de=None):
    """Chỉ các ô CÓ đáp án trong key (đủ lựa chọn của mỗi câu).

    de=None: lấy HỢP các ô của TẤT CẢ đề — khi gộp nhiều key/đề có số câu khác
    nhau, không bỏ sót ô của đề dài hơn."""
    if de is None:
        seen, out = set(), []
        for d_ in key["các đề"].values():
            grid = S.build_answer_grid(paper, tpl_path, d_)
            for c in grid["cells"]:
                if grid["answers"].get((c[0], c[1], c[2])) and c not in seen:
                    seen.add(c)
                    out.append(c)
        return out
    grid = S.build_answer_grid(paper, tpl_path, key["các đề"][de])
    return [c for c in grid["cells"] if grid["answers"].get((c[0], c[1], c[2]))]


# ---------- chấm 1 ảnh ----------

def _md_read_strong(ids_detail):
    """MÃ ĐỀ đọc từ lưới có ĐỦ CHẮC KHÔNG (ids_detail = ids['score']).

    'Chắc' = MỌI cột của lưới Mã đề đều chọn ra chữ số (không '?') với mực thô
    >= 0.55, vượt nền in >= 0.30 và gấp >= 2.5 lần số nhì. Dưới các ngưỡng này
    (vd raw 0.48 / s1 0.23 / ratio 2.4) cách đọc coi là MƠ HỒ — chấm điểm mới
    được phép phủ quyết. Ngưỡng hiệu chỉnh trên test12_20: trang 195843 (3157
    chắc chắn: raw 0.87-1.0, ratio 3.7-33) phải PASS, trang 200014 (4185 mơ
    hồ: raw 0.48-0.84, cột 1 ratio 2.4) phải FAIL."""
    md = (ids_detail or {}).get("MD")
    if not isinstance(md, list) or not md:
        return False
    try:
        return all(v[1] >= 0 and v[2] >= 0.55 and v[3] >= 0.30
                   and v[4] >= 2.5 for v in md)
    except Exception:                                        # noqa
        return False


def _finish_grade(ph, H, cells, key, paper, made, thresh, ids, q,
                  photo_path, pre, tpl_path):
    """Phần dùng chung: đọc ô -> quyết định -> chọn đề -> trả kết quả."""
    reads = S.read_cells(ph, cells, H)
    dec = S.decide(reads, None, thresh=thresh)
    # --- SBD / Mã đề: warp về hệ toạ độ mẫu rồi đọc bằng omr_idscan ---
    # (matched-filter vành khuyên + tinh chỉnh lưới theo vòng in thật +
    #  chuẩn hoá nền mực in theo hàng + ràng buộc mã đề theo key)
    ids_full = ids
    sbd = (ids or {}).get("SBD")
    md_grid = (ids or {}).get("MD")
    ids_detail = (ids or {}).get("score", {})

    # ---- SBD của mẫu 12: nếu không giải được chuỗi số thì vẫn khoanh các ô
    # đậm nhất mỗi cột (chỉ để HIỂN THỊ, không dùng để chấm/đặt tên) ----
    try:
        pos_sbd = (ids_full or {}).get("positions", {}).get("SBD")
        if paper == "12" and not pos_sbd:
            fp = ID.fallback_positions(ph, paper, tpl_path, "SBD",
                                       scale=float(np.asarray(H)[0, 0]))
            if fp:
                ids_full = dict(ids_full) if ids_full else {}
                pp = dict(ids_full.get("positions") or {})
                pp["SBD"] = fp
                ids_full["positions"] = pp
                ids_full["sbd_partial"] = True
                # BUILD CHUỖI SBD từ các cột tự tin (2026-11-08 — user báo:
                # "nhận dạng SBD chính xác nhưng không hiển thị SBD trên hình
                # kết quả" — fallback_positions chỉ VẼ vòng mà không gán
                # chuỗi). Đủ MỌI cột mới gán (có cột lệch → '?', nhường đường
                # CNN viết tay); sbd_source ghi rõ "bubble (dự phòng)".
                _cols = {}
                for _c in fp:
                    _cols[int(_c[0])] = str(_c[1])
                if _cols:
                    _n = max(_cols) + 1
                    _s = "".join(_cols.get(i, "?") for i in range(_n))
                    if "?" not in _s and len(_s) >= 4:
                        sbd = _s
                        ids_full["sbd_fallback"] = True
    except Exception:                                            # noqa
        pass

    # --- chọn đề: ưu tiên mã đề chỉ định / mã đề đọc từ lưới (nếu có trong key),
    #     sau đó tới đề cho kết quả khớp nhất ---
    des = list(key["các đề"].keys())
    scored = {}
    for de in des:
        grid = S.build_answer_grid(paper, tpl_path, key["các đề"][de])
        s1, s2, detail, stats = S.grade(dec, grid)
        scored[de] = (s1 + s2, s1, s2, detail, stats, grid)
    best_de = max(scored, key=lambda d: (scored[d][4]["P1"]["ok"] + scored[d][4]["P2"]["ok"],
                                         scored[d][0]))
    use_de = best_de
    md_source = "điểm"
    if made and made in key["các đề"]:
        use_de, md_source = made, "chỉ định"
    elif md_grid in key["các đề"]:
        # xác nhận chéo: mã đề từ lưới phải có kết quả gần với đề khớp nhất —
        # TRỪ KHI lưới đọc RÕ (mỗi chữ số đủ mực và cách xa số nhì): khi đó
        # TIN LƯỚI, vì "đề khớp điểm nhất" là nhiễu với học sinh bỏ trống nhiều
        # (vd trang 195843: lưới đọc chắc 3157 raw 0.87-1.0 / ratio 3.7-33 nhưng
        # điểm khớp nhất rơi vào đề 132 chỉ vì bài làm ít câu).
        if _md_read_strong(ids_detail) or \
                scored[md_grid][0] >= scored[best_de][0] - 0.75:
            use_de, md_source = md_grid, "lưới"
    tot, s1, s2, detail, stats, grid = scored[use_de]

    # --- BỔ SUNG MÃ ĐỀ ĐÃ DÙNG: tạo vị trí để vẽ vòng lên lưới Mã đề của phiếu
    # kể cả khi lưới không đọc được (đề suy từ đáp án). Nhờ vậy ảnh kết quả luôn
    # thể hiện rõ bài này chấm theo mã đề nào.
    try:
        idc = [c for c in S.build_id_cells(paper, tpl_path) if c[0].upper() == "MD"]
        if idc and use_de:
            cols = sorted({c[1] for c in idc})
            xs = [c[3] for c in idc]
            pitch = float(np.median(np.diff(sorted(set(round(v, 3) for v in xs))))) \
                if len(set(round(v, 3) for v in xs)) > 1 else 8.0
            rad = max(3.0, 0.55 * pitch)
            pos = []
            for i, ci in enumerate(cols):
                if i >= len(str(use_de)):
                    break
                d = str(use_de)[i]
                for (pp_, cc_, dd_, x, y) in idc:
                    if cc_ == ci and str(dd_) == d:
                        q_ = cv2.perspectiveTransform(
                            np.float32([[[x, y]]]), np.asarray(H, np.float64))[0][0]
                        pos.append((ci, d, float(q_[0]), float(q_[1]), rad))
                        break
            if pos:
                ids_full = dict(ids_full) if ids_full else {}
                pp = dict(ids_full.get("positions") or {})
                pp["MD"] = pos
                ids_full["positions"] = pp
    except Exception:                                            # noqa
        pass

    # --- CẢNH BÁO: ô đã tô nhưng KHÔNG nằm trong đáp án (ngoài phạm vi key) ---
    # Ví dụ phiếu in 20 câu PHẦN I nhưng key chỉ có 16 -> 4 câu bị bỏ khỏi điểm.
    unkeyed = []
    try:
        all_cells = S.build_template_cells(paper, tpl_path)
        keyed = {(c[0], c[1], c[2]) for c in cells}
        rest = [c for c in all_cells if (c[0], c[1], c[2]) not in keyed]
        if rest:
            reads2 = S.read_cells(ph, rest, H)
            th2 = max(S.adapt_thresh(reads2), 0.22)
            for (part_, cau_, opt_), v in reads2.items():
                if float(v[0]) >= th2:
                    unkeyed.append((part_, cau_, opt_, round(float(v[0]), 2)))
    except Exception:                                          # noqa
        pass

    # --- SBD dự phòng bằng CHỮ SỐ VIẾT TAY (CNN) khi lưới bubble không đọc
    #     được chuỗi số: bubble trước, viết tay sau (chỉ mẫu 12, ảnh đã warp) ---
    written = None
    if paper == "12" and not sbd:
        try:
            Hm = np.asarray(H, np.float64)
            sc = float(Hm[0, 0])
            # ô viết tay chỉ khớp khi H là thuần tỉ lệ (ảnh đã warp hệ mẫu)
            ok_h = (abs(Hm[0, 1]) + abs(Hm[1, 0])
                    + abs(Hm[0, 2]) + abs(Hm[1, 2])) < 1e-6
            tpl_ck = cv2.imread(str(tpl_path))
            ok_size = (tpl_ck is not None
                       and abs(ph.shape[1] - sc * tpl_ck.shape[1])
                       <= max(6.0, 0.02 * ph.shape[1]))
            if ok_h and ok_size:
                import omr_ocr as O
                if O._get_cnn() is not None:
                    written = O.read_written_ids(ph, tpl_path, paper,
                                                 scale=sc, force=True)
                    s_w = (written or {}).get("SBD")
                    if s_w and len(s_w) >= 4:
                        sbd = s_w
                        ids_full = dict(ids_full) if ids_full else {}
                        ids_full["sbd_written"] = True
                    else:
                        written = None
        except Exception:                                        # noqa
            written = None
    sbd_source = ("viết tay" if (ids_full or {}).get("sbd_written")
                  else ("bubble (dự phòng)" if (ids_full or {}).get("sbd_fallback")
                        else ("bubble" if sbd else "không có")))
    return {"photo": str(photo_path), "paper": paper, "made": use_de,
            "sbd_source": sbd_source,
            "pre_norm": None if pre is None else {"ok": pre["ok"],
                                                  "quality": pre["quality"],
                                                  "rot": pre["rot"],
                                                  "reason": pre["reason"]},
            "made_source": md_source, "made_grid": md_grid, "sbd": sbd,
            "ids_detail": ids_detail, "ids": ids_full,
            "quality": round(float(q), 3), "score": round(tot, 2),
            "p1": s1, "p2": s2, "stats": stats, "detail": detail,
            "img": ph, "decided": dec, "grid": grid,
            "unkeyed_filled": sorted(unkeyed)}


def grade_prepared(img_path, paper, tpl_path, key, ids=None, made=None,
                   thresh=None, quality=1.0):
    """CHẤM TRÊN ẢNH ĐÃ CHUẨN HOÁ của bước ⓪ — không căn lại, không đọc ID lại.

    Ảnh vào có kích thước = mẫu × scale nên chỉ cần H = scale·I; SBD/Mã đề lấy
    từ `ids` do bước ⓪ đã đọc (tránh chạy lại best_orientation 4 hướng — đây là
    nguyên nhân làm bước chấm chậm gấp đôi).
    """
    cells = keyed_cells(paper, tpl_path, key, made)
    if not cells:
        return None
    ph = cv2.imread(str(img_path))
    tpl_img = cv2.imread(str(tpl_path))
    if ph is None or tpl_img is None:
        return None
    scale = ph.shape[1] / float(tpl_img.shape[1])
    H = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    pre = {"ok": True, "quality": float(quality), "rot": 0, "reason": ""}
    return _finish_grade(ph, H, cells, key, paper, made, thresh, ids,
                         float(quality), img_path, pre, tpl_path)


def grade_one(photo_path, paper, tpl_path, key, made=None, min_quality=0.70,
              thresh=None, decode_ids=True, normalize=False, norm_scale=None,
              ink_only_input=False):
    """Chấm 1 ảnh với 1 mẫu + 1 bộ key. Trả dict kết quả hoặc None nếu không khớp mẫu.

    normalize=True: chạy BƯỚC TIỀN XỬ LÝ trước — tự xoay 4 hướng, căn mốc, warp
    ảnh về đúng khung mẫu rồi mới chấm (xem omr_prep.normalize_to_template).
    norm_scale=None: tự chọn theo mẫu (omr_prep.norm_scale_for — moi cần 3.0
    vì bubble nhỏ, sample cũ 2.0).
    ink_only_input=True: ảnh vào ĐÃ là bản chuẩn hoá của bước ⓪ -> dùng H=scale·I.
    """
    cells = keyed_cells(paper, tpl_path, key, made)
    if not cells:
        return None
    ref = S.build_template_cells(paper, tpl_path)
    pre = None
    H = ph = None
    q = 0.0
    ids = None
    if normalize and ink_only_input:
        ph = cv2.imread(str(photo_path))
        if ph is None:
            return None
        tpl_img = cv2.imread(str(tpl_path))
        norm_scale = ph.shape[1] / float(tpl_img.shape[1])
        H = np.array([[norm_scale, 0, 0], [0, norm_scale, 0],
                      [0, 0, 1]], np.float64)
        q = 1.0
        pre = {"ok": True, "quality": 1.0, "rot": 0, "reason": ""}
    elif normalize:
        pre = PREP.normalize_to_template(
            photo_path, tpl_path,
            scale=norm_scale or PREP.norm_scale_for(paper),
            cells=cells, ink=True)
        if pre.get("ok"):
            ph = pre["warped"]
            ns = norm_scale or PREP.norm_scale_for(paper)
            H = np.array([[ns, 0, 0], [0, ns, 0], [0, 0, 1]], np.float64)
            q = float(pre["quality"])
            ids = pre.get("ids_result")
    if H is None:
        H, ph, q = S.align_photo(photo_path, tpl_path, cells, ref_cells=ref)
    if H is None or q < min_quality:
        return None
    if decode_ids and ids is None:
        try:
            ids = ID.scan_ids(ph, H, paper, tpl_path,
                              md_candidates=list(key["các đề"].keys()))
        except Exception:                                            # noqa
            ids = None
    return _finish_grade(ph, H, cells, key, paper, made, thresh, ids, q,
                         photo_path, pre, tpl_path)


# ---------- xuất file ----------

def _safe(s, default="NA"):
    s = str(s or "").strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
    return s if s else default


def output_name(photo_path, r):
    """<SBD>_<Mã đề>_<ngày quét>_<giờ quét> (EXIF, fallback mtime)."""
    from omr_overlay import _scan_datetime
    ngay, gio = _scan_datetime(photo_path)
    return f"{_safe(r.get('sbd'))}_{_safe(r.get('made'))}_{ngay}_{gio}"


def output_name_now(r, ts=None):
    """<SBD>_<Mã đề>_<YYYYMMDD>_<HHMMSS> — THỜI GIAN HIỆN HÀNH (có giây).

    Dùng cho luồng chấm ảnh đã chuẩn hoá: tên phản ánh LÚC CHẤM, không phụ
    thuộc EXIF/mtime của ảnh gốc (ảnh lấy từ Drive thường mất EXIF).
    """
    from datetime import datetime
    ts = ts or datetime.now()
    return (f"{_safe(r.get('sbd'))}_{_safe(r.get('made'))}_"
            f"{ts.strftime('%Y%m%d')}_{ts.strftime('%H%M%S')}")


def write_excel(rows, details, path):
    """2 sheet: Tổng hợp + Chi tiết từng câu."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Tổng hợp"
    ws.append(["File", "Mẫu", "SBD", "Mã đề", "Nguồn mã đề", "Khớp mẫu",
               "P1 đúng", "P1 số câu", "Điểm P1", "P2 đúng", "P2 số câu", "Điểm P2",
               "Tổng điểm", "Số sai", "Trống/nhiều"])
    for r in rows:
        ws.append([r["file"], r["paper"], r.get("sbd") or "", r["made"], r.get("made_source", ""),
                   r["quality"], r["stats"]["P1"]["ok"], r["stats"]["P1"]["n"], r["p1"],
                   r["stats"]["P2"]["ok"], r["stats"]["P2"]["n"], r["p2"], r["score"],
                   r.get("n_wrong", 0), r.get("n_unk", 0)])
    ws2 = wb.create_sheet("Chi tiết")
    ws2.append(["File", "Mã đề", "Phần", "Câu", "Trạng thái", "Đáp án", "Bài làm"])
    for r in details:
        for (part, q), (st, exp, got) in sorted(r["detail"].items(),
                                                key=lambda z: (z[0][0], str(z[0][1]))):
            ws2.append([r["file"], r["made"], part,
                        str(q) if not isinstance(q, tuple) else f"{q[0]}{q[1]}",
                        STATE_VI.get(st, st), exp, got])
    for sheet in (ws, ws2):
        for col in sheet.columns:
            w = max(len(str(c.value or "")) for c in col)
            sheet.column_dimensions[col[0].column_letter].width = min(42, max(9, w + 2))
    wb.save(str(path))
    return path


def save_db(rows, label):
    """Lưu lịch sử vào CSDL SQLite (không làm hỏng chấm nếu lỗi)."""
    try:
        from omr_db import save_run
        results = []
        for r in rows:
            results.append({"file": r["file"], "grp": r["paper"], "md": r["made"],
                            "sbd": r.get("sbd"), "match": r["quality"] >= 0.9,
                            "my": {"P1": r["p1"], "P2": r["p2"], "P3": 0.0,
                                   "tong": r["score"]},
                            "counts": {"P1": r["stats"]["P1"], "P2": r["stats"]["P2"]}})
        return save_run(label, results, notes="chấm theo mẫu giấy (fiducial)")
    except Exception as e:  # noqa
        print(f"[db] bỏ qua: {e}")
        return None


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Chấm bài theo mẫu giấy + lưới đáp án")
    ap.add_argument("--template", choices=list(DEFAULT_TPLS), default=None)
    ap.add_argument("--key", default=None, help="File đáp án (.xlsx)")
    ap.add_argument("--images", default="/home/dell/Documents/tulieuchambai/Test")
    ap.add_argument("--out", default="output/theo_mau")
    ap.add_argument("--made", default=None, help="Mã đề cố định (mặc định: tự chọn)")
    ap.add_argument("--auto", action="store_true", help="Tự chọn mẫu cho từng ảnh")
    ap.add_argument("--json", default=None, help="Xuất JSON kết quả")
    ap.add_argument("--no-excel", action="store_true")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--normalize", action="store_true",
                    help="Tiền xử lý: tự xoay 4 hướng + chuẩn hoá ảnh về đúng mẫu "
                         "trước khi chấm (khuyến nghị cho ảnh điện thoại)")
    args = ap.parse_args()

    if args.template:
        papers = [args.template]
        keys = {args.template: Path(args.key) if args.key else DEFAULT_KEYS[args.template]}
    else:
        papers = list(DEFAULT_TPLS)
        keys = {p: (Path(args.key) if args.key else DEFAULT_KEYS[p]) for p in papers}
    key_cache = {}
    for p in papers:
        key, fmt, desc = load_key(keys[p])
        key_cache[p] = key
        print(f"[key] {p}: {keys[p].name} — {desc}")

    src = Path(args.images)
    if src.is_dir():
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXT])
    elif src.is_file():
        imgs = [src]
    else:
        print(f"Không thấy ảnh: {src}")
        return 2
    if not imgs:
        print("Không có ảnh.")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, details, used_names, skipped = [], [], set(), []
    print(f"\nChấm {len(imgs)} ảnh...\n")
    for i, ip in enumerate(imgs, 1):
        best = None
        for paper in papers:
            r = grade_one(ip, paper, DEFAULT_TPLS[paper], key_cache[paper],
                          made=args.made, normalize=args.normalize)
            if r is None:
                continue
            cand = (r["stats"]["P1"]["ok"] + r["stats"]["P2"]["ok"], r["score"], r["quality"])
            if best is None or cand > best[0]:
                best = (cand, r)
        if best is None:
            print(f"[{i:2d}/{len(imgs)}] {ip.name}: KHÔNG khớp mẫu (bỏ qua)")
            skipped.append(ip.name)
            continue
        r = best[1]
        # chấm ảnh đã chuẩn hoá -> tên theo THỜI GIAN HIỆN HÀNH (có giây)
        stem = output_name_now(r) if args.normalize else output_name(ip, r)
        outp = out_dir / f"{stem}.jpg"
        # đổi tên CẢ stem (như webapp): tránh vòng lặp vô hạn khi trùng tên
        base_stem, n = stem, 1
        while stem in used_names or outp.exists():
            stem = f"{base_stem}_{n}"
            outp = out_dir / f"{stem}.jpg"
            n += 1
        used_names.add(stem)
        S.render_result(r["img"], r["decided"], r["detail"], r["stats"], outp,
                        r["paper"], made=r["made"], s1=r["p1"], s2=r["p2"],
                        sbd=r.get("sbd"), ids=r.get("ids"), show_sbd=True,
                        md_inferred=(r.get("made_source") != "lưới"),
                        md_note="", tpl_path=tpl_path)
        n_wrong = sum(1 for v in r["detail"].values() if v[0] == "wrong")
        n_unk = sum(1 for v in r["detail"].values() if v[0] in ("blank", "multi"))
        rows.append({"file": ip.name, "paper": r["paper"], "sbd": r.get("sbd"),
                     "made": r["made"], "made_source": r["made_source"],
                     "quality": r["quality"], "score": r["score"],
                     "p1": r["p1"], "p2": r["p2"], "stats": r["stats"],
                     "n_wrong": n_wrong, "n_unk": n_unk})
        details.append({"file": ip.name, "made": r["made"], "detail": r["detail"]})
        print(f"[{i:2d}/{len(imgs)}] {ip.name}: mẫu {r['paper']} | SBD {r.get('sbd') or '-'} | "
              f"đề {r['made']} ({r['made_source']}) | khớp {r['quality']:.2f} | "
              f"P1 {r['stats']['P1']['ok']}/{r['stats']['P1']['n']} "
              f"P2 {r['stats']['P2']['ok']}/{r['stats']['P2']['n']} | tổng {r['score']}")

    if rows:
        if not args.no_excel:
            xp = out_dir / "ket_qua.xlsx"
            write_excel(rows, details, xp)
            print(f"\nExcel: {xp}")
        if not args.no_db:
            rid = save_db(rows, f"theo-mau {out_dir.name}")
            if rid:
                print(f"CSDL: run {rid}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"JSON: {args.json}")
    print(f"Ảnh kết quả: {out_dir.resolve()}"
          + (f" | bỏ qua {len(skipped)} ảnh không khớp mẫu" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
