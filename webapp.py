# -*- coding: utf-8 -*-
"""Web chấm trắc nghiệm (Streamlit) — chạy: .venv/bin/streamlit run webapp.py"""
import io
import json
import time
from datetime import datetime
from pathlib import Path

import openpyxl
import streamlit as st

import omr_key
import omr_templates

st.set_page_config(page_title="Chấm trắc nghiệm AI", page_icon="📝", layout="wide")

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)
SHEET_DIR = ROOT / "data/web_upload"
KEY_DIR = ROOT / "data/keys"
KEY_DIR.mkdir(parents=True, exist_ok=True)
REG = KEY_DIR / "registry.json"
BUILTIN_KEY_FILE = ROOT / "data/keys_builtin.json"

st.title("📝 Hệ thống chấm trắc nghiệm bằng AI")


def load_registry():
    """Nạp registry key; tự loại key hệ thống cũ (builtin-gk1) và key thiếu file."""
    reg = json.loads(REG.read_text(encoding="utf-8")) if REG.is_file() else {}
    changed = False
    for kid in list(reg):
        fname = str(reg[kid].get("file", ""))
        f = KEY_DIR / fname
        if kid == "builtin-gk1" or fname.endswith("builtin.json") or not f.is_file():
            reg.pop(kid)
            changed = True
    if changed:
        REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    return reg


def load_key_data(kid, reg):
    """Trả key_data cho id đã chọn (file key_<id>.xlsx trong data/keys)."""
    if not kid:
        return None
    f = KEY_DIR / f"key_{kid}.xlsx"
    if f.is_file():
        _, key_data, _ = omr_key.detect_key_format(f)
        return key_data
    return None


def load_keys_multi(kids, reg):
    """Gộp NHIỀU key thành 1 key hợp nhất: các đề = 'kid|mã đề' (không xung đột
    mã đề trùng giữa các bộ). Trả key_data hoặc None."""
    if not kids:
        return None
    des = {}
    for kid in kids:
        kd = load_key_data(kid, reg)
        if not kd:
            continue
        if kd.get("đề") == "QM":
            for de, dk in kd["các đề"].items():
                des[f"{kid}|{de}"] = dk
        else:
            des[f"{kid}|list"] = {"p1": {int(k): v["answer"] for k, v in kd.get("câu", {}).items()},
                                   "p2": {}, "p3": {}}
    return {"đề": "QM", "các đề": des} if des else None


def merge_keys_grading(kids, reg):
    """Gộp NHIỀU key đã chọn thành 1 key dùng cho CHẤM (luồng cham_theo_mau).

    Khác load_keys_multi (cho lưới mẫu v3, tiền tố 'kid|đề'): ở đây các đề GIỮ
    NGUYÊN mã (1132, 132...) để khớp với mã đề đọc từ lưới bubble; trùng mã đề
    giữa các key thì lấy của key được chọn trước. Trả key_data hoặc None."""
    if not kids:
        return None
    des = {}
    for kid in kids:
        kd = load_key_data(kid, reg)
        if not kd or kd.get("đề") != "QM":
            continue
        for de, dk in kd["các đề"].items():
            des.setdefault(str(de), dk)
    return {"đề": "QM", "các đề": des} if des else None


def process_page_import(path):
    from omr_grade_final import process_page as _pp
    return _pp(path)


def grade_batch(pages, progress=None, key_data=None):
    """pages: [(path, grp|None, krow|None)] -> results.

    Logic lai: trang CÓ pattern ô xanh (overlay) -> infer_made (mạnh nhất);
    trang trống pattern (bài mới) + có key -> chấm theo key, mã đề suy từ
    độ khớp đáp án với key (>=90% hoặc phân biệt rõ với đề nhì)."""
    from omr_grade_final import grade_page_full, infer_made, grade_with_key
    results = []
    for i, (f, grp, krow) in enumerate(pages):
        if progress is not None:
            progress.progress((i + 1) / len(pages),
                              text=f"Đang chấm {i + 1}/{len(pages)}: {Path(f).name}")
        tpl_name = st.session_state.get("template")
        tpl_grp = omr_templates.get_template(tpl_name)["grp"] if tpl_name else None
        r = grade_page_full(f, tpl_grp or grp, krow)
        if key_data and not r.get("pattern"):
            page = process_page_import(f)
            r2 = grade_with_key(page, grp, key_data)
            r2["file"] = Path(f).stem
            if r2.get("de_inferred"):
                parts = str(r2["de_inferred"]).split("|", 1)
                r2["md"] = parts[-1]
                r2["key_used"] = parts[0] if len(parts) > 1 else ""
                r2["md_source"] = "answer-key"
                r2["md_score"] = r2.get("key_score")
            else:
                r2["md"], r2["md_source"], r2["md_error"] = None, "none", r2.get("error")
            r = r2
        results.append(r)
    canon = None
    if BUILTIN_KEY_FILE.is_file():
        canon = json.loads(BUILTIN_KEY_FILE.read_text(encoding="utf-8")).get("các đề")
    n_fixed, _ = infer_made(results, canonical_refs=canon)
    if n_fixed:
        st.info(f"🔎 Suy ra/sửa mã đề từ đáp án cho {n_fixed} trang (pattern ≥ 90%)")
    if key_data:
        n_key = sum(1 for r in results if r.get("md_source") == "answer-key")
        st.info(f"📋 Key đang dùng: {n_key} trang chấm theo key "
                f"({sum(1 for r in results if r.get('md'))}/{len(results)} trang có mã đề).")
    return results


def normalize_batch(paths, tpl_path, paper, key, out_dir, progress=None,
                    ink=True, ink_only=False):
    """Chuẩn hoá nhiều ảnh về đúng mẫu + NHẬN DẠNG & TÔ ĐEN bubble đã tô.

    `key` = key_data đã nạp (dict có 'các đề') HOẶC đường dẫn file .xlsx.
    Trả (ok_paths, report_rows). Mỗi ảnh ghi 2 bản: *_chuan.jpg (dùng để chấm)
    và *_todam.jpg (bubble đã tô được tô đen để kiểm tra bằng mắt).
    """
    import omr_prep as PREP
    import cv2 as _cv
    if isinstance(key, (str, Path)):
        _fmt, key, _ = omr_key.detect_key_format(key)
    cells = CTM.keyed_cells(paper, tpl_path, key)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_paths, rows, ink_paths, extras = [], [], [], []
    for i, ip in enumerate(paths, 1):
        if progress is not None:
            progress.progress(i / max(1, len(paths)),
                              text=f"Chuẩn hoá {i}/{len(paths)}: {Path(ip).name}")
        info = PREP.normalize_to_template(ip, tpl_path, scale=2.0, cells=cells,
                                          ink=ink, ink_only=ink_only)
        if info.get("ok"):
            dst = out_dir / f"{Path(ip).stem}_chuan.jpg"
            _cv.imwrite(str(dst), info["warped"], [_cv.IMWRITE_JPEG_QUALITY, 95])
            ok_paths.append(dst)
            if info.get("inked") is not None:
                _cv.imwrite(str(out_dir / f"{Path(ip).stem}_todam.jpg"),
                            info["inked"], [_cv.IMWRITE_JPEG_QUALITY, 95])
                ink_paths.append(str(out_dir / f"{Path(ip).stem}_todam.jpg"))
            extras.append({"ids": info.get("ids_result"),
                           "quality": info.get("quality", 1.0),
                           "rot": info.get("rot", 0)})
            st = info.get("ink_stats") or {}
            rows.append({"file": Path(ip).name,
                         "chuẩn hoá": "chỉ tô đậm" if ink_only else "✓",
                         "xoay": f"{info['rot']}°",
                         "khớp mẫu": round(info["quality"], 2),
                         "ô đã tô": st.get("n_filled", "-"),
                         "ghi chú": ""})
        else:
            rows.append({"file": Path(ip).name, "chuẩn hoá": "✗", "xoay": "-",
                         "khớp mẫu": round(info.get("quality", 0.0), 2),
                         "ô đã tô": "-", "ghi chú": info.get("reason", "")})
            ink_paths.append(None)
            extras.append({"ids": None, "quality": 0.0, "rot": 0})
    return ok_paths, rows, ink_paths, extras


def make_result_pdf(img_paths, out_pdf, max_side=1600):
    """Gộp ảnh kết quả thành 1 file PDF — mỗi ảnh 1 trang, ĐÚNG thứ tự."""
    from PIL import Image
    pages = []
    for p in img_paths:
        im = Image.open(str(p)).convert("RGB")
        w, h = im.size
        s = min(1.0, max_side / float(max(w, h)))
        if s < 1.0:
            im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
        pages.append(im)
    if not pages:
        return None
    out_pdf = Path(out_pdf)
    pages[0].save(out_pdf, "PDF", resolution=150.0, save_all=True,
                  append_images=pages[1:])
    return out_pdf


def make_detail_excel(rows, details, out_xlsx, row_meta=None):
    """Excel 2 sheet: Tổng hợp + Chi tiết — ĐÚNG thứ tự trang của file PDF.

    Trang PDF chỉ chứa ảnh kết quả (ảnh chấm được), nên số trang được đánh
    tăng dần theo các dòng CÓ ảnh; dòng lỗi hiện "-" để không lệch thứ tự.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Tổng hợp"
    ws.append(["Trang PDF", "File", "SBD", "Mã đề", "Nguồn mã đề",
               "P1 đúng/số câu", "P2 đúng/số câu", "Tổng điểm",
               "Số sai", "Trống/nhiều", "Ghi chú"])
    page = 0
    for i, r in enumerate(rows):
        has_img = bool(row_meta and i < len(row_meta) and row_meta[i].get("img"))
        if has_img:
            page += 1
        ws.append([page if has_img else "-", r.get("file"), r.get("SBD"),
                   r.get("mã đề"), r.get("nguồn"),
                   r.get("P1"), r.get("P2"), r.get("tổng"),
                   r.get("sai"), r.get("trống/nhiều"), r.get("ghi chú", "")])
    for col, w in zip("ABCDEFGHIJK", (9, 26, 12, 8, 12, 15, 15, 11, 8, 12, 34)):
        ws.column_dimensions[col].width = w

    ws2 = wb.create_sheet("Chi tiết từng câu")
    ws2.append(["Trang PDF", "File", "Mã đề", "Phần", "Câu", "Đáp án",
                "Bài làm", "Trạng thái", "Điểm"])
    # giữ ĐÚNG thứ tự trang PDF
    for i, d in enumerate(details, 1):
        for (part, q), (st, exp, got) in d["detail"].items():
            cau = q if isinstance(q, int) else f"{q[0]}{q[1]}"
            ws2.append([i, d["file"], d.get("made"), part, cau, exp or "",
                        got or "", CTM.STATE_VI.get(st, st),
                        0.25 if st == "ok" else 0.0])
    for col, w in zip("ABCDEFGHI", (9, 26, 8, 7, 8, 9, 9, 12, 8)):
        ws2.column_dimensions[col].width = w
    wb.save(str(out_xlsx))
    return out_xlsx


def results_table(results):
    rows = []
    for i, r in enumerate(results, 1):
        k = r.get("kq") or {}
        rows.append({
            "STT": i, "File": r["file"], "Loại phiếu": r["grp"],
            "Mã đề": r.get("md") or k.get("made") or "",
            "Nguồn mã đề": r.get("md_source") or "",
            "SBD": r.get("sbd") or k.get("sbd") or "",
            "P1": r["my"]["P1"], "P2": r["my"]["P2"], "P3": r["my"]["P3"],
            "Tổng AI": r["my"]["tong"],
            "Tổng KQ": k.get("tong"),
            "Chênh": r.get("diff", {}).get("tong"),
            "Khớp": ("✓" if r.get("match") else "") if k else "",
        })
    return rows


def to_excel(rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "KQ chấm AI"
    ws.append(list(rows[0].keys()) if rows else ["STT"])
    for r in rows:
        ws.append(list(r.values()))
    for col, w in zip("ABCDEFGHIJKLM", (5, 14, 9, 9, 9, 12, 7, 7, 7, 9, 9, 8, 6)):
        ws.column_dimensions[col].width = w
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def save_run_db(source, results):
    try:
        from omr_db import save_run
        rid = save_run(source, results)
        st.toast(f"Đã lưu vào CSDL: run {rid}", icon="💾")
        return rid
    except Exception as e:  # noqa
        st.warning(f"Lưu CSDL bỏ qua: {e}")


tab_key, tab_sheet, tab_tpl = st.tabs(
    ["📋 Đáp án & QR", "🟩 Chấm theo mẫu",
     "🎯 Mẫu giấy & lưới đáp án"])

# ---------------- TAB 0: MẪU GIẤY & LƯỚI ĐÁP ÁN ----------------
with tab_tpl:
    st.markdown("**Quy trình:** ① chọn mẫu giấy → ② nạp đáp án (tab 📋) → ⓪ tự động "
                "chuẩn hoá ảnh + tô đậm bubble (kể cả SBD, mã đề) → ③ ảnh vào sẽ "
                "khớp lưới → đúng/sai → điểm.")
    tnames = omr_templates.list_templates()
    tname = st.selectbox("① Chọn mẫu giấy:", list(tnames), format_func=lambda k: tnames[k])
    tpl = omr_templates.get_template(tname)
    st.session_state["template"] = tname
    st.caption(f"Loại phiếu: `{tpl['grp']}` — " + tpl["structure"]["P1"]["mô_tả"])
    reg0 = load_registry()
    kids0 = st.session_state.get("key_ids") or []
    grid = None
    if not kids0:
        st.info("② Chưa nạp key nào — vào tab '📋 Đáp án & QR' để nạp đáp án.")
    else:
        kd = load_keys_multi(kids0, reg0)
        if kd:
            grid = omr_templates.build_answer_grid(tpl, kd)
            st.session_state["grid"] = grid
            c1, c2, c3 = st.columns(3)
            c1.metric("Số đề", grid["số_đề"])
            c2.metric("Số câu", grid["số_câu"])
            c3.metric("Số ô đáp án", grid["số_ô"])
            st.dataframe([{"câu": r["câu"], "phần": r["phần"], "loại": r["loại"],
                           "đáp án": r["đáp án"], "vị trí trên phiếu": r["vị trí"],
                           "đề": r["đề"]} for r in grid["ô"]],
                         width="stretch", hide_index=True)
            st.download_button("⬇️ Tải lưới đáp án (JSON)",
                               data=json.dumps(grid, ensure_ascii=False, indent=1),
                               file_name=f"luoi_dap_an_{tname}.json")
    if tpl.get("vùng") is None:
        st.warning("⚠️ Mẫu này chưa hiệu chuẩn hình học — chấm sẽ dùng auto-detect.")

# ---------------- TAB 1b: CHẤM THEO MẪU (FIDUCIAL) ----------------
with tab_sheet:
    import json as _json
    import omr_sheetgrid as SG
    import cham_theo_mau as CTM

    st.markdown("**Quy trình:** chọn mẫu giấy → chọn đáp án (tab 📋 Đáp án & QR) → "
                "tải ảnh → hệ thống khớp ảnh với mẫu bằng **ô đen định vị**, "
                "đọc SBD/Mã đề, đối chiếu bubble.")
    st.caption("🟢 đúng · 🔴 sai · 🟡 trống hoặc tô nhiều đáp án")

    TPL_CHOICES = {
        "12": ("Mẫu THPT 12.png (40 câu ABCD + 8 Đ/S)",
               str(ROOT / "data" / "maugiay" / "12.png")),
        "10_11": ("Mẫu QM 10_11.png (30 câu ABCD + 10 Đ/S)",
                  str(ROOT / "data" / "maugiay" / "10_11.png")),
    }
    # ② ĐÁP ÁN = key đã CHỌN ở tab '📋 Đáp án & QR' (bảng 'Các đáp án đã lưu')
    reg_s = load_registry()
    kids_sel = [k for k in (st.session_state.get("key_ids") or []) if k in reg_s]
    key_names = [str(reg_s[k].get("name") or reg_s[k].get("desc") or k)
                 for k in kids_sel]
    _key_sel = merge_keys_grading(kids_sel, reg_s)
    c_paper, c_keyf = st.columns([1, 2])
    with c_paper:
        paper = st.selectbox("① Mẫu giấy:", list(TPL_CHOICES),
                             format_func=lambda k: TPL_CHOICES[k][0])
    tpl_path = TPL_CHOICES[paper][1]
    with c_keyf:
        if key_names:
            st.markdown("**② Đáp án đã chọn:**  \n" + " · ".join(key_names))
        else:
            st.warning("② Chưa chọn đáp án — chọn dòng trong bảng "
                       "'Các đáp án đã lưu' ở tab '📋 Đáp án & QR'.")

    if not Path(tpl_path).is_file():
        st.error(f"Không thấy ảnh mẫu: {tpl_path}")
    elif _key_sel is None:
        st.info("Hãy upload Excel đáp án và CHỌN DÒNG key ở tab '📋 Đáp án & QR' "
                "— lần chấm sẽ dùng đúng đáp án đã chọn.")
    else:
        try:
            cells = SG.build_template_cells(paper, tpl_path)
            id_cells = SG.build_id_cells(paper, tpl_path)
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Số đề trong key", len(_key_sel["các đề"]))
            cc2.metric("Ô PHẦN I", len([c for c in cells if c[0] == "P1"]))
            cc3.metric("Ô PHẦN II", len([c for c in cells if c[0] == "P2"]))
            cc4.metric("Ô SBD/Mã đề", len(id_cells))
            st.caption(f"Đáp án: {', '.join(key_names)}")
            de0 = list(_key_sel["các đề"])[0]
            g0 = SG.build_answer_grid(paper, tpl_path, _key_sel["các đề"][de0])
            rows0 = [{"phần": p, "câu": (q if isinstance(q, int) else f"{q[0]}{q[1]}"),
                      "lựa chọn": o, "đáp án": g0["answers"].get((p, q, o)),
                      "toạ độ mẫu": f"({x:.0f},{y:.0f})"}
                     for (p, q, o, x, y) in cells[:80]]
            with st.expander(f"🔎 Xem lưới đáp án đề {de0} (80 ô đầu)"):
                st.dataframe(rows0, width="stretch", hide_index=True)
            st.download_button("⬇️ Tải lưới đáp án đầy đủ (JSON)",
                               data=_json.dumps([{"phần": p, "câu": str(q), "lựa_chọn": o,
                                                  "đáp_án": g0["answers"].get((p, q, o)),
                                                  "x": round(x, 1), "y": round(y, 1)}
                                                 for (p, q, o, x, y) in cells],
                                                ensure_ascii=False, indent=1),
                               file_name=f"luoi_dap_an_{paper}_{de0}.json")
        except Exception as _e:  # noqa
            st.warning(f"Chưa dựng được lưới: {_e}")

    st.divider()
    round_no = st.session_state.get("round_no", 0)
    files_s = st.file_uploader("③ Tải ảnh bài thi (chọn nhiều)",
                               key=f"sheet_files_{round_no}",
                               type=["jpg", "jpeg", "png", "bmp", "webp"],
                               accept_multiple_files=True)
    gdrive_files = st.session_state.get("gdrive_files") or []
    gdrive_links = st.text_area(
        "🔗 Link Google Drive (mỗi link một dòng — lấy TẤT CẢ file ảnh trong link, gồm thư mục con)",
        value="", key="sheet_gdrive_links", height=90,
        placeholder="https://drive.google.com/drive/folders/...")
    if st.button("📥 Lấy ảnh từ Google Drive", key="gd_fetch",
                 disabled=not gdrive_links.strip()):
        from omr_gdrive import scan_links
        prog = st.progress(0.0, text="Đang lấy ảnh từ Google Drive...")
        try:
            pages, report = scan_links(gdrive_links, prog)
            prog.empty()
            st.session_state["gdrive_files"] = [str(x) for x in pages]
            st.session_state["gdrive_report"] = report
            st.session_state["gdrive_stamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            if report and len(report):
                st.dataframe(report, width="stretch", hide_index=True)
            if len(pages):
                st.success(f"Đã lấy {len(pages)} ảnh từ Google Drive.")
            else:
                loi = "; ".join(str(x.get("lỗi", ""))[:80] for x in report if x.get("lỗi"))
                st.error("Không lấy được ảnh nào từ link. " +
                         ("Lý do: " + loi if loi else
                          "Hãy kiểm tra: ① link đúng chưa; ② file/thư mục đã chia sẻ "
                          "'Bất kỳ ai có liên kết' chưa; ③ link trỏ tới ảnh/PDF chứ không phải "
                          "tài liệu khác."))
        except Exception as e:                                   # noqa
            st.error(f"Lỗi khi lấy ảnh từ Google Drive: {e}")
    gdrive_files = st.session_state.get("gdrive_files") or []
    c_made, c_btn = st.columns([1, 2])
    with c_made:
        fixed_made = st.text_input("Mã đề (bỏ trống = tự chọn)", value="", key="sheet_made")
    norm_on = st.checkbox(
        "⓪ Chuẩn hoá ảnh về đúng mẫu trước khi chấm (tự xoay 4 hướng + khử nghiêng/phối cảnh)",
        value=True, key="sheet_norm",
        help="Ảnh upload / Drive / điện thoại được đưa về đúng khung mẫu giấy đã chọn "
             "(cùng kích thước, cùng góc) rồi mới chấm. Ảnh nào không khớp mẫu sẽ bị báo, "
             "không bao giờ chuẩn hoá bừa.")
    # ---- ⓪ TIỀN XỬ LÝ TỰ ĐỘNG: chuẩn hoá + nhận dạng & TÔ ĐEN bubble ----
    # Nguồn ảnh = ảnh upload + ảnh đã lấy từ Google Drive
    gdrive_files = st.session_state.get("gdrive_files") or []
    prep_ready = (files_s or gdrive_files) and Path(tpl_path).is_file() \
        and _key_sel is not None
    with st.expander("⓪ Ảnh đã chuẩn hoá & tô đậm bubble (bấm dòng để so sánh)",
                     expanded=bool(files_s or gdrive_files)):
        if not prep_ready:
            st.info("Tải ảnh lên / dán link Google Drive và chọn mẫu giấy + đáp án để bắt đầu.")
        else:
            sig = (tuple((u.name, len(u.getvalue())) for u in files_s),
                   tuple(gdrive_files), paper, tuple(kids_sel), bool(norm_on))
            if st.session_state.get("prep_sig") != sig:
                stamp_n = datetime.now().strftime("%Y%m%d_%H%M%S")
                destn = SHEET_DIR / f"norm_{stamp_n}"
                destn.mkdir(parents=True, exist_ok=True)
                pths = []
                for u in files_s:                      # ảnh upload
                    q = destn / u.name
                    q.write_bytes(u.getvalue())
                    pths.append(q)
                for gp in gdrive_files:                # ảnh đã lấy từ Drive
                    pths.append(Path(gp))
                pr = st.progress(0.0,
                                 text=("Đang chuẩn hoá & tô đậm..." if norm_on
                                       else "Đang tô đậm bubble..."))
                ok_paths, rep, ink_paths, extras = normalize_batch(
                    pths, tpl_path, paper, _key_sel, destn / "chuan_hoa", pr,
                    ink_only=not norm_on)
                pr.empty()
                st.session_state["prep_sig"] = sig
                st.session_state["prep_res"] = {
                    "rows": rep, "ok": [str(x) for x in ok_paths],
                    "ink": ink_paths, "orig": [str(x) for x in pths],
                    "extras": extras,
                    "out": str(destn / "chuan_hoa"), "stamp": stamp_n}
            res = st.session_state.get("prep_res")
            if res:
                st.caption("👉 Bấm vào một dòng để xem Ảnh gốc và ảnh đã "
                           "Chuẩn hoá / Tô đậm bên dưới.")
                psel = st.dataframe(res["rows"], width="stretch", hide_index=True,
                                    on_select="rerun", selection_mode="single-row",
                                    key="prep_table")
                prows = list(getattr(psel.selection, "rows", []) or []) \
                    if hasattr(psel, "selection") else []
                idx = min(prows[0], len(res["rows"]) - 1) if prows else 0
                c1, c2 = st.columns(2)
                c1.image(res["orig"][idx], caption="Ảnh gốc", width="stretch")
                ok_p = res["ok"][idx] if idx < len(res["ok"]) else None
                ink_p = res["ink"][idx] if idx < len(res["ink"]) else None
                show = ink_p or ok_p
                if show and Path(show).is_file():
                    cap = ("Đã chuẩn hoá + TÔ ĐEN bubble (kể cả mã đề, SBD)"
                           if ink_p else "Sau chuẩn hoá (đúng mẫu)")
                    c2.image(show, caption=cap, width="stretch")
                else:
                    c2.info("Ảnh này không chuẩn hoá được — xem cột ghi chú.")
                import zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for pth in map(Path, res["ok"]):
                        z.write(pth, pth.name)
                        td = pth.parent / f"{pth.stem.replace('_chuan', '')}_todam.jpg"
                        if td.is_file():
                            z.write(td, td.name)
                st.download_button("⬇️ Tải ZIP ảnh đã chuẩn hoá / tô đậm",
                                   data=buf.getvalue(),
                                   file_name=f"anh_chuan_hoa_{res['stamp']}.zip",
                                   key="sheet_norm_zip")

    c_run, c_new, _ = st.columns([1.2, 1.2, 2.6])
    with c_new:
        if st.button("🆕 Chấm đợt khác", key="sheet_new_round",
                     help="Xoá kết quả đợt hiện tại và nạp ảnh của đợt mới; "
                          "giữ nguyên mẫu giấy + đáp án đã chọn."):
            st.session_state["round_no"] = st.session_state.get("round_no", 0) + 1
            for k_ in ("sheet_res", "prep_res", "prep_sig",
                       "gdrive_files", "gdrive_report"):
                st.session_state.pop(k_, None)
            st.rerun()
    with c_run:
        run = st.button("▶️ Chấm theo mẫu", type="primary", key="sheet_run",
                    disabled=not ((files_s or gdrive_files)
                                  and Path(tpl_path).is_file()
                                  and _key_sel is not None
                                  and st.session_state.get("prep_res")))
    if run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # chấm trên ẢNH ĐÃ CHUẨN HOÁ / TÔ ĐẬM của bước ⓪ (không dùng ảnh gốc)
        res_p = st.session_state["prep_res"]
        paths, path_ok = [], []
        ok_list = res_p["ok"]
        for k_, orig_p in enumerate(res_p["orig"]):
            ink_p = res_p["ink"][k_] if k_ < len(res_p["ink"]) else None
            ok_p = ok_list[k_] if k_ < len(ok_list) else None
            if not (ok_p and Path(ok_p).is_file()):
                # bước ⓪ không chuẩn hoá được -> không chấm ảnh gốc
                paths.append(Path(orig_p))
                path_ok.append(False)
                continue
            use = ink_p if (ink_p and Path(ink_p).is_file()) else ok_p
            paths.append(Path(use))
            path_ok.append(True)
        _key = _key_sel
        prog = st.progress(0.0, text="Đang khớp mẫu...")
        outd = OUTDIR / "theo_mau" / f"web_{stamp}"
        outd.mkdir(parents=True, exist_ok=True)
        rows_out, details, rendered, used = [], [], [], set()
        name_by_key, dup_counts = {}, {}   # (SBD, Mã đề) -> tên chuẩn / số lần
        row_meta = []     # song song với rows_out: ảnh kết quả + mã đề + SBD
        for i, ip in enumerate(paths, 1):
            # tên hiển thị = tên ẢNH GỐC (bỏ hậu tố _todam/_chuan của bước ⓪)
            disp = Path(res_p["orig"][i - 1]).name if (i - 1) < len(res_p["orig"]) \
                else ip.name
            prog.progress(i / max(1, len(paths)), text=f"{disp} ({i}/{len(paths)})")
            # Ảnh vào đã chuẩn hoá ở bước ⓪ và SBD/Mã đề đã đọc sẵn -> chấm
            # trực tiếp, KHÔNG căn lại / KHÔNG quét ID lần 2 (nhanh gấp đôi)
            ex = res_p.get("extras") or []
            ex_i = ex[i - 1] if (i - 1) < len(ex) else {}
            if not path_ok[i - 1]:
                rows_out.append({"file": disp, "chuẩn hoá": "✗", "SBD": "-",
                                 "mã đề": "-", "nguồn": "-", "khớp mẫu": 0.0,
                                 "P1": "-", "P2": "-", "tổng": 0, "sai": 0,
                                 "trống/nhiều": 0,
                                 "ghi chú": "bước ⓪ không chuẩn hoá được — xem bảng ⓪"})
                row_meta.append({"img": None, "sbd": None, "made": None, "file": disp})
                continue
            r = CTM.grade_prepared(ip, paper, tpl_path, _key,
                                   ids=ex_i.get("ids"),
                                   made=(fixed_made or None),
                                   quality=ex_i.get("quality", 1.0))
            if r is None:
                rows_out.append({"file": disp, "chuẩn hoá": "✗", "SBD": "-", "mã đề": "-",
                                 "nguồn": "-", "khớp mẫu": 0.0, "P1": "-", "P2": "-",
                                 "tổng": 0, "sai": 0, "trống/nhiều": 0,
                                 "ghi chú": "không khớp mẫu (chụp rõ 4 góc)"})
                row_meta.append({"img": None, "sbd": None, "made": None, "file": disp})
                continue
            # Tên chuẩn: <SBD>_<Mã đề>_<ngày>_<giờ>.jpg
            # tên theo quy định: SBD_Mãđề_NgàyGiờ HIỆN HÀNH (có giây)
            base_name = CTM.output_name_now(r)
            sbd_txt = str(r.get("sbd") or "").strip()
            dup_key = (sbd_txt, str(r["made"]))
            # chỉ coi là trùng khi CÓ Số báo danh (mẫu QM không có lưới SBD)
            if sbd_txt and dup_key in name_by_key:
                # TRÙNG Số báo danh + Mã đề -> cùng tên chuẩn, thêm _2, _3...
                dup_counts[dup_key] = dup_counts.get(dup_key, 1) + 1
                stem = f"{name_by_key[dup_key]}_{dup_counts[dup_key]}"
            else:
                name_by_key[dup_key] = base_name
                dup_counts[dup_key] = 1
                stem = base_name
            outp = outd / f"{stem}.jpg"
            n = 1
            while stem in used or outp.exists():
                n += 1
                outp = outd / f"{stem}_{n}.jpg"
            used.add(stem)
            SG.render_result(r["img"], r["decided"], r["detail"], r["stats"], outp,
                             paper, made=r["made"], s1=r["p1"], s2=r["p2"],
                             sbd=r.get("sbd"), ids=r.get("ids"), show_sbd=True,
                             md_inferred=(r.get("made_source") != "lưới"),
                             md_note="", tpl_path=tpl_path)
            rendered.append(outp)
            n_wrong = sum(1 for v in r["detail"].values() if v[0] == "wrong")
            n_unk = sum(1 for v in r["detail"].values() if v[0] in ("blank", "multi"))
            pn = r.get("pre_norm") or {}
            norm_tag = ("↻%d° q%.2f" % (pn.get("rot", 0), pn.get("quality", 0))) if pn else "—"
            rows_out.append({"file": disp, "chuẩn hoá": norm_tag,
                             "SBD": r.get("sbd") or "-",
                             "mã đề": r["made"], "nguồn": r["made_source"],
                             "khớp mẫu": r["quality"],
                             "P1": f"{r['stats']['P1']['ok']}/{r['stats']['P1']['n']}",
                             "P2": f"{r['stats']['P2']['ok']}/{r['stats']['P2']['n']}",
                             "tổng": r["score"], "sai": n_wrong, "trống/nhiều": n_unk,
                             "ghi chú": ""})
            details.append({"file": disp, "made": r["made"], "detail": r["detail"]})
            row_meta.append({"img": str(outp), "sbd": r.get("sbd"),
                             "made": r["made"], "file": disp})
        # ---- SẮP THỨ TỰ THEO TÊN FILE ẢNH KẾT QUẢ ----
        # (vd NA_132_20261103_104757.jpg, NA_209_...): Excel + PDF + bảng đều
        # theo đúng thứ tự này, không theo thứ tự upload.
        order = sorted(range(len(rows_out)),
                       key=lambda i: (Path(row_meta[i]["img"]).name
                                      if row_meta[i].get("img")
                                      else "zzz_" + str(rows_out[i].get("file"))))
        rows_out = [rows_out[i] for i in order]
        row_meta = [row_meta[i] for i in order]
        # details/rendered chỉ chứa trang chấm được -> sắp theo tên ảnh
        det_pairs = list(zip(rendered, details)) if len(rendered) == len(details) else []
        if det_pairs:
            det_pairs.sort(key=lambda z: Path(z[0]).name if not isinstance(z[0], str)
                           else Path(z[0]).name)
            rendered = [p_ for p_, _ in det_pairs]
            details = [d_ for _, d_ in det_pairs]
        prog.empty()
        st.session_state["sheet_res"] = {
            "rows": rows_out, "meta": row_meta, "rendered": [str(x) for x in rendered],
            "details": details, "outd": str(outd), "stamp": stamp}
    res = st.session_state.get("sheet_res")
    if res:
        rows_out = res["rows"]; row_meta = res["meta"]
        rendered = [Path(x) for x in res["rendered"]]
        details = res["details"]; stamp = res["stamp"]
        st.success(f"Đã chấm {len(rows_out)} ảnh — lưu tại {res['outd']}")
        st.caption("👉 Bấm vào một dòng để xem ảnh kết quả bên dưới. "
                   "Màu trên ảnh: **xanh = đúng**, **đỏ = sai**, **vàng = trống/tô nhiều**, "
                   "**cam = đáp án đúng của câu đó**, **xanh dương = mã đề suy từ đáp án**.")
        sel = st.dataframe(rows_out, width="stretch", hide_index=True,
                           on_select="rerun", selection_mode="single-row",
                           key="sheet_res_table")
        picked = list(getattr(sel.selection, "rows", []) or []) \
            if hasattr(sel, "selection") else []
        if details:
            with st.expander("🔎 Vì sao ra điểm này? (chi tiết từng câu)", expanded=False):
                d0 = details[0]
                wrong = [(p_, q_, exp, got) for (p_, q_), (st, exp, got)
                         in d0["detail"].items() if st in ("wrong", "blank", "multi")]
                if wrong:
                    st.caption(f"{d0['file']}: " + " · ".join(
                        f"{p_}{q_}: key {exp} / làm {got or '—'}" for p_, q_, exp, got in wrong[:12]))
                st.dataframe([{"phần": p_, "câu": (q_ if isinstance(q_, int) else f"{q_[0]}{q_[1]}"),
                               "trạng thái": CTM.STATE_VI.get(st, st),
                               "đáp án": exp, "bài làm": got}
                              for (p_, q_), (st, exp, got) in d0["detail"].items()],
                             width="stretch", hide_index=True)
        if rendered:
            import zipfile
            # 1) PDF gộp ảnh kết quả (mỗi ảnh 1 trang, đúng thứ tự)
            pdf_p = None
            try:
                pdf_p = make_result_pdf(rendered, Path(res["outd"]) /
                                        f"ket_qua_{stamp}.pdf")
            except Exception as e:                                 # noqa
                st.warning(f"Không tạo được PDF: {e}")
            # 2) Excel thống kê chi tiết — cùng thứ tự trang PDF
            xls_p = None
            try:
                xls_p = make_detail_excel(rows_out, details,
                                          Path(res["outd"]) /
                                          f"thong_ke_chi_tiet_{stamp}.xlsx",
                                          row_meta=row_meta)
            except Exception as e:                                 # noqa
                st.warning(f"Không tạo được Excel: {e}")
            # 3) ZIP: ảnh + PDF + Excel
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for pth in rendered:
                    z.write(pth, f"anh_ket_qua/{pth.name}")
                if pdf_p and Path(pdf_p).is_file():
                    z.write(pdf_p, Path(pdf_p).name)
                if xls_p and Path(xls_p).is_file():
                    z.write(xls_p, Path(xls_p).name)
            st.download_button(
                "⬇️ Tải bộ kết quả (ảnh 3 màu + PDF + Excel chi tiết)",
                data=buf.getvalue(), file_name=f"ket_qua_theo_mau_{stamp}.zip",
                mime="application/zip")
            cdl1, cdl2 = st.columns(2)
            if pdf_p and Path(pdf_p).is_file():
                cdl1.download_button("⬇️ Chỉ tải PDF ảnh kết quả",
                                     data=Path(pdf_p).read_bytes(),
                                     file_name=Path(pdf_p).name, mime="application/pdf")
            if xls_p and Path(xls_p).is_file():
                cdl2.download_button("⬇️ Chỉ tải Excel chi tiết",
                                     data=Path(xls_p).read_bytes(),
                                     file_name=Path(xls_p).name,
                                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        idx = picked[0] if picked and picked[0] < len(row_meta) else 0
        meta = row_meta[idx] if row_meta else None
        if meta and meta.get("img") and Path(meta["img"]).is_file():
            parts = [f"**Mã đề: {meta['made']}**"]
            if paper == "12" and meta.get("sbd"):
                parts.append(f"**SBD: {meta['sbd']}**")
            st.markdown(" · ".join(parts))
            st.image(meta["img"], caption=Path(meta["img"]).name, width="stretch")


# ---------------- TAB 2: ĐÁP ÁN & QR ----------------
with tab_key:
    st.markdown("**1) Đặt tên + Upload Excel đáp án** — **2) sinh/quét QR** — "
                "**3) chọn dòng key** rồi chấm bài ở tab '🟩 Chấm theo mẫu'.")
    reg = load_registry()

    # ---- 1. TÊN ĐÁP ÁN + UPLOAD ----
    ten_dap_an = st.text_input(
        "Tên đáp án (thể hiện đáp án của kỳ kiểm tra nào):",
        key="key_name", placeholder="vd: Tin học 12 - Giữa kỳ 1 - 2026-2027")
    up_files = st.file_uploader("Upload NHIỀU file Excel đáp án (k10, k11, k12...)",
                                type=["xlsx", "xls"], key="key_up",
                                accept_multiple_files=True)
    if up_files and st.button("➕ Nhận dạng & lưu TẤT CẢ key", key="key_save"):
        n_ok = 0
        for up in up_files:
            tmp = KEY_DIR / up.name
            tmp.write_bytes(up.getvalue())
            fmt, key_data, desc = omr_key.detect_key_format(tmp)
            if not fmt:
                st.error(f"{up.name}: {desc}")
                continue
            import hashlib
            kid = hashlib.sha1(tmp.read_bytes()).hexdigest()[:8]
            dst = KEY_DIR / f"key_{kid}.xlsx"
            tmp.rename(dst)
            # Tên hiển thị: nhập tay > tên file; nếu upload nhiều file một lần
            # thì ghép thêm tên file để các dòng không trùng nhau
            if ten_dap_an.strip() and len(up_files) == 1:
                kname = ten_dap_an.strip()
            elif ten_dap_an.strip():
                kname = f"{ten_dap_an.strip()} [{Path(up.name).stem}]"
            else:
                kname = Path(up.name).stem
            reg[kid] = {"file": dst.name, "desc": desc, "name": kname,
                        "ts": datetime.now().isoformat(timespec="seconds")}
            n_ok += 1
        if n_ok:
            REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
            st.session_state["key_ids"] = list(reg.keys())
            # chọn sẵn toàn bộ các dòng trong bảng key
            st.session_state["key_tbl"] = {"selection": {"rows":
                                             list(range(len(reg)))}}
            st.success(f"Đã lưu {n_ok} key — mặc định nạp TẤT CẢ vào lần chấm.")

    # ---- 2. BẢNG KEY ĐÃ LƯU (TT | Tên đáp án | Ngày tạo) ----
    st.markdown("**Các đáp án đã lưu** (chọn các dòng để nạp vào lần chấm):")
    kids = list(reg.keys())

    def _fmt_ts(ts):
        try:
            return datetime.fromisoformat(ts).strftime("%d/%m/%Y %H:%M") if ts else "-"
        except Exception:                                        # noqa
            return ts or "-"

    df_rows = [{"TT": i + 1,
                "Tên đáp án": reg[k].get("name") or reg[k].get("desc") or k,
                "Ngày tạo": _fmt_ts(reg[k].get("ts", ""))}
               for i, k in enumerate(kids)]
    ev = st.dataframe(df_rows, width="stretch", hide_index=True,
                      on_select="rerun", selection_mode="multi-row",
                      key="key_tbl")
    sel = []
    if hasattr(ev, "selection") and hasattr(ev.selection, "rows"):
        sel = list(ev.selection.rows)
    elif isinstance(ev, dict) and "selection" in ev:
        sel_data = ev["selection"]
        if isinstance(sel_data, dict):
            sel = list(sel_data.get("rows", []))
        elif hasattr(sel_data, "rows"):
            sel = list(sel_data.rows)
    if sel:
        st.session_state["key_ids"] = [kids[i] for i in sel]
    else:
        st.session_state["key_ids"] = []
    c_tbl1, c_tbl2, _c = st.columns([1, 1, 2])
    if sel and c_tbl1.button("🗑 Xoá key đã chọn", key="key_del"):
        for i in sel:
            kid_d = kids[i]
            f_d = KEY_DIR / str(reg[kid_d].get("file", f"key_{kid_d}.xlsx"))
            if f_d.name.startswith("key_") and f_d.is_file():
                f_d.unlink(missing_ok=True)
            reg.pop(kid_d, None)
        REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
        st.session_state.pop("key_tbl", None)
        st.session_state["key_ids"] = []
        st.rerun()
    if sel:
        c_tbl2.caption(f"Đang nạp {len(sel)} key — dùng cho lần chấm tới.")

    # ---- 3. QR ----
    key_ids = st.session_state.get("key_ids") or []
    if key_ids:
        qr_id = key_ids[0]
        qr_png = omr_key.make_qr_png(omr_key.QRCODE_PREFIX + qr_id)
        if qr_png:
            g1, g2 = st.columns(2)
            g1.image(qr_png, caption=f"QR chọn bộ {qr_id}", width=180)
            g2.download_button("⬇️ Tải QR", data=qr_png, file_name=f"qr_key_{qr_id}.png")
    qr_up = st.file_uploader("Upload/scan hình mã QR (thêm key vào bộ nạp)",
                             type=["png", "jpg", "jpeg"], key="qr_up")
    if qr_up and st.button("🔍 Đọc QR", key="qr_read"):
        payload = omr_key.decode_qr_image(qr_up.getvalue())
        if payload:
            kid = payload[0][len(omr_key.QRCODE_PREFIX):]
            if kid in reg:
                cur = st.session_state.get("key_ids") or []
                if kid not in cur:
                    cur = cur + [kid]
                st.session_state["key_ids"] = cur
                st.success(f"Đã thêm key `{kid}` từ QR: "
                           f"{reg[kid].get('name') or reg[kid]['desc']}")
            else:
                st.error(f"QR trỏ key `{kid}` chưa có trong máy.")
        else:
            st.error("Không đọc được QR (payload phải là OMRKEY:...)")

# ---------------- TAB 3: GOOGLE DRIVE ----------------
# (đã bỏ tab "Lịch sử CSDL" theo yêu cầu — kết quả vẫn lưu vào CSDL qua save_run_db)
