# -*- coding: utf-8 -*-
"""Web chấm trắc nghiệm (Streamlit) — chạy: .venv/bin/streamlit run webapp.py"""
import faulthandler
import io
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
from datetime import datetime
from pathlib import Path

import openpyxl
import streamlit as st

import numpy as np

import omr_key
import omr_templates
import omr_prep as PREP
import omr_aiassist as AIA
import cham_theo_mau as CTM

# Chẩn đoán treo: kill -USR1 <pid> sẽ in stack TẤT CẢ thread ra log server
faulthandler.register(signal.SIGUSR1, file=sys.stderr)


class _StepTimeout(Exception):
    """Một bước chấm/chuẩn hoá vượt quá thời gian cho phép (đang treo)."""


def run_with_timeout(fn, timeout_s, *args, **kwargs):
    """Chạy fn trong thread riêng với giới hạn thời gian.

    Nếu vượt timeout -> raise _StepTimeout; script CHẤM TIẾP các trang còn lại
    thay vì treo cả app. Lưu ý thread bị bỏ lại vẫn chạy (không kill được), nhưng
    app và người dùng không bị khoá."""
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout_s)
    except _FutTimeout:
        raise _StepTimeout(f"vượt {timeout_s}s")
    finally:
        ex.shutdown(wait=False)


def _df_str(rows):
    """Ép MỌI ô của bảng hiển thị về chuỗi — chống lỗi pyarrow ArrowInvalid
    ('Could not convert ... tried to convert to int64') khi 1 cột trộn kiểu
    (vd 'câu' = 1, 2, '2a'; 'ô đã tô' = 12, '-') — lỗi này làm BẢNG KHÔNG
    HIỂN THỊ trên giao diện."""
    return [{k: ("" if v is None else str(v)) for k, v in r.items()}
            for r in rows]

st.set_page_config(page_title="Chấm trắc nghiệm AI", page_icon="📝", layout="wide")

# Tự RELOAD khi trình duyệt giữ chunk JS cũ (sau khi server khởi động lại,
# chunk bất biến trong cache không khớp -> bảng/khu nội dung hiển thị TRỐNG).
# Reload tối đa 1 lần / 10 giây để không lặp vô hạn.
st.html("""
<script>
function handleChunkError() {
    const lastReload = sessionStorage.getItem('last_chunk_reload');
    const now = Date.now();
    if (!lastReload || now - parseInt(lastReload) > 10000) {
        sessionStorage.setItem('last_chunk_reload', now.toString());
        window.location.reload();
    }
}
window.addEventListener('vite:preloadError', handleChunkError);
window.addEventListener('error', (e) => {
    if (e && e.message && e.message.includes('dynamically imported module')) {
        handleChunkError();
    }
});
window.addEventListener('unhandledrejection', (e) => {
    if (e && e.reason && (e.reason.message || '').includes('dynamically imported module')) {
        handleChunkError();
    }
});
</script>
""", unsafe_allow_javascript=True)

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)
SHEET_DIR = ROOT / "data/web_upload"
KEY_DIR = ROOT / "data/keys"
KEY_DIR.mkdir(parents=True, exist_ok=True)
REG = KEY_DIR / "registry.json"
BUILTIN_KEY_FILE = ROOT / "data/keys_builtin.json"

st.title("📝 Hệ thống chấm trắc nghiệm bằng AI")


# ---------- 🔑 MODAL NHẬP GEMINI API KEY KHI KHỞI ĐỘNG ----------
def _startup_key_present():
    import os as _os
    if _os.environ.get("GEMINI_API_KEY", "").strip():
        return True
    _kf = ROOT / "data" / "gemini_key.txt"
    return _kf.is_file() and bool(_kf.read_text(encoding="utf-8").strip())


@st.dialog("🔑 Gemini API key — bật trợ lý AI", width="large")
def _startup_key_dialog():
    st.markdown(
        "Trợ lý AI Gemini hỗ trợ: **🤖 chấm trực tiếp** ảnh không chuẩn hoá "
        "được, **🤖 cứu định vị** ảnh lỗi, và **đọc SBD + STT chữ viết tay**. "
        "Không có key app vẫn chấm thường, nhưng không có các hỗ trợ trên.")
    st.markdown(
        "**Lấy key miễn phí (2 phút):**\n"
        "1. Mở **aistudio.google.com/apikey** và đăng nhập tài khoản Google\n"
        "2. Bấm **“Create API key”** → copy chuỗi key (dạng `AIza…`)\n"
        "3. Dán vào ô dưới và bấm **💾 Lưu key** — key lưu tại "
        "`data/gemini_key.txt` (không đưa lên mạng, gitignore sẵn)")
    _k = st.text_input("Gemini API key:", value="", type="password",
                       key="startup_key_input",
                       placeholder="AIza… hoặc AQ.…")
    _c1, _c2, _c3 = st.columns([1, 1, 2])
    with _c1:
        _save = st.button("💾 Lưu key", type="primary",
                          disabled=not (_k or "").strip())
    with _c2:
        _skip = st.button("Bỏ qua — dùng không có AI")
    if _save:
        with st.spinner("Đang kiểm tra key…"):
            _ok, _msg = AIA.validate_key(_k.strip())
        if _ok:
            (ROOT / "data" / "gemini_key.txt").write_text(_k.strip(),
                                                          encoding="utf-8")
            st.session_state["startup_key_dismissed"] = True
            st.success("✅ Đã lưu key — AI đã sẵn sàng!")
            time.sleep(1.2)
            st.rerun()
        else:
            st.error(f"❌ Key chưa dùng được ({_msg}) — kiểm tra lại chuỗi "
                     "key đã copy đủ chưa.")
    if _skip:
        st.session_state["startup_key_dismissed"] = True
        st.rerun()


if not _startup_key_present() and \
        not st.session_state.get("startup_key_dismissed"):
    _startup_key_dialog()


def _gemini_model_sel():
    """Model AI người dùng chọn (tab Đáp án & QR ④): session > file,
    None = dùng chuỗi mặc định của từng module."""
    v = st.session_state.get("gemini_model_sel")
    if v:
        return v
    _f = ROOT / "data" / "gemini_model.txt"
    try:
        return _f.read_text(encoding="utf-8").strip() or None
    except Exception:                                            # noqa
        return None


def load_registry():
    """Nạp registry key; tự loại key hệ thống cũ (builtin-gk1) và key thiếu file.
    Tự PHỤC HỒI key còn file nhưng thiếu entry registry (chống mất danh sách
    'Các đáp án đã lưu' sau khi khởi động lại/đồng bộ mà thiếu registry.json)."""
    reg = json.loads(REG.read_text(encoding="utf-8")) if REG.is_file() else {}
    changed = False
    for kid in list(reg):
        fname = str(reg[kid].get("file", ""))
        f = KEY_DIR / fname
        if kid == "builtin-gk1" or fname.endswith("builtin.json") or not f.is_file():
            reg.pop(kid)
            changed = True
    # Self-heal: file key_<id>.xlsx tồn tại nhưng chưa có trong registry
    for f in sorted(KEY_DIR.glob("key_*.xlsx")):
        kid = f.stem[4:]
        if kid and kid not in reg:
            try:
                _, _, desc = omr_key.detect_key_format(f)
            except Exception:                                    # noqa
                desc = ""
            reg[kid] = {"file": f.name, "desc": desc or f.name,
                        "name": desc or f.name,
                        "ts": datetime.fromtimestamp(f.stat().st_mtime)
                              .isoformat(timespec="seconds")}
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
                    ink=True, ink_only=False, alt_templates=None):
    """Chuẩn hoá nhiều ảnh về đúng mẫu + NHẬN DẠNG & TÔ ĐEN bubble đã tô.

    `key` = key_data đã nạp (dict có 'các đề') HOẶC đường dẫn file .xlsx.
    Trả (ok_paths, report_rows). Mỗi ảnh ghi 2 bản: *_chuan.jpg (dùng để chấm)
    và *_todam.jpg (bubble đã tô được tô đen để kiểm tra bằng mắt).

    CHẠY SONG SONG: >=3 ảnh thì chia cho tối đa 4 tiến trình con (module
    omr_webprep, fork — không re-execute app như spawn; worker chỉ dùng
    cv2/numpy, không đụng TF); lỗi thì rơi về chạy tuần tự như cũ.
    ok_paths/ink_paths LUÔN ĐỒNG BỘ THEO THỨ TỰ
    `paths` (ảnh fail -> None) — trước đây ảnh fail làm LỆCH CHỈ SỐ và các
    trang sau nó bị chấm nhầm ảnh của trang kế bên.
    """
    import cv2 as _cv
    if isinstance(key, (str, Path)):
        _fmt, key, _ = omr_key.detect_key_format(key)
    cells = CTM.keyed_cells(paper, tpl_path, key)
    # mẫu dự phòng cho TỰ NHẬN THEO TỪNG ẢNH (lô lẫn 10_11 + moi)
    alts = []
    for _p2, _tpl2 in (alt_templates or []):
        if _p2 == paper:
            continue
        try:
            alts.append((_p2, _tpl2, CTM.keyed_cells(_p2, _tpl2, key)))
        except Exception:                                    # noqa
            continue
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in paths]
    n = len(paths)
    ok_paths = [None] * n
    ink_paths = [None] * n
    rows = [None] * n
    extras = [None] * n
    papers = [paper] * n

    def _put(i, res):
        if res.get("ok"):
            ok_paths[i] = Path(res["ok_path"])
            ink_paths[i] = res.get("ink_path")
        rows[i] = res["row"]
        extras[i] = res["extras"]
        if res.get("extras", {}).get("paper_used"):
            papers[i] = res["extras"]["paper_used"]

    results = None
    if n >= 3:
        # ---- SONG SONG (4 tiến trình, giữ thứ tự ảnh vào) ----
        try:
            import multiprocessing as _mp
            import omr_webprep as _WP
            tasks = [{"path": str(p), "tpl": str(tpl_path), "paper": paper,
                      "cells": cells, "out": str(out_dir), "ink": bool(ink),
                      "ink_only": bool(ink_only), "alt": alts} for p in paths]
            n_w = min(4, max(1, _mp.cpu_count() or 2))
            # fork (KHÔNG spawn): spawn re-execute module chính — với webapp.py
            # nghĩa là chạy lại cả app Streamlit trong tiến trình con. fork chỉ
            # sao chép bộ nhớ; worker (prep_one) chỉ dùng cv2/numpy, không đụng
            # TF nên không vướng khoá của thread TF.
            ctx = _mp.get_context("fork")
            # 1 thread OpenCV mỗi tiến trình: n worker = n lõi, hết tranh chấp.
            # (2026-11-06: cũ là cpu_count-1 -> máy 4 lõi chỉ dùng 3 worker;
            # mỗi worker cv2 1 thread nên đủ 4 worker = vừa 4 lõi.)
            # PHẢI set ở CHA TRƯỚC KHI fork (set trong con = deadlock vì
            # thread-pool cv2 của cha bị fork ở trạng thái giữ khoá).
            _cv_thr = (_cv.getNumThreads()
                       if hasattr(_cv, "getNumThreads") else 0)
            _cv.setNumThreads(1)
            try:
                with ctx.Pool(n_w) as pool:
                    for i, res in enumerate(pool.imap(_WP.prep_one, tasks)):
                        if progress is not None:
                            progress.progress((i + 1) / max(1, n),
                                              text=f"Đang chuẩn hoá {i + 1}/{n} ảnh: "
                                                   f"{paths[i].name}")
                        results = True
                        _put(i, res)
            finally:
                # trả lại thread cho khâu chấm/vẽ (0 = mặc định dùng hết lõi)
                _cv.setNumThreads(_cv_thr)
        except Exception as _e:                              # noqa
            # bất ngờ (vd không fork được) -> tuần tự; ghi rõ lý do vào log
            import traceback as _tb
            print(f"[normalize_batch] nhánh song song lỗi ({_e!r}), "
                  "chuyển tuần tự:\n" + _tb.format_exc(), file=sys.stderr)
            results = None
            ok_paths = [None] * n
            ink_paths = [None] * n
            rows = [None] * n
            extras = [None] * n
    if results is None:
        # ---- TUẦN TỰ (dự phòng / <3 ảnh) — logic giữ nguyên từ bản cũ ----
        for i, ip in enumerate(paths, 1):
            if progress is not None:
                progress.progress(i / max(1, n),
                                  text=f"Đang chuẩn hoá {i}/{n} ảnh: {ip.name}")
            try:
                import omr_prep as PREP
                info = PREP.normalize_to_template(
                    ip, tpl_path, scale=PREP.norm_scale_for(paper),
                    cells=cells, ink=ink, ink_only=ink_only)
                _paper_used = paper
                if (not info.get("ok")) and alts and not ink_only:
                    # MARGIN +0.08 cho mẫu dự phòng (đồng bộ omr_webprep):
                    # phiếu SAI mẫu warp vào alt đạt ring suýt soát ngưỡng
                    # (bug thật lô 11a8: 10_11 layout mới vào moi 0.44-0.50
                    # vừa đủ qua cổng 0.50 -> chấm sai "tự tin")
                    for _p2, _tpl2, _cells2 in alts:
                        try:
                            info2 = PREP.normalize_to_template(
                                ip, _tpl2, scale=PREP.norm_scale_for(_p2),
                                cells=_cells2, ink=ink, ink_only=False)
                        except Exception:                    # noqa
                            continue
                        if info2.get("ok"):
                            _min2 = PREP.RING_MIN_PART.get(_p2, {})
                            _need = (_min2.get("ALL", PREP.RING_MIN_DEFAULT) + 0.08
                                     if "ALL" in _min2 else None)
                            if _need is not None and \
                                    float(info2.get("ring_rate") or 0.0) < _need:
                                continue
                            info, _paper_used = info2, _p2
                            break
            except Exception as e:                           # noqa
                info = {"ok": False, "quality": 0.0, "rot": 0,
                        "reason": f"lỗi xử lý: {e}"}
                _paper_used = paper
            _res_w = _WP_RESULT(ip, info, out_dir, ink_only)
            _res_w["extras"]["paper_used"] = _paper_used
            _put(i - 1, _res_w)
    return ok_paths, rows, ink_paths, extras


def _WP_RESULT(ip, info, out_dir, ink_only):
    """Đóng gói kết quả 1 ảnh theo đúng format omr_webprep.prep_one."""
    import cv2 as _cv
    if info.get("ok"):
        dst = out_dir / f"{Path(ip).stem}_chuan.jpg"
        _cv.imwrite(str(dst), info["warped"], [_cv.IMWRITE_JPEG_QUALITY, 95])
        ink_path = None
        if info.get("inked") is not None:
            td = out_dir / f"{Path(ip).stem}_todam.jpg"
            _cv.imwrite(str(td), info["inked"], [_cv.IMWRITE_JPEG_QUALITY, 95])
            ink_path = str(td)
        st_ = info.get("ink_stats") or {}
        return {"ok": True, "ok_path": str(dst), "ink_path": ink_path,
                "row": {"file": Path(ip).name,
                        "chuẩn hoá": "chỉ tô đậm" if ink_only else "✓",
                        "xoay": f"{info['rot']}°",
                        "khớp mẫu": round(info["quality"], 2),
                        "ô đã tô": st_.get("n_filled", "-"),
                        "ghi chú": ""},
                "extras": {"ids": info.get("ids_result"),
                           "quality": info.get("quality", 1.0),
                           "rot": info.get("rot", 0)}}
    return {"ok": False, "ok_path": None, "ink_path": None,
            "row": {"file": Path(ip).name, "chuẩn hoá": "✗", "xoay": "-",
                    "khớp mẫu": round(info.get("quality", 0.0), 2),
                    "ô đã tô": "-", "ghi chú": info.get("reason", "")},
            "extras": {"ids": None, "quality": 0.0, "rot": 0}}


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


LBL_KEY, LBL_SHEET, LBL_TPL = ("📋 Đáp án & QR", "🟩 Chấm theo mẫu",
                               "🎯 Mẫu giấy & lưới đáp án")
nav = st.radio("Tab:", [LBL_KEY, LBL_SHEET, LBL_TPL], horizontal=True,
               key="tab_nav", label_visibility="collapsed")
if nav == LBL_KEY and st.session_state.get("tab_prev") != LBL_KEY:
    # VỪA CHUYỂN sang tab '📋 Đáp án & QR' -> RESET TOÀN BỘ tab '🟩 Chấm
    # theo mẫu' về trạng thái ban đầu (xoá kết quả chấm, ảnh chuẩn hoá,
    # ảnh upload, link Drive, mã đề cố định; uploader ảnh mới hoàn toàn).
    for k_ in ("sheet_res", "prep_res", "prep_sig", "prep_cache",
               "gdrive_files", "gdrive_report", "gdrive_stamp",
               "paper_det", "paper_det_sig", "sheet_gdrive_links",
               "sheet_made", "prep_table", "sheet_res_table",
               "sheet_norm_zip"):
        st.session_state.pop(k_, None)
    st.session_state["sheet_norm"] = True
    st.session_state.pop(f"sheet_files_{st.session_state.get('round_no', 0)}",
                         None)
    st.session_state["round_no"] = st.session_state.get("round_no", 0) + 1
st.session_state["tab_prev"] = nav

# ---------------- TAB 0: MẪU GIẤY & LƯỚI ĐÁP ÁN ----------------
if nav == LBL_TPL:
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
            st.dataframe(_df_str([{"câu": r["câu"], "phần": r["phần"],
                                   "loại": r["loại"], "đáp án": r["đáp án"],
                                   "vị trí trên phiếu": r["vị trí"],
                                   "đề": r["đề"]} for r in grid["ô"]]),
                         width="stretch", hide_index=True)
            st.download_button("⬇️ Tải lưới đáp án (JSON)",
                               data=json.dumps(grid, ensure_ascii=False, indent=1),
                               file_name=f"luoi_dap_an_{tname}.json")
    if tpl.get("vùng") is None:
        st.warning("⚠️ Mẫu này chưa hiệu chuẩn hình học — chấm sẽ dùng auto-detect.")

# ---------------- TAB 1b: CHẤM THEO MẪU (FIDUCIAL) ----------------
if nav == LBL_SHEET:
    import json as _json
    import omr_sheetgrid as SG

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
    # MẪU TỰ NHẬN (KHÔNG hiện trong ① Mẫu giấy): mỗi ảnh fail mẫu chính sẽ
    # được thử TỰ ĐỘNG qua alt-fallback (normalize_batch alt_templates) rồi
    # chấm bằng ĐÚNG mẫu đó — người dùng không cần chọn tay.
    TPL_EXTRA = {
        "moi": ("Mẫu mới 2025 (20 câu ABCD + 4 Đ/S + 8 điền số) — tự nhận",
                str(ROOT / "data" / "maugiay" / "moi.png")),
    }
    TPL_ALL = {**TPL_CHOICES, **TPL_EXTRA}
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
    # 2026-11-05: hiển thị TÊN ĐÁP ÁN ĐÃ CHỌN kế bên "① Mẫu giấy:"
    # (các mục lưới đáp án / tải JSON / Mã đề nằm gọn trong '⚙️ Nâng cao'
    # bên dưới — chức năng vẫn nguyên vẹn)
    with c_keyf:
        if key_names:
            st.markdown("**② Đáp án đã chọn:**  \n" + " · ".join(key_names))
        else:
            st.caption("② Chưa chọn đáp án — chọn dòng trong bảng "
                       "'Các đáp án đã lưu' ở tab '📋 Đáp án & QR'.")
    # mặc định: hệ thống tự chọn mã đề (đọc lưới / suy đáp án)
    fixed_made = ""    # (nếu nhập trong '⚙️ Nâng cao' thì dùng giá trị đó)
    if not Path(tpl_path).is_file():
        st.error(f"Không thấy ảnh mẫu: {tpl_path}")
    elif _key_sel is None:
        pass    # nút Chấm đã tự khoá khi thiếu key
    else:
        try:
            cells = SG.build_template_cells(paper, tpl_path)
            id_cells = SG.build_id_cells(paper, tpl_path)
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Số đề trong key", len(_key_sel["các đề"]))
            cc2.metric("Ô PHẦN I", len([c for c in cells if c[0] == "P1"]))
            cc3.metric("Ô PHẦN II", len([c for c in cells if c[0] == "P2"]))
            cc4.metric("Ô SBD/Mã đề", len(id_cells))
            with st.expander("⚙️ Nâng cao", expanded=False):
                de0 = list(_key_sel["các đề"])[0]
                g0 = SG.build_answer_grid(paper, tpl_path,
                                          _key_sel["các đề"][de0])
                rows0 = [{"phần": p,
                          "câu": (q if isinstance(q, int) else f"{q[0]}{q[1]}"),
                          "lựa chọn": o, "đáp án": g0["answers"].get((p, q, o)),
                          "toạ độ mẫu": f"({x:.0f},{y:.0f})"}
                         for (p, q, o, x, y) in cells[:80]]
                st.dataframe(_df_str(rows0), width="stretch",
                             hide_index=True)
                st.download_button("⬇️ Tải lưới đáp án đầy đủ (JSON)",
                                   data=_json.dumps(
                                       [{"phần": p, "câu": str(q),
                                         "lựa_chọn": o,
                                         "đáp_án": g0["answers"].get((p, q, o)),
                                         "x": round(x, 1), "y": round(y, 1)}
                                        for (p, q, o, x, y) in cells],
                                       ensure_ascii=False, indent=1),
                                   file_name=f"luoi_dap_an_{paper}_{de0}.json")
                fixed_made = st.text_input(
                    "Mã đề (bỏ trống = tự chọn)", value="", key="sheet_made")
                import os as _os
                # key mặc định: biến môi trường > data/gemini_key.txt
                _kfile = ROOT / "data" / "gemini_key.txt"
                _saved = _kfile.read_text(encoding="utf-8").strip() \
                    if _kfile.is_file() else ""
                _env_key = _os.environ.get("GEMINI_API_KEY", "") or _saved
                st.text_input(
                    "🔑 Gemini API key — nhận dạng CHỮ VIẾT TAY (SBD + STT)",
                    value=_env_key, type="password", key="gemini_key",
                    help="MẶC ĐỊNH đã cấu hình sẵn. Muốn đổi: lấy key miễn phí "
                         "tại aistudio.google.com/apikey (đăng nhập Google → "
                         "Get API key → copy) rồi dán vào đây — hoặc đặt biến "
                         "môi trường GEMINI_API_KEY. Có key: mỗi bài được đọc "
                         "SBD + STT viết tay bằng AI Gemini. Không có key: chỉ "
                         "dùng CNN nội bộ (đọc SBD ô in).")
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
                st.dataframe(_df_str(report), width="stretch",
                             hide_index=True)
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
    # (fixed_made: nếu key đã chọn thì lấy từ mục '⚙️ Nâng cao' ở trên; nếu
    #  chưa chọn key thì rỗng = hệ thống tự chọn mã đề)
    norm_on = st.checkbox(
        "⓪ Chuẩn hoá ảnh về đúng mẫu trước khi chấm (tự xoay 4 hướng + khử nghiêng/phối cảnh)",
        value=True, key="sheet_norm",
        help="Ảnh upload / Drive / điện thoại được đưa về đúng khung mẫu giấy đã chọn "
             "(cùng kích thước, cùng góc) rồi mới chấm. Ảnh nào không khớp mẫu sẽ bị báo, "
             "không bao giờ chuẩn hoá bừa.")
    # ---- ⓪ TIỀN XỬ LÝ TỰ ĐỘNG: chuẩn hoá + nhận dạng & TÔ ĐEN bubble ----
    # Nguồn ảnh = ảnh upload + ảnh đã lấy từ Google Drive
    gdrive_files = st.session_state.get("gdrive_files") or []
    # ---- TỰ DÒ MẪU GIẤY theo ảnh đầu tiên: nếu mẫu đang chọn KHÔNG khớp mà
    #      mẫu kia khớp được (corner/lưới xác nhận) thì tự chuyển, khỏi báo lỗi
    #      'không khớp mẫu' chỉ vì quên đổi selectbox (vd phiếu 10_11 đang để
    #      mẫu 12). Chỉ ghi đè khi có mẫu XÁC NHẬN ĐƯỢC (ok=True).
    _first = files_s[0] if files_s else (Path(gdrive_files[0])
                                         if gdrive_files else None)
    if _first is not None and len(TPL_CHOICES) > 1:
        try:
            # KHOÁ CACHE THEO NỘI DUNG (tên + kích thước file) — KHÔNG dùng
            # str(object): địa chỉ bộ nhớ bị reuse giữa 2 lần upload liên tiếp
            # -> detection CŨ (vd mẫu 12 của lô trước) áp NHẦM cho lô mới
            # (lỗi thật 19:46/19:50: lô test11 10_11 bị ép mẫu 12 -> 0/12).
            if hasattr(_first, "getvalue"):
                _sigd = (str(getattr(_first, "name", "")),
                         len(_first.getvalue()))
            else:
                _sigd = (str(_first), _first.stat().st_size)
        except Exception:                                    # noqa
            _sigd = str(_first)
        if st.session_state.get("paper_det_sig") != _sigd:
            import omr_prep as _PREP
            _det, _det_q = paper, -1.0
            for _p, (_lbl, _tp) in TPL_CHOICES.items():
                try:
                    _i = _PREP.normalize_to_template(_first, _tp, scale=1.0,
                                                     ink=False)
                    _s = (float(_i.get("quality") or 0.0)
                          if _i.get("ok") else -1.0)
                except Exception:                            # noqa
                    _s = -1.0
                if _s > _det_q:
                    _det, _det_q = _p, _s
            st.session_state["paper_det_sig"] = _sigd
            st.session_state["paper_det"] = _det
        _pd = st.session_state.get("paper_det")
        if _pd and _pd != paper and _pd in TPL_CHOICES:
            st.info(f"📌 Ảnh upload là phiếu mẫu '{TPL_CHOICES[_pd][0]}' — "
                    "hệ thống TỰ CHUYỂN mẫu giấy (bỏ chọn mẫu cũ).")
            paper = _pd
            tpl_path = TPL_CHOICES[paper][1]
    prep_ready = (files_s or gdrive_files) and Path(tpl_path).is_file() \
        and _key_sel is not None
    with st.expander("⓪ Ảnh đã chuẩn hoá & tô đậm bubble (bấm dòng để so sánh)",
                     expanded=bool(files_s or gdrive_files)):
        if not prep_ready:
            st.info("Tải ảnh lên / dán link Google Drive và chọn mẫu giấy + đáp án để bắt đầu.")
        else:
            sig = (tuple((u.name, len(u.getvalue())) for u in files_s),
                   tuple(gdrive_files), paper, tuple(kids_sel), bool(norm_on))
            # ---- CACHE kết quả chuẩn hoá THEO TỔ HỢP LỰA CHỌN ----
            # Chọn lại dòng key ở tab 'Đáp án & QR' (hoặc selection mất khi
            # chuyển tab) làm prep_sig đổi -> trước đây chuẩn hoá CHẠY LẠI
            # TỪ ĐẦU và xoá sạch nội dung tab 'Chấm theo mẫu'. Giờ mỗi tổ hợp
            # đã tính được lưu lại (tối đa 4) — quay lại lựa chọn cũ là có NGAY.
            prep_cache = st.session_state.setdefault("prep_cache", {})
            if st.session_state.get("prep_sig") != sig:
                if sig in prep_cache:
                    st.session_state["prep_sig"] = sig
                    st.session_state["prep_res"] = prep_cache[sig]
                else:
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
                    n_p = len(pths)
                    pr = st.progress(0.0,
                                     text=(f"Đang chuẩn hoá 0/{n_p} ảnh..." if norm_on
                                           else f"Đang tô đậm bubble 0/{n_p} ảnh..."))
                    # CHẠY TRỰC TIẾP trong script thread để cập nhật progress
                    # "i/N: tên ảnh" (Streamlit chỉ cho update widget từ script
                    # thread — bản run_with_timeout trước đây làm mất progress;
                    # an toàn vì bug treo đã sửa tận gốc + chuẩn hoá chạy song song)
                    _alts = [(p2, TPL_ALL[p2][1]) for p2 in TPL_ALL
                             if p2 != paper]
                    try:
                        ok_paths, rep, ink_paths, extras = normalize_batch(
                            pths, tpl_path, paper, _key_sel,
                            destn / "chuan_hoa", pr, ink=True,
                            ink_only=not norm_on, alt_templates=_alts)
                    except Exception as e:                           # noqa
                        pr.empty()
                        st.error(f"⓪ Chuẩn hoá lỗi: {e} — thử lại; nếu lặp lại "
                                 "hãy chụp lại ảnh rõ hơn (đủ 4 mốc đen ở góc).")
                        st.stop()
                    pr.empty()
                    # ---- PHÒNG VỆ: chuẩn hoá THẤT BẠI TOÀN BỘ + template
                    #      đang dùng khác template người dùng chọn ban đầu
                    #      (do tự-dò chuyển) -> thử lại với template KIA ----
                    _alt = [k for k in TPL_ALL if k != paper]
                    if not any(ok_paths) and _alt and len(TPL_ALL) > 1:
                        _alt_p = _alt[0]
                        pr.progress(0.0, text=(
                            f"Không khớp mẫu {TPL_CHOICES[paper][0]} — thử "
                            f"{TPL_CHOICES[_alt_p][0]}..."))
                        ok_paths, rep, ink_paths, extras = normalize_batch(
                            pths, TPL_CHOICES[_alt_p][1], _alt_p, _key_sel,
                            destn / "chuan_hoa", pr, ink=True,
                            ink_only=not norm_on)
                        pr.empty()
                        if any(ok_paths):
                            paper, tpl_path = _alt_p, TPL_CHOICES[_alt_p][1]
                            st.session_state["paper_det"] = _alt_p
                            st.info(f"📌 Đã tự chuyển sang mẫu "
                                    f"'{TPL_CHOICES[_alt_p][0]}' vì ảnh chỉ "
                                    "khớp mẫu này.")
                    # ---- 🤖 GEMINI RESCUE: ảnh ⓪ không chuẩn hoá được ----
                    # Gemini chỉ các Ô GÓC khung ngoài (toạ độ qua lưới tham
                    # chiếu) -> kiểm chứng hình học -> H khởi đầu cho
                    # normalize (ICP + corner/ring gate VẪN quyết định — AI
                    # không thể ép phiếu sai mẫu vào kết quả). Kết quả H được
                    # cache theo (tên, cỡ) để rerun không tốn API.
                    _fails = [i for i, x in enumerate(ok_paths) if not x]
                    _gkey_r = (st.session_state.get("gemini_key")
                               or AIA.load_key())
                    _ai_resc = st.session_state.setdefault("ai_rescue_cache", {})
                    _n_resc = 0
                    if _fails and _gkey_r and len(_fails) <= 12:
                        pr.progress(0.0, text=(f"🤖 Gemini cứu định vị "
                                               f"0/{len(_fails)} ảnh lỗi..."))
                        for _j, _i in enumerate(_fails):
                            _p = pths[_i]
                            try:
                                _ck = (_p.name, _p.stat().st_size)
                            except Exception:                    # noqa
                                _ck = (_p.name, 0)
                            _Hl = _ai_resc.get(_ck, False)   # False = chưa gọi
                            if _Hl is False:
                                _Hl = None
                                try:
                                    _img_r = PREP.load_bgr(_p)
                                    _H, _nb = AIA.rescue_homography(
                                        _img_r, tpl_path, _gkey_r,
                                        model=_gemini_model_sel())
                                    _Hl = _H.tolist() if _H is not None else None
                                except Exception:                # noqa
                                    _Hl = None
                                _ai_resc[_ck] = _Hl
                            _H = np.array(_Hl, np.float64) if _Hl else None
                            if _H is not None:
                                try:
                                    _info_r = PREP.normalize_to_template(
                                        _p, tpl_path, paper=paper, cells=cells,
                                        scale=PREP.norm_scale_for(paper),
                                        ink=True, init_H=_H)
                                except Exception:                # noqa
                                    _info_r = None
                                if _info_r and _info_r.get("ok"):
                                    _r = _WP_RESULT(_p, _info_r,
                                                    destn / "chuan_hoa", False)
                                    if _r["ok"]:
                                        ok_paths[_i] = _r["ok_path"]
                                        ink_paths[_i] = _r["ink_path"]
                                        _row_r = _r["row"]
                                        _row_r["ghi chú"] = "🤖 Gemini cứu định vị"
                                        rep[_i] = _row_r
                                        extras[_i] = _r["extras"]
                                        _n_resc += 1
                            pr.progress((_j + 1) / len(_fails),
                                        text=(f"🤖 Gemini cứu định vị "
                                              f"{_j + 1}/{len(_fails)} ảnh lỗi..."))
                        pr.empty()
                        if _n_resc:
                            st.caption(f"🤖 Gemini đã cứu định vị {_n_resc}/"
                                       f"{len(_fails)} ảnh lỗi (vẫn qua đủ "
                                       "kiểm chứng ring trước khi chấm).")
                    st.session_state["prep_sig"] = sig
                    st.session_state["prep_res"] = {
                        "rows": rep,
                        "ok": [str(x) if x else None for x in ok_paths],
                        "ink": [x if x else None for x in ink_paths], "orig": [str(x) for x in pths],
                        "extras": extras,
                        "out": str(destn / "chuan_hoa"), "stamp": stamp_n}
                    prep_cache[sig] = st.session_state["prep_res"]
                    while len(prep_cache) > 4:             # giữ tối đa 4 tổ hợp
                        prep_cache.pop(next(iter(prep_cache)))
            res = st.session_state.get("prep_res")
            if res:
                # ---- THỐNG KÊ chuẩn hoá: tổng / đã chuẩn hoá / không chuẩn hoá ----
                _n_total = len(res.get("orig") or [])
                _n_ok = sum(1 for x in (res.get("ok") or []) if x)
                _n_fail = _n_total - _n_ok
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Tổng số ảnh", _n_total)
                sm2.metric("Đã chuẩn hoá",
                           f"{_n_ok}/{_n_total}",
                           delta=f"{_n_ok / _n_total:.0%}" if _n_total else None,
                           delta_color="off")
                sm3.metric("Không chuẩn hoá được",
                           _n_fail,
                           delta=f"{_n_fail / _n_total:.0%}" if _n_total else None,
                           delta_color="inverse")
                st.caption("👉 Bấm vào một dòng để xem Ảnh gốc và ảnh đã "
                           "Chuẩn hoá / Tô đậm bên dưới.")
                psel = st.dataframe(_df_str(res["rows"]), width="stretch",
                                    hide_index=True,
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
                    for pth in map(Path, [x for x in res["ok"] if x]):
                        if not Path(pth).is_file():
                            continue
                        z.write(pth, pth.name)
                        td = pth.parent / f"{pth.stem.replace('_chuan', '')}_todam.jpg"
                        if td.is_file():
                            z.write(td, td.name)
                st.download_button("⬇️ Tải ZIP ảnh đã chuẩn hoá / tô đậm",
                                   data=buf.getvalue(),
                                   file_name=f"anh_chuan_hoa_{res['stamp']}.zip",
                                   key="sheet_norm_zip")

    _gkey_ui = st.session_state.get("gemini_key") or AIA.load_key()
    st.checkbox(
        "🤖 Chấm trực tiếp bằng AI cho ảnh không chuẩn hoá được",
        value=bool(_gkey_ui), key="ai_direct_on",
        help=("Ảnh mờ/nghiêng nặng không thể căn mẫu: Gemini đọc SBD + Mã đề "
              "+ đáp án PHẦN I (ABCD) và PHẦN II (Đúng/Sai) ngay trên ảnh rồi "
              "so với đáp án đã chọn. Dòng kết quả đánh dấu 'nguồn = AI' — "
              "nên rà soát tay. Cần Gemini API key (⚙️ Nâng cao)."))
    c_run, c_new, _ = st.columns([1.2, 1.2, 2.6])
    with c_new:
        if st.button("🆕 Chấm đợt khác", key="sheet_new_round",
                     help="Xoá kết quả đợt hiện tại và nạp ảnh của đợt mới; "
                          "giữ nguyên mẫu giấy + đáp án đã chọn."):
            st.session_state["round_no"] = st.session_state.get("round_no", 0) + 1
            for k_ in ("sheet_res", "prep_res", "prep_sig", "prep_cache",
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
        # ---- 🤖 CHẤM TRỰC TIẾP BẰNG AI cho ảnh ⓪ không chuẩn hoá được ----
        # (kể cả sau Gemini rescue — AI ĐỌC NỘI DUNG, thế mạnh thật của VLM;
        #  kết quả cache theo (tên, cỡ) nên rerun không tốn API)
        _gkey_g = st.session_state.get("gemini_key") or AIA.load_key()
        _ai_direct = {}
        if (st.session_state.get("ai_direct_on") and _gkey_g and _key is not None):
            _fails_g = [k_ for k_, x in enumerate(path_ok) if not x]
            if _fails_g and len(_fails_g) <= 12:
                _ai_cache = st.session_state.setdefault("ai_read_cache", {})
                _prg = st.progress(0.0, text=(f"🤖 AI đọc trực tiếp "
                                              f"0/{len(_fails_g)} ảnh..."))
                for _j, _i in enumerate(_fails_g):
                    _orig = Path(res_p["orig"][_i])
                    try:
                        _ck = (_orig.name, _orig.stat().st_size)
                    except Exception:                            # noqa
                        _ck = (_orig.name, 0)
                    if _ck in _ai_cache:
                        _rec = _ai_cache[_ck]
                    else:
                        _rec = None
                        try:
                            _img_g = PREP.load_bgr(_orig)
                            _ans = AIA.read_answers(_img_g, _gkey_g,
                                                    model=_gemini_model_sel())
                            _sc = AIA.grade_ai(_ans, _key)
                            if _sc:
                                _rec = {"sbd": _ans.get("sbd"),
                                        "made": _sc["made"],
                                        "note": _sc["note_de"],
                                        "p1": f"{_sc['p1_ok']}/{_sc['p1_n']}",
                                        "p2": f"{_sc['p2_ok']}/{_sc['p2_n']}",
                                        "tổng": _sc["diem"]}
                        except Exception as _e:                  # noqa
                            st.caption(f"🤖 AI đọc {_orig.name} lỗi: "
                                       f"{str(_e)[:70]}")
                        _ai_cache[_ck] = _rec
                    if _rec:
                        _ai_direct[_orig.name] = _rec
                    _prg.progress((_j + 1) / len(_fails_g),
                                  text=(f"🤖 AI đọc trực tiếp "
                                        f"{_j + 1}/{len(_fails_g)} ảnh..."))
                _prg.empty()
        prog = st.progress(0.0, text=f"Đang chấm 0/{len(paths)} bài...")
        outd = OUTDIR / "theo_mau" / f"web_{stamp}"
        outd.mkdir(parents=True, exist_ok=True)
        rows_out, details, rendered, used = [], [], [], set()
        name_by_key, dup_counts = {}, {}   # (SBD, Mã đề) -> tên chuẩn / số lần
        row_meta = []     # song song với rows_out: ảnh kết quả + mã đề + SBD
        for i, ip in enumerate(paths, 1):
            # tên hiển thị = tên ẢNH GỐC (bỏ hậu tố _todam/_chuan của bước ⓪)
            disp = Path(res_p["orig"][i - 1]).name if (i - 1) < len(res_p["orig"]) \
                else ip.name
            prog.progress(i / max(1, len(paths)),
                          text=f"Đang chấm {i}/{len(paths)} bài: {disp}")
            # Ảnh vào đã chuẩn hoá ở bước ⓪ và SBD/Mã đề đã đọc sẵn -> chấm
            # trực tiếp, KHÔNG căn lại / KHÔNG quét ID lần 2 (nhanh gấp đôi)
            ex = res_p.get("extras") or []
            ex_i = ex[i - 1] if (i - 1) < len(ex) else {}
            # TỰ NHẬN MẪU THEO TỪNG ẢNH: ảnh lẫn 10_11/moi -> chấm ĐÚNG mẫu
            paper_i = (ex_i.get("paper_used") or paper)
            tpl_i = TPL_ALL[paper_i][1] if paper_i in TPL_ALL \
                else tpl_path
            if not path_ok[i - 1]:
                # LÝ DO CHUẨN HOÁ THẬT từ bảng ⓪ (match theo tên file gốc)
                _prep_reason = ""
                try:
                    for _pr in (res_p.get("rows") or []):
                        if _pr.get("file") == disp:
                            _prep_reason = str(_pr.get("ghi chú") or "")
                            break
                except Exception:                            # noqa
                    pass
                _ai = _ai_direct.get(disp)
                if _ai:
                    # 🤖 AI đã đọc trực tiếp: vào BẢNG CHÍNH với nguồn 'AI'
                    rows_out.append({"file": disp, "chuẩn hoá": "🤖 AI",
                                     "SBD": _ai.get("sbd") or "-", "STT": "-",
                                     "mã đề": _ai["made"], "nguồn": _ai["note"],
                                     "khớp mẫu": "-",
                                     "P1": _ai["p1"], "P2": _ai["p2"],
                                     "tổng": _ai["tổng"], "sai": "-",
                                     "trống/nhiều": "-",
                                     "ghi chú": ("🤖 chấm trực tiếp bằng AI — "
                                                 "ảnh không chuẩn hoá được, "
                                                 "nên rà soát tay")})
                    row_meta.append({"img": None, "sbd": _ai.get("sbd"),
                                     "stt": "-", "made": _ai["made"],
                                     "file": disp, "ai": True})
                    continue
                rows_out.append({"file": disp, "chuẩn hoá": "✗", "SBD": "-",
                                 "mã đề": "-", "nguồn": "-", "khớp mẫu": 0.0,
                                 "P1": "-", "P2": "-", "tổng": 0, "sai": 0,
                                 "trống/nhiều": 0,
                                 "ghi chú": (_prep_reason or
                                             "bước ⓪ không chuẩn hoá được")})
                row_meta.append({"img": None, "sbd": None, "made": None, "file": disp})
                continue
            try:
                r = run_with_timeout(
                    CTM.grade_prepared, 120, ip, paper_i, tpl_i, _key,
                    ids=ex_i.get("ids"), made=(fixed_made or None),
                    quality=ex_i.get("quality", 1.0))
            except _StepTimeout as e:
                rows_out.append({"file": disp, "chuẩn hoá": "-", "SBD": "-",
                                 "mã đề": "-", "nguồn": "-", "khớp mẫu": 0.0,
                                 "P1": "-", "P2": "-", "tổng": 0, "sai": 0,
                                 "trống/nhiều": 0,
                                 "ghi chú": f"⚠ chấm quá lâu ({e}) — bỏ qua trang"})
                row_meta.append({"img": None, "sbd": None, "made": None,
                                 "file": disp})
                continue
            if r is None:
                rows_out.append({"file": disp, "chuẩn hoá": "✗", "SBD": "-", "mã đề": "-",
                                 "nguồn": "-", "khớp mẫu": 0.0, "P1": "-", "P2": "-",
                                 "tổng": 0, "sai": 0, "trống/nhiều": 0,
                                 "ghi chú": "không khớp mẫu (chụp rõ 4 góc)"})
                row_meta.append({"img": None, "sbd": None, "made": None, "file": disp})
                continue
            # ---- NHẬN DẠNG CHỮ VIẾT TAY (SBD + STT) — omr_hwr ----
            # Có Gemini key: đọc cả 2 trường bằng AI (cắt dải đầu trang phiếu).
            # Không key: CNN nội bộ chỉ khi bubble không đọc được SBD.
            _gem = st.session_state.get("gemini_key") or ""
            _hwr = {"SBD": None, "STT": None, "engine": "none"}
            try:
                import omr_hwr as _HWR
                _hwr = _HWR.read_fields(r["img"], tpl_i, paper_i,
                                        api_key=(_gem or None),
                                        model=_gemini_model_sel())
            except Exception as _e:                          # noqa
                st.caption(f"⚠ đọc chữ viết tay lỗi: {_e}")
            if _hwr.get("SBD") and not r.get("sbd"):
                r["sbd"] = _hwr["SBD"]
                r["sbd_source"] = ("viết tay (AI)" if _hwr.get("engine") == "gemini"
                                   else "viết tay")
            _stt = _hwr.get("STT") or "-"
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
            # ĐỔI TÊN CẢ `stem` (trước đây chỉ đổi outp => khi 2 trang cùng
            # mã đề (phiếu 10_11 không SBD -> NA_<đề>_<giờ>) được chấm trong
            # CÙNG 1 GIÂY thì `stem in used` đúng mãi -> VÒNG LẶP VÔ HẠN
            # 100 % CPU — đúng 2 lần treo 17:16 và 22:29).
            base_stem, n = stem, 1
            while stem in used or outp.exists():
                stem = f"{base_stem}_{n}"
                outp = outd / f"{stem}.jpg"
                n += 1
            used.add(stem)
            try:
                run_with_timeout(
                    SG.render_result, 120, r["img"], r["decided"], r["detail"],
                    r["stats"], outp, paper_i, made=r["made"], s1=r["p1"],
                    s2=r["p2"], sbd=r.get("sbd"), ids=r.get("ids"),
                    show_sbd=True, md_inferred=(r.get("made_source") != "lưới"),
                    md_note="", tpl_path=tpl_i)
            except _StepTimeout:
                outp = None
            if outp is not None:
                rendered.append(outp)
            n_wrong = sum(1 for v in r["detail"].values() if v[0] == "wrong")
            n_unk = sum(1 for v in r["detail"].values() if v[0] in ("blank", "multi"))
            pn = r.get("pre_norm") or {}
            norm_tag = ("↻%d° q%.2f" % (pn.get("rot", 0), pn.get("quality", 0))) if pn else "—"
            rows_out.append({"file": disp, "chuẩn hoá": norm_tag,
                             "SBD": r.get("sbd") or "-", "STT": _stt,
                             "mã đề": r["made"], "nguồn": r["made_source"],
                             "khớp mẫu": r["quality"],
                             "P1": f"{r['stats']['P1']['ok']}/{r['stats']['P1']['n']}",
                             "P2": f"{r['stats']['P2']['ok']}/{r['stats']['P2']['n']}",
                             "tổng": r["score"], "sai": n_wrong, "trống/nhiều": n_unk,
                             "ghi chú": ""})
            details.append({"file": disp, "made": r["made"], "detail": r["detail"]})
            row_meta.append({"img": (str(outp) if outp else None),
                             "sbd": r.get("sbd"), "stt": _stt,
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
        # ---- TÁCH DANH SÁCH: chấm thường / 🤖 AI trực tiếp / KHÔNG chấm ----
        _graded_rows, _graded_meta_idx = [], []
        _ai_rows, _failed_rows = [], []
        for _i, _r in enumerate(rows_out):
            if _i < len(row_meta) and row_meta[_i].get("img"):
                _graded_rows.append(_r)
                _graded_meta_idx.append(_i)
            elif _i < len(row_meta) and row_meta[_i].get("ai"):
                _ai_rows.append(_r)
            else:
                _failed_rows.append({
                    "file": _r.get("file", ""),
                    "SBD": _r.get("SBD", "-"),
                    "mã đề": _r.get("mã đề", "-"),
                    "lý do không chấm được":
                        (_r.get("ghi chú") or "không xác định")})
        # tách nhỏ: KHÔNG chuẩn hoá được (bỏ qua khi chấm) vs lỗi khi chấm
        _failed_norm = [f_ for f_ in _failed_rows
                        if "chuẩn hoá" in f_["lý do không chấm được"].lower()
                        or "khớp mẫu" in f_["lý do không chấm được"].lower()
                        or "trắng" in f_["lý do không chấm được"].lower()
                        or "bám sát" in f_["lý do không chấm được"].lower()]
        _failed_other = [f_ for f_ in _failed_rows if f_ not in _failed_norm]
        st.success(f"Đã chấm {len(_graded_rows)} ảnh"
                   + (f" + 🤖 {len(_ai_rows)} ảnh chấm trực tiếp bằng AI"
                      if _ai_rows else "")
                   + " (bỏ qua "
                   f"{len(_failed_norm)} ảnh không chuẩn hoá được"
                   + (f", {len(_failed_other)} ảnh lỗi khác"
                      if _failed_other else "") + f") — lưu tại {res['outd']}")
        if _ai_rows:
            st.warning(f"🤖 {len(_ai_rows)} ảnh KHÔNG chuẩn hoá được nhưng AI "
                       "đã đọc trực tiếp đáp án trên ảnh — kết quả mang tính "
                       "tham khảo, NÊN RÀ SOÁT TAY:")
            st.dataframe(_df_str(_ai_rows), width="stretch", key="ai_tbl")
        if _failed_rows:
            st.warning(f"⚠️ Có {len(_failed_rows)} ảnh không chấm được — "
                       "danh sách riêng bên dưới.")
        st.caption("👉 Bấm vào một dòng để xem ảnh kết quả bên dưới. "
                   "Màu trên ảnh: **xanh = đúng**, **đỏ = sai**, **vàng = trống/tô nhiều**, "
                   "**cam = đáp án đúng của câu đó**, **xanh dương = mã đề suy từ đáp án**.")
        sel = st.dataframe(_df_str(_graded_rows), width="stretch", hide_index=True,
                           on_select="rerun", selection_mode="single-row",
                           key="sheet_res_table")
        picked = list(getattr(sel.selection, "rows", []) or []) \
            if hasattr(sel, "selection") else []
        # ánh xạ chỉ số bảng (chỉ các trang đã chấm) -> chỉ số row_meta gốc
        if picked:
            picked = [_graded_meta_idx[p] for p in picked
                      if p < len(_graded_meta_idx)]
        if _failed_rows:
            with st.expander(f"⛔ Danh sách {len(_failed_rows)} ảnh KHÔNG chấm "
                             "được (bấm để xem)", expanded=bool(_failed_rows)):
                if _failed_norm:
                    st.markdown(f"**❌ Không chuẩn hoá được — đã bỏ qua khi chấm "
                                f"({len(_failed_norm)} ảnh):**")
                    st.dataframe(_df_str(_failed_norm), width="stretch",
                                 hide_index=True, key="sheet_failed_table")
                if _failed_other:
                    st.markdown(f"**⚠️ Lỗi khác trong lúc chấm "
                                f"({len(_failed_other)} ảnh):**")
                    st.dataframe(_df_str(_failed_other), width="stretch",
                                 hide_index=True, key="sheet_failed_other")
                st.download_button(
                    "⬇️ Tải danh sách ảnh không chấm được (CSV)",
                    data="\n".join(["file,SBD,mã đề,lý do"] +
                                   [f'{f_["file"]},{f_["SBD"]},{f_["mã đề"]},'
                                    f'"{f_["lý do không chấm được"]}"'
                                    for f_ in _failed_rows]),
                    file_name=f"anh_khong_cham_duoc_{stamp}.csv",
                    mime="text/csv", key="sheet_failed_csv")
        if details:
            with st.expander("🔎 Vì sao ra điểm này? (chi tiết từng câu)", expanded=False):
                d0 = details[0]
                wrong = [(p_, q_, exp, got) for (p_, q_), (st, exp, got)
                         in d0["detail"].items() if st in ("wrong", "blank", "multi")]
                if wrong:
                    st.caption(f"{d0['file']}: " + " · ".join(
                        f"{p_}{q_}: key {exp} / làm {got or '—'}" for p_, q_, exp, got in wrong[:12]))
                st.dataframe(_df_str([{"phần": p_,
                                       "câu": (q_ if isinstance(q_, int)
                                               else f"{q_[0]}{q_[1]}"),
                                       "trạng thái": CTM.STATE_VI.get(st, st),
                                       "đáp án": exp, "bài làm": got}
                                      for (p_, q_), (st, exp, got)
                                      in d0["detail"].items()]),
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
        idx = picked[0] if picked and picked[0] < len(row_meta) else \
            (_graded_meta_idx[0] if _graded_meta_idx else 0)
        meta = row_meta[idx] if row_meta else None
        if meta and meta.get("img") and Path(meta["img"]).is_file():
            parts = [f"**Mã đề: {meta['made']}**"]
            if paper == "12" and meta.get("sbd"):
                parts.append(f"**SBD: {meta['sbd']}**")
            if str(meta.get("stt") or "-") != "-":
                parts.append(f"**STT: {meta['stt']}**")
            st.markdown(" · ".join(parts))
            st.image(meta["img"], caption=Path(meta["img"]).name, width="stretch")


# ---------------- TAB 2: ĐÁP ÁN & QR ----------------
if nav == LBL_KEY:
    # (đã ẩn dòng hướng dẫn "1) Đặt tên… — 2) sinh/quét QR — 3) chọn dòng key…"
    # theo yêu cầu 2026-11-05: "ẩn nội dung trong hình")
    reg = load_registry()

    # ---- ① TÊN ĐÁP ÁN + UPLOAD ----
    st.markdown("**① Nạp đáp án từ Excel** — đặt tên rồi upload file key:")
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

    # ---- ② BẢNG KEY ĐÃ LƯU (TT | Tên đáp án | Ngày tạo) ----
    st.markdown("**② Các đáp án đã lưu** — chọn dòng để nạp vào lần chấm:")
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
    ev = st.dataframe(_df_str(df_rows), width="stretch", hide_index=True,
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

    # ---- ③ QR CHIA SẺ ĐÁP ÁN ----
    st.markdown("**③ QR chia sẻ bộ đáp án** — quét QR trên máy khác để nạp "
                "cùng bộ key:")
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

    # ---- ④ GEMINI API KEY + MODEL (AI hỗ trợ chuẩn hoá & chấm) ----
    import os as _os4
    _kf4 = ROOT / "data" / "gemini_key.txt"
    _saved4 = _kf4.read_text(encoding="utf-8").strip() \
        if _kf4.is_file() else ""
    _key_default = _os4.environ.get("GEMINI_API_KEY", "") or _saved4
    st.markdown("**④ 🔑 Gemini API key & model AI** — hỗ trợ: 🤖 chấm trực "
                "tiếp ảnh không chuẩn hoá được, 🤖 cứu định vị ảnh lỗi, "
                "đọc SBD + STT viết tay.")
    with st.expander("📖 Hướng dẫn lấy API key miễn phí (chi tiết từng bước)"):
        st.markdown(
            "1. Mở trình duyệt vào **[aistudio.google.com/apikey]"
            "(https://aistudio.google.com/apikey)** và đăng nhập tài khoản "
            "Google (bất kỳ tài khoản nào cũng được).\n"
            "2. Bấm nút **“Create API key”** (Tạo khóa API).\n"
            "3. Chọn project mặc định (hoặc tạo mới) — hệ thống sinh chuỗi "
            "key dạng `AIza…` hoặc `AQ.…`.\n"
            "4. Bấm **Copy** (sao chép) chuỗi key.\n"
            "5. Dán vào ô **“Gemini API key”** bên dưới → bấm **💾 Lưu & "
            "kiểm tra key** — app tự kiểm tra key có thật trước khi lưu vào "
            "`data/gemini_key.txt` (không đưa lên mạng).\n\n"
            "**Lưu ý:** gói miễn phí giới hạn số lần gọi mỗi phút/ngày — "
            "khi hiện lỗi 429 nghĩa là đã tạm hết lượt, chờ vài phút hoặc "
            "ngày mai dùng tiếp. App luôn có chuỗi dự phòng: model quá tải "
            "→ tự chuyển model kế tiếp.")
    _c4a, _c4b = st.columns([2, 1])
    with _c4a:
        _k_in = st.text_input(
            "Gemini API key:",
            value=_key_default, type="password", key="gemini_key",
            help="Lấy key miễn phí: aistudio.google.com/apikey → đăng nhập "
                 "Google → Create API key → copy. Key lưu tại "
                 "data/gemini_key.txt (gitignored — không lên mạng).")
    with _c4b:
        st.write("")  # căn dòng
        _save4 = st.button("💾 Lưu & kiểm tra key", key="gemini_save",
                           type="primary",
                           disabled=not (_k_in or "").strip())
    if _save4:
        with st.spinner("Đang kiểm tra key…"):
            _ok4, _msg4 = AIA.validate_key(_k_in.strip())
        if _ok4:
            _kf4.write_text(_k_in.strip(), encoding="utf-8")
            st.session_state.pop("gemini_models", None)   # nạp lại model
            st.success("✅ Key hợp lệ — đã lưu. AI sẵn sàng cho chuẩn hoá + chấm.")
        else:
            st.error(f"❌ Key chưa dùng được ({_msg4}) — không lưu.")
    # ---- chọn model (danh sách LẤY TRỰC TIẾP từ API theo key) ----
    _models = st.session_state.get("gemini_models")
    if _models is None and _key_default:
        try:
            _models = AIA.list_models(_key_default)
            st.session_state["gemini_models"] = _models
        except Exception as _e:                              # noqa
            # dự phòng: danh sách tĩnh — vẫn chọn được model khi API lỗi
            _models = list(AIA.GEMINI_MODELS)
            st.session_state["gemini_models"] = _models
            st.caption(f"⚠ Không lấy được danh sách trực tiếp "
                       f"({str(_e)[:60]}) — đang dùng danh sách dự phòng.")
    _cur_model = None
    _mf4 = ROOT / "data" / "gemini_model.txt"
    if _mf4.is_file():
        _cur_model = _mf4.read_text(encoding="utf-8").strip()
    if _models:
        _opts = list(_models)
        if _cur_model and _cur_model not in _opts:
            _opts.insert(0, _cur_model)
        _def_i = _opts.index(_cur_model) if _cur_model in _opts else \
            (_opts.index("gemini-3.8-flash") if "gemini-3.8-flash" in _opts
             else 0)
        _sel_m = st.selectbox(
            "Model AI dùng cho chấm + hỗ trợ (danh sách lấy trực tiếp từ "
            "API):", _opts, index=_def_i, key="gemini_model_sel")
        _c4c, _c4d, _c4e = st.columns([1, 1, 2])
        if _c4c.button("✅ Dùng model này", key="gemini_model_save"):
            _mf4.write_text(_sel_m, encoding="utf-8")
            st.success(f"Đã chọn **{_sel_m}** — dùng cho mọi tác vụ AI.")
        if _c4d.button("🔄 Làm mới danh sách", key="gemini_model_refresh"):
            st.session_state.pop("gemini_models", None)
            st.rerun()
        st.caption(f"Model đang dùng: **{_cur_model or _opts[_def_i]}** — "
                   "mọi tác vụ AI đều có chuỗi dự phòng khi model quá tải.")
    else:
        st.caption("Nhập key hợp lệ ở trên rồi bấm 💾 để thấy danh sách "
                   "model khả dụng.")

# ---------------- TAB 3: GOOGLE DRIVE ----------------
# (đã bỏ tab "Lịch sử CSDL" theo yêu cầu — kết quả vẫn lưu vào CSDL qua save_run_db)
