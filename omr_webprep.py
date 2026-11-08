# -*- coding: utf-8 -*-
"""Worker chuẩn hoá 1 ảnh cho webapp — TÁCH RIÊNG khỏi webapp.py vì tiến trình
con (multiprocessing spawn) sẽ import lại module này: phải KHÔNG chứa code
Streamlit ở cấp module.

Chạy song song N ảnh trên N tiến trình giúp bước ⓪ nhanh ~3-4 lần (44 s/12 ảnh
-> ~12 s) mà thuật toán chuẩn hoá/chấm GIỐNG HỆT chạy tuần tự (cùng
omr_prep.normalize_to_template, cùng cells, cùng tham số).
"""
import cv2
from pathlib import Path

import omr_prep as PREP


def prep_one(task):
    """Chuẩn hoá + tô đen 1 ảnh. `task` = dict(path, tpl, paper, cells, out,
    ink, ink_only). Trả dict: ok / ok_path / ink_path / row / extras.

    Đầu ra TƯƠNG THÍCH 1:1 với vòng lặp tuần tự cũ của webapp.normalize_batch:
    ghi <stem>_chuan.jpg (chấm) + <stem>_todam.jpg (kiểm tra bằng mắt).

    LƯU Ý: KHÔNG gọi cv2.setNumThreads trong worker — tiến trình con được tạo
    bằng FORK, thread-pool nội bộ của OpenCV bên cha bị copy ở trạng thái treo
    (khoá không chủ) => setNumThreads trong con = deadlock. Giới hạn thread
    phải làm ở TIẾN TRÌNH CHA trước khi fork (xem webapp.normalize_batch)."""
    ip = Path(task["path"])
    out_dir = Path(task["out"])
    try:
        info = PREP.normalize_to_template(
            ip, task["tpl"], scale=PREP.norm_scale_for(task.get("paper")),
            cells=task["cells"], ink=task["ink"], ink_only=task["ink_only"])
    except Exception as e:                                   # noqa
        info = {"ok": False, "quality": 0.0, "rot": 0,
                "reason": f"lỗi xử lý: {e}"}
    # ---- TỰ NHẬN MẪU THEO TỪNG ẢNH (2026-11-06): lô có thể LẪN hai loại
    # phiếu (10_11 + Mẫu mới 2025) — ảnh fail mẫu chính thì thử TỪNG mẫu dự
    # phòng; ghi paper_used vào extras để bước chấm dùng ĐÚNG mẫu đó.
    # ink_only (chỉ tô đậm) không cần mẫu -> bỏ qua fallback. ----
    # MARGIN CHỐNG NHẬN NHẦM MẪU (2026-11-06, bug thật lô 11a8): phiếu 10_11
    # layout mới warp vào mẫu moi đạt ring 0.44-0.50 — VỪA ĐỦ lách qua cổng
    # 0.50 của moi và chấm SAI "tự tin". Mẫu dự phòng (alt) phải vượt cổng
    # THÊM +0.08: phiếu ĐÚNG mẫu thật (Thu/moi 0.85-0.89) vẫn thoải mái,
    # phiếu SAI mẫu (0.44-0.50) bị loại về danh sách thất bại với lý do thật.
    paper_used = task.get("paper")
    alts = task.get("alt") or []
    if (not info.get("ok")) and alts and not task.get("ink_only"):
        for _p2, _tpl2, _cells2 in alts:
            if _p2 == paper_used:
                continue
            try:
                info2 = PREP.normalize_to_template(
                    ip, _tpl2, scale=PREP.norm_scale_for(_p2),
                    cells=_cells2, ink=task["ink"], ink_only=False)
            except Exception:                                # noqa
                continue
            if info2.get("ok"):
                _min2 = PREP.RING_MIN_PART.get(_p2, {})
                _need = (_min2.get("ALL", PREP.RING_MIN_DEFAULT) + 0.08
                         if "ALL" in _min2 else None)
                if _need is not None and \
                        float(info2.get("ring_rate") or 0.0) < _need:
                    continue          # ring suýt soát — nhiều khả năng SAI mẫu
                info = info2
                paper_used = _p2
                break
    if info.get("ok"):
        dst = out_dir / f"{ip.stem}_chuan.jpg"
        cv2.imwrite(str(dst), info["warped"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        ink_path = None
        if info.get("inked") is not None:
            td = out_dir / f"{ip.stem}_todam.jpg"
            cv2.imwrite(str(td), info["inked"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            ink_path = str(td)
        st_ = info.get("ink_stats") or {}
        _note = ""
        if paper_used != task.get("paper") and not task.get("ink_only"):
            _note = f"↔ tự nhận mẫu khác: {paper_used}"
        return {"ok": True, "ok_path": str(dst), "ink_path": ink_path,
                "row": {"file": ip.name,
                        "chuẩn hoá": "chỉ tô đậm" if task["ink_only"] else "✓",
                        "xoay": f"{info['rot']}°",
                        "khớp mẫu": round(info["quality"], 2),
                        "ô đã tô": st_.get("n_filled", "-"),
                        "ghi chú": _note},
                "extras": {"ids": info.get("ids_result"),
                           "quality": info.get("quality", 1.0),
                           "rot": info.get("rot", 0),
                           "paper_used": paper_used}}
    return {"ok": False, "ok_path": None, "ink_path": None,
            "row": {"file": ip.name, "chuẩn hoá": "✗", "xoay": "-",
                    "khớp mẫu": round(info.get("quality", 0.0), 2),
                    "ô đã tô": "-", "ghi chú": info.get("reason", "")},
            "extras": {"ids": None, "quality": 0.0, "rot": 0,
                       "paper_used": task.get("paper")}}
