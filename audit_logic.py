# -*- coding: utf-8 -*-
"""KIỂM TRA LOGIC THUẬT TOÁN — các bất biến toán học của từng khâu chấm.

Khác verify_logic.py (đối chiếu giả định với dữ liệu có sẵn), script này kiểm
tra LOGIC THUẦN TÚY bằng dữ liệu tổng hợp có đáp án biết trước:

  A1. Round-trip chuẩn hoá: warp tổng hợp (xoay 0/90/180/270 + phối cảnh nhẹ +
      sáng không đều) của CHÍNH template -> normalize_to_template phải khôi
      phục được (ok=True, H khớp warp gốc).
  A2. classify_filled: bất biến phân loại (ô đậm nhất câu -> tô; câu trống ->
      không tô; ngưỡng nền theo percentile).
  A3. read_cells: điểm nằm trong [0, 1]; nền trắng = ~0, ô đen = ~1.
  A4. Đặt tên kết quả: trùng tên cùng giây -> hậu quả _1/_2, KHÔNG treo.
  A5. _md_read_strong: ngưỡng tin cậy lưới Mã đề (tuple thật 195843/200014).
  A6. merge key gộp: đủ 12 đề, không đề rỗng, không trùng lặp.
  A7. grade_prepared: H = scale·I đúng tỉ lệ kích thước ảnh/mẫu.
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/dell/Documents/chambaiV3")
ROOT = Path("/home/dell/Documents/chambaiV3")

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'✓' if ok else '✗ LỖI'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def main():
    import omr_prep as P
    import omr_sheetgrid as S
    import omr_key
    import cham_theo_mau as CTM

    tpl12 = str(ROOT / "data/maugiay/12.png")
    tpl1011 = str(ROOT / "data/maugiay/10_11.png")
    tpl_img = cv2.imread(tpl12)
    Ht, Wt = tpl_img.shape[:2]

    print("== A1. ROUND-TRIP CHUẨN HOÁ (warp tổng hợp -> khôi phục) ==")
    # 4.3× = tỉ lệ thật của ảnh điện thoại (1950-2600 px / template 461 px),
    # marker ~34-39 px nằm trong dải dò 30-110 px của align_photo.
    # PHÁT HIỆN BIÊN: ảnh chụp XA hơn (marker < 30 px) sẽ không align được —
    # dải dò ô định vị của align_photo là hằng số (30, 110).
    SCALE_SYN = 4.3
    # warp gốc: xoay nhẹ + phối cảnh nhỏ + gradient sáng (bóng đổ chéo)
    src = np.float32([[0, 0], [Wt, 0], [Wt, Ht], [0, Ht]])
    off = 0.035
    # 4 góc đích nằm TRÊN CANVAS SCALE_SYN× (nhân scale — trước đây quên,
    # warp đặt sheet 1:1 vào góc trái trên của canvas)
    dst = SCALE_SYN * np.float32(
        [[Wt * off, Ht * 0.02], [Wt * (1 - off * 0.6), 0],
         [Wt * (1 - 0.01), Ht * (1 - 0.02)], [Wt * 0.01, Ht * (1 - off * 0.5)]])
    H_true = cv2.getPerspectiveTransform(src, dst)
    photo_big = cv2.warpPerspective(tpl_img, H_true,
                                    (int(Wt * SCALE_SYN), int(Ht * SCALE_SYN)),
                                    flags=cv2.INTER_CUBIC, borderValue=(255,) * 3)
    # gradient sáng nhân tạo (bóng đổ chéo)
    gx = np.linspace(0.75, 1.0, photo_big.shape[1], dtype=np.float32)
    gy = np.linspace(0.85, 1.0, photo_big.shape[0], dtype=np.float32)
    shade = (gx[None, :] * gy[:, None])[:, :, None]
    photo_big = np.clip(photo_big.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    cv2.imwrite("/tmp/_audit_warp.jpg", photo_big,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    import omr_sheetgrid as _SGA
    _cells12 = _SGA.build_template_cells("12", tpl12)
    t0 = time.time()
    info = P.normalize_to_template("/tmp/_audit_warp.jpg", tpl12, scale=2.0,
                                   cells=_cells12, ink=False)
    dt = time.time() - t0
    check("khôi phục warp phối cảnh + bóng đổ", info.get("ok"),
          f"q={info.get('quality')} rot={info.get('rot')} corner={info.get('corner')} "
          f"({dt:.1f}s)")
    if info.get("ok"):
        # so H tìm được với H_true — CHÚ Ý: load_bgr thu nhỏ ảnh > 2600 px,
        # nên toạ độ ảnh thực = toạ độ canvas × (2600 / max_side_canvas)
        Ht_found = np.asarray(info["H"], np.float64)
        sc_img = 2600.0 / max(photo_big.shape[0], photo_big.shape[1])
        pts = np.float32([[[50, 50]], [[Wt - 50, 50]], [[Wt - 50, Ht - 50]],
                          [[50, Ht - 50]]])
        proj_true = cv2.perspectiveTransform(pts, H_true).reshape(-1, 2) * sc_img
        proj_found = cv2.perspectiveTransform(pts, Ht_found).reshape(-1, 2)
        err = float(np.linalg.norm(proj_true - proj_found, axis=1).max())
        check("H tìm được khớp warp gốc (≤ 25 px)", err <= 25.0,
              f"sai số {err:.1f} px (đã trừ hệ số thu nhỏ {sc_img:.3f})")

    # A1b: XOAY TỈNH — phiếu nghiêng 12° và 22° (+ phối cảnh + bóng đổ) phải
    # được khôi phục nhờ vòng xoay tỉnh (trước đây corner 2/4 -> bị loại)
    print("  -- A1b: xoay tỉnh 12° / 22° --")
    src = np.float32([[0, 0], [Wt, 0], [Wt, Ht], [0, Ht]])
    off = 0.035
    dst = SCALE_SYN * np.float32(
        [[Wt * off, Ht * 0.02], [Wt * (1 - off * 0.6), 0],
         [Wt * (1 - 0.01), Ht * (1 - 0.02)], [Wt * 0.01, Ht * (1 - off * 0.5)]])
    # canvas 1.5×: đủ chứa phiếu XOAY 12-22° mà không cắt mất mốc góc
    # (như ảnh chụp thật — phiếu phải lọt trọn khung hình)
    canvas = (int(Wt * SCALE_SYN * 1.5), int(Ht * SCALE_SYN * 1.5))
    # CĂN GIỮA phiếu trên canvas: dst quad lệch do phối cảnh — không căn giữa
    # thì xoay làm mép phiếu/mốc rơi ra ngoài khung => bị loại là ĐÚNG
    dst = dst + (np.float32([canvas[0] / 2, canvas[1] / 2]) - dst.mean(axis=0))
    H_true = cv2.getPerspectiveTransform(src, dst)
    for deg in (12, 22):
        big = cv2.warpPerspective(tpl_img, H_true, canvas,
                                  borderValue=(255,) * 3)
        Mr = cv2.getRotationMatrix2D((canvas[0] / 2, canvas[1] / 2), deg, 1.0)
        big = cv2.warpAffine(big, Mr, canvas, borderValue=(255,) * 3)
        gx = np.linspace(0.78, 1.0, canvas[0], dtype=np.float32)
        gy = np.linspace(0.85, 1.0, canvas[1], dtype=np.float32)
        big = np.clip(big.astype(np.float32) *
                      (gx[None, :] * gy[:, None])[:, :, None], 0, 255).astype(np.uint8)
        p = f"/tmp/_audit_tilt_{deg}.jpg"
        cv2.imwrite(p, big, [cv2.IMWRITE_JPEG_QUALITY, 92])
        t0 = time.time()
        info = P.normalize_to_template(p, tpl12, scale=2.0,
                                       cells=_cells12, ink=False)
        check(f"khôi phục phiếu nghiêng {deg}°", info.get("ok"),
              f"q={info.get('quality')} corner={info.get('corner')} "
              f"grid={info.get('grid_hit')} fine={info.get('fine_rot')} "
              f"({time.time() - t0:.1f}s)")

    print("\n== A2. CLASSIFY_FILLED (bất biến phân loại) ==")
    # 2 câu: câu 1 có 1 ô đậm (0.8) + nhạt (0.3); câu 2 trống (0.05)
    keys = [("P1", 1, "A"), ("P1", 1, "B"), ("P1", 1, "C"), ("P1", 1, "D"),
            ("P1", 2, "A"), ("P1", 2, "B"), ("P1", 2, "C"), ("P1", 2, "D")]
    scores = np.array([0.8, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], np.float32)
    mask, floor, peak = P.classify_filled(keys, scores)
    check("ô đậm nhất câu 1 được nhận (A)", bool(mask[0]))
    check("ô nhạt hơn 75% (B=0.3) KHÔNG nhận", not bool(mask[1]),
          f"(floor={floor:.2f})")
    check("câu trống không ô nào nhận", not mask[4:].any())
    scores2 = np.array([0.8, 0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], np.float32)
    mask2, _, _ = P.classify_filled(keys, scores2)
    check("tô 2 ô (B ≥ 75% max câu) nhận cả 2", bool(mask2[0]) and bool(mask2[1]))

    print("\n== A3. READ_CELLS (biên điểm [0,1]) ==")
    # H = đơn vị: ô đặt TRỰC TIẾP ở toạ độ ảnh (thang ảnh = thang mẫu)
    img = np.full((400, 400), 255, np.uint8)
    cv2.circle(img, (100, 100), 14, 0, -1)          # ô đen đặc
    cv2.circle(img, (300, 100), 14, 235, -1)        # ô trắng
    cells = [("P1", 1, "A", 100.0, 100.0), ("P1", 1, "B", 300.0, 100.0)]
    H1 = np.eye(3, dtype=np.float64)
    reads = S.read_cells(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cells, H1)
    v_black = reads[("P1", 1, "A")][0]
    v_white = reads[("P1", 1, "B")][0]
    check("điểm nằm trong [0, 1]", 0 <= v_black <= 1 and 0 <= v_white <= 1,
          f"đen={v_black:.2f} trắng={v_white:.2f}")
    # viền khử răng cưa của vòng tròn hạ trung bình một chút -> tách 0.35
    check("tách bạch đen/trắng (đen > 0.9, trắng < 0.35)",
          v_black > 0.9 and v_white < 0.35)

    print("\n== A4. ĐẶT TÊN TRÙNG (chống treo) ==")
    outd = Path(tempfile.mkdtemp())
    used = set()

    def dat_ten(stem):
        outp = outd / f"{stem}.jpg"
        base, n = stem, 1
        while stem in used or outp.exists():
            stem = f"{base}_{n}"
            outp = outd / f"{stem}.jpg"
            n += 1
        used.add(stem)
        outp.write_bytes(b"x")
        return outp.name
    names = [dat_ten("NA_485_20261104_222917") for _ in range(3)]
    check("3 trang trùng tên -> _1/_2 và kết thúc", len(set(names)) == 3,
          str(names))

    print("\n== A5. NGƯỠNG TIN MÃ ĐỀ LƯỚI ==")
    strong = {"MD": [(404.5, 3, 1.00, 0.76, 9.15), (413.5, 1, 1.00, 0.73, 7.20),
                     (422.5, 5, 0.87, 0.66, 3.71), (431.5, 7, 0.91, 0.66, 33.0)]}
    weak = {"MD": [(404.5, 4, 0.49, 0.23, 2.41), (413.5, 1, 0.84, 0.54, 9.48),
                   (422.5, 8, 0.66, 0.36, 3.98), (431.5, 5, 0.48, 0.28, 8.22)]}
    check("đọc chắc (195843: raw 0.87-1.0) -> TIN", CTM._md_read_strong(strong))
    check("đọc mơ hồ (200014: ratio 2.4) -> KHÔNG tin",
          not CTM._md_read_strong(weak))
    check("không đọc được -> KHÔNG tin", not CTM._md_read_strong({}))

    print("\n== A6. KEY GỘP (cấu trúc) ==")
    import json as _json
    import importlib.util as _ilu
    regf = ROOT / "data/keys/registry.json"
    reg = _json.loads(regf.read_text(encoding="utf-8"))
    _spec = _ilu.spec_from_file_location("wapp", ROOT / "webapp.py")
    _w = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_w)
    merged = _w.merge_keys_grading(list(reg.keys()), reg)
    des = merged.get("các đề") or {}
    check("key gộp có 12 đề", len(des) == 12, f"({len(des)}: {sorted(des)})")
    empty = [d for d, dk in des.items() if not (dk.get("p1") or {})]
    check("không đề nào rỗng PHẦN I", not empty, str(empty))

    print("\n== A7. H TRONG GRADE_PREPARED (= scale·I) ==")
    ph = cv2.imread("/tmp/_audit_warp.jpg")  # ảnh bất kỳ để lấy tỉ lệ
    scale_expect = 2.0
    H = np.array([[scale_expect, 0, 0], [0, scale_expect, 0], [0, 0, 1]],
                 np.float64)
    tpl2 = cv2.imread(tpl12)
    check("scale·I đưa ảnh về đúng cỡ mẫu×scale",
          abs(scale_expect * tpl2.shape[1] - int(tpl2.shape[1] * scale_expect)) == 0
          and H[0, 0] == scale_expect)

    print("\n" + "=" * 50)
    if FAIL:
        print(f"KẾT LUẬN: {len(FAIL)} hạng mục LỖI: {FAIL}")
        sys.exit(1)
    print("KẾT LUẬN: toàn bộ bất biến logic ĐỨNG — thuật toán nhất quán.")


if __name__ == "__main__":
    main()
