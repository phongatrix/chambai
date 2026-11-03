# -*- coding: utf-8 -*-
"""
CSDL LƯU TRỮ CÁC LẦN QUÉT/CHẤM (omr_db.py)

SQLite file: data/omr_results.db
- bảng runs : id, thời gian, nguồn (built-in/dir/gdrive), số trang, số khớp, ghi chú
- bảng pages: run_id, số thứ tự, tên file, loại phiếu, mã đề, SBD,
              điểm P1/P2/P3/Tổng của AI, điểm phần mềm cũ (nếu có), đếm chi tiết

CLI:
  python omr_db.py init                     # tạo/kiểm tra CSDL
  python omr_db.py import-json <file.json> <label nguồn>   # nhập kết quả JSON có sẵn
  python omr_db.py list                     # liệt kê các lần chấm
  python omr_db.py show <run_id>            # chi tiết 1 lần chấm
  python omr_db.py export <run_id> [file.xlsx]  # xuất 1 lần chấm ra Excel
  python omr_db.py stats                    # tổng hợp theo mã đề
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data/omr_results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  ts       TEXT NOT NULL,
  source   TEXT NOT NULL,
  n_pages  INTEGER NOT NULL,
  n_match  INTEGER,
  notes    TEXT
);
CREATE TABLE IF NOT EXISTS pages (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  page_no     INTEGER NOT NULL,
  filename    TEXT NOT NULL,
  grp         TEXT,
  made        TEXT,
  sbd         TEXT,
  p1 REAL, p2 REAL, p3 REAL, tong REAL,
  kq_p1 REAL, kq_p2 REAL, kq_p3 REAL, kq_tong REAL,
  counts_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_run ON pages(run_id);
CREATE INDEX IF NOT EXISTS idx_pages_made ON pages(made);
"""


def connect(db=None):
    db = Path(db) if db else DB
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def save_run(source, results, notes=None, db=None):
    """Lưu 1 lần chấm (danh sách results của omr_grade_final) vào CSDL."""
    con = connect(db)
    ts = datetime.now().isoformat(timespec="seconds")
    n_match = sum(1 for r in results if r.get("match"))
    cur = con.execute("INSERT INTO runs(ts, source, n_pages, n_match, notes) VALUES (?,?,?,?,?)",
                      (ts, source, len(results), n_match, notes))
    run_id = cur.lastrowid
    for i, r in enumerate(results, 1):
        k = r.get("kq") or {}
        con.execute(
            "INSERT INTO pages(run_id, page_no, filename, grp, made, sbd, p1, p2, p3, tong,"
            " kq_p1, kq_p2, kq_p3, kq_tong, counts_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, i, r["file"], r.get("grp"), r.get("md"), r.get("sbd"),
             r["my"]["P1"], r["my"]["P2"], r["my"]["P3"], r["my"]["tong"],
             k.get("p1"), k.get("p2"), k.get("p3"), k.get("tong"),
             json.dumps(r.get("counts", {}), ensure_ascii=False)))
    con.commit()
    con.close()
    return run_id


def import_json(json_path, source_label, db=None):
    """Nhập 1 file grading_*.json có sẵn thành 1 lần chấm."""
    results = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return save_run(source_label, results,
                    notes=f"import từ {Path(json_path).name}", db=db)


def list_runs(db=None):
    con = connect(db)
    rows = con.execute("SELECT id, ts, source, n_pages, n_match, notes FROM runs ORDER BY id").fetchall()
    con.close()
    return rows


def show_run(run_id, db=None):
    con = connect(db)
    run = con.execute("SELECT id, ts, source, n_pages, n_match, notes FROM runs WHERE id=?",
                      (run_id,)).fetchone()
    pages = con.execute(
        "SELECT page_no, filename, made, sbd, p1, p2, p3, tong, kq_tong"
        " FROM pages WHERE run_id=? ORDER BY page_no", (run_id,)).fetchall()
    con.close()
    return run, pages


def export_run(run_id, xlsx, db=None):
    import openpyxl
    run, pages = show_run(run_id, db)
    if not run:
        raise ValueError(f"Không có run_id={run_id}")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"Run {run_id}"
    ws.append(["Run", "Thời gian", "Nguồn", "Số trang", "Số khớp"])
    ws.append(list(run))
    ws.append([])
    ws.append(["STT", "File", "Mã đề", "SBD", "P1", "P2", "P3", "Tổng AI",
               "Tổng KQ", "Chênh"])
    for (no, fn, made, sbd, p1, p2, p3, tong, kq_tong) in pages:
        ws.append([no, fn, made, sbd, p1, p2, p3, tong, kq_tong,
                   None if kq_tong is None else round((tong or 0) - kq_tong, 2)])
    for col, wdt in zip("ABCDEFGHIJ", (5, 14, 8, 12, 7, 7, 7, 9, 9, 8)):
        ws.column_dimensions[col].width = wdt
    Path(xlsx).parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx)
    return len(pages)


def stats_by_made(db=None):
    con = connect(db)
    rows = con.execute(
        "SELECT made, COUNT(*), AVG(p1), AVG(p2), AVG(tong), MIN(tong), MAX(tong)"
        " FROM pages WHERE made IS NOT NULL GROUP BY made ORDER BY made").fetchall()
    con.close()
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(2)
    cmd = a[0]
    if cmd == "init":
        connect()
        print(f"CSDL sẵn sàng: {DB}")
    elif cmd == "import-json":
        rid = import_json(a[1], a[2])
        print(f"Đã nhập {a[1]} -> run {rid}")
    elif cmd == "list":
        print(f"{'id':>3} {'thời gian':20} {'nguồn':22} {'số trang':>8} {'khớp':>5}  ghi chú")
        for r in list_runs():
            print(f"{r[0]:>3} {r[1]:20} {r[2]:22} {r[3]:>8} {str(r[4] or '-'):>5}  {r[5] or ''}")
    elif cmd == "show":
        run, pages = show_run(int(a[1]))
        print(f"Run {run[0]} | {run[1]} | nguồn: {run[2]} | {run[3]} trang | khớp: {run[4]}")
        print(f"{'STT':>3} {'file':12} {'mã đề':6} {'SBD':12} {'P1':>5} {'P2':>5} {'P3':>4} {'Tổng':>6} {'KQ':>6}")
        for p in pages:
            print(f"{p[0]:>3} {p[1]:12} {str(p[2] or '-'):6} {str(p[3] or '-'):12} "
                  f"{str(p[4] or '-'):>5} {str(p[5] or '-'):>5} {str(p[6] or '-'):>4} "
                  f"{str(p[7] or '-'):>6} {str(p[8] or '-'):>6}")
    elif cmd == "export":
        n = export_run(int(a[1]), a[2] if len(a) > 2 else f"output/run_{a[1]}.xlsx")
        print(f"Đã xuất {n} trang")
    elif cmd == "stats":
        print(f"{'mã đề':6} {'n':>3} {'TB P1':>6} {'TB P2':>6} {'TB tổng':>8} {'min':>5} {'max':>5}")
        for made, n, ap1, ap2, at, mn, mx in stats_by_made():
            print(f"{made:6} {n:>3} {ap1:>6.2f} {ap2:>6.2f} {at:>8.2f} {mn:>5.2f} {mx:>5.2f}")
    else:
        print(f"Lệnh không rõ: {cmd}")
        sys.exit(2)
