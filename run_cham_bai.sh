#!/bin/bash
# Chạy chấm toàn bộ (mặc định: bộ 49 trang tích hợp; đối chiếu KQ phần mềm cũ).
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PY="$DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "Chưa cài môi trường — chạy: bash setup.sh" >&2
    exit 1
fi

echo "=== 1. CHẤM ĐIỂM (nguồn: tích hợp / hoặc tham số \$1 = 'gdrive:<link>' / 'dir:<thư mục>') ==="
if [[ $# -ge 1 ]]; then
    "$PY" -I omr_grade_final.py --source "$1"
else
    "$PY" -I omr_grade_final.py
fi

echo
echo "=== 2. KIỂM ĐỊNH LOGIC ==="
"$PY" -I verify_logic.py || true

echo
echo "=== 3. LỊCH SỬ CÁC LẦN CHẤM (CSDL) ==="
"$PY" -I omr_db.py list

echo
echo "=== KẾT QUẢ ==="
echo "Excel : $DIR/output/"
echo "CSDL  : $DIR/data/omr_results.db  (xem: .venv/bin/python -I omr_db.py show <run_id>)"
