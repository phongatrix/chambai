#!/bin/bash
# Khởi động web chấm trắc nghiệm (mặc định cổng 8502).
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
if [[ ! -x "$DIR/.venv/bin/streamlit" ]]; then
    echo "Chưa cài môi trường — chạy: bash setup.sh" >&2
    exit 1
fi
PORT="${1:-8502}"
echo "Web: http://127.0.0.1:${PORT}  (Ctrl+C để dừng)"
exec "$DIR/.venv/bin/streamlit" run "$DIR/webapp.py" \
    --server.address 0.0.0.0 --server.port "$PORT" --browser.gatherUsageStats false
