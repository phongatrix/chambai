#!/bin/bash
# Cài môi trường cho gói chấm trắc nghiệm OMR (chạy 1 lần).
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
UV="$DIR/.tools/uv"
VENV="$DIR/.venv"

if [[ ! -x "$UV" ]]; then
    echo "Tải uv..."
    curl -fsSL https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
    UV_NO_MODIFY_PATH=1 UV_UNMANAGED_INSTALL="$DIR/.tools" sh /tmp/uv-install.sh
fi
if ! command -v tesseract >/dev/null 2>&1; then
    echo "Cảnh báo: thiếu tesseract (chỉ dùng cho OCR phụ). Cài: sudo apt install tesseract-ocr"
fi
echo "Cài Python 3.12 + venv..."
"$UV" python install 3.12
"$UV" venv --python 3.12 "$VENV"
"$UV" pip install --python "$VENV/bin/python" \
    opencv-python-headless numpy pandas openpyxl pillow onnxruntime tf2onnx \
    tensorflow scikit-learn reportlab pdf2image streamlit pytesseract \
    google-generativeai python-dotenv gdown
echo "=== Cài đặt xong. Chạy: bash run_cham_bai.sh ==="
