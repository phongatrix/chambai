#!/bin/bash
# ============================================================
# SYNC — đồng bộ code từ bản GỐC sang các bản copy
# ============================================================
# Bản gốc (nguồn) TỪ 2026-11-04: /home/dell/Documents/chambaiV3
#   (chỉ định dùng bản này làm chính; server chạy từ đây, cổng 8502)
# Bản copy được cập nhật (chỉ để dự phòng, KHÔNG dùng làm chính):
#   1. /home/dell/Desktop/omr_ai_chambai
#   2. /home/dell/Desktop/omr_ai_project
#   3. /home/dell/Desktop/omr_ai_project (2)
#
# Đồng bộ: toàn bộ *.py, README, báo cáo, run_*.sh, setup.sh,
#          requirements/packages, .streamlit, data/keys (kèm registry.json),
#          data/maugiay, data/dapan, keys_builtin.json, omr_results.db, tulieu
# KHÔNG đồng bộ: .venv, output/, data/web_upload, data/scan_pages*,
#                data/gdrive_in, __pycache__ (dữ liệu chạy cục bộ)
# ============================================================
set -Eeuo pipefail

SRC="/home/dell/Documents/chambaiV3"
DESTS=(
  "/home/dell/Desktop/omr_ai_chambai"
  "/home/dell/Desktop/omr_ai_project"
  "/home/dell/Desktop/omr_ai_project (2)"
)

cd "$SRC"

# Danh sách file đồng bộ (mọi thứ trừ phần chạy cục bộ)
FILES=(
  *.py
  README.md
  BAO_CAO_CHAM_BAI_THUC_TE.md
  PROMPT_VA_THUAT_TOAN.md
  requirements.txt packages.txt .gitignore
  run_web.sh run_cham_bai.sh setup.sh
)

for DEST in "${DESTS[@]}"; do
  if [ ! -d "$DEST" ]; then
    echo "[!] Bỏ qua (không tồn tại): $DEST"
    continue
  fi
  mkdir -p "$DEST/data/keys" "$DEST/data/tulieu" "$DEST/.streamlit"
  cp -f .streamlit/config.toml "$DEST/.streamlit/" 2>/dev/null || true
  # code + tài liệu
  for f in "${FILES[@]}"; do
    [ -e "$f" ] && cp -f "$f" "$DEST/"
  done
  # dữ liệu dùng chung: mẫu giấy + đáp án + key + CSDL
  mkdir -p "$DEST/data/maugiay" "$DEST/data/dapan"
  cp -f data/maugiay/*.png "$DEST/data/maugiay/" 2>/dev/null || true
  # lưới ô của mẫu moi (build_moi_template.py) — không có thì mẫu moi KHÔNG chạy
  cp -f data/maugiay/*.json "$DEST/data/maugiay/" 2>/dev/null || true
  cp -f data/dapan/*.xlsx "$DEST/data/dapan/" 2>/dev/null || true
  cp -f data/keys_builtin.json "$DEST/data/" 2>/dev/null || true
  cp -f data/omr_results.db    "$DEST/data/" 2>/dev/null || true
  # models phải nằm TRONG thư mục dự án (tự chủ, không tải lúc chạy):
  # digit_cnn.keras (SBD viết tay) + docshadow_sd7k.onnx + uvdoc/*.onnx
  mkdir -p "$DEST/data/models/uvdoc"
  cp -f data/models/digit_cnn.keras data/models/digit_cnn_meta.json \
        data/models/docshadow_sd7k.onnx "$DEST/data/models/" 2>/dev/null || true
  cp -f data/models/uvdoc/*.onnx "$DEST/data/models/uvdoc/" 2>/dev/null || true
  cp -f data/keys/*.xlsx       "$DEST/data/keys/" 2>/dev/null || true
  # registry.json = danh sách 'Các đáp án đã lưu' — phải đồng bộ cùng key,
  # nếu thiếu thì bản đích mất danh sách key sau khi khởi động lại
  cp -f data/keys/registry.json "$DEST/data/keys/" 2>/dev/null || true
  if [ -d data/tulieu ]; then
    cp -rf data/tulieu/. "$DEST/data/tulieu/" 2>/dev/null || true
  fi
  # dọn cache cũ của bản đích
  rm -rf "$DEST/__pycache__" 2>/dev/null || true
  echo "[✓] Đã đồng bộ -> $DEST"
done

echo "Hoàn tất: bản gốc = $SRC"
