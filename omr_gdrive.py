# -*- coding: utf-8 -*-
"""
NHẬN ẢNH QUÉT TỪ GOOGLE DRIVE (omr_gdrive.py)

Hỗ trợ link công khai:
- File bất kỳ (ảnh/PDF): https://drive.google.com/file/d/<ID>/view?usp=sharing
- Thư mục: https://drive.google.com/drive/folders/<ID>

Quy trình: parse link -> gdown tải về data/gdrive_in -> PDF tách thành ảnh
(pdfimages) -> ảnh chép vào thư mục trang -> trả về thư mục để pipeline chấm.
Chỉ hỗ trợ link công khai (không cần OAuth); link riêng tư cần chia sẻ
"Bất kỳ ai có đường liên kết".
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GDRIVE_IN = ROOT / "data/gdrive_in"


def parse_gdrive_link(url):
    """Trả ("file"|"folder", id) hoặc None nếu không nhận dạng được."""
    u = url.strip()
    if "/file/d/" in u:
        return "file", u.split("/file/d/")[1].split("/")[0]
    if "/folders/" in u:
        return "folder", u.split("/folders/")[1].split("?")[0].strip("/")
    if "uc?id=" in u or "open?id=" in u:
        tail = u.split("id=")[1]
        return "file", tail.split("&")[0]
    if u.isdigit() and len(u) >= 20:
        return "file", u
    return None


def fetch_gdrive(url, dest=None):
    """Tải file/thư mục Drive về dest; trả (kind, id, thư mục chứa nội dung)."""
    import gdown
    parsed = parse_gdrive_link(url)
    if not parsed:
        raise ValueError(f"Không nhận dạng được link Google Drive: {url}")
    kind, fid = parsed
    dest = Path(dest) if dest else GDRIVE_IN / fid
    dest.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        out = dest / "drive_file"
        path = gdown.download(id=fid, output=str(out), quiet=False, fuzzy=True)
        if not path:
            raise RuntimeError(f"Tải file Drive thất bại (id={fid}) — kiểm tra link công khai.")
        return kind, fid, Path(path).parent
    # folder
    # gdown 6.x đã bỏ kwarg `remaining_ok` (gặp lỗi "unexpected keyword
    # argument 'remaining_ok'" với gdown 6.5.1) — gọi KHÔNG kwarg đó.
    try:
        got = gdown.download_folder(id=fid, output=str(dest), quiet=False,
                                    use_cookies=False)
    except TypeError:
        got = gdown.download_folder(id=fid, output=str(dest), quiet=False,
                                    use_cookies=False, remaining_ok=True)
    if not got:
        raise RuntimeError(f"Tải thư mục Drive thất bại (id={fid}) — kiểm tra link công khai.")
    return kind, fid, dest


def prepare_pages(content_dir, pages_dir):
    """Chuẩn hóa nội dung tải về thành thư mục ảnh trang (PDF -> pdfimages)."""
    content_dir, pages_dir = Path(content_dir), Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    n_img = n_pdf = 0
    for f in sorted(content_dir.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            dst = pages_dir / f.name
            if dst.resolve() == f.resolve():
                continue  # file đã nằm sẵn trong thư mục trang
            shutil.copy2(f, dst)
            n_img += 1
        elif ext == ".pdf":
            base = f.stem.replace(" ", "_")[:24] or "pdf"
            subprocess.run(["pdfimages", "-j", str(f), str(pages_dir / base)], check=True)
            # Fallback: PDF vector không có ảnh nhúng -> render từng trang bằng pdftoppm
            if not list(pages_dir.glob(f"{base}-*.jpg")):
                subprocess.run(["pdftoppm", "-png", "-r", "150", str(f),
                                str(pages_dir / base)], check=True)
            n_pdf += 1
    n_pages = len([p for p in pages_dir.iterdir() if p.is_file()])
    return n_img, n_pdf, n_pages


def fetch_and_prepare(url, fresh=True):
    """Trọn gói: link Drive -> thư mục ảnh trang.

    fresh=True: XÓA cache trang cũ của link trước khi quét — mỗi lần nhập link
    quét lại toàn bộ hình hiện có (gồm thư mục con), không trộn kết quả cũ.
    Trả (thư mục trang, tổng số ảnh)."""
    kind, fid, content = fetch_gdrive(url)
    pages = ROOT / "data/scan_pages_gdrive" / fid
    if fresh and pages.is_dir():
        import shutil
        shutil.rmtree(pages, ignore_errors=True)
    n_img, n_pdf, n_pages = prepare_pages(content, pages)
    print(f"Drive [{kind} id={fid}]: {n_img} ảnh, {n_pdf} PDF -> {n_pages} trang ở {pages}")
    if n_pages == 0:
        raise RuntimeError("Không nhận được ảnh trang nào từ Drive.")
    return pages, n_pages


def scan_links(links_text, progress=None):
    """Quét NHIỀU link (mỗi dòng một link Drive hoặc 'dir:<thư mục>').

    Trả (danh sách file ảnh gộp, báo cáo từng link)."""
    links = [l.strip() for l in links_text.splitlines() if l.strip()]
    all_pages, report = [], []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for i, link in enumerate(links, 1):
        try:
            if link.startswith("dir:"):
                pages_dir = Path(link[4:])
                pages = sorted(p for p in pages_dir.iterdir()
                               if p.is_file() and p.suffix.lower() in exts)
                report.append({"link": link[:60], "trang": len(pages),
                               "thư mục": str(pages_dir), "lỗi": ""})
            else:
                pages_dir, n = fetch_and_prepare(link, fresh=True)
                pages = sorted(p for p in pages_dir.iterdir()
                               if p.is_file() and p.suffix.lower() in exts)
                report.append({"link": link[:60], "trang": len(pages),
                               "thư mục": str(pages_dir), "lỗi": ""})
            all_pages += pages
        except Exception as e:  # noqa
            report.append({"link": link[:60], "trang": 0, "thư mục": "", "lỗi": str(e)[:90]})
        if progress:
            progress.progress(i / max(1, len(links)), text=f"Link {i}/{len(links)}")
    return all_pages, report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Dùng: python omr_gdrive.py <link_file_hoac_thu_muc_GDrive>")
        sys.exit(2)
    fetch_and_prepare(sys.argv[1])
