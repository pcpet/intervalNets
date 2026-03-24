from __future__ import annotations

from pathlib import Path
import textwrap

PAGE_W = 595
PAGE_H = 842
MARGIN = 48
LINE_H = 13
FONT_SIZE = 10
MAX_CHARS = 98


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_line(line: str, width: int = MAX_CHARS) -> list[str]:
    if not line.strip():
        return [""]
    if line.startswith("```"):
        return [line]
    if line.startswith("    ") or line.startswith("\t"):
        return [line[:width]] + ([line[width:]] if len(line) > width else [])
    return textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def markdown_to_lines(md: str) -> list[str]:
    out: list[str] = []
    in_code = False
    for raw in md.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("```"):
            in_code = not in_code
            out.append("[diagram block]" if in_code else "")
            continue
        if in_code:
            out.append("    " + line)
            continue
        if line.startswith("# "):
            out.append(line[2:].upper())
            out.append("")
            continue
        if line.startswith("## "):
            out.append(line[3:])
            out.append("")
            continue
        if line.startswith("### "):
            out.append(line[4:])
            continue
        if line.startswith("#### "):
            out.append(line[5:])
            continue
        out.append(line)
    return out


def paginate(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    max_lines = (PAGE_H - 2 * MARGIN) // LINE_H
    for line in lines:
        wrapped = wrap_line(line)
        for part in wrapped:
            if len(current) >= max_lines:
                pages.append(current)
                current = []
            current.append(part)
    if current:
        pages.append(current)
    return pages


def build_pdf(pages: list[list[str]], output: Path) -> None:
    objects: list[bytes] = []

    def add_obj(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_obj = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_objs: list[int] = []
    content_objs: list[int] = []

    for page in pages:
        commands = ["BT", f"/F1 {FONT_SIZE} Tf"]
        y = PAGE_H - MARGIN
        for line in page:
            escaped = pdf_escape(line)
            commands.append(f"1 0 0 1 {MARGIN} {y} Tm ({escaped}) Tj")
            y -= LINE_H
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        content = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        content_id = add_obj(content)
        content_objs.append(content_id)

    pages_kids_placeholder = "PAGES_PLACEHOLDER"
    for content_id in content_objs:
        page_data = (
            f"<< /Type /Page /Parent {pages_kids_placeholder} 0 R "
            f"/MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode()
        page_objs.append(add_obj(page_data))

    kids = " ".join(f"{pid} 0 R" for pid in page_objs)
    pages_obj = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_objs)} >>".encode())

    for idx, pid in enumerate(page_objs):
        objects[pid - 1] = objects[pid - 1].replace(pages_kids_placeholder.encode(), str(pages_obj).encode())

    catalog_obj = add_obj(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode())

    xref: list[int] = [0]
    pdf = bytearray(b"%PDF-1.4\n")
    for i, obj in enumerate(objects, start=1):
        xref.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for off in xref[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects)+1} /Root {catalog_obj} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode()
    )

    output.write_bytes(pdf)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    md_path = repo_root / "docs" / "whitepaper" / "whitepaper.md"
    out_pdf = repo_root / "whitepaper.pdf"

    md_text = md_path.read_text(encoding="utf-8")
    lines = markdown_to_lines(md_text)
    pages = paginate(lines)
    build_pdf(pages, out_pdf)
    print(f"Wrote {out_pdf} with {len(pages)} pages.")
