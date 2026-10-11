"""Structured extraction for private uploaded documents (PDF, TXT, Markdown, CSV, DOCX, PPTX)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def validate_pdf(filename: str, content: bytes) -> None:
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
        raise ValueError("Only PDF, TXT, Markdown (.md), CSV, DOCX and PPTX uploads are supported.")


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_pdf_pages(path: Path) -> list[dict]:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path), page_chunks=True)


# Task 57-B / phase 4: non-PDF formats, all turned into the same page-dict
# shape pymupdf4llm returns so they reuse structured_chunks() unchanged.
# Formats without pages are one "page 1"; PPTX uses one page per slide.
# DOCX/PPTX are read with zipfile + ElementTree (no new dependencies).
_MAX_UNZIPPED_BYTES = 50 * 1024 * 1024  # zip-bomb guard for DOCX/PPTX
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _page(text: str, number: int = 1) -> dict:
    return {"metadata": {"page_number": number}, "text": text}


def _decode_text(content: bytes) -> str:
    if b"\x00" in content:
        raise ValueError("The file does not look like a text document.")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text uploads must be UTF-8 encoded.") from exc


def text_pages(content: bytes) -> list[dict]:
    return [_page(_decode_text(content))]


def csv_pages(content: bytes) -> list[dict]:
    """One line per row as "Header: value; ..." so a retrieved chunk is
    self-describing even when it doesn't contain the header row."""
    import csv
    import io
    rows = [r for r in csv.reader(io.StringIO(_decode_text(content))) if any(c.strip() for c in r)]
    if not rows:
        return [_page("")]
    header, body = rows[0], rows[1:] or [rows[0]]
    lines = ["; ".join(f"{h.strip() or f'col{i + 1}'}: {v.strip()}" for i, (h, v) in enumerate(zip(header, r)) if v.strip())
             for r in body]
    return [_page("\n".join(lines))]


def _open_office_zip(content: bytes):
    import io
    import zipfile
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("The file is not a valid Office document.") from exc
    if sum(i.file_size for i in archive.infolist()) > _MAX_UNZIPPED_BYTES:
        raise ValueError("The document is too large once decompressed.")
    return archive


def _runs_text(element, ns: str) -> str:
    """Text of one paragraph; soft line breaks and tabs become spaces
    (otherwise "Market<br/>Industry" reads as "MarketIndustry")."""
    parts = []
    for node in element.iter():
        if node.tag == f"{ns}t":
            parts.append(node.text or "")
        elif node.tag in (f"{ns}br", f"{ns}tab"):
            parts.append(" ")
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def _xml(archive, name: str):
    import xml.etree.ElementTree as ET
    try:
        return ET.fromstring(archive.read(name))
    except (KeyError, ET.ParseError) as exc:
        raise ValueError("The file is not a valid Office document.") from exc


def docx_pages(content: bytes) -> list[dict]:
    """Paragraphs in order; Heading N styles become markdown headings (so
    heading paths/citations work like PDFs); table rows become "a | b"."""
    body = _xml(_open_office_zip(content), "word/document.xml").find(f"{_W}body")
    lines = []
    for block in body if body is not None else []:
        if block.tag == f"{_W}p":
            text = _runs_text(block, _W)
            style = block.find(f"{_W}pPr/{_W}pStyle")
            level = re.match(r"heading\s*(\d)", (style.get(f"{_W}val", "") if style is not None else ""), re.I)
            if text:
                lines.append(f"{'#' * min(6, int(level.group(1)))} {text}" if level else text)
        elif block.tag == f"{_W}tbl":
            for row in block.iter(f"{_W}tr"):
                cells = [_runs_text(cell, _W) for cell in row.iter(f"{_W}tc")]
                if any(cells):
                    lines.append(" | ".join(cells))
    return [_page("\n".join(lines))]


def pptx_pages(content: bytes) -> list[dict]:
    archive = _open_office_zip(content)
    slides = sorted((n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
    pages = []
    for number, name in enumerate(slides, 1):
        paragraphs = [_runs_text(p, _A) for p in _xml(archive, name).iter(f"{_A}p")]
        pages.append(_page("\n".join(p for p in paragraphs if p), number))
    return pages


# extension -> (MIME type, "# " lines are headings, extractor(content) -> pages)
DOCUMENT_FORMATS = {
    ".txt": ("text/plain", False, text_pages),
    ".md": ("text/markdown", True, text_pages),
    ".markdown": ("text/markdown", True, text_pages),
    ".csv": ("text/csv", False, csv_pages),
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", True, docx_pages),
    ".pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", False, pptx_pages),
}


def document_format(filename: str, content: bytes) -> tuple[str, str]:
    """Validates the upload and returns (kind, mime_type); kind is "pdf" or
    the file extension. Raises ValueError for anything unsupported."""
    suffix = Path(filename.lower()).suffix
    if suffix in DOCUMENT_FORMATS:
        return suffix, DOCUMENT_FORMATS[suffix][0]
    validate_pdf(filename, content)
    return "pdf", "application/pdf"


def extract_pages(kind: str, content: bytes) -> tuple[list[dict], bool]:
    """(pages, headings) for a non-PDF kind returned by document_format()."""
    _mime, headings, extractor = DOCUMENT_FORMATS[kind]
    return extractor(content), headings


def structured_chunks(pages: list[dict], max_tokens: int = 400, headings: bool = True) -> list[dict]:
    chunks, heading_path, pending = [], [], []
    start_page = end_page = None

    def flush() -> None:
        nonlocal pending, start_page, end_page
        text = "\n\n".join(pending).strip()
        if text:
            chunks.append({
                "text": text,
                "heading_path": list(heading_path),
                "page": start_page,
                "page_end": end_page,
            })
        pending = []
        start_page = end_page = None

    for page in pages:
        page_no = int(page["metadata"]["page_number"])
        for raw in page["text"].splitlines():
            line = raw.strip()
            if not line:
                continue
            heading_match = headings and re.match(r"^(#{1,6})\s+(.+)", line)
            if heading_match:
                flush()
                level = len(heading_match.group(1))
                heading_path = heading_path[:level - 1] + [heading_match.group(2).strip()]
                continue
            if start_page is None:
                start_page = page_no
            end_page = page_no
            line_tokens = len(re.findall(r"\w+|[^\w\s]", line))
            pending_tokens = sum(
                len(re.findall(r"\w+|[^\w\s]", item)) for item in pending
            )
            if pending and pending_tokens + line_tokens > max_tokens:
                flush()
                start_page = page_no
                end_page = page_no
            pending.append(line)
    flush()
    return chunks
