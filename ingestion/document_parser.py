"""Structured extraction for private uploaded documents (PDF, TXT, Markdown)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def validate_pdf(filename: str, content: bytes) -> None:
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
        raise ValueError("Only PDF, TXT and Markdown (.md) uploads are supported.")


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_pdf_pages(path: Path) -> list[dict]:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path), page_chunks=True)


# Task 57-B: plain-text formats. They have no pages, so the whole file is
# "page 1" in the same page-dict shape pymupdf4llm returns, and reuse the
# same chunker. Extension -> (MIME type, whether "# " lines are headings).
TEXT_FORMATS = {".txt": ("text/plain", False), ".md": ("text/markdown", True),
                ".markdown": ("text/markdown", True)}


def text_pages(filename: str, content: bytes) -> list[dict]:
    if b"\x00" in content:
        raise ValueError("The file does not look like a text document.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text uploads must be UTF-8 encoded.") from exc
    return [{"metadata": {"page_number": 1}, "text": text}]


def document_format(filename: str, content: bytes) -> tuple[str, str]:
    """Validates the upload and returns (kind, mime_type); kind is "pdf" or
    the text extension. Raises ValueError for anything unsupported."""
    suffix = Path(filename.lower()).suffix
    if suffix in TEXT_FORMATS:
        return suffix, TEXT_FORMATS[suffix][0]
    validate_pdf(filename, content)
    return "pdf", "application/pdf"


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
