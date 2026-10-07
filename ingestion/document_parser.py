"""PDF-only structured extraction for private uploaded documents."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def validate_pdf(filename: str, content: bytes) -> None:
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
        raise ValueError("Only valid PDF uploads are supported.")


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_pdf_pages(path: Path) -> list[dict]:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path), page_chunks=True)


def structured_chunks(pages: list[dict], max_tokens: int = 400) -> list[dict]:
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
            heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
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
