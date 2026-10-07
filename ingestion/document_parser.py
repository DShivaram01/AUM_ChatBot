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


def structured_chunks(pages: list[dict], max_chars: int = 2200) -> list[dict]:
    chunks, heading, pending = [], "", []
    def flush() -> None:
        nonlocal pending
        text = "\n\n".join(pending).strip()
        if text:
            chunks.append({"text": text, "heading_path": [heading] if heading else []})
        pending = []
    for page in pages:
        page_no = int(page["metadata"]["page_number"])
        for raw in page["text"].splitlines():
            line = raw.strip()
            if not line: continue
            if re.match(r"^#{1,6}\s+", line):
                flush(); heading = re.sub(r"^#+\s+", "", line); continue
            pending.append(line)
            if sum(map(len, pending)) >= max_chars: flush()
        if pending and chunks: chunks[-1].setdefault("page_end", page_no)
    flush()
    return chunks
