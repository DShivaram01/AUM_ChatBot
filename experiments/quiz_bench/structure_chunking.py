"""Experiment 2: markdown/structure-aware splitting.

Builds blocks from the production (pymupdf4llm) markdown: headings set the
section path, paragraphs and whole markdown tables are atomic blocks. Blocks
are packed within a section up to `size` Mistral tokens; a section never
shares a chunk with the next one; an oversized paragraph is split at sentence
boundaries; a table is never split. Tiny sections (< MIN_TOKENS) are merged
into the following section of the same parent so headings like "2.3.1 Cosine
Similarity" (whose equation the extractor dropped) are not stranded alone.

`filtered` variants drop non-quizzable back/front matter from the index:
the References section, the author/affiliation block before the first
heading, and Acknowledgments/Funding/Data/Code availability/Competing
interests/Authors' contributions paragraphs.

Also registers `plain_recursive512`: plain PyMuPDF text (no markdown) split
the same way, to compare extraction fidelity at equal chunk size.
"""

import re
from functools import lru_cache

from experiments.quiz_bench.chunkers import register

MIN_TOKENS = 60
BACK_MATTER_HEADINGS = re.compile(r"^(references|bibliography)$", re.I)
BACK_MATTER_PARAS = re.compile(r"^\**\s*(acknowledg|funding|data availability|code availability|competing interests|authors'? contributions)", re.I)


@lru_cache(maxsize=1)
def _tok():
    import config
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.LLM_ID)


def ntok(t: str) -> int:
    return len(_tok()(t, add_special_tokens=False)["input_ids"])


def _clean_heading(line: str) -> str:
    return re.sub(r"[*#]", "", line).strip()


def blocks(pages: list[dict]) -> list[dict]:
    """[{"kind": heading|para|table, "level", "text", "page"}] in reading order."""
    out = []
    for p in pages:
        pg = p["metadata"]["page_number"]
        lines = p["text"].splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line.strip():
                i += 1
                continue
            m = re.match(r"^(#{1,6})\s+(.+)", line.strip())
            if m:
                out.append({"kind": "heading", "level": len(m.group(1)), "text": _clean_heading(m.group(2)), "page": pg})
                i += 1
                continue
            if line.lstrip().startswith("|"):
                rows = []
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    rows.append(lines[i].rstrip()); i += 1
                out.append({"kind": "table", "text": "\n".join(rows), "page": pg})
                continue
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("|", "#")):
                para.append(lines[i].strip()); i += 1
            out.append({"kind": "para", "text": " ".join(para), "page": pg})
    return out


def _sentences(text: str, size: int) -> list[str]:
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and ntok(cur + " " + s) > size:
            parts.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    return parts + ([cur] if cur else [])


def structure_chunks(pages: list[dict], size: int = 512, filtered: bool = False) -> list[dict]:
    # 1) sections: (heading_path, [blocks])
    sections, path, levels, current = [], [], [], {"path": [], "blocks": []}
    seen_heading = False
    for b in blocks(pages):
        if b["kind"] == "heading":
            sections.append(current)
            while levels and levels[-1] >= b["level"]:
                levels.pop(); path.pop()
            levels.append(b["level"]); path.append(b["text"])
            current = {"path": list(path), "blocks": []}
            seen_heading = True
            continue
        if filtered and not seen_heading and len(sections) == 0:
            continue                                   # author/affiliation front matter
        if filtered and b["kind"] == "para" and BACK_MATTER_PARAS.match(b["text"]):
            continue
        current["blocks"].append(b)
    sections.append(current)
    if filtered:
        sections = [s for s in sections if not any(BACK_MATTER_HEADINGS.match(h) for h in s["path"])]
    sections = [s for s in sections if s["blocks"]]

    # 2) merge tiny sections into the next section with the same parent
    merged = []
    for s in sections:
        if merged and sum(ntok(b["text"]) for b in merged[-1]["blocks"]) < MIN_TOKENS \
                and merged[-1]["path"][:-1] == s["path"][:-1]:
            prev = merged.pop()
            s = {"path": s["path"][:-1] or s["path"], "blocks": prev["blocks"] + s["blocks"],
                 "merged_from": prev.get("merged_from", [prev["path"]]) + [s["path"]]}
        merged.append(s)

    # 3) pack blocks within each section
    chunks = []
    for s in merged:
        cur, cur_tok, pages_ = [], 0, []
        units = []
        for b in s["blocks"]:
            if b["kind"] == "para" and ntok(b["text"]) > size:
                units += [{"text": t, "page": b["page"]} for t in _sentences(b["text"], size)]
            else:
                units.append(b)                         # tables stay whole even if > size
        for u in units:
            t = ntok(u["text"])
            if cur and cur_tok + t > size:
                chunks.append({"text": "\n\n".join(cur), "page": pages_[0], "page_end": pages_[-1], "heading_path": s["path"]})
                cur, cur_tok, pages_ = [], 0, []
            cur.append(u["text"]); cur_tok += t; pages_.append(u["page"])
        if cur:
            chunks.append({"text": "\n\n".join(cur), "page": pages_[0], "page_end": pages_[-1], "heading_path": s["path"]})
    return chunks


@lru_cache(maxsize=1)
def plain_pages() -> list[dict]:
    import pymupdf
    from experiments.quiz_bench.fixture import paper_bytes
    doc = pymupdf.open(stream=paper_bytes(), filetype="pdf")
    out = [{"metadata": {"page_number": i + 1}, "text": page.get_text("text")} for i, page in enumerate(doc)]
    doc.close()
    return out


def plain_recursive(size: int) -> list[dict]:
    """Plain-text extraction, paragraph/sentence packing to `size` (no headings)."""
    chunks, cur, cur_tok, pg = [], [], 0, []
    for p in plain_pages():
        for para in re.split(r"\n\s*\n", p["text"]):
            para = " ".join(para.split())
            if not para:
                continue
            for unit in (_sentences(para, size) if ntok(para) > size else [para]):
                t = ntok(unit)
                if cur and cur_tok + t > size:
                    chunks.append({"text": "\n\n".join(cur), "page": pg[0], "page_end": pg[-1], "heading_path": []})
                    cur, cur_tok, pg = [], 0, []
                cur.append(unit); cur_tok += t; pg.append(p["metadata"]["page_number"])
    if cur:
        chunks.append({"text": "\n\n".join(cur), "page": pg[0], "page_end": pg[-1], "heading_path": []})
    return chunks


for _size in (256, 400, 512, 768):
    register(f"section{_size}")(lambda pages, s=_size: structure_chunks(pages, s, False))
    register(f"section{_size}_filtered")(lambda pages, s=_size: structure_chunks(pages, s, True))
register("plain_recursive512")(lambda pages: plain_recursive(512))
