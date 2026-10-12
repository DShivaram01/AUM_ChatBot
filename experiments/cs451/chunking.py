"""Chunkers for the CS 451 retrieval matrix. All return
[{"text", "start", "end"}] with character offsets into units.document().

Token counts use the production generator tokenizer (Mistral), per the
evaluator guide (boundaries depend on the consuming pipeline's tokenizer).

  PROD          production ingestion.document_parser.structured_chunks
  F{s}_ov{o}    fixed token windows of s tokens, o% overlap
  R512          recursive paragraph -> line -> sentence -> word, 512, 15 %
  H512          structure-aware: one labelled unit (passage + its tables /
                equations) never split across chunks unless > 512; units packed
                within a section; markdown tables never split
  SW            400-token windows, 200-token stride (50 % overlap)
  PC            children R256 (no overlap) -> parents H1024 (handled in
                retrieval: children are searched, their parents returned)
"""

import re
from functools import lru_cache

from experiments.cs451.units import document, norm

SEPARATORS = ["\n\n", "\n", ". ", " "]


@lru_cache(maxsize=1)
def tokenizer():
    import config
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.LLM_ID)


def ntok(t: str) -> int:
    return len(tokenizer()(t, add_special_tokens=False)["input_ids"])


@lru_cache(maxsize=1)
def token_offsets() -> list[tuple[int, int]]:
    text, _ = document()
    return tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]


def fixed(size: int, overlap_frac: float) -> list[dict]:
    text, _ = document()
    offs = token_offsets()
    step = max(1, size - int(size * overlap_frac))
    out = []
    for i in range(0, len(offs), step):
        j = min(len(offs), i + size)
        a, b = offs[i][0], offs[j - 1][1]
        out.append({"text": text[a:b], "start": a, "end": b})
        if j == len(offs):
            break
    return out


def _pieces(a: int, b: int, size: int, seps: list[str]) -> list[tuple[int, int]]:
    text, _ = document()
    if ntok(text[a:b]) <= size or not seps:
        return [(a, b)]
    sep = seps[0]
    out, pos = [], a
    while pos < b:
        k = text.find(sep, pos, b)
        end = b if k == -1 else k + len(sep)
        out += _pieces(pos, end, size, seps[1:]) if ntok(text[pos:end]) > size else [(pos, end)]
        pos = end
    return out


def recursive(size: int, overlap_frac: float) -> list[dict]:
    text, _ = document()
    pieces = _pieces(0, len(text), size, SEPARATORS)
    overlap = int(size * overlap_frac)
    out, cur, cur_tok = [], [], 0
    for p in pieces:
        t = ntok(text[p[0]:p[1]])
        if cur and cur_tok + t > size:
            out.append((cur[0][0], cur[-1][1]))
            carry, ct = [], 0
            for q in reversed(cur):
                qt = ntok(text[q[0]:q[1]])
                if ct + qt > overlap:
                    break
                carry.insert(0, q); ct += qt
            cur, cur_tok = carry, ct
        cur.append(p); cur_tok += t
    if cur:
        out.append((cur[0][0], cur[-1][1]))
    return [{"text": text[a:b], "start": a, "end": b} for a, b in out if text[a:b].strip()]


def _blocks() -> list[tuple[int, int, str]]:
    """(start, end, kind) for headings, paragraphs and whole tables."""
    text, _ = document()
    out, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    blocks = []
    for a, b in out:
        s = text[a:b].lstrip()
        kind = "heading" if s.startswith("#") else "table" if s.startswith("|") else "para"
        if kind == "table" and blocks and blocks[-1][2] == "table" and text[blocks[-1][1]:a].strip() == "":
            blocks[-1] = (blocks[-1][0], b, "table")
        else:
            blocks.append((a, b, kind))
    return blocks


def structure(size: int) -> list[dict]:
    """Group blocks into units starting at each labelled-passage heading
    (###### **Sx-Pyy ...**) or section heading; pack whole units up to `size`;
    a unit larger than `size` is split at block boundaries; tables never split."""
    text, _ = document()
    units, cur = [], []
    for a, b, kind in _blocks():
        s = text[a:b].lstrip()
        starts_unit = kind == "heading" and (re.search(r"S[1-6]-P\d{2}", s) or s.startswith(("# ", "## ", "### ", "#### ")))
        if starts_unit and cur:
            units.append(cur); cur = []
        cur.append((a, b))
    if cur:
        units.append(cur)
    out, pack = [], []
    def flush():
        if pack:
            out.append((pack[0][0], pack[-1][1]))
    tok = 0
    for u in units:
        ut = ntok(text[u[0][0]:u[-1][1]])
        if ut > size:
            flush(); pack, tok = [], 0
            for blk in u:
                bt = ntok(text[blk[0]:blk[1]])
                if pack and tok + bt > size:
                    flush(); pack, tok = [], 0
                pack.append(blk); tok += bt
            flush(); pack, tok = [], 0
            continue
        if pack and tok + ut > size:
            flush(); pack, tok = [], 0
        pack += u; tok += ut
    flush()
    return [{"text": text[a:b], "start": a, "end": b} for a, b in out if text[a:b].strip()]


def production() -> list[dict]:
    from ingestion.document_parser import structured_chunks
    from experiments.cs451.units import pages
    text, _ = document()
    out = []
    for c in structured_chunks(pages()):
        out.append({"text": c["text"], "start": None, "end": None, "page": c["page"], "heading_path": c["heading_path"]})
    return out


def build(name: str) -> list[dict]:
    if name == "PROD":
        return production()
    if m := re.fullmatch(r"F(\d+)_ov(\d+)", name):
        return fixed(int(m.group(1)), int(m.group(2)) / 100)
    if m := re.fullmatch(r"R(\d+)", name):
        return recursive(int(m.group(1)), 0.15)
    if m := re.fullmatch(r"H(\d+)", name):
        return structure(int(m.group(1)))
    if name == "SW":
        return fixed(400, 0.5)
    raise ValueError(name)


def parent_child() -> tuple[list[dict], list[dict], list[int]]:
    """children (R256, no overlap), parents (H1024), parent index per child."""
    children = recursive(256, 0.0)
    parents = structure(1024)
    owner = []
    for c in children:
        mid = (c["start"] + c["end"]) // 2
        owner.append(next((i for i, p in enumerate(parents) if p["start"] <= mid < p["end"]), len(parents) - 1))
    return children, parents, owner
