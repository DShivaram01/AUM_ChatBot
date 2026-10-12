"""CS 451 RAG benchmark: document loading, evidence-unit spans, containment.

Only the PDF is ingested by the system under test. The ground-truth JSON is
read here, by the evaluator, to score retrieval -- never passed to a
retriever, prompt or index.

Evidence units are the labelled passages (Sx-Pyy), tables (Sx-Tyy) and
equations (Sx-Eyy). A unit's span starts at its label on the designated page
(the contents page p.2 is ignored) and ends at the next label of any kind
(capped at MAX_SPAN chars). A chunk CONTAINS a unit when it holds at least
CONTAIN_FRAC of the unit's normalised 8-word shingles -- the substantive
text, not merely the label (per evaluator guide section 5).
"""

import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

BENCH = Path(os.environ.get("CS451_DIR", Path.home() / "aum_benchmarks" / "cs451_rag"))
PDF_SHA256 = "a8d7420c7a9488e8b4295c139377f74b8d2c52f428c74bf7c35fc2502d540191"
MAX_SPAN = 3000
CONTAIN_FRAC = 0.5
LABEL_RE = re.compile(r"\bS[1-6]-[PTEF]\d{2}\b")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_#`|]", " ", t)).strip().lower()


def pdf_bytes() -> bytes:
    data = (BENCH / "paper.pdf").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PDF_SHA256:
        raise RuntimeError(f"benchmark PDF hash mismatch: {digest}")
    return data


@lru_cache(maxsize=1)
def ground_truth() -> dict:
    return json.loads((BENCH / "ground_truth.json").read_text())


@lru_cache(maxsize=1)
def pages() -> list[dict]:
    from ingestion.document_parser import extract_pdf_pages
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_bytes()); f.flush()
        return [{"metadata": {"page_number": int(p["metadata"]["page_number"])}, "text": p["text"]}
                for p in extract_pdf_pages(Path(f.name))]


@lru_cache(maxsize=1)
def document() -> tuple[str, list[int]]:
    """(full text, page start offsets) -- pages joined with blank lines."""
    text, starts = "", []
    for p in pages():
        starts.append(len(text))
        text += p["text"] + "\n\n"
    return text, starts


def page_of(offset: int) -> int:
    _, starts = document()
    import bisect
    return pages()[bisect.bisect_right(starts, offset) - 1]["metadata"]["page_number"]


@lru_cache(maxsize=1)
def unit_spans() -> dict[str, tuple[int, int]]:
    gt = ground_truth()
    regs = {**gt["passage_registry"], **gt["table_registry"], **gt["equation_registry"]}
    text, starts = document()
    occ = []
    for m in LABEL_RE.finditer(text):
        if page_of(m.start()) != 2:                      # skip the contents page
            occ.append((m.start(), m.group(0)))
    spans = {}
    for i, (pos, lid) in enumerate(occ):
        if lid not in regs or lid in spans or page_of(pos) != regs[lid]["page_number"]:
            continue
        end = occ[i + 1][0] if i + 1 < len(occ) else len(text)
        spans[lid] = (pos, min(end, pos + MAX_SPAN))
    missing = set(regs) - set(spans)
    if missing:
        raise RuntimeError(f"units without a span: {sorted(missing)}")
    return spans


def _shingles(t: str, n: int = 8) -> set[str]:
    w = norm(LABEL_RE.sub(" ", t)).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if w else set()


@lru_cache(maxsize=1)
def unit_shingles() -> dict[str, set[str]]:
    text, _ = document()
    return {lid: _shingles(text[a:b]) for lid, (a, b) in unit_spans().items()}


def contained_units(chunk_text: str) -> list[str]:
    """Units whose substantive text the chunk holds, in document order."""
    cs = _shingles(chunk_text)
    out = []
    for lid, us in unit_shingles().items():
        if us and len(us & cs) / len(us) >= CONTAIN_FRAC:
            out.append(lid)
    spans = unit_spans()
    return sorted(out, key=lambda l: spans[l][0])
