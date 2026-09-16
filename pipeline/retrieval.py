"""
pipeline/retrieval.py
======================
Data loading, indexing, name matching, and hybrid (BM25 + FAISS + RRF)
retrieval, extracted from backend.py.

NOTE on the RetrievalCandidate import inside retrieve_cos_rrf(): it's a
local import, not a top-level one, on purpose. pipeline/classifier.py
imports query_name_index/build_name_index/NAME_INDEX from this module, and
retrieve_cos_rrf() needs RetrievalCandidate from pipeline/classifier.py --
a top-level import either direction would be circular. Deferring the import
to call time (inside the function body) breaks the cycle since by the time
retrieve_cos_rrf() actually runs, both modules have finished importing.
"""

import os
import json
import re
import shutil
import time
import time as _time
import hashlib
from datetime import datetime
from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

import config
from pipeline.memory import logger

SCRATCH   = config.SCRATCH
EMB_STORE = config.EMB_STORE
RRF_K     = config.RRF_K


# ── Normalizers + fuzzy name matching (backend.py:448-475) ──────────

def norm_text(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


DEPARTMENT_CANONICAL_NAMES = (
    "Biology and Environmental Science",
    "Chemistry",
    "Computer Science and Computer Information Systems",
    "Mathematics",
    "Psychology",
)


def canonicalize_departments(raw_department):
    """Normalize source spellings at ingest time into the five COS departments.

    A record may genuinely span more than one department, so the normalized
    value is a list rather than an unsafe substring-derived single label.
    """
    value = norm_text(raw_department)
    found = []
    aliases = (
        ("Biology and Environmental Science", ("biology", "environmental science", "marine science")),
        ("Chemistry", ("chemistry",)),
        ("Computer Science and Computer Information Systems", ("computer science", "computer sciences", "computer information systems", "computer and computer information systems")),
        ("Mathematics", ("mathematics",)),
        ("Psychology", ("psychology",)),
    )
    for canonical, terms in aliases:
        if any(term in value for term in terms):
            found.append(canonical)
    return found

def norm_name(name):
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"\b(dr\.?|prof\.?|professor)\b", "", name)
    name = name.replace(".", " ")
    return re.sub(r"\s+", " ", name).strip()

def name_tokens(name):
    return [t for t in norm_name(name).split() if len(t) > 2]

def names_overlap(query_name, corpus_name):
    qt = name_tokens(query_name)
    ct = name_tokens(corpus_name)
    if not qt or not ct:
        return False
    for q in qt:
        for c in ct:
            if q == c:
                return True
            if len(q) >= 6 and len(c) >= 6 and q[:6] == c[:6]:
                return True
    return False


# ── Scratch copy (backend.py:419-441, warmup call moved to main.py) ──

def scratch_copy(src, dst):
    if os.path.exists(dst):
        return dst
    if not os.path.exists(src):
        logger.warning(f"[Scratch] Source not found: {src}")
        return src
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    t0 = time.time()
    shutil.copy2(src, dst)
    mb = os.path.getsize(dst) / 1e6
    logger.info(f"[Scratch] {os.path.basename(dst)} ({mb:.1f} MB, {time.time()-t0:.1f}s)")
    return dst


# ── COS data loading + embedding store (backend.py:480-558) ─────────

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                meta = row.setdefault("metadata", {})
                departments = canonicalize_departments(meta.get("department", ""))
                meta["departments"] = departments
                if departments:
                    meta["department"] = " / ".join(departments)
                rows.append(row)
    logger.info(f"[JSONL] Loaded {len(rows)} rows from {path}")
    return rows

def _manifest_path():
    return f"{SCRATCH}/embeddings_manifest.json"

def _save_manifest(model_id, n, dim, text_hash):
    m = {"model_name": model_id, "n_vectors": n, "dim": dim,
         "text_hash": text_hash, "created_at": datetime.utcnow().isoformat() + "Z"}
    p = _manifest_path()
    with open(p, "w") as f:
        json.dump(m, f, indent=2)
    shutil.copy2(p, f"{EMB_STORE}/embeddings_manifest.json")
    logger.info(f"[Manifest] Saved - model={model_id}, n={n}, dim={dim}")

def _check_manifest(model_id, text_hash):
    p = _manifest_path()
    if not os.path.exists(p):
        logger.info("[Manifest] Not found - will build store")
        return False
    try:
        with open(p) as f:
            m = json.load(f)
        if m.get("model_name") != model_id:
            logger.warning(f"[Manifest] Model mismatch: stored={m.get('model_name')} current={model_id}")
            return False
        if m.get("text_hash") != text_hash:
            logger.warning("[Manifest] Corpus changed - rebuilding")
            return False
        logger.info(f"[Manifest] Valid - {m['n_vectors']} vectors")
        return True
    except Exception as e:
        logger.warning(f"[Manifest] Error: {e}")
        return False

def _texts_hash(texts, metadata=None):
    """Fingerprint all retrieval-visible corpus fields, not just abstracts."""
    return hashlib.md5(json.dumps(
        {"texts": texts, "metadata": metadata}, sort_keys=True
    ).encode()).hexdigest()

def build_cos_store(rows, embedder):
    IDS   = [r["id"]       for r in rows]
    TEXTS = [r["text"]     for r in rows]
    META  = [r["metadata"] for r in rows]
    th    = _texts_hash(TEXTS, META)
    required = ["embeddings.npy","faiss_ip.index","ids.json","metadata.json","texts.json"]
    files_ok = all(os.path.exists(f"{SCRATCH}/{fn}") for fn in required)

    if not files_ok or not _check_manifest(config.EMBED_MODEL_ID, th):
        logger.info(f"[COS Build] Encoding {len(TEXTS)} texts...")
        t0  = time.time()
        EMB = embedder.encode(TEXTS, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        logger.info(f"[COS Build] Done in {time.time()-t0:.1f}s, shape={EMB.shape}")
        idx = faiss.IndexIDMap2(faiss.IndexFlatIP(EMB.shape[1]))
        idx.add_with_ids(EMB, np.arange(len(IDS)).astype(np.int64))
        np.save(f"{SCRATCH}/embeddings.npy", EMB)
        faiss.write_index(idx, f"{SCRATCH}/faiss_ip.index")
        with open(f"{SCRATCH}/ids.json",      "w") as f: json.dump(IDS,  f)
        with open(f"{SCRATCH}/metadata.json", "w") as f: json.dump(META, f, indent=2)
        with open(f"{SCRATCH}/texts.json",    "w") as f: json.dump(TEXTS, f)
        _save_manifest(config.EMBED_MODEL_ID, len(IDS), EMB.shape[1], th)
        for fn in required:
            shutil.copy2(f"{SCRATCH}/{fn}", f"{EMB_STORE}/{fn}")
        logger.info("[COS Build] Saved to emb_store")
    else:
        logger.info("[COS] Using cached store")

    EMB   = np.load(f"{SCRATCH}/embeddings.npy")
    index = faiss.read_index(f"{SCRATCH}/faiss_ip.index")
    with open(f"{SCRATCH}/ids.json")      as f: IDS   = json.load(f)
    with open(f"{SCRATCH}/metadata.json") as f: META  = json.load(f)
    with open(f"{SCRATCH}/texts.json")    as f: TEXTS = json.load(f)
    logger.info(f"[COS] Index ready: {index.ntotal} vectors")
    return index, EMB, IDS, META, TEXTS


# ── Housing PDF chunking + index ─────────────────────────────────────
# Housing policies are extracted as page-aware Markdown.  The Markdown
# headings are produced by pymupdf4llm's layout-aware PDF parser; this avoids
# treating arbitrary raw text lines (including the table of contents) as a
# policy heading.

_HRL_MARKDOWN_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(?:\*\*)?(HRL\.\d{4}\b.*?)(?:\*\*)?\s*$",
    re.IGNORECASE,
)
_MAJOR_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,2}\s+")


def _is_housing_boilerplate(line):
    normalized = re.sub(r"[`*_#|]", "", line)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return (
        not normalized
        or normalized == "table of contents"
        or normalized.startswith("aum housing and residence life contractual obligations")
        or re.match(r"^page \d+( of \d+)?$", normalized) is not None
        or normalized.replace("-", "") == ""
    )

def _markdown_text_blocks(page_markdown, page_number):
    """Yield clean Markdown text blocks while retaining their source page."""
    block = []
    for raw_line in page_markdown.splitlines():
        line = raw_line.strip()
        if not line:
            if block:
                yield page_number, " ".join(block)
                block = []
            continue
        if _is_housing_boilerplate(line):
            continue
        # Table rows are TOC artifacts in this document. Content tables can
        # later be handled as structured Markdown rather than line fragments.
        if line.startswith("|") and line.endswith("|"):
            continue
        block.append(line)
    if block:
        yield page_number, " ".join(block)


def _append_housing_chunk(chunks, section, blocks):
    """Emit page-attributed chunks at paragraph boundaries, never raw offsets."""
    current = []
    current_chars = 0

    def flush():
        text = "\n\n".join(text for _, text in current).strip()
        if len(text) <= 60:
            return
        pages = sorted({page for page, _ in current})
        chunks.append({
            "text": text,
            "section": section,
            "page": pages[0],
            "page_end": pages[-1],
            "source": "AUM Housing Policy",
            "_chunk_version": 4,
        })

    for page, text in blocks:
        # The parser preserves paragraph boundaries, so a normal chunk does
        # not cut through a sentence as the former character slicer could.
        if current and current_chars + len(text) + 2 > 700:
            flush()
            current = []
            current_chars = 0
        current.append((page, text))
        current_chars += len(text) + 2
    flush()


def extract_housing_chunks(pdf_path):
    """Convert a policy PDF into page-aware HRL Markdown chunks.

    The generic ``pdf_path -> chunk records`` interface can be reused by a
    future document-ingestion feature without coupling that feature to Housing.
    """
    try:
        import pymupdf4llm
    except ImportError:
        raise ImportError("Install pymupdf4llm==1.28.2 to extract Housing PDFs")

    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    logger.info(f"[Housing] Markdown extraction produced {len(pages)} pages")
    chunks = []
    section = None
    section_blocks = []
    for page in pages:
        page_number = int(page["metadata"]["page_number"])
        lines = page["text"].splitlines()
        body_lines = []
        for line in lines:
            match = _HRL_MARKDOWN_HEADING_RE.match(line)
            if match:
                if section:
                    for block in _markdown_text_blocks("\n".join(body_lines), page_number):
                        section_blocks.append(block)
                    _append_housing_chunk(chunks, section, section_blocks)
                section = re.sub(r"\s+", " ", match.group(1)).strip()
                section_blocks = []
                body_lines = []
            elif section and _MAJOR_MARKDOWN_HEADING_RE.match(line):
                # A new top-level Markdown section ends the final HRL policy
                # even when the source document does not introduce another
                # HRL code afterwards (for example, Medical Services).
                for block in _markdown_text_blocks("\n".join(body_lines), page_number):
                    section_blocks.append(block)
                _append_housing_chunk(chunks, section, section_blocks)
                section = None
                section_blocks = []
                body_lines = []
            elif section:
                body_lines.append(line)
        if section:
            section_blocks.extend(_markdown_text_blocks("\n".join(body_lines), page_number))
    if section:
        _append_housing_chunk(chunks, section, section_blocks)
    return chunks

def load_or_build_housing(pdf_path, embedder):
    cc = f"{SCRATCH}/housing_chunks.json"
    he = f"{SCRATCH}/housing_emb.npy"
    hi = f"{SCRATCH}/housing_faiss.index"

    if all(os.path.exists(p) for p in [cc, he, hi]):
        logger.info("[Housing] Loading cached index...")
        with open(cc) as f: chunks = json.load(f)
        if chunks and all(chunk.get("_chunk_version") == 4 for chunk in chunks):
            H_EMB   = np.load(he)
            H_index = faiss.read_index(hi)
            logger.info(f"[Housing] {len(chunks)} chunks, {H_index.ntotal} vectors")
            return H_index, H_EMB, chunks
        logger.info("[Housing] Cached chunks predate Markdown extraction; rebuilding")

    logger.info("[Housing] Building from PDF...")
    if not os.path.exists(pdf_path):
        logger.error(f"[Housing] PDF not found: {pdf_path}")
        return None, None, []

    chunks = extract_housing_chunks(pdf_path)
    texts  = [c["text"] for c in chunks]
    H_EMB  = embedder.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    H_index = faiss.IndexIDMap2(faiss.IndexFlatIP(H_EMB.shape[1]))
    H_index.add_with_ids(H_EMB, np.arange(len(chunks)).astype(np.int64))

    with open(cc, "w") as f: json.dump(chunks, f, indent=2)
    np.save(he, H_EMB)
    faiss.write_index(H_index, hi)
    for src, fn in [(cc,"housing_chunks.json"),(he,"housing_emb.npy"),(hi,"housing_faiss.index")]:
        shutil.copy2(src, f"{EMB_STORE}/{fn}")
    logger.info(f"[Housing] Built - {len(chunks)} chunks")
    return H_index, H_EMB, chunks


# ── BM25 index (backend.py:658-682) ──────────────────────────────────

def bm25_tokenize(text):
    """Stable tokenization for both BM25 indexing and query scoring."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def build_bm25(meta_list, texts_list):
    corpus = []
    for meta, text in zip(meta_list, texts_list):
        abstract = text.split("Abstract:", 1)[-1][:600] if "Abstract:" in text else ""
        combined = " ".join([
            meta.get("title", ""),
            meta.get("mentor", "") or "",
            " ".join(meta.get("lead_presenters", []) or []),
            " ".join(meta.get("other_authors",   []) or []),
            meta.get("department", "") or "",
            meta.get("category",   "") or "",
            str(meta.get("year", "") or ""),
            " ".join(meta.get("keywords",        []) or []),
            abstract,
        ])
        corpus.append(bm25_tokenize(combined))
    bm = BM25Okapi(corpus)
    logger.info(f"[BM25] Built - {len(corpus)} documents")
    return bm

def bm25_search(bm25_index, query, top_k=60):
    tokens = bm25_tokenize(query)
    scores = np.array(bm25_index.get_scores(tokens), dtype=np.float32)
    idxs   = np.argsort(scores)[::-1][:min(top_k, len(scores))]
    return idxs, scores


# ── Name index (backend.py:1693-1889) ────────────────────────────────
# Listed as "build_name_index(), query_name_index(), NAME_INDEX" in the
# task spec; _index_one_name isn't listed but build_name_index() calls it
# directly, so it has to live here too.

NAME_INDEX = {
    "full":   defaultdict(set),   # norm_full_name  -> {original_name, ...}
    "last":   defaultdict(set),   # last_token      -> {original_name, ...}
    "first":  defaultdict(set),   # first_token     -> {original_name, ...}
    "prefix": defaultdict(set),   # token[:5]       -> {original_name, ...}
}

def _index_one_name(raw_name: str):
    """Add one raw name string to NAME_INDEX."""
    if not raw_name or not raw_name.strip():
        return
    cleaned = re.sub(r"\b(dr\.?|prof\.?|professor)\b", "", raw_name, flags=re.IGNORECASE)
    cleaned = re.sub(r"\d+", "", cleaned)
    cleaned = re.sub(r"[^\w\s\-\']", " ", cleaned)
    norm    = re.sub(r"\s+", " ", cleaned).strip().lower()

    if len(norm) < 2:
        return

    tokens = norm.split()
    if not tokens:
        return

    NAME_INDEX["full"][norm].add(raw_name.strip())

    last = tokens[-1]
    if len(last) >= 3:
        NAME_INDEX["last"][last].add(raw_name.strip())

    first = tokens[0]
    if len(first) >= 4:
        NAME_INDEX["first"][first].add(raw_name.strip())

    for tok in tokens:
        if len(tok) >= 5:
            NAME_INDEX["prefix"][tok[:5]].add(raw_name.strip())


def build_name_index(meta_list):
    """Build NAME_INDEX from all mentor and presenter fields in corpus."""
    for bucket in NAME_INDEX.values():
        bucket.clear()
    for meta in meta_list:
        mentor_raw = meta.get("mentor", "") or ""
        for part in re.split(r"\band\b|&|,", mentor_raw, flags=re.IGNORECASE):
            _index_one_name(part.strip())

        for name in (meta.get("lead_presenters", []) or []):
            _index_one_name(name)

        for name in (meta.get("other_authors", []) or []):
            _index_one_name(name)

    total = sum(len(v) for v in NAME_INDEX["last"].values())
    logger.info(
        f"[NameIndex] Built — "
        f"{len(NAME_INDEX['full'])} full names, "
        f"{len(NAME_INDEX['last'])} last names, "
        f"{len(NAME_INDEX['first'])} first names, "
        f"{len(NAME_INDEX['prefix'])} prefixes, "
        f"{total} total entries"
    )


def query_name_index(query_text: str, with_scores=False):
    """
    Scan query_text against NAME_INDEX.
    Returns list of CANONICAL matched name strings, best match first.
    """
    q_norm = re.sub(r"\b(dr\.?|prof\.?|professor|ms\.?|mr\.?)\b", "",
                    query_text, flags=re.IGNORECASE)
    q_norm = re.sub(r"\d+", "", q_norm)
    q_norm = re.sub(r"[^\w\s]", " ", q_norm)
    q_norm = re.sub(r"\s+", " ", q_norm).strip().lower()
    q_tokens = [t for t in q_norm.split() if len(t) >= 2]

    matched = {}  # canonical_name -> match_score

    def _canonicalize(raw_name: str) -> str:
        c = re.sub(r"\b(dr\.?|prof\.?|professor|ms\.?|mr\.?)\b", "",
                   raw_name, flags=re.IGNORECASE)
        c = re.sub(r"\d+", "", c)
        c = re.sub(r"[^\w\s\-\']", " ", c)
        return re.sub(r"\s+", " ", c).strip()

    def _store_match(orig_name, score):
        canon = _canonicalize(orig_name)
        if len(canon) >= 3:
            matched[canon] = max(matched.get(canon, 0), score)

    for size in range(4, 0, -1):
        for start in range(len(q_tokens) - size + 1):
            window = q_tokens[start:start + size]
            candidate = " ".join(window)

            if size >= 2:
                first_tok = window[0]
                last_tok  = window[-1]

                if candidate in NAME_INDEX["full"]:
                    for orig in NAME_INDEX["full"][candidate]:
                        _store_match(orig, 40 + size)
                else:
                    candidates_first = set()
                    if len(first_tok) >= 4 and first_tok in NAME_INDEX["first"]:
                        candidates_first |= NAME_INDEX["first"][first_tok]
                    if len(first_tok) >= 5 and first_tok[:5] in NAME_INDEX["prefix"]:
                        candidates_first |= NAME_INDEX["prefix"][first_tok[:5]]

                    candidates_last = set()
                    if len(last_tok) >= 3 and last_tok in NAME_INDEX["last"]:
                        candidates_last |= NAME_INDEX["last"][last_tok]
                    if len(last_tok) >= 5 and last_tok[:5] in NAME_INDEX["prefix"]:
                        candidates_last |= NAME_INDEX["prefix"][last_tok[:5]]

                    both = candidates_first & candidates_last
                    for orig in both:
                        _store_match(orig, 35)

            else:
                tok = window[0]

                if len(tok) >= 3 and tok in NAME_INDEX["last"]:
                    for orig in NAME_INDEX["last"][tok]:
                        _store_match(orig, 30)

                if len(tok) >= 4 and tok in NAME_INDEX["first"]:
                    for orig in NAME_INDEX["first"][tok]:
                        _store_match(orig, 20)

                if len(tok) >= 5:
                    pfx = tok[:5]
                    if pfx in NAME_INDEX["prefix"]:
                        for orig in NAME_INDEX["prefix"][pfx]:
                            _store_match(orig, 10)

    if not matched:
        return []

    # A full name must not be diluted by a same-surname match.  Preserve a
    # tie only when the query is genuinely ambiguous (usually a bare surname).
    top_score = max(matched.values())

    seen_norm = set()
    results   = []
    for canon, score in sorted(matched.items(), key=lambda x: (-x[1], x[0])):
        if score != top_score:
            continue
        norm = canon.lower().strip()
        if norm not in seen_norm and len(norm) >= 3:
            seen_norm.add(norm)
            results.append((canon, score) if with_scores else canon)

    return results


# ── Person exact-match + rerank + RRF retrieval (backend.py:726-1033) ─
# NOTE: _relative_threshold() (backend.py:701-708) is NOT here -- the task
# spec assigns it to pipeline/answer.py instead, and nothing in this file
# calls it (build_cos_answer_streaming reimplements the same top/mean/std
# logic inline rather than calling _relative_threshold() -- true in the
# original backend.py too, this isn't a change in behavior).

def _exact_person_cands(person_hints, META_LIST, TEXTS_LIST):
    """
    Exact metadata scan for person queries.
    Search mentor, presenter, and co-author fields for every query, then rank
    the merged result set.  Co-author matches must not disappear merely
    because another record matched a primary field.
    """
    def _norm_hint(h):
        n = re.sub(r"\b(dr\.?|prof\.?|professor|ms\.?|mr\.?)\b", "",
                   h, flags=re.IGNORECASE)
        n = re.sub(r"\d+", "", n)
        n = re.sub(r"[^\w\s]", " ", n)
        return re.sub(r"\s+", " ", n).strip().lower()

    norm_hints = [_norm_hint(h) for h in person_hints if h.strip()]
    norm_hints = [h for h in norm_hints if len(h) >= 2]

    if not norm_hints:
        return []

    def _name_matches_hint(stored_name: str, hint_norm: str) -> bool:
        stored = re.sub(r"\d+", "", stored_name)
        stored = re.sub(r"\b(dr\.?|prof\.?|professor)\b", "",
                        stored, flags=re.IGNORECASE)
        stored = re.sub(r"[^\w\s]", " ", stored)
        stored = re.sub(r"\s+", " ", stored).strip().lower()

        hint_tokens   = hint_norm.split()
        stored_tokens = stored.split()

        if not hint_tokens or not stored_tokens:
            return False

        if len(hint_tokens) == 1:
            tok = hint_tokens[0]
            if tok == stored_tokens[-1]:
                return True
            if tok == stored_tokens[0] and len(tok) >= 4:
                return True
            if len(tok) >= 5:
                return any(len(s) >= 5 and tok[:5] == s[:5] for s in stored_tokens)
            return False
        else:
            first_hint = hint_tokens[0]
            last_hint  = hint_tokens[-1]

            first_ok = any(
                s == first_hint or
                (len(first_hint) >= 5 and len(s) >= 5 and first_hint[:5] == s[:5])
                for s in stored_tokens
            )
            last_ok = any(
                s == last_hint or
                (len(last_hint) >= 5 and len(s) >= 5 and last_hint[:5] == s[:5])
                for s in stored_tokens
            )
            return first_ok and last_ok

    def _scan_fields():
        project_scores = {}
        for idx, meta in enumerate(META_LIST):
            mentor_raw = meta.get("mentor", "") or ""
            mentor_parts = [
                p.strip() for p in
                re.split(r"\band\b|&|,", mentor_raw, flags=re.IGNORECASE)
                if p.strip()
            ]
            primary_names = mentor_parts + (meta.get("lead_presenters", []) or [])

            primary_names += (meta.get("other_authors", []) or [])

            matched_hints = 0
            for hint in norm_hints:
                if any(_name_matches_hint(name, hint) for name in primary_names):
                    matched_hints += 1

            if matched_hints > 0:
                project_scores[idx] = matched_hints

        return project_scores

    scores = _scan_fields()

    if not scores:
        return []

    sorted_idxs = sorted(scores, key=scores.get, reverse=True)
    return [{
        "idx":      idx,
        "vec":      1.0,
        "bm25":     float(scores[idx]),
        "combined": float(scores[idx]),
        "rerank":   None,
        "meta":     META_LIST[idx],
        "text":     TEXTS_LIST[idx],
        "_bm25_rank":  9999,
        "_faiss_rank": 9999,
    } for idx in sorted_idxs]


def _rerank(query, cands, reranker, max_rerank, qid, trace=None):
    keep   = min(len(cands), max_rerank)
    pairs  = [(query, cands[i]["text"]) for i in range(keep)]
    t0     = _time.time()
    scores = reranker.predict(pairs)
    ms     = (_time.time() - t0) * 1000
    logger.info(f"[{qid}] Rerank: {keep} pairs in {ms:.1f}ms")
    for i, s in enumerate(scores):
        cands[i]["rerank"] = float(s)
    cands[:keep] = sorted(cands[:keep], key=lambda x: x["rerank"], reverse=True)
    for rank, c in enumerate(cands[:5], 1):
        m = c["meta"]
        logger.info(
            f"[{qid}]   Rerank #{rank}: score={c['rerank']:.4f}  "
            f"title={m.get('title','')[:50]}  mentor={m.get('mentor','')}"
        )
    return cands


def retrieve_cos_rrf(
    query, qinfo, embedder, index, EMB,
    META_LIST, TEXTS_LIST, bm25_index, reranker,
    query_id="Q", top_k_bm25=60, top_k_vec=30, max_rerank=25,
    trace=None,
):
    # Local import -- see module docstring for why this can't be a
    # top-level import (breaks a circular dependency with classifier.py).
    from pipeline.classifier import RetrievalCandidate

    t_start = _time.time()
    logger.info(f"[{query_id}] == RETRIEVAL START ==")
    logger.info(f"[{query_id}] Query: {repr(query)}")
    logger.info(
        f"[{query_id}] Intent: {qinfo['type']} | "
        f"names={qinfo.get('person_hints',[])} "
        f"year={qinfo['year_hint']} dept={qinfo['dept_hint']}"
    )

    # ── Path A: Person — exact metadata scan ─────────────────────
    if qinfo["type"] == "TYPE_PERSON" and qinfo.get("person_hints"):
        hints = qinfo["person_hints"]
        logger.info(f"[{query_id}] Path A: Exact person scan {hints}")
        cands = _exact_person_cands(hints, META_LIST, TEXTS_LIST)
        logger.info(f"[{query_id}] Exact matches: {len(cands)}")
        for i, c in enumerate(cands[:5], 1):
            m = c["meta"]
            logger.info(
                f"[{query_id}]   Match #{i}: "
                f"title={m.get('title','')[:50]}  mentor={m.get('mentor','')}"
            )

        if trace:
            trace.bm25_hits  = len(cands)
            trace.faiss_hits = 0
            trace.rrf_total  = len(cands)

        if cands:
            t0    = _time.time()
            cands = _rerank(query, cands, reranker, max_rerank, query_id, trace)
            if trace:
                trace.t_rerank = (_time.time() - t0) * 1000
                trace.t_total  = (_time.time() - t_start) * 1000
                trace.candidates = []
                for rank, c in enumerate(cands[:10], 1):
                    m = c["meta"]
                    trace.candidates.append(RetrievalCandidate(
                        idx=c["idx"], title=m.get("title",""),
                        mentor=m.get("mentor",""), year=m.get("year",""),
                        department=m.get("department",""),
                        bm25_score=c["bm25"], rrf_score=c["combined"],
                        rerank_score=c.get("rerank", 0.0), final_rank=rank,
                    ))
            return cands

        logger.warning(f"[{query_id}] Exact match found nothing — falling to RRF")

    # ── Path B: RRF (BM25 + FAISS) ──────────────────────────────
    logger.info(f"[{query_id}] Path B: RRF")

    t0 = _time.time()
    bm_idxs, bm_scores = bm25_search(bm25_index, query, top_k=top_k_bm25)
    t_bm25 = (_time.time() - t0) * 1000
    pos_bm = [(int(i), float(bm_scores[i])) for i in bm_idxs if bm_scores[i] > 0]
    if trace:
        trace.t_bm25    = t_bm25
        trace.bm25_hits = len(pos_bm)
    logger.info(f"[{query_id}] BM25: {t_bm25:.1f}ms  hits: {len(pos_bm)}")
    for rank, (idx, score) in enumerate(pos_bm[:5], 1):
        logger.info(f"[{query_id}]   BM25 #{rank}: {score:.4f}  {META_LIST[idx].get('title','')[:50]}")

    t0    = _time.time()
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I  = index.search(q_vec, top_k_vec)
    t_faiss    = (_time.time() - t0) * 1000
    faiss_hits = sum(1 for x in I[0] if x != -1)
    if trace:
        trace.t_faiss    = t_faiss
        trace.faiss_hits = faiss_hits
    logger.info(f"[{query_id}] FAISS: {t_faiss:.1f}ms  hits: {faiss_hits}")
    for rank, idx in enumerate(I[0][:5]):
        if idx == -1: continue
        logger.info(f"[{query_id}]   FAISS #{rank+1}: {D[0][rank]:.4f}  {META_LIST[int(idx)].get('title','')[:50]}")

    bm25_rank  = {int(i): rank for rank, i in enumerate(bm_idxs)}
    faiss_rank = {int(I[0][r]): r for r in range(len(I[0])) if I[0][r] != -1}
    all_idxs   = set(bm25_rank) | set(faiss_rank)

    if qinfo["year_hint"]:
        all_idxs = {i for i in all_idxs if META_LIST[i].get("year") == qinfo["year_hint"]}
        logger.info(f"[{query_id}] Year filter {qinfo['year_hint']}: {len(all_idxs)} remain")
    if qinfo["dept_hint"]:
        all_idxs = {i for i in all_idxs
                    if qinfo["dept_hint"] in META_LIST[i].get("departments", [])}
        logger.info(f"[{query_id}] Dept filter '{qinfo['dept_hint']}': {len(all_idxs)} remain")

    if not all_idxs:
        logger.warning(f"[{query_id}] No candidates after filters")
        if trace: trace.t_total = (_time.time() - t_start) * 1000
        return []

    rrf = {
        idx: (1.0 / (RRF_K + bm25_rank.get(idx, top_k_bm25)) +
              1.0 / (RRF_K + faiss_rank.get(idx, top_k_vec)))
        for idx in all_idxs
    }
    sorted_idxs = sorted(rrf, key=rrf.get, reverse=True)
    if trace: trace.rrf_total = len(sorted_idxs)
    logger.info(f"[{query_id}] RRF combined: {len(sorted_idxs)} candidates")
    for rank, idx in enumerate(sorted_idxs[:5], 1):
        logger.info(f"[{query_id}]   RRF #{rank}: {rrf[idx]:.5f}  {META_LIST[idx].get('title','')[:50]}")

    cands = [{
        "idx": idx,
        "vec":      float(D[0][faiss_rank[idx]]) if idx in faiss_rank else 0.0,
        "bm25":     float(bm_scores[idx])        if idx < len(bm_scores) else 0.0,
        "combined": rrf[idx], "rerank": None,
        "meta":     META_LIST[idx], "text": TEXTS_LIST[idx],
        "_bm25_rank": bm25_rank.get(idx, 9999), "_faiss_rank": faiss_rank.get(idx, 9999),
    } for idx in sorted_idxs]

    t0    = _time.time()
    cands = _rerank(query, cands, reranker, max_rerank, query_id, trace)
    if trace:
        trace.t_rerank = (_time.time() - t0) * 1000
        trace.candidates = []
        for rank, c in enumerate(cands[:10], 1):
            m = c["meta"]
            trace.candidates.append(RetrievalCandidate(
                idx=c["idx"], title=m.get("title",""),
                mentor=m.get("mentor",""), year=m.get("year",""),
                department=m.get("department",""),
                bm25_score=c["bm25"],
                bm25_rank=c.get("_bm25_rank", 9999),
                faiss_score=c["vec"],
                faiss_rank=c.get("_faiss_rank", 9999),
                rrf_score=c["combined"],
                rerank_score=c.get("rerank", 0.0),
                final_rank=rank,
            ))

    if trace: trace.t_total = (_time.time() - t_start) * 1000
    return cands


def retrieve_housing_logged(query, embedder, H_index, H_EMB,
                            housing_chunks, query_id, top_k=4):
    logger.info(f"[{query_id}] == HOUSING RETRIEVAL ==")
    logger.info(f"[{query_id}] Query: {repr(query)}")
    if H_index is None:
        logger.error(f"[{query_id}] H_index is None")
        return []
    t0    = _time.time()
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I  = H_index.search(q_vec, top_k)
    logger.info(f"[{query_id}] Housing search: {(_time.time()-t0)*1000:.1f}ms")
    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1: continue
        chunk = housing_chunks[int(idx)]
        score = float(D[0][rank])
        logger.info(
            f"[{query_id}]   #{rank+1}: score={score:.4f}  "
            f"section={chunk.get('section','')[:45]}  page={chunk.get('page','')}"
        )
        results.append({"chunk": chunk, "score": score})
    return results
