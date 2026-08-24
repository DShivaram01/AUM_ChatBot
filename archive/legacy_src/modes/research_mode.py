"""
research_mode.py

COS Research Mode for your CS-BOT.

Responsibilities:
- Load cos_data.jsonl (COS research abstracts with metadata)
- Build / load Sentence-BERT embeddings + FAISS IP index
- Build BM25 over metadata (title, mentor, lead presenters, dept, category, year, keywords)
- Layered retrieval: BM25 -> FAISS fallback + CrossEncoder rerank
- Generate single-project summaries with Phi-2
- Switch between 'single' and 'list' mode based on rerank scores

Public API:
- init_research_mode(data_file, out_dir, rebuild_store=False, ...)
- research_query(query, max_rows=50) -> (mode, combined_text, table_rows)

Debug CLI:
    python -m src.modes.research_mode --rebuild

It will:
- auto-resolve default dataset + emb_store paths relative to this file
- init Research Mode
- open an interactive Q&A loop where each query prints either a summary or a list.
"""

from __future__ import annotations

import os
import json
import time
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple

from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================================
# GLOBALS
# =====================================================================

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_MODEL_ID_DEFAULT: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_ID_DEFAULT: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PHI_MODEL_ID_DEFAULT: str = "microsoft/phi-2"
PHI_MODEL_ID_SMALL: str = "microsoft/phi-1_5"

# Embedding / index paths (set by init_research_mode)
EMB_PATH: str = ""
ID_PATH: str = ""
META_PATH: str = ""
TEXTS_PATH: str = ""
FAISS_PATH: str = ""

# Data + models
ROWS: List[Dict[str, Any]] = []       # raw rows (id, text, metadata)
IDS: List[str] = []
TEXTS: List[str] = []
META: List[Dict[str, Any]] = []

EMB: Optional[np.ndarray] = None
index: Optional[faiss.Index] = None

bm25: Optional[BM25Okapi] = None

embedder: Optional[SentenceTransformer] = None
reranker: Optional[CrossEncoder] = None

phi_tok: Optional[AutoTokenizer] = None
phi_model: Optional[AutoModelForCausalLM] = None

# Name normalization / indexes (currently not used heavily but kept for future)
INDEX_LEAD_PRESENTERS = defaultdict(list)
INDEX_MENTOR = defaultdict(list)
INDEX_DEPARTMENT = defaultdict(list)
INDEX_CATEGORY = defaultdict(list)
INDEX_YEAR = defaultdict(list)
INDEX_KEYWORD = defaultdict(list)

ALL_LEAD_PRESENTERS = set()
ALL_MENTORS = set()
ALL_DEPARTMENTS = set()
ALL_CATEGORIES = set()
ALL_YEARS = set()
ALL_KEYWORDS = set()


# =====================================================================
# UTIL: LOAD JSONL
# =====================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows_local = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_local.append(json.loads(line))
    return rows_local


# =====================================================================
# NORMALIZERS
# =====================================================================

def norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"\b(dr\.?|prof\.?|professor)\b", "", name)
    name = name.replace(".", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# =====================================================================
# BUILD / LOAD STORE
# =====================================================================

def _build_store(data_file: str) -> None:
    """
    Build:
      - ROWS, IDS, TEXTS, META
      - embeddings (cosine-normalized SBERT)
      - FAISS IndexFlatIP + IDMap
      - Save everything to disk
    """
    global ROWS, IDS, TEXTS, META, EMB, index

    if embedder is None:
        raise RuntimeError("embedder must be initialized before _build_store().")

    print(f"\n📂 Loading COS research data from: {data_file}")
    ROWS = load_jsonl(data_file)
    print(f"Loaded {len(ROWS)} rows")

    IDS = [r["id"] for r in ROWS]
    TEXTS = [r["text"] for r in ROWS]
    META = [r["metadata"] for r in ROWS]

    print("\nSample row (keys):")
    if ROWS:
        print("id:", IDS[0])
        print("metadata keys:", list(META[0].keys())[:10])
        print("text first 200 chars:\n", TEXTS[0][:200], "...")
    else:
        print("No rows found in dataset!")

    # Embeddings
    print("\n[Build] Encoding texts for embeddings (cosine)...")
    EMB = embedder.encode(TEXTS, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    print("[Build] Building FAISS (IndexFlatIP + IDMap)...")
    base_index = faiss.IndexFlatIP(EMB.shape[1])
    id_map = faiss.IndexIDMap2(base_index)
    id_map.add_with_ids(EMB, np.arange(len(IDS)).astype(np.int64))
    index = id_map

    # Save
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(EMB_PATH, EMB)
    with open(ID_PATH, "w", encoding="utf-8") as f:
        json.dump(IDS, f)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(META, f, indent=2)
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(TEXTS, f)
    faiss.write_index(index, FAISS_PATH)
    print("✅ COS store built & saved:", index.ntotal, "vectors")


def _load_store() -> None:
    """
    Load embeddings / index / ids / metadata / texts from disk.
    """
    global ROWS, IDS, TEXTS, META, EMB, index

    print("\n📥 Loading COS emb_store from disk...")
    index = faiss.read_index(FAISS_PATH)
    EMB = np.load(EMB_PATH)
    with open(ID_PATH, "r", encoding="utf-8") as f:
        IDS_local = json.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        META_local = json.load(f)
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        TEXTS_local = json.load(f)

    IDS.clear()
    META.clear()
    TEXTS.clear()

    IDS.extend(IDS_local)
    META.extend(META_local)
    TEXTS.extend(TEXTS_local)

    ROWS.clear()
    for i, _id in enumerate(IDS):
        ROWS.append({"id": _id, "text": TEXTS[i], "metadata": META[i]})

    print(f"✅ Loaded COS store: {len(ROWS)} rows | index size = {index.ntotal}")


# =====================================================================
# BM25 + INDEXES
# =====================================================================

def build_bm25_corpus() -> BM25Okapi:
    """
    Build BM25 over *metadata only* so that name / year / dept queries
    hit exactly the structured fields and not random mentions in text.
    """
    corpus_tokens: List[List[str]] = []

    for meta, full_text in zip(META, TEXTS):
        title   = meta.get("title", "") or ""
        mentor  = meta.get("mentor", "") or ""
        leads   = " ".join(meta.get("lead_presenters", []) or [])
        dept    = meta.get("department", "") or ""
        cat     = meta.get("category", "") or ""
        year    = str(meta.get("year", "") or "")
        keys    = " ".join(meta.get("keywords", []) or [])

        combined = " ".join([title, mentor, leads, dept, cat, year, keys])
        tokens = combined.lower().split()
        corpus_tokens.append(tokens)

    return BM25Okapi(corpus_tokens)


def _build_meta_indexes() -> None:
    """
    Build various helper indexes & canonical sets (lead presenters, mentor,
    department, category, year, keywords).
    """
    for idx, meta in enumerate(META):
        # lead_presenters
        for lp in meta.get("lead_presenters", []) or []:
            lp = lp.strip()
            if not lp:
                continue
            ALL_LEAD_PRESENTERS.add(lp)
            lp_norm = norm_name(lp)
            if lp_norm:
                INDEX_LEAD_PRESENTERS[lp_norm].append(idx)

        # mentor
        mentor = (meta.get("mentor") or "").strip()
        if mentor:
            ALL_MENTORS.add(mentor)
            m_norm = norm_name(mentor)
            if m_norm:
                INDEX_MENTOR[m_norm].append(idx)

        # dept
        dept = (meta.get("department") or "").strip()
        if dept:
            ALL_DEPARTMENTS.add(dept)
            d_norm = norm_text(dept)
            if d_norm:
                INDEX_DEPARTMENT[d_norm].append(idx)

        # category
        cat = (meta.get("category") or "").strip()
        if cat:
            ALL_CATEGORIES.add(cat)
            c_norm = norm_text(cat)
            if c_norm:
                INDEX_CATEGORY[c_norm].append(idx)

        # year
        year = meta.get("year")
        if year is not None:
            year_str = str(year).strip()
            if year_str:
                ALL_YEARS.add(year_str)
                INDEX_YEAR[year_str].append(idx)

        # keywords
        for kw in meta.get("keywords", []) or []:
            kw = kw.strip()
            if not kw:
                continue
            ALL_KEYWORDS.add(kw)
            kw_norm = norm_text(kw)
            if kw_norm:
                INDEX_KEYWORD[kw_norm].append(idx)


# =====================================================================
# INITIALIZATION
# =====================================================================

def init_research_mode(
    data_file: str,
    out_dir: str,
    rebuild_store: bool = False,
    embed_model_id: str = EMBED_MODEL_ID_DEFAULT,
    reranker_id: str = RERANKER_ID_DEFAULT,
    use_phi15: bool = False,
    phi_model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    Initialize Research Mode.

    Args:
        data_file:     Path to cos_data.jsonl
        out_dir:       Directory to store embeddings/index/metadata (emb_store)
        rebuild_store: If True, always rebuild emb_store from JSONL
        embed_model_id: SentenceTransformer model ID
        reranker_id:   CrossEncoder model ID
        use_phi15:     Use phi-1_5 instead of phi-2
        phi_model_id:  Manual override for Phi model ID
        device:        "cuda" or "cpu". If None, auto-detect
    """
    global DEVICE, EMB_PATH, ID_PATH, META_PATH, TEXTS_PATH, FAISS_PATH
    global embedder, bm25, reranker, phi_tok, phi_model

    if device is not None:
        DEVICE = device

    os.makedirs(out_dir, exist_ok=True)
    EMB_PATH   = os.path.join(out_dir, "embeddings.npy")
    ID_PATH    = os.path.join(out_dir, "ids.json")
    META_PATH  = os.path.join(out_dir, "metadata.json")
    TEXTS_PATH = os.path.join(out_dir, "texts.json")
    FAISS_PATH = os.path.join(out_dir, "faiss_ip.index")

    # Sentence-BERT
    print(f"\n🧠 Loading SentenceTransformer for Research Mode: {embed_model_id} (device={DEVICE})")
    embedder = SentenceTransformer(embed_model_id, device=DEVICE)

    # Build or load store
    need_build = rebuild_store or not (
        os.path.exists(FAISS_PATH)
        and os.path.exists(EMB_PATH)
        and os.path.exists(ID_PATH)
        and os.path.exists(META_PATH)
        and os.path.exists(TEXTS_PATH)
    )

    if need_build:
        _build_store(data_file)
    else:
        _load_store()

    # BM25 + meta indexes
    print("\n[Build] BM25 corpus...")
    bm25 = build_bm25_corpus()
    print("[Build] BM25 ready.")

    print("\n[Build] Meta indexes...")
    _build_meta_indexes()
    print("[Build] Meta indexes ready.")

    # Reranker
    print(f"\n🧪 Loading CrossEncoder reranker: {reranker_id}")
    reranker = CrossEncoder(reranker_id, device=DEVICE)

    # Phi model
    if phi_model_id is not None:
        chosen_phi_id = phi_model_id
    else:
        chosen_phi_id = PHI_MODEL_ID_SMALL if use_phi15 else PHI_MODEL_ID_DEFAULT

    print(f"\n🧪 Loading Phi model for Research Mode: {chosen_phi_id}")
    phi_tok = AutoTokenizer.from_pretrained(chosen_phi_id)
    phi_model = AutoModelForCausalLM.from_pretrained(
        chosen_phi_id,
        device_map="auto",
        torch_dtype="auto",
    )

    # Warmup
    _ = phi_model.generate(
        **phi_tok("hi", return_tensors="pt").to(phi_model.device),
        max_new_tokens=1,
    )
    print("✅ Research Mode initialized (emb_store + BM25 + reranker + Phi ready).")


# =====================================================================
# RETRIEVAL + SUMMARY HELPERS
# =====================================================================

def bm25_search(query: str, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    BM25 keyword search over metadata-only corpus.
    Returns:
        top_idxs: indices of top docs
        scores:   BM25 scores for ALL docs
    """
    if bm25 is None:
        raise RuntimeError("BM25 not initialized. Call init_research_mode() first.")

    q_tokens = query.lower().split()
    scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)

    if top_k >= len(scores):
        top_idxs = np.argsort(scores)[::-1]
    else:
        top_idxs = np.argsort(scores)[::-1][:top_k]

    return top_idxs, scores


def extract_abstract(full_text: str) -> str:
    parts = full_text.split("Abstract:", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return full_text.strip()


META_MARKERS = (
    "Title:",
    "Year:",
    "Lead Presenters:",
    "Other Authors:",
    "Mentor:",
    "Department:",
    "Category:",
    "Keywords:",
    "Abstract:",
)


def clean_summary(text: str) -> str:
    if "Summary:" in text:
        text = text.split("Summary:", 1)[1]

    for m in META_MARKERS:
        if m in text:
            text = text.split(m, 1)[0]

    text = text.strip().split("\n\n", 1)[0].strip()

    sents = re.split(r'(?<=[.!?])\s+', text)
    seen, dedup = set(), []
    for s in sents:
        ss = s.strip()
        if ss and ss.lower() not in seen:
            seen.add(ss.lower())
            dedup.append(ss)
    return " ".join(dedup).strip()


def build_summary_prompt(meta: dict, abstract: str) -> str:
    title   = meta.get("title", "")
    leads   = ", ".join(meta.get("lead_presenters", []))
    others  = ", ".join(meta.get("other_authors", []))
    mentor  = meta.get("mentor", "")
    dept    = meta.get("department", "")
    cat     = meta.get("category", "")
    year    = meta.get("year", "XXXX")
    keys    = ", ".join(meta.get("keywords", []))

    return f"""Task: Write a concise research summary.

Rules:
- Output exactly ONE paragraph of 5–6 sentences.
- Do NOT include headings or bullet points.
- Do NOT repeat the input or metadata.
- Use only the given facts; no speculation.

Context (for reference only):
Title: {title}
Year: {year}
Lead Presenters: {leads}
Other Authors: {others}
Mentor: {mentor}
Department: {dept}
Category: {cat}
Keywords: {keys}


Summary:"""


def format_meta_details(meta: dict) -> str:
    title  = meta.get("title", "")
    leads  = ", ".join(meta.get("lead_presenters", []))
    others = ", ".join(meta.get("other_authors", []))
    mentor = meta.get("mentor", "")
    dept   = meta.get("department", "")
    cat    = meta.get("category", "")
    year   = meta.get("year", "XXXX")
    keys   = ", ".join(meta.get("keywords", []))

    lines = [
        f"Title: {title}",
        f"Lead Presenters: {leads}" if leads else "Lead Presenters: -",
        f"Other Authors: {others}" if others else "Other Authors: -",
        f"Mentor: {mentor}" if mentor else "Mentor: -",
        f"Department: {dept}" if dept else "Department: -",
        f"Category: {cat}" if cat else "Category: -",
        f"Year: {year}",
        f"Keywords: {keys}" if keys else "Keywords: -",
    ]
    return "\n".join(lines)


def summarize_best(best: Dict[str, Any], max_new_tokens: int = 140) -> Tuple[str, str]:
    if phi_model is None or phi_tok is None:
        raise RuntimeError("Phi model not initialized. Call init_research_mode() first.")

    abstract = extract_abstract(best["text"])
    prompt = build_summary_prompt(best["meta"], abstract)

    inputs = phi_tok(prompt, return_tensors="pt", truncation=True).to(phi_model.device)
    out = phi_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        no_repeat_ngram_size=4,
        repetition_penalty=1.12,
        eos_token_id=phi_tok.eos_token_id,
    )
    raw = phi_tok.decode(out[0], skip_special_tokens=True)
    summary = clean_summary(raw)
    details = format_meta_details(best["meta"])
    return summary, details


def retrieve_layered(
    query: str,
    top_k_bm25: int = 50,
    top_k_vec: int = 30,
    max_for_rerank: int = 30,
) -> List[Dict[str, Any]]:
    """
    Layered retrieve:
    1) BM25 over metadata-only corpus.
    2) If BM25 produces candidates → rerank them with CrossEncoder (no FAISS).
    3) If BM25 is empty → fallback to FAISS semantic search + rerank.
    """
    if embedder is None or index is None or reranker is None:
        raise RuntimeError("Research Mode not initialized. Call init_research_mode() first.")

    print(f"\n🧠 Query (layered): {query}")

    # 1) BM25
    t_bm0 = time.time()
    bm25_idxs, bm25_scores = bm25_search(query, top_k=top_k_bm25)
    print(f"BM25 time: {(time.time()-t_bm0)*1000:.1f} ms")

    bm25_candidates: Dict[int, float] = {}
    for idx_ in bm25_idxs:
        score = float(bm25_scores[idx_])
        if score > 0.0:
            bm25_candidates[int(idx_)] = score

    cands: List[Dict[str, Any]] = []

    if bm25_candidates:
        max_bm = max(bm25_candidates.values()) if bm25_candidates else 1.0

        for idx_, score in bm25_candidates.items():
            meta = META[idx_]
            text = TEXTS[idx_]
            cands.append(
                {
                    "idx": int(idx_),
                    "vec": 0.0,
                    "bm25": float(score),
                    "combined": float(score / max_bm),
                    "rerank": None,
                    "meta": meta,
                    "text": text,
                }
            )

        cands.sort(key=lambda x: x["bm25"], reverse=True)

        print("\n📥 Candidates from BM25 (before rerank):")
        for i, c in enumerate(cands[:5], 1):
            print(f"{i}. bm25={c['bm25']:.3f}")
            print("   Title:", c["meta"].get("title", "NO TITLE"))
            print("   Mentor:", c["meta"].get("mentor", ""))
            print("   Year:", c["meta"].get("year", ""))

    else:
        # 2) FAISS fallback
        print("BM25 produced no candidates. Falling back to FAISS.")
        t0 = time.time()
        q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = index.search(q, top_k_vec)
        print(f"FAISS time: {(time.time()-t0)*1000:.1f} ms")

        vec_candidates: Dict[int, float] = {}
        for rank, idx_ in enumerate(I[0]):
            if idx_ == -1:
                continue
            vec_candidates[int(idx_)] = float(D[0][rank])

        if not vec_candidates:
            print("No candidates from FAISS either.")
            return []

        max_vec = max(vec_candidates.values()) if vec_candidates else 1.0

        for idx_, vscore in vec_candidates.items():
            meta = META[idx_]
            text = TEXTS[idx_]
            cands.append(
                {
                    "idx": int(idx_),
                    "vec": float(vscore),
                    "bm25": 0.0,
                    "combined": float(vscore / max_vec),
                    "rerank": None,
                    "meta": meta,
                    "text": text,
                }
            )

        print("\n📥 Candidates from FAISS (before rerank):")
        for i, c in enumerate(cands[:5], 1):
            print(f"{i}. vec={c['vec']:.3f}")
            print("   Title:", c["meta"].get("title", "NO TITLE"))
            print("   Mentor:", c["meta"].get("mentor", ""))
            print("   Year:", c["meta"].get("year", ""))

    # 3) CrossEncoder rerank
    keep = min(len(cands), max_for_rerank)
    if keep > 0:
        t2 = time.time()
        pairs = [(query, cands[i]["text"]) for i in range(keep)]
        scores = reranker.predict(pairs)
        for i, s in enumerate(scores):
            cands[i]["rerank"] = float(s)
        cands[:keep] = sorted(cands[:keep], key=lambda x: x["rerank"], reverse=True)
        print(f"\n🔁 Rerank time: {(time.time()-t2)*1000:.1f} ms")
        print("\n📦 Top (after rerank):")
        for i, c in enumerate(cands[:5], 1):
            print(f"{i}. rerank={c['rerank']:.3f}")
            print("   Title:", c["meta"].get("title", "NO TITLE"))
            print("   Mentor:", c["meta"].get("mentor", ""))
            print("   Year:", c["meta"].get("year", ""))

    return cands


def infer_mode_from_rerank_candidates(cands: List[Dict[str, Any]]) -> str:
    """
    Decide 'single' vs 'list' based on CrossEncoder rerank scores.

    Logic:
    - Use the rerank scores.
    - If no rerank scores -> 'single'
    - If exactly one -> 'single'
    - Otherwise, compare top1 vs top2 ratio.
    """
    scores = sorted(
        [c["rerank"] for c in cands if c.get("rerank") is not None],
        reverse=True,
    )

    if not scores:
        return "single"
    if len(scores) == 1:
        return "single"

    top1, top2 = scores[0], scores[1]

    if top1 <= 0:
        return "list"
    if top2 <= 0:
        return "single"

    ratio = top1 / top2
    if ratio >= 1.5:
        return "single"
    return "list"


# =====================================================================
# PUBLIC QUERY FUNCTION
# =====================================================================

def research_query(
    query: str,
    max_rows: int = 50,
) -> Tuple[str, str, List[List[Any]]]:
    """
    High-level handler similar to run_query in the notebook.

    Returns:
        mode: "single" or "list"
        combined_text: summary + details (for single) or "" (for list)
        table_rows: list of [rank, title, year, mentor, department, rerank_score]
    """
    if index is None or bm25 is None or embedder is None or reranker is None:
        raise RuntimeError("Research Mode not initialized. Call init_research_mode() first.")

    topk = len(IDS)
    cands = retrieve_layered(
        query,
        top_k_bm25=min(len(IDS), 500),
        top_k_vec=int(topk),
        max_for_rerank=min(len(IDS), 50),
    )
    if not cands:
        return "none", "No results.", []

    mode = infer_mode_from_rerank_candidates(cands)
    print(f"Mode: {mode}")

    # table rows
    rows = []
    for i, c in enumerate(cands[:max_rows], 1):
        m = c["meta"]
        rows.append(
            [
                i,
                m.get("title", ""),
                m.get("year", ""),
                m.get("mentor", ""),
                m.get("department", ""),
                round(c.get("rerank", 0.0), 4),
            ]
        )

    if mode == "list":
        combined = ""
        return mode, combined, rows
    else:
        best = cands[0]
        summary, details = summarize_best(best)
        combined = f"{summary}\n\n---\n{details}"
        return mode, combined, rows


# =====================================================================
# DEBUG CLI
# =====================================================================

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug CLI for Research Mode"
    )

    # research_mode.py is in src/modes/
    here = Path(__file__).resolve().parent   # .../src/modes

    # Walk up until we find a parent containing "datasets"
    base = here
    while base != base.parent and not (base / "datasets").exists():
        base = base.parent

    datasets_root = base / "datasets"
    default_data = datasets_root / "COS_research_data" / "cos_data.jsonl"
    default_out  = datasets_root / "COS_research_data" / "emb_store"

    parser.add_argument(
        "--data-file",
        type=str,
        default=str(default_data),
        help=f"Path to cos_data.jsonl (default: {default_data})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out),
        help=f"Directory for COS emb_store (default: {default_out})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of embeddings / FAISS even if files exist.",
    )
    parser.add_argument(
        "--phi15",
        action="store_true",
        help="Use microsoft/phi-1_5 instead of phi-2.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device (cuda/cpu). If omitted, auto-detect.",
    )
    return parser.parse_args()


def _run_cli() -> None:
    args = _parse_cli_args()

    print("\n=== Research Mode CLI ===")
    print(f"Data file : {args.data_file}")
    print(f"Out dir   : {args.out_dir}")
    print(f"Rebuild   : {args.rebuild}")
    print(f"Use phi-1_5: {args.phi15}")
    print(f"Device    : {args.device or DEVICE}")
    print("========================\n")

    init_research_mode(
        data_file=args.data_file,
        out_dir=args.out_dir,
        rebuild_store=args.rebuild,
        use_phi15=args.phi15,
        device=args.device,
    )

    print("Type your COS research queries. Type 'exit' or 'quit' to stop.\n")

    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Bye 👋")
                break

            mode, combined, rows = research_query(q, max_rows=10)

            if mode == "none":
                print("\nBot: No results.\n")
                continue

            if mode == "single":
                print("\nBot (single mode):")
                print(combined)
                print("\nTop hits:")
                for r in rows[:5]:
                    print(f"{r[0]}. {r[1]} ({r[2]}) | Mentor: {r[3]} | Dept: {r[4]} | Rerank={r[5]}")
                print()
            else:
                print("\nBot (list mode): showing top hits:")
                for r in rows[:10]:
                    print(f"{r[0]}. {r[1]} ({r[2]}) | Mentor: {r[3]} | Dept: {r[4]} | Rerank={r[5]}")
                print()

    except KeyboardInterrupt:
        print("\nInterrupted. Bye 👋")

# =====================================================================
# Gradio / main_app-friendly chat handler
# =====================================================================

def research_chat_handler(
    query: str,
    history=None,
    memory=None,
):
    """
    Thin wrapper so main_app.py can treat research mode as a chat handler.

    Expected behavior:
    - history: list of {"role": "...", "content": "..."} messages
    - memory:  (currently unused, but kept for API symmetry)
    - Returns: (updated_history, memory)
    """
    if history is None:
        history = []
    if memory is None:
        memory = []

    # Use our high-level research_query()
    mode, combined, rows = research_query(query, max_rows=10)

    if mode == "none":
        answer = "No results found for your query."
    elif mode == "single":
        # combined already has: summary + --- + meta details
        # Optionally append a tiny top-hits view
        top_lines = []
        for r in rows[:5]:
            rank, title, year, mentor, dept, score = r
            top_lines.append(
                f"{rank}. {title} ({year}) | Mentor: {mentor} | Dept: {dept} | Score: {score}"
            )
        hits_block = "\n\nTop relevant projects:\n" + "\n".join(top_lines) if top_lines else ""
        answer = combined + hits_block
    else:  # list mode
        # Build a list-style answer summarizing the top hits
        lines = ["Here are the most relevant research projects I found:\n"]
        for r in rows[:10]:
            rank, title, year, mentor, dept, score = r
            lines.append(
                f"{rank}. {title} ({year})\n   Mentor: {mentor} | Dept: {dept} | Score: {score}"
            )
        answer = "\n".join(lines)

    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    # research mode doesn't use memory yet, so we just pass it through unchanged
    return history, memory

if __name__ == "__main__":
    _run_cli()
