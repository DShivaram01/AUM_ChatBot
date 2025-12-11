"""
research_mode.py

Modular version of research_2.0_frozen.ipynb.

Provides:
- init_research_mode(...)       # load COS JSONL, build/load FAISS + BM25 + models
- research_chat_handler(...)    # main handler for COS research Q&A (chat-style)
- retrieve_layered(...)         # core layered retriever (BM25 -> FAISS -> rerank)
- summarize_best(...)           # Phi-2 research summary for best hit

Intended usage from main_app.py:

    from research_mode import init_research_mode, research_chat_handler

    init_research_mode(
        data_file="COS_dataset/cos_data.jsonl",
        out_dir="COS_dataset/emb_store",
    )

    history, memory = [], []
    history, memory = research_chat_handler("projects about protein homology", history, memory)
"""

from __future__ import annotations

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from collections import defaultdict


# =====================================================================
# GLOBALS (initialized by init_research_mode)
# =====================================================================

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PHI_MODEL_ID: str = "microsoft/phi-2"

embedder: Optional[SentenceTransformer] = None
reranker: Optional[CrossEncoder] = None
phi_tok: Optional[AutoTokenizer] = None
phi_model: Optional[AutoModelForCausalLM] = None

# COS data
ROWS: List[Dict[str, Any]] = []
IDS: List[Any] = []
TEXTS: List[str] = []
META: List[Dict[str, Any]] = []

# Store files
EMB_PATH: str = ""
ID_PATH: str = ""
META_PATH: str = ""
TEXTS_PATH: str = ""
FAISS_PATH: str = ""

# Loaded store
index: Optional[faiss.Index] = None
EMB: Optional[np.ndarray] = None
ID_LIST: List[Any] = []
META_LIST: List[Dict[str, Any]] = []
TEXTS_LIST: List[str] = []

# BM25 & name/field indexes
bm25: Optional[BM25Okapi] = None

INDEX_LEAD_PRESENTERS = defaultdict(list)   # norm_name(lead_presenter) -> [row_idx,...]
INDEX_MENTOR          = defaultdict(list)   # norm_name(mentor)        -> [row_idx,...]
INDEX_DEPARTMENT      = defaultdict(list)   # norm_text(dept)          -> [row_idx,...]
INDEX_CATEGORY        = defaultdict(list)   # norm_text(category)      -> [row_idx,...]
INDEX_YEAR            = defaultdict(list)   # "2024"                   -> [row_idx,...]
INDEX_KEYWORD         = defaultdict(list)   # norm_text(keyword)       -> [row_idx,...]

ALL_LEAD_PRESENTERS = set()
ALL_MENTORS         = set()
ALL_DEPARTMENTS     = set()
ALL_CATEGORIES      = set()
ALL_YEARS           = set()
ALL_KEYWORDS        = set()

# Simple chat memory (not heavily used here but kept for consistency)
MAX_MEMORY = 5


# =====================================================================
# JSONL LOADER
# =====================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# =====================================================================
# NORMALIZERS
# =====================================================================

def norm_text(s: str) -> str:
    """Generic cleaner for text fields (dept, category, keywords)."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_name(name: str) -> str:
    """Name-specific cleaner (presenters, mentor)."""
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"\b(dr\.?|prof\.?|professor)\b", "", name)
    name = name.replace(".", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# =====================================================================
# STORE BUILD / LOAD
# =====================================================================

def _build_store() -> None:
    """
    Build embeddings + FAISS ip index from global TEXTS / IDS
    using global paths defined in init_research_mode.
    """
    global EMB, index

    if embedder is None:
        raise RuntimeError("Embedder not initialized before _build_store().")

    print("\n[Build] Encoding texts for embeddings (cosine)...")
    EMB = embedder.encode(
        TEXTS,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("[Build] Building FAISS (IndexFlatIP + IDMap)...")
    faiss_index = faiss.IndexFlatIP(EMB.shape[1])
    faiss_index = faiss.IndexIDMap2(faiss_index)
    faiss_index.add_with_ids(EMB, np.arange(len(IDS)).astype(np.int64))

    np.save(EMB_PATH, EMB)
    with open(ID_PATH, "w", encoding="utf-8") as f:
        json.dump(IDS, f)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(META, f, indent=2)
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(TEXTS, f)

    faiss.write_index(faiss_index, FAISS_PATH)
    index = faiss_index
    print("✅ Store built & saved:", index.ntotal, "vectors")


def _load_store() -> None:
    """
    Load FAISS + embeddings + meta from disk into globals.
    """
    global index, EMB, ID_LIST, META_LIST, TEXTS_LIST

    index = faiss.read_index(FAISS_PATH)
    EMB = np.load(EMB_PATH)

    with open(ID_PATH, "r", encoding="utf-8") as f:
        ID_LIST = json.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        META_LIST = json.load(f)
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        TEXTS_LIST = json.load(f)

    print(f"Index size: {index.ntotal} | Device: {DEVICE}")


# =====================================================================
# BM25 OVER METADATA
# =====================================================================

def build_bm25_corpus() -> BM25Okapi:
    """
    Build BM25 over metadata-only corpus.
    We intentionally drop the full text here to keep BM25 aligned to structured fields:
    title, mentor, leads, dept, category, year, keywords.
    """
    corpus_tokens: List[List[str]] = []

    for meta, full_text in zip(META_LIST, TEXTS_LIST):
        title   = meta.get("title", "") or ""
        mentor  = meta.get("mentor", "") or ""
        leads   = " ".join(meta.get("lead_presenters", []) or [])
        dept    = meta.get("department", "") or ""
        cat     = meta.get("category", "") or ""
        year    = str(meta.get("year", "") or "")
        keys    = " ".join(meta.get("keywords", []) or [])

        combined = " ".join(
            [
                title,
                mentor,
                leads,
                dept,
                cat,
                year,
                keys,
            ]
        )

        tokens = combined.lower().split()
        corpus_tokens.append(tokens)

    return BM25Okapi(corpus_tokens)


def _build_field_indexes() -> None:
    """
    Populate name/field indexes and canonical sets from META_LIST
    for possible future structured querying (by mentor, department, etc.).
    """
    for idx, meta in enumerate(META_LIST):
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

        # department
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
    embed_model_id: str = EMBED_MODEL_ID,
    reranker_id: str = RERANKER_ID,
    phi_model_id: str = PHI_MODEL_ID,
    device: Optional[str] = None,
    rebuild_store: bool = False,
) -> None:
    """
    Initialize COS research mode from a JSONL file.

    Args:
        data_file:      Path to cos_data.jsonl (id, text, metadata)
        out_dir:        Directory to store/load embeddings/index files
        embed_model_id: SentenceTransformer model ID
        reranker_id:    CrossEncoder model ID
        phi_model_id:   Phi model ID (default microsoft/phi-2)
        device:         "cuda" or "cpu" (None -> auto)
        rebuild_store:  If True, force rebuild of FAISS/embeddings from JSONL
    """
    global DEVICE, EMBED_MODEL_ID, RERANKER_ID, PHI_MODEL_ID
    global EMB_PATH, ID_PATH, META_PATH, TEXTS_PATH, FAISS_PATH
    global embedder, reranker, phi_tok, phi_model
    global ROWS, IDS, TEXTS, META, bm25

    if device is not None:
        DEVICE = device

    EMBED_MODEL_ID = embed_model_id
    RERANKER_ID = reranker_id
    PHI_MODEL_ID = phi_model_id

    os.makedirs(out_dir, exist_ok=True)
    EMB_PATH   = os.path.join(out_dir, "embeddings.npy")
    ID_PATH    = os.path.join(out_dir, "ids.json")
    META_PATH  = os.path.join(out_dir, "metadata.json")
    TEXTS_PATH = os.path.join(out_dir, "texts.json")
    FAISS_PATH = os.path.join(out_dir, "faiss_ip.index")

    # 1) Load JSONL
    print(f"📂 Loading COS data from: {data_file}")
    ROWS = load_jsonl(data_file)
    print(f"Loaded {len(ROWS)} rows.")

    IDS   = [r["id"] for r in ROWS]
    TEXTS = [r["text"] for r in ROWS]
    META  = [r["metadata"] for r in ROWS]

    if ROWS:
        print("\nSample row (keys):")
        print("id:", IDS[0])
        print("metadata keys:", list(META[0].keys())[:10])
        print("text first 200 chars:\n", TEXTS[0][:200], "...")

    # 2) Load embedder
    print(f"\n🧠 Loading SentenceTransformer: {EMBED_MODEL_ID} (device={DEVICE})")
    embedder = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)

    # 3) Build or load store
    need_build = rebuild_store or not (
        os.path.exists(FAISS_PATH)
        and os.path.exists(EMB_PATH)
        and os.path.exists(ID_PATH)
        and os.path.exists(META_PATH)
        and os.path.exists(TEXTS_PATH)
    )
    if need_build:
        _build_store()
    else:
        _load_store()

    # 4) Build BM25 and field indexes
    print("\n[Build] BM25 corpus...")
    global bm25
    bm25 = build_bm25_corpus()
    print("[Build] BM25 ready.")

    print("[Build] Field indexes (names, dept, year, keywords)...")
    _build_field_indexes()

    # 5) Load reranker (CrossEncoder)
    print(f"\n🧪 Loading CrossEncoder reranker: {RERANKER_ID}")
    reranker = CrossEncoder(RERANKER_ID, device=DEVICE)

    # 6) Load Phi model
    print(f"🧪 Loading Phi model: {PHI_MODEL_ID}")
    phi_tok = AutoTokenizer.from_pretrained(PHI_MODEL_ID)
    phi_model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL_ID,
        device_map="auto",
        torch_dtype=(torch.float16 if DEVICE == "cuda" else torch.float32),
    )

    # Warmup
    _ = phi_model.generate(
        **phi_tok("hi", return_tensors="pt").to(phi_model.device),
        max_new_tokens=1,
    )
    print("✅ Research mode models ready.")


# =====================================================================
# RETRIEVAL & RERANK
# =====================================================================

def bm25_search(query: str, top_k: int = 50):
    """
    BM25 keyword search over the metadata corpus.
    Returns:
        top_idxs: numpy array of indices (int)
        scores:   numpy array of BM25 scores for all docs
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
    """
    Split on the first 'Abstract:' and take the rest.
    If not found, return full_text stripped.
    """
    parts = full_text.split("Abstract:", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return full_text.strip()


def print_hit(rank: int, meta: Dict[str, Any], vec: float, rr: Optional[float]):
    title = meta.get("title", "")
    year  = meta.get("year", "")
    dept  = meta.get("department", "")
    cats  = meta.get("category", "")
    print(f"\n--- Hit {rank} ---")
    print(f"Title: {title}")
    print(f"Year: {year} | Dept: {dept} | Category: {cats}")
    print(f"VecScore: {vec:.4f} | Rerank: {0.0 if rr is None else float(rr):.4f}")


def retrieve_layered(
    query: str,
    top_k_bm25: int = 50,
    top_k_vec: int = 30,
    max_for_rerank: int = 30,
) -> List[Dict[str, Any]]:
    """
    Layered retrieve:
    1) Try BM25 over metadata-only corpus.
    2) If BM25 produces candidates -> rerank them with CrossEncoder (no FAISS).
    3) If BM25 is empty -> fallback to FAISS semantic search + rerank.
    Returns:
        List of candidate dicts with keys:
        { idx, vec, bm25, combined, rerank, meta, text }
    """
    if index is None or embedder is None or reranker is None:
        raise RuntimeError("Research mode not initialized. Call init_research_mode() first.")

    print(f"\n🧠 Query (layered): {query}")

    # 1) BM25 keyword search (metadata)
    t_bm0 = time.time()
    bm25_idxs, bm25_scores = bm25_search(query, top_k=top_k_bm25)
    print(f"BM25 time: {(time.time() - t_bm0) * 1000:.1f} ms")

    bm25_candidates: Dict[int, float] = {}
    for idx_ in bm25_idxs:
        score = float(bm25_scores[idx_])
        if score > 0.0:
            bm25_candidates[int(idx_)] = score

    cands: List[Dict[str, Any]] = []

    if bm25_candidates:
        max_bm = max(bm25_candidates.values()) if bm25_candidates else 1.0
        for idx_, score in bm25_candidates.items():
            meta = META_LIST[idx_]
            text = TEXTS_LIST[idx_]
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
        # 2) Fallback: FAISS semantic search
        print("BM25 produced no candidates. Falling back to FAISS.")
        t0 = time.time()
        q_emb = embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        D, I = index.search(q_emb, top_k_vec)
        print(f"FAISS time: {(time.time() - t0) * 1000:.1f} ms")

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
            meta = META_LIST[idx_]
            text = TEXTS_LIST[idx_]
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
        print(f"\n🔁 Rerank time: {(time.time() - t2) * 1000:.1f} ms")

        print("\n📦 Top (after rerank):")
        for i, c in enumerate(cands[:5], 1):
            print(f"{i}. rerank={c['rerank']:.3f}")
            print("   Title:", c["meta"].get("title", "NO TITLE"))
            print("   Mentor:", c["meta"].get("mentor", ""))
            print("   Year:", c["meta"].get("year", ""))

    return cands


# =====================================================================
# SUMMARY HELPERS
# =====================================================================

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


def clean_summary(text: str) -> str:
    # Keep only content after the "Summary:" anchor (if present)
    if "Summary:" in text:
        text = text.split("Summary:", 1)[1]

    # Stop before any metadata markers if the model starts echoing them
    for m in META_MARKERS:
        if m in text:
            text = text.split(m, 1)[0]

    # Keep only the first paragraph
    text = text.strip().split("\n\n", 1)[0].strip()

    # Remove duplicate sentences while preserving order
    sents = re.split(r"(?<=[.!?])\s+", text)
    seen, dedup = set(), []
    for s in sents:
        ss = s.strip()
        if ss and ss.lower() not in seen:
            seen.add(ss.lower())
            dedup.append(ss)
    return " ".join(dedup).strip()


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
    """
    Summarize the best candidate's abstract with Phi-2.
    Returns:
        summary, meta_details
    """
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


# =====================================================================
# MODE INFERENCE: SINGLE vs LIST
# =====================================================================

def infer_mode_from_rerank_candidates(cands: List[Dict[str, Any]]) -> str:
    """
    Decide 'single' vs 'list' based on CrossEncoder rerank scores.

    Logic:
    - Use the rerank scores (more semantic, see full text).
    - If we have no rerank scores → 'single'
    - If exactly one candidate has a rerank score → 'single'.
    - Otherwise, look at how sharply the top score stands out:
        - If top1 is clearly stronger than top2 (ratio), treat as 'single'.
        - Else, treat as 'list'.
    """

    scores = sorted(
        [c["rerank"] for c in cands if c.get("rerank") is not None],
        reverse=True,
    )

    if not scores:
        return "single"

    if len(scores) == 1:
        return "single"

    top1 = scores[0]
    top2 = scores[1]

    if top1 <= 0:
        return "list"
    if top2 <= 0:
        return "single"

    ratio = top1 / top2
    if ratio >= 1.5:
        return "single"
    return "list"


# =====================================================================
# SIMPLE MEMORY
# =====================================================================

def update_memory(
    memory: List[Dict[str, str]],
    query: str,
    answer: str,
    max_length: int = MAX_MEMORY,
) -> List[Dict[str, str]]:
    memory.append({"user": query, "assistant": answer})
    return memory[-max_length:]


# =====================================================================
# CHAT HANDLER (FOR MAIN ROUTER)
# =====================================================================

def research_chat_handler(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    memory: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Main COS research Q&A handler.
    - Uses layered retrieval (BM25 + FAISS + CrossEncoder)
    - Infers single vs list mode
    - For single: returns Phi-2 summary + metadata
    - For list: returns a markdown list of top hits

    Returns:
        updated_history, updated_memory
    """
    if history is None:
        history = []
    if memory is None:
        memory = []

    if index is None:
        raise RuntimeError("Research mode not initialized. Call init_research_mode() first.")

    topk = len(IDS) if IDS else 50
    cands = retrieve_layered(
        query,
        top_k_bm25=min(len(IDS), 500) if IDS else 500,
        top_k_vec=int(topk),
        max_for_rerank=min(len(IDS), 50) if IDS else 50,
    )

    if not cands:
        answer = "I couldn't find any COS projects matching that query."
        history += [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        memory = update_memory(memory, query, answer)
        return history, memory

    mode = infer_mode_from_rerank_candidates(cands)
    print(f"Mode: {mode}")

    if mode == "single":
        best = cands[0]
        summary, details = summarize_best(best)
        answer = f"{summary}\n\n---\n{details}"
    else:
        # Build a markdown list of results
        lines: List[str] = []
        max_rows = 10
        for i, c in enumerate(cands[:max_rows], 1):
            m = c["meta"]
            title = m.get("title", "")
            year = m.get("year", "")
            mentor = m.get("mentor", "")
            dept = m.get("department", "")
            score = c.get("rerank", 0.0)
            lines.append(
                f"**{i}. {title}**\n"
                f"- Year: {year}\n"
                f"- Mentor: {mentor}\n"
                f"- Department: {dept}\n"
                f"- Relevance Score: {score:.3f}"
            )
        answer = "\n\n".join(lines)

    history += [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    memory = update_memory(memory, query, answer)
    return history, memory
