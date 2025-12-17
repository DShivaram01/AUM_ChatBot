"""
faculty_mode.py

Faculty Research Mode for your CS-BOT.

Responsibilities:
- Load faculty_data.json (raw faculty profiles)
- Build rich "content" blocks per professor
- Build / load Sentence-BERT embeddings + FAISS index
- Build BM25 over metadata (name, dept, interests, keywords, etc.)
- Use layered retrieval: BM25 → FAISS fallback + heuristic rerank
- Generate answers with Phi (phi-2 by default) using retrieved context
- Maintain a small memory for pronoun resolution ("him", "her", etc.)

Public API:
- init_faculty_mode(data_file, out_dir, rebuild_store=False, ...)
- faculty_pipeline(query, history=None, memory=None)

Debug CLI:
    python -m src.faculty_mode
or (depending on layout)
    python src/faculty_mode.py

It will:
- auto-resolve default dataset + emb_store paths relative to this file
- init Faculty Mode
- open an interactive Q&A loop until you type 'exit' or Ctrl+C
"""

from __future__ import annotations

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from collections import defaultdict
from pathlib import Path
import argparse

import torch
from sentence_transformers import SentenceTransformer
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

# Data + models
faculty_metadata: List[Dict[str, Any]] = []  # each contains "content" field
faculty_bm25: Optional[BM25Okapi] = None
reranker = None  # CrossEncoder

phi_model: Optional[AutoModelForCausalLM] = None
phi_tokenizer: Optional[AutoTokenizer] = None

# Layer-1 Filter Indexes (requested)
INDEX_PROFESSORS   = defaultdict(set)  # norm_name(full_name) -> {idx,...}
INDEX_DEPARTMENTS  = defaultdict(set)  # norm_text(dept)      -> {idx,...}
INDEX_COURSES_EXACT = defaultdict(set) # norm_text(course)    -> {idx,...}
INDEX_COURSES_TOKEN = defaultdict(set) # token               -> {idx,...}
INDEX_DESIGNATIONS = defaultdict(set)  # norm_text(desig)     -> {idx,...}

# Optional helper (fast name matching)
INDEX_PROF_TOKENS  = defaultdict(set)  # token -> {idx,...}

# Canonical sets (debug / future UI)
ALL_PROFESSORS  = set()
ALL_DEPARTMENTS = set()
ALL_COURSES     = set()
ALL_DESIGNATIONS = set()

# Memory settings (as per your new spec)
MAX_MEMORY = 5

# Stopwords for robust course-token matching
COURSE_STOPWORDS = {
    "and","or","of","the","a","an","to","in","for","with","on","at","by",
    "i","ii","iii","iv",
    "intro","introduction","advanced","fundamentals","principles",
    "seminar","topics","special","studies","lab","laboratory"
}



# =====================================================================
# NORMALIZERS
# =====================================================================

def norm_text(s: str) -> str:
    """Generic cleaner for texty fields (dept, interests, keywords, courses)."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_name(name: str) -> str:
    """Name-specific cleaner for faculty names."""
    if not name:
        return ""
    name = name.lower()
    # remove typical title words
    name = re.sub(r"\b(dr\.?|prof\.?|professor|mr\.?|ms\.?|mrs\.?)\b", "", name)
    name = name.replace(".", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# =====================================================================
# BUILD / LOAD STORE
# =====================================================================

def _build_faculty_store(data_file: str) -> None:
    """
    Build:
      - faculty_metadata (with a rich 'content' block)
      - faculty_embeddings (Sentence-BERT)
      - FAISS index (L2)
      - METADATA json

    Saves them into paths under out_dir.
    """
    global faculty_metadata, faculty_embeddings, faculty_faiss_index, bert_model

    if bert_model is None:
        raise RuntimeError("bert_model must be initialized before _build_faculty_store().")

    print(f"\n📂 Loading faculty data from: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    texts: List[str] = []
    faculty_metadata = []

    for prof in raw_data:
        try:
            content = f"""
Name: {prof['name']}
Designation: {prof['designation']}
Department: {prof['department']}
Email: {prof['email']}
Courses Taught: {', '.join(prof.get('courses_taught', []))}
Recent Publications: {', '.join(prof.get('recent_publications', []))}
Ongoing Projects: {prof.get('ongoing_projects', '')}
Research Interests: {prof.get('research_interests', '')}
Research Keywords: {', '.join(prof.get('research_keywords', []))}
""".strip()

            texts.append(content)
            faculty_metadata.append(
                {
                    "name": prof["name"],
                    "designation": prof["designation"],
                    "department": prof["department"],
                    "email": prof["email"],
                    "courses_taught": prof.get("courses_taught", []),
                    "recent_publications": prof.get("recent_publications", []),
                    "ongoing_projects": prof.get("ongoing_projects", ""),
                    "research_interests": prof.get("research_interests", ""),
                    "research_keywords": prof.get("research_keywords", []),
                    "content": content,
                }
            )
        except Exception as e:
            print(f"❌ Error processing entry: {prof.get('name', 'Unknown')} - {e}")

    print(f"✅ Processed {len(faculty_metadata)} faculty entries.")

    # Encode & build FAISS
    print("\n[Build] Encoding faculty texts for embeddings...")
    faculty_embeddings = bert_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(faculty_embeddings.shape[1])
    index.add(faculty_embeddings)
    faculty_faiss_index = index

    # Save to disk
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_PATH)
    np.save(EMB_PATH, faculty_embeddings)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(faculty_metadata, f, indent=2)

    print("✅ Faculty index and metadata saved.")


def _load_faculty_store() -> None:
    """
    Load:
      - faculty_metadata
      - faculty_embeddings
      - FAISS index
    from the previously saved files.
    """
    global faculty_metadata, faculty_embeddings, faculty_faiss_index

    print("\n📥 Loading faculty emb_store from disk...")
    faculty_faiss_index = faiss.read_index(FAISS_PATH)
    faculty_embeddings = np.load(EMB_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        faculty_metadata = json.load(f)

    print(f"✅ Loaded faculty store: {len(faculty_metadata)} profiles | index size = {faculty_faiss_index.ntotal}")


# =====================================================================
# BM25 + INDEXES
# =====================================================================

def build_faculty_bm25(metadata_list: List[Dict[str, Any]]) -> BM25Okapi:
    """
    Build BM25 over metadata-only for each faculty:
    - name
    - designation
    - department
    - research interests
    - research keywords
    - courses taught
    - recent publications
    - ongoing projects
    Also populates INDEX_NAME, INDEX_DEPARTMENT, INDEX_KEYWORD and canonical sets.
    """
    global ALL_NAMES, ALL_DEPARTMENTS, ALL_KEYWORDS

    corpus_tokens: List[List[str]] = []

    for idx, prof in enumerate(metadata_list):
        name   = prof.get("name", "") or ""
        desig  = prof.get("designation", "") or ""
        dept   = prof.get("department", "") or ""
        r_int  = prof.get("research_interests", "") or ""
        r_keys = " ".join(prof.get("research_keywords", []) or [])
        courses = " ".join(prof.get("courses_taught", []) or [])
        pubs   = " ".join(prof.get("recent_publications", []) or [])
        proj   = prof.get("ongoing_projects", "") or ""

        combined = " ".join(
            [
                name,
                desig,
                dept,
                r_int,
                r_keys,
                courses,
                pubs,
                proj,
            ]
        )

        tokens = combined.lower().split()
        corpus_tokens.append(tokens)

        # Small indexes for meta lookups
        n_norm = norm_name(name)
        if n_norm:
            ALL_NAMES.add(name)
            INDEX_NAME[n_norm].append(idx)

        d_norm = norm_text(dept)
        if d_norm:
            ALL_DEPARTMENTS.add(dept)
            INDEX_DEPARTMENT[d_norm].append(idx)

        for kw in prof.get("research_keywords", []) or []:
            kw = kw.strip()
            if not kw:
                continue
            ALL_KEYWORDS.add(kw)
            kw_norm = norm_text(kw)
            if kw_norm:
                INDEX_KEYWORD[kw_norm].append(idx)

    return BM25Okapi(corpus_tokens)



# =====================================================================
# INITIALIZATION
# =====================================================================

def init_faculty_mode(
    data_file: str,
    out_dir: str,
    rebuild_store: bool = False,
    embed_model_id: str = EMBED_MODEL_ID_DEFAULT,
    use_phi15: bool = False,
    phi_model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    Initialize Faculty Research Mode.

    Args:
        data_file:     Path to faculty_data.json (raw faculty info).
        out_dir:       Directory to store embeddings/index/metadata (emb_store).
        rebuild_store: If True, always rebuild emb_store from faculty_data.json.
        embed_model_id: SentenceTransformer model ID (default all-MiniLM-L6-v2).
        use_phi15:     If True, use microsoft/phi-1_5 instead of phi-2.
        phi_model_id:  Optional manual override for Phi model ID.
        device:        "cuda" or "cpu". If None, auto-detect.
    """
    global DEVICE, EMB_PATH, FAISS_PATH, METADATA_PATH
    global bert_model, faculty_bm25
    global phi_model, phi_tokenizer

    if device is not None:
        DEVICE = device

    # Paths
    os.makedirs(out_dir, exist_ok=True)
    EMB_PATH = os.path.join(out_dir, "faculty_embeddings.npy")
    FAISS_PATH = os.path.join(out_dir, "faculty_faiss_index.bin")
    METADATA_PATH = os.path.join(out_dir, "faculty_metadata.json")

    # Load SBERT
    print(f"\n🧠 Loading SentenceTransformer for Faculty Mode: {embed_model_id} (device={DEVICE})")
    bert_model = SentenceTransformer(embed_model_id, device=DEVICE)

    # Build or load store
    need_build = rebuild_store or not (
        os.path.exists(EMB_PATH)
        and os.path.exists(FAISS_PATH)
        and os.path.exists(METADATA_PATH)
    )

    if need_build:
        _build_faculty_store(data_file)
    else:
        _load_faculty_store()

    # Build BM25
    print("\n[Build] Faculty BM25 corpus...")
    faculty_bm25 = build_faculty_bm25(faculty_metadata)
    print("[Build] Faculty BM25 ready.")
    
    # Build Layer-1 indexes
    print("\n[Build] Layer-1 indexes (prof/department/course/designation)...")
    build_layer1_indexes(faculty_metadata)
    print("✅ Layer-1 indexes ready.")
    
    # Load CrossEncoder reranker
    from sentence_transformers import CrossEncoder
    print(f"\n🧠 Loading CrossEncoder reranker: {RERANKER_ID_DEFAULT} (device={DEVICE})")
    global reranker
    reranker = CrossEncoder(RERANKER_ID_DEFAULT, device=DEVICE)
    print("✅ Reranker ready.")

    # Load Phi model
    if phi_model_id is not None:
        chosen_phi_id = phi_model_id
    else:
        chosen_phi_id = PHI_MODEL_ID_SMALL if use_phi15 else PHI_MODEL_ID_DEFAULT

    print(f"\n🧪 Loading Phi model for Faculty Mode: {chosen_phi_id}")
    phi_tokenizer = AutoTokenizer.from_pretrained(chosen_phi_id)
    phi_model = AutoModelForCausalLM.from_pretrained(
        chosen_phi_id,
        device_map="auto",
        torch_dtype="auto",
    )

    # Warmup
    _ = phi_model.generate(
        **phi_tokenizer("warmup", return_tensors="pt").to(phi_model.device),
        max_new_tokens=1,
    )
    print("✅ Faculty Mode initialized (emb_store + BM25 + Phi ready).")


# ====================================================================

# Layer-1 Filter

# =====================================================================

def _tokenize(s: str) -> List[str]:
    s = norm_text(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []


def _course_tokens(course: str) -> List[str]:
    toks = _tokenize(course)
    return [t for t in toks if t not in COURSE_STOPWORDS and len(t) >= 3]


def build_layer1_indexes(metadata_list: List[Dict[str, Any]]) -> None:
    """
    Build Layer-1 indexes:
    - professors, departments, courses (exact + token), designations
    """
    global INDEX_PROFESSORS, INDEX_DEPARTMENTS, INDEX_COURSES_EXACT, INDEX_COURSES_TOKEN, INDEX_DESIGNATIONS, INDEX_PROF_TOKENS
    global ALL_PROFESSORS, ALL_DEPARTMENTS, ALL_COURSES, ALL_DESIGNATIONS

    # reset
    INDEX_PROFESSORS.clear()
    INDEX_DEPARTMENTS.clear()
    INDEX_COURSES_EXACT.clear()
    INDEX_COURSES_TOKEN.clear()
    INDEX_DESIGNATIONS.clear()
    INDEX_PROF_TOKENS.clear()

    ALL_PROFESSORS.clear()
    ALL_DEPARTMENTS.clear()
    ALL_COURSES.clear()
    ALL_DESIGNATIONS.clear()

    for idx, prof in enumerate(metadata_list):
        name = prof.get("name","") or ""
        dept = prof.get("department","") or ""
        desig = prof.get("designation","") or ""
        courses = prof.get("courses_taught", []) or []

        # Professors
        n_norm = norm_name(name)
        if n_norm:
            INDEX_PROFESSORS[n_norm].add(idx)
            ALL_PROFESSORS.add(name)

            for t in _tokenize(n_norm):
                if len(t) >= 2:
                    INDEX_PROF_TOKENS[t].add(idx)

        # Departments
        d_norm = norm_text(dept)
        if d_norm:
            INDEX_DEPARTMENTS[d_norm].add(idx)
            ALL_DEPARTMENTS.add(dept)

        # Designations
        z_norm = norm_text(desig)
        if z_norm:
            INDEX_DESIGNATIONS[z_norm].add(idx)
            ALL_DESIGNATIONS.add(desig)

        # Courses (robust)
        for c in courses:
            if not isinstance(c, str):
                continue
            c = c.strip()
            if not c:
                continue
            ALL_COURSES.add(c)

            c_norm = norm_text(c)
            INDEX_COURSES_EXACT[c_norm].add(idx)

            toks = _course_tokens(c)
            for t in toks:
                INDEX_COURSES_TOKEN[t].add(idx)


def _layer1_find_professors(query: str) -> List[int]:
    """
    Direct name retrieval:
    - If query contains multiple tokens that match a professor's name tokens,
      we return those prof IDs.
    """
    q = norm_name(query)
    q_tokens = set(_tokenize(q))

    # Quick token-based candidate union
    token_cands = set()
    for t in q_tokens:
        token_cands |= INDEX_PROF_TOKENS.get(t, set())

    if not token_cands:
        return []

    # Strengthen: require >=2 overlapping tokens with that prof name (unless query has only 1 token)
    min_overlap = 2 if len(q_tokens) >= 2 else 1
    out = []
    for idx in token_cands:
        p_name = faculty_metadata[idx].get("name","")
        p_tokens = set(_tokenize(norm_name(p_name)))
        if len(p_tokens & q_tokens) >= min_overlap:
            out.append(idx)

    # stable order
    return sorted(set(out))


def _layer1_filter_candidates(query: str) -> Tuple[str, List[int]]:
    """
    Return (reason, candidate_ids) from Layer-1 indexes.
    Reason is one of: "professor" | "metadata_subset" | "none"
    """
    q_norm = norm_text(query)
    q_tokens = set(_tokenize(q_norm))

    # 1) Professor direct hit
    prof_ids = _layer1_find_professors(query)
    if prof_ids:
        return "professor", prof_ids

    # 2) Department / designation / course matches
    cands = set()

    # department: match by phrase-in-query OR token overlap with full dept phrase
    for dept_norm, ids in INDEX_DEPARTMENTS.items():
        if dept_norm and dept_norm in q_norm:
            cands |= ids

    for desig_norm, ids in INDEX_DESIGNATIONS.items():
        if desig_norm and desig_norm in q_norm:
            cands |= ids

    # courses:
    # - exact phrase
    for course_norm, ids in INDEX_COURSES_EXACT.items():
        if course_norm and course_norm in q_norm:
            cands |= ids

    # - token index (robust): require >=2 meaningful course tokens from query if possible
    q_course_tokens = [t for t in q_tokens if t not in COURSE_STOPWORDS and len(t) >= 3]
    token_hits = set()
    for t in q_course_tokens:
        token_hits |= INDEX_COURSES_TOKEN.get(t, set())

    if token_hits:
        # enforce “>=2 token” evidence when query has enough tokens
        if len(q_course_tokens) >= 2:
            refined = []
            qt = set(q_course_tokens)
            for idx in token_hits:
                prof_courses = faculty_metadata[idx].get("courses_taught", []) or []
                best_overlap = 0
                for c in prof_courses:
                    toks = set(_course_tokens(c if isinstance(c, str) else ""))
                    best_overlap = max(best_overlap, len(toks & qt))
                if best_overlap >= 2:
                    refined.append(idx)
            cands |= set(refined)
        else:
            cands |= token_hits

    if cands:
        return "metadata_subset", sorted(cands)

    return "none", []

# =====================================================================
# RETRIEVAL HELPERS
# =====================================================================

def rerank_results(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple heuristic reranker on top of BM25/FAISS candidates.
    Gives more weight to:
    - Department phrase matches
    - Research keywords / interests
    - Courses and projects
    """
    priority = []
    query_lower = query.lower()
    query_tokens = set(query_lower.split())

    for c in chunks:
        score = 0.0
        dept = (c.get("department", "") or "").lower()
        r_keywords = [kw.lower() for kw in c.get("research_keywords", [])]
        r_interests = (c.get("research_interests", "") or "").lower()
        projects = (c.get("ongoing_projects", "") or "").lower()
        courses = [
            c_name.lower() for c_name in c.get("courses_taught", []) if isinstance(c_name, str)
        ]

        # Department matching (stricter)
        dept_tokens = set(dept.split())

        if dept and dept in query_lower:
            score += 2.0
        else:
            overlap = dept_tokens & query_tokens
            if len(overlap) >= 2:
                score += 1.5

        # Keywords / interests / projects / courses
        if any(kw in query_lower for kw in r_keywords):
            score += 1.0
        if r_interests and r_interests in query_lower:
            score += 1.0
        if any(cname in query_lower for cname in courses):
            score += 0.5
        if projects and any(word in query_tokens for word in projects.split()):
            score += 0.5

        priority.append((score, c))

    priority = [x for x in priority if x[0] > 0]
    priority.sort(reverse=True, key=lambda x: x[0])

    print("🔍 Top reranked candidates:")
    for score, chunk in priority[:5]:
        print(f"✔️ {chunk['name']} | Score: {score:.1f} | Dept: {chunk['department']}")

    return [x[1] for x in priority]


def _bm25_rank_ids(query: str, candidate_ids: Optional[List[int]] = None) -> List[int]:
    """
    Returns BM25-ranked ids (descending) with score > 0.
    If candidate_ids is provided, we only consider those ids.
    """
    if faculty_bm25 is None:
        raise RuntimeError("faculty_bm25 is not initialized.")
    q_tokens = _tokenize(query)
    scores = np.array(faculty_bm25.get_scores(q_tokens), dtype=np.float32)

    if candidate_ids is None:
        pos = np.where(scores > 0)[0]
    else:
        cand = np.array(list(set(candidate_ids)), dtype=np.int64)
        pos = cand[scores[cand] > 0]

    if len(pos) == 0:
        return []

    # sort by score desc
    pos = pos[np.argsort(scores[pos])[::-1]]
    return [int(i) for i in pos]


def _cross_encoder_rerank(query: str, ids: List[int]) -> List[int]:
    """
    CrossEncoder rerank over the provided ids.
    Keep only positive rerank scores (>0).
    """
    if reranker is None:
        raise RuntimeError("reranker is not initialized.")

    if not ids:
        return []

    pairs = [(query, faculty_metadata[i]["content"]) for i in ids]
    scores = reranker.predict(pairs)

    scored = [(int(i), float(s)) for i, s in zip(ids, scores)]
    scored = [(i, s) for (i, s) in scored if s > 0.0]  # keep positive only

    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored]


def retrieve_faculty_new(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    New retrieval:
    1) Layer-1 indexes:
       - professor → direct candidates
       - else subset candidates → BM25(subset) → CrossEncoder
    2) If no Layer-1 candidates → BM25(global) → CrossEncoder
    3) If still empty → return []
    Returns: (path, chunks)
      path in {"direct_professor", "subset_bm25_ce", "global_bm25_ce", "none"}
    """
    if not faculty_metadata:
        raise RuntimeError("Faculty metadata is empty. Did you call init_faculty_mode()?")

    reason, layer1_ids = _layer1_filter_candidates(query)

    # A) Professor direct
    if reason == "professor" and layer1_ids:
        chunks = [faculty_metadata[i] for i in layer1_ids]
        return "direct_professor", chunks

    # B) Subset path (dept/course/designation)
    if reason == "metadata_subset" and layer1_ids:
        bm_ids = _bm25_rank_ids(query, candidate_ids=layer1_ids)
        ce_ids = _cross_encoder_rerank(query, bm_ids if bm_ids else layer1_ids)
        chunks = [faculty_metadata[i] for i in ce_ids]
        return ("subset_bm25_ce" if chunks else "none"), chunks

    # C) Global BM25 + CE
    bm_ids = _bm25_rank_ids(query, candidate_ids=None)
    ce_ids = _cross_encoder_rerank(query, bm_ids)
    chunks = [faculty_metadata[i] for i in ce_ids]
    return ("global_bm25_ce" if chunks else "none"), chunks



# =====================================================================
# MEMORY + PRONOUN RESOLUTION
# =====================================================================

def update_memory(
    memory: List[Dict[str, str]],
    query: str,
    answer: str,
    max_length: int = MAX_MEMORY,
) -> List[Dict[str, str]]:
    memory.append({"user": query, "assistant": answer})
    return memory[-max_length:]


def is_ambiguous_query(query: str) -> bool:
    pronouns = ["him", "her", "he", "she", "his", "they", "them"]
    return any(p in query.lower().split() for p in pronouns)


def resolve_query_via_memory_last_turn(
    memory: List[Dict[str, str]],
) -> List[int]:
    """
    Only inspect memory[-1] (last saved turn), extract professor ids if names appear.
    """
    if not memory:
        return []

    last = memory[-1]
    hay = f"{last.get('user','')} {last.get('assistant','')}"
    return _layer1_find_professors(hay)



def truncate(text: str, max_words: int = 400) -> str:
    return " ".join(text.split()[:max_words])


def build_context(top_chunks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            f"{m['name']} ({m['designation']}, {m['department']}):\n{m['content']}"
            for m in top_chunks
        ]
    )
def _clean_model_answer(text: str) -> str:
    """
    Prevent the model from emitting fake multi-turn Q/A blocks.
    Keep only the first answer segment and strip any "Question:" echoes.
    """
    t = text.strip()

    # Keep after "Answer:" if present
    if "Answer:" in t:
        t = t.split("Answer:", 1)[1].strip()

    # Stop if it starts inventing a new QA
    for stop in ["\nQuestion:", "\nQ:", "\nUser:", "\nAssistant:"]:
        if stop in t:
            t = t.split(stop, 1)[0].strip()

    # first paragraph only (optional)
    t = t.split("\n\n", 1)[0].strip()
    return t


def build_strict_prompt(context: str, query: str) -> str:
    """
    Stronger instruction: answer only once; never create extra Q/A.
    """
    return f"""You are a helpful academic assistant.

Rules (strict):
- Use ONLY the provided faculty context. If the answer is not in the context, say: "I don't have that information in the faculty dataset."
- Answer the user's question ONCE. Do NOT create follow-up questions or additional Q/A.
- Do NOT invent courses, departments, projects, or research topics.

Faculty Context:
{context}

User Question: {query}

Answer:"""


def build_memory_prompt(
    memory: List[Dict[str, str]],
    context: str,
    query: str,
    include_memory: bool = True,
    max_turns: int = 5,
    debug_print: bool = True,
) -> str:
    """
    Build the final prompt for Phi.

    include_memory = False -> do NOT feed prior conversation to the model
                              (use this when name is already resolved).
    """
    recent = memory[-max_turns:] if memory else []

    if debug_print and recent:
        print("\n🧠 Memory (latest first shown last):")
        for i, m in enumerate(recent, start=1):
            print(f"Q{i}: {m.get('user','')}")
            print(f"A{i}: {m.get('assistant','')}\n")

    if include_memory and recent:
        memory_str = "\n".join(
            [f"Q: {m.get('user','')}\nA: {m.get('assistant','')}" for m in recent]
        )
    else:
        memory_str = ""
        if debug_print and recent:
            print("⚠️ Memory is NOT included in the prompt for this turn.")

    prompt = f"""You are a helpful academic assistant. Use the faculty data context to answer accurately.
Only respond based on the faculty data. Do not make up facts.

Previous Conversation:
{memory_str}

Context from Faculty Data:
{context}

Question: {query}

Answer:"""

    return prompt


# =====================================================================
# FACULTY PIPELINE
# =====================================================================

def faculty_pipeline(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    memory: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    global phi_model, phi_tokenizer

    if history is None:
        history = []
    if memory is None:
        memory = []

    if phi_model is None or phi_tokenizer is None or not faculty_metadata:
        raise RuntimeError("Faculty Mode not initialized. Call init_faculty_mode() first.")

    print(f"\n🧠 Incoming Faculty Query: {query}")

    # 1) Ambiguous → memory[-1] name-lock
    if is_ambiguous_query(query):
        prof_ids = resolve_query_via_memory_last_turn(memory)
        if prof_ids:
            print(f"🎯 Ambiguous query resolved via memory[-1] → {prof_ids}")
            top_chunks = [faculty_metadata[i] for i in prof_ids]
            context = truncate(build_context(top_chunks))
            prompt = build_strict_prompt(context, query)

            inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
            outputs = phi_model.generate(**inputs, max_new_tokens=280)
            raw = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = _clean_model_answer(raw)

            memory = update_memory(memory, query, answer, max_length=MAX_MEMORY)
            history = history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
            return history, memory

    # 2) Not ambiguous → Layer-1 indexes → subset/global BM25+CE → or none
    path, chunks = retrieve_faculty_new(query)
    print(f"🧭 Retrieval path: {path}")

    if not chunks:
        answer = "I couldn’t find anything related to your query in the faculty dataset. Please try a different query (e.g., a professor name, department, course, or designation)."
        memory = update_memory(memory, query, answer, max_length=MAX_MEMORY)
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        return history, memory

    context = truncate(build_context(chunks))
    prompt = build_strict_prompt(context, query)

    inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    outputs = phi_model.generate(**inputs, max_new_tokens=280)
    raw = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = _clean_model_answer(raw)

    memory = update_memory(memory, query, answer, max_length=MAX_MEMORY)
    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    return history, memory


# =====================================================================
# Gradio / main_app-friendly chat handler
# =====================================================================

def faculty_chat_handler(
    query: str,
    history=None,
    memory=None,
):
    """
    Thin wrapper so main_app.py can treat faculty mode as a chat handler.

    Expected behavior:
    - history: list of {"role": "...", "content": "..."} messages
    - memory:  list of past QA pairs for disambiguation
    - Returns: (updated_history, updated_memory)
    """
    if history is None:
        history = []
    if memory is None:
        memory = []

    updated_history, updated_memory = faculty_pipeline(
        query=query,
        history=history,
        memory=memory,
    )
    return updated_history, updated_memory

# =====================================================================
# DEBUG CLI
# =====================================================================

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug CLI for Faculty Research Mode"
    )

    # faculty_mode.py is in src/modes/
    here = Path(__file__).resolve().parent           # .../src/modes
    # Find the first parent that has a "datasets" folder
    base = here
    while base != base.parent and not (base / "datasets").exists():
        base = base.parent

    datasets_root = base / "datasets"

    default_data = datasets_root / "faculty_dataset" / "faculty_data.json"
    default_out  = datasets_root / "faculty_dataset" / "emb_store"

    parser.add_argument(
        "--data-file",
        type=str,
        default=str(default_data),
        help=f"Path to faculty_data.json (default: {default_data})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out),
        help=f"Directory for emb_store (default: {default_out})",
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

    print("\n=== Faculty Mode CLI ===")
    print(f"Data file : {args.data_file}")
    print(f"Out dir   : {args.out_dir}")
    print(f"Rebuild   : {args.rebuild}")
    print(f"Use phi-1_5: {args.phi15}")
    print(f"Device    : {args.device or DEVICE}")
    print("========================\n")

    # Init mode
    init_faculty_mode(
        data_file=args.data_file,
        out_dir=args.out_dir,
        rebuild_store=args.rebuild,
        use_phi15=args.phi15,
        device=args.device,
    )

    history: List[Dict[str, str]] = []
    memory: List[Dict[str, str]] = []

    print("Type your questions about faculty. Type 'exit' or 'quit' to stop.\n")

    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Bye 👋")
                break

            history, memory = faculty_pipeline(q, history, memory)
            last_answer = history[-1]["content"]
            print(f"\nBot: {last_answer}\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Bye 👋")


if __name__ == "__main__":
    _run_cli()
