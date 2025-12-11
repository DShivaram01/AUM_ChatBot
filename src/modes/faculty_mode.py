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
PHI_MODEL_ID_DEFAULT: str = "microsoft/phi-2"
PHI_MODEL_ID_SMALL: str = "microsoft/phi-1_5"

# Embedding / index paths (set by init_faculty_mode)
EMB_PATH: str = ""
FAISS_PATH: str = ""
METADATA_PATH: str = ""

# Data + models
faculty_metadata: List[Dict[str, Any]] = []  # each contains "content" field
faculty_embeddings: Optional[np.ndarray] = None
faculty_faiss_index: Optional[faiss.Index] = None
bert_model: Optional[SentenceTransformer] = None
faculty_bm25: Optional[BM25Okapi] = None

phi_model: Optional[AutoModelForCausalLM] = None
phi_tokenizer: Optional[AutoTokenizer] = None

# Indexes and canonical sets
INDEX_NAME = defaultdict(list)       # norm_name(name)      -> [idx,...]
INDEX_DEPARTMENT = defaultdict(list) # norm_text(dept)      -> [idx,...]
INDEX_KEYWORD = defaultdict(list)    # norm_text(keyword)   -> [idx,...]

ALL_NAMES = set()
ALL_DEPARTMENTS = set()
ALL_KEYWORDS = set()

# Memory settings
MAX_MEMORY = 10


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


def retrieve_faculty_layered(query: str) -> List[Dict[str, Any]]:
    """
    Layered retrieval for faculty:
    1) BM25 over metadata-only corpus for ALL faculty.
    2) If BM25 has hits, rerank and return them.
    3) If BM25 is empty, fallback to FAISS semantic search over ALL faculty, then rerank.
    """
    if faculty_bm25 is None or faculty_faiss_index is None or bert_model is None:
        raise RuntimeError("Faculty Mode not initialized. Call init_faculty_mode() first.")

    print(f"\n🧠 Faculty layered retrieve: {query!r}")

    # 1) BM25
    q_tokens = query.lower().split()
    scores = np.array(faculty_bm25.get_scores(q_tokens), dtype=np.float32)
    positive_idxs = np.where(scores > 0)[0]

    if len(positive_idxs) > 0:
        sorted_positive = positive_idxs[np.argsort(scores[positive_idxs])[::-1]]
        bm25_chunks = [faculty_metadata[int(i)] for i in sorted_positive]

        print(f"📥 BM25 hits: {len(bm25_chunks)} (metadata-only)")
        reranked = rerank_results(query, bm25_chunks)
        return reranked if reranked else bm25_chunks

    # 2) FAISS fallback
    print("⚠️ BM25 returned no hits. Falling back to FAISS semantic search.")
    q_embed = bert_model.encode([query], convert_to_numpy=True)
    k_all = faculty_faiss_index.ntotal
    D, I = faculty_faiss_index.search(q_embed, k=k_all)
    faiss_chunks = [faculty_metadata[int(i)] for i in I[0]]

    reranked = rerank_results(query, faiss_chunks)
    return reranked if reranked else faiss_chunks


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


def resolve_query_via_memory(
    query: str,
    memory: List[Dict[str, str]],
    metadata_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    If the query is ambiguous (pronouns) and we have memory,
    scan memory from newest to oldest and return the most recently
    mentioned professor (full metadata dict). Otherwise return None.
    """
    if not is_ambiguous_query(query) or not memory:
        return None

    for turn in reversed(memory):
        hay = f"{turn.get('user','')} {turn.get('assistant','')}".lower()
        for prof in metadata_list:
            if prof["name"].lower() in hay:
                print(f"🔁 Ambiguity resolved via memory → using: {prof['name']}")
                return prof

    return None


def truncate(text: str, max_words: int = 400) -> str:
    return " ".join(text.split()[:max_words])


def build_context(top_chunks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            f"{m['name']} ({m['designation']}, {m['department']}):\n{m['content']}"
            for m in top_chunks
        ]
    )


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
    """
    Main Faculty Research pipeline.

    Steps:
    1) If query is ambiguous (pronouns), try resolving via memory
       to lock onto a specific professor.
    2) If resolved, skip retrieval and just use that professor's data.
    3) Otherwise, use layered retrieval (BM25 -> FAISS fallback).
    4) Build context and send to Phi for generation.
    5) Update memory and history.

    Returns:
        updated_history, updated_memory
    """
    global faculty_metadata, phi_model, phi_tokenizer

    if history is None:
        history = []
    if memory is None:
        memory = []

    if phi_model is None or phi_tokenizer is None or not faculty_metadata:
        raise RuntimeError("Faculty Mode not initialized. Call init_faculty_mode() first.")

    print(f"\n🧠 Incoming Faculty Query: {query}")

    matched_prof: Optional[Dict[str, Any]] = None

    # 1) Ambiguous pronoun → try memory
    if is_ambiguous_query(query):
        matched_prof = resolve_query_via_memory(query, memory, faculty_metadata)

    # 2) If resolved, skip retrieval
    if matched_prof:
        print(f"🎯 Ambiguous query resolved via memory: {matched_prof['name']} (skip BM25/FAISS)")
        top_chunks = [matched_prof]
        name_path = True
    else:
        # 3) Layered retrieval
        top_chunks = retrieve_faculty_layered(query)
        name_path = False

    # Log retrieved chunks
    print("\n📥 Retrieved Faculty Context Chunks:\n")
    for idx, chunk in enumerate(top_chunks[:5]):
        print(f"--- Chunk {idx+1} ---")
        print(f"Name: {chunk['name']}")
        print(f"Department: {chunk['department']}")
        print(f"Email: {chunk['email']}")
        print(f"Courses: {chunk.get('courses_taught', [])}")
        print(f"Content:\n{chunk['content'][:500]}...\n")

    context = truncate(build_context(top_chunks))
    include_memory = not name_path

    prompt = build_memory_prompt(
        memory,
        context,
        query,
        include_memory=include_memory,
    )

    print(f"\n🧩 include_memory_in_prompt = {include_memory} and 📤 Final Prompt Sent to Phi:")
    print(prompt)

    # Generate with Phi
    inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    outputs = phi_model.generate(**inputs, max_new_tokens=300)
    raw_output = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = raw_output.split("Answer:", 1)[-1].strip()

    # Update memory + history
    memory = update_memory(memory, query, answer)
    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]

    return history, memory


# =====================================================================
# DEBUG CLI
# =====================================================================

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug CLI for Faculty Research Mode"
    )

    # Resolve defaults relative to this file: src/faculty_mode.py
    here = Path(__file__).resolve().parent
    default_data = here / "datasets" / "faculty_dataset" / "faculty_data.json"
    default_out = here / "datasets" / "faculty_dataset" / "emb_store"

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
