"""
faculty_mode.py

Modular version of Faculty_Mode.ipynb for use in a multi-mode CS-BOT.

Provides:
- build_faculty_store(...)  # optional one-time index builder
- init_faculty_mode(...)    # load models, metadata, FAISS, BM25
- faculty_pipeline(...)     # main faculty research Q&A
- exam_qa_pipeline(...)     # exam question generator

Intended usage (from main_app.py or a notebook):

    from faculty_mode import init_faculty_mode, faculty_pipeline

    init_faculty_mode(
        metadata_path="embeddings/faculty_metadata.json",
        index_path="embeddings/faculty_faiss_index.bin",
        use_phi15=False,
    )

    history, memory = [], []
    history, memory = faculty_pipeline("What does Dr. X work on?", history, memory)
"""

from __future__ import annotations

import os
import json
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict


# =====================================================================
# GLOBALS (initialized by init_faculty_mode)
# =====================================================================

bert_model: Optional[SentenceTransformer] = None
phi_model: Optional[AutoModelForCausalLM] = None
phi_tokenizer: Optional[AutoTokenizer] = None
metadata: List[Dict] = []
faiss_index: Optional[faiss.Index] = None
faculty_bm25: Optional[BM25Okapi] = None

INDEX_NAME        = defaultdict(list)   # norm_name(name)       -> [idx,...]
INDEX_DEPARTMENT  = defaultdict(list)   # norm_text(dept)       -> [idx,...]
INDEX_KEYWORD     = defaultdict(list)   # norm_text(keyword)    -> [idx,...]

ALL_NAMES        = set()
ALL_DEPARTMENTS  = set()
ALL_KEYWORDS     = set()

MAX_MEMORY = 10  # for memory trimming


# =====================================================================
# OPTIONAL: one-time builder for embeddings + FAISS index + metadata
# =====================================================================

def build_faculty_store(
    data_file: str,
    embedding_output_path: str,
    metadata_output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """
    One-time function to:
    - Read `faculty_data.json`
    - Build text content
    - Encode with SentenceTransformer
    - Build FAISS index
    - Save index and metadata to disk

    Run this offline or in a notebook; then point init_faculty_mode() at the outputs.
    """
    print(f"📂 Loading faculty data from: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    bert = SentenceTransformer(model_name)
    metadata_list: List[Dict] = []
    texts: List[str] = []

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
            metadata_list.append(
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

    print("🧠 Encoding faculty texts...")
    embeddings = bert.encode(texts, convert_to_numpy=True)

    print("📦 Building FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(embedding_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)

    faiss.write_index(index, embedding_output_path)
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print("✅ Data converted and index saved.")
    print(f"   FAISS index: {embedding_output_path}")
    print(f"   Metadata   : {metadata_output_path}")


# =====================================================================
# NORMALIZATION UTILITIES
# =====================================================================

def norm_text(s: str) -> str:
    """Generic cleaner for text fields (department, interests, keywords, etc.)."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_name(name: str) -> str:
    """Name-specific cleaner (faculty names)."""
    if not name:
        return ""
    name = name.lower()
    # remove typical title words
    name = re.sub(r"\b(dr\.?|prof\.?|professor|mr\.?|ms\.?|mrs\.?)\b", "", name)
    name = name.replace(".", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# =====================================================================
# BM25 INDEX BUILDER
# =====================================================================

def build_faculty_bm25(metadata_list: List[Dict]) -> BM25Okapi:
    """
    Build BM25 over metadata-only for each faculty:
    - name, designation, department
    - research interests, research keywords
    - courses taught, recent publications, ongoing projects
    Also fills:
    - INDEX_NAME, INDEX_DEPARTMENT, INDEX_KEYWORD
    - ALL_NAMES, ALL_DEPARTMENTS, ALL_KEYWORDS
    """
    global INDEX_NAME, INDEX_DEPARTMENT, INDEX_KEYWORD
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

        # Small inverted indexes for potential future use
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
    metadata_path: str,
    index_path: str,
    use_phi15: bool = False,
    phi_model_id: Optional[str] = None,
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """
    Initialize global models, metadata, FAISS index, and BM25.

    - metadata_path: path to faculty_metadata.json (from build_faculty_store)
    - index_path:    path to faculty_faiss_index.bin
    - use_phi15:     if True, use microsoft/phi-1_5 instead of phi-2
    - phi_model_id:  override for custom phi model ID
    - embed_model_name: SentenceTransformer model (default MiniLM-L6-v2)
    """
    global metadata, faiss_index, bert_model, phi_model, phi_tokenizer, faculty_bm25

    print(f"📂 Loading metadata from: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"📦 Loading FAISS index from: {index_path}")
    faiss_index = faiss.read_index(index_path)

    print("🧠 Loading SentenceTransformer for query embeddings...")
    bert_model = SentenceTransformer(embed_model_name)

    # Decide phi model ID
    if phi_model_id is not None:
        chosen_phi = phi_model_id
    else:
        chosen_phi = "microsoft/phi-1_5" if use_phi15 else "microsoft/phi-2"

    print(f"🧪 Loading Phi model: {chosen_phi}")
    phi_tokenizer = AutoTokenizer.from_pretrained(chosen_phi)
    phi_model = AutoModelForCausalLM.from_pretrained(
        chosen_phi,
        device_map="auto",
        torch_dtype="auto",
    )

    # Warmup
    _ = phi_model.generate(
        **phi_tokenizer("warmup", return_tensors="pt").to(phi_model.device),
        max_new_tokens=1,
    )

    print("\n[Build] Faculty BM25 corpus...")
    faculty_bm25 = build_faculty_bm25(metadata)
    print("[Build] Faculty BM25 ready.")

    print("✅ Faculty mode initialized successfully.")


# =====================================================================
# MEMORY MANAGEMENT
# =====================================================================

def update_memory(
    memory: List[Dict[str, str]],
    query: str,
    answer: str,
    max_length: int = 5,
) -> List[Dict[str, str]]:
    memory.append(
        {
            "user": query,
            "assistant": answer,
        }
    )
    return memory[-max_length:]


def build_memory_prompt(
    memory: List[Dict[str, str]],
    context: str,
    query: str,
    include_memory: bool = True,
    max_turns: int = 5,
    debug_print: bool = True,
) -> str:
    """
    Build the final prompt for Phi-2.

    include_memory = False:
        do NOT feed prior conversation to the model
        (use this when a professor name is already resolved)
    max_turns:
        how many recent QA pairs to include if include_memory=True
    debug_print:
        print memory contents to console (for debugging)
    """
    recent = memory[-max_turns:] if memory else []
    if debug_print and recent:
        print("\n🧠 Memory (latest first shown last):")
        for i, m in enumerate(recent, start=1):
            print(f"Q{i}: {m.get('user', '')}")
            print(f"A{i}: {m.get('assistant', '')}\n")

    memory_str = ""
    if include_memory and recent:
        memory_str = "\n".join(
            [f"Q: {m.get('user', '')}\nA: {m.get('assistant', '')}" for m in recent]
        )
    elif debug_print and recent:
        print(
            "⚠️ Memory is NOT included in the prompt for this turn "
            "(name-locked or disambiguated query)."
        )

    prompt = f"""You are a helpful academic assistant. Use the context to answer accurately.
Only respond based on the faculty data. Do not make up facts.

Previous Conversation:
{memory_str}

Context from Faculty Data:
{context}

Question: {query}

Answer:"""

    return prompt


# =====================================================================
# NAME / MEMORY UTILITIES
# =====================================================================

def find_prof_by_name(query: str, metadata_list: List[Dict]) -> Optional[Dict]:
    """Return metadata entry if a professor's name is mentioned in the query."""
    query_lower = query.lower()
    for prof in metadata_list:
        name_parts = prof["name"].lower().split()
        if all(part in query_lower for part in name_parts):
            return prof  # full match
        elif any(part in query_lower for part in name_parts):
            return prof  # partial match
    return None


def is_ambiguous_query(query: str) -> bool:
    pronouns = ["him", "her", "he", "she", "his", "they", "them"]
    tokens = query.lower().split()
    return any(p in tokens for p in pronouns)


def resolve_query_via_memory(
    query: str,
    memory: List[Dict[str, str]],
    metadata_list: List[Dict],
) -> Optional[Dict]:
    """
    If the query is ambiguous (pronouns) and we have memory,
    scan memory from newest to oldest and return the most recently
    mentioned professor (full metadata dict). Otherwise return None.
    """
    if not is_ambiguous_query(query) or not memory:
        return None

    for turn in reversed(memory):
        hay = f"{turn.get('user', '')} {turn.get('assistant', '')}".lower()
        for prof in metadata_list:
            if prof["name"].lower() in hay:
                print(f"🔁 Ambiguity resolved via memory → using: {prof['name']}")
                return prof
    return None


# =====================================================================
# RERANKING & STRUCTURED LOGIC
# =====================================================================

def rerank_results(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Simple heuristic reranker using department, research keywords, interests, projects, and courses.
    """
    priority: List[Tuple[float, Dict]] = []
    query_lower = query.lower()
    query_tokens = set(query_lower.split())

    for c in chunks:
        score = 0.0
        dept = (c.get("department", "") or "").lower()
        r_keywords = [kw.lower() for kw in c.get("research_keywords", [])]
        r_interests = (c.get("research_interests", "") or "").lower()
        projects = (c.get("ongoing_projects", "") or "").lower()
        courses = [
            c_name.lower()
            for c_name in c.get("courses_taught", [])
            if isinstance(c_name, str)
        ]

        # Department match
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


def structured_query_response(query: str) -> Optional[str]:
    """
    Very simple direct response when the query clearly mentions names.
    Not heavily used in the main pipeline, but can be useful as a fallback.
    """
    query_lower = query.lower()
    results: List[str] = []

    for prof in metadata:
        name = prof.get("name", "").lower()
        dept = prof.get("department", "").lower()
        research = prof.get("research_interests", "").lower()

        if any(name_part in query_lower for name_part in name.split()):
            results.append(
                f"👤 {prof['name']} ({prof['designation']}, {prof['department']})\n"
                f"📘 Research Interests: {prof['research_interests']}"
            )

    return "\n\n".join(results) if results else None


# =====================================================================
# CONTEXT BUILDERS
# =====================================================================

def truncate(text: str, max_words: int = 400) -> str:
    return " ".join(text.split()[:max_words])


def build_context(top_chunks: List[Dict]) -> str:
    return "\n\n".join(
        [
            f"{m['name']} ({m['designation']}, {m['department']}):\n{m['content']}"
            for m in top_chunks
        ]
    )


# =====================================================================
# RETRIEVAL (BM25 + FAISS)
# =====================================================================

def retrieve_faculty_layered(query: str) -> List[Dict]:
    """
    Layered retrieval for faculty:
    1) Try BM25 over metadata-only corpus using ALL faculty.
    2) If BM25 has hits, return all positive-score candidates (optionally re-ranked).
    3) If BM25 is empty, fallback to FAISS semantic search over ALL faculty.
    """
    if faculty_bm25 is None or faiss_index is None or bert_model is None:
        raise RuntimeError("Faculty mode not initialized. Call init_faculty_mode() first.")

    print(f"\n🧠 Faculty layered retrieve: {query!r}")

    # 1) BM25 over metadata
    q_tokens = query.lower().split()
    scores = np.array(faculty_bm25.get_scores(q_tokens), dtype=np.float32)
    positive_idxs = np.where(scores > 0)[0]

    if len(positive_idxs) > 0:
        sorted_positive = positive_idxs[np.argsort(scores[positive_idxs])[::-1]]
        bm25_chunks = [metadata[int(i)] for i in sorted_positive]

        print(f"📥 BM25 hits: {len(bm25_chunks)} (metadata-only)")
        reranked = rerank_results(query, bm25_chunks)
        if reranked:
            return reranked
        return bm25_chunks

    # 2) FAISS fallback
    print("⚠️ BM25 returned no hits. Falling back to FAISS semantic search.")
    q_embed = bert_model.encode([query], convert_to_numpy=True)
    k_all = faiss_index.ntotal
    D, I = faiss_index.search(q_embed, k=k_all)
    faiss_chunks = [metadata[int(i)] for i in I[0]]

    reranked = rerank_results(query, faiss_chunks)
    if reranked:
        return reranked
    return faiss_chunks


# =====================================================================
# PIPELINE 1: FACULTY RESEARCH PIPELINE
# =====================================================================

def faculty_pipeline(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    memory: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Main faculty research Q&A pipeline.
    Returns:
        updated_history, updated_memory
    """
    if history is None:
        history = []
    if memory is None:
        memory = []

    if phi_model is None or phi_tokenizer is None:
        raise RuntimeError("Faculty mode not initialized. Call init_faculty_mode() first.")

    print(f"\n🧠 Incoming Faculty Query: {query}")

    matched_prof: Optional[Dict] = None
    if is_ambiguous_query(query):
        matched_prof = resolve_query_via_memory(query, memory, metadata)

    if matched_prof:
        print(
            f"🎯 Ambiguous query resolved via memory: "
            f"{matched_prof['name']} (skip BM25/FAISS)"
        )
        top_chunks = [matched_prof]
        name_path = True
    else:
        top_chunks = retrieve_faculty_layered(query)
        name_path = False

    # Log retrieved chunks
    print("\n📥 Retrieved Context Chunks:\n")
    for idx, chunk in enumerate(top_chunks):
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
    print(
        f"\n🧩 include_memory_in_prompt = {include_memory} "
        f"and 📤 Final Prompt Sent to Phi:"
    )
    print(prompt)

    inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    outputs = phi_model.generate(**inputs, max_new_tokens=300)
    raw_output = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = raw_output.split("Answer:")[-1].strip()

    memory = update_memory(memory, query, answer)
    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, memory


# =====================================================================
# PIPELINE 2: EXAM QA PIPELINE
# =====================================================================

def exam_qa_pipeline(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Exam QA pipeline:
    Given a topic, generate exactly 10 MCQs with options and answers.
    """
    if history is None:
        history = []

    if phi_model is None or phi_tokenizer is None:
        raise RuntimeError("Faculty mode not initialized. Call init_faculty_mode() first.")

    print(f"\n🧠 Incoming Exam QA Query: {query}")

    prompt = f"""You are a university teaching assistant helping students prepare for exams.

{query}

Generate exactly 10 multiple choice questions with answers.
Each question should be clearly numbered 1 to 10 and followed by:
A. ...
B. ...
C. ...
D. ...
Answer: <Correct option letter>

Begin below:
"""

    inputs = phi_tokenizer(prompt, return_tensors="pt").to(phi_model.device)
    outputs = phi_model.generate(**inputs, max_new_tokens=700)
    response = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Begin below:")[-1].strip()

    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]
    return updated_history
