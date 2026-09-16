"""
pipeline/classifier.py
========================
Query classification, extracted from backend.py.

Two names in the task's file list do not exist anywhere in backend.py and
are intentionally NOT included here (see workspace.md TASK 11 log entry):
  - _PERSON_RE: no such regex exists in the source. Person detection is
    corpus-aware (query_name_index() in pipeline/retrieval.py), not
    regex-based -- there never was a _PERSON_RE to extract.

Two additions beyond the task's list, because things that already exist
in backend.py need somewhere to live and nothing else claimed them:
  - get_trace(query_id): operates on SESSION_TRACES, used by the trace
    inspector UI (backend.py:293-297).
  - classify_topic(): the five-way auto router used by server/api_server.py.
    It handles COS, Housing, general AUM, general-purpose, and declined
    requests. It is placed here because it is built directly on
    classify_query() and the local hint-word sets.

TASK 12 introduced the Mistral COS-vs-Housing router. TASK 16 added
general-AUM and declined outcomes; TASK 26 added GENERAL for ordinary
open-ended questions. The COS/Housing heuristic remains a fallback only for
those domains; otherwise fallback routing is GENERAL_AUM or GENERAL.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch

from pipeline.memory import logger
from pipeline.retrieval import query_name_index


# ── QueryTrace / RetrievalCandidate (backend.py:194-297) ─────────────

@dataclass
class RetrievalCandidate:
    idx:          int
    title:        str
    mentor:       str
    year:         Any
    department:   str
    bm25_score:   float = 0.0
    bm25_rank:    int   = 9999
    faiss_score:  float = 0.0
    faiss_rank:   int   = 9999
    rrf_score:    float = 0.0
    rerank_score: float = 0.0
    final_rank:   int   = 9999

    def row(self):
        """Returns a display row for the debug table."""
        return {
            "rank":    self.final_rank,
            "title":   self.title[:52],
            "mentor":  self.mentor[:25],
            "year":    self.year,
            "bm25_r":  self.bm25_rank  if self.bm25_rank  < 9999 else "-",
            "faiss_r": self.faiss_rank if self.faiss_rank < 9999 else "-",
            "rrf":     f"{self.rrf_score:.5f}"    if self.rrf_score    else "-",
            "rerank":  f"{self.rerank_score:.4f}" if self.rerank_score else "-",
        }


@dataclass
class QueryTrace:
    # ── Identity ──────────────────────────────────────────────────
    query_id:     str   = ""
    query:        str   = ""
    tab:          str   = ""          # "cos" or "housing"
    timestamp:    str   = ""

    # ── Classification ────────────────────────────────────────────
    intent_type:  str   = ""
    person_hint:  Optional[str] = None
    year_hint:    Optional[int] = None
    dept_hint:    Optional[str] = None
    is_broad:     bool  = False

    # ── Retrieval ─────────────────────────────────────────────────
    bm25_hits:    int   = 0
    faiss_hits:   int   = 0
    rrf_total:    int   = 0
    candidates:   List[RetrievalCandidate] = field(default_factory=list)

    # ── Threshold decision ────────────────────────────────────────
    threshold_top:    float = 0.0
    threshold_mean:   float = 0.0
    threshold_std:    float = 0.0
    threshold_cutoff: float = 0.0
    threshold_passed: bool  = False
    response_state:   str   = ""   # "paragraph" | "list" | "not_found" | "selection"

    # ── Prompt ───────────────────────────────────────────────────
    history_turns_used: int  = 0
    history_text:       str  = ""
    full_prompt:        str  = ""   # COMPLETE prompt, not truncated
    prompt_chars:       int  = 0
    prompt_tokens:      int  = 0

    # ── Generation ───────────────────────────────────────────────
    raw_llm_output:  str   = ""
    final_answer:    str   = ""
    output_tokens:   int   = 0

    # ── State ────────────────────────────────────────────────────
    pending_cands_stored: int  = 0   # how many cands stored in gr.State
    selected_index:       Optional[int] = None

    # ── Timing (ms) ──────────────────────────────────────────────
    t_classify:   float = 0.0
    t_bm25:       float = 0.0
    t_faiss:      float = 0.0
    t_rerank:     float = 0.0
    t_prompt:     float = 0.0
    t_generate:   float = 0.0
    t_total:      float = 0.0

    def timing_summary(self):
        return (
            f"classify={self.t_classify:.0f}ms  "
            f"bm25={self.t_bm25:.0f}ms  "
            f"faiss={self.t_faiss:.0f}ms  "
            f"rerank={self.t_rerank:.0f}ms  "
            f"generate={self.t_generate:.0f}ms  "
            f"total={self.t_total:.0f}ms"
        )


# ── Session trace store (backend.py:288-297) ─────────────────────────
SESSION_TRACES: List[QueryTrace] = []

def get_trace(query_id: str) -> Optional[QueryTrace]:
    for t in reversed(SESSION_TRACES):
        if t.query_id == query_id:
            return t
    return None

logger.info("[Trace] QueryTrace ready. SESSION_TRACES initialized.")


# ── Query ID counter (backend.py:140-144) ─────────────────────────────
_query_counter = 0
def next_query_id(prefix="Q"):
    global _query_counter
    _query_counter += 1
    return f"{prefix}{_query_counter:04d}"


# ── Classifier patterns (backend.py:1926-1960) ────────────────────────

_YEAR_RE = re.compile(r"\b(20\d{2})\b")

_LIST_RE = re.compile(
    r"\b(list|give\s+me|show\s+me|name\s+\d|find\s+\d|\d+\s+projects?|"
    r"examples?\s+of|what\s+are\s+some|which\s+projects?|what\s+.+projects?)\b",
    re.IGNORECASE,
)

_DEPT_PATTERNS = [
    (re.compile(r"\b(math(?:ematics)?)\b", re.IGNORECASE), "Mathematics"),
    (re.compile(r"\b(biol(?:ogy)?|environmental\s+science)\b", re.IGNORECASE), "Biology and Environmental Science"),
    (re.compile(r"\b(chem(?:istry)?)\b", re.IGNORECASE), "Chemistry"),
    (re.compile(r"\b(computer\s+science|comp\s+sci)\b", re.IGNORECASE), "Computer Science and Computer Information Systems"),
    (re.compile(r"\b(psychology|psych)\b", re.IGNORECASE), "Psychology"),
]

_STOPWORDS = {
    "what", "which", "who", "when", "where", "how", "why", "tell",
    "show", "give", "list", "find", "name", "about", "from", "with",
    "that", "this", "have", "were", "been", "does", "did", "they",
    "them", "some", "more", "than", "also", "into", "only", "over",
    "year", "years", "paper", "papers", "project", "projects",
    "research", "studies", "study", "work", "works", "related",
    "presented", "published", "submitted", "done", "made",
}


def classify_query(query: str) -> dict:
    q  = query.strip()
    ql = q.lower()

    result = {
        "type":         "TYPE_TOPIC",
        "person_hints": [],    # list of matched raw name strings
        "person_hint":  None,  # first match (backward compat)
        "year_hint":    None,
        "dept_hint":    None,
        "is_broad":     False,
    }

    # ── 1. Year (always reliable) ─────────────────────────────────
    year_m = _YEAR_RE.search(q)
    if year_m:
        result["year_hint"] = int(year_m.group(1))
        result["is_broad"]  = True
        result["type"]      = "TYPE_BROAD"

    # ── 2. Department (word-boundary only) ───────────────────────
    for pattern, canonical in _DEPT_PATTERNS:
        if pattern.search(q):
            result["dept_hint"] = canonical
            result["is_broad"]  = True
            result["type"]      = "TYPE_BROAD"
            break

    # ── 3. List intent ───────────────────────────────────────────
    if _LIST_RE.search(q):
        result["is_broad"] = True
        if result["type"] == "TYPE_TOPIC":
            result["type"] = "TYPE_BROAD"

    # ── 4. Corpus-aware name detection (replaces regex person) ───
    q_clean_tokens = [
        t for t in ql.replace("?", " ").replace(",", " ").split()
        if t not in _STOPWORDS and len(t) >= 3
    ]
    q_clean = " ".join(q_clean_tokens)

    matched_scored = query_name_index(q_clean, with_scores=True)
    matched_names = [name for name, _score in matched_scored]

    if matched_names:
        result["person_hints"] = matched_names
        result["person_hint"]  = matched_names[0]
        result["type"]         = "TYPE_PERSON"
        result["is_broad"]     = False   # person overrides broad
        result["person_ambiguous"] = len(matched_names) > 1

    return result


# ── COS-vs-Housing auto router (backend.py:1514-1580) ─────────────────
# See module docstring: not assigned to a module in the task spec,
# placed here since it's built directly on classify_query() above.

_COS_HINT_WORDS = {
    "research", "project", "projects", "study", "studies", "paper", "papers",
    "abstract", "mentor", "mentored", "presented", "presenter", "presenters",
    "symposium", "department", "professor", "dr", "author", "authors",
    "thesis", "publication", "published", "lab", "synthesis", "cos",
}
_HOUSING_HINT_WORDS = {
    "dorm", "dorms", "dormitory", "housing", "room", "roommate", "quiet",
    "hours", "guest", "guests", "visitor", "visitors", "pet", "pets",
    "lease", "rent", "fire", "alarm", "curfew", "ra", "resident",
    "residence", "residential", "laundry", "maintenance", "policy",
    "policies", "alcohol", "drink", "drinking",
}


def _classify_topic_heuristic(query: str, embedder, cos_index, H_index, housing_ok: bool) -> str:
    """
    Original embedding+keyword COS-vs-Housing classifier (backend.py:1529-1580).
    Kept as the fallback path for classify_topic() (see its docstring,
    TASK 12) -- used when the Mistral router's output can't be parsed, or
    if the LLM call itself raises.

    Two signals, in order of trust (the caller already handled the
    person-name override before calling this):
      1. Embedding similarity: top-1 FAISS score against both the COS and
         Housing indices (cosine similarity, embeddings are normalized).
      2. Keyword bias: a small nudge from domain words (e.g. "research",
         "mentor" -> cos; "dorm", "quiet hours" -> housing).
    """
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    cos_score = -1.0
    if cos_index is not None and cos_index.ntotal > 0:
        D, _ = cos_index.search(q_vec, 1)
        cos_score = float(D[0][0])

    housing_score = -1.0
    if housing_ok and H_index is not None and H_index.ntotal > 0:
        D, _ = H_index.search(q_vec, 1)
        housing_score = float(D[0][0])

    ql_tokens = set(re.findall(r"[a-z]+", query.lower()))
    cos_hits = len(ql_tokens & _COS_HINT_WORDS)
    housing_hits = len(ql_tokens & _HOUSING_HINT_WORDS)
    BIAS = 0.08  # per matching keyword; small enough not to override a clear embedding signal
    cos_adj = cos_score + BIAS * cos_hits
    housing_adj = housing_score + BIAS * housing_hits

    topic = "housing" if housing_adj > cos_adj else "cos"
    logger.info(
        f"[classify_topic:heuristic] '{query[:60]}' -> {topic} "
        f"(cos_top1={cos_score:.3f}+{BIAS*cos_hits:.2f} "
        f"housing_top1={housing_score:.3f}+{BIAS*housing_hits:.2f})"
    )
    return topic


_GENERAL_AUM_HINT_WORDS = {
    "aum", "auburn montgomery", "tuition", "financial aid", "admissions",
    "registrar", "enrollment", "semester", "academic calendar", "campus",
    "parking", "library", "bookstore", "commencement", "scholarship",
}

_TOPIC_ROUTER_PROMPT = (
    "<s>[INST] You are a routing classifier for an AUM academic assistant. "
    "Reply with exactly one of these five labels, and nothing else:\n"
    "- cos: undergraduate research symposium projects — mentors, presenters, "
    "departments, abstracts, papers, research topics.\n"
    "- housing: AUM Housing and Community Standards policy — dorms, roommates, "
    "quiet hours, guests, pets, fire alarms, leases, maintenance.\n"
    "- general_aum: an AUM university-information question not covered by "
    "the two connected sources, such as admissions, tuition, registration, "
    "academic calendar, parking, or library information.\n"
    "- general: an ordinary open-ended question not specifically about AUM, "
    "such as programming, explanations, writing, weather, recipes, or "
    "entertainment.\n"
    "- out_of_scope: a request that should be declined under policy.\n\n"
    "Message: {query}\n"
    "[/INST]"
)


def _classify_topic_llm(query: str, llm_tok, llm_model) -> str:
    """Ask Mistral for one of the five routing labels without answering."""
    prompt = _TOPIC_ROUTER_PROMPT.format(query=query)
    inputs = llm_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(llm_model.device)
    with torch.no_grad():
        out = llm_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=llm_tok.eos_token_id,
            pad_token_id=llm_tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return llm_tok.decode(new_tokens, skip_special_tokens=True).strip().lower()


def _classify_topic_fallback(query: str, embedder, cos_index, H_index, housing_ok: bool) -> str:
    """Conservative fallback: retain AUM routes, otherwise answer generally."""
    tokens = set(re.findall(r"[a-z]+", query.lower()))
    normalized = " ".join(re.findall(r"[a-z]+", query.lower()))

    cos_hits = len(tokens & _COS_HINT_WORDS)
    housing_hits = len(tokens & _HOUSING_HINT_WORDS)
    if cos_hits or housing_hits:
        return _classify_topic_heuristic(query, embedder, cos_index, H_index, housing_ok)

    if tokens & _GENERAL_AUM_HINT_WORDS or "auburn montgomery" in normalized:
        return "general_aum"

    return "general"


def classify_topic(query: str, embedder, cos_index, H_index, housing_ok: bool,
                   llm_tok=None, llm_model=None) -> str:
    """
    Route automatic-topic requests to COS, Housing, GENERAL_AUM, GENERAL,
    or OUT_OF_SCOPE. COS person-name matches remain a hard override.

    GENERAL bypasses AUM retrieval and is answered with the loaded Mistral
    model under a non-grounded prompt. GENERAL_AUM remains non-retrieval
    because no authoritative general-AUM collection is connected.
    """
    qinfo = classify_query(query)
    if qinfo.get("person_hints"):
        logger.info(f"[classify_topic] '{query[:60]}' -> cos (person match: {qinfo['person_hints'][:2]})")
        return "cos"

    if llm_tok is not None and llm_model is not None:
        try:
            decoded = _classify_topic_llm(query, llm_tok, llm_model)
            labels = {"general_aum", "out_of_scope", "housing", "general", "cos"}
            topic = re.sub(r"[^a-z_]", "", decoded)
            if topic in labels:
                logger.info(
                    f"[classify_topic] '{query[:60]}' -> {topic} "
                    f"(Mistral router, raw={decoded!r})"
                )
                return topic
            logger.warning(
                f"[classify_topic] Mistral router gave unparseable output {decoded!r} "
                f"for '{query[:60]}' -- using conservative fallback"
            )
        except Exception as exc:
            logger.warning(
                f"[classify_topic] Mistral router raised {exc!r} -- "
                "using conservative fallback"
            )

    topic = _classify_topic_fallback(query, embedder, cos_index, H_index, housing_ok)
    logger.info(f"[classify_topic] '{query[:60]}' -> {topic} (fallback)")
    return topic


def run_smoke_tests(cos_meta):
    """
    NAME_INDEX + classify_query() self-tests (backend.py:1894-1912,
    2017-2051). Not runnable at import time here like backend.py's flat
    script did, because they need NAME_INDEX populated first, which needs
    cos_meta loaded, which needs the embedder loaded -- so main.py calls
    this explicitly after data load, instead of it firing as a side effect
    of importing this module.
    """
    from pipeline.retrieval import query_name_index

    _name_tests = [
        ("bhattacharya",            True,  "last name only"),
        ("sutanu bhattacharya",     True,  "full name"),
        ("kursun",                  True,  "last name Kursun"),
        ("priscilla",               True,  "first name only"),
        ("robert spicer",           True,  "presenter full name"),
        ("spicer",                  True,  "presenter last name"),
        ("jerome goddard",          True,  "mentor full name"),
        ("protein sequences",       False, "topic word, not a name"),
        ("biology",                 False, "department word, not a name"),
        ("2024",                    False, "year, not a name"),
    ]
    for q, should_match, desc in _name_tests:
        hits = query_name_index(q)
        ok   = bool(hits) == should_match
        logger.info(
            f"[NameIndex] {'OK' if ok else 'FAIL'}  '{q}' → {hits[:2]} ({desc})"
        )

    _smoke = [
        ("List 3 projects related to Dr. Sutanu Bhattacharya", "TYPE_PERSON", "Bhattacharya"),
        ("list projects related to dr. olcay kursun",          "TYPE_PERSON", "Kursun"),
        ("Name few papers mentored by Dr. Olcay Kursun",       "TYPE_PERSON", "Kursun"),
        ("What paper did Robert Spicer present in 2024?",      "TYPE_PERSON", "Spicer"),
        ("show me a paper presented by Priscilla",             "TYPE_PERSON", "Priscilla"),
        ("what paper did robert spicer present at AUM",        "TYPE_PERSON", "Spicer"),
        ("projects by kursun and okeke",                       "TYPE_PERSON", "Kursun"),
        ("Tell me about Jerome Goddard",                       "TYPE_PERSON", "Goddard"),
        ("What Biology projects were presented in 2024?",      "TYPE_BROAD",  None),
        ("What research has been done in the Mathematics dept?","TYPE_BROAD", None),
        ("Name a research project related to protein sequences","TYPE_TOPIC", None),
        ("List all papers in 2025",                            "TYPE_BROAD",  None),
        ("Tell me about the GoFold project",                   "TYPE_TOPIC",  None),
    ]

    smoke_pass = 0
    for q, exp_type, exp_name in _smoke:
        got     = classify_query(q)
        type_ok = got["type"] == exp_type
        name_ok = (exp_name is None) or any(
            exp_name.lower() in h.lower() for h in got["person_hints"]
        )
        ok = type_ok and name_ok
        if ok:
            smoke_pass += 1
        logger.info(
            f"[Classifier] {'OK' if ok else 'FAIL'}  '{q[:60]}'"
            f" -> {got['type']} names={[h[:20] for h in got['person_hints'][:2]]}"
            f" dept={got['dept_hint']} year={got['year_hint']}"
        )

    logger.info(f"[Classifier] Smoke tests: {smoke_pass}/{len(_smoke)} passed")
    return smoke_pass, len(_smoke)
