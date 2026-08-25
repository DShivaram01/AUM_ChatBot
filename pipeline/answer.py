"""
pipeline/answer.py
====================
Answer templates, LLM prompt construction, and streaming generation,
extracted from backend.py.

Two names in the task's file list needed a decision, not a verbatim copy
(see workspace.md TASK 11 log entry for full detail):

  - _cos_prompt_single(): did not exist as a separate function in
    backend.py -- the prompt string was built inline inside
    _explain_one_project() (backend.py:1269-1280). Factored out here into
    its own function to match the requested name; this is a pure
    extraction of the same literal prompt text, not a behavior change.

  - _cos_prompt_multi(): does NOT exist anywhere in backend.py and has no
    corresponding logic to extract. The comment at backend.py:1406-1408
    explicitly says multi-project LLM synthesis was REMOVED ("Instead of
    LLM synthesising 3 projects (breaks names/years), show top result as
    template"). Adding a stub function here would misrepresent a
    capability that was deliberately removed, so it is intentionally
    omitted rather than fabricated.

generate_streaming() is not in the task's file list for any module, but
every function below calls it, so it's included here rather than left
orphaned -- see the models/loader.py vs. here placement note in
workspace.md TASK 11 log entry.
"""

import time
import threading

import numpy as np
from transformers import TextIteratorStreamer

import config
from pipeline.memory import logger
from pipeline.classifier import QueryTrace, RetrievalCandidate, SESSION_TRACES

SIGMA = config.SIGMA


# ── Relative threshold helper (backend.py:701-708) ────────────────────

def _relative_threshold(cands, top_n=10):
    scores = [c["rerank"] for c in cands[:top_n] if c.get("rerank") is not None]
    if not scores:
        return False, 0.0, 0.0, 0.0
    top  = scores[0]
    mean = float(np.mean(scores))
    std  = float(np.std(scores)) if len(scores) > 1 else 0.0
    return top >= mean + SIGMA * std, top, mean, std


# ── LLM streaming generation (backend.py:1041-1101) ────────────────────

def generate_streaming(
    prompt, tokenizer, model,
    query_id="Q", max_new_tokens=300,
    trace: QueryTrace = None,
):
    MAX_CHARS = 3200
    if len(prompt) > MAX_CHARS:
        prompt = prompt[:MAX_CHARS]
        logger.warning(f"[{query_id}] Prompt truncated to {MAX_CHARS} chars")

    if trace:
        trace.full_prompt   = prompt
        trace.prompt_chars  = len(prompt)

    logger.info(f"[{query_id}] == LLM PROMPT ==")
    logger.info(f"[{query_id}] {prompt[:500]}{'...' if len(prompt) > 500 else ''}")

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    if trace:
        trace.prompt_tokens = int(inputs["input_ids"].shape[1])
    logger.info(f"[{query_id}] Input tokens: {inputs['input_ids'].shape[1]}")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=90.0
    )
    gen_kw = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        no_repeat_ngram_size=4,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    t0     = time.time()
    thread = threading.Thread(target=model.generate, kwargs=gen_kw)
    thread.start()

    collected = []
    for token in streamer:
        collected.append(token)
        yield "".join(collected)

    thread.join()
    full = "".join(collected).strip()
    ms   = (time.time() - t0) * 1000

    if trace:
        trace.raw_llm_output = full
        trace.output_tokens  = len(tokenizer.encode(full))
        trace.t_generate     = ms

    logger.info(f"[{query_id}] == LLM OUTPUT ==")
    logger.info(f"[{query_id}] {full[:600]}{'...' if len(full) > 600 else ''}")
    logger.info(f"[{query_id}] Generation: {ms:.0f}ms  {len(full)} chars")


# ── Citation / text helpers (backend.py:1114-1134) ─────────────────────

def extract_abstract(full_text):
    parts = full_text.split("Abstract:", 1)
    return parts[1].strip() if len(parts) == 2 else full_text.strip()

def _trunc(text, n):
    return text[:n] + "..." if len(text) > n else text

def _cos_cite(meta):
    year  = meta.get("year", "")
    title = meta.get("title", "")
    leads = meta.get("lead_presenters", []) or []
    if leads:
        last = leads[0].split()[-1] if leads[0].strip() else ""
        pres = f"{last} et al." if len(leads) > 1 else leads[0]
    else:
        pres = "Unknown"
    return f'({year}, "{_trunc(title, 55)}", {pres})'

def _housing_cite(chunk):
    return (f"(AUM Housing Policy, "
            f"section {chunk.get('section','General')}, p.{chunk.get('page','?')})")


# ── Template: numbered list (backend.py:1138-1162) ─────────────────────

def _format_chooser_list(cands, qinfo):
    show  = cands[:8]
    lines = []
    for i, c in enumerate(show, 1):
        m     = c["meta"]
        title = m.get("title", "Unknown")
        year  = m.get("year", "")
        dept  = m.get("department", "")
        leads = ", ".join(m.get("lead_presenters", []) or [])
        mentor = m.get("mentor", "")
        lines.append(
            f"{i}. {title}\n"
            f"   Year: {year}  |  Department: {dept}\n"
            f"   Presenters: {leads}\n"
            f"   Mentor: {mentor}"
        )
    parts = ["I found the following projects"]
    if qinfo.get("dept_hint"):
        parts.append(f"in {qinfo['dept_hint'].title()}")
    if qinfo.get("year_hint"):
        parts.append(f"from {qinfo['year_hint']}")
    header = " ".join(parts) + ":"
    body   = "\n\n".join(lines)
    footer = '\nReply with a number (e.g. "3") to get a detailed summary.'
    return header + "\n\n" + body + footer, show


# ── Template: person query answer — NO LLM (backend.py:1181-1231) ──────

def _format_person_answer(cands, person_hints):
    """
    Deterministic template output for person queries.
    Exact facts from metadata. Zero LLM. Zero hallucination.
    """
    seen  = set()
    clean = []
    for h in person_hints:
        key = h.lower().strip()
        if key not in seen and len(key) >= 3:
            seen.add(key)
            clean.append(h.strip())

    if clean:
        names_str = " and ".join(clean)
    else:
        names_str = "the requested person"

    show  = cands[:6]
    lines = []
    for i, c in enumerate(show, 1):
        m      = c["meta"]
        title  = m.get("title", "Unknown")
        year   = m.get("year", "")
        dept   = m.get("department", "")
        leads  = ", ".join(m.get("lead_presenters", []) or [])
        mentor = m.get("mentor", "")
        others = ", ".join(m.get("other_authors",   []) or [])
        cit    = _cos_cite(m)

        block = (
            f"{i}. {title}\n"
            f"   Year: {year}  |  Department: {dept}\n"
            f"   Lead Presenters: {leads}"
        )
        if others:
            block += f"\n   Other Authors: {others}"
        block += f"\n   Mentor: {mentor}\n   Citation: {cit}"
        lines.append(block)

    header = f"Here are projects associated with {names_str} at AUM:"
    body   = "\n\n".join(lines)
    footer = '\nReply with a number (e.g. "2") for a detailed summary of that project.'
    return header + "\n\n" + body + "\n" + footer


# ── Prompt builder: single-project abstract explanation ────────────────
# Factored out of _explain_one_project() (backend.py:1269-1280) -- see
# module docstring.

def _cos_prompt_single(abstract: str) -> str:
    return (
        "<s>[INST] "
        "You are a research explainer. Summarise the research abstract below "
        "in ONE plain-English paragraph (3-5 sentences). "
        "Do NOT mention institution names, presenter names, or years — "
        "those are already shown to the user. "
        "Do NOT repeat the title. Just explain what the research is about "
        "and why it matters.\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Write one plain-English paragraph explaining this research. "
        "[/INST]"
    )


# ── LLM prompt: explain ONE selected project (backend.py:1238-1292) ────

def _explain_one_project(cand, query_id, tokenizer, model):
    """
    Returns plain string (not streaming) for a single selected project.
    Uses LLM only to paraphrase the abstract in plain English.
    All factual fields (title, year, dept, presenters, mentor, citation)
    are written by template BEFORE and AFTER the LLM paragraph.
    LLM only touches: abstract explanation.
    """
    m        = cand["meta"]
    title    = m.get("title", "Unknown")
    year     = m.get("year", "")
    dept     = m.get("department", "")
    leads    = ", ".join(m.get("lead_presenters", []) or [])
    others   = ", ".join(m.get("other_authors",   []) or [])
    mentor   = m.get("mentor", "")
    cit      = _cos_cite(m)
    abstract = _trunc(extract_abstract(cand["text"]), 700)

    header = (
        f"{title}\n"
        f"Year: {year}  |  Department: {dept}\n"
        f"Lead Presenters: {leads}"
    )
    if others:
        header += f"\nOther Authors: {others}"
    header += f"\nMentor: {mentor}\n"

    prompt = _cos_prompt_single(abstract)

    logger.info(f"[{query_id}] LLM call: explain abstract only ({len(prompt)} chars)")

    full_explanation = ""
    for partial in generate_streaming(prompt, tokenizer, model, query_id, max_new_tokens=450):
        full_explanation = partial

    footer = f"\nCitation: {cit}"

    return header + "\n" + full_explanation.strip() + footer


# ── Main COS answer builder (backend.py:1296-1415) ──────────────────────

def build_cos_answer_streaming(
    query, cands, qinfo,
    tokenizer, model,
    query_id,
    history_text="",      # kept for signature compat — not used
    selected_cand=None,
):
    from datetime import datetime

    trace = QueryTrace(
        query_id     = query_id,
        query        = query,
        tab          = "cos",
        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        intent_type  = qinfo["type"],
        person_hint  = qinfo.get("person_hint"),
        year_hint    = qinfo.get("year_hint"),
        dept_hint    = qinfo.get("dept_hint"),
        is_broad     = qinfo.get("is_broad", False),
        history_turns_used = 0,   # memory removed
        history_text       = "",
    )

    # ── STATE 3: nothing found ────────────────────────────────────
    if not cands and selected_cand is None:
        msg = (
            "I could not find any COS symposium projects matching your query.\n"
            "Try asking about a specific department, year, mentor, or project title."
        )
        trace.response_state = "not_found"
        trace.final_answer   = msg
        SESSION_TRACES.append(trace)
        logger.info(f"[{query_id}] STATE 3: not-found")
        yield msg, None, trace
        return

    # ── User selected a numbered item → explain that one project ─
    if selected_cand is not None:
        logger.info(f"[{query_id}] STATE 1 (selection): template + abstract LLM")
        trace.response_state = "selection"
        trace.selected_index = selected_cand.get("_selection_n")

        answer = _explain_one_project(selected_cand, query_id, tokenizer, model)
        trace.final_answer = answer
        SESSION_TRACES.append(trace)
        logger.info(f"[{query_id}] FINAL ANSWER (selection): {answer[:200]}")
        yield answer, None, trace
        return

    # ── Person query → template answer, NO LLM ───────────────────
    if qinfo["type"] == "TYPE_PERSON":
        logger.info(f"[{query_id}] STATE PERSON: template output, no LLM")
        trace.response_state = "person_template"

        for rank, c in enumerate(cands[:10], 1):
            m = c["meta"]
            trace.candidates.append(RetrievalCandidate(
                idx=c["idx"], title=m.get("title",""),
                mentor=m.get("mentor",""), year=m.get("year",""),
                department=m.get("department",""),
                bm25_score=c["bm25"], rrf_score=c["combined"],
                rerank_score=c.get("rerank", 0.0), final_rank=rank,
            ))

        answer = _format_person_answer(cands, qinfo.get("person_hints", []))
        trace.pending_cands_stored = min(len(cands), 6)
        trace.final_answer         = answer
        SESSION_TRACES.append(trace)
        logger.info(f"[{query_id}] FINAL ANSWER (person): {answer[:200]}")
        yield answer, cands[:6], trace
        return

    # ── Broad query → numbered list, NO LLM ──────────────────────
    scores = [c["rerank"] for c in cands[:10] if c.get("rerank") is not None]
    if scores:
        top  = scores[0]
        mean = float(np.mean(scores))
        std  = float(np.std(scores)) if len(scores) > 1 else 0.0
        is_strong = top >= mean + SIGMA * std
    else:
        top = mean = std = 0.0
        is_strong = False

    trace.threshold_top    = top
    trace.threshold_mean   = mean
    trace.threshold_std    = std
    trace.threshold_cutoff = mean + SIGMA * std
    trace.threshold_passed = is_strong

    if qinfo["is_broad"] or not is_strong:
        logger.info(f"[{query_id}] STATE 2: list (broad={qinfo['is_broad']} strong={is_strong})")
        trace.response_state = "list"
        list_text, stored = _format_chooser_list(cands, qinfo)
        trace.pending_cands_stored = len(stored)
        trace.final_answer         = list_text
        for rank, c in enumerate(cands[:10], 1):
            m = c["meta"]
            trace.candidates.append(RetrievalCandidate(
                idx=c["idx"], title=m.get("title",""),
                mentor=m.get("mentor",""), year=m.get("year",""),
                department=m.get("department",""),
                bm25_score=c["bm25"], rrf_score=c["combined"],
                rerank_score=c.get("rerank", 0.0), final_rank=rank,
            ))
        SESSION_TRACES.append(trace)
        yield list_text, stored, trace
        return

    # ── Topic query, strong hit → explain top project, no multi-doc LLM ──
    logger.info(f"[{query_id}] STATE 1 (topic): template + abstract explanation")
    trace.response_state = "topic_explain"
    answer = _explain_one_project(cands[0], query_id, tokenizer, model)
    trace.final_answer = answer
    SESSION_TRACES.append(trace)
    logger.info(f"[{query_id}] FINAL ANSWER (topic): {answer[:200]}")
    yield answer, None, trace


# ── Housing answer builder (backend.py:1419-1450) ────────────────────

def build_housing_answer_streaming(
    query, hits, tokenizer, model, query_id, history_text="", search_ms=0.0,
):
    """Build a Housing answer and retain a QueryTrace for the inspector and feedback log.

    Housing has no BM25/RRF candidate shape. Its FAISS hits are stored in the
    existing RetrievalCandidate list with the policy section as ``title`` and
    the page number in ``mentor`` so render_trace_md() can display both paths
    without a parallel trace schema.
    """
    from datetime import datetime

    started = time.time()
    trace = QueryTrace(
        query_id=query_id,
        query=query,
        tab="housing",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        intent_type="housing",
        history_turns_used=0,
        history_text=history_text,
        t_faiss=search_ms,
    )

    for rank, hit in enumerate(hits, 1):
        chunk = hit["chunk"]
        trace.candidates.append(RetrievalCandidate(
            idx=rank - 1,
            title=chunk.get("section", "General"),
            mentor=f"p.{chunk.get('page', '?')}",
            year="",
            department="Housing policy",
            faiss_score=hit.get("score", 0.0),
            faiss_rank=rank,
            final_rank=rank,
        ))
    trace.faiss_hits = len(trace.candidates)

    if not hits:
        msg = (
            "I could not find relevant information in the AUM Housing "
            "and Community Standards.\n"
            "Please check the official policy document or contact the Housing Office."
        )
        trace.response_state = "not_found"
        trace.final_answer = msg
        trace.t_total = (time.time() - started) * 1000
        SESSION_TRACES.append(trace)
        logger.info(f"[{query_id}] No housing hits")
        yield msg, trace
        return

    ctx = "\n\n---\n\n".join(_trunc(h["chunk"]["text"], 500) for h in hits)
    cits = "  ".join(_housing_cite(h["chunk"]) for h in hits)

    prompt = (
        "<s>[INST] "
        "You are an AUM Housing policy assistant. "
        "Use ONLY the policy text inside <context> tags. "
        "Do NOT invent rules. Write ONE paragraph. No bullet points. "
        "End with the citations.\n\n"
        f"Question: {query}\n\n"
        f"<context>\n{ctx}\n</context>\n\n"
        f"Write one paragraph answering the student. End with: {cits} "
        "[/INST]"
    )
    trace.response_state = "paragraph"
    full = ""
    for partial in generate_streaming(
        prompt, tokenizer, model, query_id, max_new_tokens=550, trace=trace
    ):
        full = partial
        yield partial, trace
    trace.final_answer = full
    trace.t_total = (time.time() - started) * 1000
    SESSION_TRACES.append(trace)
    logger.info(f"[{query_id}] FINAL HOUSING ANSWER: {full[:300]}")


# ── Trace renderer (backend.py:320-413) ─────────────────────────────

def render_trace_md(trace: QueryTrace) -> str:
    if trace is None:
        return "_No trace available._"

    lines = []

    lines.append(f"## Trace: `{trace.query_id}`  —  {trace.timestamp}")
    lines.append(f"Query: `{trace.query}`")
    lines.append(f"Tab: {trace.tab}  |  State: `{trace.response_state}`")
    lines.append("")

    lines.append("### 1. Classification")
    lines.append(f"- Type: `{trace.intent_type}`")
    lines.append(f"- Person hint: `{trace.person_hint}`")
    lines.append(f"- Year hint: `{trace.year_hint}`")
    lines.append(f"- Dept hint: `{trace.dept_hint}`")
    lines.append(f"- Is broad: `{trace.is_broad}`")
    lines.append("")

    lines.append("### 2. Retrieval Candidates (after rerank)")
    lines.append(f"BM25 hits: {trace.bm25_hits}  |  FAISS hits: {trace.faiss_hits}  |  RRF pool: {trace.rrf_total}")
    lines.append("")

    lines.append("| Rank | Title | Mentor | Yr | BM25_r | FAISS_r | RRF | Rerank |")
    lines.append("|------|-------|--------|----|--------|---------|-----|--------|")
    for c in trace.candidates[:8]:
        r = c.row()
        lines.append(
            f"| {r['rank']} | {r['title']} | {r['mentor']} | {r['year']} "
            f"| {r['bm25_r']} | {r['faiss_r']} | {r['rrf']} | {r['rerank']} |"
        )
    lines.append("")

    lines.append("### 3. Threshold Decision")
    lines.append(
        f"top={trace.threshold_top:.4f}  "
        f"mean={trace.threshold_mean:.4f}  "
        f"std={trace.threshold_std:.4f}  "
        f"cutoff={trace.threshold_cutoff:.4f}  "
        f"passed={trace.threshold_passed}"
    )
    lines.append(f"→ Response state: {trace.response_state}")
    lines.append("")

    lines.append("### 4. Chat History Injected")
    lines.append(f"Turns used: {trace.history_turns_used}")
    if trace.history_text:
        lines.append("```")
        lines.append(trace.history_text[:600])
        lines.append("```")
    else:
        lines.append("_None (first turn or cleared)_")
    lines.append("")

    lines.append("### 5. Temporary State (gr.State)")
    lines.append(f"pending_cands stored this turn: {trace.pending_cands_stored}")
    if trace.selected_index is not None:
        lines.append(f"User selected item: #{trace.selected_index}")
    lines.append("")

    lines.append("### 6. Full Prompt to LLM")
    lines.append(f"Length: {trace.prompt_chars} chars  |  tokens: {trace.prompt_tokens}")
    lines.append("```")
    lines.append(trace.full_prompt)
    lines.append("```")
    lines.append("")

    lines.append("### 7. LLM Output")
    lines.append(f"Output tokens: {trace.output_tokens}")
    lines.append("Raw output:")
    lines.append("```")
    lines.append(trace.raw_llm_output)
    lines.append("```")
    lines.append("Final answer:")
    lines.append("```")
    lines.append(trace.final_answer)
    lines.append("```")
    lines.append("")

    lines.append("### 8. Timing")
    lines.append(f"`{trace.timing_summary()}`")
    lines.append("")

    return "\n".join(lines)
