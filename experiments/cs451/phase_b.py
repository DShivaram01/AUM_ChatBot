"""Phase B: answer and quiz generation on the CS 451 benchmark (real Mistral,
run in the `qzgen` env during an approved maintenance window).

    python -m experiments.cs451.phase_b --out experiments/cs451/results/phase_b.json

ANSWERS (90 non-quiz queries: 80 answerable + 10 intentionally unanswerable)
  A0  production AssistantService.document_chat, unchanged (production chunks,
      MiniLM, top-5, evidence gate, 500-char cut, production prompt)
  P1  same retrieval (production chunks + MiniLM top-5) but whole chunks and a
      prompt requiring unit-ID citations or "Not specified in the document."
      -> isolates the prompt/evidence-presentation effect
  P2  best measured retrieval (R512 chunks + Qwen3-Embedding-0.6B top-5,
      precomputed in rankings_R512_qwen06.json) with the P1 prompt
      -> isolates the retrieval effect

QUIZZES (Q091-Q100)
  live      production routing: does detect_quiz_intent recognise the prompt?
            if not, what production returns (document_chat prose)
  QP        production pipeline.quiz.generate_quiz, unchanged, with production
            evidence (top-6 production chunks for the prompt) and the
            requested count -- what production would produce IF routed
  QI        integrated candidate: word-number-aware request parsing, CS451
            blueprint (sections from unit labels; whole / topic / numeric /
            terminology / integrative modes), evidence sufficiency check for
            topic requests (one short Mistral yes/no per candidate chunk;
            none -> decline with supported alternatives), Outlines
            schema-constrained batches of 5, unit-ID citations in the key,
            targeted per-slot repair
  QI_nochk  QI without the sufficiency check (topic prompts only) -- ablation

Automatic scores are PROXIES (numeric/keyword match, lexical support,
embedding duplicates); all outputs are saved for human adjudication per the
evaluator guide.
"""

import argparse
import json
import re
import time
from functools import lru_cache

import numpy as np

from experiments.cs451 import chunking as C
from experiments.cs451.units import contained_units, ground_truth, norm, pdf_bytes

FLOOR = 0.35
ANSWER_TOKENS = 300
ABSTAIN_RE = re.compile(r"could not find enough information|not specified in the document|does not (?:state|specify|mention|provide|include|say)|"
                        r"(?:is|are) not (?:stated|specified|provided|mentioned|given)|no information|cannot be determined from", re.I)
ID_RE = re.compile(r"\bS[1-6]-[PTE]\d{2}\b")
WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                      "fifteen sixteen seventeen eighteen nineteen twenty".split())}
STOP = set("the a an of and or to in on for with is are be by as at from that this which what how why when does do its it "
           "than then into per not no".split())


# ---------------------------------------------------------------- shared data

@lru_cache(maxsize=1)
def emb():
    from sentence_transformers import SentenceTransformer
    import config
    return SentenceTransformer(config.EMBED_MODEL_ID, device="cpu")


@lru_cache(maxsize=None)
def chunk_store(name):
    ch = C.build(name)
    units = [contained_units(c["text"]) for c in ch]
    D = emb().encode([c["text"] for c in ch], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return ch, units, D


def dense(name, query, k):
    ch, units, D = chunk_store(name)
    s = D @ emb().encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    order = np.argsort(-s)[:k]
    return [(int(i), float(s[i])) for i in order]


def section_of_units(units):
    secs = [u[:2] for u in units if u[3] == "P"] or [u[:2] for u in units]
    return max(set(secs), key=secs.count) if secs else None


# ---------------------------------------------------------------- answers

def answer_prompt(question, excerpts):
    ev = "\n\n".join(f"<excerpt {i}>\n{t}\n</excerpt {i}>" for i, t in enumerate(excerpts, 1))
    return ("<s>[INST] Answer the question using ONLY the course document excerpts below. The excerpts contain labelled "
            "units such as S1-P02 (passage), S1-T01 (table) and S1-E02 (equation). Cite every unit you rely on in square "
            "brackets, e.g. [S1-P02]. Keep scenario qualifiers exact (e.g. teaching processor vs ResearchGrid deployment, "
            "classroom vs deployed values). If the excerpts do not contain the answer, reply exactly: Not specified in the "
            f"document.\n\n{ev}\n\nQuestion: {question} [/INST]")


NUM_RE = re.compile(r"(?<![A-Za-z0-9.\-])\d+(?:\.\d+)?(?![A-Za-z0-9])")


def numbers(t):
    """Standalone numbers only: digits inside identifiers (L1D, L2, S1-P02) are not values."""
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in NUM_RE.findall(t.replace(",", ""))}


def score_answer(q, text):
    abst = bool(ABSTAIN_RE.search(text))
    exp = q["expected_answer"]
    nums, got_nums = numbers(exp), numbers(ID_RE.sub(" ", text))
    words = [w for w in re.findall(r"[a-z][a-z\-]{3,}", norm(exp)) if w not in STOP]
    stem = lambda w: w[:-1] if w.endswith("s") and len(w) > 4 else w  # noqa: E731
    kw = sum(stem(w) in norm(text) for w in words) / len(words) if words else 1.0
    if q["expected_abstention"]:
        correct = abst
    else:
        correct = (not abst) and nums <= got_nums and kw >= (0.5 if nums else 0.6)
    cited = set(ID_RE.findall(text))
    req = set(q["required_evidence_passage_ids"])
    ctx = set(q.get("context_passage_ids") or [])
    return {"abstained": abst, "auto_correct": bool(correct), "keyword_overlap": round(kw, 2),
            "numbers_expected": sorted(nums), "numbers_missing": sorted(nums - got_nums),
            "cited_ids": sorted(cited),
            "citation_precision": round(len(cited & (req | ctx)) / len(cited), 3) if cited else None,
            "citation_recall": round(len(cited & req) / len(req), 3) if req else None}


def run_answers(lm, svc, doc_id, session):
    import core.assistant_service as AS
    gt = ground_truth()
    qs = [q for q in gt["queries"] if q["category"] != "quiz_generation"]
    rk = json.load(open("experiments/cs451/results/rankings_R512_qwen06.json"))["rankings"]
    r512 = C.build("R512")
    prod, _, _ = chunk_store("PROD")
    out = {"A0": {}, "P1": {}, "P2": {}}
    shim = lm.production_streaming_shim()
    orig = AS.generate_streaming
    for q in qs:
        b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens); t = time.perf_counter()
        AS.generate_streaming = shim
        try:
            final = ""
            for partial, *_ in svc.document_chat(q["query_text"], [doc_id], session):
                final = partial
        finally:
            AS.generate_streaming = orig
        out["A0"][q["query_id"]] = {"answer": final, "seconds": round(time.perf_counter() - t, 2),
                                    "calls": lm.log.calls - b[0], "input_tokens": lm.log.input_tokens - b[1],
                                    "output_tokens": lm.log.output_tokens - b[2], **score_answer(q, final)}
        for arm, excerpts in (("P1", [prod[i]["text"] for i, _ in dense("PROD", q["query_text"], 5)]),
                              ("P2", [r512[i]["text"] for i in rk[q["query_id"]]["top10"][:5]])):
            b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens); t = time.perf_counter()
            text = lm.generate(answer_prompt(q["query_text"], excerpts), ANSWER_TOKENS)
            out[arm][q["query_id"]] = {"answer": text, "seconds": round(time.perf_counter() - t, 2), "calls": 1,
                                       "input_tokens": lm.log.input_tokens - b[1], "output_tokens": lm.log.output_tokens - b[2],
                                       **score_answer(q, text)}
        print(q["query_id"], q["category"][:12], " ".join(f"{a}:{'✓' if out[a][q['query_id']]['auto_correct'] else '✗'}"
                                                         f"{'(abst)' if out[a][q['query_id']]['abstained'] else ''}" for a in out), flush=True)
    summary = {}
    for arm, recs in out.items():
        ans = [q for q in qs if not q["expected_abstention"]]
        una = [q for q in qs if q["expected_abstention"]]
        cats = sorted({q["category"] for q in ans})
        prec = [recs[q["query_id"]]["citation_precision"] for q in ans if recs[q["query_id"]]["citation_precision"] is not None]
        summary[arm] = {
            "auto_correct_answerable": round(sum(recs[q["query_id"]]["auto_correct"] for q in ans) / len(ans), 3),
            "auto_correct_by_category": {c: round(sum(recs[q["query_id"]]["auto_correct"] for q in ans if q["category"] == c) /
                                                  sum(1 for q in ans if q["category"] == c), 3) for c in cats},
            "false_abstention_rate": round(sum(recs[q["query_id"]]["abstained"] for q in ans) / len(ans), 3),
            "unanswerable_abstention_rate": round(sum(recs[q["query_id"]]["abstained"] for q in una) / len(una), 3),
            "unsupported_answer_rate": round(sum(not recs[q["query_id"]]["abstained"] for q in una) / len(una), 3),
            "answers_with_citation": round(sum(bool(recs[q["query_id"]]["cited_ids"]) for q in ans) / len(ans), 3),
            "citation_precision_mean": round(sum(prec) / len(prec), 3) if prec else None,
            "latency_s_median": float(np.median([r["seconds"] for r in recs.values()])),
            "latency_s_p95": float(np.percentile([r["seconds"] for r in recs.values()], 95)),
            "input_tokens_total": sum(r["input_tokens"] for r in recs.values()),
            "output_tokens_total": sum(r["output_tokens"] for r in recs.values()),
        }
    return summary, out


# ---------------------------------------------------------------- quizzes

def parse_request(text):
    t = norm(text)
    m = re.search(r"\b(\d{1,2})\b", t)
    count = int(m.group(1)) if m else next((WORDNUM[w] for w in t.split() if w in WORDNUM), 5)
    modes = set()
    if re.search(r"entire|whole|spanning|throughout|across the|course pdf|the pdf|cs 451 document", t):
        modes.add("whole")
    if re.search(r"integrative|different sections|cross-section|across sections", t):
        modes |= {"integrative", "whole"}
    if re.search(r"numeric|numerical|quantitative", t):
        modes |= {"numeric", "whole"}
    if re.search(r"terminolog|differentiat|similar terms|distinguish", t):
        modes |= {"terminology", "whole"}
    if "whole" not in modes:
        modes.add("topic")
    topic = re.sub(r"^(make|create|generate|produce|write)\s+(\w+\s+)?(\w+\s+)?(quiz\s+|exam\s+)?questions?\s+(on|about|from)?\s*", "", t)
    return {"count": min(20, max(1, count)), "modes": sorted(modes), "topic": topic or t}


OBJECTIVES = {"default": ["recall a stated fact, value or definition", "apply it (compute or interpret a value, unit or scenario)",
                          "distinguish it from a related idea or a differently scoped value"],
              "numeric": ["compute or read an exact value with its unit and scenario from a table or equation"],
              "terminology": ["distinguish two similar terms or scopes that the passage explicitly contrasts"],
              "integrative": ["combine one fact from EACH of the two passages (different sections) in a single question"]}


def spread(seq):
    """Evenly spread order (van der Corput): first, middle, quarter, three-quarter, ...
    so a few slots reach early, middle and late passages of a long section."""
    n = len(seq)
    if n <= 2:
        return list(seq)
    order, seen, k = [], set(), 0
    while len(order) < n:
        x, f, kk = 0.0, 0.5, k
        while kk:
            x += f * (kk & 1); kk >>= 1; f /= 2
        i = int(x * n)
        if i not in seen:
            seen.add(i); order.append(seq[i])
        k += 1
    return order


def eligible(name="PROD"):
    ch, units, D = chunk_store(name)
    return [i for i, u in enumerate(units) if any(x[3] in "PTE" for x in u) and C.ntok(ch[i]["text"]) >= 40]


def build_blueprint(req, lm=None, check=True):
    """-> (slots, note, check_log). slot = {"slot", "chunks": [idx...], "objective", "sections"}"""
    ch, units, D = chunk_store("PROD")
    elig = eligible()
    n = req["count"]
    note, check_log = "", None
    if "topic" in req["modes"]:
        hits = [(i, s) for i, s in dense("PROD", req["topic"], 40) if i in elig and s >= FLOOR][:12]
        if check and lm is not None and hits:
            kept, check_log = [], []
            for i, s in hits:
                ans = lm.generate(f"<s>[INST] Passage:\n{ch[i]['text']}\n\nCan a factual multiple-choice question specifically about "
                                  f"\"{req['topic']}\" be answered from this passage alone, without outside knowledge? "
                                  "Answer only yes or no. [/INST]", 3)
                ok = ans.strip().lower().startswith("yes")
                check_log.append({"chunk": i, "score": round(s, 3), "answer": ans.strip()[:10]})
                if ok:
                    kept.append((i, s))
            hits = kept
        if not hits:
            alt = sorted({section_of_units(units[i]) for i in elig})
            return [], f"declined: the document does not support questions on '{req['topic']}'. Supported sections: {', '.join(alt)}", check_log
        pool = [[i] for i, _ in hits]
        objectives = OBJECTIVES["default"]
    else:
        by_sec = {}
        for i in elig:
            by_sec.setdefault(section_of_units(units[i]), []).append(i)
        by_sec = {k: spread(v) for k, v in by_sec.items()}
        secs = sorted(k for k in by_sec if k)
        if "numeric" in req["modes"]:
            by_sec = {s: ([i for i in v if any(u[3] in "TE" for u in units[i])] or v) for s, v in by_sec.items()}
            objectives = OBJECTIVES["numeric"]
        elif "terminology" in req["modes"]:
            pat = re.compile(r"\bnot\b|versus|distractor|separable|different|distinct", re.I)
            by_sec = {s: ([i for i in v if pat.search(ch[i]["text"][:200])] or v) for s, v in by_sec.items()}
            objectives = OBJECTIVES["terminology"]
        elif "integrative" in req["modes"]:
            objectives = OBJECTIVES["integrative"]
        else:
            objectives = OBJECTIVES["default"]
        pool, pos = [], {s: 0 for s in secs}
        while len(pool) < n * len(objectives) and any(pos[s] < len(by_sec[s]) for s in secs):
            for s in secs:
                if pos[s] < len(by_sec[s]):
                    pool.append([by_sec[s][pos[s]]]); pos[s] += 1
        if "integrative" in req["modes"]:
            paired = []
            for grp in pool:
                i = grp[0]; si = section_of_units(units[i])
                cands = [j for j in elig if section_of_units(units[j]) not in (si, None)]
                j = max(cands, key=lambda j: float(D[i] @ D[j]))
                paired.append([i, j])
            pool = paired
    slots, used = [], {}
    k = 0
    while len(slots) < n and k < len(pool) * len(objectives):
        grp = pool[k % len(pool)]
        key = tuple(grp); m = used.get(key, 0)
        if m < len(objectives):
            used[key] = m + 1
            slots.append({"slot": len(slots) + 1, "chunks": grp, "objective": objectives[m],
                          "sections": sorted({section_of_units(units[i]) for i in grp})})
        k += 1
    if len(slots) < n:
        note = f"the document supports only {len(slots)} distinct slots for this request (requested {n})"
    return slots, note, check_log


def quiz_prompt(slots, chunks_text, avoid):
    ids = []
    for s in slots:
        for i in s["chunks"]:
            if i not in ids:
                ids.append(i)
    num = {i: k for k, i in enumerate(ids, 1)}
    ev = "\n\n".join(f"[{num[i]}] {chunks_text[i]}" for i in ids)
    spec = "\n".join(f"- slot {s['slot']}: use evidence {' and '.join(f'[{num[i]}]' for i in s['chunks'])}; objective: {s['objective']}" for s in slots)
    avoid_b = ("\n\nDo not repeat or rephrase these existing questions:\n" + "\n".join(f"- {a}" for a in avoid)) if avoid else ""
    return ("<s>[INST] You write multiple-choice quiz questions as strict JSON. Base every question ONLY on its assigned "
            "evidence; no outside knowledge. Keep scenario qualifiers exact (teaching vs ResearchGrid deployment, classroom vs "
            "deployed values). Four distinct options, exactly one correct. In each explanation cite the unit labels that "
            "support the answer (e.g. S2-P06, S3-T02).\n\n"
            f"<evidence>\n{ev}\n</evidence>\n\nWrite exactly {len(slots)} questions, one per slot:\n{spec}{avoid_b}\n\n"
            'Output ONLY: {"questions": [{"slot": <n>, "question": "...", "options": ["...", "...", "...", "..."], '
            '"correct_index": <0-3>, "explanation": "...", "evidence_refs": [<evidence numbers>]}]} [/INST]'), num


def fill_slots(gen_lm, slots, chunks_text):
    from pipeline.quiz import _repair_json_text, _normalize_keys, _validate_question, _coerce_ref
    from sentence_transformers import util
    accepted, rows = {}, []
    stats = {"batches": 0, "invalid_items": 0, "repairs": 0, "parse_failures": 0, "first_result_seconds": None}
    t0 = time.perf_counter()
    pending = [slots[i:i + 5] for i in range(0, len(slots), 5)]
    attempts = {s["slot"]: 0 for s in slots}
    while pending:
        group = pending.pop(0)
        for s in group:
            attempts[s["slot"]] += 1
        avoid = [accepted[k]["question"] for k in sorted(accepted)]
        prompt, num = quiz_prompt(group, chunks_text, avoid)
        raw = gen_lm.generate(prompt, 230 * len(group) + 40, avoid_texts=avoid, in_answer_guard=False)
        stats["batches"] += 1
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            items = _normalize_keys(json.loads(_repair_json_text(m.group(0)))).get("questions", [])
        except Exception:
            stats["parse_failures"] += 1
            items = []
        by_slot = {}
        for idx, q in enumerate(items):
            sl = q.get("slot") if isinstance(q, dict) else None
            if not (isinstance(sl, int) and any(s["slot"] == sl for s in group)):
                sl = group[idx]["slot"] if idx < len(group) else None
            if sl is not None:
                by_slot.setdefault(sl, q)
        failed = []
        for s in group:
            q = by_slot.get(s["slot"])
            errs = ["missing"] if q is None else _validate_question(q, require_evidence=True)
            if not errs:
                refs = {_coerce_ref(r) for r in q.get("evidence_refs") or []}
                if not {num[i] for i in s["chunks"]} <= refs:
                    errs = ["evidence_refs do not cover the assigned passage(s)"]
            if not errs and rows:
                e = emb().encode([str(q["question"])])
                if float(util.cos_sim(e, np.array(rows)).max()) >= 0.9:
                    errs = ["semantic duplicate"]
            if errs:
                stats["invalid_items"] += 1; failed.append(s); continue
            accepted[s["slot"]] = {"slot": s["slot"], "question": str(q["question"]).strip(),
                                   "options": [str(o).strip() for o in q["options"]], "correct_index": q["correct_index"],
                                   "explanation": str(q.get("explanation", "")).strip(), "evidence_chunks": s["chunks"],
                                   "sections": s["sections"]}
            rows.append(emb().encode([accepted[s["slot"]]["question"]])[0])
            if stats["first_result_seconds"] is None:
                stats["first_result_seconds"] = round(time.perf_counter() - t0, 1)
        for s in failed:
            if attempts[s["slot"]] <= 2:
                stats["repairs"] += 1; pending.append([s])
    return [accepted[k] for k in sorted(accepted)], stats


def score_quiz(q, items, chunk_units, chunks_text):
    from sentence_transformers import util
    c = q["quiz_constraints"]
    out = {"requested": c["requested_question_count"], "n_items": len(items)}
    out["count_compliant"] = len(items) == c["requested_question_count"]
    if not items:
        return out
    E = emb().encode([it["question"] for it in items])
    S = util.cos_sim(E, E)
    groups, seen = [], set()
    for i in range(len(items)):
        if i in seen:
            continue
        g = [j for j in range(i, len(items)) if j == i or float(S[i][j]) >= 0.85]
        seen |= set(g); groups.append(g)
    out["extra_duplicates"] = sum(len(g) - 1 for g in groups)
    out["duplicates_within_allowance"] = out["extra_duplicates"] <= c["maximum_acceptable_semantic_duplication"]
    used_units = [set(u for i in it["evidence_chunks"] for u in chunk_units[i]) for it in items]
    passages = {u for us in used_units for u in us if u[3] == "P"}
    out["distinct_passages_used"] = len(passages)
    out["min_distinct_concepts"] = c["min_distinct_concepts"]
    out["sections_covered"] = sorted({u[:2] for us in used_units for u in us})
    exp_secs = set(c["expected_topic_or_section_coverage"])
    out["section_coverage"] = round(len(exp_secs & set(out["sections_covered"])) / len(exp_secs), 3) if exp_secs else None
    cand = set(c["evidence_coverage_requirements"]["candidate_evidence_passage_ids"])
    out["candidate_evidence_hit"] = f"{len(cand & {u for us in used_units for u in us})}/{len(cand)}"
    cited_ok, lex = 0, 0
    for it, us in zip(items, used_units):
        cited = set(ID_RE.findall(it["explanation"]))
        cited_ok += bool(cited & us)
        ev = norm(" ".join(chunks_text[i] for i in it["evidence_chunks"]))
        words = [w for w in re.findall(r"[a-z0-9().+*/^%=-]+", norm(it["options"][it["correct_index"]])) if len(w) > 2 or any(ch.isdigit() for ch in w)]
        lex += bool(words) and sum(w in ev for w in words) / len(words) >= 0.5
    out["items_citing_valid_unit_ids"] = cited_ok
    out["lexically_supported"] = lex
    out["multi_section_items"] = sum(len(it["sections"]) > 1 for it in items)
    return out


def run_quizzes(lm, gen_lm, svc, doc_id, session):
    import core.assistant_service as AS
    import pipeline.quiz as pq
    from pipeline.quiz_intent import detect_quiz_intent
    gt = ground_truth()
    qz = [q for q in gt["queries"] if q["category"] == "quiz_generation"]
    ch, units, _ = chunk_store("PROD")
    text = [c["text"] for c in ch]
    res, private = {}, {}
    for q in qz:
        rq, rec, priv = parse_request(q["query_text"]), {}, {}
        # live production routing
        intent = detect_quiz_intent(q["query_text"], has_attached_document=True)
        live = {"intent_recognised": intent.activity == "quiz_create"}
        if not live["intent_recognised"]:
            AS.generate_streaming = lm.production_streaming_shim()
            try:
                final = ""
                for partial, *_ in svc.document_chat(q["query_text"], [doc_id], session):
                    final = partial
            finally:
                import pipeline.answer as PA
                AS.generate_streaming = PA.generate_streaming
            live["production_returns"] = "document_chat prose (no quiz)"
            priv["live"] = final
        rec["live"] = live
        # QP: production generator if it had been routed
        hits = dense("PROD", q["query_text"], 6)
        evmap = {i + 1: f"chunk{c}" for i, (c, _) in enumerate(hits)}
        b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens); t = time.perf_counter()
        orig = pq.generate_streaming
        pq.generate_streaming = lm.production_streaming_shim()
        try:
            quiz, errors = pq.generate_quiz(rq["topic"], q["quiz_constraints"]["requested_question_count"], lm.tok, lm.model,
                                            "QZCS", context_chunks=[text[c] for c, _ in hits], evidence_map=evmap, require_evidence=True)
        finally:
            pq.generate_streaming = orig
        items = []
        for it in (quiz or {}).get("questions", []):
            items.append({**it, "evidence_chunks": [int(e[5:]) for e in it.get("evidence_ids", [])] or [hits[0][0]],
                          "sections": sorted({section_of_units(units[int(e[5:])]) for e in it.get("evidence_ids", [])} - {None})})
        rec["QP"] = {"valid": quiz is not None, "errors": errors, "seconds": round(time.perf_counter() - t, 1),
                     "calls": lm.log.calls - b[0], "input_tokens": lm.log.input_tokens - b[1], "output_tokens": lm.log.output_tokens - b[2],
                     "truncated_prompts": None, **score_quiz(q, items, units, text)}
        priv["QP"] = items
        # QI (+ ablation without check for topic prompts)
        arms = [("QI", True)] + ([("QI_nochk", False)] if "topic" in rq["modes"] else [])
        for arm, chk in arms:
            b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens); t = time.perf_counter()
            slots, note, check_log = build_blueprint(rq, lm, chk)
            check_calls = lm.log.calls - b[0]
            items, stats = fill_slots(gen_lm, slots, text) if slots else ([], {})
            rec[arm] = {"parsed": rq, "planned_slots": len(slots), "note": note, "declined": not slots,
                        "sufficiency_check_calls": check_calls if chk else 0, "seconds": round(time.perf_counter() - t, 1),
                        "calls": lm.log.calls - b[0], "input_tokens": lm.log.input_tokens - b[1], "output_tokens": lm.log.output_tokens - b[2],
                        **stats, **score_quiz(q, items, units, text)}
            priv[arm] = {"slots": slots, "check_log": check_log, "items": items}
        res[q["query_id"]], private[q["query_id"]] = rec, priv
        print(q["query_id"], json.dumps({a: {k: rec[a].get(k) for k in ("n_items", "declined", "count_compliant", "extra_duplicates",
                                                                        "distinct_passages_used", "section_coverage", "lexically_supported",
                                                                        "items_citing_valid_unit_ids", "seconds")} for a in ("QP", "QI", "QI_nochk") if a in rec},
                                        ensure_ascii=False), "| live intent:", live["intent_recognised"], flush=True)
    return res, private


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="answers,quizzes")
    a = ap.parse_args()
    from experiments.quiz_bench import gen_harness as H
    from experiments.quiz_bench.structured import OutlinesLM
    from core.assistant_service import AssistantService
    from core.runtime_manager import RuntimeManager
    lm = H.MistralLM()
    gen_lm = OutlinesLM(lm)
    svc = AssistantService(RuntimeManager(embedder=emb(), llm_tok=lm.tok, llm_model=lm.model))
    session = "eval-cs451"
    doc_id = svc.ingest_document("AUM_RAG_BENCHMARK_COURSE_DOCUMENT.pdf", pdf_bytes(), session)["document_id"]
    out = {"load_seconds": lm.load_seconds}
    private = {}
    if "answers" in a.parts:
        out["answers_summary"], private["answers"] = run_answers(lm, svc, doc_id, session)
        out["answers_per_query"] = {arm: {k: {kk: vv for kk, vv in v.items() if kk != "answer"} for k, v in recs.items()}
                                    for arm, recs in private["answers"].items()}
        print(json.dumps(out["answers_summary"], indent=1), flush=True)
    if "quizzes" in a.parts:
        out["quizzes"], private["quizzes"] = run_quizzes(lm, gen_lm, svc, doc_id, session)
    out["gpu_peak_gib"] = H.gpu_peak_gib()
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump(private, open(a.out.replace(".json", "_outputs.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
