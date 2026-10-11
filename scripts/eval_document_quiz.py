"""Task 51: measure Document QA and Quiz Mode against a LIVE server.

    python scripts/eval_document_quiz.py --server http://127.0.0.1:8000 \
        --server-log logs/session_<server start>.log --out results.json

Uses tests/fixtures/document_qa_eval.json (fictional PDFs built in memory,
so the model cannot answer from pretrained knowledge). Sessions are named
eval-t51-* so this traffic is identifiable in the server's logs/events.

Retrieval scores are not returned by the API, so they are recomputed
offline with the same embedder and the same DocumentService (ingest is
deterministic: same session + bytes -> same chunk ids), which also lets
quiz evidence_ids be checked against the real chunk text.
"""

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

import pymupdf
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "document_qa_eval.json"
ABSTAIN_RE = re.compile(
    r"could not find enough information|does not (?:mention|say|state|specify|provide|contain|include)|"
    r"(?:is|are) not (?:explicitly )?(?:mentioned|stated|specified|provided|included)|"
    r"no (?:specific |explicit )?(?:information|mention|details)|(?:remains?|is) unspecified|not (?:found|available) in",
    re.IGNORECASE,
)
PAGE_RE = re.compile(r"p\.(\d+)(?:-(\d+))?")


def build_pdf(pages: list[str]) -> bytes:
    pdf = pymupdf.open()
    for text in pages:
        pdf.new_page().insert_textbox(pymupdf.Rect(72, 72, 540, 760), text, fontsize=11)
    data = pdf.tobytes()
    pdf.close()
    return data


def cited_pages(answer: str) -> set[int]:
    pages = set()
    for a, b in PAGE_RE.findall(answer):
        pages.update(range(int(a), int(b or a) + 1))
    return pages


def has_keywords(answer: str, keywords: list[str]) -> bool:
    return all(re.search(rf"(?<![\w.]){alt_group(k)}(?![\w])", answer, re.IGNORECASE) for k in keywords)


def alt_group(k: str) -> str:
    return "(?:" + "|".join(re.escape(x) for x in k.split("|")) + ")"


def ask(server, payload, timeout=300):
    t = time.perf_counter()
    r = requests.post(f"{server}/api/ask", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t) * 1000


def run_docqa(server, spec, run_id):
    from sentence_transformers import SentenceTransformer

    import config
    from core.document_service import DocumentService

    offline = DocumentService(SentenceTransformer(config.EMBED_MODEL_ID, device="cpu"))
    rows, uploaded, isolation = [], {}, []
    for doc in spec["documents"]:
        data = build_pdf(doc["pages"])
        session = f"eval-t51-{run_id}-{doc['id']}"
        r = requests.post(f"{server}/api/documents", data={"session_id": session},
                          files={"file": (doc["filename"], data, "application/pdf")}, timeout=120)
        r.raise_for_status()
        doc_id = r.json()["document_id"]
        offline.ingest(doc["filename"], data, session)
        uploaded[doc["id"]] = (doc_id, session, data)

        # adversarial cross-session isolation: another session holding the id
        other = f"eval-t51-{run_id}-intruder"
        got = requests.get(f"{server}/api/documents/{doc_id}", params={"session_id": other}, timeout=30)
        body, _ = ask(server, {"question": doc["questions"][0]["q"], "session_id": other, "document_ids": [doc_id]})
        leaked = any(k.lower() in body["answer"].lower() for k in doc["questions"][0]["keywords"])
        isolation.append({"doc": doc["id"], "metadata_status": got.status_code, "chat_topic": body["topic_used"],
                          "chat_answer_leaked_fact": leaked, "chat_answer": body["answer"][:160]})

        for q in doc["questions"]:
            body, ms = ask(server, {"question": q["q"], "session_id": session, "document_ids": [doc_id]})
            answer = body["answer"]
            hits = offline.retrieve(doc_id, session, q["q"], limit=5)
            abstained = bool(ABSTAIN_RE.search(answer))
            gate = "could not find enough information" in answer
            row = {"doc": doc["id"], "q": q["q"], "expect": q["expect"], "topic_used": body["topic_used"],
                   "answer": answer, "latency_ms": ms, "abstained": abstained, "gate_abstained": gate,
                   "top_score": hits[0]["score"] if hits else None,
                   "top_page": hits[0]["page"] if hits else None}
            if q["expect"] == "answer":
                row["correct"] = (not abstained) and has_keywords(answer, q["keywords"])
                pages = cited_pages(answer)
                row["cited_pages"] = sorted(pages)
                row["citation_ok"] = q["page"] in pages
                row["citation_exact"] = pages == {q["page"]}
                row["retrieval_top1_ok"] = row["top_page"] == q["page"]
            else:
                row["correct"] = abstained
            rows.append(row)
            print(f"  [{doc['id']}] {q['expect']:7s} ok={row['correct']!s:5s} top={row['top_score']:.3f} {q['q'][:50]}",
                  flush=True)
    return rows, isolation, uploaded, offline


ATTEMPT_RE = re.compile(r"\[(QZ\d+)-q(\d+)a(\d+)\] == LLM PROMPT")


def log_lines(log_path):
    return Path(log_path).read_text(errors="replace").splitlines() if log_path else []


def quiz_attempts(lines):
    """Attempt/retry stats for the ONE quiz whose log lines these are. The
    chat-quiz API returns query_id="" so quizzes are matched by log window
    (the eval is sequential), not by id."""
    qids = sorted({m.group(1) for l in lines if (m := ATTEMPT_RE.search(l))})
    if len(qids) != 1:
        return {"error": f"expected 1 quiz id in window, found {qids}"} if lines else None
    qid = qids[0]
    attempts = [(int(m.group(2)), int(m.group(3))) for l in lines if (m := ATTEMPT_RE.search(l))]
    warns = [l for l in lines if f"[{qid}] Quiz q" in l]
    return {"query_id": qid, "model_calls": len(attempts),
            "retries": sum(a > 1 for _, a in attempts),
            "failed_attempts": len(warns),
            "duplicate_failures": sum("duplicate" in l for l in warns),
            "evidence_failures": sum("evidence_refs" in l for l in warns),
            "parse_failures": sum("unparseable" in l for l in warns)}


def run_quizzes(server, spec, uploaded, offline, log_path, run_id):
    from sentence_transformers import SentenceTransformer, util

    import config

    emb = SentenceTransformer(config.EMBED_MODEL_ID, device="cpu")
    out = []
    for i, qz in enumerate(spec["quizzes"]):
        payload = {"question": qz["prompt"], "topic": "general" if qz.get("open_ended") else "auto"}
        chunk_text = {}
        if qz["doc"]:
            doc_id, session, _ = uploaded[qz["doc"]]
            payload.update(session_id=session, document_ids=[doc_id])
            chunk_text = {c.chunk_id: c.text for c in offline.store.get(doc_id, session).chunks}
        else:
            payload["session_id"] = f"eval-t51-{run_id}-quiz{i}"
        offset = len(log_lines(log_path))
        body, ms = ask(server, payload, timeout=900)
        rec = {"prompt": qz["prompt"], "doc": qz["doc"], "breadth": qz["breadth"], "kind": body["kind"],
               "source_mode": body.get("source_mode"), "query_id": body["query_id"], "latency_ms": ms,
               "valid": body["kind"] == "quiz" and bool(body.get("quiz")), "answer": body["answer"][:200]}
        time.sleep(0.5)
        rec["attempts"] = quiz_attempts(log_lines(log_path)[offset:])
        if rec["valid"]:
            qs = body["quiz"]["questions"]
            rec["n_questions"] = len(qs)
            texts = [q["question"] for q in qs]
            sims = util.cos_sim(emb.encode(texts), emb.encode(texts))
            rec["near_duplicate_pairs"] = sum(float(sims[a][b]) >= 0.85 for a in range(len(qs)) for b in range(a + 1, len(qs)))
            answers = [q["options"][q["correct_index"]].strip().lower() for q in qs]
            rec["repeated_correct_answers"] = len(answers) - len(set(answers))
            if chunk_text:
                grounded = 0
                for q in qs:
                    ev = " ".join(chunk_text.get(e, "") for e in q.get("evidence_ids", [])).lower()
                    words = [w for w in re.findall(r"[a-z0-9.$%]+", q["options"][q["correct_index"]].lower()) if len(w) > 2 or w.isdigit()]
                    grounded += bool(ev) and bool(words) and sum(w in ev for w in words) / len(words) >= 0.5
                rec["evidence_supports_answer"] = grounded
                rec["questions_with_evidence_ids"] = sum(bool(q.get("evidence_ids")) for q in qs)
            rec["questions"] = [{"q": q["question"], "answer": q["options"][q["correct_index"]],
                                 "evidence_ids": q.get("evidence_ids", [])} for q in qs]
        print(f"  quiz[{qz['breadth']:6s}] {qz['prompt'][:45]:45s} valid={rec['valid']} attempts={rec['attempts']}",
              flush=True)
        out.append(rec)
    return out


def summarize(rows, isolation, quizzes, floor_now):
    ans = [r for r in rows if r["expect"] == "answer"]
    abst = [r for r in rows if r["expect"] == "abstain"]
    s = {
        "docqa": {
            "n_answerable": len(ans), "n_unanswerable": len(abst),
            "answer_accuracy": sum(r["correct"] for r in ans) / len(ans),
            "false_abstention_rate": sum(r["abstained"] for r in ans) / len(ans),
            "citation_contains_expected_page": sum(r["citation_ok"] for r in ans) / len(ans),
            "citation_exact_page": sum(r["citation_exact"] for r in ans) / len(ans),
            "retrieval_top1_page": sum(r["retrieval_top1_ok"] for r in ans) / len(ans),
            "abstention_correct": sum(r["correct"] for r in abst) / len(abst),
            "abstained_by_gate": sum(r["gate_abstained"] for r in abst),
            "abstained_by_model": sum(r["abstained"] and not r["gate_abstained"] for r in abst),
            "latency_ms_median": sorted(r["latency_ms"] for r in rows)[len(rows) // 2],
        },
        "isolation": {"all_blocked": all(i["metadata_status"] in (403, 404) and not i["chat_answer_leaked_fact"] for i in isolation),
                      "detail": isolation},
    }
    # evidence-floor sweep (offline top scores): gate abstains iff top < floor
    sweep = []
    for f in [0.20, 0.25, 0.30, floor_now, 0.40, 0.45, 0.50]:
        sweep.append({"floor": f,
                      "answerable_blocked": sum((r["top_score"] or 0) < f for r in ans),
                      "unanswerable_blocked": sum((r["top_score"] or 0) < f for r in abst)})
    s["docqa"]["floor_sweep"] = sweep
    s["docqa"]["top_score_answerable"] = sorted(round(r["top_score"], 3) for r in ans)
    s["docqa"]["top_score_unanswerable"] = sorted(round(r["top_score"], 3) for r in abst)

    if not quizzes:
        return s
    valid = [q for q in quizzes if q["valid"]]
    att = [q["attempts"] for q in quizzes if q["attempts"] and "error" not in q["attempts"]]
    total_q = sum(q.get("n_questions", 0) for q in valid)
    s["quiz"] = {
        "n_quizzes": len(quizzes), "validity_rate": len(valid) / len(quizzes),
        "questions_generated": total_q,
        "model_calls": sum(a["model_calls"] for a in att),
        "retries_total": sum(a["retries"] for a in att),
        "retry_rate_per_question": sum(a["retries"] for a in att) / max(1, sum(a["model_calls"] - a["retries"] for a in att)),
        "failed_attempts_by_cause": {k: sum(a[k] for a in att) for k in ("duplicate_failures", "evidence_failures", "parse_failures")},
        "near_duplicate_pairs": sum(q["near_duplicate_pairs"] for q in valid),
        "repeated_correct_answers": sum(q["repeated_correct_answers"] for q in valid),
        "document_evidence_supports_answer": f"{sum(q.get('evidence_supports_answer', 0) for q in valid)}/"
                                             f"{sum(q['n_questions'] for q in valid if q['doc'])}",
        "by_breadth": {b: {"valid": f"{sum(q['valid'] for q in quizzes if q['breadth'] == b)}/{sum(q['breadth'] == b for q in quizzes)}",
                           "retries": sum((q["attempts"] or {}).get("retries", 0) for q in quizzes if q["breadth"] == b),
                           "duplicate_failures": sum((q["attempts"] or {}).get("duplicate_failures", 0) for q in quizzes if q["breadth"] == b)}
                       for b in ("broad", "narrow")},
        "latency_s_median": sorted(q["latency_ms"] for q in quizzes)[len(quizzes) // 2] / 1000,
    }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--server-log", help="the live server's session log (for quiz retry counts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-quiz", action="store_true")
    ap.add_argument("--rescore", help="re-score a saved results file (no server calls)")
    a = ap.parse_args()

    import config
    if a.rescore:
        saved = json.loads(Path(a.rescore).read_text())
        spec = {(d["id"], q["q"]): q for d in json.loads(FIXTURE.read_text())["documents"] for q in d["questions"]}
        for r in saved["docqa_rows"]:
            q = spec[(r["doc"], r["q"])]
            r["abstained"] = bool(ABSTAIN_RE.search(r["answer"]))
            r["correct"] = ((not r["abstained"]) and has_keywords(r["answer"], q["keywords"])) if q["expect"] == "answer" else r["abstained"]
        if a.server_log:
            by_qid = {}
            for l in log_lines(a.server_log):
                m = ATTEMPT_RE.search(l)
                qid = m.group(1) if m else next((q for q in by_qid if f"[{q}] Quiz q" in l), None)
                if qid:
                    by_qid.setdefault(qid, []).append(l)
            ids = sorted(by_qid, key=lambda q: int(q[2:]))[-len(saved["quizzes"]):]
            for rec, qid in zip(saved["quizzes"], ids):
                rec["attempts"] = quiz_attempts(by_qid[qid])
        saved["summary"] = summarize(saved["docqa_rows"], saved["summary"]["isolation"]["detail"], saved["quizzes"],
                                     config.DOCUMENT_EVIDENCE_SCORE_FLOOR)
        Path(a.out).write_text(json.dumps(saved, indent=1))
        print(json.dumps(saved["summary"], indent=1))
        return
    spec = json.loads(FIXTURE.read_text())
    run_id = uuid.uuid4().hex[:6]
    print("Document QA ...", flush=True)
    rows, isolation, uploaded, offline = run_docqa(a.server, spec, run_id)
    quizzes = []
    if not a.skip_quiz:
        print("Quizzes ...", flush=True)
        quizzes = run_quizzes(a.server, spec, uploaded, offline, a.server_log, run_id)
    summary = summarize(rows, isolation, quizzes, config.DOCUMENT_EVIDENCE_SCORE_FLOOR)
    Path(a.out).write_text(json.dumps({"summary": summary, "docqa_rows": rows, "quizzes": quizzes}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
