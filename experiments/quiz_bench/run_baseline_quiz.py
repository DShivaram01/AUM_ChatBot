"""Phase 1: run quiz requests A-F through the LIVE production pipeline.

    python -m experiments.quiz_bench.run_baseline_quiz --server-log logs/session_<start>.log \
        --out experiments/quiz_bench/results/baseline_quiz_metrics.json

Uploads the private benchmark paper under an eval-qz-* session, sends each
request as a natural-language chat message (the only production quiz path),
and records: validity, question count, model calls/retries and per-call input
tokens + truncation warnings (from the server log window), the chunk IDs each
question cites (mapped to sections/pages via a deterministic offline ingest),
lexical evidence support, semantic duplication and latency. Full generated
questions are written only to $QZ_BENCH_DIR (they quote the paper); the
committed --out file holds metrics and IDs only.
"""

import argparse
import json
import re
import statistics
import threading
import time
import uuid
from pathlib import Path

import requests

from experiments.quiz_bench.fixture import bench_dir, gold, norm, paper_bytes

CALL_RE = re.compile(r"\[(QZ\d+)-q(\d+)a(\d+)\] (Input tokens: (\d+)|== LLM PROMPT == len=(\d+))")


def log_lines(path):
    return Path(path).read_text(errors="replace").splitlines() if path else []


def gpu_sampler(stop, samples):
    import torch
    while not stop.is_set():
        free, total = torch.cuda.mem_get_info()
        samples.append((total - free) / 2**30)
        time.sleep(0.5)


def window_stats(lines):
    qids = sorted({m.group(1) for l in lines if (m := CALL_RE.search(l))})
    calls = [m for l in lines if (m := CALL_RE.search(l)) and m.group(5)]
    toks = [int(m.group(5)) for m in calls]
    warns = [l for l in lines if "] Quiz q" in l and "attempt" in l]
    return {"quiz_ids": qids, "model_calls": len(toks), "retries": sum(int(m.group(3)) > 1 for m in calls),
            "input_tokens_per_call": {"min": min(toks), "median": statistics.median(toks), "max": max(toks)} if toks else None,
            "input_tokens_total": sum(toks),
            "truncation_warnings": sum("Prompt truncated" in l for l in lines),
            "failures": {"duplicate": sum("duplicate" in l for l in warns), "evidence": sum("evidence_refs" in l for l in warns),
                         "parse": sum("unparseable" in l for l in warns), "other": sum(not any(k in l for k in ("duplicate", "evidence_refs", "unparseable")) for l in warns)},
            "generation_failed": any("Quiz generation failed" in l for l in lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated request keys, e.g. A,D")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer, util
    import config
    from core.document_service import DocumentService

    g = gold()
    data = paper_bytes()
    session = f"eval-qz-{uuid.uuid4().hex[:6]}"
    r = requests.post(f"{a.server}/api/documents", data={"session_id": session},
                      files={"file": ("benchmark_paper.pdf", data, "application/pdf")}, timeout=300)
    r.raise_for_status()
    doc_id = r.json()["document_id"]
    emb = SentenceTransformer(config.EMBED_MODEL_ID, device="cpu")
    offline = DocumentService(emb)
    offline.ingest("benchmark_paper.pdf", data, session)   # deterministic: same chunk ids as the server
    chunks = {c.chunk_id: c for c in offline.store.get(doc_id, session).chunks}

    metrics, private = {"session": session, "document_id": doc_id, "n_chunks": len(chunks), "requests": {}}, {}
    keys = a.only.split(",") if a.only else list(g["quiz_requests"])
    for key in keys:
        req = g["quiz_requests"][key]
        offset = len(log_lines(a.server_log))
        stop, samples = threading.Event(), []
        sampler = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
        sampler.start()
        t = time.perf_counter()
        body = requests.post(f"{a.server}/api/ask", json={"question": req["prompt"], "session_id": session,
                                                           "document_ids": [doc_id]}, timeout=3600).json()
        latency = time.perf_counter() - t
        stop.set(); sampler.join()
        time.sleep(0.5)
        rec = {"prompt": req["prompt"], "scope": req["scope"], "latency_s": round(latency, 1), "kind": body.get("kind"),
               "topic_used": body.get("topic_used"), "valid": body.get("kind") == "quiz" and bool(body.get("quiz")),
               "gpu_used_gib_max": round(max(samples), 2) if samples else None,
               **window_stats(log_lines(a.server_log)[offset:])}
        requested = int(re.search(r"(\d+)", req["prompt"]).group(1))
        rec["requested"] = requested
        if rec["valid"]:
            qs = body["quiz"]["questions"]
            rec["n_questions"] = len(qs)
            texts = [q["question"] for q in qs]
            sims = util.cos_sim(emb.encode(texts), emb.encode(texts))
            rec["near_duplicate_pairs"] = sum(float(sims[i][j]) >= 0.85 for i in range(len(qs)) for j in range(i + 1, len(qs)))
            answers = [norm(q["options"][q["correct_index"]]) for q in qs]
            rec["repeated_correct_answers"] = len(answers) - len(set(answers))
            cited = [e for q in qs for e in q.get("evidence_ids", [])]
            rec["distinct_cited_chunks"] = len(set(cited))
            rec["cited_pages"] = sorted({chunks[e].page for e in cited if e in chunks})
            rec["cited_sections"] = sorted({" > ".join(chunks[e].heading_path) or "(none)" for e in cited if e in chunks})
            rec["cites_references_section"] = sum("references" in norm(" ".join(chunks[e].heading_path)) for e in set(cited) if e in chunks)
            supported, gold_hits = 0, set()
            for q in qs:
                ev = " ".join(chunks[e].text for e in q.get("evidence_ids", []) if e in chunks)
                words = [w for w in re.findall(r"[a-z0-9.$%]+", norm(q["options"][q["correct_index"]])) if len(w) > 2 or w.isdigit()]
                supported += bool(ev) and bool(words) and sum(w in norm(ev) for w in words) / len(words) >= 0.5
                for c in g["concepts"]:
                    if any(norm(an) in norm(ev) for an in c["anchors"]) and any(w in norm(q["question"] + " " + q["options"][q["correct_index"]]) for w in norm(c["answer"]).split() if len(w) > 3):
                        gold_hits.add(c["id"])
            rec["lexically_supported_answers"] = supported
            rec["gold_concepts_touched"] = sorted(gold_hits)
            private[key] = {"questions": qs}
        else:
            rec["answer"] = body.get("answer", "")[:200]
        metrics["requests"][key] = rec
        print(key, json.dumps({k: rec.get(k) for k in ("valid", "requested", "n_questions", "latency_s", "model_calls", "retries",
                                                        "truncation_warnings", "near_duplicate_pairs", "distinct_cited_chunks",
                                                        "cited_pages", "lexically_supported_answers", "failures")}), flush=True)
    Path(a.out).write_text(json.dumps(metrics, indent=1))
    out_private = bench_dir() / f"baseline_quiz_outputs_{session}.json"
    out_private.write_text(json.dumps(private, indent=1))
    out_private.chmod(0o600)
    print("metrics ->", a.out, "| full outputs (private) ->", out_private)


if __name__ == "__main__":
    main()
