"""Retrieval-level evaluation against the gold anchors.

    python -m experiments.quiz_bench.retrieval_eval --chunker structured400 --out report.json

Relevance of a chunk to a gold concept = the chunk contains the concept's
anchor phrase. "Visible" = the anchor lies inside the part of the chunk the
quiz prompt actually shows the model (production: first 500 chars). For a
whole-document quiz the production query is literally the intent topic
("this document"); coverage = how many distinct gold concepts are visible in
the evidence handed to generation.
"""

import argparse
import json
import math
import time
from statistics import median

import numpy as np

from experiments.quiz_bench import chunkers as chunker_registry
from experiments.quiz_bench.fixture import gold, norm, production_pages

PROMPT_VIEW_CHARS = 500   # pipeline/quiz.py _MAX_CONTEXT_CHARS_PER_CHUNK
QUIZ_TOP_K = 6            # core/assistant_service.py quiz(): hits[:6]


class DenseRetriever:
    """Production-equivalent dense retrieval: normalised embeddings, inner product."""

    def __init__(self, embedder, texts: list[str]):
        self.embedder = embedder
        t = time.perf_counter()
        self.matrix = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        self.index_seconds = time.perf_counter() - t

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


def relevant(chunks: list[dict], anchors: list[str], view: int | None = None) -> set[int]:
    out = set()
    for i, c in enumerate(chunks):
        text = norm(c["text"][:view] if view else c["text"])
        if all(norm(a) in text for a in anchors):
            out.add(i)
    return out


def ranking_metrics(ranked: list[int], rel: set[int]) -> dict:
    first = next((r for r, i in enumerate(ranked[:10], 1) if i in rel), None)
    dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(ranked[:10], 1) if i in rel)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(rel), 10) + 1))
    return {"hit@1": float(bool(ranked[:1]) and ranked[0] in rel),
            "recall@5": len(rel & set(ranked[:5])) / len(rel), "recall@10": len(rel & set(ranked[:10])) / len(rel),
            "mrr@10": 1 / first if first else 0.0, "ndcg@10": dcg / idcg if idcg else 0.0}


def duplicate_ratio(texts: list[str]) -> float:
    """Share of 8-word shingles in the evidence set that occur more than once
    (measures overlap-induced duplication, e.g. from chunk overlap)."""
    seen, dup, total = {}, 0, 0
    for t in texts:
        words = norm(t).split()
        for s in {" ".join(words[i:i + 8]) for i in range(max(0, len(words) - 7))}:
            seen[s] = seen.get(s, 0) + 1
    total = sum(seen.values())
    dup = sum(v for v in seen.values() if v > 1)
    return dup / total if total else 0.0


def evaluate(chunks: list[dict], embedder, view_chars: int | None = PROMPT_VIEW_CHARS, tokenizer=None) -> dict:
    g = gold()
    retriever = DenseRetriever(embedder, [c["text"] for c in chunks])
    per_query, lat = [], []
    for c in g["concepts"]:
        rel = relevant(chunks, c["anchors"])
        t = time.perf_counter()
        ranked = [i for i, _ in retriever.search(c["query"], 10)]
        lat.append((time.perf_counter() - t) * 1000)
        m = ranking_metrics(ranked, rel) if rel else {"hit@1": 0, "recall@5": 0, "recall@10": 0, "mrr@10": 0, "ndcg@10": 0}
        vis = relevant(chunks, c["anchors"], view_chars)
        m["visible_in_top6"] = float(bool(vis & set(ranked[:QUIZ_TOP_K])))
        m["anchor_chunks"] = sorted(rel)
        per_query.append({"id": c["id"], **m})
    agg = {k: sum(q[k] for q in per_query) / len(per_query)
           for k in ("hit@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "visible_in_top6")}
    # quiz-request evidence: what the production quiz path would hand to generation
    requests = {}
    for key, req in g["quiz_requests"].items():
        from pipeline.quiz_intent import detect_quiz_intent
        topic = detect_quiz_intent(req["prompt"], has_attached_document=True).topic
        hits = retriever.search(topic, QUIZ_TOP_K)
        ids = [i for i, _ in hits]
        shown = [chunks[i]["text"][:view_chars] if view_chars else chunks[i]["text"] for i in ids]
        covered = sorted(c["id"] for c in g["concepts"] if relevant([{"text": s} for s in shown], c["anchors"]))
        target = req.get("concepts")
        requests[key] = {
            "query": topic, "chunk_ids": ids, "scores": [round(s, 3) for _, s in hits],
            "sections": sorted({" > ".join(chunks[i]["heading_path"]) or "(none)" for i in ids}),
            "pages": sorted({chunks[i]["page"] for i in ids}),
            "visible_concepts": covered, "n_visible_concepts": len(covered),
            "target_concepts_visible": (f"{len(set(covered) & set(target))}/{len(target)}" if target else None),
            "evidence_chars_retrieved": sum(len(chunks[i]["text"]) for i in ids),
            "evidence_chars_shown": sum(len(s) for s in shown),
            "evidence_tokens_shown": (sum(len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in shown) if tokenizer else None),
            "reference_chunks": sum("references" in norm(" ".join(chunks[i]["heading_path"])) for i in ids),
            "duplicate_ratio": round(duplicate_ratio(shown), 3),
        }
    sizes = [len(c["text"]) for c in chunks]
    tok = [len(tokenizer(c["text"], add_special_tokens=False)["input_ids"]) for c in chunks] if tokenizer else None
    return {
        "n_chunks": len(chunks), "chunk_chars": {"min": min(sizes), "median": median(sizes), "max": max(sizes)},
        "chunk_tokens": ({"min": min(tok), "median": median(tok), "max": max(tok), "total": sum(tok)} if tok else None),
        "chunks_exceeding_view": sum(s > view_chars for s in sizes) if view_chars else 0,
        "concepts_with_anchor_chunk": sum(bool(q["anchor_chunks"]) for q in per_query),
        "concepts_visible_anywhere": sum(bool(relevant(chunks, c["anchors"], view_chars)) for c in g["concepts"]),
        "index_seconds": round(retriever.index_seconds, 3),
        "query_ms_p50": round(median(lat), 2),
        "retrieval": agg, "per_query": per_query, "quiz_requests": requests,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="structured400", choices=sorted(chunker_registry.CHUNKERS))
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--view", type=int, default=PROMPT_VIEW_CHARS, help="chars of each chunk shown to the model; 0 = whole chunk")
    ap.add_argument("--out")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import config
    chunks = chunker_registry.CHUNKERS[a.chunker](production_pages())
    report = evaluate(chunks, SentenceTransformer(a.embedder, device="cpu"), a.view or None,
                      AutoTokenizer.from_pretrained(config.LLM_ID))
    report.update(chunker=a.chunker, embedder=a.embedder, view_chars=a.view or None)
    text = json.dumps(report, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    print(json.dumps({k: report[k] for k in ("chunker", "n_chunks", "chunk_tokens", "chunks_exceeding_view",
                                             "concepts_visible_anywhere", "retrieval")}, indent=1))
    for k, r in report["quiz_requests"].items():
        print(k, r["query"][:40], "| visible concepts", r["n_visible_concepts"], "| target", r["target_concepts_visible"],
              "| pages", r["pages"], "| refs", r["reference_chunks"], "| tokens shown", r["evidence_tokens_shown"])


if __name__ == "__main__":
    main()
