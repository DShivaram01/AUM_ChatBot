"""Experiment 5: embedding models x retrieval strategies on the benchmark paper.

Run ONE model per process (clean memory numbers), in the isolated CPU venv
~/.venvs/embbench (Qwen3/EmbeddingGemma need transformers newer than the
production pin):

    ~/.venvs/embbench/bin/python -m experiments.quiz_bench.embedding_bench \
        --model Qwen/Qwen3-Embedding-0.6B --out results/emb_qwen06.json

Chunks are fixed to the production chunker so only the retriever varies.
Each model gets its own in-memory index (never shared or cached across
models). Query/document formatting uses sentence-transformers'
encode_query / encode_document, i.e. each model's own published prompts.

Strategies: dense, BM25, hybrid (dense + BM25 by reciprocal rank fusion,
k=60), MMR (lambda 0.7 over the dense top 20). Metrics per strategy: R@5/10,
MRR@10, nDCG@10 over 30 gold concepts, coverage when whole chunks are packed
into a 1000-token budget, target coverage for topic requests D/E, and
retrieval latency.
"""

import argparse
import json
import math
import os
import re
import time
from statistics import median

import numpy as np
import psutil

from experiments.quiz_bench.fixture import gold, norm, production_pages

RRF_K = 60
MMR_LAMBDA = 0.7
BUDGET = 1000


def rss_gib():
    return psutil.Process(os.getpid()).memory_info().rss / 2**30


class Bm25:
    def __init__(self, texts):
        from rank_bm25 import BM25Okapi
        self.bm = BM25Okapi([self.tok(t) for t in texts])

    @staticmethod
    def tok(t):
        return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", norm(t))

    def rank(self, q):
        return list(np.argsort(-self.bm.get_scores(self.tok(q))))


def rrf(*rankings):
    score = {}
    for r in rankings:
        for pos, i in enumerate(r):
            score[i] = score.get(i, 0) + 1 / (RRF_K + pos + 1)
    return sorted(score, key=lambda i: -score[i])


def mmr(qv, docs, cand, k=10):
    chosen = []
    while cand and len(chosen) < k:
        best = max(cand, key=lambda i: MMR_LAMBDA * float(docs[i] @ qv)
                   - (1 - MMR_LAMBDA) * max((float(docs[i] @ docs[j]) for j in chosen), default=0.0))
        chosen.append(best); cand = [c for c in cand if c != best]
    return chosen


def metrics(ranked, rel):
    first = next((r for r, i in enumerate(ranked[:10], 1) if i in rel), None)
    dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(ranked[:10], 1) if i in rel)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(rel), 10) + 1))
    return {"recall@5": len(rel & set(ranked[:5])) / len(rel), "recall@10": len(rel & set(ranked[:10])) / len(rel),
            "mrr@10": 1 / first if first else 0.0, "ndcg@10": dcg / idcg if idcg else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from ingestion.document_parser import structured_chunks
    from pipeline.quiz_intent import detect_quiz_intent

    mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    ntok = lambda t: len(mistral(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    g = gold()
    chunks = structured_chunks(production_pages())
    texts = [c["text"] for c in chunks]

    rss0 = rss_gib()
    t = time.perf_counter()
    model = SentenceTransformer(a.model, device="cpu", trust_remote_code=False)
    load_s = time.perf_counter() - t
    rss_loaded = rss_gib()
    enc_q = getattr(model, "encode_query", model.encode)
    enc_d = getattr(model, "encode_document", model.encode)
    t = time.perf_counter()
    D = enc_d(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=8).astype(np.float32)
    index_s = time.perf_counter() - t
    rss_indexed = rss_gib()
    truncated = sum(len(model.tokenizer(x)["input_ids"]) > model.max_seq_length for x in texts)
    bm25 = Bm25(texts)

    queries = [(c["id"], c["query"], [c]) for c in g["concepts"]]
    for key in ("D", "E"):
        req = g["quiz_requests"][key]
        queries.append((key, detect_quiz_intent(req["prompt"], has_attached_document=True).topic,
                        [c for c in g["concepts"] if c["id"] in req["concepts"]]))
    rel = {c["id"]: {i for i, x in enumerate(texts) if all(norm(an) in norm(x) for an in c["anchors"])} for c in g["concepts"]}

    def packed_cover(ranked, targets):
        ev, used = [], 0
        for i in ranked:
            n = ntok(texts[i])
            if used + n > BUDGET:
                continue
            ev.append(texts[i]); used += n
        body = norm(" ".join(ev))
        return sum(all(norm(an) in body for an in t["anchors"]) for t in targets) / len(targets)

    results, lat = {}, {s: [] for s in ("dense", "bm25", "hybrid_rrf", "mmr")}
    for s in lat:
        results[s] = {"per_query": {}, "coverage@1000": {}}
    for qid, q, targets in queries:
        t0 = time.perf_counter()
        qv = enc_q([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        dense = list(np.argsort(-(D @ qv)))
        t_dense = time.perf_counter() - t0
        t0 = time.perf_counter(); lex = bm25.rank(q); t_bm = time.perf_counter() - t0
        t0 = time.perf_counter(); hyb = rrf(dense[:30], lex[:30]); t_h = time.perf_counter() - t0
        t0 = time.perf_counter(); mm = mmr(qv, D, dense[:20]) + [i for i in dense if i not in dense[:20]]; t_m = time.perf_counter() - t0
        rankings = {"dense": (dense, t_dense), "bm25": (lex, t_bm), "hybrid_rrf": (hyb, t_dense + t_bm + t_h), "mmr": (mm, t_dense + t_m)}
        for s, (r, secs) in rankings.items():
            r = [int(i) for i in r]
            if qid in rel:
                results[s]["per_query"][qid] = metrics(r, rel[qid])
            results[s]["coverage@1000"][qid] = packed_cover(r, targets)
            lat[s].append(secs * 1000)
    summary = {}
    for s, r in results.items():
        pq = r["per_query"].values()
        concept_ids = [c["id"] for c in g["concepts"]]
        summary[s] = {k: sum(x[k] for x in pq) / len(pq) for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")}
        summary[s]["coverage@1000"] = sum(r["coverage@1000"][i] for i in concept_ids) / len(concept_ids)
        summary[s]["D_target_coverage@1000"] = r["coverage@1000"]["D"]
        summary[s]["E_target_coverage@1000"] = r["coverage@1000"]["E"]
        summary[s]["query_ms_p50"] = round(median(lat[s]), 2)
        summary[s]["query_ms_p95"] = round(sorted(lat[s])[int(0.95 * (len(lat[s]) - 1))], 2)
    out = {"model": a.model, "dim": int(D.shape[1]), "max_seq_length": model.max_seq_length,
           "chunks": len(texts), "chunks_truncated_by_model": truncated,
           "load_seconds": round(load_s, 1), "index_seconds": round(index_s, 2),
           "chunks_per_second": round(len(texts) / index_s, 2),
           "rss_gib": {"start": round(rss0, 2), "loaded": round(rss_loaded, 2), "after_index": round(rss_indexed, 2),
                       "peak_process": round(psutil.Process().memory_info().rss / 2**30, 2)},
           "index_bytes_fp32": int(D.nbytes), "threads": a.threads,
           "summary": summary, "per_query": {s: r["per_query"] for s, r in results.items()},
           "coverage": {s: r["coverage@1000"] for s, r in results.items()}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.model}: dim={out['dim']} maxlen={out['max_seq_length']} trunc={truncated}/{len(texts)} load={out['load_seconds']}s "
          f"index={out['index_seconds']}s rss_loaded={out['rss_gib']['loaded']}GiB")
    for s, m in summary.items():
        print(f"  {s:11s} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} MRR={m['mrr@10']:.3f} nDCG={m['ndcg@10']:.3f} "
              f"cov@1000={m['coverage@1000']:.3f} D={m['D_target_coverage@1000']:.2f} E={m['E_target_coverage@1000']:.2f} "
              f"p50={m['query_ms_p50']}ms")


if __name__ == "__main__":
    main()
