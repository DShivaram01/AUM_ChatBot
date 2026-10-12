"""Phase A: retrieval metrics for the evaluator guide's controlled matrix.

    python -m experiments.cs451.eval_retrieval --embedder sentence-transformers/all-MiniLM-L6-v2 \
        --out experiments/cs451/results/retrieval_minilm.json

Configurations (chunker x retriever), everything else held constant:
  PROD/D, F256/F400/F512/F768 at 15 % and 0 % overlap (dense), R512/D,
  H512/D, SW/D, PC (children R256 -> parents H1024, dense on children),
  and on PROD chunks: K (BM25), HY (RRF k=60 dense+BM25), DIV (MMR 0.65).

Metrics on the 80 answerable non-quiz queries (guide section 5), top-k by
returned unit (chunk, or parent for PC):
  Recall@5/@10 (macro), strict all-evidence Recall@10, MRR@10, nDCG@10 with
  binary gains (required unit rel=2, everything else 0; stated explicitly),
  micro evidence recall@10, per-category Recall@10, distractor-in-top-5 rate
  for lexical_ambiguity, and Mistral tokens of the top-10 context.
Unanswerable queries are excluded from ranking metrics; their top dense
score is reported for the abstention analysis.
"""

import argparse
import json
import math
import re
import time
from statistics import median

import numpy as np

from experiments.cs451 import chunking as C
from experiments.cs451.units import contained_units, ground_truth, norm

RRF_K = 60
MMR_LAMBDA = 0.65


class Bm25:
    def __init__(self, texts):
        from rank_bm25 import BM25Okapi
        self.bm = BM25Okapi([self.tok(t) for t in texts])

    @staticmethod
    def tok(t):
        return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", norm(t))

    def rank(self, q):
        return [int(i) for i in np.argsort(-self.bm.get_scores(self.tok(q)))]


def embed(model, texts, query=False):
    fn = getattr(model, "encode_query" if query else "encode_document", model.encode)
    return fn(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=16).astype(np.float32)


def rrf(*rankings):
    s = {}
    for r in rankings:
        for pos, i in enumerate(r):
            s[i] = s.get(i, 0) + 1 / (RRF_K + pos + 1)
    return sorted(s, key=lambda i: -s[i])


def mmr(qv, D, cand, k=10):
    chosen = []
    while cand and len(chosen) < k:
        best = max(cand, key=lambda i: MMR_LAMBDA * float(D[i] @ qv)
                   - (1 - MMR_LAMBDA) * max((float(D[i] @ D[j]) for j in chosen), default=0.0))
        chosen.append(best); cand = [c for c in cand if c != best]
    return chosen


def score_query(ranked_units: list[list[str]], required: set[str], k_list=(5, 10)):
    """ranked_units: per returned unit (rank order), the evidence units it contains."""
    seen_at = {}
    for rank, units in enumerate(ranked_units[:10], 1):
        for u in units:
            seen_at.setdefault(u, rank)
    out = {}
    for k in k_list:
        got = {u for u, r in seen_at.items() if r <= k}
        out[f"recall@{k}"] = len(got & required) / len(required)
    out["strict_recall@10"] = float(required <= {u for u, r in seen_at.items() if r <= 10})
    first = min((r for u, r in seen_at.items() if u in required), default=None)
    out["mrr@10"] = 1 / first if first else 0.0
    gains = {}
    for u in required:
        if u in seen_at:
            gains[seen_at[u]] = gains.get(seen_at[u], 0) + 3          # 2^2 - 1 per newly covered required unit
    dcg = sum(g / math.log2(r + 1) for r, g in gains.items())
    idcg = sum(3 / math.log2(r + 1) for r in range(1, min(len(required), 10) + 1))
    out["ndcg@10"] = dcg / idcg
    out["found@10"] = len({u for u, r in seen_at.items() if r <= 10} & required)
    return out


def run_config(name, units_per_chunk, rank_fn, queries, tok_of_chunk):
    per, lat = {}, []
    for q in queries:
        t = time.perf_counter()
        order = rank_fn(q["query_text"])
        lat.append((time.perf_counter() - t) * 1000)
        ranked = [units_per_chunk[i] for i in order[:10]]
        req = set(q["required_evidence_passage_ids"])
        m = score_query(ranked, req)
        top5 = {u for us in ranked[:5] for u in us}
        m["distractor_in_top5"] = float(bool(set(q.get("distractor_passage_ids") or []) & top5)) if q.get("distractor_passage_ids") else None
        m["context_tokens@10"] = sum(tok_of_chunk[i] for i in order[:10])
        m["category"] = q["category"]
        m["top10"] = [int(i) for i in order[:10]]
        per[q["query_id"]] = m
    vals = list(per.values())
    agg = {k: round(sum(v[k] for v in vals) / len(vals), 4) for k in ("recall@5", "recall@10", "strict_recall@10", "mrr@10", "ndcg@10")}
    agg["micro_evidence_recall@10"] = round(sum(v["found@10"] for v in vals) /
                                            sum(len(q["required_evidence_passage_ids"]) for q in queries), 4)
    cats = sorted({v["category"] for v in vals})
    agg["recall@10_by_category"] = {c: round(sum(v["recall@10"] for v in vals if v["category"] == c) /
                                             sum(1 for v in vals if v["category"] == c), 4) for c in cats}
    agg["strict_recall@10_multihop"] = round(sum(v["strict_recall@10"] for v in vals if v["category"] == "boundary_multihop") /
                                             max(1, sum(1 for v in vals if v["category"] == "boundary_multihop")), 4)
    d = [v["distractor_in_top5"] for v in vals if v["distractor_in_top5"] is not None]
    agg["distractor_in_top5_rate"] = round(sum(d) / len(d), 4) if d else None
    agg["context_tokens@10_median"] = median(v["context_tokens@10"] for v in vals)
    agg["query_ms_p50"] = round(median(lat), 2)
    return {"summary": agg, "per_query": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--configs", nargs="+", default=["PROD", "F256_ov15", "F400_ov15", "F512_ov15", "F768_ov15",
                                                      "F256_ov0", "F400_ov0", "F512_ov0", "F768_ov0", "R512", "H512",
                                                      "SW", "PC", "K", "HY", "DIV"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(a.embedder, device="cpu")
    gt = ground_truth()
    answerable = [q for q in gt["queries"] if q["category"] not in ("quiz_generation", "intentionally_unanswerable")]
    unanswerable = [q for q in gt["queries"] if q["category"] == "intentionally_unanswerable"]
    results = {"embedder": a.embedder, "n_answerable": len(answerable), "configs": {}}
    cache = {}

    def chunks_for(name):
        if name not in cache:
            ch = C.build(name)
            t = time.perf_counter()
            D = embed(model, [c["text"] for c in ch])
            cache[name] = (ch, [contained_units(c["text"]) for c in ch], D, time.perf_counter() - t,
                           [C.ntok(c["text"]) for c in ch])
        return cache[name]

    for cfg in a.configs:
        if cfg == "PC":
            children, parents, owner = C.parent_child()
            Dc = embed(model, [c["text"] for c in children])
            pu = [contained_units(p["text"]) for p in parents]
            ptok = [C.ntok(p["text"]) for p in parents]

            def rank_fn(q):
                qv = embed(model, [q], query=True)[0]
                out = []
                for i in np.argsort(-(Dc @ qv)):
                    if owner[i] not in out:
                        out.append(owner[i])
                return out
            res = run_config(cfg, pu, rank_fn, answerable, ptok)
            res["summary"].update(n_units=len(parents), n_children=len(children))
        else:
            base = "PROD" if cfg in ("K", "HY", "DIV") else cfg
            ch, units, D, idx_s, toks = chunks_for(base)
            bm = Bm25([c["text"] for c in ch]) if cfg in ("K", "HY") else None

            def rank_fn(q, D=D, bm=bm, cfg=cfg):
                if cfg == "K":
                    return bm.rank(q)
                qv = embed(model, [q], query=True)[0]
                dense = [int(i) for i in np.argsort(-(D @ qv))]
                if cfg == "HY":
                    return rrf(dense[:50], bm.rank(q)[:50])
                if cfg == "DIV":
                    top = mmr(qv, D, dense[:20])
                    return top + [i for i in dense if i not in top]
                return dense
            res = run_config(cfg, units, rank_fn, answerable, toks)
            res["summary"].update(n_units=len(ch), index_seconds=round(idx_s, 2),
                                  units_split_across_chunks=None)
        results["configs"][cfg] = res
        s = res["summary"]
        print(f"{cfg:10s} n={s['n_units']:4d} R@5={s['recall@5']:.3f} R@10={s['recall@10']:.3f} strict@10={s['strict_recall@10']:.3f} "
              f"MRR={s['mrr@10']:.3f} nDCG={s['ndcg@10']:.3f} micro={s['micro_evidence_recall@10']:.3f} "
              f"multihopStrict={s['strict_recall@10_multihop']:.3f} distr5={s['distractor_in_top5_rate']} ctxTok={s['context_tokens@10_median']:.0f}",
              flush=True)

    # unanswerable: top dense score on production chunks (abstention signal, not a ranking metric)
    ch, units, D, _, _ = chunks_for("PROD")
    results["unanswerable_top_scores"] = {q["query_id"]: round(float((D @ embed(model, [q["query_text"]], query=True)[0]).max()), 3)
                                          for q in unanswerable}
    results["answerable_top_scores"] = {q["query_id"]: round(float((D @ embed(model, [q["query_text"]], query=True)[0]).max()), 3)
                                        for q in answerable}
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
