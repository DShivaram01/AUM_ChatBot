"""Paired bootstrap over gold queries for two runs' per-query metrics.

    python -m experiments.quiz_bench.paired_bootstrap RESULTS.json A B [--metric mrr@10]
"""
import json
import random
import sys


def paired_ci(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rnd = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), boots[int(0.025 * n)], boots[int(0.975 * n)]


if __name__ == "__main__":
    from experiments.quiz_bench.fixture import production_pages
    from experiments.quiz_bench import chunkers as reg, structure_chunking  # noqa: F401
    from experiments.quiz_bench.retrieval_eval import evaluate
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    base, cand = sys.argv[1], sys.argv[2]
    runs = {n: {q["id"]: q for q in evaluate(reg.CHUNKERS[n](production_pages()), emb, None)["per_query"]} for n in (base, cand)}
    ids = sorted(runs[base])
    for m in ("recall@5", "mrr@10", "ndcg@10", "visible_in_top6"):
        d, lo, hi = paired_ci([runs[base][i][m] for i in ids], [runs[cand][i][m] for i in ids])
        print(f"{cand} - {base}  {m:16s} diff={d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(ids)}")
