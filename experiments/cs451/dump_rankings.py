"""Precompute ranked chunk indices (top 10) for every non-quiz query under a
given chunker/embedder, so Phase B generation (in the production-pinned env)
can use retrieval from embedders that need newer transformers.

    ~/.venvs/embbench/bin/python -m experiments.cs451.dump_rankings --chunker R512 \
        --embedder Qwen/Qwen3-Embedding-0.6B --out experiments/cs451/results/rankings_R512_qwen06.json
"""
import argparse
import json

import numpy as np

from experiments.cs451 import chunking as C
from experiments.cs451.eval_retrieval import embed
from experiments.cs451.units import ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(a.embedder, device="cpu")
    ch = C.build(a.chunker)
    D = embed(model, [c["text"] for c in ch])
    out = {"chunker": a.chunker, "embedder": a.embedder, "n_chunks": len(ch), "rankings": {}}
    for q in ground_truth()["queries"]:
        if q["category"] == "quiz_generation":
            continue
        s = D @ embed(model, [q["query_text"]], query=True)[0]
        order = [int(i) for i in np.argsort(-s)[:10]]
        out["rankings"][q["query_id"]] = {"top10": order, "scores": [round(float(s[i]), 4) for i in order]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", len(out["rankings"]), "rankings")


if __name__ == "__main__":
    main()
