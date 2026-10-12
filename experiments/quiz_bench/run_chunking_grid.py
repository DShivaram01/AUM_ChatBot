"""Run the retrieval evaluation over a grid of chunkers, at two evidence views:
the current prompt view (first 500 chars per chunk) and the full chunk (what a
token-aware evidence packer would show).

    python -m experiments.quiz_bench.run_chunking_grid --prefix recursive structured400 --out results/x.json
"""

import argparse
import json
import time
import tracemalloc

from experiments.quiz_bench import chunkers as reg
from experiments.quiz_bench.fixture import production_pages
from experiments.quiz_bench.retrieval_eval import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", required=True, help="chunker names or name prefixes")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import importlib
    for mod in ("recursive_chunking", "structure_chunking", "parent_child"):
        try:
            importlib.import_module(f"experiments.quiz_bench.{mod}")
        except ModuleNotFoundError:
            pass
    import config
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    emb = SentenceTransformer(a.embedder, device="cpu")
    tok = AutoTokenizer.from_pretrained(config.LLM_ID)
    pages = production_pages()
    names = [n for n in sorted(reg.CHUNKERS) if any(n == p or n.startswith(p) for p in a.prefix)]
    rows = {}
    for name in names:
        tracemalloc.start()
        t = time.perf_counter()
        chunks = reg.CHUNKERS[name](pages)
        chunk_s = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        r500 = evaluate(chunks, emb, 500, tok)
        rfull = evaluate(chunks, emb, None, tok)
        truncated_for_embedder = sum(len(emb.tokenizer(c["text"])["input_ids"]) > emb.max_seq_length for c in chunks)
        rows[name] = {
            "n_chunks": r500["n_chunks"], "chunk_tokens": r500["chunk_tokens"], "chunking_seconds": round(chunk_s, 2),
            "chunking_peak_mib": round(peak, 1), "index_seconds": r500["index_seconds"],
            "chunks_truncated_by_embedder": truncated_for_embedder,
            "concepts_with_anchor_chunk": r500["concepts_with_anchor_chunk"],
            "retrieval": r500["retrieval"],
            "visible_in_top6_view500": r500["retrieval"]["visible_in_top6"],
            "visible_in_top6_full": rfull["retrieval"]["visible_in_top6"],
            "top6_evidence_tokens_full": {k: v["evidence_tokens_shown"] for k, v in rfull["quiz_requests"].items()},
            "top6_duplicate_ratio_full": {k: v["duplicate_ratio"] for k, v in rfull["quiz_requests"].items()},
            "quiz_request_visible_concepts_full": {k: v["n_visible_concepts"] for k, v in rfull["quiz_requests"].items()},
            "quiz_request_target_visible_full": {k: v["target_concepts_visible"] for k, v in rfull["quiz_requests"].items() if v["target_concepts_visible"]},
            "per_query_hit": {q["id"]: q["recall@5"] for q in r500["per_query"]},
        }
        r = rows[name]
        print(f"{name:22s} n={r['n_chunks']:3d} medTok={r['chunk_tokens']['median']:6.0f} embTrunc={truncated_for_embedder:3d} "
              f"anchors={r["concepts_with_anchor_chunk"]:2d} R@5={r['retrieval']['recall@5']:.3f} R@10={r['retrieval']['recall@10']:.3f} "
              f"MRR={r['retrieval']['mrr@10']:.3f} nDCG={r['retrieval']['ndcg@10']:.3f} vis500={r['visible_in_top6_view500']:.3f} "
              f"visFull={r['visible_in_top6_full']:.3f} D_tok={r['top6_evidence_tokens_full']['D']} D_tgt={r['quiz_request_target_visible_full'].get('D')} "
              f"E_tgt={r['quiz_request_target_visible_full'].get('E')} dupD={r['top6_duplicate_ratio_full']['D']}", flush=True)
    json.dump({"embedder": a.embedder, "results": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
