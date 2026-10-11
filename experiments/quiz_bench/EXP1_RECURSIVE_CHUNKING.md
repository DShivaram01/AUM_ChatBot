# Experiment 1 — Recursive chunking (`exp/quiz-recursive-chunking`)

Common baseline: `e7c28d3` (branch `exp/quiz-benchmark-baseline`). Fixture: benchmark paper sha256 `6bc726b1…` (20 pages, 14,793 Mistral tokens), 30 provisional gold concepts (not human-reviewed). Embedder fixed at MiniLM; retrieval = production dense inner product; quiz top-K = 6.

## Hypothesis
Token-aware recursive splitting (paragraph → line → sentence → word) retrieves gold evidence better than the production heading/page-aware chunker (`structured_chunks`, ~400 regex tokens), and some overlap helps across boundaries.

## Implementation
`recursive_chunking.py`. No new dependencies. Sizes are in **Mistral** tokens: 256 / 400 / 512 / 768 / 1024, each with overlap 0 % and 15 %. Page provenance is kept via character offsets; headings are provenance only. Grid runner: `run_chunking_grid.py`. Paired bootstrap: `paired_bootstrap.py`.

## Commands
```
python -m experiments.quiz_bench.run_chunking_grid --prefix structured400 recursive \
    --out experiments/quiz_bench/results/chunking_recursive_grid.json
python -m experiments.quiz_bench.paired_bootstrap structured400 recursive512_ov0
```

## Results (n = 30 concept queries; CPU)

| Chunker | Chunks | Median tok | Chunks MiniLM truncates | R@5 | R@10 | MRR@10 | nDCG@10 | Visible in top-6, 500-char view | Visible, full chunk | Top-6 dup. (D) |
|---|---|---|---|---|---|---|---|---|---|---|
| structured400 (prod) | 43 | 339 | 29 | 0.850 | 0.950 | 0.620 | 0.695 | 0.333 | 0.867 | 0.049 |
| recursive256 ov0 | 69 | 215 | 1 | 0.800 | 0.850 | 0.540 | 0.613 | **0.500** | 0.833 | 0.000 |
| recursive256 ov15 | 71 | 216 | 1 | 0.817 | 0.850 | 0.549 | 0.617 | 0.500 | 0.833 | 0.000 |
| recursive400 ov0 | 44 | 354 | 36 | 0.800 | 0.883 | 0.528 | 0.614 | 0.333 | 0.800 | 0.051 |
| recursive400 ov15 | 47 | 353 | 38 | 0.733 | 0.850 | 0.537 | 0.611 | 0.300 | 0.800 | 0.059 |
| **recursive512 ov0** | 37 | 428 | 33 | **0.883** | 0.950 | **0.748** | **0.791** | 0.333 | **0.900** | 0.045 |
| recursive512 ov15 | 38 | 448 | 37 | 0.867 | 0.917 | 0.631 | 0.698 | 0.300 | 0.900 | 0.069 |
| recursive768 ov0 | 22 | 715 | 21 | 0.667 | 0.800 | 0.591 | 0.633 | 0.133 | 0.733 | 0.028 |
| recursive768 ov15 | 24 | 716 | 23 | 0.783 | 0.850 | 0.650 | 0.694 | 0.067 | 0.800 | 0.072 |
| recursive1024 ov0 | 17 | 914 | 17 | 0.633 | 0.800 | 0.506 | 0.565 | 0.200 | 0.700 | 0.014 |
| recursive1024 ov15 | 18 | 952 | 18 | 0.767 | 0.900 | 0.583 | 0.648 | 0.067 | 0.867 | 0.046 |

- Paired bootstrap, recursive512 ov0 − production (10k resamples):
  - MRR@10: **+0.128**, 95 % CI [+0.045, +0.218]
  - nDCG@10: **+0.095** [+0.033, +0.163]
  - Recall@5: +0.033 [0, +0.100] (one query)
  - Visible evidence: +0.033 [0, +0.100]
- No chunker lost a concept at a boundary: all 30 anchors fall inside a single chunk for every configuration.
- Chunking takes ≤ 1 s for this paper; indexing 43 chunks with MiniLM on CPU takes about 1–2 s.

## What improved
512-token recursive chunks without overlap rank the supporting chunk higher than production (MRR +0.13, CI excludes 0).

## What became worse
- **Overlap** (15 %) never helped reliably and adds duplicated text to the evidence (512: dup 0.045 → 0.069; 768: 0.028 → 0.072).
- **Chunks ≥ 768 tokens** lose recall. MiniLM embeds only the first 256 word pieces, so the tail of a large chunk is invisible to retrieval.
- **Small chunks** (256) see more within the 500-char prompt view but rank worse.

## The dominant factor is not chunking
For every chunker, the supporting fact is in the top-6 evidence 0.80–0.90 of the time when the whole chunk is shown, but only 0.07–0.50 under the current 500-char-per-chunk prompt view. The production chunker already retrieves 4/5 BLASTP-topic concepts into the top 6 (quiz request D); the 500-char cut hides all of them (0/5, Phase 0). Token-aware evidence packing (Experiment 6) is the larger lever.

## Hypothesis supported?
Partly:
- For ranking: yes, at 512 tokens with no overlap.
- For recall and visible evidence: no measurable difference.
- Overlap: not supported.

Downstream quiz quality was **not** measured here (needs Mistral; see the consolidated report).

## Recommendation
**Investigate further.** Carry `recursive512_ov0` into the integrated candidate only together with token-aware evidence packing, and re-test with a stronger long-context embedder (Experiment 5), since MiniLM truncation confounds sizes ≥ 400. Reject overlap.

## Limitations
- One document, with a provisional, machine-drafted gold set.
- One concept per query.
- Retrieval-level metrics only.
