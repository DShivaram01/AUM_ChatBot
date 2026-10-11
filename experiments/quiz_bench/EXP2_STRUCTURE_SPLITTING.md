# Experiment 2 — Markdown and structure-aware splitting (`exp/quiz-structure-splitting`)

Common baseline `e7c28d3`. Same fixture, provisional gold set (30 concepts), MiniLM, and production dense retrieval as Experiment 1.

## Hypothesis
Splitting along the paper's section hierarchy (headings, paragraphs, whole tables) and excluding non-quizzable back matter improves retrieval and evidence quality over the production chunker, which also follows headings but packs across paragraphs at about 400 regex tokens.

## Implementation (`structure_chunking.py`)
- **Blocks.** Built from the production `pymupdf4llm` markdown: headings, paragraphs, and markdown tables (never split).
- **Packing.** Blocks are packed per section up to 256/400/512/768 Mistral tokens; sections never share a chunk; oversized paragraphs split at sentence boundaries; sections under 60 tokens merge into the next sibling.
- **`_filtered` variants.** These drop the References section, the author/affiliation block, and Acknowledgments/Funding/Data/Code availability/Competing-interests/Authors'-contributions paragraphs.
- **Extraction fidelity.** `plain_recursive512` uses plain PyMuPDF text at the same chunk size, to compare against `pymupdf4llm`.

Not evaluated: GROBID, Docling, Marker, Nougat. None is installed, and each needs a separate heavy environment (models and/or a Java service). Installing them into the production env was out of scope by policy. They are candidates for a follow-up in an isolated venv.

## Commands
```
python -m experiments.quiz_bench.run_chunking_grid --prefix structured400 section plain_recursive512 \
    --out experiments/quiz_bench/results/chunking_structure_grid.json
python -m experiments.quiz_bench.paired_bootstrap structured400 section512_filtered
```

## Extraction fidelity (benchmark paper)

| | pymupdf4llm (production) | plain PyMuPDF |
|---|---|---|
| Gold anchors present in extracted text | **30/30** | 28/30 (loses C13 and C30: hyphen/line-join differences) |
| Table 2 (Spearman table) rows intact | yes (markdown table) | yes (flattened lines) |
| Table 1 (printed rotated) | garbled into split-word columns | garbled |
| Display equations (Eq. 1–9) | **dropped** (blank) | dropped |
| Figures | legends only; no images are in the PDF | same |
| Section headings | recovered as markdown `#` levels | none |

`pymupdf4llm` is the better extractor for this paper. Neither recovers the equations, so quiz items about the metric formulas cannot be grounded from extracted text.

## Retrieval results (n = 30)

| Chunker | Chunks | R@5 | R@10 | MRR@10 | nDCG@10 | Visible top-6 (500-char) | Visible top-6 (full) | Ref. chunks in quiz evidence A/B/C/D/E/F |
|---|---|---|---|---|---|---|---|---|
| structured400 (prod) | 43 | 0.850 | 0.950 | 0.620 | 0.695 | 0.333 | 0.867 | 1/1/1/0/1/2 |
| section256 | 78 | 0.767 | 0.850 | 0.488 | 0.572 | 0.433 | 0.767 | |
| section256_filtered | 61 | 0.783 | 0.850 | 0.500 | 0.578 | 0.467 | 0.800 | |
| section400 | 51 | 0.833 | 0.950 | 0.589 | 0.671 | 0.367 | 0.900 | |
| section400_filtered | 41 | 0.833 | 0.950 | 0.616 | 0.684 | 0.367 | 0.900 | |
| section512 | 43 | 0.850 | 0.917 | 0.636 | 0.699 | 0.333 | 0.867 | 2/2/2/0/1/3 |
| **section512_filtered** | 34 | 0.850 | 0.933 | **0.663** | **0.718** | 0.333 | 0.867 | **0/0/0/0/0/0** |
| section768 | 34 | 0.817 | 0.883 | 0.621 | 0.679 | 0.333 | 0.833 | |
| section768_filtered | 28 | 0.817 | 0.900 | 0.643 | 0.695 | 0.333 | 0.833 | |
| plain_recursive512 | 35 | 0.783 | 0.883 | 0.643 | 0.701 | 0.233 | 0.900 | |

Paired bootstrap, section512_filtered − production:
- Recall@5: 0.000 [−0.10, +0.10]
- MRR@10: +0.042 [−0.073, +0.153]
- nDCG@10: +0.022 [−0.078, +0.110]

**No significant ranking difference.**

## What improved
- **Back-matter filtering** removes bibliography chunks from quiz evidence entirely. Production hands a reference-list chunk to 5 of 6 quiz requests, and 2 to the unsupported-topic request. The index also shrinks by 21 % (43 → 34 chunks).
- **Whole tables.** Tables stay intact and sections never mix.

## What became worse / did not help
- Small section chunks (256) rank worse.
- Ranking gains over production are within noise. The production chunker already respects headings, so most of this design's benefit was already present.
- The whole-document request still sees only 1–2 gold concepts, because the query "this document" is meaningless regardless of chunking. That needs a coverage strategy (Experiment 4 / blueprint), not a splitter.

## Hypothesis supported?
- **Retrieval ranking:** not supported (no significant difference).
- **Evidence quality:** supported for back-matter filtering, which removes a concrete source of off-topic quiz evidence.

## Recommendation
- **Adopt the back-matter filter** in the integrated candidate. Low cost, clear benefit; also worth a separate small production change after review.
- **Keep `pymupdf4llm`.**
- **Investigate further:** a layout-model extractor (Docling/GROBID) in an isolated env, for equations and the rotated table.

## Limitations
- One paper.
- Provisional gold set.
- The filter's regexes are tuned on this paper's conventions; they need testing on other document types before production use.
