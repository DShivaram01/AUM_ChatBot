# Advanced quiz retrieval and generation benchmark

**Dates:** 2026-10-11 to 2026-10-12.
**Repository baseline:** `main` at `fc5648a`. Production behaviour was unchanged throughout, and nothing is merged.
**Common baselines:** `e7c28d3` (retrieval harness) and `397f21f` (generation harness), both on `exp/quiz-benchmark-baseline`.

**Headline.** The production quiz pipeline's problems on documents are not mainly about chunking or embeddings. They come from **what evidence it chooses** and **how it shows that evidence to the model**:
- A whole-document request searches for the literal phrase "this document".
- Each chunk is cut to 500 characters.
- The prompt is cut at 3,200 characters once the list of earlier questions grows.
- Topics with no support get no relevance check.

A model-independent **quiz blueprint** (evidence plan) combined with **Outlines** schema-constrained JSON fixes these issues. On the CS-201 notes:

| Metric (20-question quiz) | Production | Integrated candidate |
|---|---|---|
| Valid questions | 20/20 | 20/20 |
| Near-duplicate question pairs | **110** | **0** |
| Gold concepts covered | 3 | 21 |
| Answers lexically supported by cited evidence | 2/20 | 18/20 |
| Total time | 116 s | 136 s |

Off-topic requests are declined instead of answered from neighbouring chapters.

## 1. Repository baseline and constraints
- Production: FastAPI with Mistral-7B-Instruct-v0.3 (fp16, GPU) and MiniLM embeddings. Quiz requests are natural-language chat requests (no quiz button, no COS quizzes); uploaded documents are the quiz source.
- **Experiment code** lives only on `exp/quiz-*` branches, each in its own git worktree.
- **Generation experiments** ran in `qzgen`, a clone of the production `aum_env` with torch 2.6.0+cu124 and transformers 4.46.3 pinned. Structured-output frameworks were installed only there.
- **Embedding experiments** ran in a separate CPU venv (`~/.venvs/embbench`).
- **GPU:** the two owner-approved maintenance windows stopped production (2026-10-11 19:17–20:31 and 22:57–23:20). Both times production was restarted and passed its health check.

## 2. Lab hardware
| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4000 Ada, 21 GB (≈4.9 GB free while production Mistral is loaded; 20 GB during the windows) |
| CPU / RAM | 32 cores, 60 GB RAM |
| Disk | 1.6 TB free |

`nvidia-smi` is unavailable (NVML mismatch), so GPU memory comes from `torch.cuda`. Peak GPU allocated during generation: 13.7–14.1 GiB.

## 3. Benchmark documents
| Fixture | Pages | Mistral tokens | Chunks (production) | SHA-256 |
|---|---|---|---|---|
| Research paper (unpublished manuscript; **inactive at the owner's request** after the retrieval phase) | 20 | 14,793 | 43 | `6bc726b1…` |
| CS-201 *Data Structures & Algorithms* course notes (**primary**) | 10 | 4,678 | 37 | `d5b33bba…` |

Gold sets (30 and 35 concepts with anchor phrases, pages, queries and answers, plus unanswerable questions and confusable pairs) are **provisional**: machine-drafted and not human-reviewed. Every result below inherits that limitation.

## 4. Document extraction analysis (production `pymupdf4llm`)
**Research paper:**
- All nine display equations are dropped.
- The rotated Table 1 comes out as split-word columns; Table 2 is intact.
- There are no figure images (legends only).
- Line-break hyphens are joined ("sequencelevel").
- Plain PyMuPDF loses 2/30 gold anchors; `pymupdf4llm` loses none.

**CS-201 notes:**
- Code blocks are clipped to about 90 characters in the text layer itself, so the code is unrecoverable.
- Sub- and superscripts are flattened.
- Tables are intact.
- Page footers leak into the text.
- Each page's "Quiz Benchmark Questions" box becomes its own chunk, which production can hand to the model and the model then copies.

## 5. Chunking (`exp/quiz-recursive-chunking`, Experiment 1)
| Chunker | Paper MRR@10 | Notes MRR@10 |
|---|---|---|
| Production `structured400` | 0.620 | **0.881** |
| Recursive 512, no overlap | **0.748** (+0.128, 95% CI [+0.045, +0.218]) | 0.661 |

- Overlap never helped reliably and added duplicated evidence.
- Chunks of 768 tokens or more lose recall, because MiniLM embeds only the first 256 word pieces.
- **The paper's chunk-size gain did not generalise to the notes. Recommendation: keep the production chunker.**

## 6. Structure-aware splitting (`exp/quiz-structure-splitting`, Experiment 2)
- **Ranking:** no significant difference from production on either document (paper MRR +0.042, CI [−0.073, +0.153]).
- **Back-matter filter:** excluding references and acknowledgements removes bibliography chunks from quiz evidence. Production fed one to 5 of 6 requests on the paper. On the notes the filter is a no-op.
- **Recommendation:** adopt the filter; keep `pymupdf4llm`.
- **Not evaluated:** layout-model extractors (Docling, GROBID, Marker), which need an isolated heavy environment.

## 7. Parent-child retrieval (`exp/quiz-parent-child`, Experiment 3)
**Research paper:**
- Packing **whole** retrieved chunks into a 1,000-token budget instead of six 500-character cuts raised gold-concept coverage from 0.33 to 0.80 (+0.467, CI [+0.267, +0.667]) at roughly the same token cost.
- Parent-child c256/p1024 reached 0.93 against 0.83 for flat chunks at 2,000 tokens, which is not significant.

**Notes:** every strategy reaches the ceiling (1.00), because the chunks are already short.

**Recommendation:** adopt token-budgeted whole-chunk evidence; re-test parent-child on a longer document.

## 8. Map-reduce and sliding window (`exp/quiz-map-reduce`, Experiment 4; real Mistral)
| Selection method (N = 10) | Chapters | Gold concepts in evidence | Extra LLM time | Valid questions |
|---|---|---|---|---|
| Direct top-6 (production) | 4 | 6 | – | 7/10 |
| **Blueprint (no LLM)** | **10** | **17** | **0 s** | 9/10 |
| LLM map per chapter + embedding reduce | 3 | 9 | 120 s | 5/10 |
| LLM map over sliding windows | 5 | 9 | 109 s | 7/10 |

- Mistral returned only 11–12 usable facts from 9–10 map calls.
- At N = 20, the blueprint's evidence holds 34 gold concepts.
- Hierarchical reduction was not needed for a 10-page document.
- **Recommendation: reject map-reduce for this model; use the blueprint.**

## 9. Embedding models (`exp/quiz-embedding-benchmark`, Experiment 5; CPU)
Dense MRR@10, with the production chunker fixed. Each model got its own index and its published query/document prompts.

| Model | Paper | Notes | Query P50 (CPU) | Notes |
|---|---|---|---|---|
| MiniLM (production) | 0.620 | 0.881 | 3 ms | |
| Qwen3-Embedding-0.6B | 0.758 | **0.952** | 36 ms | best on both documents |
| BGE-M3 | **0.812** | 0.910 | 49–59 ms | |
| Qwen3-Embedding-4B | 0.790 | 0.862 | 175–184 ms | its RSS reading was not credible; re-measure |
| EmbeddingGemma-300M | — | — | — | not run: gated, no access |
| EmbeddingGemma-2 | — | — | — | failed: missing `pillow` |

- **Hybrid BM25 + dense (RRF):** raised Recall@5 to 0.92–0.95 on the term-heavy paper, but **hurt** the hash-table topic on the notes (target coverage 1.00 → 0.33). Not uniformly beneficial.
- **Recommendation: investigate further.** Qwen3-0.6B is the strongest candidate, but it needs transformers ≥ 4.51 (production pins 4.46.3) and costs about 10× MiniLM's latency. It has not yet been measured in a quiz end to end.

## 10. Batch generation and blueprint (`exp/quiz-batch-generation`, Experiment 6; real Mistral)
The blueprint fills slots by chapter round-robin for whole-document requests. For topic requests it takes chunks at or above the production document floor (0.35) plus the top hit's chapter, and it declines if nothing clears the floor. It plans at most 3 objectives per chunk and never invents slots.

20-question whole-document quiz on the notes:

| Arm | Valid | Near-dup pairs | Chapters | Gold concepts | Supported | Time | First question |
|---|---|---|---|---|---|---|---|
| Production | 20 | 110 | 3 | 3 | 2/20 | 116 s | ~6 s |
| Blueprint, 1 per call | 19/20 | 0 | 10 | 17 | 18/19 | 128 s | 5.7 s |
| Blueprint, batch 5 | 18/20 | 0 | 10 | 15 | 17/18 | 205 s | 37 s |
| Blueprint, batch 10 | 19/20 | 0 | 10 | 16 | 18/19 | 267 s | 90 s |
| Blueprint, adaptive | 19/20 | 0 | 10 | 15 | 16/19 | 168 s | 43 s |

- **Batching does not reduce wall time on this GPU**, because decoding dominates (~20 tokens/s).
- **One question per call** gives the fastest first result.
- The **hash-table quiz** that production failed now yields 4–6 of the 6 planned questions.
- The **red-black request** is declined in every arm.

## 11. Structured JSON (`exp/quiz-structured-output`, Experiment 7; real Mistral)
Same blueprint, batch 5, one canonical Pydantic schema (`QuizBatch` with an exact question count).

| Mechanism | 20-question quiz | Calls | Parse failures | Invalid items | Time | Decode speed |
|---|---|---|---|---|---|---|
| Prompt + production repair | 18/20 | 11 | 5 | 9 | 205 s | 19.9 tok/s |
| lm-format-enforcer | 20/20 | 9 | 1 | 5 | 209 s | 19.3 tok/s |
| **Outlines 1.3** | **20/20** | **4** | **0** | **0** | **136 s** | 19.7 tok/s |

- **LMQL 0.7.3** runs with transformers 4.46 but loads its own model copy, which can't share the GPU; it is effectively unmaintained. Not benchmarked.
- **Instructor** has no in-process transformers backend, and its validate-and-retry role duplicates the slot filler. Not applicable.
- Constraint overhead on decode speed is negligible. **Valid JSON ≠ grounded:** lexical support stays at 17–19/20 in every arm.

## 12. Factual grounding
"Lexically supported" means ≥ 50 % of the correct answer's content words appear in the cited evidence chunk. This is a proxy, not human-verified correctness.

| Configuration | Answers supported |
|---|---|
| Production, 20-question quiz | 2/20 (generated from 2–3 pages, mostly rewordings) |
| Blueprint arms | 15–18/20 |

- **Embedded quiz boxes:** the notes' own questions were copied 1–3 times per request by every configuration that hands them to the model (direct top-6 retrieval, the no-blueprint ablation), and 0 times when the blueprint excludes them.
- **Human review is still required** for factual correctness and question quality. All generated questions are in `experiments/quiz_bench/dumps/*/generation_outputs/` on each branch.

## 13. Performance and resources
| Item | Measurement |
|---|---|
| Mistral load (fp16, GPU) | ~13–20 s |
| Decoding | ~20 tok/s in every arm |
| Peak GPU allocated | 13.7–14.1 GiB |
| Per question (one per call) | ~6 s |
| 20-question quiz | 116–267 s depending on arm |
| Blueprint, rejection gate, evidence packing | no model calls; negligible CPU |
| Embedding indexing (CPU, 37–43 chunks) | 0.2 s (MiniLM) to 34 s (Qwen3-4B) |

## 14. Integrated candidate (`exp/quiz-integrated-candidate`, Phase 10; real Mistral)
Components:
- the production chunker and MiniLM;
- the blueprint, with the back-matter filter and the production document floor;
- whole-chunk, token-aware evidence;
- Outlines;
- targeted per-slot repair.

| Arm | A (5) | B (10) | C (20) | D (hash tables) | E (AVL) | F (red-black, unsupported) |
|---|---|---|---|---|---|---|
| **I2: blueprint + Outlines, batch 5** | 5/5 | 10/10 | **20/20**; 0 dup; 21 concepts; 18/20 supported; 136 s | **6/6** | 8/9 | **declined** (0 calls) |
| I1: blueprint + Outlines, 1 per call | 5/5 | 10/10 | 17/20; first question in 6 s | 4/6 | 7/9 | declined |
| A1: ablation without blueprint | 5/5 | 9/10 | 13/20; 3 chapters + title page; copies notes' questions | 10/10* | 9/10* | **answered with 9 AVL questions** |
| Production baseline | 5/5 | 10/10 | 20/20 but 110 dup pairs, 2/20 supported | **failed** | 10/10 | failed by accident |

\* A1 is not capped by the blueprint, so it plans the full requested count from the top-6 chunks.

**Attribution:**
- **The blueprint** drives coverage, grounding, duplicate removal and the decline of unsupported topics.
- **Outlines** drives JSON validity and removes most repair calls.
- **The batch size** trades first-result latency (1 per call: ~6 s) against completeness (batch 5 was more complete: 20/20 vs 17/20).

## 15. Recommended production configuration (pending review)
1. **Blueprint for evidence selection:**
   - whole-document requests: chapter round-robin;
   - topic requests: at or above the document floor;
   - decline below the floor;
   - never invent slots;
   - exclude embedded quiz boxes and back matter.
2. **Whole-chunk, token-budgeted evidence**, replacing the 500-character cut and the 3,200-character prompt truncation.
3. **Outlines schema-constrained JSON**, with the canonical Pydantic schema and targeted per-slot repair.
4. **Batch size:** 5 for completeness, or 1 per call where time to first question matters. Stream per batch either way.
5. **Keep** the production chunker and MiniLM for now. Evaluate Qwen3-Embedding-0.6B end to end as a separate change.

Before deployment:
- Add Outlines to the production environment, after a pin-compatibility review.
- Write unit and integration tests.
- Have a human review the generated quizzes.
- Measure on at least two more authorized documents of different lengths and structure.

## 16. Limitations, risks, rollback
- **Sample size:** two documents, of which only the notes ran through generation.
- **Gold sets:** provisional and not human-reviewed.
- **Grounding metric:** lexical support is a proxy.
- **Nondeterminism:** greedy decoding is deterministic, so there was no repeated sampling. The in-harness baseline reproduced the live server exactly.
- **Retrieval ceiling:** the notes are too easy to rank retrieval methods; a longer, harder document is needed.
- **Floor calibration:** the evidence floor of 0.35 is reused from document QA and was not calibrated for quizzes.
- **Outlines integration:** adds a dependency and couples JSON generation to the transformers backend.
- **Rollback:** every change is isolated on unmerged branches; production code and config are untouched. A future deployment should sit behind a config flag (old quiz path as the default) so it can be switched off instantly.

## 17. Branches and commits (remote `origin`, private repository)
| Branch | Commit | Contents |
|---|---|---|
| `exp/quiz-benchmark-baseline` | `7c3b356` | harness, fixtures, Phase 0/1 baselines (both documents), extracted pages, production chunks, gold sets |
| `exp/quiz-recursive-chunking` | `589360a` | Experiment 1 + dumps of all 11 recursive chunkers |
| `exp/quiz-structure-splitting` | `0dbe45d` | Experiment 2 + dumps of all 10 structure chunkers |
| `exp/quiz-parent-child` | `d003a91` | Experiment 3 + parent/child hierarchy dumps |
| `exp/quiz-map-reduce` | `55c05eb` | Experiment 4 + extracted facts / questions |
| `exp/quiz-embedding-benchmark` | `ab6d3af` | Experiment 5, both documents |
| `exp/quiz-batch-generation` | `4af2a41` | Blueprint, Experiment 6 + blueprints / questions |
| `exp/quiz-structured-output` | `388ba62` | Experiment 7 + questions |
| `exp/quiz-integrated-candidate` | `c01a9a7` (+ this report) | Phase 10 + questions + this report |

**Per-experiment verdicts:**
| Experiment | Verdict |
|---|---|
| Recursive chunking | investigate (did not generalise) |
| Structure splitting | adopt the back-matter filter only |
| Parent-child | adopt budgeted packing; investigate parent-child |
| Map-reduce | reject |
| Embeddings | investigate Qwen3-0.6B |
| Batch | adopt blueprint; batch size 1 or 5 by UX need |
| Structured output | adopt Outlines |
| Integrated | candidate for a reviewed production change |
