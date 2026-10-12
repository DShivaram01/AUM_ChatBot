# AUM-CS451-RAG-2026-001: results for the AUM chatbot

**Run:** 2026-10-12. Branch `exp/cs451-rag-benchmark`.
**Inputs:** the benchmark PDF (24 pages, SHA-256 `a8d7420c…`) was the only file ingested. The ground truth and evaluator guide were read only by the evaluator code and never indexed or prompted.
**Generator:** Mistral-7B-Instruct-v0.3, fp16, greedy (deterministic), in a clone of the production environment, during an owner-approved maintenance window (production down 01:36–02:41, restarted and healthy).
**Scoring:**
- Retrieval metrics are computed exactly per guide §5.
- Answer correctness is **model adjudication by Claude** using the guide's 1 / 0.5 / 0 rubric. It is **not** human review.
- Quiz metrics are automatic proxies.
- Every output is saved for human adjudication.

## Panel 1 — Retrieval (80 answerable non-quiz queries; guide §5)
Evidence units: 88 passages, 13 tables, 24 equations. A chunk contains a unit only if it holds at least 50 % of the unit's 8-word shingles; labels alone don't count, and the contents page is excluded. nDCG uses binary gains (required units only).

| Config (MiniLM unless noted) | R@5 | R@10 | Strict all-evidence R@10 | Multi-hop strict | MRR@10 | nDCG@10 | Distractor in top-5 | Top-10 context tokens |
|---|---|---|---|---|---|---|---|---|
| **PROD** (production chunks, dense) | 0.771 | 0.856 | 0.725 | 0.667 | 0.753 | 0.732 | 0.5 | 1,220 |
| F256 / F400 / F512 / F768, 15 % overlap | 0.81–0.85 | 0.91–0.92 | 0.81–0.86 | 0.53–0.73 | 0.66–0.79 | 0.73–0.81 | 0.5–0.9 | 2.6k–7.7k |
| F256 / F400 / F512 / F768, 0 % overlap | 0.69–0.73 | 0.79–0.85 | 0.68–0.76 | 0.47–0.73 | 0.57–0.67 | 0.65–0.69 | 0.6–0.8 | 2.6k–7.7k |
| R512 | 0.762 | 0.865 | 0.775 | 0.600 | 0.702 | 0.742 | 0.7 | 4,738 |
| H512 | 0.752 | 0.846 | 0.762 | 0.533 | 0.624 | 0.680 | 0.8 | 4,004 |
| SW (400 / stride 200) | 0.877 | 0.940 | 0.863 | 0.667 | 0.754 | 0.779 | 0.8 | 4,002 |
| PC (child 256 → parent ≤ 1024) | **0.925** | **0.973** | **0.938** | **0.867** | 0.793 | **0.851** | 0.8 | 8,604 |
| K (BM25, PROD chunks) | 0.771 | 0.849 | 0.700 | 0.333 | 0.769 | 0.727 | 0.6 | 1,180 |
| **HY** (RRF k=60, PROD chunks) | 0.809 | 0.916 | 0.825 | 0.667 | **0.803** | 0.778 | 0.7 | **1,217** |
| DIV (MMR 0.65) | 0.699 | 0.798 | 0.662 | 0.467 | 0.728 | 0.687 | 0.4 | 1,202 |
| PROD, Qwen3-Embedding-0.6B | 0.774 | 0.890 | 0.800 | 0.733 | 0.800 | 0.777 | 0.6 | 1,202 |
| R512, Qwen3-0.6B | 0.904 | 0.967 | 0.938 | 0.867 | 0.753 | 0.813 | 0.6 | 4,755 |
| PC, Qwen3-0.6B | 0.912 | **0.992** | **0.975** | 0.867 | 0.795 | 0.853 | 0.9 | 8,376 |

**Boundary analysis** (guide §3, measured with the Mistral tokenizer):
- With 15 % overlap, **no** unit is split at any window size.
- With zero overlap, units are split at every size:

| Window | Units split | Examples |
|---|---|---|
| F256 | 7 | S2-T02, S6-E02, S6-T01 |
| F400 | 7 | |
| F512 | 1 | |
| F768 | 4 | |

- Units with a "vulnerable edge" (< 25 tokens from a window boundary): 45 / 28 / 24 / 8 for the four sizes. Being near an edge is not the same as being split.

**Findings:**
- **HY** improves on PROD at the same context cost.
- **PC** finds the most evidence, including multi-hop, but uses about 7× the context.
- **Overlap matters** for fixed windows.
- **Wrong-domain distractors** reach the top 5 for 40–90 % of lexical-ambiguity queries under *every* configuration. Retrieval alone cannot solve the lexical traps.

## Panel 2 — Grounded answers (90 queries)
| Arm | Correctness (80 answerable) | Full / partial / wrong | Fabricated on 10 unanswerable | Median latency | Output tokens |
|---|---|---|---|---|---|
| **A0** production `document_chat` (unchanged) | **0.794** | 56 / 15 / 9 | 1 (invented a campus ISP) | 10.6 s | 20.8k |
| P1 production retrieval, whole chunks, "cite IDs or say Not specified" prompt | 0.744 | 52 / 15 / 13 | **0** | 3.0 s | 6.7k |
| P2 R512 + Qwen3-0.6B retrieval, P1 prompt | 0.719 | 51 / 13 / 16 | **0** | 4.3 s | 7.2k |

By category:

| Category | A0 | P1 | P2 |
|---|---|---|---|
| direct_factual | 0.84 | **0.90** | 0.72 |
| paraphrased_semantic | **0.80** | 0.73 | 0.75 |
| boundary_multihop | 0.67 | 0.50 | **0.70** |
| table_equation | 0.85 | 0.80 | **1.00** |
| lexical_ambiguity | **0.80** | 0.70 | 0.40 |

**Findings:**
- **Production's answer prompt is the most accurate overall.**
- **The strict-abstention prompt** removes fabrication (0/10) but causes false abstentions: 8 in P2, mainly on lexical-ambiguity and direct questions.
- **Better retrieval** (P2) helps exactly where the retrieval panel predicts: table/equation (1.00) and multi-hop (0.70).
- **Citations:**
  - Production cites file > heading > page, not unit IDs.
  - P1 can't cite unit IDs, because **production chunks strip headings, including passage labels, from the chunk text** (they live only in `heading_path`). P1 cited "[Excerpt n]" or invented IDs.
  - P2's recursive chunks keep labels in the text: 93 % of its answers cite IDs, with precision 0.68.
- **The automatic proxy is misleading.** It scored A0 0.575, P1 0.450 and P2 0.412, rewarding verbosity and penalising concise correct answers ("8 ms" vs "8 milliseconds"). Use the adjudicated numbers.

## Panel 3 — Quiz scenarios (guide §4, Q091–Q100)
**Live production:** `detect_quiz_intent` recognises **0 of 10** prompts. The counts are spelled out ("five", "ten", "twenty"), and phrasings like "Produce … questions" aren't covered. Every quiz request would be answered as document Q&A prose.

| Scenario | QP: production generator, if it had been routed | QI: integrated candidate | QI_nochk (ablation) |
|---|---|---|---|
| Q091 whole, 5 | 5 items, 1 section | 5/5, 5 sections | – |
| Q092 whole, 10 | **failed** (JSON) | **10/10, 6/6 sections, 0 dup** | – |
| Q093 whole, 20 | **failed** | **20/20, 6/6 sections, 13 passages, 0 dup** | – |
| Q094 hashing | 10, 6 duplicates | 8/10, sections S2 + S6, 0 dup (21 calls, 164 s) | 9/10, S2 only (5 calls, 86 s) |
| Q095 networking | **failed** | 10/10, 2 dup | 10/10, 0 dup |
| Q096 consistency | **failed** | 10/10 | 10/10 |
| Q097 cross-section | 10, 1 section, 0 multi-section items | 6/10, **all 6 span 2 sections** | – |
| Q098 GPU (unsupported) | **10 near-identical invented items** | **declined** with supported sections listed | 10 items (fails) |
| Q099 numeric | 5, 2 sections | 5/5, 5 sections | – |
| Q100 terminology | 10 | 7/10, 6 sections | – |

How QI works:
- word-number-aware request parsing;
- a section blueprint built from unit labels, with an evenly spread within-section order;
- whole / topic / numeric / terminology / integrative modes;
- an LLM evidence-sufficiency check for topic requests;
- Outlines schema-constrained JSON in batches of 5.

**Findings:**
- **The evidence-sufficiency check is what declines the unsupported GPU request.** The 0.35 retrieval floor alone cannot: the topic entity is in the document, so the request scores 0.595.
- **The check has a price** on supported topics: more calls, and it can over-prune (Q094). It needs tuning.
- **The unit-ID citation metric is 0 in every arm.** Production chunks lack labels (see Panel 2), so a model citation can only be a guess. Fixing this means prepending `heading_path` to each chunk's text.
- **Quiz factual correctness is not yet adjudicated.** All items are in `phase_b_outputs.json`.

## Panel 4 — Operations
- Mistral load: ~15 s.
- Answers: median 3–11 s per query.
- Quizzes: 5 questions ≈ 33–40 s; 20 questions ≈ 146 s (QI).
- Peak GPU: reported in `phase_b.json`.
- Retrieval: MiniLM ~3 ms/query, Qwen3-0.6B ~36 ms/query (CPU).

## Recommendations (pending review; nothing merged)
1. **Fix quiz intent detection.** Recognise spelled-out counts and "make / create / produce / generate … questions" phrasing. Today the live chatbot never produces a quiz for these prompts.
2. **Prepend each chunk's heading path to its text,** for both embedding and prompting. This restores the passage labels and section titles that production chunking strips, and enables real citations.
3. **Keep production's answer prompt.** Do not adopt the strict "say Not specified" prompt: it trades one fabrication for eight false abstentions. Consider a softer abstention instruction, and test it.
4. **Retrieval:** adopt hybrid RRF on production chunks, a near-free gain. Evaluate Qwen3-0.6B (+ R512 or parent-child) separately, because it improves multi-hop and table questions but costs context and latency.
5. **Quizzes:** adopt the blueprint, the sufficiency check for topic requests, and Outlines. Tune the check, which over-prunes Q094.
6. **Next:** human adjudication of quiz items and borderline answers, three repeats if sampling is ever enabled, and re-running after fixes 1–2.
