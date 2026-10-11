# Experiment 3 — Hierarchical parent-child retrieval (`exp/quiz-parent-child`)

Common baseline `e7c28d3`. Same fixture, provisional gold set (30 concepts plus topic requests D and E), MiniLM, inner-product retrieval.

## Hypothesis
Searching small child chunks and returning their larger section-bounded parents as evidence covers more gold facts per token than flat retrieval, without flooding the model.

## Implementation (`parent_child.py`)
- **Hierarchy.** Document → section (markdown headings) → paragraph → child.
  - Children: 128 / 256 / 384 Mistral tokens, sentence-safe.
  - Parents: 512 / 1024 / 1536 tokens.
  - A parent never crosses a section boundary, and each child belongs to one parent.
- **Evidence strategies.** All are packed in rank order, deduplicated, under the same token budget (1000 or 2000):
  - `parent`: child hits expanded to their parents.
  - `neighbors`: a child plus its previous/next child in the same section.
  - `child-only`.
  - `flat structured400`: production chunks shown whole.
  - `flat500`: today's production prompt, the top 6 chunks at 500 chars each (~780 tokens).
- **Coverage.** A concept counts as covered when its anchor phrase is inside the packed evidence.

## Command
```
python -m experiments.quiz_bench.parent_child --out experiments/quiz_bench/results/parent_child.json
```

## Results (n = 30 concept queries; D = 5 target concepts, E = 3)

| Strategy | Budget (tok) | Concept coverage | D targets | E targets |
|---|---|---|---|---|
| **flat500 — production prompt** | ~778 used | **0.333** | 0.00 | 0.33 |
| flat structured400 (whole chunks) | 1000 | **0.800** | 0.40 | 0.33 |
| parent c128/p512 | 1000 | 0.767 | 0.60 | 0.33 |
| parent c384/p512 | 1000 | 0.833 | 0.40 | 0.67 |
| neighbors c128 | 1000 | 0.800 | 0.60 | 0.67 |
| child-only c128 | 1000 | 0.800 | 0.80 | 0.33 |
| parent c*/p1536 | 1000 | 0.600–0.667 | 0.40 | 0.00–0.67 |
| flat structured400 (whole chunks) | 2000 | 0.833 | 0.60 | 0.33 |
| parent c128/p512 | 2000 | 0.900 | 0.60 | 1.00 |
| **parent c256/p1024** | 2000 | **0.933** | 0.60 | 1.00 |
| neighbors c256 | 2000 | 0.900 | 0.40 | 0.67 |
| child-only c128 | 2000 | 0.867 | **1.00** | 0.67 |
| parent c*/p1536 | 2000 | 0.800–0.833 | 0.40–0.60 | 0.00–1.00 |

Full grid: `results/parent_child.json`.

Paired bootstrap (10k resamples, per-concept coverage):

| Comparison | Diff | 95 % CI |
|---|---|---|
| flat whole chunks @1000 − production prompt | **+0.467** | **[+0.267, +0.667]** |
| parent c256/p1024 @2000 − flat @2000 | +0.100 | [−0.033, +0.233] |
| parent c128/p512 @2000 − flat @2000 | +0.067 | [0.000, +0.167] |
| parent c384/p512 @1000 − flat @1000 | +0.033 | [−0.067, +0.133] |

## What improved
- **Token-budgeted packing of whole retrieved passages** is the decisive change. It raises coverage from 0.33 to 0.80 at roughly the same token cost as today's six 500-char snippets. This is a prompt-construction change, independent of parent-child.
- Parent-child at a 2000-token budget gives the highest coverage observed (0.93 vs 0.83 flat), but the gain is not significant at n = 30.

## What became worse
- **Large parents (1536)** consume the budget with a few passages and reduce coverage, which confirms the plan's warning not to include entire parent sections.
- **Topic-level coverage (D/E)** is volatile with only 5 and 3 targets; no strategy is reliably best there.

## Hypothesis supported?
Not significantly for parent-child per se. Strongly supported for the underlying idea that evidence should be whole, deduplicated passages packed to a token budget rather than cut to 500 chars each.

## Recommendation
- **Adopt** token-budgeted whole-passage packing (about 1000–2000 tokens) for the integrated candidate. This is the single largest measured improvement in the retrieval phase.
- **Investigate further:** parent-child c256/p1024 or c128/p512 on more documents and a reviewed gold set before adopting.
- **Reject** parents ≥ 1536 tokens.

## Limitations
- One paper, provisional gold set, retrieval-level only.
- Downstream quiz quality was not measured (needs Mistral).
- Coverage counts anchor presence, not whether the model uses the evidence.
