# Testing and Evaluation

This project requires two parallel quality systems:

1. traditional software testing;
2. AI/retrieval evaluation.

Neither substitutes for the other.

---

# Part I — Software tests

## 1. Unit tests

### Classification / routing primitives

- year detection;
- department detection;
- list intent;
- person detection;
- ambiguous names;
- negative/common-word cases.

### Name/entity resolution

- full name;
- first/last name;
- aliases;
- canonical IDs;
- duplicate-name handling;
- no topic-word false positives.

### Retrieval

- BM25 ordering;
- dense result mapping;
- RRF calculation;
- metadata filters;
- exact-person path;
- reranking;
- per-capability collection selection.

### Evidence gate

- supported;
- unsupported;
- partial;
- conflict;
- boundary cases.

### Provenance/versioning

- new source;
- new document version;
- content-hash change;
- unchanged source;
- stale index;
- candidate snapshot promotion;
- source/version citation mapping.

### Conversation state

- result selection;
- pronoun/entity continuation;
- session isolation;
- topic switch;
- catalog-year state;
- expiration/reset.

---

## 2. Integration tests

Examples:

```text
question
→ route
→ retrieve
→ evidence gate
→ response state
→ structured sources
```

Mock LLM generation for most integration tests.

---

## 3. API tests

Validate:

- request schema;
- typed response schema;
- structured sources;
- capability labels;
- trace IDs;
- streaming;
- invalid input;
- session behavior;
- errors/cancellation.

---

# Part II — AUM Gold Set

## 4. Initial target

Build approximately **250–400 carefully curated queries** spanning current and near-term domains.

Suggested JSONL record:

```json
{
  "query_id": "cos_person_001",
  "question": "What projects did Dr. X mentor?",
  "query_class": "person_research",
  "answerable": true,
  "required_capabilities": ["people", "research"],
  "gold_source_ids": ["..."],
  "gold_evidence_ids": ["..."],
  "gold_claims": ["..."],
  "expected_citations": ["..."],
  "expected_routes": ["people", "research"],
  "catalog_or_policy_version": null,
  "multi_turn_parent_id": null,
  "notes": ""
}
```

Keep a stable human-reviewed control subset.

---

## 5. Retrieval metrics

Primary:

- Recall@K.

Additional:

- Precision@K;
- MRR;
- nDCG@K.

Report by:

- capability/domain;
- query class;
- answerable/unanswerable;
- single-hop/multi-hop where applicable.

Do not hide domain failures behind one average.

---

## 6. Routing metrics

For multi-label routing:

- exact route-set match;
- micro-F1;
- macro-F1;
- missed-capability rate;
- unnecessary-capability rate.

---

## 7. Answerability / abstention metrics

Track:

```text
answerable → correct answer
unanswerable → correct rejection
```

Error categories:

- false answer;
- false abstention;
- partial support treated as full support;
- conflict ignored;
- missing capability;
- retrieval miss;
- reranker miss;
- generation hallucination;
- citation mismatch.

---

## 8. Grounding/citation evaluation

Assess:

- claim correctness;
- claim-level support;
- answer relevance;
- citation support;
- citation completeness;
- unsupported additions;
- contradiction.

Automated frameworks/judges may accelerate evaluation, but they are not the sole release gate.

---

## 9. Multi-turn evaluation

Include cases with:

- pronouns;
- “the second one”;
- later-turn underspecification;
- topic shifts;
- return to previous topic;
- correction of a mistaken premise;
- catalog-year change;
- unanswerable follow-up;
- multi-capability query.

---

## 10. Conflict/version evaluation

Create cases where:

- old catalog differs from current catalog;
- two sources disagree;
- source has been superseded;
- a relation changed over time.

Expected behavior should include source/version awareness.

---

## 11. KG evaluation prerequisites

Before building a meaningful KG, define 20–30 questions that are relational enough that graph structure might help.

After a vertical graph exists, compare:

```text
hybrid text RAG
vs
KG-assisted retrieval
```

Retain graph machinery only where it measurably improves:

- correctness;
- coverage;
- citation quality;
- interpretability.

---

# Part III — Regression suite

## 12. Known issues to preserve as regression cases

- numbered COS selection routing;
- concurrent generation serialization;
- prompt-format compatibility;
- cache invalidation;
- stale client timeout behavior;
- Housing unrelated-query abstention after evidence gate exists;
- structured source propagation.

---

# Part IV — Serving benchmarks

## 13. Benchmark matrix

Input context:

```text
1K
4K
8K
16K
```

Output:

```text
128
512
1024
```

Concurrency:

```text
1
2
4
8
16
```

Compare at minimum:

```text
direct Transformers
vs
vLLM
```

Record:

- p50/p95/p99 TTFT;
- inter-token latency;
- tokens/sec/request;
- aggregate tokens/sec;
- queue time;
- GPU utilization;
- VRAM high-water mark;
- CPU RAM;
- OOM/error events;
- cancellation behavior;
- retrieval/rerank/end-to-end latency.

---

## 14. Quality gate

Before merging substantial changes:

```text
software tests: PASS
Gold Set: no unexplained regression
retrieval metrics: acceptable
routing metrics: acceptable
answerability/grounding: acceptable
source/version integrity: PASS
performance: no unexplained regression
real runtime smoke test: PASS
```
